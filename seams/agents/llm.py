import json
import os
import time
from pathlib import Path

from seams.agents.replay_cache import ReplayCache, request_key

OPENROUTER_URL = "https://openrouter.ai/api/v1"


class CacheMiss(RuntimeError):
    pass


def _api_key():
    if "OPENROUTER_API_KEY" not in os.environ:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    return os.environ["OPENROUTER_API_KEY"]


def nullable_enums_as_anyof(schema):
    if isinstance(schema, list):
        return [nullable_enums_as_anyof(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: nullable_enums_as_anyof(v) for k, v in schema.items()}
    t = out.get("type")
    if "enum" in out and isinstance(t, list) and "null" in t:
        base = [x for x in t if x != "null"]
        vals = [v for v in out.pop("enum") if v is not None]
        del out["type"]
        out["anyOf"] = [{"type": base[0] if len(base) == 1 else base, "enum": vals}, {"type": "null"}]
    return out


SCHEMA_COMPAT = {None: lambda s: s, "anyof_enums": nullable_enums_as_anyof}


class LLM:
    def __init__(self, backbone, cache, mode="record", client=None):
        self.b = backbone
        self.model = backbone["model"]
        self.cache = cache if isinstance(cache, ReplayCache) else ReplayCache(cache)
        if mode not in ("record", "replay"):
            raise ValueError(f"agents.mode: {mode!r}")
        self.mode = mode
        self._client = client
        self.usage = {"calls": 0, "api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=OPENROUTER_URL, api_key=_api_key(), timeout=60, max_retries=3)
        return self._client

    def _complete(self, messages, name, schema):
        # temperature: null -> the provider default, no seed (models that accept neither, e.g. Haiku 5.5)
        temp = self.b.get("temperature", 0.0)
        temp = None if temp is None else float(temp)
        key = request_key(self.model, temp, messages, schema)
        hit = self.cache.get(key)
        if hit is None:
            if self.mode == "replay":
                raise CacheMiss(f"{self.model}: no recorded response for request {key[:12]}")
            extra = {"provider": {"require_parameters": True}}
            if self.b.get("reasoning"):
                extra["reasoning"] = {"effort": self.b["reasoning"]}
            sampling = {} if temp is None else {"temperature": temp, "seed": 0}
            t0 = time.perf_counter()
            r = self.client.chat.completions.create(
                model=self.model, max_tokens=int(self.b.get("max_tokens", 4000)),
                messages=messages, extra_body=extra, **sampling,
                response_format={"type": "json_schema", "json_schema": {
                    "name": name, "strict": True, "schema": SCHEMA_COMPAT[self.b.get("schema_compat")](schema)}})
            latency = time.perf_counter() - t0
            text = r.choices[0].message.content or ""
            pt, ct = r.usage.prompt_tokens, r.usage.completion_tokens
            cost = (pt * self.b["price_in_per_mtok"] + ct * self.b["price_out_per_mtok"]) / 1e6
            self.cache.put(key, self.model, {"messages": messages, "schema": schema}, text, latency, pt, ct, cost)
            self.usage["api_calls"] += 1
            hit = {"response": text, "latency_s": latency, "prompt_tokens": pt, "completion_tokens": ct, "cost_usd": cost}
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += hit["prompt_tokens"]
        self.usage["completion_tokens"] += hit["completion_tokens"]
        self.usage["cost_usd"] += hit["cost_usd"]
        return hit["response"], hit["latency_s"]

    def call(self, system, user, name, schema, validate=None):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        latency, err = 0.0, None
        for _ in range(2):
            text, lat = self._complete(messages, name, schema)
            latency += lat
            try:
                obj = json.loads(text)
                if validate is not None:
                    validate(obj)
                return obj, latency, None
            except (ValueError, TypeError, KeyError) as e:
                err = f"{type(e).__name__}: {e}"
                messages = messages + [{"role": "assistant", "content": text},
                                       {"role": "user", "content": f"That reply was rejected ({err}). "
                                                                   "Reply again with valid JSON in the schema."}]
        return None, latency, err
