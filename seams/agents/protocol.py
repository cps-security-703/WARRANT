r"""Negotiation protocol (guideline, RQ4), one LangGraph run per renegotiation:

  propose -> verify -> rank -> decide --agreement--> done
                                     \--none, round 0--> revise -> verify -> rank -> decide -> done

1. The master broadcasts the blame, C^deg and the grammar (Context); the default outcome is the
   classical path's decision (its optimized contract, or theta_safe).
2. Each agent submits up to k candidates (JSON, fixed schema); malformed ones are discarded.
3. The verifier gate certifies each (V1, V2, exit-aware); rejected ones carry the counterexample.
4. Agents accept and rank the certified candidates by their private objective; one revision round
   with the counterexamples if no candidate is accepted by all.
5. Agreement = a certified candidate every agent accepts (the best rank sum among them);
   otherwise, or past the deadline, the default is installed.
6. Everything is logged (the returned record).

Agents in a stage run in parallel: a stage's latency is its slowest agent. The time to agreement
is the sum over stages plus the verifier's time, compared with the deadline (one contract period).

RQ5: a stalling (Byzantine) agent proposes nothing and accepts nothing, without an LLM call.
Mitigation (optional, agents.stall_limit = n): an agent that accepted no certified candidate in n
consecutive negotiations that had one is excluded from later negotiations (no proposals, no veto);
safety is unaffected, since every candidate still passes the gate.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from seams.agents import gate
from seams.agents.operator_agent import RANK_SCHEMA, clean_template, proposal_schema
from seams.contracts import grammar


class State(TypedDict, total=False):
    candidates: list      # {id, source, templates, ok, check, cex, schedule, own_cost{role}}
    rankings: dict        # role -> {accept, ranking, rationale}
    round: int
    latency_s: float
    verify_ms: float
    malformed: int
    log: list
    outcome: Any


class Negotiation:
    def __init__(self, agents, cfg):
        self.agents = {a.role: a for a in agents}
        a = cfg["agents"]
        self.deadline_s = float(a.get("deadline_s", cfg["clock"]["contract_period_s"]))
        self.revise_rounds = int(a.get("revise_rounds", 1))
        self.stall_limit = a.get("stall_limit")
        self.stall_count = {r: 0 for r in self.agents}
        self.excluded = set()
        g = StateGraph(State)
        g.add_node("propose", self._propose)
        g.add_node("verify", self._verify)
        g.add_node("rank", self._rank)
        g.add_node("revise", self._revise)
        g.set_entry_point("propose")
        g.add_edge("propose", "verify")
        g.add_edge("revise", "verify")
        g.add_edge("verify", "rank")
        g.add_conditional_edges("rank", self._decide, {"revise": "revise", "done": END})
        self.graph = g.compile()

    # --- one negotiation -------------------------------------------------------------
    def run(self, ctx, history, default_theta, default_label):
        self.ctx, self.history = ctx, history
        # if even the most conservative schedule the grammar can express (every P_bar 0 at every
        # lead, the shortest validity window) fails the gate, no candidate can certify: skip the
        # agents (no LLM call)
        zero = gate.from_theta(ctx.base.replace("zero", P_bar={j: 0 for j in ctx.base.P_bar}), ctx)
        zero.valid = ctx.min_valid
        pre = gate.certify_schedule(ctx, zero, "zero")
        if not pre.ok:
            return None, {"agreement": False, "timeout": False, "chosen": None, "chosen_source": None, "rounds": 0,
                          "n_candidates": 0, "n_certified": 0, "n_malformed": 0, "latency_s": 0.0, "verify_ms": 0.0,
                          "graph_wall_s": 0.0, "default_cost": {}, "chosen_cost": None, "candidates": [],
                          "log": [{"event": "infeasible", "check": pre.check, "cex": pre.cex}], "skipped": "infeasible"}
        self.views = {r: a.view(ctx, history) for r, a in self.agents.items()}
        dsched = gate.from_theta(default_theta, ctx)
        self.default = {r: {"label": default_label, "cost": a.cost(dsched, self.views[r], ctx)}
                        for r, a in self.agents.items()}
        t0 = time.perf_counter()
        st = self.graph.invoke({"candidates": [], "rankings": {}, "round": 0, "latency_s": 0.0,
                                "verify_ms": 0.0, "malformed": 0, "log": []})
        st["graph_wall_s"] = time.perf_counter() - t0
        return self._finish(st)

    @property
    def active(self):
        return {r: a for r, a in self.agents.items() if r not in self.excluded}

    def _parallel(self, fn, stalled):
        """fn per active agent, in parallel; a stalling agent answers `stalled` at once."""
        active = self.active
        live = [a for a in active.values() if not a.stalls]
        with ThreadPoolExecutor(max(1, len(live))) as ex:
            out = dict(zip([a.role for a in live], ex.map(fn, live)))
        out.update({r: (stalled, 0.0) for r, a in active.items() if a.stalls})
        return out, max((lat for _, lat in out.values()), default=0.0)

    def _add_proposals(self, st, replies):
        cands, malformed, log = list(st["candidates"]), st["malformed"], list(st["log"])
        for role, (obj, _lat) in replies.items():
            if obj is None:
                malformed += 1
                continue
            for c in obj["candidates"][: self.agents[role].k]:
                cid = f"{role[0].upper()}{st['round']}{len([x for x in cands if x['source'] == role])}"
                tmpl = [clean_template(t) for t in c["templates"]]
                try:
                    parsed = gate.parse(tmpl, self.ctx)
                except ValueError as e:
                    malformed += 1
                    log.append({"event": "malformed", "role": role, "error": str(e), "templates": tmpl})
                    continue
                cands.append({"id": cid, "source": role, "templates": tmpl, "parsed": parsed,
                              "rationale": c["rationale"], "round": st["round"]})
        return cands, malformed, log

    def _propose(self, st):
        def ask(agent):
            obj, lat, err = agent.llm.call(agent.system(self.ctx),
                                           agent.propose_prompt(self.ctx, self.views[agent.role], self.default[agent.role]),
                                           "proposal", proposal_schema(agent.k))
            return obj, lat
        replies, lat = self._parallel(ask, {"candidates": []})
        cands, malformed, log = self._add_proposals(st, replies)
        return {"candidates": cands, "malformed": malformed, "log": log, "latency_s": st["latency_s"] + lat}

    def _verify(self, st):
        t0 = time.perf_counter()
        cands = []
        for c in st["candidates"]:
            if "ok" not in c:
                r = gate.certify(self.ctx, c["parsed"], c["id"])
                c = {**c, "ok": r.ok, "check": r.check, "cex": r.cex, "schedule": r.schedule,
                     "own_cost": {role: a.cost(r.schedule, self.views[role], self.ctx) for role, a in self.agents.items()}}
            cands.append(c)
        ms = 1000.0 * (time.perf_counter() - t0)
        return {"candidates": cands, "verify_ms": st["verify_ms"] + ms, "latency_s": st["latency_s"] + ms / 1000.0}

    def _certified(self, st, role):
        return [{"id": c["id"], "source": c["source"], "templates": c["templates"], "own_cost": c["own_cost"][role]}
                for c in st["candidates"] if c["ok"]]

    def _rank(self, st):
        if not any(c["ok"] for c in st["candidates"]):
            return {"rankings": {}}
        ids = {c["id"] for c in st["candidates"] if c["ok"]}

        def validate(obj):
            bad = [x for x in obj["accept"] + obj["ranking"] if x not in ids]
            if bad:
                raise ValueError(f"unknown candidate ids {bad}")

        def ask(agent):
            obj, lat, err = agent.llm.call(agent.system(self.ctx),
                                           agent.rank_prompt(self.ctx, self.views[agent.role], self.default[agent.role],
                                                             self._certified(st, agent.role)),
                                           "ranking", RANK_SCHEMA, validate)
            return obj, lat
        replies, lat = self._parallel(ask, {"accept": [], "ranking": [], "rationale": "stall"})
        rankings = {r: (obj or {"accept": [], "ranking": [], "rationale": "malformed"}) for r, (obj, _) in replies.items()}
        log = st["log"] + [{"event": "rank", "round": st["round"], "rankings": rankings}]
        return {"rankings": rankings, "log": log, "latency_s": st["latency_s"] + lat}

    def _agreed(self, st):
        rk = st.get("rankings") or {}
        if len(rk) < len(self.active):
            return []
        return sorted(set.intersection(*(set(r["accept"]) for r in rk.values())))

    def _decide(self, st):
        if self._agreed(st) or st["round"] >= self.revise_rounds or st["latency_s"] > self.deadline_s:
            return "done"
        return "revise"

    def _revise(self, st):
        st = {**st, "round": st["round"] + 1}

        def ask(agent):
            mine = [{"id": c["id"], "ok": c["ok"], "check": c["check"], "cex": c["cex"], "templates": c["templates"]}
                    for c in st["candidates"] if c["source"] == agent.role]
            obj, lat, err = agent.llm.call(agent.system(self.ctx),
                                           agent.revise_prompt(self.ctx, self.views[agent.role], self.default[agent.role],
                                                               mine, self._certified(st, agent.role)),
                                           "proposal", proposal_schema(agent.k))
            return obj, lat
        replies, lat = self._parallel(ask, {"candidates": []})
        cands, malformed, log = self._add_proposals(st, replies)
        return {"candidates": cands, "malformed": malformed, "log": log, "round": st["round"],
                "latency_s": st["latency_s"] + lat, "rankings": {}}

    def _finish(self, st):
        agreed = self._agreed(st)
        rk = st.get("rankings") or {}
        timeout = st["latency_s"] > self.deadline_s
        chosen = None
        if agreed and not timeout:
            def borda(cid):
                return sum(r["ranking"].index(cid) if cid in r["ranking"] else len(r["ranking"]) for r in rk.values())
            chosen = next(c for c in st["candidates"] if c["id"] == min(agreed, key=lambda x: (borda(x), x)))
        n = len(st["candidates"])
        record = {
            "agreement": chosen is not None, "timeout": timeout, "chosen": chosen["id"] if chosen else None,
            "chosen_source": chosen["source"] if chosen else None, "rounds": st["round"] + 1,
            "n_candidates": n, "n_certified": sum(c["ok"] for c in st["candidates"]), "n_malformed": st["malformed"],
            "latency_s": st["latency_s"], "verify_ms": st["verify_ms"], "graph_wall_s": st["graph_wall_s"],
            "default_cost": {r: d["cost"] for r, d in self.default.items()},
            "chosen_cost": chosen["own_cost"] if chosen else None,
            "candidates": [{k: v for k, v in c.items() if k not in ("parsed", "schedule")} for c in st["candidates"]],
            "log": st["log"], "excluded": sorted(self.excluded),
        }
        self._count_stalls(st, record)
        return (chosen["schedule"] if chosen else None), record

    def _count_stalls(self, st, record):
        """Stall-limit mitigation: count consecutive negotiations, with a certified candidate on the
        table, in which an agent accepted none; exclude it at the limit."""
        if not any(c["ok"] for c in st["candidates"]):
            return
        rk = st.get("rankings") or {}
        for r in self.active:
            self.stall_count[r] = self.stall_count[r] + 1 if not (rk.get(r) or {}).get("accept") else 0
            if self.stall_limit is not None and self.stall_count[r] >= int(self.stall_limit):
                self.excluded.add(r)
                record["log"].append({"event": "excluded", "role": r, "after_stalls": self.stall_count[r]})
