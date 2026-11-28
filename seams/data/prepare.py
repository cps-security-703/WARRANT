"""Real-data inputs: raw public datasets -> small processed tables the testbed samples from.

    python -m seams.data.prepare [--raw ~/seams_data/raw] [--spear SPEAR/evcs_data]

Sources (raw files stay outside the project; processed tables go to seams/data/processed):
  sessions.parquet   per hub profile, weekday L2 sessions: plug-in time of day, dwell, energy
      workplace  ACN-Data, Caltech (Lee, Li & Low, e-Energy 2019), Pasadena CA, Sep-Nov 2019
      public     EV WATTS (DOE/Energetics), Pacific region, Multi-use Parking Garage/Lot
      retail     EV WATTS, Pacific region, Retail
      home       EV WATTS, Pacific region, Single Family Residential (home chargers on the
                 feeder; 50k-session sample; with the charger's average power rate_kw)
      dcfc       EV WATTS, Pacific region, DC fast chargers (the fast-charging hub variant,
                 scenario.hub_kind: dcfc; 50k-session sample; energy capped at a 150 kW port)
  soc_start.npy      EV WATTS vehicle-side sessions: battery state of charge at plug-in (BEV)
  trip_km.npy        EV WATTS vehicle trips: trip length [km]
  trip_start_hour.npy  EV WATTS vehicle trips: weekday share of trip starts per hour of day
  load_la.parquet    NREL ResStock + ComStock 2021 (AMY 2018), Los Angeles County aggregate:
                     total building electricity per 15 min, one row per day, normalized to the
                     annual peak (residential/commercial mix is the county's own), plus the
                     same day's hourly ambient temperature t00..t23 [C] (NREL AMY 2018 weather)
What is *not* taken from data is stated where it is set: the demand level (sessions per port
per day; real public L2 runs at 0.2-0.5, far below the busy-hub scenario) and the port rating.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pc

from seams.cosim.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "seams" / "data" / "processed"
MIN_DWELL_S, MAX_DWELL_S, MIN_KWH = 600, 24 * 3600, 0.5
PORT_KW = 6.66  # our L2 port rating (208 V x 32 A); energy above dwell x rating is capped
DCFC_PORT_KW = 150.0  # the fast-charging hub's port rating (NEVI minimum, 23 CFR 680.106)


def _clean(df, source, cap=True, port_kw=PORT_KW):
    """Drop implausible sessions; for hub profiles cap energy at what one of our ports can
    deliver in the dwell (home sessions keep their own charger's rate instead)."""
    df = df[(df.dwell_s >= MIN_DWELL_S) & (df.dwell_s <= MAX_DWELL_S) & (df.kwh >= MIN_KWH)].copy()
    limit = df.dwell_s / 3600 * port_kw * 0.98 if cap else np.inf
    df["capped"] = df.kwh > limit
    df["kwh"] = np.minimum(df.kwh, limit)
    df["source"] = source
    return df


def acn_workplace(spear):
    d = pd.read_csv(spear.parent / "acndata" / "caltech_sessions_summary.csv")
    t0 = pd.to_datetime(d.connectionTime, utc=True).dt.tz_convert("America/Los_Angeles")
    t1 = pd.to_datetime(d.disconnectTime, utc=True).dt.tz_convert("America/Los_Angeles")
    df = pd.DataFrame({
        "date": t0.dt.date.astype(str),
        "arr_s": (t0 - t0.dt.normalize()).dt.total_seconds(),
        "dwell_s": (t1 - t0).dt.total_seconds(),
        "kwh": d.kWhDelivered,
        "weekday": t0.dt.day_of_week < 5,
        "port": d.stationID,
    })
    return _clean(df[df.weekday], "ACN-Data Caltech")


def evwatts_venue(spear, venues):
    root = spear / "evwatts.public"
    evse = pd.read_csv(root / "evwatts.public.evse.csv", encoding="utf-8-sig")
    evse = evse[(evse.region == "Pacific") & (evse.charge_level == "L2") & evse.venue.isin(venues.values())]
    s = pc.read_csv(root / "evwatts.public.session.csv", convert_options=pc.ConvertOptions(
        include_columns=["evse_id", "connector_id", "start_datetime", "end_datetime", "charge_duration",
                         "energy_kwh", "flag_id"],
    )).to_pandas()
    s = s[(s.flag_id == 0) & s.evse_id.isin(evse.evse_id)].merge(evse[["evse_id", "venue"]], on="evse_id")
    t0, t1 = pd.to_datetime(s.start_datetime), pd.to_datetime(s.end_datetime)  # local time
    out = {}
    for profile, venue in venues.items():
        m = (s.venue == venue) & (t0.dt.day_of_week < 5)
        df = pd.DataFrame({
            "date": t0[m].dt.date.astype(str),
            "arr_s": (t0[m] - t0[m].dt.normalize()).dt.total_seconds(),
            "dwell_s": (t1[m] - t0[m]).dt.total_seconds(),
            "kwh": s.energy_kwh[m], "weekday": True, "port": s.connector_id[m].astype(str),
            # average charging power while energy flowed [kW]
            "rate_kw": s.energy_kwh[m] / s.charge_duration[m].clip(lower=0.05),
        })
        out[profile] = _clean(df, f"EV WATTS Pacific {venue}", cap=profile != "home")
    return out


def evwatts_dcfc(spear):
    """Weekday sessions at EV WATTS Pacific DC fast chargers (all venues), for the fast-charging
    hub variant; 50k-session sample like the home profile."""
    root = spear / "evwatts.public"
    evse = pd.read_csv(root / "evwatts.public.evse.csv", encoding="utf-8-sig")
    evse = evse[(evse.region == "Pacific") & (evse.charge_level == "DCFC")]
    s = pc.read_csv(root / "evwatts.public.session.csv", convert_options=pc.ConvertOptions(
        include_columns=["evse_id", "connector_id", "start_datetime", "end_datetime", "charge_duration",
                         "energy_kwh", "flag_id"],
    )).to_pandas()
    s = s[(s.flag_id == 0) & s.evse_id.isin(evse.evse_id)]
    t0, t1 = pd.to_datetime(s.start_datetime), pd.to_datetime(s.end_datetime)  # local time
    m = t0.dt.day_of_week < 5
    df = pd.DataFrame({
        "date": t0[m].dt.date.astype(str),
        "arr_s": (t0[m] - t0[m].dt.normalize()).dt.total_seconds(),
        "dwell_s": (t1[m] - t0[m]).dt.total_seconds(),
        "kwh": s.energy_kwh[m], "weekday": True, "port": s.connector_id[m].astype(str),
        "rate_kw": s.energy_kwh[m] / s.charge_duration[m].clip(lower=0.05),
    })
    df = _clean(df, "EV WATTS Pacific DCFC", port_kw=DCFC_PORT_KW)
    return df.sample(n=min(len(df), 50_000), random_state=0)


def soc_start(spear):
    root = spear / "evwatts.public"
    v = pd.read_csv(root / "evwatts.public.vehicles.csv")
    bev = v[(v.vehicle_type == "PASSENGER CAR") & v.electrification_level.str.startswith("BEV")].id
    s = pd.read_csv(root / "evwatts.public.vehiclesessions.csv")
    s = s[(s.flag_id == 0) & s.vehicle_id.isin(bev) & s.soc_start.between(1, 99)]
    return (s.soc_start.to_numpy() / 100.0).astype(np.float32)


def trips(spear):
    """Trip length [km] (subsampled to keep the file small) and the weekday hour-of-day share of
    trip starts (through traffic departs on it)."""
    t = pd.read_csv(spear / "evwatts.public" / "evwatts.public.vehicletrips.csv",
                    usecols=["km", "start_datetime", "flag_id"])
    t = t[(t.flag_id == 0) & t.km.between(0.2, 50)]
    km = t.km.to_numpy().astype(np.float32)
    km = np.random.default_rng(0).choice(km, size=min(len(km), 100_000), replace=False)
    start = pd.to_datetime(t.start_datetime)
    hours = start[start.dt.day_of_week < 5].dt.hour
    return km, np.bincount(hours, minlength=24) / len(hours)


def load_la(raw):
    files = sorted((raw / "nrel_la").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no NREL files in {raw / 'nrel_la'} (see docstring)")
    total = None
    col = "out.electricity.total.energy_consumption"
    for f in files:
        d = pd.read_csv(f, usecols=["timestamp", col])
        s = d.set_index(pd.to_datetime(d.timestamp))[col]
        total = s if total is None else total.add(s, fill_value=0.0)
    # NREL timestamps mark the period end, in local standard time (PST, no DST); the session
    # data is in local clock time, so convert to America/Los_Angeles (DST-transition days drop
    # out below because they do not have 96 periods)
    start = total.index - pd.Timedelta(minutes=15) + pd.Timedelta(hours=8)
    total.index = start.tz_localize("UTC").tz_convert("America/Los_Angeles").tz_localize(None)
    total = total[total.index.year == 2018] / total.max()
    # ambient dry-bulb temperature, same county and year (NREL AMY 2018 weather), hourly
    w = pd.read_csv(raw / "nrel_la_weather" / "G0600370_2018.csv")
    temp = pd.Series(w.iloc[:, 1].to_numpy(), index=pd.to_datetime(w.date_time))
    start = temp.index - pd.Timedelta(hours=1) + pd.Timedelta(hours=8)
    temp.index = start.tz_localize("UTC").tz_convert("America/Los_Angeles").tz_localize(None)
    temps = {day: t.to_numpy() for day, t in temp.groupby(temp.index.date) if len(t) == 24}
    rows = [{"date": str(day), "month": day.month, "weekday": day.weekday() < 5, "p": s.to_numpy(), "t": temps[day]}
            for day, s in total.groupby(total.index.date) if len(s) == 96 and day in temps]
    df = pd.DataFrame(rows)
    return pd.concat([df.drop(columns=["p", "t"]),
                      pd.DataFrame(np.vstack(df.p), columns=[f"q{i:02d}" for i in range(96)]),
                      pd.DataFrame(np.vstack(df.t), columns=[f"t{i:02d}" for i in range(24)])], axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="~/seams_data/raw")
    ap.add_argument("--spear", default=str(PROJECT_ROOT / "SPEAR" / "evcs_data"))
    args = ap.parse_args(argv)
    raw, spear = Path(args.raw).expanduser(), Path(args.spear)
    OUT.mkdir(parents=True, exist_ok=True)

    profiles = {"workplace": acn_workplace(spear)}
    profiles.update(evwatts_venue(spear, {"public": "Multi-use Parking Garage/Lot", "retail": "Retail",
                                          "home": "Single Family Residential"}))
    home = profiles["home"]
    home = home[home.rate_kw.between(1.0, 11.5)]  # L1 trickle and implausible rates out
    profiles["home"] = home.sample(n=min(len(home), 50_000), random_state=0)  # keeps the file small
    profiles["dcfc"] = evwatts_dcfc(spear)
    sessions = pd.concat([df.assign(profile=p) for p, df in profiles.items()], ignore_index=True)
    sessions[["profile", "source", "date", "port", "arr_s", "dwell_s", "kwh", "rate_kw", "capped"]].to_parquet(
        OUT / "sessions.parquet", index=False)
    soc, load = soc_start(spear), load_la(raw)
    km, start_hour = trips(spear)
    np.save(OUT / "soc_start.npy", soc)
    np.save(OUT / "trip_km.npy", km)
    np.save(OUT / "trip_start_hour.npy", start_hour)
    load.to_parquet(OUT / "load_la.parquet", index=False)

    summary = {
        "sessions": {p: {
            "n": int(len(df)), "source": df.source.iloc[0], "capped_frac": round(float(df.capped.mean()), 4),
            "dwell_h_median": round(float(df.dwell_s.median() / 3600), 2),
            "kwh_median": round(float(df.kwh.median()), 2),
            "sessions_per_port_day": round(float(len(df) / df.groupby("port").date.nunique().sum()), 3),
        } for p, df in profiles.items()},
        "soc_start": {"n": int(len(soc)), "median": round(float(np.median(soc)), 3)},
        "trip_km": {"n": int(len(km)), "median": round(float(np.median(km)), 2)},
        "trip_start_hour": [round(float(x), 4) for x in start_hour],
        "load_la": {"days": int(len(load)), "peak_date": str(load.loc[load.filter(like="q").max(axis=1).idxmax(), "date"]),
                    "summer_weekday_tmax_C_median": round(float(load[load.month.between(6, 9) & load.weekday].filter(regex="^t[0-9]").max(axis=1).median()), 1)},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
