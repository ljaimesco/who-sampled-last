"""SUMO follow-up: does dist's realized advantage depend on forecast quality? And does it survive matched spending?

Phase 1 ("yesterday"): for each evaluation seed s, one calibration mission with the SAME PsIs and demand pattern
  but different background traffic, speed field and SUMO RNG (world seed 700 + s), played by mean-time under the
  global forecast. Its probes' segment traversals (first and last segment of each route excluded, unfinished
  records dropped) refit a per-seed forecast, shrunk toward the global one (fit_probe_forecast, n0 = 5).
Phase 2: the 30 evaluation seeds x 4 estimators rerun under their per-seed forecast (scale prior 1.0, no spread
  inflation; online rescaling kept), dist missions with ex-ante regret scoring. Raw: results/sumo_fq_raw.json.
Phase 3: matched spending under the ORIGINAL global forecast: dist's reward calibrated on seeds 100-103 to match
  mean-time's payout, then dist rerun on seeds 0-29. Raw: results/sumo_matched_raw.json.
"""
import json, os, sys, time, xml.etree.ElementTree as ET
from multiprocessing import Pool
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav import sumo_mission as smm, sumo_world as sw

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
R, TOL, ESTS, N = 400.0, 0.01, ("dist", "ordermean", "fixorder", "mean"), 30
KEYS = ("deadline_violation", "worst_gap", "net_per_vehicle", "payout", "inversion_rate", "teleports",
        "never_arrived", "forecast_bias", "forecast_mae", "forecast_cover80", "detours")
FQ = lambda s: os.path.join(sw.DATA, f"sumo_forecast_seed{s}.npz")


def calib_day(s):
    out = os.path.join(sw.WORK, f"cal_{s}.vehroutes.xml")
    o, _ = smm.run(s, "mean", R, world_seed=700 + s,
                   sumo_extra=["--vehroute-output", out, "--vehroute-output.exit-times", "true"])
    W = sw.world(); samples = {}
    for veh in ET.parse(out).getroot().iter("vehicle"):
        if not veh.attrib["id"].startswith("p"):
            continue
        r = veh.find("route")
        if r is None or "exitTimes" not in r.attrib:
            continue
        edges = r.attrib["edges"].split(); ex = [float(x) for x in r.attrib["exitTimes"].split()]
        if len(ex) != len(edges) or np.any(np.diff(ex) < 0):
            continue
        for i in range(1, len(edges) - 1):
            if edges[i] in W["idx"]:
                samples.setdefault(W["idx"][edges[i]], []).append((ex[i - 1], ex[i] - ex[i - 1]))
    os.remove(out)
    g = np.load(smm.FORECAST)
    mean, std, rho, diag = sw.fit_probe_forecast([samples], g["mean"].copy(), g["std"].copy(), n0=5.0)
    np.savez(FQ(s), mean=mean, std=std, rho=rho)
    return s, diag


def eval_job(a):
    s, est = a
    try:
        o, reg = smm.run(s, est, R, regret=(512, 1024) if est == "dist" else None,
                         forecast_npz=FQ(s), scale_prior=1.0, sigma_gain=1.0)
    except Exception as ex:
        return list(a), {"error": repr(ex)[:300]}, None
    return list(a), {k: float(o[k]) for k in KEYS}, ({e: reg[e]["win"] for e in ESTS} if reg else None)


def match_job(a):
    s, est, Rr = a
    try:
        o, _ = smm.run(s, est, Rr)
    except Exception as ex:
        return list(a), {"error": repr(ex)[:300]}
    return list(a), {k: float(o[k]) for k in KEYS}


def resumable(pool, fn, jobs, path, key):
    have = {}
    if os.path.exists(path):
        for r in json.load(open(path)):
            if "error" not in r[1]:
                have[key(r)] = r
    res = list(have.values())
    for r in pool.imap_unordered(fn, [j for j in jobs if tuple(j) not in have]):
        res.append(r); json.dump(res, open(path, "w"))
    return res


if __name__ == "__main__":
    t = time.time()
    with Pool(2) as pool:
        pool.map(sw.background, [700 + s for s in range(N)] + [300 + s for s in (100, 101, 102, 103)])
        todo = [s for s in range(N) if not os.path.exists(FQ(s))]
        for s, d in pool.imap_unordered(calib_day, todo):
            print("calibration day", s, json.dumps(d), round(time.time() - t), flush=True)
        fq = resumable(pool, eval_job, [(s, e) for s in range(N) for e in ESTS], f"{OUT}/sumo_fq_raw.json", lambda r: tuple(r[0]))
        print("phase 2 done", round(time.time() - t), flush=True)
        # phase 3: matched spending under the original forecast
        mp = np.mean([o["payout"] for a, o in pool.map(match_job, [(s, "mean", R) for s in (100, 101, 102, 103)])])
        Rd = R
        for _ in range(2):
            dp = np.mean([o["payout"] for a, o in pool.map(match_job, [(s, "dist", Rd) for s in (100, 101, 102, 103)])])
            Rd *= mp / dp
        print("matched reward for dist", round(Rd, 1), round(time.time() - t), flush=True)
        mt = resumable(pool, match_job, [(s, "dist", float(Rd)) for s in range(N)], f"{OUT}/sumo_matched_raw.json", lambda r: tuple(r[0][:2]))
    json.dump({"Rd": float(Rd)}, open(f"{OUT}/sumo_matched_R.json", "w"))
    print("all done", round(time.time() - t), flush=True)
