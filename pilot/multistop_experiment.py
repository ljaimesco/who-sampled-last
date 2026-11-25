"""Multi-stop (up to two PsIs) on the London-calibrated model, rho = 0.5, ell = 300 s, 20 paired seeds
(LondonScenario seeds 5000-5019, the same scenarios as the single-stop London experiment).
Decisions use S = 1024 draws for every estimator (mean averages them); dist missions score all estimators'
profiles at each window state (ex-ante regret, independent 512 / 1024 draws). Raw: results/multistop_raw.json."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav import london_mission as lm, multistop as ms
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
R, TOL, ESTS, N = 400.0, 0.01, ("dist", "ordermean", "fixorder", "mean"), 20
KEYS = ("deadline_violation", "worst_gap", "net_per_vehicle", "payout", "inversion_rate", "pair_routes", "samples", "detours", "breaks")

def job(a):
    s, est = a
    try:
        ml = ms.MultiLondon(lm.LondonScenario(5000 + s, 0.5), R, max_stops=2)
        o, reg = ms.run_london(ml, est, R, S=1024, seed=s + 1, regret=(512, 1024) if est == "dist" else None)
    except Exception as ex:
        return [list(a), {"error": repr(ex)[:300]}, None]
    return [list(a), {k: float(o[k]) for k in KEYS}, ({e: reg[e]["win"] for e in ESTS} if reg else None)]

if __name__ == "__main__":
    t = time.time(); path = f"{OUT}/multistop_raw.json"
    have = {tuple(r[0]): r for r in json.load(open(path)) if "error" not in r[1]} if os.path.exists(path) else {}
    res = list(have.values())
    jobs = [(s, e) for s in range(N) for e in ESTS if (s, e) not in have]
    jobs.sort(key=lambda a: a[1] != "dist")            # long regret missions first
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r); json.dump(res, open(path, "w"))
            print("missions", len(res), round(time.time() - t), flush=True)
    print("done", round(time.time() - t), flush=True)
