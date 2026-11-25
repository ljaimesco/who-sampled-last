"""Multi-stop, expenditure approximately matched (held-out), London model then SUMO.
dist's reward is calibrated so its mean payout equals mean-time's at R = 400 on calibration seeds that are never
evaluated (London: scenarios 5100-5119; SUMO: worlds 100-105), two multiplicative updates. dist is then rerun at the
calibrated reward on the evaluation seeds (London 5000-5019, SUMO 0-29); the other estimators' results on those
seeds are reused from multistop_raw.json / sumo_ms_raw.json (same code path, R = 400).
Raw: results/multistop_matched_raw.json; summary: results/multistop_matched.md."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav import london_mission as lm, multistop as ms
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
KEYS = ("deadline_violation", "worst_gap", "net_per_vehicle", "payout", "samples")
PATH = f"{OUT}/multistop_matched_raw.json"

def job(a):
    world, scn_seed, est, R = a
    try:
        if world == "london":
            ml = ms.MultiLondon(lm.LondonScenario(scn_seed, 0.5), R, max_stops=2)
            o, _ = ms.run_london(ml, est, R, S=1024, seed=scn_seed - 5000 + 1)
        else:
            from spa_uav import sumo_multistop as sms
            o, _ = sms.run(scn_seed, est, R, S=1024)
    except Exception as ex:
        return [list(a), {"error": repr(ex)[:300]}]
    return [list(a), {k: float(o[k]) for k in KEYS}]

CACHE = {}
def runs(pool, jobs):
    todo = [j for j in jobs if tuple(j) not in CACHE]
    for a, o in pool.imap_unordered(job, todo):
        if "error" in o: print("failed", a, o, flush=True); continue
        CACHE[tuple(a)] = o; json.dump([[list(k), v] for k, v in CACHE.items()], open(PATH, "w"))
    return [CACHE[tuple(j)] for j in jobs if tuple(j) in CACHE]

def calibrate(pool, world, cal):
    mp = np.mean([o["payout"] for o in runs(pool, [(world, s, "mean", 400.0) for s in cal])])
    Rd = 400.0
    for _ in range(2):
        dp = np.mean([o["payout"] for o in runs(pool, [(world, s, "dist", Rd) for s in cal])])
        Rd = round(Rd * mp / dp, 2)
    return Rd

if __name__ == "__main__":
    t = time.time()
    if os.path.exists(PATH):
        CACHE.update({tuple(a): o for a, o in json.load(open(PATH))})
    raw_l = {(a[0] + 5000, a[1]): o for a, o, w in json.load(open(f"{OUT}/multistop_raw.json")) if "error" not in o}
    raw_s = {(a[0], a[1]): o for a, o, w in json.load(open(f"{OUT}/sumo_ms_raw.json")) if "error" not in o}
    L = ["# Multi-stop: expenditure approximately matched (held-out)", "",
         "dist's reward is calibrated on seeds never evaluated so its payout matches mean-time's at R = 400; other estimators keep R = 400.", ""]
    with Pool(2) as pool:
        for world, cal, ev, raw in (("london", range(5100, 5120), range(5000, 5020), raw_l), ("sumo", range(100, 106), range(30), raw_s)):
            if world == "sumo":
                from spa_uav import sumo_world as sw
                pool.map(sw.background, [300 + s for s in cal])
            Rd = calibrate(pool, world, list(cal))
            print(world, "calibrated reward", Rd, round(time.time() - t), flush=True)
            dres = runs(pool, [(world, s, "dist", Rd) for s in ev])
            ok = [s for s, o in zip(ev, dres)]
            d = {k: np.array([CACHE[(world, s, "dist", Rd)][k] for s in ev if (world, s, "dist", Rd) in CACHE]) for k in KEYS}
            L += [f"## {world.upper()} (calibrated dist reward {Rd:.1f}, {len(d['payout'])} evaluation seeds)", "",
                  "| Comparison | Payout difference | p | Time past deadline | p | Longest gap | p | Net per vehicle | p | Samples |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            evs = [s for s in ev if (world, s, "dist", Rd) in CACHE]
            for o in ("mean", "fixorder", "ordermean"):
                m = {k: np.array([raw[(s, o)][k] for s in evs]) for k in KEYS}
                row = [f"dist (R = {Rd:.0f}) vs {o} (R = 400)"]
                for k in ("payout", "deadline_violation", "worst_gap", "net_per_vehicle"):
                    row += [f"{(d[k].mean()-m[k].mean())/abs(m[k].mean())*100:+.1f}%", f"{stats.wilcoxon(d[k], m[k]).pvalue:.2g}"]
                row.append(f"{d['samples'].mean():.0f} vs {m['samples'].mean():.0f}")
                L.append("| " + " | ".join(row) + " |")
            L.append("")
            open(f"{OUT}/multistop_matched.md", "w").write("\n".join(L) + "\n"); print("\n".join(L), flush=True)
    print("done", round(time.time() - t), flush=True)
