"""London, expenditure approximately matched: dist's reward is calibrated on seeds 5100-5129 so its mean
payout equals mean-time's at R = 400; decisions are recomputed under it and evaluated on held-out paired
seeds 5000-5029 (mean-time results on those seeds are reused from london_raw.json; same code path)."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav import london_mission as lm
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
KEYS = ("deadline_violation", "worst_gap", "net_per_vehicle", "payout")

def job(a):
    rho, seed, est, R = a
    o, _, _ = lm.run(lm.LondonScenario(seed, rho), est, R, S=256, seed=seed - 5000 + 1)
    return a, {k: float(o[k]) for k in KEYS}

def batch(pool, rho, est, R, seeds):
    res = dict(pool.map(job, [(rho, s, est, R) for s in seeds]))
    return {k: np.array([res[(rho, s, est, R)][k] for s in seeds]) for k in KEYS}

if __name__ == "__main__":
    t = time.time()
    raw = json.load(open(f"{OUT}/london_raw.json"))
    cal_seeds, ev_seeds = list(range(5100, 5130)), list(range(5000, 5030))
    L = ["# London: expenditure approximately matched (held-out)", "",
         "| rho | R (dist, calibrated) | Held-out payout mean vs dist (difference, p) | Time past deadline dist vs mean | p | Longest gap | p | Net per vehicle | p |",
         "|---|---|---|---|---|---|---|---|---|"]
    with Pool(2) as pool:
        for rho in (0.0, 0.5, 0.8):
            cal = batch(pool, rho, "mean", 400.0, cal_seeds)
            Rd = 400.0
            for _ in range(2):
                d = batch(pool, rho, "dist", Rd, cal_seeds)
                Rd *= cal["payout"].mean() / d["payout"].mean()
            d = batch(pool, rho, "dist", Rd, ev_seeds)
            mo = {a[1]: o for a, o in raw["outs"] if a[0] == rho and a[2] == "mean"}
            m = {k: np.array([mo[s - 5000][k] for s in ev_seeds]) for k in KEYS}
            gap = (d["payout"].mean() - m["payout"].mean()) / m["payout"].mean() * 100
            row = [str(rho), f"{Rd:.1f}", f"{m['payout'].mean():.0f} vs {d['payout'].mean():.0f} ({gap:+.2f}%, p = {stats.wilcoxon(d['payout'], m['payout']).pvalue:.2g})"]
            for k in ("deadline_violation", "worst_gap", "net_per_vehicle"):
                row += [f"{(d[k].mean()-m[k].mean())/m[k].mean()*100:+.1f}%", f"{stats.wilcoxon(d[k], m[k]).pvalue:.2g}"]
            L.append("| " + " | ".join(row) + " |"); print(L[-1], round(time.time() - t), flush=True)
    open(f"{OUT}/london_matched.md", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("elapsed", round(time.time() - t))
