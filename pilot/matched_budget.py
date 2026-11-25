"""Matched expenditure: scale dist's reward R_d until its mean total payout equals
mean-time's payout at R = 400, then compare mission outcomes (paired Wilcoxon over missions)."""
import os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav.corr_mission import CorrScenario, run
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
KEYS = ("deadline_violation", "worst_gap", "net_per_vehicle", "payout", "detours")

def job(a):
    cv, rho, m, est, R = a
    o, _ = run(CorrScenario(3000 + m, cv, rho), est, R, S=256, seed=m + 1)
    return a, {k: o[k] for k in KEYS}

def batch(pool, cv, rho, est, R, n, off=0):
    res = pool.map(job, [(cv, rho, m, est, R) for m in range(off, off + n)])
    return {k: np.array([r[k] for _, r in res]) for k in KEYS}

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    t = time.time(); lines = ["# Matched expenditure: dist reward calibrated on missions 100-129, evaluated on held-out paired missions 0-29", "",
        "| cv | rho | R (mean) | R (dist, matched) | Held-out payout mean / dist (difference) | Time past deadline: dist vs mean | p | Longest gap: dist vs mean | p | Net per vehicle: dist vs mean | p |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    with Pool(2) as pool:
        for cv, rho in ((0.15, 0.5), (0.3, 0.5)):
            # calibrate R_d on missions 100..100+n-1, evaluate on held-out missions 0..n-1
            cal = batch(pool, cv, rho, "mean", 400.0, n, off=100)
            Rd = 400.0
            for _ in range(3):
                d = batch(pool, cv, rho, "dist", Rd, n, off=100)
                Rd *= cal["payout"].mean() / d["payout"].mean()
            base = batch(pool, cv, rho, "mean", 400.0, n)
            d = batch(pool, cv, rho, "dist", Rd, n)               # decisions recomputed under the scaled reward
            gap = (d["payout"].mean() - base["payout"].mean()) / base["payout"].mean() * 100
            row = [f"{cv}", f"{rho}", "400", f"{Rd:.1f}", f"{base['payout'].mean():.0f} / {d['payout'].mean():.0f} ({gap:+.2f}%, p = {stats.wilcoxon(d['payout'], base['payout']).pvalue:.2g})"]
            for k in ("deadline_violation", "worst_gap", "net_per_vehicle"):
                rel = (d[k].mean() - base[k].mean()) / base[k].mean() * 100
                row += [f"{rel:+.1f}%", f"{stats.wilcoxon(d[k], base[k]).pvalue:.2g}"]
            lines.append("| " + " | ".join(row) + " |")
    open(f"{OUT}/matched_budget.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("elapsed", round(time.time() - t))
