"""Ablations on matched window states (dist mission), ex-ante regret.

Estimators: dist (full joint law), ordermean (random predecessor identity,
gap replaced by its conditional mean), fixorder (random gaps, predecessor
fixed from mean times), mean (all arrivals at their means).
Control: nr_dist / nr_mean solve and score a reward that ignores every other
vehicle (gap measured from the last realized sample only), so any regret
there comes from the nonlinearity of the reward alone.
Intervals: 95% mission-clustered bootstrap."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav.corr_mission import CorrScenario, run
R, TOL = 400.0, 0.01
ESTS = ("dist", "ordermean", "fixorder", "mean", "nr_dist", "nr_mean")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def job(a):
    cv, rho, m = a
    _, reg = run(CorrScenario(3000 + m, cv, rho), "dist", R, S=256, seed=m + 1, regret=(512, 1024), ests=ESTS)
    return a, {e: [x / R for x in reg[e]["win"]] for e in ESTS}

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cfg = [(0.15, 0.5), (0.3, 0.0), (0.3, 0.5), (0.3, 0.8)]
    t = time.time()
    with Pool(2) as p:
        res = p.map(job, [(cv, rho, m) for cv, rho in cfg for m in range(n)])
    json.dump([dict(cv=a[0], rho=a[1], mission=a[2], win=r) for a, r in res], open(f"{OUT}/ablation_regret.json", "w"))
    rng = np.random.default_rng(0)
    lines = ["# Ablation: ex-ante window regret on matched states", "",
             f"{n} missions per row, 95% mission-clustered bootstrap. Stable = regret <= 1% of R.", "",
             "| cv | rho | Estimator | Stable windows | Stable share (95% CI) | Mean regret (95% CI) | P95 | Max |",
             "|---|---|---|---|---|---|---|---|"]
    for cv, rho in cfg:
        for e in ESTS:
            per = [np.array(r[e]) for a, r in res if a[:2] == (cv, rho)]
            w = np.concatenate(per)
            bs = [np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))]) for _ in range(3000)]
            st = [(b <= TOL).mean() for b in bs]; mu = [b.mean() for b in bs]
            lines.append(f"| {cv} | {rho} | {e} | {int((w<=TOL).sum())}/{len(w)} | {np.percentile(st,2.5)*100:.1f} to {np.percentile(st,97.5)*100:.1f}% | "
                         f"{w.mean()*100:.2f}% ({np.percentile(mu,2.5)*100:.2f} to {np.percentile(mu,97.5)*100:.2f}) | {np.percentile(w,95)*100:.1f}% | {w.max()*100:.1f}% |")
    open(f"{OUT}/ablation_summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("elapsed", round(time.time() - t))
