"""Monte Carlo uncertainty of the reversal analysis (results/theory_records.npz, 'pairs').
For every (estimator, window, player): true-best option k_true and estimator-best k_est on the 4096 evaluation
draws, the paired utility difference u(k_true) - u(k_est) and its standard error (same draws for both options).
Reports the paired standard errors, the reversals whose paired 95% interval excludes zero, and mission-clustered
bootstrap 95% CIs of the reversal rates."""
import os
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
z = np.load(f"{OUT}/theory_records.npz")
P = z["pairs"]; F = list(z["pfields"]); c = {f: P[:, i] for i, f in enumerate(F)}
R = 400.0; ESTS = ("dist", "ordermean", "fixorder", "mean")
rng = np.random.default_rng(0)
L = ["# Monte Carlo uncertainty of the reversal analysis", "",
     "Paired standard error of u(true best) - u(estimator's best) on the 4096 evaluation draws, percent of R (decisions where the two differ).", "",
     "| Estimator | Decisions | Differing choices | Paired SE: median / 95th pct | Reversals, 95% interval excludes 0 | Rate (mission-clustered 95% CI) | Reversals > 2% of R, interval excludes 2% |",
     "|---|---|---|---|---|---|---|"]
for e, name in enumerate(ESTS):
    m = c["est"] == e
    diff, se, seeds = c["diff"][m] / R, c["se_diff"][m] / R, c["seed"][m]
    differ = c["k_true"][m] != c["k_est"][m]
    sig = differ & (diff - 1.96 * se > 0)
    sig2 = differ & (diff - 1.96 * se > 0.02)
    us = np.unique(seeds)
    bs = []
    for _ in range(5000):
        pick = rng.choice(us, len(us))
        idx = np.concatenate([np.nonzero(seeds == s)[0] for s in pick])
        bs.append(sig[idx].mean())
    sd = se[differ] * 100 if differ.any() else np.zeros(1)
    L.append(f"| {name} | {m.sum()} | {differ.sum()} | {np.median(sd):.2f} / {np.percentile(sd,95):.2f} | {sig.sum()} | "
             f"{sig.mean():.1%} ({np.percentile(bs,2.5):.1%} to {np.percentile(bs,97.5):.1%}) | {sig2.sum()} ({sig2.mean():.1%}) |")
open(f"{OUT}/theory_mc.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
