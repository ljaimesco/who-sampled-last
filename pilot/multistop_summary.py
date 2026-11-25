"""Summarize the multi-stop London experiment and compare with single-stop results on the same seeds."""
import json, os
import numpy as np
from scipy import stats
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ESTS, TOL = ("dist", "ordermean", "fixorder", "mean"), 0.01
rng = np.random.default_rng(0)
raw = [r for r in json.load(open(f"{OUT}/multistop_raw.json")) if "error" not in r[1]]
out = {tuple(r[0]): r[1] for r in raw}
regs = {r[0][0]: r[2] for r in raw if r[2]}
seeds = sorted(s for s in regs if all((s, e) in out for e in ESTS))
single = json.load(open(f"{OUT}/london_raw.json"))
s_out = {(a[1], a[2]): o for a, o in single["outs"] if a[0] == 0.5}
s_reg = {a[1]: w for a, w, d in single["regs"] if a[0] == 0.5}
L = ["# Multi-stop routes (up to two PsIs), London-calibrated model, rho = 0.5", "",
     f"{len(seeds)} paired seeds (scenarios 5000+). Multi-stop decisions use 1024 draws; the single-stop comparison is the "
     "London experiment on the same seeds (256 draws). Stable = ex-ante regret <= 1% of R; intervals are mission-clustered bootstrap.", "",
     "## Stability", "", "| Estimator | Single-stop stable | Multi-stop stable (95% CI) | Multi-stop mean regret | P95 |", "|---|---|---|---|---|"]
for e in ESTS:
    per = [np.array(regs[s][e]) for s in seeds]; x = np.concatenate(per)
    bs = [(np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))]) <= TOL).mean() for _ in range(3000)]
    y = np.concatenate([np.array(s_reg[s][e]) for s in seeds if s in s_reg])
    L.append(f"| {e} | {int((y<=TOL).sum())}/{len(y)} ({(y<=TOL).mean()*100:.0f}%) | {int((x<=TOL).sum())}/{len(x)} "
             f"({np.percentile(bs,2.5)*100:.0f} to {np.percentile(bs,97.5)*100:.0f}%) | {x.mean()*100:.2f}% | {np.percentile(x,95)*100:.1f}% |")
col = lambda e, k: np.array([out[(s, e)][k] for s in seeds])
scol = lambda e, k: np.array([s_out[(s, e)][k] for s in seeds])
L += ["", "## Outcomes (mean over seeds)", "", "| Estimator | Routes with two stops | Samples per mission | Time past deadline (single -> multi) | Longest gap | Net per vehicle | Payout |", "|---|---|---|---|---|---|---|"]
for e in ESTS:
    L.append(f"| {e} | {col(e,'pair_routes').mean()/col(e,'detours').mean()*100:.0f}% | {col(e,'samples').mean():.0f} | "
             f"{scol(e,'deadline_violation').mean():.3f} -> {col(e,'deadline_violation').mean():.3f} | {scol(e,'worst_gap').mean():.0f} -> {col(e,'worst_gap').mean():.0f} | "
             f"{scol(e,'net_per_vehicle').mean():.0f} -> {col(e,'net_per_vehicle').mean():.0f} | {scol(e,'payout').mean():.0f} -> {col(e,'payout').mean():.0f} |")
L += ["", "## Paired Wilcoxon, multi-stop, dist vs each estimator", "", "| Comparison | Time past deadline | p | Longest gap | p | Net per vehicle | p | Payout | p |", "|---|---|---|---|---|---|---|---|---|"]
for o in ("mean", "fixorder", "ordermean"):
    row = [f"dist vs {o}"]
    for k in ("deadline_violation", "worst_gap", "net_per_vehicle", "payout"):
        d, m = col("dist", k), col(o, k)
        row += [f"{(d.mean()-m.mean())/m.mean()*100:+.1f}%", f"{stats.wilcoxon(d, m).pvalue:.2g}" if np.any(d != m) else "n/a"]
    L.append("| " + " | ".join(row) + " |")
L += ["", f"Reward cuts (cycle breaking) per mission, dist: {col('dist','breaks').mean():.1f}; mean: {col('mean','breaks').mean():.1f}."]
open(f"{OUT}/multistop_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
