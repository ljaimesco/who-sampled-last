"""Single-stop vs multi-stop in SUMO (same 30 seeds), and dist vs every estimator within multi-stop."""
import json, os
import numpy as np
from scipy import stats
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ESTS = ("dist", "ordermean", "fixorder", "mean")
load = lambda f: {(a[0], a[1]): o for a, o, w in json.load(open(f"{OUT}/{f}")) if "error" not in o}
s1, s2 = load("sumo_raw.json"), load("sumo_ms_raw.json")
seeds = sorted({s for s, e in s2 if all((s, x) in s2 and (s, x) in s1 for x in ESTS)})
L = ["# SUMO: single-stop vs multi-stop", "", f"{len(seeds)} paired seeds.", "",
     "| Estimator | Time past deadline (single to multi) | Longest gap | Net per vehicle | Payout | Samples (multi) |", "|---|---|---|---|---|---|"]
m = lambda d, e, k: np.array([d[(s, e)][k] for s in seeds])
for e in ESTS:
    L.append(f"| {e} | {m(s1,e,'deadline_violation').mean():.3f} to {m(s2,e,'deadline_violation').mean():.3f} | {m(s1,e,'worst_gap').mean():.0f} to {m(s2,e,'worst_gap').mean():.0f} | "
             f"{m(s1,e,'net_per_vehicle').mean():.0f} to {m(s2,e,'net_per_vehicle').mean():.0f} | {m(s1,e,'payout').mean():.0f} to {m(s2,e,'payout').mean():.0f} | {m(s2,e,'samples').mean():.0f} |")
L += ["", "Paired Wilcoxon within multi-stop:", "", "| Comparison | Deadline | p | Longest gap | p | Net per vehicle | p | Payout | p |", "|---|---|---|---|---|---|---|---|---|"]
for o in ("mean", "fixorder", "ordermean"):
    c = []
    for k in ("deadline_violation", "worst_gap", "net_per_vehicle", "payout"):
        d_, x_ = m(s2, "dist", k), m(s2, o, k)
        c += [f"{(d_.mean()-x_.mean())/abs(x_.mean())*100:+.1f}%", f"{stats.wilcoxon(d_, x_).pvalue:.2g}"]
    L.append(f"| dist vs {o} | " + " | ".join(c) + " |")
open(f"{OUT}/sumo_ms_compare.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
