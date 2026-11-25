"""Summary of the announced-reward rerun -> results/announced_summary.md."""
import json, os
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
D = {tuple(a): (o, r) for a, o, r in json.load(open(f"{OUT}/announced_raw.json")) if "error" not in o}
ESTS = ("dist", "ordermean", "fixorder", "mean")
NAMES = {"syn": "Synthetic (cv 0.3, rho 0.5)", "london1": "London model, single stop", "sumo1": "SUMO, single stop",
         "london2": "London model, two stops", "sumo2": "SUMO, two stops"}
rng = np.random.default_rng(0)
L = ["# Announced reward: stability under R_w and R_0, reward cuts, payments", ""]
for s in ("syn", "london1", "sumo1", "london2", "sumo2"):
    seeds = sorted({a[1] for a in D if a[0] == s and all((s, a[1], e) in D for e in ESTS)})
    if not seeds:
        continue
    L += [f"## {NAMES[s]} ({len(seeds)} missions)", "",
          "| Estimator | Stable under R_w (95% CI) | Stable under R_0 = 400 (95% CI) | Windows with a cut | Mean R_w/R_0 in cut windows | Min R_w/R_0 | Time past deadline | Payout at R_w | Payout at R_0 | Net per vehicle at R_w |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for e in ESTS:
        per = [D[(s, m, "dist")][1][e] for m in seeds]
        w = [np.array(p["win"]) for p in per]; w4 = [np.array(p["win400"]) for p in per]
        cut = np.concatenate([p["cut"] for p in per]); rf = np.concatenate([p["rfinal"] for p in per])
        def ci(xs):
            st = np.array([(x <= 0.01).mean() for x in xs]); n = np.array([len(x) for x in xs])
            tot = (st * n).sum() / n.sum()
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(xs), len(xs)); bs.append((st[i] * n[i]).sum() / n[i].sum())
            return f"{tot:.0%} ({np.percentile(bs, 2.5):.0%} to {np.percentile(bs, 97.5):.0%})"
        o = [D[(s, m, e)][0] for m in seeds]
        g = lambda k: np.mean([x[k] for x in o])
        L.append(f"| {e} | {ci(w)} | {ci(w4)} | {cut.mean():.0%} | {rf[cut == 1].mean() if cut.any() else 1:.2f} | {rf.min():.2f} | "
                 f"{g('deadline_violation'):.3f} | {g('payout'):.0f} | {g('payout_nominal'):.0f} | {g('net_per_vehicle'):.0f} |")
    L.append("")
open(f"{OUT}/announced_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
