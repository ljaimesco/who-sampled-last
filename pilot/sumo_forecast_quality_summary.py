"""Summarize the SUMO forecast-quality and matched-spending follow-ups (results/sumo_followup.md)."""
import json, os
import numpy as np
from scipy import stats
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ESTS, TOL = ("dist", "ordermean", "fixorder", "mean"), 0.01
rng = np.random.default_rng(0)

def load(path):
    r = json.load(open(path))
    return {tuple(x[0][:2]): x[1] for x in r if "error" not in x[1]}, [x[2] for x in r if len(x) > 2 and x[2]]

orig, oreg = load(f"{OUT}/sumo_raw.json")
fq, freg = load(f"{OUT}/sumo_fq_raw.json")
seeds = sorted({s for s, e in fq if all((s, x) in fq for x in ESTS)})
col = lambda D, e, k, S=seeds: np.array([D[(s, e)][k] for s in S])
L = ["# SUMO follow-up: forecast quality and matched spending", "",
     f"{len(seeds)} paired seeds. Original = global warm-up forecast (scale prior 1.285, spread x1.5). "
     "Per-seed = forecast refit from a calibration day with the same PsIs and demand pattern (different traffic), no inflation.", "",
     "## Forecast accuracy (all estimators' missions)", "", "| Metric | Original | Per-seed |", "|---|---|---|"]
for k, f in (("forecast_bias", "{:.1f} s"), ("forecast_mae", "{:.1f} s"), ("forecast_cover80", "{:.2f}"), ("inversion_rate", "{:.1%}")):
    a = np.mean([orig[(s, e)][k] for s in seeds for e in ESTS]); b = np.mean([fq[(s, e)][k] for s in seeds for e in ESTS])
    L.append(f"| {k} | {f.format(a)} | {f.format(b)} |")
L += ["", "## Ex-ante stability (forecast model)", "", "| Estimator | Original stable | Per-seed stable |", "|---|---|---|"]
for e in ESTS:
    a = np.concatenate([w[e] for w in oreg]); b = np.concatenate([w[e] for w in freg])
    L.append(f"| {e} | {int((a<=TOL).sum())}/{len(a)} ({(a<=TOL).mean()*100:.0f}%) | {int((b<=TOL).sum())}/{len(b)} ({(b<=TOL).mean()*100:.0f}%) |")
L += ["", "## Realized outcomes: paired Wilcoxon, dist vs each estimator", "",
      "| Forecast | Comparison | Time past deadline | p | Longest gap | p | Net per vehicle | p | Payout | p |", "|---|---|---|---|---|---|---|---|---|---|"]
for name, D in (("original", orig), ("per-seed", fq)):
    for o in ("mean", "fixorder", "ordermean"):
        row = [name, f"dist vs {o}"]
        for k in ("deadline_violation", "worst_gap", "net_per_vehicle", "payout"):
            d, m = col(D, "dist", k), col(D, o, k)
            row += [f"{(d.mean()-m.mean())/m.mean()*100:+.1f}%", f"{stats.wilcoxon(d, m).pvalue:.2g}"]
        L.append("| " + " | ".join(row) + " |")
L += ["", "## Mean outcomes per estimator", "", "| Forecast | Estimator | Time past deadline | Longest gap | Net per vehicle | Payout |", "|---|---|---|---|---|---|"]
for name, D in (("original", orig), ("per-seed", fq)):
    for e in ESTS:
        L.append(f"| {name} | {e} | {col(D,e,'deadline_violation').mean():.3f} | {col(D,e,'worst_gap').mean():.0f} | {col(D,e,'net_per_vehicle').mean():.0f} | {col(D,e,'payout').mean():.0f} |")
if os.path.exists(f"{OUT}/sumo_matched_raw.json") and os.path.exists(f"{OUT}/sumo_matched_R.json"):
    Rd = json.load(open(f"{OUT}/sumo_matched_R.json"))["Rd"]
    mt = {x[0][0]: x[1] for x in json.load(open(f"{OUT}/sumo_matched_raw.json")) if "error" not in x[1]}
    S2 = [s for s in sorted(mt) if (s, "mean") in orig]
    L += ["", f"## Matched spending (original forecast): dist reward {Rd:.1f}, calibrated on seeds 100-103", "",
          "| Comparison | Payout | p | Time past deadline | p | Longest gap | p | Net per vehicle | p |", "|---|---|---|---|---|---|---|---|---|"]
    for o in ("mean", "fixorder"):
        row = [f"dist (matched) vs {o}"]
        for k in ("payout", "deadline_violation", "worst_gap", "net_per_vehicle"):
            d = np.array([mt[s][k] for s in S2]); m = np.array([orig[(s, o)][k] for s in S2])
            row += [f"{(d.mean()-m.mean())/m.mean()*100:+.1f}%", f"{stats.wilcoxon(d, m).pvalue:.2g}"]
        L.append("| " + " | ".join(row) + " |")
open(f"{OUT}/sumo_followup.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
