"""Summary of the greedy baseline against the estimators (same seeds, R = 400) -> results/greedy_summary.md."""
import json, os
import numpy as np
from scipy import stats
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
G = {tuple(a): (o, r) for a, o, r in json.load(open(f"{OUT}/greedy_raw.json")) if "error" not in o}
Lr = json.load(open(f"{OUT}/london_raw.json"))
ANN = [x for x in json.load(open(f"{OUT}/announced_raw.json")) if "error" not in x[1]]
MAIN = {s: {(a[1], a[2]): o for a, o, w in ANN if a[0] == s} for s in ("london1", "london2", "sumo1", "sumo2")}   # estimators paid at the announced reward
NAMES = {"london1": "London model, single stop", "london2": "London model, two stops", "sumo1": "SUMO, single stop", "sumo2": "SUMO, two stops"}
L = ["# Greedy freshness baseline against the estimators", "",
     "greedy: predicted reduction in time past deadline per unit of detour, followed as is (hypothetical full compliance at the same payment rule); one-pass reaction: each driver, in random order, best-responds once (dist evaluation) and commits; reaction to equilibrium: best-response dynamics from the greedy profile. Stability of the greedy profiles is scored at the greedy mission's window states; estimators' stability is in the main summaries.", ""]
for s in ("london1", "sumo2", "london2", "sumo1"):
    seeds = sorted({a[1] for a in G if a[0] == s and all((s, a[1], e) in G for e in ("greedy", "greedy_seq", "greedy_br"))})
    seeds = [m for m in seeds if all((m, e) in MAIN[s] for e in ("dist", "mean", "fixorder", "ordermean"))]
    if not seeds:
        continue
    get = lambda e, k: np.array([(G[(s, m, e)][0] if e.startswith("greedy") else MAIN[s][(m, e)])[k] for m in seeds])
    regs = {e: np.concatenate([G[(s, m, "greedy")][1][e]["win"] for m in seeds]) for e in ("greedy", "greedy_seq", "greedy_br")}
    chg = {e: sum(G[(s, m, e)][0]["changed"] for m in seeds) / max(1, sum(G[(s, m, e)][0]["players"] for m in seeds)) for e in ("greedy_seq", "greedy_br")}
    L += [f"## {NAMES[s]} ({len(seeds)} paired seeds)", "",
          "| Policy | Stable windows | Median window regret, % of R | Drivers changing | Time past deadline | Longest gap, s | Net per vehicle | Payout | Detours |",
          "|---|---|---|---|---|---|---|---|---|"]
    for e in ("dist", "ordermean", "fixorder", "mean", "greedy", "greedy_seq", "greedy_br"):
        st = (f"{(regs[e] <= 0.01).mean():.0%}", f"{np.median(regs[e]) * 100:.1f}") if e in regs else ("see main", "")
        name = {"greedy": "greedy (compliance)", "greedy_seq": "greedy + one-pass reaction", "greedy_br": "greedy + reaction to equilibrium"}.get(e, e)
        ch = f"{chg[e]:.0%}" if e in chg else ("0%" if e == "greedy" else "")
        L.append(f"| {name} | {st[0]} | {st[1]} | {ch} | {get(e, 'deadline_violation').mean():.3f} | {get(e, 'worst_gap').mean():.0f} | "
                 f"{get(e, 'net_per_vehicle').mean():.0f} | {get(e, 'payout').mean():.0f} | {get(e, 'detours').mean():.0f} |")
    L += ["", "| Paired comparison | Time past deadline | p | Net per vehicle | p | Payout |", "|---|---|---|---|---|---|"]
    for a, b in (("greedy", "dist"), ("greedy_seq", "dist"), ("greedy_seq", "greedy"), ("greedy_br", "dist"), ("greedy", "mean"), ("greedy_seq", "mean")):
        c = []
        for k in ("deadline_violation", "net_per_vehicle"):
            x, y = get(a, k), get(b, k)
            c += [f"{(x.mean() / y.mean() - 1) * 100:+.1f}%", f"{stats.wilcoxon(x, y).pvalue:.2g}"]
        c.append(f"{(get(a, 'payout').mean() / get(b, 'payout').mean() - 1) * 100:+.1f}%")
        L.append(f"| {a} vs {b} | " + " | ".join(c) + " |")
    L.append("")
open(f"{OUT}/greedy_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
