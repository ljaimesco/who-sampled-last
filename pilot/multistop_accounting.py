"""Cost accounting and paired intervals for the multi-stop matched-spending comparisons.
Net payoff per vehicle = (total payments - total detour costs) / N, N = all vehicles in the scenario (opt-outs included),
so total detour cost = payout - N * net. Paired 95% bootstrap CIs (10000 resamples of seeds) for dist minus other,
as a percent of the other's mean; equivalence on payout judged with a 90% CI against a +/-2% tolerance (TOST logic).
Holm adjustment over the 3 outcomes x 2 comparisons (mean, fixorder) per world; primary test = deadline vs mean."""
import json, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav import london_mission as lm
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rng = np.random.default_rng(0)
mt = {tuple(a): o for a, o in json.load(open(f"{OUT}/multistop_matched_raw.json"))}
raw_l = {(a[0] + 5000, a[1]): o for a, o, w in json.load(open(f"{OUT}/multistop_raw.json")) if "error" not in o}
raw_s = {(a[0], a[1]): o for a, o, w in json.load(open(f"{OUT}/sumo_ms_raw.json")) if "error" not in o}
L = ["# Multi-stop matched spending: cost accounting and paired intervals", ""]
def ci(d, x, lvl):
    b = rng.integers(0, len(d), (10000, len(d)))
    r = (d[b].mean(1) - x[b].mean(1)) / x[b].mean(1) * 100
    return np.percentile(r, [(100 - lvl) / 2, 100 - (100 - lvl) / 2])
for world, raw, ev, nfun in (("london", raw_l, range(5000, 5020), lambda s: lm.LondonScenario(s, 0.5).n),
                              ("sumo", raw_s, range(30), lambda s: lm.LondonScenario(5000 + s, 0.5).n)):
    Rd = [a[3] for a in mt if a[0] == world and a[2] == "dist" and a[1] in ev][0]
    N = np.array([nfun(s) for s in ev], float)
    def get(est, k):
        src = (lambda s: mt[(world, s, "dist", Rd)]) if est == "dist" else (lambda s: raw[(s, est)])
        return np.array([src(s)[k] for s in ev])
    L += [f"## {world.upper()} (dist at R = {Rd:.1f}, others at R = 400, {len(ev)} seeds; N = all vehicles, mean {N.mean():.0f})", "",
          "| Estimator | Payout | Detour cost | Samples | Pay per sample | Cost per sample | Net per vehicle |", "|---|---|---|---|---|---|---|"]
    acc = {}
    for e in ("dist", "mean", "fixorder"):
        pay, net, smp = get(e, "payout"), get(e, "net_per_vehicle"), get(e, "samples")
        cost = pay - N * net; acc[e] = dict(payout=pay, cost=cost, net=net, deadline=get(e, "deadline_violation"), gap=get(e, "worst_gap"))
        L.append(f"| {e} | {pay.mean():.0f} | {cost.mean():.0f} | {smp.mean():.0f} | {(pay/smp).mean():.1f} | {(cost/smp).mean():.1f} | {net.mean():.1f} |")
    L += ["", "| dist vs | Payout diff (90% CI) | Within +/-2%? | Detour cost diff (95% CI) | Deadline diff (95% CI) | Holm p | Longest gap (95% CI) | Holm p | Net per vehicle (95% CI) | Holm p |", "|---|---|---|---|---|---|---|---|---|---|"]
    ps, rows = [], []
    for o in ("mean", "fixorder"):
        pl, ph = ci(acc["dist"]["payout"], acc[o]["payout"], 90)
        pd = (acc["dist"]["payout"].mean() / acc[o]["payout"].mean() - 1) * 100
        cd = (acc["dist"]["cost"].mean() / acc[o]["cost"].mean() - 1) * 100; cl, ch = ci(acc["dist"]["cost"], acc[o]["cost"], 95)
        cells = [f"{pd:+.1f}% ({pl:+.1f} to {ph:+.1f})", "yes" if -2 < pl and ph < 2 else "no", f"{cd:+.1f}% ({cl:+.1f} to {ch:+.1f})"]
        for k in ("deadline", "gap", "net"):
            d_, x_ = acc["dist"][k], acc[o][k]
            lo, hi = ci(d_, x_, 95)
            cells.append(f"{(d_.mean()/x_.mean()-1)*100:+.1f}% ({lo:+.1f} to {hi:+.1f})"); cells.append(None)
            ps.append(stats.wilcoxon(d_, x_).pvalue)
        rows.append((o, cells))
    # Holm over the 6 tests in this world
    order = np.argsort(ps); adj = np.empty(6); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (6 - r) * ps[i])); adj[i] = run
    j = 0
    for o, cells in rows:
        out = []
        for c in cells:
            if c is None: out.append(f"{adj[j]:.2g}"); j += 1
            else: out.append(c)
        L.append(f"| {o} | " + " | ".join(out) + " |")
    L.append("")
open(f"{OUT}/multistop_accounting.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
