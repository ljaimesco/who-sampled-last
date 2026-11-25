"""Full missions with correlated traffic: three estimators, matched scenarios.

For each (cv, rho) and each of the missions:
  * run the mission three times, each time playing the window equilibrium
    computed with one estimator (dist, fixorder, mean); same vehicles, options,
    payments and realized traffic;
  * inside the dist mission, score the profiles of all three estimators at
    each window state: unilateral gain in expected utility given the
    information at decision time, best deviation selected on 512 draws and
    evaluated on 1024 independent draws.
Tolerance fixed in advance: a window is "stable" if its largest unilateral
gain is at most 1% of R (R = maximum reward per sample).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav.corr_mission import ESTIMATORS, CorrScenario, run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
R = 400.0
TOL = 0.01


def job(args):
    cv, rho, m, S = args
    scn = CorrScenario(3000 + m, cv, rho)
    rows, regs = [], None
    for est in ESTIMATORS:
        o, reg = run(scn, est, R, S=S, seed=m + 1, regret=(512, 1024) if est == "dist" else None)
        o.update(mission=m, S=S)
        rows.append(o)
        if reg:
            regs = {e: dict(win=[x / R for x in reg[e]["win"]], player=[x / R for x in reg[e]["player"]],
                            cut=reg[e]["cut"], breaks=reg[e]["breaks"]) for e in reg}
    return (cv, rho, m, S), rows, regs


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


if __name__ == "__main__":
    missions = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    os.makedirs(OUT, exist_ok=True)
    configs = [(0.0, 0.0, 256)] + [(cv, rho, 256) for cv in (0.15, 0.3) for rho in (0.0, 0.5, 0.8)]
    configs += [(0.3, 0.5, 64), (0.3, 0.5, 1024)]           # sensitivity to the number of decision draws
    jobs = [(cv, rho, m, S) for (cv, rho, S) in configs for m in range(missions)]
    t = time.time()
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=4)
    rows = [r for _, rs, _ in res for r in rs]
    with open(os.path.join(OUT, "corr_missions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump([dict(cv=k[0], rho=k[1], mission=k[2], S=k[3], regret=g) for k, _, g in res],
              open(os.path.join(OUT, "corr_regret.json"), "w"))
    L = ["# Correlated traffic: missions and window stability", "",
         f"R = {R:.0f} credits (maximum reward per sample), {missions} matched missions per row. "
         f"Regret = largest unilateral gain in a window, as a share of R. Stable = regret <= {100*TOL:.0f}% of R.", ""]
    L += ["## Window stability (matched states, independent evaluation draws)", "",
          "| cv | rho | S | Estimator | Median | P95 | Max | Max without reward cuts | Stable windows | Mean over missions (95% CI) | Per-player P95 | Windows | Windows with a reward cut |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (cv, rho, S) in configs:
        sel = [g for k, _, g in res if k[0] == cv and k[1] == rho and k[3] == S]
        for e in ESTIMATORS:
            wv = np.concatenate([g[e]["win"] for g in sel])
            pv = np.concatenate([g[e]["player"] for g in sel])
            cu = np.concatenate([g[e]["cut"] for g in sel]).astype(bool)
            mm, hh = ci([np.mean(g[e]["win"]) for g in sel])
            L.append(f"| {cv} | {rho} | {S} | {e} | {100*np.median(wv):.2f}% | {100*np.quantile(wv,.95):.1f}% | "
                     f"{100*wv.max():.1f}% | {100*wv[~cu].max():.1f}% | {100*(wv<=TOL).mean():.0f}% | {100*mm:.2f}% ± {100*hh:.2f} | "
                     f"{100*np.quantile(pv,.95):.1f}% | {len(wv)} | {int(cu.sum())} |")
    L += ["", "## Mission outcomes (mean ± 95% CI half-width)", "",
          "| cv | rho | S | Estimator | Time past deadline | Longest gap (s) | Coverage | Net per vehicle | Total payout | Forecast bias per detour | Order inversions |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    f2 = lambda v: "%.3g ± %.2g" % ci(v)
    for (cv, rho, S) in configs:
        for e in ESTIMATORS:
            r = [x for x in rows if x["cv"] == cv and x["rho"] == rho and x["S"] == S and x["est"] == e]
            g = lambda m: [x[m] for x in r]
            L.append(f"| {cv} | {rho} | {S} | {e} | {f2(g('deadline_violation'))} | {f2(g('worst_gap'))} | "
                     f"{np.mean(g('coverage')):.3f} | {f2(g('net_per_vehicle'))} | {f2(g('payout'))} | "
                     f"{f2(g('forecast_bias'))} | {100*np.mean(g('inversion_rate')):.1f}% |")
    L += ["", "## Paired comparisons over missions (Wilcoxon signed-rank, two-sided)", "",
          "| cv | rho | S | Pair | Time past deadline: relative difference | p | Net per vehicle: relative difference | p |",
          "|---|---|---|---|---|---|---|---|"]
    for (cv, rho, S) in configs:
        if cv == 0:
            continue
        get = lambda e, m: np.array([x[m] for x in sorted((x for x in rows if x["cv"] == cv and x["rho"] == rho
                                     and x["S"] == S and x["est"] == e), key=lambda x: x["mission"])])
        for a, b in (("dist", "mean"), ("dist", "fixorder"), ("fixorder", "mean")):
            out = []
            for m in ("deadline_violation", "net_per_vehicle"):
                x, y = get(a, m), get(b, m)
                p = stats.wilcoxon(x, y).pvalue if np.any(x != y) else float("nan")
                out += [f"{100*(x.mean()-y.mean())/y.mean():+.1f}%", f"{p:.2g}"]
            L.append(f"| {cv} | {rho} | {S} | {a} vs {b} | " + " | ".join(out) + " |")
    unstable = sum(x["unstable_games"] for x in rows)
    L += ["", f"Window games in which SPA did not end at a verified equilibrium of its own sample game: {unstable} of "
              f"{sum(x['games'] for x in rows)}. Reward cuts across all missions: {sum(x['breaks'] for x in rows)}.", ""]
    txt = "\n".join(L) + "\n"
    open(os.path.join(OUT, "corr_summary.md"), "w").write(txt)
    print(txt)
    print("elapsed %.0f s" % (time.time() - t))
