"""Does the split between "random gaps" and "random predecessor" depend on how
the fixed-order estimator treats draws in which the assumed predecessor
arrives later?  Three conventions, same missions and traffic:
  fixorder       inverted draw -> gap 0
  fixorder_abs   inverted draw -> gap of the same size
  fixorder_cond  inverted draws dropped (gap law given that the order holds)
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav.corr_mission import CorrScenario, run

ESTS = ("dist", "fixorder", "fixorder_abs", "fixorder_cond", "mean")
R, TOL = 400.0, 0.01


def job(args):
    cv, rho, m = args
    scn = CorrScenario(3000 + m, cv, rho)
    out = {}
    for e in ESTS:
        o, reg = run(scn, e, R, S=256, seed=m + 1, regret=(512, 1024) if e == "dist" else None, ests=ESTS)
        out[e] = o
        if reg:
            rg = {x: [w / R for w in reg[x]["win"]] for x in reg}
    return (cv, rho, m), out, rg


if __name__ == "__main__":
    missions = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    configs = [(0.15, 0.0), (0.3, 0.0), (0.3, 0.5)]
    with Pool(2) as pool:
        res = pool.map(job, [(cv, rho, m) for (cv, rho) in configs for m in range(missions)], chunksize=4)
    L = ["| cv | rho | Estimator | Stable windows | Regret P95 | Time past deadline vs mean | p | Net per vehicle vs mean | p |",
         "|---|---|---|---|---|---|---|---|---|"]
    for (cv, rho) in configs:
        sel = sorted([r for r in res if r[0][:2] == (cv, rho)], key=lambda r: r[0][2])
        vm = np.array([r[1]["mean"]["deadline_violation"] for r in sel])
        nm = np.array([r[1]["mean"]["net_per_vehicle"] for r in sel])
        for e in ESTS:
            w = np.concatenate([r[2][e] for r in sel])
            v = np.array([r[1][e]["deadline_violation"] for r in sel])
            n = np.array([r[1][e]["net_per_vehicle"] for r in sel])
            if e == "mean":
                cmp_ = ["", "", "", ""]
            else:
                cmp_ = [f"{100*(v.mean()-vm.mean())/vm.mean():+.1f}%", f"{stats.wilcoxon(v, vm).pvalue:.2g}",
                        f"{100*(n.mean()-nm.mean())/nm.mean():+.1f}%", f"{stats.wilcoxon(n, nm).pvalue:.2g}"]
            L.append(f"| {cv} | {rho} | {e} | {100*(w<=TOL).mean():.0f}% | {100*np.quantile(w,.95):.1f}% | " + " | ".join(cmp_) + " |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "fixorder_variants.md"), "w").write(txt)
    print(txt)
