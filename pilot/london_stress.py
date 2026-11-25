"""Stronger dependence stress test on the London grid (rho = 0.8, ell = 300 s, 30 paired seeds):
  area      : congestion fully area-wide (w = 1, 1 km cells): strong cross-vehicle dependence; route cv ~0.31
  spread    : no area component, every edge's log-spread scaled by 1.65 (edge means fixed): same route cv, weak dependence
Comparing the two separates the effect of dependence from that of higher travel-time variability."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav import london_mission as lm
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
R, TOL, ESTS = 400.0, 0.01, ("dist", "ordermean", "fixorder", "mean")
CFG = {"area": dict(w=1.0, sig_scale=1.0), "spread": dict(w=0.0, sig_scale=1.65)}
scn_ = lambda c, m: lm.LondonScenario(5000 + m, 0.8, **CFG[c])

def marg_job(a):
    c, m = a; scn = scn_(c, m)
    players = [v for v in range(scn.n) if 600 <= scn.t_dep[v] < 660]
    A, _ = lm.window_samples(scn, players, 600.0, [], 3000, np.random.default_rng(m), R)
    return a, [float((A[i, k] - scn.t_dep[v]).std() / (A[i, k] - scn.t_dep[v]).mean()) for k in range(scn.M)
               for i, v in enumerate(players) if scn.cost[v, k] < R and len(scn.path[v][k]) > 0]

def reg_job(a):
    c, m = a
    _, reg, dg = lm.run(scn_(c, m), "dist", R, S=256, seed=m + 1, regret=(512, 1024), diag=True)
    return a, {e: reg[e]["win"] for e in ESTS}, dg

def out_job(a):
    c, m, e = a
    o, _, _ = lm.run(scn_(c, m), e, R, S=256, seed=m + 1)
    return a, {k: float(o[k]) for k in ("deadline_violation", "worst_gap", "net_per_vehicle", "payout", "inversion_rate")}

if __name__ == "__main__":
    n = 30; t = time.time(); rng = np.random.default_rng(0)
    with Pool(2) as pool:
        marg = pool.map(marg_job, [(c, m) for c in CFG for m in range(10)])
        regs = list(pool.imap_unordered(reg_job, [(c, m) for c in CFG for m in range(n)]))
        print("regret done", round(time.time() - t), flush=True)
        outs = list(pool.imap_unordered(out_job, [(c, m, e) for c in CFG for m in range(n) for e in ESTS]))
    json.dump(dict(marg=marg, regs=[[a, w, d] for a, w, d in regs], outs=outs), open(f"{OUT}/london_stress_raw.json", "w"))
    L = ["# London stress test: stronger dependence vs matched variability (rho = 0.8)", "",
         "| Setting | Median route cv | Pair arrival correlation (median, p90) | Var(Ti - Tj)/(Var Ti + Var Tj) (median, p10) |", "|---|---|---|---|"]
    for c in CFG:
        cv = np.concatenate([o for a, o in marg if a[0] == c])
        co = np.concatenate([d["corr"] for a, w, d in regs if a[0] == c]); ra = np.concatenate([d["ratio"] for a, w, d in regs if a[0] == c])
        L.append(f"| {c} | {np.median(cv):.3f} | {np.median(co):.3f}, {np.percentile(co,90):.3f} | {np.median(ra):.3f}, {np.percentile(ra,10):.3f} |")
    L += ["", "| Setting | Estimator | Stable windows | Stable share (95% CI) | Mean regret | P95 regret |", "|---|---|---|---|---|---|"]
    for c in CFG:
        for e in ESTS:
            per = [np.array(w[e]) for a, w, d in regs if a[0] == c]; x = np.concatenate(per)
            bs = [(np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))]) <= TOL).mean() for _ in range(3000)]
            L.append(f"| {c} | {e} | {int((x<=TOL).sum())}/{len(x)} | {np.percentile(bs,2.5)*100:.1f} to {np.percentile(bs,97.5)*100:.1f}% | {x.mean()*100:.2f}% | {np.percentile(x,95)*100:.1f}% |")
    L += ["", "| Setting | Time past deadline dist vs mean | p | Net per vehicle | p | Payout | p | Inversions dist / mean |", "|---|---|---|---|---|---|---|---|"]
    def col(c, e, k):
        d = {a[1]: o[k] for a, o in outs if a[0] == c and a[2] == e}; return np.array([d[m] for m in sorted(d)])
    for c in CFG:
        row = [c]
        for k in ("deadline_violation", "net_per_vehicle", "payout"):
            d_, m_ = col(c, "dist", k), col(c, "mean", k)
            row += [f"{d_.mean():.3f} vs {m_.mean():.3f} ({(d_.mean()-m_.mean())/m_.mean()*100:+.1f}%)" if k == "deadline_violation" else f"{(d_.mean()-m_.mean())/m_.mean()*100:+.1f}%", f"{stats.wilcoxon(d_, m_).pvalue:.2g}"]
        row.append(f"{col(c,'dist','inversion_rate').mean():.1%} / {col(c,'mean','inversion_rate').mean():.1%}")
        L.append("| " + " | ".join(row) + " |")
    open(f"{OUT}/london_stress.md", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("elapsed", round(time.time() - t))
