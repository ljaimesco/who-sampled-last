"""London-calibrated correlation sensitivity experiment (single-stop decisions).

Per rho in {0, 0.5, 0.8} (ell = 300 s), same 30 scenario seeds (geometry and demand do not depend on rho):
  1. marginals: route travel-time mean, sd, p10, p90 for the same routes under each rho;
  2. dependence: arrival correlation and Var(Ti - Tj) / (Var Ti + Var Tj) for co-located pairs;
  3. strategy: ex-ante window regret and stable windows for dist, ordermean, fixorder, mean
     (all solve the same decision draws; scored on independent 512 / 1024 draws);
  4. outcomes: one mission per estimator, paired by seed (same true world), with mission-level
     bootstrap intervals and Wilcoxon tests.
"""
import json, math, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav import london_mission as lm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
R, TOL, RHOS, ESTS = 400.0, 0.01, (0.0, 0.5, 0.8), ("dist", "ordermean", "fixorder", "mean")


def marg_job(a):
    rho, m = a
    scn = lm.LondonScenario(5000 + m, rho)
    players = [v for v in range(scn.n) if 600 <= scn.t_dep[v] < 660]
    A, _ = lm.window_samples(scn, players, 600.0, [], 4000, np.random.default_rng(m), R)
    out = []
    for a_, v in enumerate(players):
        for k in range(scn.M):
            if scn.cost[v, k] < R and len(scn.path[v][k]) > 0:
                x = A[a_, k] - scn.t_dep[v]
                out.append((v, k, x.mean(), x.std(), np.percentile(x, 10), np.percentile(x, 90)))
    return a, out


def reg_job(a):
    rho, m = a
    _, reg, dg = lm.run(lm.LondonScenario(5000 + m, rho), "dist", R, S=256, seed=m + 1, regret=(512, 1024), diag=True)
    return a, {e: reg[e]["win"] for e in ESTS}, {"corr": dg["corr"], "ratio": dg["ratio"]}


def out_job(a):
    rho, m, est = a
    o, _, _ = lm.run(lm.LondonScenario(5000 + m, rho), est, R, S=256, seed=m + 1)
    return a, {k: float(o[k]) for k in ("deadline_violation", "worst_gap", "net_per_vehicle", "payout", "inversion_rate", "detours")}


def boot(per, f, rng, n=3000):
    v = [f(np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))])) for _ in range(n)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    t = time.time()
    with Pool(2) as pool:
        marg = pool.map(marg_job, [(rho, m) for rho in RHOS for m in range(min(n, 10))])
        print("marginals done", round(time.time() - t), flush=True)
        regs = []
        for r in pool.imap_unordered(reg_job, [(rho, m) for rho in RHOS for m in range(n)]):
            regs.append(r)
            if len(regs) % 10 == 0: print("regret missions", len(regs), round(time.time() - t), flush=True)
        outs = []
        for r in pool.imap_unordered(out_job, [(rho, m, e) for rho in RHOS for m in range(n) for e in ESTS]):
            outs.append(r)
            if len(outs) % 40 == 0: print("outcome missions", len(outs), round(time.time() - t), flush=True)
    json.dump(dict(marg=[[list(a), o] for a, o in marg], regs=[[list(a), w, d] for a, w, d in regs],
                   outs=[[list(a), o] for a, o in outs]), open(f"{OUT}/london_raw.json", "w"))
    rng = np.random.default_rng(0)
    L = ["# London-calibrated correlation sensitivity experiment", "",
         f"ANTE-Mobility London grid (Uber per-edge travel-time means and stds), central 2 km x 2 km, 20 PsIs, single-stop decisions, "
         f"{n} paired scenario seeds per rho, congestion correlation length 300 s. Intervals: 95% mission-clustered bootstrap.", "",
         "## 1. Route travel-time marginals (same routes, 4000 draws each)", "",
         "| rho | Routes | Median relative change in mean vs rho = 0 | in sd | in p10 | in p90 | Median route cv |", "|---|---|---|---|---|---|---|"]
    base = {(a[1], v, k): (mu, sd, p1, p9) for a, o in marg if a[0] == 0.0 for (v, k, mu, sd, p1, p9) in o}
    for rho in RHOS:
        rows = [(base[(a[1], v, k)], (mu, sd, p1, p9)) for a, o in marg if a[0] == rho for (v, k, mu, sd, p1, p9) in o]
        rel = np.array([[(y[i] - x[i]) / x[i] for i in range(4)] for x, y in rows])
        cvs = np.array([y[1] / y[0] for x, y in rows])
        L.append(f"| {rho} | {len(rows)} | {np.median(rel[:,0])*100:+.2f}% | {np.median(rel[:,1])*100:+.2f}% | {np.median(rel[:,2])*100:+.2f}% | {np.median(rel[:,3])*100:+.2f}% | {np.median(cvs):.3f} |")
    L += ["", "## 2. Dependence relevant to order flips (co-located pairs in each window, decision draws)", "",
          "| rho | Pairs | Arrival correlation (median, p90) | Var(Ti - Tj) / (Var Ti + Var Tj) (median, p10) |", "|---|---|---|---|"]
    for rho in RHOS:
        c = np.concatenate([d["corr"] for a, w, d in regs if a[0] == rho]); r = np.concatenate([d["ratio"] for a, w, d in regs if a[0] == rho])
        L.append(f"| {rho} | {len(c)} | {np.median(c):.3f}, {np.percentile(c,90):.3f} | {np.median(r):.3f}, {np.percentile(r,10):.3f} |")
    L += ["", "## 3. Ex-ante window regret (matched states)", "",
          "| rho | Estimator | Stable windows | Stable share (95% CI) | Mean regret (95% CI) | P95 regret |", "|---|---|---|---|---|---|"]
    for rho in RHOS:
        for e in ESTS:
            per = [np.array(w[e]) for a, w, d in regs if a[0] == rho]; x = np.concatenate(per)
            s_lo, s_hi = boot(per, lambda z: (z <= TOL).mean(), rng); m_lo, m_hi = boot(per, np.mean, rng)
            L.append(f"| {rho} | {e} | {int((x<=TOL).sum())}/{len(x)} | {s_lo*100:.1f} to {s_hi*100:.1f}% | {x.mean()*100:.2f}% ({m_lo*100:.2f} to {m_hi*100:.2f}) | {np.percentile(x,95)*100:.1f}% |")
    L += ["", "## 4. Mission outcomes (paired seeds; mean over missions, 95% bootstrap CI)", "",
          "| rho | Estimator | Time past deadline | Longest gap (s) | Net per vehicle | Payout | Order inversions |", "|---|---|---|---|---|---|---|"]
    def col(rho, e, k):
        d = {a[1]: o[k] for a, o in outs if a[0] == rho and a[2] == e}
        return np.array([d[m] for m in sorted(d)])
    for rho in RHOS:
        for e in ESTS:
            cells = []
            for k in ("deadline_violation", "worst_gap", "net_per_vehicle", "payout", "inversion_rate"):
                x = col(rho, e, k); b = [rng.choice(x, len(x)).mean() for _ in range(3000)]
                fmt = "{:.3f}" if k == "deadline_violation" else ("{:.1%}" if k == "inversion_rate" else "{:.0f}")
                cells.append(f"{fmt.format(x.mean())} ({fmt.format(np.percentile(b,2.5))} to {fmt.format(np.percentile(b,97.5))})")
            L.append(f"| {rho} | {e} | " + " | ".join(cells) + " |")
    L += ["", "Paired Wilcoxon, dist vs mean:", "", "| rho | Time past deadline | p | Net per vehicle | p | Payout | p |", "|---|---|---|---|---|---|---|"]
    for rho in RHOS:
        row = [str(rho)]
        for k in ("deadline_violation", "net_per_vehicle", "payout"):
            d_, m_ = col(rho, "dist", k), col(rho, "mean", k)
            row += [f"{(d_.mean()-m_.mean())/m_.mean()*100:+.1f}%", f"{stats.wilcoxon(d_, m_).pvalue:.2g}"]
        L.append("| " + " | ".join(row) + " |")
    open(f"{OUT}/london_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L)); print("elapsed", round(time.time() - t))
