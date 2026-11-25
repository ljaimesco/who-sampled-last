"""Summary of the theory check (results/theory_records.npz) -> results/theory_check.md."""
import os
import numpy as np
from scipy import stats
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
z = np.load(f"{OUT}/theory_records.npz")
F = list(z["fields"]); r = z["recs"]; W = z["wins"]
c = {f: r[:, i] for i, f in enumerate(F)}
ESTS = ("dist", "ordermean", "fixorder", "mean")
R = 400.0
import sys
TOL = float(sys.argv[1]) * R if len(sys.argv) > 1 else 0.005 * R   # reversals smaller than TOL are treated as ties
L = ["# Theory check on London single-stop windows (rho = 0.5)", "",
     f"{int(len(np.unique(W[:,0])))} missions, {int(len(W)/4)} windows; {len(r)} (estimator, player, PsI) evaluations on 4096 independent draws.", ""]
# 1. bounds hold
ok_o = np.abs(c["order"]) <= c["pchange"] + 1e-9
ok_p = c["pchange"] <= c["order_b"] + 1e-9
ok_c = np.abs(c["curv"]) <= c["curv_b"] + 1e-9
ok_m = (c["om"] >= -1e-9) & (c["om"] <= c["om_b"] + 1e-9)
L += ["## 1. The moment bounds hold", "",
      "| Statement | Holds in |", "|---|---|",
      f"| order term <= predecessor-change probability | {ok_o.mean():.2%} |",
      f"| predecessor-change probability <= Cantelli sum | {ok_p.mean():.2%} |",
      f"| curvature term <= Taylor + Scarf bound | {ok_c.mean():.2%} |",
      f"| 0 <= ordermean error <= conditional-variance bound | {ok_m.mean():.2%} |", ""]
# 2. size of each term (percent of R, per evaluation of the profile's own options)
pct = lambda x: f"{np.mean(x)*100:.2f}% / {np.percentile(x,95)*100:.1f}%"
sel = c["est"] == 0
L += ["## 2. Size of the terms (mean / 95th percentile, percent of R; dist profiles)", "",
      "| Term | Exact | Bound |", "|---|---|---|",
      f"| order (predecessor change), fixorder and mean | {pct(np.abs(c['order'][sel]))} | {pct(c['order_b'][sel])} |",
      f"| curvature (gap randomness at fixed order), mean | {pct(np.abs(c['curv'][sel]))} | {pct(c['curv_b'][sel])} |",
      f"| conditional curvature, ordermean | {pct(c['om'][sel])} | {pct(c['om_b'][sel])} |",
      f"| predecessor-change probability | {pct(c['pchange'][sel])} | |", ""]
# estimator errors vs truth (includes decision-sampling noise)
L += ["Estimator error |u_hat - u| (credits as percent of R, each estimator's own profiles):", "",
      "| Estimator | Mean | 95th pct | Share of evaluations above 2% of R |", "|---|---|---|---|"]
for e, name in enumerate(ESTS):
    m = c["est"] == e
    err = np.abs(c["uhat"][m] - c["u"][m]) / R
    L.append(f"| {name} | {err.mean()*100:.2f}% | {np.percentile(err,95)*100:.1f}% | {(err>0.02).mean():.1%} |")
L.append("")
# 3. best-response reversals (Lemma 1)
def bound(e, m):
    Rk = c["Rk"][m]
    if e == 0: return np.zeros(m.sum())
    if e == 1: return Rk * c["om_b"][m]
    if e == 2: return Rk * c["order_b"][m]
    return Rk * (c["order_b"][m] + c["curv_b"][m])
def exact(e, m):
    Rk = c["Rk"][m]
    if e == 0: return np.zeros(m.sum())
    if e == 1: return Rk * np.abs(c["om"][m])
    if e == 2: return Rk * np.abs(c["order"][m])
    return Rk * (np.abs(c["order"][m]) + np.abs(c["curv"][m]))
L += ["## 3. Best responses (Lemma 1): reversal needs bound(true best) + bound(chosen) >= true margin", "",
      "A reversal: the estimator's best option is worth at least 0.5% of R less than the true best (opt-out has utility 0).", "",
      "| Estimator | Player decisions | Reversals | Necessary condition holds (exact terms) | Holds (bounds) | Sufficient condition certifies no reversal | Dominant term in reversals |",
      "|---|---|---|---|---|---|---|"]
key = np.stack([c["seed"], c["win"], c["est"], c["i"]], 1)
order = np.lexsort(key.T[::-1]); key = key[order]
cut = np.nonzero(np.any(np.diff(key, axis=0) != 0, axis=1))[0] + 1
groups = np.split(order, cut)
for e, name in enumerate(ESTS):
    nd = nrev = nec_x = nec_b = suff = dom_o = dom_c = 0
    for g in groups:
        if c["est"][g[0]] != e: continue
        m = np.zeros(len(r), bool); m[g] = True
        u = np.concatenate([[0.0], c["u"][g]]); uh = np.concatenate([[0.0], c["uhat"][g]])
        bx = np.concatenate([[0.0], exact(e, m)]); bb = np.concatenate([[0.0], bound(e, m)])
        ks, ke = int(np.argmax(u)), int(np.argmax(uh))
        nd += 1
        srt = np.sort(u)[::-1]; margin = srt[0] - srt[1] if len(u) > 1 else np.inf
        suff += int(margin > 2 * bb.max())
        if u[ks] - u[ke] > TOL:
            nrev += 1
            gap = u[ks] - u[ke]
            nec_x += int(bx[ks] + bx[ke] >= gap); nec_b += int(bb[ks] + bb[ke] >= gap)
            if e == 3:
                o = c["Rk"][g] * np.abs(c["order"][g]); q = c["Rk"][g] * np.abs(c["curv"][g])
                o = np.concatenate([[0], o]); q = np.concatenate([[0], q])
                if o[ks] + o[ke] >= q[ks] + q[ke]: dom_o += 1
                else: dom_c += 1
    dom = f"order {dom_o}, curvature {dom_c}" if e == 3 else ("order" if e == 2 else ("curvature" if e == 1 else "sampling"))
    L.append(f"| {name} | {nd} | {nrev} ({nrev/nd:.1%}) | {nec_x}/{nrev} | {nec_b}/{nrev} | {suff/nd:.1%} | {dom} |")
L.append("")
# 4. windows: approximation terms vs regret (indices fixed before looking at the result: mean over the window's evaluations)
L += ["## 4. Windows: larger approximation terms go with instability", "",
      "Per window and estimator, two indices over all the players' options under the estimator's own profile: the mean exact error term and the mean moment bound (percent of R). Thirds are by rank.", "",
      "| Estimator | Spearman with regret: exact / bound | Unstable share by third of exact index (low / middle / high) | By third of bound index |", "|---|---|---|---|"]
for e, name in enumerate(ESTS[1:], 1):
    wi = W[W[:, 2] == e]
    ix, ib = [], []
    me = c["est"] == e
    for (s_, w_) in wi[:, :2]:
        m = me & (c["seed"] == s_) & (c["win"] == w_)
        ix.append(exact(e, m).mean() / R if m.any() else 0.0); ib.append(bound(e, m).mean() / R if m.any() else 0.0)
    ix, ib, reg = np.array(ix), np.array(ib), wi[:, 3]
    def thirds(x):
        rk = stats.rankdata(x, method="ordinal"); t = np.minimum((3 * (rk - 1)) // len(x), 2)
        return " / ".join(f"{(reg[t==q] > 0.01).mean():.0%}" for q in range(3))
    L.append(f"| {name} | {stats.spearmanr(ix, reg).correlation:.2f} / {stats.spearmanr(ib, reg).correlation:.2f} | {thirds(ix)} | {thirds(ib)} |")
L.append("")
L += ["## 5. Why predecessor uncertainty is small or large", "",
      "Exact predecessor-change probability by Var(T_i - T_p) relative to the expected gap (dist profiles, PsIs with a predicted predecessor):", "",
      "| sd(T_i - T_p) / E(T_i - T_p) | Evaluations | Mean predecessor-change probability | Mean order bound |", "|---|---|---|---|"]
m = sel & (c["has_pred"] == 1) & (c["uM"] < 1 - 1e-9)
meanD = -c["tau"][m] * np.log(1 - c["uM"][m])            # u_M = f(E D)  =>  E D = -tau log(1 - u_M)
ratio = np.sqrt(c["varD"][m]) / np.maximum(meanD, 1e-9)
for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, np.inf)):
    b = (ratio >= lo) & (ratio < hi)
    if b.any():
        L.append(f"| {lo:g} to {hi:g} | {b.sum()} | {c['pchange'][m][b].mean():.2f} | {c['order_b'][m][b].mean():.2f} |")
L.append("")
open(f"{OUT}/theory_check.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
