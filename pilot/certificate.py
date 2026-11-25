"""Certificates of non-existence of a pure-strategy equilibrium (finite action set as modeled).

For each game: enumerate EVERY action profile; for each, find a player and a
unilateral deviation with positive gain, using exact numerical integration
at three node resolutions. The certificate holds if the smallest such gain
(over profiles) exceeds twice the largest change in any utility between
resolutions (each gain is a difference of two utilities). An independent
Monte Carlo check (2e6 draws) re-evaluates the tightest profile."""
import csv, itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exact_small import ExactGame
from small_games import make
from spa_uav import OPT_OUT
from spa_uav.stochastic import lognorm_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RES = ((100, 200), (150, 300), (200, 400))

def table(g):
    profs = list(itertools.product(*g.options))
    U = {}
    for p in profs:
        a = list(p)
        for i in range(g.N):
            for k in g.options[i]:
                b = a.copy(); b[i] = k
                U[(p, i, k)] = g.utility(i, k, b)
    return profs, U

def mc_utility(inst, cv, R, i, k, prof, n=2_000_000, v=10.0, seed=7):
    psi, tau, last, tdep, dop, cost = inst
    if k == OPT_OUT: return 0.0
    rng = np.random.default_rng(seed); s, mu = lognorm_params(cv)
    T = lambda j: tdep[j] + dop[j, k] / (v * np.exp(mu + s * rng.standard_normal(n)))
    a = T(i); prev = np.full(n, last[k])
    for j, kk in enumerate(prof):
        if kk == k and j != i:
            aj = T(j); ok = (aj < a) & (aj > prev); prev = np.where(ok, aj, prev)
    return R * float(np.mean(1 - np.exp(-(a - prev) / tau[k]))) - cost[i, k]

if __name__ == "__main__":
    lines = []
    for (N, M, cv, R, seed) in ((6, 2, 0.3, 400.0, 271), (5, 3, 0.5, 800.0, 276)):
        inst = make(seed, N, M, cv, R, 256)
        games = [ExactGame(inst, cv, R, nz, nx) for nz, nx in RES]
        tabs = [table(g) for g in games]
        assert all(np.isfinite(list(t[1].values())).all() for t in tabs)
        profs = tabs[0][0]
        err = max(abs(tabs[-1][1][key] - tabs[-2][1][key]) for key in tabs[-1][1])
        err_coarse = max(abs(tabs[1][1][key] - tabs[0][1][key]) for key in tabs[1][1])
        U = tabs[-1][1]
        rows, margin, tight = [], np.inf, None
        for p in profs:
            best = (-np.inf, None, None)
            for i in range(N):
                for k in games[-1].options[i]:
                    gain = U[(p, i, k)] - U[(p, i, p[i])]
                    if gain > best[0]: best = (gain, i, k)
            rows.append((p, *best))
            if best[0] < margin: margin, tight = best[0], (p, best[1], best[2])
        with open(f"{OUT}/certificate_seed{seed}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["profile (PsI per vehicle, -1 = opt out)", "deviating vehicle", "deviation", "gain (credits)"])
            for p, gn, i, k in rows: w.writerow([list(p), i, k, f"{gn:.4f}"])
        p, i, k = tight
        mc_dev, mc_cur = mc_utility(inst, cv, R, i, k, p), mc_utility(inst, cv, R, i, p[i], p, seed=8)
        ok = margin > 2 * err
        lines.append(f"seed {seed} (N={N}, M={M}, cv={cv}, R={R:.0f}): {len(profs)} profiles, every one has a profitable deviation; "
                     f"smallest gain {margin:.4f} credits ({margin/R*100:.3f}% of R) at profile {list(p)}, vehicle {i} -> {k}; "
                     f"largest utility change between the two finest resolutions {err:.2e} (coarse pair {err_coarse:.2e}); "
                     f"certificate {'HOLDS' if ok else 'FAILS'} (margin/2err = {margin/(2*err):.1e}); "
                     f"Monte Carlo 2e6 draws on the tightest deviation: gain {mc_dev-mc_cur:.4f}")
    open(f"{OUT}/certificates.md", "w").write("\n\n".join(lines) + "\n")
    print("\n\n".join(lines))
