"""Small games, re-checked.

1. Regret with selection and evaluation on independent draws (1024 / 4096),
   for the equilibria computed with the three estimators; reported as median,
   p95, max and share within the fixed tolerance (1% of R), over all games and
   over the unstable ones only. R is the maximum reward per sample.
2. Games whose 256-draw approximation had no pure equilibrium or had a
   best-response cycle are enumerated again with 8192 draws, three times with
   independent draws.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav import exact, spa
from spa_uav.stochastic import StochGame, profile_regret, sample_f
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from small_games import make

TOL = 0.01


def game(inst, cv, R, S, rng, est="dist", v=10.0):
    psi, tau, last, tdep, dop, cost = inst
    N, M = dop.shape
    F = sample_f(rng, cv, (N, S))
    A = tdep[:, None, None] + dop[:, :, None] / (v * F[:, None, :])
    if est == "mean":
        A = A.mean(axis=2, keepdims=True)
    P = [np.zeros((0, A.shape[2])) for _ in range(M)]
    return StochGame(A, P, last, tau, cost, R, mode="fixorder" if est == "fixorder" else "dist")


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    for (N, M, cv, R) in ((5, 3, 0.3, 400.0), (6, 2, 0.3, 400.0), (5, 3, 0.5, 800.0)):
        reg = {e: [] for e in ("dist", "fixorder", "mean")}
        cut = {e: 0 for e in reg}
        suspects = []
        for s in range(games):
            inst = make(s, N, M, cv, R, 256)
            ex = exact(game(inst, cv, R, 256, np.random.default_rng(s)))
            if not ex["nash"] or ex["br_cycle"]:
                suspects.append((s, len(ex["nash"]), ex["br_cycle"]))
            g_sel = game(inst, cv, R, 1024, np.random.default_rng(20_000 + s))
            g_eval = game(inst, cv, R, 4096, np.random.default_rng(40_000 + s))
            for e in reg:
                g = game(inst, cv, R, 256, np.random.default_rng(s), e)
                r = spa(g, random.Random(s), adapt="global")
                cut[e] += int(r.breaks > 0)
                reg[e].append(max(profile_regret(g_sel, g_eval, r.assign)) / R)
        print(f"N={N} M={M} cv={cv} R={R:.0f}, {games} games")
        for e in reg:
            x = np.array(reg[e])
            bad = x[x > TOL]
            print(f"  {e:8s} all games: median {100*np.median(x):.2f}%  p95 {100*np.quantile(x,.95):.1f}%  max {100*x.max():.1f}%  "
                  f"mean {100*x.mean():.2f}%  within 1% of R: {100*(x<=TOL).mean():.0f}%  | unstable only ({len(bad)}): "
                  f"median {100*np.median(bad) if len(bad) else 0:.1f}%  mean {100*bad.mean() if len(bad) else 0:.1f}%  | reward cuts {cut[e]}")
        for (s, n_ne, cyc) in suspects:
            inst = make(s, N, M, cv, R, 256)
            res = []
            for rep in range(3):
                ex = exact(game(inst, cv, R, 8192, np.random.default_rng(90_000 + 10 * s + rep)))
                res.append((len(ex["nash"]), int(ex["br_cycle"])))
            print(f"  seed {s}: 256 draws -> equilibria {n_ne}, cycle {int(cyc)};  8192 draws x3 -> (equilibria, cycle) {res}")
