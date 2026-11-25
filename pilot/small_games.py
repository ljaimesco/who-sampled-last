"""Small expected-utility games enumerated exhaustively.

For each random game (N vehicles departing in one 60 s window, M PsIs,
lognormal speed noise): enumerate every profile; find all pure equilibria
and whether the best-response graph has a cycle (exact, for the sample-average
game); check what SPA returns; and measure, with FRESH draws, how much a
vehicle could gain by deviating unilaterally from (a) the SPA equilibrium of
the distribution game and (b) the equilibrium of the mean-time game.
"""
from __future__ import annotations

import math
import random
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
from spa_uav import OPT_OUT, exact, spa
from spa_uav.stochastic import StochGame, sample_f


def make(seed, N, M, cv, R, S, mean=False, side=1000.0, v=10.0, alpha=0.4):
    rs = np.random.default_rng(seed)
    psi = rs.uniform(0, side, (M, 2))
    tau = rs.choice((120.0, 180.0, 300.0), M) / math.log(10.0)
    last = -rs.uniform(0, 60, M)
    tdep = rs.uniform(0, 60, N)
    o, d = rs.uniform(0, side, (N, 2)), rs.uniform(0, side, (N, 2))
    dop = np.linalg.norm(o[:, None] - psi[None], axis=2)
    cost = alpha * np.maximum(dop + np.linalg.norm(psi[None] - d[:, None], axis=2)
                              - np.linalg.norm(o - d, axis=1)[:, None], 0.0)
    return psi, tau, last, tdep, dop, cost


def game(inst, cv, R, S, rng, mean=False, v=10.0):
    psi, tau, last, tdep, dop, cost = inst
    N, M = dop.shape
    F = sample_f(rng, cv, (N, S))
    A = tdep[:, None, None] + dop[:, :, None] / (v * F[:, None, :])
    if mean:
        A = A.mean(axis=2, keepdims=True)
    P = [np.zeros((0, A.shape[2])) for _ in range(M)]
    return StochGame(A, P, last, tau, cost, R)


def regret(g, assign, R):
    return max((g.best_response(i, assign)[1] - g.best_response(i, assign)[2]) for i in range(g.N)) / R


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    for (N, M, cv, R) in ((5, 3, 0.3, 400.0), (6, 2, 0.3, 400.0), (5, 3, 0.5, 800.0)):
        no_ne = cyc = spa_ok = brk = 0
        reg_spa, reg_mean, mean_is_true_ne = [], [], 0
        diff_profile = 0
        for s in range(games):
            inst = make(s, N, M, cv, R, 256)
            g = game(inst, cv, R, 256, np.random.default_rng(s))
            ex = exact(g)
            no_ne += int(len(ex["nash"]) == 0)
            cyc += int(ex["br_cycle"])
            gs = game(inst, cv, R, 256, np.random.default_rng(s))
            r = spa(gs, random.Random(s), adapt="global")
            brk += int(r.breaks > 0)
            spa_ok += int(r.breaks == 0 and r.assign in [a for a, _ in ex["nash"]])
            gm = game(inst, cv, R, 1, np.random.default_rng(s), mean=True)
            rm = spa(gm, random.Random(s), adapt="global")
            diff_profile += int(rm.assign != r.assign)
            fresh = game(inst, cv, R, 2048, np.random.default_rng(10_000 + s))
            reg_spa.append(regret(fresh, r.assign, R))
            rmg = regret(fresh, rm.assign, R)
            reg_mean.append(rmg)
            mean_is_true_ne += int(rmg <= 0.005)
        rs_, rm_ = np.array(reg_spa), np.array(reg_mean)
        print(f"N={N} M={M} cv={cv} R={R:.0f}: games={games}  profiles={(M+1)**N}")
        print(f"  no pure equilibrium (sample game): {no_ne}   best-response cycle exists: {cyc}   SPA needed a reward cut: {brk}")
        print(f"  SPA returned one of the enumerated equilibria (no cut): {spa_ok}/{games - brk}")
        print(f"  mean-time equilibrium differs from distribution equilibrium: {diff_profile}/{games}")
        print(f"  max unilateral gain (fresh draws, share of R): SPA-dist mean {rs_.mean():.4f}, p95 {np.quantile(rs_,0.95):.4f} | "
              f"mean-time eq mean {rm_.mean():.4f}, p95 {np.quantile(rm_,0.95):.4f}, max {rm_.max():.3f}")
        print(f"  mean-time equilibrium is an (0.5% of R) equilibrium of the true game in {mean_is_true_ne}/{games}")
