"""Small games with EXACT expected utilities (numerical integration, no sampling).

Setting of small_games.py: N vehicles depart in one window, no vehicle en
route, independent lognormal speed factors. For player i at PsI k with rival
set J, conditioning on its own arrival a:

    P(prev <= x | a) = prod_{j in J} [ F_j(x) + 1 - F_j(a) ],   last <= x < a
    delta(a) = (1/tau) * int_last^a exp(-(a - x)/tau) * P(prev <= x | a) dx
    delta    = E_a[ delta(a) ]

The outer expectation uses Gauss-Hermite nodes, the inner integral
Gauss-Legendre nodes. Every profile is then enumerated: pure equilibria,
cycles of the best-response graph, and the smallest "largest unilateral gain"
over all profiles (how far the most stable profile is from an equilibrium).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spa_uav import OPT_OUT, Game, Params, exact
from spa_uav.stochastic import lognorm_params
from small_games import make


class ExactGame(Game):
    def __init__(self, inst, cv, R, nz=120, nx=200, v=10.0):
        psi, tau, last, tdep, dop, cost = inst
        self.p = Params()
        self.N, self.M = dop.shape
        self.tau, self.last, self.tdep, self.c = tau, last, tdep, dop / v
        self.cost = cost
        self.R = np.full(self.M, float(R))
        self.feas = self.cost < self.R[None, :]
        self.options = [[OPT_OUT] + [k for k in range(self.M) if self.feas[i, k]] for i in range(self.N)]
        self.evals = 0
        self.sigma, self.mu = lognorm_params(cv)
        z, wz = np.polynomial.hermite_e.hermegauss(nz)
        self.z, self.wz = z, wz / math.sqrt(2 * math.pi)
        u, wu = np.polynomial.legendre.leggauss(nx)
        self.u, self.wu = (u + 1) / 2, wu / 2
        self._cache = {}

    def F(self, j, k, x):
        """P(vehicle j has reached k by time x)."""
        t = x - self.tdep[j]
        out = np.zeros_like(t, dtype=float)
        ok = t > 0
        out[ok] = ndtr((np.log(t[ok] / self.c[j, k]) + self.mu) / self.sigma)
        return out

    def delta(self, i, k, assign):
        rivals = tuple(j for j, kk in enumerate(assign) if kk == k and j != i)
        key = (i, k, rivals)
        if key in self._cache:
            return self._cache[key]
        a = self.tdep[i] + self.c[i, k] * np.exp(-self.mu - self.sigma * self.z)      # (nz,)
        L = self.last[k]
        x = L + (a[:, None] - L) * self.u[None, :]                                      # (nz, nx)
        G = np.ones_like(x)
        for j in rivals:
            G *= self.F(j, k, x) + 1.0 - self.F(j, k, a)[:, None]
        inner = ((a - L)[:, None] * self.wu[None, :] * np.exp(-(a[:, None] - x) / self.tau[k]) * G).sum(1) / self.tau[k]
        val = float((self.wz * inner).sum())
        self._cache[key] = val
        return val


def analyse(g):
    ex = exact(g)
    best = float("inf")
    import itertools
    for prof in itertools.product(*g.options):
        a = list(prof)
        gain = max(g.best_response(i, a)[1] - g.best_response(i, a)[2] for i in range(g.N))
        best = min(best, gain)
    return len(ex["nash"]), int(ex["br_cycle"]), best


if __name__ == "__main__":
    print("Suspect games (256-draw approximation had no pure equilibrium or a cycle):")
    for (N, M, cv, R, seed) in ((5, 3, 0.3, 400.0, 137), (6, 2, 0.3, 400.0, 59), (6, 2, 0.3, 400.0, 271), (5, 3, 0.5, 800.0, 189)):
        inst = make(seed, N, M, cv, R, 256)
        r1 = analyse(ExactGame(inst, cv, R, 120, 200))
        r2 = analyse(ExactGame(inst, cv, R, 240, 400))
        print(f"  N={N} M={M} cv={cv} R={R:.0f} seed {seed}: pure equilibria {r1[0]}, cycle {r1[1]}, "
              f"smallest largest-gain over profiles {r1[2]:.4f} credits | with doubled nodes: {r2[0]}, {r2[1]}, {r2[2]:.4f}")
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print(f"\nAll {games} games per configuration, exact utilities:")
    for (N, M, cv, R) in ((5, 3, 0.3, 400.0), (6, 2, 0.3, 400.0), (5, 3, 0.5, 800.0)):
        none, cyc, multi = [], [], 0
        for s in range(games):
            g = ExactGame(make(s, N, M, cv, R, 256), cv, R, 80, 120)
            ex = exact(g)
            if not ex["nash"]:
                none.append(s)
            if ex["br_cycle"]:
                cyc.append(s)
            multi += int(len(ex["nash"]) > 1)
        print(f"  N={N} M={M} cv={cv} R={R:.0f}: no pure equilibrium in {len(none)} {none}; best-response cycle in {len(cyc)} {cyc}; "
              f"more than one pure equilibrium in {multi}")
