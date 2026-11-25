"""Strategic competition for freshness with uncertain arrival times.

One decision window is one game among the vehicles that depart in it. Each
vehicle chooses one PsI to detour to, or opts out. Travel times are random,
so the gap to the previous sample, and who the previous sampler is, are
random too. A vehicle's payoff for PsI k is

    R_k * E[ 1 - exp(-(a_i - prev_i) / tau_k) ] - c_ik,

where prev_i is the latest sample at k before a_i: the last realized sample,
a vehicle from an earlier window still en route to k, or another player of
this window who picked k and arrives first.

The expectation is a sample average over S fixed draws, so each game is a
finite, deterministic APPROXIMATION of the expected-utility game. Payoffs stay
affine in R, so the corrected SPA and break_cycle apply unchanged.

Three estimators of the payoff (same code path, different inputs):
  dist      full joint samples: random gaps and random predecessor;
  fixorder  random gaps, but the predecessor is whoever is ahead ON AVERAGE
            (captures the concavity of the reward, ignores order flips);
  mean      every arrival replaced by its mean (pass S = 1 arrays).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .model import OPT_OUT, Game, Params


def lognorm_params(cv: float):
    s2 = math.log(1.0 + cv * cv)
    return math.sqrt(s2), -s2 / 2.0


def sample_f(rng, cv, size, fmax=None):
    """Speed factors (mean 1); optionally conditioned on f < fmax."""
    if cv == 0:
        return np.ones(size)
    sigma, mu = lognorm_params(cv)
    if fmax is None:
        return np.exp(mu + sigma * rng.standard_normal(size))
    p = stats.norm.cdf((math.log(fmax) - mu) / sigma) if fmax > 0 else 0.0
    u = rng.uniform(1e-12, max(p, 1e-12), size)
    return np.exp(mu + sigma * stats.norm.ppf(u))


class StochGame(Game):
    """Sample-average expected-utility game of one window. Reuses
    Game.best_response, utility, is_nash, welfare and break_cycle."""

    def __init__(self, arr_own, arr_pend, last, tau, cost, R, params: Params = Params(),
                 mode: str = "dist", rivals: bool = True):
        """arr_own:  (N, M, S) sampled arrival times of the players at each PsI
        arr_pend: list of M arrays (n_k, S), vehicles en route to k
        last:     (M,) time of the last realized sample at each PsI
        tau:      (M,) freshness constants;  cost: (N, M) detour costs."""
        self.p, self.mode, self.rivals = params, mode, rivals
        self.N, self.M, self.S = arr_own.shape
        self.A, self.P, self.last = arr_own, arr_pend, np.asarray(last, float)
        self.tau = np.asarray(tau, float)
        self.cost = np.asarray(cost, float)
        self.R = np.full(self.M, float(R)) if np.isscalar(R) else np.asarray(R, float).copy()
        self.feas = self.cost < self.R[None, :]          # a detour costing more than R can never pay
        self.options = [[OPT_OUT] + [k for k in range(self.M) if self.feas[i, k]] for i in range(self.N)]
        self.evals = 0
        self._cache = {}
        if mode.startswith("fixorder"):
            self.Am = self.A.mean(axis=2)
            self.Pm = [P.mean(axis=1) if len(P) else np.zeros(0) for P in self.P]
            return
        if mode == "ordermean":
            return
        # part of prev that does not depend on the other players: last sample and vehicles en route
        self.base = np.empty((self.N, self.M, self.S))
        for k in range(self.M):
            b = np.full((self.N, self.S), self.last[k])
            for row in (self.P[k] if rivals else []):     # en-route vehicle arriving before me
                ok = (row[None, :] < self.A[:, k, :]) & (row[None, :] > b)
                b = np.where(ok, row[None, :], b)
            self.base[:, k, :] = b

    def delta(self, i, k, assign):
        """Expected freshness multiplier (sample average)."""
        rivals = tuple(j for j, kk in enumerate(assign) if kk == k and j != i) if self.rivals else ()
        key = (i, k, rivals)
        v = self._cache.get(key)
        if v is not None:
            return v
        a = self.A[i, k]
        if self.mode == "ordermean":
            # ablation: the predecessor's identity stays random, but within each identity the gap is
            # replaced by its conditional mean (removes gap randomness, keeps order uncertainty)
            prev = np.full(self.S, self.last[k]); pid = np.full(self.S, -1)
            for r, row in enumerate(self.P[k]):
                ok = (row < a) & (row > prev)
                prev, pid = np.where(ok, row, prev), np.where(ok, 10_000 + r, pid)
            for j in rivals:
                aj = self.A[j, k]
                ok = (aj < a) & (aj > prev)
                prev, pid = np.where(ok, aj, prev), np.where(ok, j, pid)
            gap = np.maximum(a - prev, 0.0)
            v = 0.0
            for c in np.unique(pid):
                m = pid == c
                v += m.mean() * (1.0 - math.exp(-gap[m].mean() / self.tau[k]))
            v = float(v)
            self._cache[key] = v
            return v
        if self.mode.startswith("fixorder"):
            am, pm, pred = self.Am[i, k], self.last[k], self.last[k]
            for row, m in zip(self.P[k], self.Pm[k]):
                if pm < m < am:
                    pm, pred = m, row
            for j in rivals:
                m = self.Am[j, k]
                if (pm, -1) < (m, j) < (am, i):
                    pm, pred = m, self.A[j, k]
            gap = a - pred
            if self.mode == "fixorder_abs":          # an inverted draw counts as a gap of the same size
                gap = np.abs(gap)
            elif self.mode == "fixorder_cond" and np.any(gap > 0):   # keep only draws in which the order holds
                gap = gap[gap > 0]
            else:                                    # an inverted draw counts as a zero gap
                gap = np.maximum(gap, 0.0)
        else:
            prev = self.base[i, k].copy()
            for j in rivals:
                aj = self.A[j, k]
                ok = (aj < a) & (aj > prev)
                prev = np.where(ok, aj, prev)
            gap = np.maximum(a - prev, 0.0)
        v = float(np.mean(1.0 - np.exp(-gap / self.tau[k])))
        self._cache[key] = v
        return v

    def break_cycle(self, states, moves, mode="global"):
        r, t = super().break_cycle(states, moves, mode)
        self.feas = self.cost < self.R[None, :]
        return r, t


def profile_regret(g_sel: StochGame, g_eval: StochGame, assign):
    """Unilateral gains available at `assign`, per player, in credits.

    The best deviation is SELECTED on one sample set (g_sel) and its gain is
    EVALUATED on an independent one (g_eval), so the estimate is not inflated
    by taking a maximum over noisy utilities. It can be slightly negative."""
    out = []
    for i in range(g_sel.N):
        cur = assign[i]
        opts = set(g_sel.options[i]) | {cur}
        best = max(opts, key=lambda k: (g_sel.utility(i, k, assign), k == cur))
        out.append(0.0 if best == cur else
                   g_eval.utility(i, best, assign) - g_eval.utility(i, cur, assign))
    return out
