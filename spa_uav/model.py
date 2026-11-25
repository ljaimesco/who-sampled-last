"""Game model for delivery-constrained, freshness-driven UAV sensing.

One decision epoch is one fixed game:
  * each UAV i flies a delivery leg orig[i] -> dest[i] and may detour to at
    most one intersection (PsI) k, or opt out (option -1);
  * a detour is feasible only if it respects the delivery slack and leaves the
    battery reserve intact;
  * the reward for a sample is R[k] * delta, where delta grows with the time
    since the previous sample at k (committed history or another UAV in this
    epoch that arrives earlier);
  * the cost is the extra energy of the detour, converted to credits.

Arrival times and the committed history are frozen for the whole epoch, so
utilities depend only on the joint assignment and on R.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np

OPT_OUT = -1


@dataclass(frozen=True)
class Params:
    """Physical and economic constants. All values are placeholders to be
    replaced by the values the co-authors confirm."""
    speed: float = 15.0      # cruise speed, m/s
    p_cruise: float = 360.0  # cruise power, W  (360 W = 0.1 Wh/s)
    p_hover: float = 400.0   # hover power, W
    dwell: float = 2.0       # hover time per observation, s
    kappa: float = 180.0     # price of energy, credits per Wh
    reserve: float = 20.0    # battery reserve that must remain, Wh
    eps: float = 1e-9        # tolerance for "strict" improvement

    def energy_wh(self, dist_m: float, hover_s: float = 0.0) -> float:
        """distance -> time -> energy (Wh)."""
        return (self.p_cruise * dist_m / self.speed + self.p_hover * hover_s) / 3600.0


@dataclass
class Epoch:
    """Exogenous data of one decision epoch."""
    t0: float               # decision time, s
    orig: np.ndarray        # (N, 2) leg origins, m
    dest: np.ndarray        # (N, 2) leg destinations, m
    energy: np.ndarray      # (N,) battery energy available at t0, Wh
    slack: np.ndarray       # (N,) delivery delay the operator tolerates, s
    psi: np.ndarray         # (M, 2) intersection positions, m
    tau: np.ndarray         # (M,) freshness recovery constant per intersection, s
    history: list           # M sorted lists of committed sample times, s


class Game:
    """The one-deviation allocation game of a single epoch."""

    def __init__(self, ep: Epoch, R, params: Params = Params(), enforce: bool = True):
        self.ep, self.p = ep, params
        self.N, self.M = len(ep.orig), len(ep.psi)
        self.R = np.full(self.M, float(R)) if np.isscalar(R) else np.asarray(R, float).copy()
        self.tau = np.asarray(ep.tau, float)
        self.hist = ep.history

        d_op = np.linalg.norm(ep.orig[:, None, :] - ep.psi[None, :, :], axis=2)   # (N, M)
        d_pd = np.linalg.norm(ep.psi[None, :, :] - ep.dest[:, None, :], axis=2)
        d_od = np.linalg.norm(ep.orig - ep.dest, axis=1)[:, None]
        self.detour = np.maximum(d_op + d_pd - d_od, 0.0)                         # extra metres
        self.arr = ep.t0 + d_op / params.speed                                   # arrival at PsI
        self.delay = self.detour / params.speed + params.dwell                   # extra seconds
        extra_wh = (params.p_cruise * self.detour / params.speed
                    + params.p_hover * params.dwell) / 3600.0
        self.cost = params.kappa * extra_wh                                      # credits
        leg_wh = params.p_cruise * d_od / params.speed / 3600.0
        ok_time = self.delay <= np.asarray(ep.slack, float)[:, None]
        ok_energy = leg_wh + extra_wh <= (np.asarray(ep.energy, float)[:, None] - params.reserve)
        self.truly_feasible = ok_time & ok_energy
        # enforce=False reproduces the original SPA, which ignores both limits
        self.feas = self.truly_feasible if enforce else np.ones_like(self.truly_feasible)
        self.options = [[OPT_OUT] + [k for k in range(self.M) if self.feas[i, k]]
                        for i in range(self.N)]
        self.evals = 0  # number of utility evaluations (for the scaling study)

    # ---- utilities -------------------------------------------------------
    def delta(self, i: int, k: int, assign) -> float:
        """Freshness multiplier of UAV i sampling PsI k, given the others."""
        a = self.arr[i, k]
        prev = -math.inf
        h = self.hist[k]
        j = bisect.bisect_right(h, a)
        if j > 0:
            prev = h[j - 1]
        for other, ko in enumerate(assign):
            if other != i and ko == k:
                ao = self.arr[other, k]
                if (ao, other) < (a, i) and ao > prev:
                    prev = ao
        if prev == -math.inf:
            return 1.0
        return 1.0 - math.exp(-(a - prev) / self.tau[k])

    def utility(self, i: int, k: int, assign) -> float:
        self.evals += 1
        if k == OPT_OUT:
            return 0.0
        return self.R[k] * self.delta(i, k, assign) - self.cost[i, k]

    def best_response(self, i: int, assign):
        """Return (best option, its utility, current utility).

        The current choice is kept unless another option is strictly better
        (by more than eps). Among equally good alternatives, opting out wins,
        then the lowest index, so the rule is deterministic."""
        cur = assign[i]
        u_cur = self.utility(i, cur, assign)
        best, u_best = cur, u_cur
        for k in self.options[i]:
            if k == cur:
                continue
            u = self.utility(i, k, assign)
            if u > u_best + self.p.eps:
                best, u_best = k, u
        return best, u_best, u_cur

    def is_nash(self, assign) -> bool:
        """Pure-strategy Nash equilibrium check by exhaustive unilateral deviation."""
        return all(self.best_response(i, assign)[0] == assign[i] for i in range(self.N))

    def welfare(self, assign) -> float:
        """Sum of UAV utilities (platform value R*delta minus detour costs)."""
        return sum(self.utility(i, k, assign) for i, k in enumerate(assign))

    # ---- reward adaptation ------------------------------------------------
    def break_cycle(self, states, moves, mode: str = "global"):
        """Lower the reward just enough to disable one transition of a cycle.

        states[t] is the assignment before moves[t] = (i, x, y).
        A move into y is a best response only while
            R_y * d_y - C_y > max(R_x * d_x - C_x, 0).
        global: one scalar R. The move dies once R <= C_y/d_y (y no better than
                opting out) or, if d_y > d_x, once R <= (C_y - C_x)/(d_y - d_x).
                We take the largest such threshold over the cycle, i.e. the
                smallest reduction. Every cycle has a move with y != OPT_OUT, so
                a threshold always exists, and it is strictly below R.
        local:  only R_y of one intersection is lowered, to
                (max(u_x, 0) + C_y) / d_y. Experimental: termination unproven.
        Returns (new reward value, index of the disabled move)."""
        best = None
        for t, (s, (i, x, y)) in enumerate(zip(states, moves)):
            if y == OPT_OUT:
                continue
            a = list(s)
            dy, cy = self.delta(i, y, a), self.cost[i, y]
            dx, cx = (self.delta(i, x, a), self.cost[i, x]) if x != OPT_OUT else (0.0, 0.0)
            if dy <= 0.0:
                continue
            if mode == "global":
                r = cy / dy
                if dy > dx and cy > cx:
                    r = max(r, (cy - cx) / (dy - dx))
                score = r
            else:
                ux = max(self.R[x] * dx - cx, 0.0) if x != OPT_OUT else 0.0
                r = (ux + cy) / dy
                score = r / self.R[y]          # smallest relative reduction
            if best is None or score > best[0]:
                best = (score, r, t, y)
        if best is None:                        # cannot happen for a real cycle
            raise RuntimeError("cycle without an eligible transition")
        _, r, t, y = best
        if mode == "global":
            self.R[:] = r
        else:
            self.R[y] = r
        return r, t
