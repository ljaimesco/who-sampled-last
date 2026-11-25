"""Platform-oriented baseline: greedy assignment by overdue time per unit of detour, without solving the game.

For player i and feasible action b (one PsI, or an ordered pair), with forecast mean arrival t_k at each stop,
  score(i, b) = sum_k max(0, t_k - prev_k(t_k) - Delta_k) / (detour_i(b) + eps),
where prev_k(t) is the latest known or planned sample at k before t (last realized sample, en-route vehicles,
vehicles already assigned in this window, all at forecast mean times) and detour = cost / beta. The pair with the
highest positive score is assigned, its planned samples are added to the schedule, and the scores are recomputed;
players left without a positive score opt out. Uses the same feasible actions, draws and forecast as the estimators.

Estimator names understood by `solve`:
  "greedy"        the greedy profile, followed as is (full compliance),
  "greedy_seq"    drivers react once: in random order, each driver best-responds (dist evaluation) to the current
                  profile, which holds earlier drivers' reactions and later drivers' recommendations, and commits,
  "greedy_br"     best-response dynamics under the dist evaluation, started from the greedy profile, until no driver
                  wants to change (this reaches an equilibrium of the same game dist solves).
Default score: predicted reduction in time past deadline per unit of detour (a sample at t between planned samples
prev and next saves (next-prev-D)^+ - (t-prev-D)^+ - (next-t-D)^+). "greedy_od", "greedy_od_br": the simpler score
of overdue time at arrival, (t-prev-D)^+, which never values a visit before the deadline.
"""
from __future__ import annotations

import bisect
import math

import numpy as np

from .model import OPT_OUT
from .policies import spa

GREEDY = ("greedy", "greedy_seq", "greedy_br", "greedy_od", "greedy_od_br")
COUNT = {}         # per estimator name: [players decided, players whose action differs from the greedy recommendation]
EPS = 10.0         # seconds, fixed in advance
HORIZON = 1800.0   # mission horizon (s): with no later planned sample, a PsI is assumed unsampled until then
SCORE = "reduction"


def base(e: str) -> str:
    """Game evaluation used by an estimator name (greedy variants are scored and best-respond under dist)."""
    return "dist" if e in GREEDY else e


def bases(ests, est) -> set:
    return {base(e) for e in set(ests) | {est}}


def greedy_profile(g, beta: float = 1.5, eps: float = EPS, score: str = "reduction") -> list:
    deadline = g.tau * math.log(10.0)
    multi = hasattr(g, "routes")
    sched = [sorted([float(g.last[k])] + [float(x) for x in (g.P[k].mean(axis=1) if len(g.P[k]) else [])])
             for k in range(g.M)]
    if multi:
        means = {key: [float(x.mean()) for x in v] for key, v in g.own.items()}
        stops = lambda r: g.routes[r]
        arr = lambda i, r: means[(i, r)]
        detour = lambda i, r: g.costd[(i, r)] / beta
    else:
        Am = g.A.mean(axis=2)
        stops = lambda r: (r,)
        arr = lambda i, r: [float(Am[i, r])]
        detour = lambda i, r: float(g.cost[i, r]) / beta

    def overdue(k, t):
        j = bisect.bisect_left(sched[k], t)
        prev = sched[k][j - 1] if j > 0 else t
        return max(0.0, t - prev - deadline[k])

    def reduction(k, t):
        """Predicted reduction in time past deadline at k from a sample at t, given the planned schedule."""
        j = bisect.bisect_left(sched[k], t)
        prev = sched[k][j - 1] if j > 0 else t
        nxt = sched[k][j] if j < len(sched[k]) else max(HORIZON, t)
        d = deadline[k]
        return max(0.0, nxt - prev - d) - max(0.0, t - prev - d) - max(0.0, nxt - t - d)

    value = reduction if score == "reduction" else overdue

    assign = [OPT_OUT] * g.N
    free = set(range(g.N))
    while free:
        best, bi, br = 0.0, None, None
        for i in free:
            for r in g.options[i]:
                if r == OPT_OUT:
                    continue
                s = sum(value(k, t) for k, t in zip(stops(r), arr(i, r))) / (detour(i, r) + eps)
                if s > best:
                    best, bi, br = s, i, r
        if bi is None:
            break
        assign[bi] = br
        for k, t in zip(stops(br), arr(bi, br)):
            bisect.insort(sched[k], t)
        free.discard(bi)
    return assign


class _Res:
    def __init__(self, assign, breaks=0):
        self.assign, self.breaks, self.cycled, self.converged = assign, breaks, False, True


def solve(e: str, games: dict, rng, adapt="global"):
    """Profile chosen by estimator name e. games[e] is built with evaluation base(e), one object per name, so the
    reward cuts of one estimator's dynamics never leak into another's game."""
    if e in GREEDY:
        g = games[e]
        prof = greedy_profile(g, score="overdue" if "_od" in e else "reduction")
        if e.endswith("_br"):
            res = spa(g, rng, adapt=adapt, init=list(prof))
        elif e == "greedy_seq":                       # one pass: each driver, in random order, best-responds once
            out = list(prof)
            for i in rng.sample(range(g.N), g.N):
                out[i] = g.best_response(i, out)[0]
            res = _Res(out)
        else:
            res = _Res(prof)
        c = COUNT.setdefault(e, [0, 0])
        c[0] += g.N; c[1] += sum(int(x != y) for x, y in zip(res.assign, prof))
        return res
    return spa(games[e], rng, adapt=adapt)
