"""Missions for the uncertain-arrival game (ground vehicles, straight legs).

Vehicles depart over the horizon (Poisson). Decisions are taken in windows of
`window` seconds: the vehicles departing in a window play one game, knowing
the realized samples so far and the plans of vehicles still en route (and
that those have not arrived yet). Plans are then committed and the true
arrival times, drawn once per vehicle, decide the realized order and pay.

Policies (identical scenarios and identical true speed factors):
  eq_dist   equilibrium of the expected-utility game (distribution of arrivals)
  eq_mean   equilibrium of the game played with mean arrival times
  greedy    one pass in random order, distribution-aware, no revision
  age       each vehicle best-responds to the last realized sample only
"""
from __future__ import annotations

import bisect
import math
import random

import numpy as np

from .model import OPT_OUT, Params
from .policies import greedy as greedy_pass, spa
from .stochastic import StochGame, sample_f


class StochScenario:
    def __init__(self, seed, cv, M=20, side=2000.0, horizon=1800.0, rate=1 / 6.0, speed=10.0,
                 deadlines=(120.0, 180.0, 300.0), alpha=0.4, window=60.0):
        rng = np.random.default_rng(seed)
        self.M, self.T, self.v, self.cv, self.alpha, self.window = M, horizon, speed, cv, alpha, window
        self.psi = rng.uniform(0, side, (M, 2))
        self.deadline = rng.choice(deadlines, M)
        self.tau = self.deadline / math.log(10.0)
        self.init_last = -rng.uniform(0, 1, M) * self.deadline
        n = rng.poisson(rate * horizon)
        self.n = n
        self.t_dep = np.sort(rng.uniform(0, horizon, n))
        orig, dest = rng.uniform(0, side, (n, 2)), rng.uniform(0, side, (n, 2))
        d_op = np.linalg.norm(orig[:, None] - self.psi[None], axis=2)
        d_pd = np.linalg.norm(self.psi[None] - dest[:, None], axis=2)
        d_od = np.linalg.norm(orig - dest, axis=1)[:, None]
        self.d_op = d_op
        self.cost = alpha * np.maximum(d_op + d_pd - d_od, 0.0)
        self.f_true = sample_f(rng, cv, n)


POLICIES = ("eq_dist", "eq_mean", "greedy", "age")


def _game(scn, players, t0, pending, last, R, S, rng, mode):
    """Build the window game. pending: list of (vehicle, k) still en route."""
    N, M = len(players), scn.M
    if mode == "mean":
        S_eff = 1
    else:
        S_eff = S
    F = sample_f(rng, scn.cv, (N, S))                          # one speed factor per trip and draw
    tt = scn.d_op[players][:, :, None] / (scn.v * F[:, None, :])  # (N, M, S)
    A = scn.t_dep[players][:, None, None] + tt
    P = [[] for _ in range(M)]
    for (j, k) in pending:
        tj = scn.t_dep[j]
        fmax = scn.d_op[j, k] / (scn.v * (t0 - tj)) if t0 > tj else None
        P[k].append(tj + scn.d_op[j, k] / (scn.v * sample_f(rng, scn.cv, S, fmax)))
    if mode == "mean":
        A = A.mean(axis=2, keepdims=True)
        P = [np.array([[row.mean()] for row in Pk]) if Pk else np.zeros((0, 1)) for Pk in P]
    elif mode == "age":
        P = [np.zeros((0, A.shape[2])) for _ in range(M)]
    else:
        P = [np.array(Pk) if Pk else np.zeros((0, S)) for Pk in P]
    return StochGame(A, P, last, scn.tau, scn.cost[players], R)


def run(scn: StochScenario, policy: str, R: float, S: int = 64, seed: int = 0, eps_check: int = 0):
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    done = [[t] for t in scn.init_last]
    pending = []                    # (true arrival, vehicle, k)
    stats = dict(switches=0, breaks=0, cycles=0, games=0, unstable=0, expected=0.0, eps=[])
    commits = {}
    w = scn.window
    t0 = 0.0
    idx = 0
    while idx < scn.n:
        t1 = t0 + w
        players = []
        while idx < scn.n and scn.t_dep[idx] < t1:
            players.append(idx)
            idx += 1
        # release samples realized before the window starts
        keep = []
        for (a, j, k) in pending:
            if a <= t0:
                bisect.insort(done[k], a)
            else:
                keep.append((a, j, k))
        pending = keep
        if players:
            last = np.array([d[-1] for d in done])
            mode = {"eq_dist": "dist", "greedy": "dist", "eq_mean": "mean", "age": "age"}[policy]
            g = _game(scn, players, t0, [(j, k) for (_, j, k) in pending], last, R, S, rng, mode)
            stats["games"] += 1
            if policy == "greedy":
                assign = greedy_pass(g, prng)
            elif policy == "age":
                assign = [g.best_response(i, [OPT_OUT] * g.N)[0] for i in range(g.N)]
            else:
                r = spa(g, prng, adapt="global")
                assign = r.assign
                stats["switches"] += r.switches
                stats["breaks"] += r.breaks
                stats["cycles"] += int(r.cycled)
                stats["unstable"] += int(not (r.converged and g.is_nash(assign)))
            if eps_check and policy in ("eq_dist", "eq_mean"):
                # regret of the chosen profile in the TRUE expected game, estimated with fresh draws
                gt = _game(scn, players, t0, [(j, k) for (_, j, k) in pending], last, R, eps_check, rng, "dist")
                for i in range(gt.N):
                    _, u_best, u_cur = gt.best_response(i, assign)
                    stats["eps"].append((u_best - u_cur) / R)
            for i, k in enumerate(assign):
                if k == OPT_OUT:
                    continue
                v = players[i]
                a_true = scn.t_dep[v] + scn.d_op[v, k] / (scn.v * scn.f_true[v])
                pending.append((a_true, v, k))
                commits[v] = (k, a_true, scn.cost[v, k])
                stats["expected"] += g.R[k] * g.delta(i, k, assign) if policy != "age" else 0.0
        t0 = t1
    for (a, j, k) in pending:
        bisect.insort(done[k], a)
    # pay on realized order
    pay = {}
    for k in range(scn.M):
        s = done[k]
        for p_, a in zip(s[:-1], s[1:]):
            pay[(k, a)] = R * (1 - math.exp(-(a - p_) / scn.tau[k]))
    payout = sum(pay[(k, a)] for (k, a, c) in commits.values())
    cost = sum(c for (k, a, c) in commits.values())
    viol, worst = 0.0, []
    for k in range(scn.M):
        pts = [t for t in done[k] if t <= scn.T] + [scn.T]
        gaps = []
        for a, b in zip(pts[:-1], pts[1:]):
            gaps.append(b - a)
            viol += max(0.0, b - max(a + scn.deadline[k], 0.0))
        worst.append(max(gaps))
    nd = max(len(commits), 1)
    out = dict(policy=policy, R=R, cv=scn.cv,
               deadline_violation=viol / (scn.M * scn.T), worst_gap=float(np.mean(worst)),
               detours=len(commits), payout=payout, net_per_vehicle=(payout - cost) / scn.n,
               forecast_bias=((payout - stats["expected"]) / nd) if policy != "age" else float("nan"),
               switches_per_game=stats["switches"] / max(stats["games"], 1),
               breaks=stats["breaks"], cycle_games=stats["cycles"], unstable_games=stats["unstable"])
    if stats["eps"]:
        e = np.array(stats["eps"])
        out["eps_mean"] = float(e.mean())
        out["eps_p95"] = float(np.quantile(e, 0.95))
        out["eps_max"] = float(e.max())
    return out
