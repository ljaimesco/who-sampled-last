"""Mission simulator: a sequence of decision epochs over one horizon.

The scenario (intersections, deadlines, delivery legs, batteries, slacks) is
generated once from a seed and reused unchanged for every policy, so policies
are compared on matched scenarios.
"""
from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass

import numpy as np

from .model import OPT_OUT, Epoch, Game, Params
from . import policies as P


@dataclass(frozen=True)
class Scenario:
    """All values are working assumptions, to be set from the application."""
    N: int = 10                     # delivery legs starting per epoch
    M: int = 20                     # monitored intersections
    map_m: float = 2000.0           # side of the square service area, m
    epoch_len: float = 60.0         # s between decision epochs
    epochs: int = 30                # horizon = epochs * epoch_len
    deadlines: tuple = (120.0, 180.0, 300.0)   # revisit deadline classes, s
    tau_ratio: float = 1.0 / math.log(10.0)    # tau = ratio * deadline (delta = 0.9 at the deadline)
    energy_range: tuple = (30.0, 100.0)        # battery at leg start, Wh
    slack_range: tuple = (20.0, 90.0)          # tolerated delivery delay, s

    @property
    def horizon(self) -> float:
        return self.epochs * self.epoch_len


def generate(scn: Scenario, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    psi = rng.uniform(0, scn.map_m, (scn.M, 2))
    deadline = rng.choice(scn.deadlines, scn.M)
    init_age = rng.uniform(0, 1, scn.M) * deadline          # age of the last sample at t = 0
    legs = []
    for _ in range(scn.epochs):
        legs.append(dict(
            orig=rng.uniform(0, scn.map_m, (scn.N, 2)),
            dest=rng.uniform(0, scn.map_m, (scn.N, 2)),
            energy=rng.uniform(*scn.energy_range, scn.N),
            slack=rng.uniform(*scn.slack_range, scn.N)))
    return dict(psi=psi, deadline=deadline, tau=scn.tau_ratio * deadline,
                init_age=init_age, legs=legs)


POLICIES = ("spa", "spa_local", "spa_noadapt", "legacy", "greedy", "nearest", "none")


def _allocate(policy: str, game: Game, rng: random.Random):
    if policy in ("spa", "legacy"):
        r = P.spa(game, rng, adapt="global")
        return r.assign, r
    if policy == "spa_local":
        r = P.spa(game, rng, adapt="local")
        return r.assign, r
    if policy == "spa_noadapt":
        r = P.spa(game, rng, adapt=None)
        return r.assign, r
    if policy == "greedy":
        return P.greedy(game, rng), None
    if policy == "nearest":
        return P.nearest(game, rng), None
    if policy == "none":
        return [OPT_OUT] * game.N, None
    raise ValueError(policy)


def run_mission(scn: Scenario, data: dict, policy: str, R: float,
                params: Params = Params(), seed: int = 0) -> dict:
    rng = random.Random(seed)
    history = [[-float(a)] for a in data["init_age"]]
    H = scn.horizon
    tot = dict(payout=0.0, cost=0.0, samples=0, violations=0, legs=0,
               switches=0, checks=0, breaks=0, cycles=0, unstable=0, evals=0, reward_sum=0.0)
    for e, leg in enumerate(data["legs"]):
        ep = Epoch(t0=e * scn.epoch_len, orig=leg["orig"], dest=leg["dest"],
                   energy=leg["energy"], slack=leg["slack"], psi=data["psi"],
                   tau=data["tau"], history=history)
        game = Game(ep, R, params, enforce=(policy != "legacy"))
        assign, r = _allocate(policy, game, rng)
        if r is not None:
            tot["switches"] += r.switches
            tot["checks"] += r.checks
            tot["breaks"] += r.breaks
            tot["cycles"] += int(r.cycled)
            tot["unstable"] += int(not r.converged)
        tot["evals"] += game.evals
        tot["reward_sum"] += float(game.R.mean())
        tot["legs"] += game.N
        commits = []
        for i, k in enumerate(assign):
            if k == OPT_OUT:
                continue
            tot["payout"] += game.R[k] * game.delta(i, k, assign)
            tot["cost"] += game.cost[i, k]
            tot["samples"] += 1
            tot["violations"] += int(not game.truly_feasible[i, k])
            commits.append((k, float(game.arr[i, k])))
        for k, t in commits:                     # commit after pricing the whole epoch
            bisect.insort(history[k], t)

    # ---- mission metrics ---------------------------------------------------
    viol_time, worst_gap, ent, never = 0.0, [], [], 0
    for k in range(scn.M):
        T = float(data["deadline"][k])
        s = [t for t in history[k] if t <= H]
        inside = [t for t in s if t >= 0]
        never += int(not inside)
        pts = s + [H]
        gaps = []
        for a, b in zip(pts[:-1], pts[1:]):
            gaps.append(b - a)
            lo, hi = max(a + T, 0.0), b           # interval where the age exceeds the deadline
            viol_time += max(0.0, hi - lo)
        worst_gap.append(max(gaps))
        iv = np.diff([0.0] + inside + [H])
        iv = iv[iv > 0] / H
        n = len(inside)
        ent.append(float(-(iv * np.log(iv)).sum() / math.log(n + 1)) if n > 0 else 0.0)
    part = tot["samples"] / tot["legs"]
    return dict(
        policy=policy, R=R,
        deadline_violation=viol_time / (scn.M * H),      # share of time past the revisit deadline
        worst_gap_mean=float(np.mean(worst_gap)),        # mean over intersections of longest gap, s
        worst_gap_max=float(np.max(worst_gap)),
        never_sampled=never / scn.M,
        entropy=float(np.mean(ent)),                     # normalized, secondary metric
        samples=tot["samples"], participation=part,
        payout=tot["payout"],
        uav_utility=(tot["payout"] - tot["cost"]) / tot["legs"],   # mean net credits per leg
        infeasible_share=tot["violations"] / max(tot["samples"], 1),
        switches_per_epoch=tot["switches"] / scn.epochs,
        checks_per_epoch=tot["checks"] / scn.epochs,
        evals_per_epoch=tot["evals"] / scn.epochs,
        breaks=tot["breaks"], cycle_epochs=tot["cycles"], unstable_epochs=tot["unstable"],
        mean_reward=tot["reward_sum"] / scn.epochs)
