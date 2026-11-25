"""Allocation policies: corrected SPA, greedy baselines, exact enumeration."""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .model import OPT_OUT, Game


@dataclass
class SpaResult:
    assign: list
    switches: int = 0            # best-response moves executed
    checks: int = 0              # best-response evaluations
    breaks: int = 0              # break_cycle invocations
    cycled: bool = False         # a cycle was met (with or without adaptation)
    converged: bool = True       # ended in a verified stable state
    reward_trace: list = field(default_factory=list)   # reward after each break
    cycle: list = field(default_factory=list)          # first cycle: [(state, move), ...]


def spa(game: Game, rng: random.Random, adapt: str | None = "global",
        init=None, max_steps: int = 200_000) -> SpaResult:
    """Smart PsI Allocation, corrected.

    * strict-improvement best responses, current choice kept on ties;
    * a cycle is detected as soon as ANY previously visited assignment recurs,
      checked right after every switch;
    * adapt=None stops at the first cycle; "global"/"local" call break_cycle,
      clear the history (the game changed) and continue.
    """
    N = game.N
    assign = list(init) if init is not None else [OPT_OUT] * N
    res = SpaResult(assign=assign)
    pending = set(range(N))
    trace, moves = [tuple(assign)], []
    seen = {trace[0]: 0}
    steps = 0
    while pending:
        steps += 1
        if steps > max_steps:
            res.converged = False
            break
        i = rng.choice(sorted(pending))
        pending.discard(i)
        y, _, _ = game.best_response(i, assign)
        res.checks += 1
        x = assign[i]
        if y == x:
            continue
        assign[i] = y
        res.switches += 1
        moves.append((i, x, y))
        key = tuple(assign)
        trace.append(key)
        # anyone who can use the old or the new intersection may now prefer otherwise
        for j in range(N):
            if j != i and ((x != OPT_OUT and game.feas[j, x]) or (y != OPT_OUT and game.feas[j, y])):
                pending.add(j)
        if key in seen:
            k0 = seen[key]
            res.cycled = True
            if not res.cycle:
                res.cycle = list(zip(trace[k0:-1], moves[k0:]))
            if adapt is None:
                res.converged = False
                break
            r, _ = game.break_cycle(trace[k0:-1], moves[k0:], adapt)
            res.breaks += 1
            res.reward_trace.append(float(r))
            trace, moves, seen = [key], [], {key: 0}
            pending = set(range(N))
        else:
            seen[key] = len(trace) - 1
    return res


def greedy(game: Game, rng: random.Random) -> list:
    """One pass: UAVs choose once, in random order, the option that is best
    given the choices already made. No revision."""
    assign = [OPT_OUT] * game.N
    order = list(range(game.N))
    rng.shuffle(order)
    for i in order:
        assign[i] = game.best_response(i, assign)[0]
    return assign


def nearest(game: Game, rng: random.Random) -> list:
    """Each UAV takes its cheapest feasible detour, then drops it if the
    resulting utility is negative (the Greedy of the original code)."""
    assign = [OPT_OUT] * game.N
    for i in range(game.N):
        ks = [k for k in game.options[i] if k != OPT_OUT]
        if ks:
            assign[i] = min(ks, key=lambda k: game.cost[i, k])
    for i in range(game.N):
        if assign[i] != OPT_OUT and game.utility(i, assign[i], assign) < 0:
            assign[i] = OPT_OUT
    return assign


# ---- exact analysis of small games -----------------------------------------
def all_profiles(game: Game):
    return itertools.product(*game.options)


def exact(game: Game, limit: int = 2_000_000):
    """Enumerate every assignment. Returns all pure-strategy Nash equilibria,
    the welfare-optimal assignment, and whether the best-response graph
    contains a cycle (an exact property of the game, not of one random run)."""
    size = int(np.prod([len(o) for o in game.options], dtype=float))
    if size > limit:
        raise ValueError(f"{size} profiles: too many to enumerate")
    states = list(all_profiles(game))
    index = {s: n for n, s in enumerate(states)}
    nash, best, w_best, src, dst = [], None, -float("inf"), [], []
    for s in states:
        a = list(s)
        w = game.welfare(a)
        if w > w_best:
            best, w_best = a, w
        stable = True
        for i in range(game.N):
            y = game.best_response(i, a)[0]
            if y != a[i]:
                stable = False
                src.append(index[s])
                dst.append(index[s[:i] + (y,) + s[i + 1:]])
        if stable:
            nash.append((a, w))
    # the best-response graph has a cycle iff some strongly connected
    # component holds more than one state (there are no self-loops)
    n = len(states)
    graph = csr_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n))
    _, labels = connected_components(graph, directed=True, connection="strong")
    has_cycle = bool((np.bincount(labels) > 1).any())
    return dict(size=size, nash=nash, opt=best, w_opt=w_best, br_cycle=has_cycle)
