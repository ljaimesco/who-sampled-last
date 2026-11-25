"""Core checks. Run with:  python -m pytest -q   (or python tests/test_core.py)"""
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spa_uav import OPT_OUT, Epoch, Game, Params, Scenario, exact, generate, greedy, run_mission, spa


def small_game(seed, N=5, M=2, R=800.0, tau=15.0, side=500.0, enforce=False, params=Params()):
    rng = np.random.default_rng(seed)
    ep = Epoch(t0=0.0, orig=rng.uniform(0, side, (N, 2)), dest=rng.uniform(0, side, (N, 2)),
               energy=np.full(N, 100.0), slack=np.full(N, 1e9), psi=rng.uniform(0, side, (M, 2)),
               tau=np.full(M, tau), history=[[] for _ in range(M)])
    return Game(ep, R, params, enforce=enforce)


def test_energy_chain():
    p = Params()
    assert math.isclose(p.energy_wh(p.speed), 0.1)          # 360 W for 1 s = 0.1 Wh
    assert math.isclose(p.p_cruise / 3600.0, 0.1)


def test_delta_bounds_and_order():
    g = small_game(1)
    a = [0, 0, OPT_OUT, OPT_OUT, OPT_OUT]
    first, second = sorted([0, 1], key=lambda i: g.arr[i, 0])
    assert g.delta(first, 0, a) == 1.0
    assert 0.0 <= g.delta(second, 0, a) < 1.0


def test_spa_output_is_nash():
    for seed in range(200):
        g = small_game(seed, N=6, M=3, R=600.0, tau=40.0)
        r = spa(g, random.Random(seed), adapt="global")
        assert r.converged
        assert g.is_nash(r.assign), seed


def test_spa_agrees_with_enumeration():
    for seed in range(60):
        g = small_game(seed, N=4, M=2, R=500.0, tau=20.0)
        r = spa(g, random.Random(seed), adapt=None)
        if r.converged:
            nash = [a for a, _ in exact(g)["nash"]]
            assert r.assign in nash, seed


def test_break_cycle_lowers_reward_and_terminates():
    found = 0
    for seed in range(3000):
        g = small_game(seed)
        r0 = spa(g, random.Random(seed), adapt=None)
        if not r0.cycled:
            continue
        found += 1
        g2 = small_game(seed)
        r = spa(g2, random.Random(seed), adapt="global")
        assert r.breaks >= 1 and r.converged
        assert g2.R[0] < 800.0
        assert all(b2 < b1 for b1, b2 in zip(r.reward_trace, r.reward_trace[1:]))
        assert g2.is_nash(r.assign)
        if found >= 5:
            break
    assert found >= 1, "no cycle instance found in the stress regime"


def test_feasibility_respected():
    scn = Scenario(epochs=6)
    data = generate(scn, 3)
    m = run_mission(scn, data, "spa", 400.0, seed=3)
    assert m["infeasible_share"] == 0.0
    assert 0.0 <= m["deadline_violation"] <= 1.0


def test_mission_reproducible():
    scn = Scenario(epochs=6)
    a = run_mission(scn, generate(scn, 5), "spa", 400.0, seed=5)
    b = run_mission(scn, generate(scn, 5), "spa", 400.0, seed=5)
    assert a == b


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
