"""MultiGame restricted to single stops must reproduce StochGame exactly, for every estimator."""
import os, random, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav.model import OPT_OUT
from spa_uav.multistop import make_multi
from spa_uav.policies import spa
from spa_uav.stochastic import StochGame


def instance(seed, N=6, M=4, S=64):
    rng = np.random.default_rng(seed)
    A = 50 + rng.uniform(0, 100, (N, M))[:, :, None] * rng.lognormal(0, 0.3, (N, M, S))
    P = [np.sort(30 + rng.uniform(0, 150, (rng.integers(0, 3), 1)) * rng.lognormal(0, 0.3, (1, S)), axis=0) for _ in range(M)]
    P = [p if len(p) else np.zeros((0, S)) for p in P]
    last = -rng.uniform(0, 100, M); tau = rng.uniform(50, 130, M); cost = rng.uniform(0, 300, (N, M))
    return A, P, last, tau, cost


def test_single_stop_equivalence():
    for seed in range(40):
        A, P, last, tau, cost = instance(seed)
        N, M, S = A.shape
        routes = [(k,) for k in range(M)]
        for est in ("dist", "ordermean", "fixorder", "mean"):
            Ae, Pe = (A.mean(axis=2, keepdims=True), [p.mean(axis=1, keepdims=True) for p in P]) if est == "mean" else (A, P)
            sg = StochGame(Ae, Pe, last, tau, cost, 400.0, mode="dist" if est == "mean" else est)
            opts = [[OPT_OUT] + [k for k in range(M) if cost[i, k] < 400.0] for i in range(N)]
            own = {(i, k): [A[i, k]] for i in range(N) for k in range(M)}
            cst = {(i, k): float(cost[i, k]) for i in range(N) for k in range(M)}
            mg = make_multi(routes, opts, own, P, last, tau, cst, 400.0, est)
            rng = random.Random(seed)
            for _ in range(30):
                prof = [rng.choice(opts[i]) for i in range(N)]
                for i in range(N):
                    for k in opts[i]:
                        b = list(prof); b[i] = k
                        assert abs(sg.utility(i, k, b) - mg.utility(i, k, b)) < 1e-9, (seed, est, i, k)
            r1 = spa(sg, random.Random(seed), adapt="global"); r2 = spa(mg, random.Random(seed), adapt="global")
            assert r1.assign == r2.assign and r1.breaks == r2.breaks, (seed, est)


if __name__ == "__main__":
    test_single_stop_equivalence(); print("multistop equivalence: ok")
