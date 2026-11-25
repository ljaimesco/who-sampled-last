"""Empirical check of the theory on the London single-stop windows (rho = 0.5, the experiment's scenarios).

The mission is played with dist (the states of the London experiment). At every window, each estimator's SPA
profile is computed from the same 1024 decision draws, exactly as in london_mission.run. For every player i, every
option k of that estimator, and the rivals given by that estimator's profile, we evaluate on 4096 independent draws
(the model's ground truth): true utility, fixorder / mean / ordermean values, the exact error terms and their moment
bounds (spa_uav.theory). Records go to results/theory_records.npz; per window, the estimator's ex-ante regret.
"""
import bisect, json, math, os, random, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav import london_mission as lm
from spa_uav.model import OPT_OUT
from spa_uav.policies import spa
from spa_uav.stochastic import StochGame, profile_regret
from spa_uav.theory import terms
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
R, S, S_EVAL, ESTS = 400.0, 1024, 4096, ("dist", "ordermean", "fixorder", "mean")
PFIELDS = ("seed", "win", "est", "i", "k_true", "k_est", "diff", "se_diff")
FIELDS = ("seed", "win", "est", "i", "k", "chosen", "u", "uhat", "uF", "uM", "uO", "order", "curv", "om", "pchange",
          "order_b", "curv_b", "om_b", "Rk", "varD", "tau", "has_pred")


def mission(seed):
    scn = lm.LondonScenario(5000 + seed, 0.5)
    rng = np.random.default_rng(seed + 1)
    done = [[t] for t in scn.init_last]
    pending = []
    recs, wins, pairs = [], [], []
    t0, idx, widx = 0.0, 0, 0
    while idx < scn.n:
        t1 = t0 + scn.window
        players = []
        while idx < scn.n and scn.t_dep[idx] < t1:
            players.append(idx); idx += 1
        keep = []
        for (a, j, k) in pending:
            if a <= t0: bisect.insort(done[k], a)
            else: keep.append((a, j, k))
        pending = keep
        if players:
            last = np.array([d[-1] for d in done])
            pend = [(j, k) for (_, j, k) in pending]
            A, P = lm.window_samples(scn, players, t0, pend, S, rng, R)
            Ae, Pe = lm.window_samples(scn, players, t0, pend, S_EVAL, np.random.default_rng(7_000_003 * (seed + 1) + widx), R)
            As, Ps = lm.window_samples(scn, players, t0, pend, 512, np.random.default_rng(1_000_003 * (seed + 1) + widx), R)
            games = {e: lm.make_game(scn, players, A, P, last, R, e) for e in ESTS}
            prof = {}
            for e in ESTS:
                r = spa(games[e], random.Random(seed * 31 + widx), adapt="global")
                prof[e] = (r.assign, games[e].R.copy(), r.breaks)
            for e in ESTS:
                assign, Re, nb = prof[e]
                g = games[e]
                gs = StochGame(As, Ps, last, scn.tau, scn.cost[players], Re)
                ge = StochGame(Ae, Pe, last, scn.tau, scn.cost[players], Re)
                regs = profile_regret(gs, ge, assign)
                wins.append((seed, widx, ESTS.index(e), max(regs) / R, int(nb > 0), len(players)))
                for i in range(len(players)):
                    pays = {OPT_OUT: (np.full(S_EVAL, 0.0), 0.0, 0.0)}   # per-draw utility, mean utility, estimate
                    for k in g.options[i]:
                        if k == OPT_OUT:
                            continue
                        comps = [Pe[k][q] for q in range(len(Pe[k]))]
                        comps += [Ae[j, k] for j, kk in enumerate(assign) if kk == k and j != i]
                        tm = terms(Ae[i, k], comps, last[k], scn.tau[k])
                        uhat = Re[k] * g.delta(i, k, assign) - scn.cost[players[i], k]
                        recs.append((seed, widx, ESTS.index(e), i, k, int(assign[i] == k), Re[k] * tm["u"] - scn.cost[players[i], k],
                                     uhat, tm["uF"], tm["uM"], tm["uO"], tm["order"], tm["curv"], tm["om"], tm["pchange"],
                                     tm["order_b"], tm["curv_b"], tm["om_b"], Re[k], tm["varD"], scn.tau[k], int(tm["has_pred"])))
                        ud = Re[k] * tm["pay"] - scn.cost[players[i], k]
                        pays[k] = (ud, float(ud.mean()), uhat)
                    kt = max(pays, key=lambda q: pays[q][1]); ke = max(pays, key=lambda q: pays[q][2])
                    d = pays[kt][0] - pays[ke][0]
                    pairs.append((seed, widx, ESTS.index(e), i, kt, ke, float(d.mean()), float(d.std(ddof=1) / math.sqrt(S_EVAL))))
            assign = prof["dist"][0]
            for i, k in enumerate(assign):
                if k == OPT_OUT: continue
                v = players[i]
                pending.append((float(scn.arr_true[v, k]), v, k))
        t0, widx = t1, widx + 1
    return np.array(recs, dtype=float), np.array(wins, dtype=float), np.array(pairs, dtype=float)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    t = time.time()
    with Pool(2) as pool:
        out = pool.map(mission, range(n))
    recs = np.concatenate([o[0] for o in out]); wins = np.concatenate([o[1] for o in out]); pairs = np.concatenate([o[2] for o in out])
    np.savez_compressed(f"{OUT}/theory_records.npz", recs=recs, wins=wins, pairs=pairs, fields=np.array(FIELDS), pfields=np.array(PFIELDS))
    print("records", recs.shape, "windows", wins.shape, "elapsed", round(time.time() - t))
