"""Missions with correlated traffic (ground vehicles, straight legs).

Travel time of vehicle i to PsI k:

    t = (d_ik / v) * u_i * sum_z w_ikz * s_{z, p(i)}

  s_{z,p}  slowness of zone z (G x G grid) in period p: shared by every
           vehicle that departs in p and crosses z (lognormal, mean 1);
  u_i      individual factor of the trip (lognormal, mean 1);
  w_ikz    share of the path i -> k that lies in zone z.
rho is the share of the log-variance assigned to the shared component. The
zone variance is calibrated so that the mean squared coefficient of variation
of the travel time, over all vehicle-PsI pairs, equals cv^2 for every rho.
The underlying standard-normal shocks depend only on the seed, so scenarios
with the same seed share geometry and shocks across cv and rho.

Information at a decision window: realized samples, the plans of vehicles
still en route and the fact that they have not arrived, and the prior law of
the traffic. Beliefs about zone slowness are NOT updated from observations
(a stated simplification); the individual factor of an en-route vehicle is
conditioned exactly on "not yet arrived" given each zone draw.

Plans are committed at departure; there is no replanning.
"""
from __future__ import annotations

import bisect
import math
import random

import numpy as np
from scipy.special import ndtr, ndtri

from .model import OPT_OUT
from .policies import spa
from .stochastic import StochGame, profile_regret

ESTIMATORS = ("dist", "fixorder", "mean")


class CorrScenario:
    def __init__(self, seed, cv, rho, M=20, side=2000.0, horizon=1800.0, rate=1 / 6.0, speed=10.0,
                 deadlines=(120.0, 180.0, 300.0), alpha=0.4, window=60.0, G=4, period=300.0, Q=40):
        rng = np.random.default_rng(seed)
        self.M, self.T, self.cv, self.rho, self.window = M, horizon, cv, rho, window
        self.psi = rng.uniform(0, side, (M, 2))
        self.deadline = rng.choice(deadlines, M)
        self.tau = self.deadline / math.log(10.0)
        self.init_last = -rng.uniform(0, 1, M) * self.deadline
        n = self.n = rng.poisson(rate * horizon)
        self.t_dep = np.sort(rng.uniform(0, horizon, n))
        orig, dest = rng.uniform(0, side, (n, 2)), rng.uniform(0, side, (n, 2))
        self.Z = G * G
        n_per = int(horizon // period) + 2
        xi_zone = rng.standard_normal((n_per, self.Z))
        xi_u = rng.standard_normal(n)
        d_op = np.linalg.norm(orig[:, None] - self.psi[None], axis=2)
        d_pd = np.linalg.norm(self.psi[None] - dest[:, None], axis=2)
        d_od = np.linalg.norm(orig - dest, axis=1)[:, None]
        self.cost = alpha * np.maximum(d_op + d_pd - d_od, 0.0)
        self.base = d_op / speed                                   # mean travel time to the PsI
        self.per = (self.t_dep // period).astype(int)
        # zone weights of each path
        frac = (np.arange(Q) + 0.5) / Q
        pts = orig[:, None, None, :] + (self.psi[None, :, None, :] - orig[:, None, None, :]) * frac[None, None, :, None]
        iz = np.clip((pts / (side / G)).astype(int), 0, G - 1)
        z = iz[..., 0] * G + iz[..., 1]
        self.w = np.zeros((n, M, self.Z))
        ii, kk = np.indices((n, M))
        for q in range(Q):
            np.add.at(self.w, (ii, kk, z[:, :, q]), 1.0 / Q)
        # variance split, calibrated to a common mean squared CV
        s2 = math.log(1.0 + cv * cv)
        self.sig_u = math.sqrt((1.0 - rho) * s2)
        cvu2 = math.exp(self.sig_u ** 2) - 1.0
        H = float((self.w ** 2).sum(-1).mean())
        cvc2 = max(((1.0 + cv * cv) / (1.0 + cvu2) - 1.0) / H, 0.0)
        self.sig_c = math.sqrt(math.log(1.0 + cvc2))
        s_true = np.exp(self.sig_c * xi_zone - self.sig_c ** 2 / 2)
        u_true = np.exp(self.sig_u * xi_u - self.sig_u ** 2 / 2)
        mix = np.einsum("ikz,iz->ik", self.w, s_true[self.per])
        self.arr_true = self.t_dep[:, None] + self.base * mix * u_true[:, None]
        self.arr_mean = self.t_dep[:, None] + self.base            # prior mean arrival


def build(scn: CorrScenario, players, t0, pending, last, R, S, rng, est, rivals=True):
    """Window game seen through estimator `est`. pending: [(vehicle, k)] en route."""
    N, M = len(players), scn.M
    pers = sorted({int(scn.per[v]) for v in players} | {int(scn.per[j]) for j, _ in pending})
    Zs = {p: np.exp(scn.sig_c * rng.standard_normal((scn.Z, S)) - scn.sig_c ** 2 / 2) for p in pers}
    U = np.exp(scn.sig_u * rng.standard_normal((N, S)) - scn.sig_u ** 2 / 2)
    A = np.empty((N, M, S))
    for a, v in enumerate(players):
        A[a] = scn.t_dep[v] + scn.base[v][:, None] * (scn.w[v] @ Zs[int(scn.per[v])]) * U[a][None, :]
    P = [[] for _ in range(M)]
    for (j, k) in pending:
        mix = scn.w[j, k] @ Zs[int(scn.per[j])]
        if scn.sig_u > 0:                                   # u_j conditioned on "not yet arrived"
            lo = np.maximum((t0 - scn.t_dep[j]) / (scn.base[j, k] * mix), 1e-300)
            plo = ndtr((np.log(lo) + scn.sig_u ** 2 / 2) / scn.sig_u)
            vv = np.clip(plo + (1.0 - plo) * rng.uniform(0, 1, S), 1e-12, 1 - 1e-12)
            u = np.exp(-scn.sig_u ** 2 / 2 + scn.sig_u * ndtri(vv))
        else:
            u = 1.0
        P[k].append(np.maximum(scn.t_dep[j] + scn.base[j, k] * mix * u, t0))
    if est == "mean":
        A = A.mean(axis=2, keepdims=True)
        P = [np.array([[r.mean()] for r in Pk]) if Pk else np.zeros((0, 1)) for Pk in P]
    else:
        P = [np.array(Pk) if Pk else np.zeros((0, S)) for Pk in P]
    return StochGame(A, P, last, scn.tau, scn.cost[players], R,
                     mode=est if (est.startswith("fixorder") or est == "ordermean") else "dist", rivals=rivals)


def run(scn: CorrScenario, est: str, R: float, S: int = 64, seed: int = 0, regret=None, ests=ESTIMATORS):
    """Mission in which every window plays the equilibrium computed with
    estimator `est`. If regret=(S_sel, S_eval), the profiles of all three
    estimators are also scored at each window state of THIS mission."""
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    done = [[t] for t in scn.init_last]
    pending, commits = [], {}
    st = dict(breaks=0, cycles=0, unstable=0, games=0, expected=0.0)
    reg = {e: dict(win=[], player=[], cut=[], breaks=0, win400=[], rfinal=[]) for e in ests} if regret else None
    wR = {}
    t0, idx, widx = 0.0, 0, 0
    while idx < scn.n:
        t1 = t0 + scn.window
        players = []
        while idx < scn.n and scn.t_dep[idx] < t1:
            players.append(idx)
            idx += 1
        keep = []
        for (a, j, k) in pending:
            if a <= t0:
                bisect.insort(done[k], a)
            else:
                keep.append((a, j, k))
        pending = keep
        if players:
            last = np.array([d[-1] for d in done])
            pend = [(j, k) for (_, j, k) in pending]
            g = build(scn, players, t0, pend, last, R, S, rng, est)
            r = spa(g, random.Random(seed * 31 + widx), adapt="global")   # same SPA order for every estimator
            assign = r.assign
            st["games"] += 1
            st["breaks"] += r.breaks
            st["cycles"] += int(r.cycled)
            st["unstable"] += int(not (r.converged and g.is_nash(assign)))
            if regret:
                eseed = 1_000_003 * (seed + 1) + widx
                for e in ests:
                    rv = not e.startswith("nr_")              # nr_*: control reward that ignores other vehicles
                    be = e if rv else e[3:]
                    if e == est:
                        prof, nb, Re = assign, r.breaks, g.R.copy()
                    else:
                        ge = build(scn, players, t0, pend, last, R, S, np.random.default_rng(eseed + 7), be, rivals=rv)
                        re_ = spa(ge, random.Random(seed * 31 + widx), adapt="global")
                        prof, nb, Re = re_.assign, re_.breaks, ge.R.copy()
                    # score against the game actually offered: the reward after any cuts, same draws for all
                    erng = np.random.default_rng(eseed)
                    g_sel = build(scn, players, t0, pend, last, Re, regret[0], erng, "dist", rivals=rv)
                    g_eval = build(scn, players, t0, pend, last, Re, regret[1], erng, "dist", rivals=rv)
                    pr = profile_regret(g_sel, g_eval, prof)
                    reg[e]["win"].append(max(pr))
                    reg[e]["cut"].append(int(nb > 0))
                    reg[e]["player"].extend(pr)
                    reg[e]["breaks"] += nb
                    reg[e]["rfinal"].append(float(np.min(Re)) / R)
                    if nb > 0:   # same profile under the nominal reward R, same evaluation draws
                        erng0 = np.random.default_rng(eseed)
                        g_sel0 = build(scn, players, t0, pend, last, R, regret[0], erng0, "dist", rivals=rv)
                        g_eval0 = build(scn, players, t0, pend, last, R, regret[1], erng0, "dist", rivals=rv)
                        reg[e]["win400"].append(max(profile_regret(g_sel0, g_eval0, prof)))
                    else:
                        reg[e]["win400"].append(reg[e]["win"][-1])
            for i, k in enumerate(assign):
                if k == OPT_OUT:
                    continue
                v = players[i]
                a_true = float(scn.arr_true[v, k])
                pending.append((a_true, v, k))
                commits[v] = (k, a_true, float(scn.cost[v, k]))
                wR[v] = float(np.min(g.R))
                st["expected"] += g.R[k] * g.delta(i, k, assign)
        t0, widx = t1, widx + 1
    for (a, j, k) in pending:
        bisect.insort(done[k], a)
    pay = {}
    for k in range(scn.M):
        s = done[k]
        for p_, a in zip(s[:-1], s[1:]):
            pay[(k, a)] = R * (1 - math.exp(-(a - p_) / scn.tau[k]))
    payout_nominal = sum(pay[(k, a)] for (k, a, c) in commits.values())
    payout = sum(pay[(k, a)] * wR[v] / R for v, (k, a, c) in commits.items())   # paid at the window's announced reward
    cost = sum(c for (k, a, c) in commits.values())
    viol, worst, covered = 0.0, [], 0
    for k in range(scn.M):
        inside = [t for t in done[k] if 0 <= t <= scn.T]
        covered += int(len(inside) > 0)
        pts = [t for t in done[k] if t <= scn.T] + [scn.T]
        gaps = []
        for a, b in zip(pts[:-1], pts[1:]):
            gaps.append(b - a)
            viol += max(0.0, b - max(a + scn.deadline[k], 0.0))
        worst.append(max(gaps))
    # how often the realized order of consecutive samplers differs from the mean-time order
    inv = tot = 0
    by_k = {}
    for v, (k, a, c) in commits.items():
        by_k.setdefault(k, []).append((float(scn.arr_mean[v, k]), a))
    for lst in by_k.values():
        lst.sort()
        for (m1, a1), (m2, a2) in zip(lst[:-1], lst[1:]):
            tot += 1
            inv += int(a2 < a1)
    nd = max(len(commits), 1)
    out = dict(est=est, R=R, cv=scn.cv, rho=scn.rho,
               deadline_violation=viol / (scn.M * scn.T), worst_gap=float(np.mean(worst)),
               coverage=covered / scn.M, detours=len(commits), vehicles=scn.n,
               payout_nominal=payout_nominal, net_nominal=(payout_nominal - cost) / scn.n,
               payout=payout, net_per_vehicle=(payout - cost) / scn.n,
               forecast_bias=(payout - st["expected"]) / nd,
               inversion_rate=inv / max(tot, 1),
               breaks=st["breaks"], cycle_games=st["cycles"], unstable_games=st["unstable"], games=st["games"])
    return out, reg
