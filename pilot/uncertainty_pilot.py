"""Pilot: does modelling the arrival-time DISTRIBUTION change detour decisions
for the better, compared with point estimates?

Setting (ground vehicles, simplified geometry):
  * vehicles depart one at a time (Poisson arrivals) on straight legs;
  * at departure a vehicle picks at most one PsI to detour to, or opts out;
  * travel time = distance / (v * f), with a per-trip speed factor f
    (lognormal, mean 1, coefficient of variation CV), unknown in advance;
  * reward at PsI k = R * (1 - exp(-(a - prev)/tau_k)), where prev is the
    time of the previous sample at k (realized order, not planned order);
  * cost = alpha * detour length.
  The decider knows past samples, the plans of vehicles still en route
  (and that they have not arrived yet), and the distribution of f.

Three estimators of the expected reward of each option:
  dist : Monte Carlo over own and others' arrival times (spread and order);
  mean : the same inputs reduced to conditional mean arrival times
         (a SenseNow-style deterministic plan, but with the best point estimate);
  age  : ignores vehicles still en route, uses only the last realized sample.
All policies face identical scenarios and identical realized speed factors.

Limitation: vehicles respond to earlier vehicles only (sequential, not a
simultaneous game). It tests whether uncertainty matters, nothing more.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def lognorm_params(cv):
    s2 = math.log(1.0 + cv * cv)
    return math.sqrt(s2), -s2 / 2.0          # sigma, mu  (mean of f = 1)


class Mission:
    def __init__(self, seed, cv, M=20, side=2000.0, horizon=1800.0, rate=1 / 6.0,
                 speed=10.0, deadlines=(120.0, 180.0, 300.0)):
        rng = np.random.default_rng(seed)
        self.M, self.T, self.v, self.cv = M, horizon, speed, cv
        self.psi = rng.uniform(0, side, (M, 2))
        self.deadline = rng.choice(deadlines, M)
        self.tau = self.deadline / math.log(10.0)
        self.init_last = -rng.uniform(0, 1, M) * self.deadline
        n = rng.poisson(rate * horizon)
        self.t_dep = np.sort(rng.uniform(0, horizon, n))
        self.orig = rng.uniform(0, side, (n, 2))
        self.dest = rng.uniform(0, side, (n, 2))
        self.sigma, self.mu = lognorm_params(cv)
        self.f_true = (np.exp(self.mu + self.sigma * rng.standard_normal(n)) if cv > 0 else np.ones(n))
        d_op = np.linalg.norm(self.orig[:, None] - self.psi[None], axis=2)
        d_pd = np.linalg.norm(self.psi[None] - self.dest[:, None], axis=2)
        d_od = np.linalg.norm(self.orig - self.dest, axis=1)[:, None]
        self.d_op = d_op
        self.detour = np.maximum(d_op + d_pd - d_od, 0.0)
        self.n = n

    def sample_f(self, rng, size, fmax=None):
        """Speed factors, optionally conditioned on f < fmax (vehicle not yet arrived)."""
        if self.cv == 0:
            return np.ones(size)
        if fmax is None:
            return np.exp(self.mu + self.sigma * rng.standard_normal(size))
        p = stats.norm.cdf((math.log(fmax) - self.mu) / self.sigma) if fmax > 0 else 0.0
        u = rng.uniform(1e-12, max(p, 1e-12), size)
        return np.exp(self.mu + self.sigma * stats.norm.ppf(u))


def run(mis: Mission, policy: str, R: float, alpha: float, S: int, seed: int):
    rng = np.random.default_rng(seed)
    M, v = mis.M, mis.v
    done = [[t] for t in mis.init_last]        # realized samples per PsI (sorted)
    pending = []                               # (arrival_true, k, vehicle)
    pend_by_k = [[] for _ in range(M)]         # vehicles en route to k
    payout = cost = 0.0
    n_det = 0
    exp_err = []
    for i in range(mis.n):
        t0 = mis.t_dep[i]
        # release samples that happened before t0
        still = []
        for (a, k, j) in pending:
            if a <= t0:
                done[k].append(a)
                done[k].sort()
                pend_by_k[k].remove(j)
            else:
                still.append((a, k, j))
        pending = still
        # evaluate options
        f_own = mis.sample_f(rng, S)
        best_k, best_u = -1, 0.0
        for k in range(M):
            c = alpha * mis.detour[i, k]
            if c >= R:
                continue
            a_own = t0 + mis.d_op[i, k] / (v * f_own)                       # (S,)
            others = []
            for j in pend_by_k[k]:
                tj = mis.t_dep[j]
                fmax = mis.d_op[j, k] / (v * (t0 - tj)) if t0 > tj else None
                others.append(tj + mis.d_op[j, k] / (v * mis.sample_f(rng, S, fmax)))
            last = done[k][-1]
            if policy == "age":
                prev = np.full(S, last)
                a_use = np.full(S, a_own.mean())
            elif policy == "mean":
                a_use = np.full(S, a_own.mean())
                prev = np.full(S, last)
                for o in others:
                    m = o.mean()
                    prev = np.where((m < a_use) & (m > prev), m, prev)
            else:  # dist
                a_use = a_own
                prev = np.full(S, last)
                for o in others:
                    prev = np.where((o < a_use) & (o > prev), o, prev)
            gap = np.maximum(a_use - prev, 0.0)
            u = float((R * (1 - np.exp(-gap / mis.tau[k]))).mean()) - c
            if u > best_u:
                best_k, best_u = k, u
        if best_k < 0:
            continue
        k = best_k
        a_true = t0 + mis.d_op[i, k] / (v * mis.f_true[i])
        pending.append((a_true, k, i))
        pend_by_k[k].append(i)
        n_det += 1
        cost += alpha * mis.detour[i, k]
        exp_err.append(best_u + alpha * mis.detour[i, k])   # expected reward, scored below
    # flush, then score rewards on realized order
    for (a, k, j) in pending:
        done[k].append(a)
    rewards = 0.0
    realized = {}
    viol, worst = 0.0, []
    for k in range(M):
        s = sorted(done[k])
        for p_, a in zip(s[:-1], s[1:]):
            realized[(k, a)] = R * (1 - math.exp(-(a - p_) / mis.tau[k]))
        pts = [t for t in s if t <= mis.T] + [mis.T]
        gaps = []
        for a, b in zip(pts[:-1], pts[1:]):
            gaps.append(b - a)
            viol += max(0.0, b - max(a + mis.deadline[k], 0.0))
        worst.append(max(gaps))
    payout = sum(realized.values()) if realized else 0.0
    # realized minus expected reward per detour = forecast bias of the policy
    exp_total = sum(exp_err)
    return dict(policy=policy, R=R, cv=mis.cv,
                deadline_violation=viol / (M * mis.T),
                worst_gap=float(np.mean(worst)),
                detours=n_det, payout=payout,
                net_per_vehicle=(payout - cost) / mis.n,
                net_per_detour=(payout - cost) / max(n_det, 1),
                forecast_bias=(payout - exp_total) / max(n_det, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", type=int, default=30)
    ap.add_argument("--S", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    t = time.time()
    for cv in (0.0, 0.15, 0.3, 0.5):
        for R in (150.0, 400.0):
            for m in range(a.missions):
                mis = Mission(1000 + m, cv)
                for pol in ("dist", "mean", "age"):
                    r = run(mis, pol, R, alpha=0.4, S=a.S, seed=7 * m + 1)
                    r["mission"] = m
                    rows.append(r)
        print("cv", cv, "done", round(time.time() - t, 1), "s", flush=True)
    with open(os.path.join(OUT, "pilot.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
