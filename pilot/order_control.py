"""Causality control: is the gain of the distribution-aware estimator due to
the uncertain ARRIVAL ORDER, or only to the concavity of the reward
(E[delta(G)] != delta(E[G])) ?

Travel time = d / v + eps, with additive noise of standard deviation sigma.
Two noise models with the SAME marginal spread:
  common      : one shock per (PsI, decision window), shared by every vehicle
                heading to that PsI in that window. Gaps between those
                vehicles are unchanged, so their order never flips; only the
                gap to already realized samples is random (concavity only).
  independent : one shock per vehicle (order between vehicles can flip).
Policies (sequential, as in the first pilot): dist uses the correct joint
distribution of the noise model; mean uses mean arrival times.
If the dist-vs-mean gain is similar under both models, it comes from
concavity. If it appears mainly under independent noise, it comes from the
uncertain order.
"""
from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def truncnorm_above(rng, lo, sigma, size):
    """N(0, sigma) conditioned on eps > lo."""
    a = lo / sigma
    return stats.truncnorm.rvs(a, np.inf, scale=sigma, size=size, random_state=rng)


def run(seed, sigma, model, policy, R, S=128, M=20, side=2000.0, T=1800.0, rate=1 / 6.0, v=10.0,
        alpha=0.4, window=60.0):
    rs = np.random.default_rng(seed)
    psi = rs.uniform(0, side, (M, 2))
    deadline = rs.choice((120.0, 180.0, 300.0), M)
    tau = deadline / math.log(10.0)
    done = [[-x] for x in rs.uniform(0, 1, M) * deadline]
    n = rs.poisson(rate * T)
    tdep = np.sort(rs.uniform(0, T, n))
    o, d = rs.uniform(0, side, (n, 2)), rs.uniform(0, side, (n, 2))
    dop = np.linalg.norm(o[:, None] - psi[None], axis=2)
    cost = alpha * np.maximum(dop + np.linalg.norm(psi[None] - d[:, None], axis=2)
                              - np.linalg.norm(o - d, axis=1)[:, None], 0.0)
    W = int(T // window) + 1
    eps_common = rs.normal(0, sigma, (M, W))     # truth, common model
    eps_indiv = rs.normal(0, sigma, n)           # truth, independent model
    win = (tdep // window).astype(int)
    rng = np.random.default_rng(seed + 99)
    pending, by_k = [], [[] for _ in range(M)]
    payout = cst = 0.0
    commits = []
    for i in range(n):
        t0 = tdep[i]
        keep = []
        for (a, k, j) in pending:
            if a <= t0:
                done[k].append(a); done[k].sort(); by_k[k].remove(j)
            else:
                keep.append((a, k, j))
        pending = keep
        best, bu = -1, 0.0
        for k in range(M):
            c = cost[i, k]
            if c >= R:
                continue
            base_i = tdep[i] + dop[i, k] / v
            last = done[k][-1]
            if policy == "mean":
                a_i = base_i
                prev = last
                for j in by_k[k]:
                    bj = tdep[j] + dop[j, k] / v
                    lo = t0 - bj                      # j has not arrived yet: eps_j > lo
                    mj = bj + (stats.truncnorm.mean(lo / sigma, np.inf, scale=sigma) if sigma > 0 else 0.0)
                    if mj < a_i and mj > prev:
                        prev = mj
                gap = max(a_i - prev, 0.0)
                u = R * (1 - math.exp(-gap / tau[k])) - c
            else:
                if model == "common":
                    # shock shared with vehicles of the same window heading to k
                    lo = -np.inf
                    same = [j for j in by_k[k] if win[j] == win[i]]
                    for j in same:
                        lo = max(lo, t0 - (tdep[j] + dop[j, k] / v))
                    e = truncnorm_above(rng, lo, sigma, S) if (sigma > 0 and same) else rng.normal(0, sigma, S) if sigma > 0 else np.zeros(S)
                    a_i = base_i + e
                    prev = np.full(S, last)
                    for j in by_k[k]:
                        bj = tdep[j] + dop[j, k] / v
                        if win[j] == win[i]:
                            aj = bj + e
                        else:
                            aj = bj + (truncnorm_above(rng, t0 - bj, sigma, S) if sigma > 0 else 0.0)
                        prev = np.where((aj < a_i) & (aj > prev), aj, prev)
                else:
                    a_i = base_i + rng.normal(0, sigma, S)
                    prev = np.full(S, last)
                    for j in by_k[k]:
                        bj = tdep[j] + dop[j, k] / v
                        aj = bj + (truncnorm_above(rng, t0 - bj, sigma, S) if sigma > 0 else 0.0)
                        prev = np.where((aj < a_i) & (aj > prev), aj, prev)
                gap = np.maximum(a_i - prev, 0.0)
                u = float((R * (1 - np.exp(-gap / tau[k]))).mean()) - c
            if u > bu:
                best, bu = k, u
        if best < 0:
            continue
        k = best
        e = eps_common[k, win[i]] if model == "common" else eps_indiv[i]
        a = max(tdep[i] + dop[i, k] / v + e, t0 + 1e-6)
        pending.append((a, k, i)); by_k[k].append(i)
        commits.append((k, a)); cst += cost[i, k]
    for (a, k, j) in pending:
        done[k].append(a)
    viol = 0.0
    pay = 0.0
    for k in range(M):
        s = sorted(done[k])
        for p_, a in zip(s[:-1], s[1:]):
            if (k, a) in set(commits):
                pay += R * (1 - math.exp(-(a - p_) / tau[k]))
        pts = [t for t in s if t <= T] + [T]
        for x, y in zip(pts[:-1], pts[1:]):
            viol += max(0.0, y - max(x + deadline[k], 0.0))
    return viol / (M * T), (pay - cst) / n


if __name__ == "__main__":
    missions = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sigma in (15.0, 30.0, 60.0):
        for model in ("common", "independent"):
            for m in range(missions):
                for pol in ("dist", "mean"):
                    vi, ne = run(500 + m, sigma, model, pol, 400.0)
                    rows.append(dict(sigma=sigma, model=model, policy=pol, mission=m, viol=vi, net=ne))
        print("sigma", sigma, "done", flush=True)
    with open(os.path.join(OUT, "order_control.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
