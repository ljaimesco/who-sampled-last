"""Error terms of the simplified estimators and their moment bounds (theory section).

For player i at PsI k, given the competitors at k (vehicles en route and rivals choosing k) and the last realized
sample L, with f(g) = 1 - exp(-g / tau) (payment / R):

  true    u   = E f(G),           G = T_i - max({L} U {T_j : T_j < T_i})
  fixorder u_F = E f((T_i - T_p)^+), p = predicted predecessor (latest competitor whose MEAN precedes i's mean, else L)
  mean    u_M = f(mu_i - mu_p)
  ordermean u_O = sum_c P(c) f(E[G | c]), c = realized predecessor identity

Decomposition u_M - u = (u_F - u) + (u_M - u_F) = -order term + curvature term, with

  |u - u_F| <= P(E^c) <= sum_pairs Cantelli                  (order / predecessor-change term)
  |u_M - u_F| <= Var(Y) / (2 tau^2) + Scarf(D) / tau          (curvature term), D = T_i - T_p, Y = D^+
  0 <= u_O - u <= sum_c P(c) Var(G | c) / (2 tau^2)           (ordermean: curvature only)

Cantelli: P(X <= 0) <= s^2 / (s^2 + m^2) for E X = m > 0;  Scarf: E[D^-] <= (sqrt(s^2 + m^2) - m) / 2.
All values are per unit of R (multiply by R_k for credits).
"""
from __future__ import annotations

import math

import numpy as np


def cantelli(x: np.ndarray) -> float:
    """Moment bound on P(X <= 0) from E X > 0 (1 if E X <= 0)."""
    m, v = float(x.mean()), float(x.var())
    return 1.0 if m <= 0 else (v / (v + m * m) if v > 0 else 0.0)


def terms(a: np.ndarray, comps: list, L: float, tau: float) -> dict:
    """a: (S,) draws of my arrival; comps: list of (S,) competitor draws at this PsI; L: last sample time."""
    S = len(a)
    f = lambda g: 1.0 - np.exp(-g / tau)
    # true predecessor
    prev = np.full(S, float(L)); pid = np.full(S, -1)
    for c, t in enumerate(comps):
        ok = (t < a) & (t > prev)
        prev, pid = np.where(ok, t, prev), np.where(ok, c, pid)
    G = np.maximum(a - prev, 0.0)
    u = float(f(G).mean())
    # predicted predecessor by means
    mu = float(a.mean()); means = [float(t.mean()) for t in comps]
    before = [c for c, m in enumerate(means) if m < mu and m > L]
    p = max(before, key=lambda c: means[c]) if before else None
    Tp = comps[p] if p is not None else np.full(S, float(L))
    D = a - Tp; Y = np.maximum(D, 0.0)
    uF = float(f(Y).mean())
    uM = float(f(np.array(max(mu - (means[p] if p is not None else L), 0.0))))
    # ordermean
    uO = 0.0; om_b = 0.0
    for c in np.unique(pid):
        m = pid == c; w = m.mean()
        uO += w * float(f(np.array(G[m].mean())))
        om_b += w * float(G[m].var()) / (2 * tau * tau)
    # order-term bound: pairs whose sign must hold for p to be the realized predecessor
    flips, pe = 0.0, 0.0
    if p is not None:
        flips += cantelli(a - Tp)
    mp = means[p] if p is not None else float(L)
    for c, t in enumerate(comps):
        if c == p:
            continue
        if means[c] >= mu:                      # predicted at or after me, could arrive before
            flips += cantelli(t - a)
        else:                                   # predicted before p (or before L), could arrive between p and me
            flips += cantelli(Tp - t)
    # exact predecessor-change probability (event E^c)
    if p is not None:
        pe = float(np.mean(pid != p))
    else:
        pe = float(np.mean(pid != -1))
    # curvature bound
    s2 = float(D.var()); m1 = float(D.mean())
    scarf = 0.0 if p is None else (math.sqrt(s2 + m1 * m1) - m1) / 2 if m1 > 0 else float("inf")
    curv_b = float(Y.var()) / (2 * tau * tau) + scarf / tau
    return dict(u=u, uF=uF, uM=uM, uO=uO, order=uF - u, curv=uM - uF, om=uO - u, pchange=pe,
                order_b=min(flips, 1.0), curv_b=curv_b, om_b=om_b, varD=s2, meanD=m1, n_comp=len(comps),
                has_pred=p is not None, pay=f(G))
