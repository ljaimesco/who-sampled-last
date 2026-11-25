"""Single-stop missions on the ANTE-Mobility London grid with edge-and-time congestion.

Travel time of a vehicle that ENTERS edge e at time t:

    T = exp( mu_e + sigma_e * ( sqrt(rho) * Z_e(t) + sqrt(1 - rho) * eps ) ),

  mu_e, sigma_e  lognormal parameters matched by moments to the edge's real mean
                 and std in ANTE-Mobility/uberspeed.xml (Uber movement data);
  Z_e(t)         stationary standard-normal congestion process of edge e, shared
                 by every vehicle entering e, with correlation exp(-|t - t'|/ell)
                 (AR(1) on a grid of dt seconds, piecewise constant within a bin);
                 independent across edges;
  eps            the vehicle's own standard-normal shock on that edge.

Every edge's marginal travel-time law is exactly the calibrated lognormal for
every rho; rho only moves variance from the individual to the shared component.
Vehicles are simulated edge by edge from their departure, so a delay upstream
changes the time, and hence the congestion, at which they enter downstream edges.
The data supplies marginals only, so this is a London-CALIBRATED correlation
sensitivity model, not a validated dependence model.

Information at a decision window: realized samples, the plans of vehicles still
en route and the fact that they have not arrived, and the prior law of traffic
(observed congestion is not used to update beliefs; stated simplification).
Conditioning on "not yet arrived" is approximate: offending draws get fresh
individual shocks up to 20 times, then are set to the window start.
"""
from __future__ import annotations

import bisect
import math
import os
import random

import networkx as nx
import numpy as np
from scipy.signal import lfilter

from . import real_london as rl
from .model import OPT_OUT
from .policies import spa
from . import fresh_greedy as _gr
from .stochastic import StochGame, profile_regret

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")   # copies from ANTE-Mobility
NOD = os.path.join(_DATA, "london-seg4.100.nod.xml")
UBER = os.path.join(_DATA, "uberspeed.xml")
_G = None


def set_edge_stats(mean, std):
    """Replace every edge's travel-time mean/std (arrays in graph edge order) and refresh mu, sigma and the
    routing weights. Used to plug in a forecast model learned from SUMO warm-up runs."""
    G = graph()
    g = G["g"]
    for i, e in enumerate(G["edges"]):
        m, sd = float(mean[i]), float(std[i])
        s2 = math.log1p((sd / m) ** 2)
        g.edges[e].update(mean=m, std=sd, mu=math.log(m) - s2 / 2, sigma=math.sqrt(s2))
        G["mu"][i], G["sg"][i] = math.log(m) - s2 / 2, math.sqrt(s2)
    G["rev"] = g.reverse(copy=True)


def graph():
    global _G
    if _G is None:
        lg = rl.largest_component(rl.load(NOD, UBER))
        g = lg.graph
        edges = list(g.edges())
        eid = {e: i for i, e in enumerate(edges)}
        mu = np.array([g.edges[e]["mu"] for e in edges])
        sg = np.array([g.edges[e]["sigma"] for e in edges])
        xy = np.array([[(lg.coords[a][0] + lg.coords[b][0]) / 2, (lg.coords[a][1] + lg.coords[b][1]) / 2] for a, b in edges])
        _G = dict(g=g, rev=g.reverse(copy=True), coords=lg.coords, eid=eid, mu=mu, sg=sg, xy=xy, edges=edges)
    return _G


class LondonScenario:
    def __init__(self, seed, rho, ell=300.0, M=20, center=(2550.0, 2550.0), half=1000.0, horizon=1800.0,
                 rate=1 / 6.0, deadlines=(120.0, 180.0, 300.0), beta=1.5, window=60.0, dt=30.0,
                 area=1000.0, w=0.0, sig_scale=1.0):
        """w: share of each edge's congestion process that is an AREA process shared by every edge whose
        midpoint lies in the same area x area cell (stationary AR(1), same correlation length); w = 0 gives
        purely edge-level congestion. Edge marginals are unchanged for every (rho, w)."""
        G = graph()
        rng = np.random.default_rng(seed)
        self.seed, self.rho, self.ell, self.dt, self.w = seed, rho, ell, dt, w
        cell = np.floor(G["xy"] / area).astype(int)
        self.zone = cell[:, 0] * 100_000 + cell[:, 1]                         # area cell of every edge
        self.M, self.T, self.window = M, horizon, window
        # sig_scale > 1 widens every edge's log-spread while keeping its mean fixed (variance-matched control)
        self.sg = G["sg"] * sig_scale
        self.mu = G["mu"] + (G["sg"] ** 2 - self.sg ** 2) / 2
        nodes = [v for v, (x, y) in G["coords"].items() if v in G["g"]
                 and abs(x - center[0]) <= half and abs(y - center[1]) <= half]
        nodes.sort()
        self.psi = [nodes[i] for i in rng.choice(len(nodes), M, replace=False)]
        self.deadline = rng.choice(deadlines, M)
        self.tau = self.deadline / math.log(10.0)
        self.init_last = -rng.uniform(0, 1, M) * self.deadline
        n = self.n = rng.poisson(rate * horizon)
        self.t_dep = np.sort(rng.uniform(0, horizon, n))
        orig = [nodes[i] for i in rng.integers(0, len(nodes), n)]
        dest = [nodes[i] for i in rng.integers(0, len(nodes), n)]
        self.orig, self.dest = orig, dest
        # shortest (mean-time) paths: origin -> PsI via reverse Dijkstra from each PsI
        self.path = [[None] * M for _ in range(n)]
        t_op = np.zeros((n, M)); t_pd = np.zeros((n, M))
        for k, p in enumerate(self.psi):
            dist_in, paths_in = nx.single_source_dijkstra(G["rev"], p, weight="mean")
            dist_out = nx.single_source_dijkstra_path_length(G["g"], p, weight="mean")
            for v in range(n):
                nd = list(reversed(paths_in[orig[v]]))                 # orig -> ... -> psi
                self.path[v][k] = np.array([G["eid"][(a, b)] for a, b in zip(nd, nd[1:])], dtype=int)
                t_op[v, k] = sum(math.exp(self.mu[e] + self.sg[e] ** 2 / 2) for e in self.path[v][k])
                t_pd[v, k] = dist_out[dest[v]]
        t_od = np.array([nx.shortest_path_length(G["g"], orig[v], dest[v], weight="mean") for v in range(n)])
        self.cost = beta * np.maximum(t_op + t_pd - t_od[:, None], 0.0)    # credits per second of expected detour
        self.base = t_op
        self.t_pd, self.t_od, self.beta = t_pd, t_od, beta          # kept for multi-stop route costs
        self.arr_mean = self.t_dep[:, None] + t_op
        # the true world: one realization of every congestion process and every individual shock
        self.edges = np.unique(np.concatenate([self.path[v][k] for v in range(n) for k in range(M)]))
        self.e_pos = {e: i for i, e in enumerate(self.edges)}
        self.t_origin = -600.0
        B = int((horizon + 3000.0 - self.t_origin) // dt) + 1
        self.Z_true = ar1(rng, len(self.edges), B, 1, ell, dt)[:, :, 0]
        self.eps_true = rng.standard_normal((n, len(self.edges))).astype(np.float32)
        if w > 0:
            zu, zi = np.unique(self.zone[self.edges], return_inverse=True)
            Za = ar1(rng, len(zu), B, 1, ell, dt)[:, :, 0]
            self.Z_true = math.sqrt(1 - w) * self.Z_true + math.sqrt(w) * Za[zi]
        self.arr_true = np.zeros((n, M))
        for v in range(n):
            for k in range(M):
                t = self.t_dep[v]
                for e in self.path[v][k]:
                    j = self.e_pos[e]
                    b = min(int((t - self.t_origin) // dt), B - 1)
                    t += math.exp(self.mu[e] + self.sg[e] * (math.sqrt(rho) * self.Z_true[j, b]
                                                             + math.sqrt(1 - rho) * self.eps_true[v, j]))
                self.arr_true[v, k] = t


def ar1(rng, E, B, S, ell, dt):
    """E independent stationary AR(1) standard-normal processes, B bins, S draws (float32), shape (E, B, S)."""
    phi = math.exp(-dt / ell); q = math.sqrt(1 - phi * phi)
    xi = rng.standard_normal((B, E, S)); xi[0] /= q
    return np.moveaxis(lfilter([q], [1.0, -phi], xi, axis=0), 0, 1).astype(np.float32)


def _simulate(scn, rows_v, E_idx, U_idx, eps, Z, t_from, B, sr, si):
    """Advance all routes together, one edge step at a time, entering each edge at its simulated time."""
    S = eps.shape[1]
    ar = np.arange(S)[None, :]
    t = np.repeat(scn.t_dep[rows_v][:, None], S, axis=1)
    for l in range(E_idx.shape[1]):
        act = np.nonzero(E_idx[:, l] >= 0)[0]
        if len(act) == 0:
            break
        el = E_idx[act, l]
        b = np.clip(((t[act] - t_from) // scn.dt).astype(int), 0, B - 1)
        z = Z[b, el[:, None], ar]
        ge = scn.edges_local_g[el]
        t[act] += np.exp(scn.mu[ge][:, None] + scn.sg[ge][:, None] * (sr * z + si * eps[U_idx[act, l]]))
    return t


def window_samples(scn, players, t0, pending, S, rng, R):
    """Joint arrival samples for one window: per draw, one congestion field shared by all vehicles;
    each vehicle keeps the same individual shock on an edge across its alternative routes."""
    M = scn.M
    pairs = [(v, k) for v in players for k in range(M) if scn.cost[v, k] < R]
    pp = [(j, k) for (j, k) in pending]
    allp = pairs + pp
    t_from = min([scn.t_dep[v] for v, _ in allp] + [t0]) - scn.dt
    B = int((t0 + scn.window + 1500.0 - t_from) // scn.dt) + 1
    loc, keys = {}, {}
    L = max(len(scn.path[v][k]) for v, k in allp)
    E_idx = -np.ones((len(allp), L), dtype=int); U_idx = np.zeros((len(allp), L), dtype=int)
    for r, (v, k) in enumerate(allp):
        for l, e in enumerate(scn.path[v][k]):
            E_idx[r, l] = loc.setdefault(e, len(loc))
            U_idx[r, l] = keys.setdefault((v, e), len(keys))
    scn.edges_local_g = np.array(list(loc.keys()), dtype=int)
    phi = math.exp(-scn.dt / scn.ell); q = math.sqrt(1 - phi * phi)
    xi = rng.standard_normal((B, len(loc), S), dtype=np.float32); xi[0] /= q   # stationary start
    Z = lfilter(np.float32([q]), np.float32([1.0, -phi]), xi, axis=0)
    del xi
    eps = rng.standard_normal((len(keys), S), dtype=np.float32)
    if scn.w > 0:
        zu, zi = np.unique(scn.zone[scn.edges_local_g], return_inverse=True)
        xa = rng.standard_normal((B, len(zu), S), dtype=np.float32); xa[0] /= q
        Za = lfilter(np.float32([q]), np.float32([1.0, -phi]), xa, axis=0)
        Z = np.float32(math.sqrt(1 - scn.w)) * Z + np.float32(math.sqrt(scn.w)) * Za[:, zi, :]
        del xa, Za
    sr, si = math.sqrt(scn.rho), math.sqrt(1 - scn.rho)
    rows_v = np.array([v for v, _ in allp], dtype=int)
    T = _simulate(scn, rows_v, E_idx, U_idx, eps, Z, t_from, B, sr, si)
    # en-route vehicles: condition on "not yet arrived" by redrawing their individual shocks
    pr = np.arange(len(pairs), len(allp))
    if len(pr):
        Tp = T[pr]
        for _ in range(20):
            bad = Tp <= t0
            if not bad.any():
                break
            eps2 = eps.copy()
            u = np.unique(U_idx[pr][E_idx[pr] >= 0])
            eps2[u] = rng.standard_normal((len(u), S), dtype=np.float32)
            Tn = _simulate(scn, rows_v[pr], E_idx[pr], U_idx[pr], eps2, Z, t_from, B, sr, si)
            Tp = np.where(bad, Tn, Tp)
        T[pr] = np.maximum(Tp, t0 + 1e-3)
    A = np.repeat(scn.arr_mean[players][:, :, None], S, axis=2).astype(float)
    pos = {v: a for a, v in enumerate(players)}
    for r, (v, k) in enumerate(pairs):
        A[pos[v], k] = T[r]
    P = [[] for _ in range(M)]
    for r, (j, k) in enumerate(pp):
        P[k].append(T[len(pairs) + r])
    P = [np.array(p) if p else np.zeros((0, S)) for p in P]
    return A, P


def make_game(scn, players, A, P, last, R, est, rivals=True):
    if est == "mean":
        A = A.mean(axis=2, keepdims=True)
        P = [p.mean(axis=1, keepdims=True) if len(p) else np.zeros((0, 1)) for p in P]
    mode = est if (est.startswith("fixorder") or est == "ordermean") else "dist"
    return StochGame(A, P, last, scn.tau, scn.cost[players], R, mode=mode, rivals=rivals)


ESTS = ("dist", "ordermean", "fixorder", "mean")


def run(scn: LondonScenario, est: str, R: float, S: int = 256, seed: int = 0, regret=None, ests=ESTS, diag=False):
    """Mission playing estimator `est` each window. regret=(S_sel, S_eval): also score every
    estimator's profile at each window state on independent draws (all estimators solve the SAME
    decision draws). diag: collect arrival correlation and Var(Ti - Tj) ratios of co-located pairs."""
    rng = np.random.default_rng(seed)
    done = [[t] for t in scn.init_last]
    pending, commits = [], {}
    st = dict(breaks=0, cycles=0, unstable=0, games=0, expected=0.0)
    reg = {e: dict(win=[], cut=[], win400=[], rfinal=[]) for e in ests} if regret else None
    dg = dict(corr=[], ratio=[]) if diag else None
    wR = {}   # vehicle -> reward in force for its window after any cycle cuts
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
            A, P = window_samples(scn, players, t0, pend, S, rng, R)
            if diag and len(players) > 1:
                for k in range(scn.M):
                    ok = [a for a, v in enumerate(players) if scn.cost[v, k] < R]
                    if len(ok) < 2: continue
                    C = np.cov(A[ok, k, :])
                    for x in range(len(ok)):
                        for y in range(x + 1, len(ok)):
                            vx, vy, c = C[x, x], C[y, y], C[x, y]
                            if vx > 0 and vy > 0:
                                dg["corr"].append(c / math.sqrt(vx * vy))
                                dg["ratio"].append((vx + vy - 2 * c) / (vx + vy))
            games = {e: make_game(scn, players, A, P, last, R, _gr.base(e)) for e in set(ests) | {est}}
            g = games[est]
            r = _gr.solve(est, games, random.Random(seed * 31 + widx))
            assign = r.assign
            st["games"] += 1; st["breaks"] += r.breaks; st["cycles"] += int(r.cycled)
            st["unstable"] += int(not (r.converged and g.is_nash(assign)))
            if regret:
                As, Ps = window_samples(scn, players, t0, pend, regret[0], np.random.default_rng(1_000_003 * (seed + 1) + widx), R)
                Ae, Pe = window_samples(scn, players, t0, pend, regret[1], np.random.default_rng(2_000_003 * (seed + 1) + widx), R)
                for e in ests:
                    if e == est:
                        prof, nb, Re = assign, r.breaks, g.R.copy()
                    else:
                        re_ = _gr.solve(e, games, random.Random(seed * 31 + widx))
                        prof, nb, Re = re_.assign, re_.breaks, games[e].R.copy()
                    gs = StochGame(As, Ps, last, scn.tau, scn.cost[players], Re)
                    ge = StochGame(Ae, Pe, last, scn.tau, scn.cost[players], Re)
                    reg[e]["win"].append(max(profile_regret(gs, ge, prof)) / R)
                    reg[e]["cut"].append(int(nb > 0))
                    reg[e]["rfinal"].append(float(Re.min()) / R)
                    if nb > 0:   # regret of the same profile under the announced reward R (no cut)
                        R0 = np.full(scn.M, R)
                        gs0 = StochGame(As, Ps, last, scn.tau, scn.cost[players], R0)
                        ge0 = StochGame(Ae, Pe, last, scn.tau, scn.cost[players], R0)
                        reg[e]["win400"].append(max(profile_regret(gs0, ge0, prof)) / R)
                    else:
                        reg[e]["win400"].append(reg[e]["win"][-1])
            for i, k in enumerate(assign):
                if k == OPT_OUT: continue
                v = players[i]
                a_true = float(scn.arr_true[v, k])
                pending.append((a_true, v, k))
                commits[v] = (k, a_true, float(scn.cost[v, k]))
                wR[v] = float(g.R[k])
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
    payout = sum(pay[(k, a)] * wR[v] / R for v, (k, a, c) in commits.items())   # each vehicle paid at its window's announced reward
    cost = sum(c for (k, a, c) in commits.values())
    viol, worst = 0.0, []
    for k in range(scn.M):
        pts = [t for t in done[k] if t <= scn.T] + [scn.T]
        gaps = []
        for a, b in zip(pts[:-1], pts[1:]):
            gaps.append(b - a)
            viol += max(0.0, b - max(a + scn.deadline[k], 0.0))
        worst.append(max(gaps))
    inv = tot = 0
    by_k = {}
    for v, (k, a, c) in commits.items():
        by_k.setdefault(k, []).append((float(scn.arr_mean[v, k]), a))
    for lst in by_k.values():
        lst.sort()
        for (m1, a1), (m2, a2) in zip(lst[:-1], lst[1:]):
            tot += 1; inv += int(a2 < a1)
    out = dict(payout_nominal=payout_nominal, net_nominal=(payout_nominal - cost) / scn.n, est=est, rho=scn.rho, ell=scn.ell, deadline_violation=viol / (scn.M * scn.T),
               worst_gap=float(np.mean(worst)), detours=len(commits), vehicles=scn.n, payout=payout,
               net_per_vehicle=(payout - cost) / scn.n, inversion_rate=inv / max(tot, 1),
               breaks=st["breaks"], unstable_games=st["unstable"], games=st["games"])
    return out, reg, dg
