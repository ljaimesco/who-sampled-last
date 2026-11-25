"""Multi-stop extension: each vehicle may sample up to two PsIs in an ordered route (as in SenseNow).

Action of vehicle i: opt out, a single PsI (k,), or an ordered pair (k1, k2), k1 != k2. Utility:
    U_i = sum over the route's stops s of R_k * delta_i(s) - cost_i(route),
where delta_i(s) = E[1 - exp(-(a_i^s - prev)/tau_k)] and prev is the latest earlier sample at that PsI: the last
realized sample, a vehicle still en route to it (as any of its stops), or another player whose chosen route
contains it (at that player's arrival time there under its route). The second stop's arrival includes the
random first leg and the random k1 -> k2 leg, sampled jointly under the same congestion field.
Estimators (dist, ordermean, fixorder, mean) are defined exactly as in stochastic.StochGame, stop by stop.
With routes restricted to single stops the game coincides with StochGame (tested in tests/test_multistop.py).
"""
from __future__ import annotations

import bisect
import math
import random

import networkx as nx
import numpy as np
from scipy.signal import lfilter

from . import london_mission as lm
from .model import Game, OPT_OUT, Params
from .policies import spa
from . import fresh_greedy as _gr
from .stochastic import profile_regret


class MultiGame(Game):
    def __init__(self, routes, options, own, pend, last, tau, cost, R, mode="dist", rivals=True, params=Params()):
        """routes: list of route tuples (global ids); options[i]: [OPT_OUT] + route ids open to player i;
        own[(i, r)]: list of (S,) arrays, arrival of i at each stop of route r; pend: list of M arrays (n_k, S);
        cost[(i, r)]: detour cost of route r for i; R: scalar or (M,)."""
        self.p, self.mode, self.rivals = params, mode, rivals
        self.routes, self.options, self.own, self.costd = routes, options, own, cost
        self.N, self.M = len(options), len(last)
        self.last, self.tau = np.asarray(last, float), np.asarray(tau, float)
        self.R = np.full(self.M, float(R)) if np.isscalar(R) else np.asarray(R, float).copy()
        self.P = pend
        any_arr = next(iter(own.values()))[0] if own else np.zeros(1)
        self.S = len(any_arr)
        touched = [set(k for r in options[j] if r != OPT_OUT for k in routes[r]) for j in range(self.N)]
        self.feas = np.array([[bool(touched[j] & set(routes[r])) for r in range(len(routes))] for j in range(self.N)])
        self.evals, self._cache, self._base = 0, {}, {}
        if mode.startswith("fixorder"):
            self.ownm = {key: [float(x.mean()) for x in v] for key, v in own.items()}
            self.Pm = [P.mean(axis=1) if len(P) else np.zeros(0) for P in pend]

    def _rivals(self, i, k, assign):
        if not self.rivals:
            return ()
        out = []
        for j, rj in enumerate(assign):
            if j != i and rj != OPT_OUT and k in self.routes[rj]:
                out.append((j, rj, self.routes[rj].index(k)))
        return tuple(out)

    def delta(self, i, r, s, assign):
        k = self.routes[r][s]
        riv = self._rivals(i, k, assign)
        key = (i, r, s, riv)
        v = self._cache.get(key)
        if v is not None:
            return v
        a = self.own[(i, r)][s]
        tau = self.tau[k]
        if self.mode == "ordermean":
            prev = np.full(self.S, self.last[k]); pid = np.full(self.S, -1)
            for q, row in enumerate(self.P[k] if self.rivals else []):
                ok = (row < a) & (row > prev)
                prev, pid = np.where(ok, row, prev), np.where(ok, 10_000 + q, pid)
            for (j, rj, sj) in riv:
                aj = self.own[(j, rj)][sj]
                ok = (aj < a) & (aj > prev)
                prev, pid = np.where(ok, aj, prev), np.where(ok, j, pid)
            gap = np.maximum(a - prev, 0.0)
            v = 0.0
            for c in np.unique(pid):
                m = pid == c
                v += m.mean() * (1.0 - math.exp(-gap[m].mean() / tau))
            v = float(v)
        elif self.mode.startswith("fixorder"):
            am, pm, pred = self.ownm[(i, r)][s], self.last[k], self.last[k]
            for row, m in zip(self.P[k] if self.rivals else [], self.Pm[k] if self.rivals else []):
                if pm < m < am:
                    pm, pred = m, row
            for (j, rj, sj) in riv:
                m = self.ownm[(j, rj)][sj]
                if (pm, -1) < (m, j) < (am, i):
                    pm, pred = m, self.own[(j, rj)][sj]
            gap = np.maximum(a - pred, 0.0)
            v = float(np.mean(1.0 - np.exp(-gap / tau)))
        else:
            bk = (i, r, s)
            prev = self._base.get(bk)
            if prev is None:
                prev = np.full(self.S, self.last[k])
                for row in (self.P[k] if self.rivals else []):
                    ok = (row < a) & (row > prev)
                    prev = np.where(ok, row, prev)
                self._base[bk] = prev
            prev = prev.copy()
            for (j, rj, sj) in riv:
                aj = self.own[(j, rj)][sj]
                ok = (aj < a) & (aj > prev)
                prev = np.where(ok, aj, prev)
            gap = np.maximum(a - prev, 0.0)
            v = float(np.mean(1.0 - np.exp(-gap / tau)))
        self._cache[key] = v
        return v

    def D(self, i, r, assign):
        """Sum of reward-weighted multipliers / R (for uniform R): sum_s delta."""
        return sum(self.delta(i, r, s, assign) for s in range(len(self.routes[r])))

    def utility(self, i, r, assign):
        self.evals += 1
        if r == OPT_OUT:
            return 0.0
        return sum(self.R[k] * self.delta(i, r, s, assign) for s, k in enumerate(self.routes[r])) - self.costd[(i, r)]

    def break_cycle(self, states, moves, mode="global"):
        """As Game.break_cycle (global mode), with a route's multiplier D = sum of its stops' deltas:
        a move x -> y is a best response only while R D_y - C_y > max(R D_x - C_x, 0)."""
        best = None
        for t, (st, (i, x, y)) in enumerate(zip(states, moves)):
            if y == OPT_OUT:
                continue
            a = list(st)
            dy, cy = self.D(i, y, a), self.costd[(i, y)]
            dx, cx = (self.D(i, x, a), self.costd[(i, x)]) if x != OPT_OUT else (0.0, 0.0)
            if dy <= 0:
                continue
            r = cy / dy
            if dy > dx and cy > cx:
                r = max(r, (cy - cx) / (dy - dx))
            if best is None or r > best[0]:
                best = (r, t)
        if best is None:
            raise RuntimeError("cycle without an eligible transition")
        self.R[:] = best[0]
        self._cache.clear()
        return best[0], best[1]


def make_multi(routes, options, own, pend, last, tau, cost, R, est, rivals=True):
    if est == "mean":
        own = {key: [x.mean(keepdims=True) for x in v] for key, v in own.items()}
        pend = [p.mean(axis=1, keepdims=True) if len(p) else np.zeros((0, 1)) for p in pend]
    mode = est if (est.startswith("fixorder") or est == "ordermean") else "dist"
    return MultiGame(routes, options, own, pend, last, tau, cost, R, mode=mode, rivals=rivals)


class RouteSampler:
    """Joint samples of checkpoint times along edge lists under the London edge-and-time congestion model
    (one shared field per draw; a vehicle keeps its individual shock on an edge across its routes)."""

    def __init__(self, scn, rows, t0, S, rng, scale=1.0, sigma_gain=1.0):
        """rows: list of (vehicle, start time, [edge indices], fraction of the first edge left, [checkpoints]),
        a checkpoint c meaning 'time after the first c edges' (0 = start)."""
        self.scn, self.rows, self.S, self.rng, self.scale = scn, rows, S, rng, scale
        starts = [r[1] for r in rows]
        self.t_from = min(starts + [t0]) - scn.dt
        self.B = int((t0 + scn.window + 2100.0 - self.t_from) // scn.dt) + 1
        loc, keys = {}, {}
        L = max([len(r[2]) for r in rows] + [1])
        self.E = -np.ones((len(rows), L), dtype=int); self.U = np.zeros((len(rows), L), dtype=int)
        self.frac = np.ones((len(rows), L), dtype=np.float32)
        for i, (v, st, edges, f0, cps) in enumerate(rows):
            for l, e in enumerate(edges):
                self.E[i, l] = loc.setdefault(e, len(loc))
                self.U[i, l] = keys.setdefault((v, e), len(keys))
            if edges:
                self.frac[i, 0] = f0
        self.g = np.array(list(loc.keys()), dtype=int)
        phi = math.exp(-scn.dt / scn.ell); q = math.sqrt(1 - phi * phi)
        xi = rng.standard_normal((self.B, max(len(loc), 1), S), dtype=np.float32); xi[0] /= q
        self.Z = lfilter(np.float32([q]), np.float32([1.0, -phi]), xi, axis=0)
        self.eps = rng.standard_normal((max(len(keys), 1), S), dtype=np.float32)
        self.sg = scn.sg * sigma_gain
        self.mu = scn.mu + (scn.sg ** 2 - self.sg ** 2) / 2
        self.sr, self.si = math.sqrt(scn.rho), math.sqrt(1 - scn.rho)

    def sim(self, idx=None, fresh=False):
        """Checkpoint times for rows idx (default all): list of lists of (S,) arrays."""
        idx = np.arange(len(self.rows)) if idx is None else np.asarray(idx)
        S, ar = self.S, np.arange(self.S)[None, :]
        E, U, fr = self.E[idx], self.U[idx], self.frac[idx]
        eps = self.eps if not fresh else self.rng.standard_normal(self.eps.shape, dtype=np.float32)
        t = np.repeat(np.array([self.rows[i][1] for i in idx], dtype=float)[:, None], S, axis=1)
        cps = [self.rows[i][4] for i in idx]
        out = [[None] * len(c) for c in cps]
        for a, c in enumerate(cps):
            for q, cp in enumerate(c):
                if cp == 0:
                    out[a][q] = t[a].copy()
        for l in range(E.shape[1]):
            act = np.nonzero(E[:, l] >= 0)[0]
            if len(act):
                el = E[act, l]
                b = np.clip(((t[act] - self.t_from) // self.scn.dt).astype(int), 0, self.B - 1)
                z = self.Z[b, el[:, None], ar]
                ge = self.g[el]
                t[act] += self.scale * fr[act, l][:, None] * np.exp(self.mu[ge][:, None] + self.sg[ge][:, None] * (self.sr * z + self.si * eps[U[act, l]]))
            for a, c in enumerate(cps):
                for q, cp in enumerate(c):
                    if cp == l + 1:
                        out[a][q] = t[a].copy()
        return out


class MultiLondon:
    """Multi-stop layer on a LondonScenario: PsI-to-PsI paths, candidate routes, route costs and true arrivals."""

    def __init__(self, scn: lm.LondonScenario, R: float = 400.0, max_stops: int = 2, K: int = 6):
        G = lm.graph()
        self.scn, self.max_stops, self.R = scn, max_stops, R
        M, n = scn.M, scn.n
        self.pp_path = [[None] * M for _ in range(M)]
        self.t_pp = np.zeros((M, M))
        for a, p in enumerate(scn.psi):
            dist, paths = nx.single_source_dijkstra(G["g"], p, weight="mean")
            for b, q in enumerate(scn.psi):
                nd = paths[q]
                self.pp_path[a][b] = np.array([G["eid"][(x, y)] for x, y in zip(nd, nd[1:])], dtype=int)
                self.t_pp[a, b] = dist[q]
        self.routes = [(k,) for k in range(M)] + [(a, b) for a in range(M) for b in range(M) if a != b]
        self.rid = {r: i for i, r in enumerate(self.routes)}
        self.opts, self.cost = [], {}
        for v in range(n):
            singles = [k for k in range(M) if scn.cost[v, k] < R]
            o = [OPT_OUT] + [self.rid[(k,)] for k in singles]
            for k in singles:
                self.cost[(v, self.rid[(k,)])] = float(scn.cost[v, k])
            if max_stops >= 2:
                cand = sorted(singles, key=lambda k: scn.cost[v, k])[:K]
                for a in cand:
                    for b in cand:
                        if a == b:
                            continue
                        c = scn.beta * max(scn.base[v, a] + self.t_pp[a, b] + scn.t_pd[v, b] - scn.t_od[v], 0.0)
                        if c < 2 * R:
                            r = self.rid[(a, b)]; o.append(r); self.cost[(v, r)] = float(c)
            self.opts.append(o)
        self._true = {}

    def edges(self, v, r):
        rt = self.routes[r]
        e1 = list(self.scn.path[v][rt[0]])
        if len(rt) == 1:
            return e1, [len(e1)]
        e2 = list(self.pp_path[rt[0]][rt[1]])
        return e1 + e2, [len(e1), len(e1) + len(e2)]

    def mean_arrival(self, v, r):
        rt = self.routes[r]
        a1 = self.scn.arr_mean[v, rt[0]]
        return [a1] if len(rt) == 1 else [a1, a1 + self.t_pp[rt[0], rt[1]]]

    def true_arrivals(self, v, r):
        """Realized arrival at each stop in the scenario's true world (edge-level congestion, as LondonScenario);
        edges outside the scenario's precomputed set get deterministic extra processes and shocks."""
        key = (v, r)
        if key in self._true:
            return self._true[key]
        scn = self.scn
        edges, cps = self.edges(v, r)
        t, out, B = scn.t_dep[v], [], scn.Z_true.shape[1]
        if cps[0] == 0:
            out.append(t)
        for l, e in enumerate(edges):
            b = min(int((t - scn.t_origin) // scn.dt), B - 1)
            if e in scn.e_pos:
                j = scn.e_pos[e]; z, ep = scn.Z_true[j, b], scn.eps_true[v, j]
            else:
                z = _extra_field(scn, e)[b]
                ep = float(np.random.default_rng([scn.seed, 7, v, int(e)]).standard_normal())
            t += math.exp(scn.mu[e] + scn.sg[e] * (math.sqrt(scn.rho) * z + math.sqrt(1 - scn.rho) * ep))
            if l + 1 in cps:
                out.append(t)
        self._true[key] = out
        return out


_EXTRA = {}


def _extra_field(scn, e):
    key = (scn.seed, int(e), scn.Z_true.shape[1])
    if key not in _EXTRA:
        rng = np.random.default_rng([scn.seed, 11, int(e)])
        phi = math.exp(-scn.dt / scn.ell); q = math.sqrt(1 - phi * phi)
        xi = rng.standard_normal(scn.Z_true.shape[1]); xi[0] /= q
        _EXTRA[key] = lfilter([q], [1.0, -phi], xi)
    return _EXTRA[key]


ESTS = ("dist", "ordermean", "fixorder", "mean")


def window_draws(ml, players, t0, pending, S, rng, R):
    """own[(i, r)] for every option of every player, and pending arrays per PsI (conditioned on not yet arrived)."""
    scn = ml.scn
    rows, keys = [], []
    for i, v in enumerate(players):
        for r in ml.opts[v]:
            if r == OPT_OUT:
                continue
            e, c = ml.edges(v, r)
            rows.append((v, scn.t_dep[v], e, 1.0, c)); keys.append((i, r))
    prow0 = len(rows)
    for (v, r, stops) in pending:                       # stops: indices of stops not yet realized
        e, c = ml.edges(v, r)
        rows.append((v, scn.t_dep[v], e, 1.0, c))
    smp = RouteSampler(scn, rows, t0, S, rng)
    out = smp.sim()
    own = {keys[a]: out[a] for a in range(prow0)}
    P = [[] for _ in range(scn.M)]
    for q, (v, r, stops) in enumerate(pending):
        a = prow0 + q
        x = out[a]
        for _ in range(20):                            # condition on "not yet arrived" at each pending stop
            bad = np.zeros(S, bool)
            for s in stops:
                bad |= x[s] <= t0
            if not bad.any():
                break
            y = smp.sim([a], fresh=True)[0]
            x = [np.where(bad, y[s], x[s]) for s in range(len(x))]
        for s in stops:
            P[ml.routes[r][s]].append(np.maximum(x[s], t0 + 1e-3))
    P = [np.array(p) if p else np.zeros((0, S)) for p in P]
    return own, P


def run_london(ml: MultiLondon, est: str, R: float = 400.0, S: int = 256, seed: int = 0, regret=None, ests=ESTS):
    scn = ml.scn
    rng = np.random.default_rng(seed)
    done = [[t] for t in scn.init_last]
    pending, commits = [], {}            # pending: list of [v, r, {stop: true arrival}]
    reg = {e: dict(win=[], cut=[], win400=[], rfinal=[]) for e in ests} if regret else None
    st = dict(breaks=0, games=0, pairs=0, singles=0)
    wR = {}
    t0, idx, widx = 0.0, 0, 0
    while idx < scn.n:
        t1 = t0 + scn.window
        players = []
        while idx < scn.n and scn.t_dep[idx] < t1:
            players.append(idx); idx += 1
        keep = []
        for item in pending:
            v, r, arr = item
            for s in list(arr):
                if arr[s] <= t0:
                    bisect.insort(done[ml.routes[r][s]], arr[s]); del arr[s]
            if arr:
                keep.append(item)
        pending = keep
        if players:
            last = np.array([d[-1] for d in done])
            pend = [(v, r, sorted(arr)) for v, r, arr in pending]
            opts = [ml.opts[v] for v in players]
            cost = {(i, r): ml.cost[(v, r)] for i, v in enumerate(players) for r in ml.opts[v] if r != OPT_OUT}
            own, P = window_draws(ml, players, t0, pend, S, rng, R)
            games = {e: make_multi(ml.routes, opts, own, P, last, scn.tau, cost, R, _gr.base(e)) for e in set(ests) | {est}}
            g = games[est]
            res = _gr.solve(est, games, random.Random(seed * 31 + widx))
            assign = res.assign
            st["games"] += 1; st["breaks"] += res.breaks
            st["nonconv"] = st.get("nonconv", 0) + int(not (res.converged and g.is_nash(res.assign)))
            if regret:
                os_, Ps = window_draws(ml, players, t0, pend, regret[0], np.random.default_rng(1_000_003 * (seed + 1) + widx), R)
                oe, Pe = window_draws(ml, players, t0, pend, regret[1], np.random.default_rng(2_000_003 * (seed + 1) + widx), R)
                for e in ests:
                    if e == est:
                        prof, nb, Re = assign, res.breaks, g.R.copy()
                    else:
                        r2 = _gr.solve(e, games, random.Random(seed * 31 + widx))
                        prof, nb, Re = r2.assign, r2.breaks, games[e].R.copy()
                    gs = MultiGame(ml.routes, opts, os_, Ps, last, scn.tau, cost, Re)
                    ge = MultiGame(ml.routes, opts, oe, Pe, last, scn.tau, cost, Re)
                    reg[e]["win"].append(max(profile_regret(gs, ge, prof)) / R)
                    reg[e]["cut"].append(int(nb > 0))
                    reg[e]["rfinal"].append(float(np.min(Re)) / R)
                    if nb > 0:   # regret of the same profile under the announced reward R (no cut)
                        R0 = np.full(scn.M, R)
                        gs0 = MultiGame(ml.routes, opts, os_, Ps, last, scn.tau, cost, R0)
                        ge0 = MultiGame(ml.routes, opts, oe, Pe, last, scn.tau, cost, R0)
                        reg[e]["win400"].append(max(profile_regret(gs0, ge0, prof)) / R)
                    else:
                        reg[e]["win400"].append(reg[e]["win"][-1])
            for i, r in enumerate(assign):
                if r == OPT_OUT:
                    continue
                v = players[i]
                ta = ml.true_arrivals(v, r)
                pending.append([v, r, {s: ta[s] for s in range(len(ta))}])
                commits[v] = (r, ta, ml.cost[(v, r)])
                wR[v] = float(np.min(g.R))
                st["pairs" if len(ml.routes[r]) == 2 else "singles"] += 1
        t0, widx = t1, widx + 1
    for v, r, arr in pending:
        for s, a in arr.items():
            bisect.insort(done[ml.routes[r][s]], a)
    pay = {}
    for k in range(scn.M):
        s = done[k]
        for p_, a in zip(s[:-1], s[1:]):
            pay[(k, a)] = R * (1 - math.exp(-(a - p_) / scn.tau[k]))
    payout_nominal = sum(pay[(k, a)] for v, (r, ta, c) in commits.items() for k, a in zip(ml.routes[r], ta))
    payout = sum(pay[(k, a)] * wR[v] / R for v, (r, ta, c) in commits.items() for k, a in zip(ml.routes[r], ta))
    cost = sum(c for (r, ta, c) in commits.values())
    viol, worst = 0.0, []
    for k in range(scn.M):
        pts = [t for t in done[k] if t <= scn.T] + [scn.T]
        gaps = []
        for a, b in zip(pts[:-1], pts[1:]):
            gaps.append(b - a); viol += max(0.0, b - max(a + scn.deadline[k], 0.0))
        worst.append(max(gaps))
    inv = tot = 0
    by_k = {}
    for v, (r, ta, c) in commits.items():
        for s, (k, a) in enumerate(zip(ml.routes[r], ta)):
            by_k.setdefault(k, []).append((ml.mean_arrival(v, r)[s], a))
    for lst in by_k.values():
        lst.sort()
        for (m1, a1), (m2, a2) in zip(lst[:-1], lst[1:]):
            tot += 1; inv += int(a2 < a1)
    out = dict(nonconv=st.get("nonconv", 0), payout_nominal=payout_nominal, net_nominal=(payout_nominal - cost) / scn.n, est=est, max_stops=ml.max_stops, deadline_violation=viol / (scn.M * scn.T), worst_gap=float(np.mean(worst)),
               detours=len(commits), pair_routes=st["pairs"], samples=sum(len(ml.routes[r]) for r, _, _ in commits.values()),
               vehicles=scn.n, payout=payout, net_per_vehicle=(payout - cost) / scn.n, inversion_rate=inv / max(tot, 1),
               breaks=st["breaks"], games=st["games"], psi_counts=[len(done[k]) - 1 for k in range(scn.M)])
    return out, reg
