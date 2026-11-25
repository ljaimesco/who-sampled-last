"""Closed-loop multi-stop missions with SUMO as ground truth (generalizes sumo_mission.run).

Each vehicle may sample up to two PsIs in order (o -> k1 -> k2 -> d). SUMO timestamps each stop when the vehicle
moves past the last edge into that PsI. At every window, committed vehicles still owing a sample are located in
SUMO and their remaining route is sampled from their current position to each pending stop. Decisions use the
learned forecast (sumo_mission.FORECAST) with the same calibration constants and online rescaling as the
single-stop SUMO experiment; all estimators face the same demand, background traffic and speed field.
"""
from __future__ import annotations

import bisect
import math
import os
import random

import numpy as np

from . import london_mission as lm
from . import multistop as ms
from . import sumo_mission as smm
from . import sumo_world as sw
from .model import OPT_OUT
from .policies import spa
from . import fresh_greedy as _gr
from .stochastic import profile_regret


class Probe:
    __slots__ = ("v", "r", "route", "stops_idx", "depart", "arr", "inserted", "actual_depart")

    def __init__(self, v, r, route, stops_idx, depart):
        self.v, self.r, self.route, self.stops_idx, self.depart = v, r, route, stops_idx, depart
        self.arr = [None] * len(stops_idx)
        self.inserted, self.actual_depart = False, None

    def pending(self):
        return [s for s, a in enumerate(self.arr) if a is None]


def run(seed: int, est: str, R: float = 400.0, S: int = 1024, regret=None, ests=ms.ESTS, max_stops: int = 2,
        horizon: float = 1800.0, scale_prior=None, sigma_gain=None):
    import traci
    T0 = smm.T0
    rho = smm.forecast(smm.FORECAST)
    sp = smm.SCALE_PRIOR if scale_prior is None else scale_prior
    sgain = smm.SIGMA_GAIN if sigma_gain is None else sigma_gain
    G = lm.graph(); W = sw.world()
    scn = lm.LondonScenario(5000 + seed, rho, horizon=horizon)
    ml = ms.MultiLondon(scn, R, max_stops=max_stops)
    label = f"ms{seed}_{est}_{os.getpid()}"
    errlog = os.path.join(sw.WORK, f"err_{label}.log")
    traci.start(sw.sumo_cmd(300 + seed, ["--error-log", errlog]), label=label)
    tr = traci.getConnection(label)
    field = sw.SpeedField(300 + seed)
    rng = np.random.default_rng(seed)
    done = [[t] for t in scn.init_last]
    probes, awaiting, zero = {}, {}, []
    reg = {e: dict(win=[], cut=[], win400=[], rfinal=[]) for e in ests} if regret else None
    wR = {}
    fc = dict(err=[], cover=[], scale=[])
    st = dict(breaks=0, games=0, teleported=0, pairs=0)
    sim_t = 0.0
    RI = traci.constants.VAR_ROUTE_INDEX

    def stamp(p, s, t):
        p.arr[s] = t
        bisect.insort(done[ml.routes[p.r][s]], t)

    def advance(until):
        nonlocal sim_t
        while sim_t < until:
            if sim_t % sw.DT_FIELD == 0:
                field.apply(tr, sim_t)
            tr.simulationStep()
            sim_t = tr.simulation.getTime()
            for vid in tr.simulation.getDepartedIDList():
                if vid in awaiting:
                    p = awaiting[vid]; p.inserted = True; p.actual_depart = sim_t - T0
                    for s in p.pending():
                        if p.stops_idx[s] < 0:                 # origin is the PsI
                            stamp(p, s, sim_t - T0)
                    tr.vehicle.subscribe(vid, [RI])
            res = tr.vehicle.getAllSubscriptionResults()
            for vid in list(awaiting):
                p = awaiting[vid]
                if not p.inserted:
                    continue
                r = res.get(vid)
                for s in p.pending():
                    if r is None or r[RI] > p.stops_idx[s]:
                        stamp(p, s, sim_t - T0)
                if not p.pending():
                    del awaiting[vid]
            st["teleported"] += tr.simulation.getStartingTeleportNumber()

    def row_from_position(p, t0):
        """Remaining edges from the vehicle's current SUMO position to its last pending stop, with checkpoints."""
        pend = p.pending()
        vid = f"p{p.v}"
        if p.inserted and vid in tr.vehicle.getIDList():
            ri = tr.vehicle.getRouteIndex(vid)
            road = tr.vehicle.getRoadID(vid)
            if road.startswith(":"):
                start, f0 = ri + 1, 1.0
            else:
                L = tr.lane.getLength(tr.vehicle.getLaneID(vid))
                start, f0 = ri, max(0.0, 1.0 - tr.vehicle.getLanePosition(vid) / L)
            t_start = t0
        else:
            start, f0, t_start = 0, 1.0, max(t0, p.depart)
        last_idx = p.stops_idx[pend[-1]]
        edges = [W["idx"][x] for x in p.route[start:last_idx + 1]]
        cps = [max(0, p.stops_idx[s] + 1 - start) for s in pend]
        return (p.v, t_start, edges, f0, cps), pend

    base_tt = {}
    t0, idx, widx = 0.0, 0, 0
    while idx < scn.n:
        advance(T0 + t0)
        t1 = t0 + scn.window
        players = []
        while idx < scn.n and scn.t_dep[idx] < t1:
            players.append(idx); idx += 1
        if players:
            last = np.array([d[-1] for d in done])
            comp = [(p.arr[s] - p.actual_depart, base_tt[(p.v, p.r, s)]) for p in probes.values() if p.actual_depart is not None
                    for s in range(len(p.arr)) if p.arr[s] is not None and base_tt.get((p.v, p.r, s), 0) > 0]
            mtt = scn.arr_mean - scn.t_dep[:, None]
            c0 = 10 * float(np.median(mtt[mtt > 0]))
            scale = (sum(a for a, _ in comp) + sp * c0) / (sum(b for _, b in comp) + c0)
            fc["scale"].append(scale)
            prows, pinfo = [], []
            for vid, p in awaiting.items():
                row, pend = row_from_position(p, t0)
                prows.append(row); pinfo.append((p, pend))
            opts = [ml.opts[v] for v in players]
            cost = {(i, r): ml.cost[(v, r)] for i, v in enumerate(players) for r in ml.opts[v] if r != OPT_OUT}

            def draws(SS, g_rng):
                rows, keys = [], []
                for i, v in enumerate(players):
                    for r in ml.opts[v]:
                        if r != OPT_OUT:
                            e, c = ml.edges(v, r); rows.append((v, scn.t_dep[v], e, 1.0, c)); keys.append((i, r))
                smp = ms.RouteSampler(scn, rows + prows, t0, SS, g_rng, scale=scale, sigma_gain=sgain)
                out = smp.sim()
                own = {keys[a]: out[a] for a in range(len(rows))}
                P = [[] for _ in range(scn.M)]
                for q, (p, pend) in enumerate(pinfo):
                    for j, s in enumerate(pend):
                        P[ml.routes[p.r][s]].append(np.maximum(out[len(rows) + q][j], t0 + 1e-3))
                P = [np.array(x) if x else np.zeros((0, SS)) for x in P]
                return own, P

            own, P = draws(S, rng)
            games = {e: ms.make_multi(ml.routes, opts, own, P, last, scn.tau, cost, R, _gr.base(e)) for e in set(ests) | {est}}
            g = games[est]
            res = _gr.solve(est, games, random.Random(seed * 31 + widx))
            assign = res.assign
            st["games"] += 1; st["breaks"] += res.breaks
            st["nonconv"] = st.get("nonconv", 0) + int(not (res.converged and g.is_nash(res.assign)))
            if regret:
                os_, Ps = draws(regret[0], np.random.default_rng(1_000_003 * (seed + 1) + widx))
                oe, Pe = draws(regret[1], np.random.default_rng(2_000_003 * (seed + 1) + widx))
                for e in ests:
                    if e == est:
                        prof, nb, Re = assign, res.breaks, g.R.copy()
                    else:
                        r2 = _gr.solve(e, games, random.Random(seed * 31 + widx))
                        prof, nb, Re = r2.assign, r2.breaks, games[e].R.copy()
                    gs = ms.MultiGame(ml.routes, opts, os_, Ps, last, scn.tau, cost, Re)
                    ge = ms.MultiGame(ml.routes, opts, oe, Pe, last, scn.tau, cost, Re)
                    reg[e]["win"].append(max(profile_regret(gs, ge, prof)) / R)
                    reg[e]["cut"].append(int(nb > 0))
                    reg[e]["rfinal"].append(float(np.min(Re)) / R)
                    if nb > 0:
                        R0 = np.full(scn.M, R)
                        reg[e]["win400"].append(max(profile_regret(ms.MultiGame(ml.routes, opts, os_, Ps, last, scn.tau, cost, R0), ms.MultiGame(ml.routes, opts, oe, Pe, last, scn.tau, cost, R0), prof)) / R)
                    else:
                        reg[e]["win400"].append(reg[e]["win"][-1])
            for i, r in enumerate(assign):
                v = players[i]
                o, d = scn.orig[v], scn.dest[v]
                if r == OPT_OUT:
                    nodes, stops_idx = smm.node_path(G["g"], o, d), []
                else:
                    seq = [o] + [scn.psi[k] for k in ml.routes[r]] + [d]
                    nodes, stops_idx = [o], []
                    for j, (a, b) in enumerate(zip(seq, seq[1:])):
                        nodes += smm.node_path(G["g"], a, b)[1:]
                        if j < len(ml.routes[r]):              # legs 0..n-1 end at the stops, the last at d
                            stops_idx.append(len(nodes) - 2)   # index of the last edge into the stop (-1: at origin)
                edges = [W["sid"][e] for e in smm.edges_of(G, nodes)]
                if not edges:                                  # origin = PsI = destination: sampled at departure
                    for s, k in enumerate(ml.routes[r] if r != OPT_OUT else ()):
                        bisect.insort(done[k], scn.t_dep[v])
                        zero.append((v, r, k, float(scn.t_dep[v]), float(np.min(g.R))))
                    continue
                vid = f"p{v}"
                tr.route.add(f"r{v}", edges)
                tr.vehicle.add(vid, f"r{v}", depart=str(T0 + scn.t_dep[v]))
                if r != OPT_OUT:
                    p = Probe(v, r, edges, stops_idx, scn.t_dep[v])
                    probes[v] = p; awaiting[vid] = p
                    wR[v] = float(np.min(g.R))
                    st["pairs"] += int(len(ml.routes[r]) == 2)
                    for s in range(len(ml.routes[r])):
                        base_tt[(v, r, s)] = ml.mean_arrival(v, r)[s] - scn.t_dep[v]
                        a = own[(i, r)][s]
                        fc.setdefault("pred", []).append((v, s, float(a.mean()), float(np.percentile(a, 10)), float(np.percentile(a, 90))))
        t0, widx = t1, widx + 1
    advance(min(sw.END - 1, T0 + horizon + 900.0))
    never = sum(len(p.pending()) for p in awaiting.values())
    tr.close()
    ties = sum(len(d) - len(set(d)) for d in done)   # samples sharing a timestamp with another at the same PsI
    pay = {}
    for k in range(scn.M):   # gap to the latest sample STRICTLY before a (simultaneous arrivals share the same predecessor)
        s = done[k]
        for a in set(s[1:]):
            j = bisect.bisect_left(s, a)
            if j > 0:
                pay[(k, a)] = R * (1 - math.exp(-(a - s[j - 1]) / scn.tau[k]))
    payout = payout_nominal = cost_total = 0.0
    for v, p in probes.items():
        for s, k in enumerate(ml.routes[p.r]):
            if p.arr[s] is not None:
                payout_nominal += pay.get((k, p.arr[s]), 0.0)
                payout += pay.get((k, p.arr[s]), 0.0) * wR[v] / R      # paid at the window's announced reward
        cost_total += ml.cost[(v, p.r)]
    for v, r, k, t, rw in zero:                                # zero-edge visits are paid like any other sample
        payout_nominal += pay.get((k, t), 0.0); payout += pay.get((k, t), 0.0) * rw / R
        cost_total += ml.cost[(v, r)]
    for (v, s, m, lo, hi) in fc.get("pred", []):
        a = probes[v].arr[s]
        if a is not None:
            fc["err"].append(a - m); fc["cover"].append(int(lo <= a <= hi))
    viol, worst = 0.0, []
    for k in range(scn.M):
        pts = [t for t in done[k] if t <= scn.T] + [scn.T]
        gaps = []
        for a, b in zip(pts[:-1], pts[1:]):
            gaps.append(b - a); viol += max(0.0, b - max(a + scn.deadline[k], 0.0))
        worst.append(max(gaps))
    inv = tot = 0
    by_k = {}
    for v, p in probes.items():
        for s, k in enumerate(ml.routes[p.r]):
            if p.arr[s] is not None:
                by_k.setdefault(k, []).append((ml.mean_arrival(v, p.r)[s], p.arr[s]))
    for lst in by_k.values():
        lst.sort()
        for (m1, a1), (m2, a2) in zip(lst[:-1], lst[1:]):
            tot += 1; inv += int(a2 < a1)
    err = np.array(fc["err"]) if fc["err"] else np.zeros(1)
    out = dict(nonconv=st.get("nonconv", 0), ties=ties, payout_nominal=payout_nominal, net_nominal=(payout_nominal - cost_total) / scn.n, seed=seed, est=est, max_stops=max_stops, deadline_violation=viol / (scn.M * scn.T), worst_gap=float(np.mean(worst)),
               detours=len(probes) + len({(v, r) for v, r, *_ in zero}), zero_edge=len(zero), pair_routes=st["pairs"], samples=sum(len(ml.routes[p.r]) for p in probes.values()) + len(zero),
               vehicles=scn.n, payout=payout, net_per_vehicle=(payout - cost_total) / scn.n, inversion_rate=inv / max(tot, 1),
               breaks=st["breaks"], games=st["games"], teleports=st["teleported"], never_sampled=never,
               forecast_bias=float(err.mean()), forecast_mae=float(np.abs(err).mean()),
               forecast_cover80=float(np.mean(fc["cover"])) if fc["cover"] else float("nan"))
    return out, reg
