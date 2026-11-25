"""Closed-loop single-stop missions with SUMO as ground truth.

Every 60 s decision window:
  1. SUMO is advanced to the window start (speed field updated every 60 s; probe arrivals at their PsI are
     timestamped at 1 s resolution from the vehicles' route index);
  2. vehicles still en route to a PsI are located in SUMO (current edge and position), so their remaining
     travel is sampled from where they are, not from their departure;
  3. the window's players solve the game under the FORECAST model (per-edge lognormal learned from warm-up
     runs, edge-and-time congestion with the estimated shared fraction rho_hat), with estimator `est`;
  4. each player's chosen detour (origin -> PsI -> destination, or origin -> destination if opting out) is
     inserted into SUMO as a real vehicle departing at its scheduled time.
Payments, gaps and deadline violations are computed from the SUMO-realized sample times.
Scenario geometry, demand, deadlines and costs come from london_mission.LondonScenario under the forecast
model, so every estimator in a paired comparison faces the same demand, background traffic and speed field.
"""
from __future__ import annotations

import bisect
import math
import os
import random

import networkx as nx
import numpy as np
from scipy.signal import lfilter

from . import london_mission as lm
from . import sumo_world as sw
from .model import OPT_OUT
from .policies import spa
from . import fresh_greedy as _gr
from .stochastic import StochGame, profile_regret

T0 = 600.0                 # probe demand starts after background traffic has built up (simulation seconds)
CALIBRATE = True           # online rescaling of forecast travel times from completed probe trips
SCALE_PRIOR = float(os.environ.get("SPA_SCALE_PRIOR", "1.285"))   # prior travel-time scale (from calibration missions)
SIGMA_GAIN = float(os.environ.get("SPA_SIGMA_GAIN", "1.5"))     # forecast log-spread inflation (edge means held fixed)
FORECAST = os.path.join(sw.DATA, "sumo_forecast.npz")
_F = None


def forecast(path=None):
    """Load a forecast file into the London graph (every call when a path is given)."""
    global _F
    if path is not None:
        f = np.load(path); lm.set_edge_stats(f["mean"], f["std"]); _F = None
        return float(f["rho"])
    if _F is None:
        f = np.load(FORECAST)
        lm.set_edge_stats(f["mean"], f["std"])
        _F = float(f["rho"])
    return _F


def sample_rows(scn, rows, t0, S, rng, scale=1.0, sigma_gain=None):
    """rows: list of (vehicle, start_time, [edge indices], fraction of first edge still to travel).
    Joint samples of the time each row reaches the end of its edge list, under the forecast model:
    one edge-and-time congestion field per draw shared by all rows; a vehicle keeps its individual shock on an
    edge across its alternative routes. Returns array (len(rows), S)."""
    starts = [r[1] for r in rows]
    t_from = min(starts + [t0]) - scn.dt
    B = int((t0 + scn.window + 1500.0 - t_from) // scn.dt) + 1
    loc, keys = {}, {}
    L = max([len(r[2]) for r in rows] + [1])
    E_idx = -np.ones((len(rows), L), dtype=int); U_idx = np.zeros((len(rows), L), dtype=int)
    frac = np.ones((len(rows), L), dtype=np.float32)
    for i, (v, st, edges, f0) in enumerate(rows):
        for l, e in enumerate(edges):
            E_idx[i, l] = loc.setdefault(e, len(loc))
            U_idx[i, l] = keys.setdefault((v, e), len(keys))
        if edges:
            frac[i, 0] = f0
    g_ids = np.array(list(loc.keys()), dtype=int)
    phi = math.exp(-scn.dt / scn.ell); q = math.sqrt(1 - phi * phi)
    xi = rng.standard_normal((B, max(len(loc), 1), S), dtype=np.float32); xi[0] /= q
    Z = lfilter(np.float32([q]), np.float32([1.0, -phi]), xi, axis=0)
    eps = rng.standard_normal((max(len(keys), 1), S), dtype=np.float32)
    sr, si = math.sqrt(scn.rho), math.sqrt(1 - scn.rho)
    sg_all = scn.sg * (SIGMA_GAIN if sigma_gain is None else sigma_gain)
    mu_all = scn.mu + (scn.sg ** 2 - sg_all ** 2) / 2
    ar = np.arange(S)[None, :]
    t = np.repeat(np.array(starts, dtype=float)[:, None], S, axis=1)
    for l in range(L):
        act = np.nonzero(E_idx[:, l] >= 0)[0]
        if len(act) == 0:
            break
        el = E_idx[act, l]
        b = np.clip(((t[act] - t_from) // scn.dt).astype(int), 0, B - 1)
        z = Z[b, el[:, None], ar]
        ge = g_ids[el]
        t[act] += scale * frac[act, l][:, None] * np.exp(mu_all[ge][:, None] + sg_all[ge][:, None] * (sr * z + si * eps[U_idx[act, l]]))
    return t


class Probe:
    __slots__ = ("v", "k", "route", "i_psi", "depart", "arr", "inserted", "actual_depart")

    def __init__(self, v, k, route, i_psi, depart):
        self.v, self.k, self.route, self.i_psi, self.depart = v, k, route, i_psi, depart
        self.arr, self.inserted, self.actual_depart = None, False, None


def node_path(g, a, b):
    return nx.shortest_path(g, a, b, weight="mean") if a != b else [a]


def edges_of(G, nodes):
    return [G["eid"][(x, y)] for x, y in zip(nodes, nodes[1:])]


def run(seed: int, est: str, R: float = 400.0, S: int = 256, regret=None, ests=lm.ESTS, horizon: float = 1800.0, sumo_extra=(),
        forecast_npz=None, world_seed=None, scale_prior=None, sigma_gain=None):
    """forecast_npz: forecast file (default: the global warm-up forecast); world_seed: seed of the background traffic,
    speed field and SUMO RNG (default 300 + seed); scale_prior / sigma_gain override the module constants."""
    import traci
    rho = forecast(forecast_npz) if forecast_npz else forecast(FORECAST)
    wseed = 300 + seed if world_seed is None else world_seed
    sp = SCALE_PRIOR if scale_prior is None else scale_prior
    G = lm.graph(); W = sw.world()
    scn = lm.LondonScenario(5000 + seed, rho, horizon=horizon)
    label = f"m{seed}_{est}_{os.getpid()}"
    errlog = os.path.join(sw.WORK, f"err_{label}.log")
    traci.start(sw.sumo_cmd(wseed, ["--error-log", errlog, *sumo_extra]), label=label)
    tr = traci.getConnection(label)
    field = sw.SpeedField(wseed)
    rng = np.random.default_rng(seed)
    done = [[t] for t in scn.init_last]
    probes, awaiting, zero = {}, {}, []
    wR = {}
    reg = {e: dict(win=[], cut=[], win400=[], rfinal=[]) for e in ests} if regret else None
    fc = dict(err=[], cover=[], scale=[])
    base_tt = scn.arr_mean - scn.t_dep[:, None]          # forecast mean travel time to each PsI
    st = dict(breaks=0, games=0, teleported=0, never=0)
    sim_t = 0.0

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
                    if p.i_psi < 0:                                    # origin is the PsI: sample at departure
                        p.arr = sim_t - T0; bisect.insort(done[p.k], p.arr); del awaiting[vid]
                    else:
                        tr.vehicle.subscribe(vid, [traci.constants.VAR_ROUTE_INDEX])
            res = tr.vehicle.getAllSubscriptionResults()
            for vid in list(awaiting):
                p = awaiting[vid]
                if not p.inserted:
                    continue
                r = res.get(vid)
                if r is not None and r[traci.constants.VAR_ROUTE_INDEX] > p.i_psi:
                    p.arr = sim_t - T0
                elif r is None:                                        # arrived at destination (PsI on last edge)
                    p.arr = sim_t - T0
                if p.arr is not None:
                    bisect.insort(done[p.k], p.arr); del awaiting[vid]
            st["teleported"] += tr.simulation.getStartingTeleportNumber()

    t0, idx, widx = 0.0, 0, 0
    while idx < scn.n:
        advance(T0 + t0)
        t1 = t0 + scn.window
        players = []
        while idx < scn.n and scn.t_dep[idx] < t1:
            players.append(idx); idx += 1
        if players:
            last = np.array([d[-1] for d in done])
            # online calibration: ratio of realized to forecast travel over completed probe trips, shrunk to 1
            comp = [(p.arr - p.actual_depart, base_tt[p.v, p.k]) for p in probes.values()
                    if p.arr is not None and p.actual_depart is not None and base_tt[p.v, p.k] > 0]
            c0 = 10 * float(np.median(base_tt[base_tt > 0]))
            scale = (sum(a for a, _ in comp) + sp * c0) / (sum(b for _, b in comp) + c0) if CALIBRATE else sp
            fc["scale"].append(scale)
            # en-route probes: remaining path from their current SUMO position
            pend_rows, pend_k = [], []
            for vid, p in awaiting.items():
                if p.inserted and vid in tr.vehicle.getIDList():
                    ri = tr.vehicle.getRouteIndex(vid)
                    road = tr.vehicle.getRoadID(vid)
                    if road.startswith(":"):
                        rem, f0 = p.route[ri + 1:p.i_psi + 1], 1.0
                    else:
                        L = tr.lane.getLength(tr.vehicle.getLaneID(vid))
                        rem, f0 = p.route[ri:p.i_psi + 1], max(0.0, 1.0 - tr.vehicle.getLanePosition(vid) / L)
                    pend_rows.append((p.v, t0, [W["idx"][s] for s in rem], f0))
                else:
                    pend_rows.append((p.v, max(t0, p.depart), [W["idx"][s] for s in p.route[:p.i_psi + 1]], 1.0))
                pend_k.append(p.k)

            def draws(SS, g_rng):
                prow = [(v, scn.t_dep[v], list(scn.path[v][k]), 1.0) for v in players for k in range(scn.M) if scn.cost[v, k] < R]
                T = sample_rows(scn, prow + pend_rows, t0, SS, g_rng, scale, sigma_gain)
                A = np.repeat((scn.t_dep[players][:, None] + scale * base_tt[players])[:, :, None], SS, axis=2).astype(float)
                pos = {v: a for a, v in enumerate(players)}
                for r, (v, _, _, _) in enumerate(prow):
                    k = [kk for kk in range(scn.M) if scn.cost[v, kk] < R][sum(1 for x in prow[:r] if x[0] == v)]
                    A[pos[v], k] = T[r]
                P = [[] for _ in range(scn.M)]
                for r, k in enumerate(pend_k):
                    P[k].append(np.maximum(T[len(prow) + r], t0 + 1e-3))
                P = [np.array(p) if p else np.zeros((0, SS)) for p in P]
                return A, P

            A, P = draws(S, rng)
            games = {e: lm.make_game(scn, players, A, P, last, R, _gr.base(e)) for e in set(ests) | {est}}
            g = games[est]
            r = _gr.solve(est, games, random.Random(seed * 31 + widx))
            assign = r.assign
            st["games"] += 1; st["breaks"] += r.breaks
            st["nonconv"] = st.get("nonconv", 0) + int(not (r.converged and g.is_nash(r.assign)))
            if regret:
                As, Ps = draws(regret[0], np.random.default_rng(1_000_003 * (seed + 1) + widx))
                Ae, Pe = draws(regret[1], np.random.default_rng(2_000_003 * (seed + 1) + widx))
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
                    reg[e]["rfinal"].append(float(np.min(Re)) / R)
                    if nb > 0:
                        R0 = np.full(scn.M, R)
                        reg[e]["win400"].append(max(profile_regret(StochGame(As, Ps, last, scn.tau, scn.cost[players], R0), StochGame(Ae, Pe, last, scn.tau, scn.cost[players], R0), prof)) / R)
                    else:
                        reg[e]["win400"].append(reg[e]["win"][-1])
            for i, k in enumerate(assign):
                v = players[i]
                o, d = scn.orig[v], scn.dest[v]
                if k == OPT_OUT:
                    nodes = node_path(G["g"], o, d); i_psi = None
                else:
                    a_nodes = node_path(G["g"], o, scn.psi[k]); b_nodes = node_path(G["g"], scn.psi[k], d)
                    nodes = a_nodes + b_nodes[1:]; i_psi = len(a_nodes) - 2
                edges = [W["sid"][e] for e in edges_of(G, nodes)]
                if not edges:                                          # origin = PsI = destination: sampled at departure
                    if k != OPT_OUT:
                        bisect.insort(done[k], scn.t_dep[v])
                        zero.append((v, k, float(scn.t_dep[v]), float(np.min(g.R))))
                    continue
                vid = f"p{v}"
                tr.route.add(f"r{v}", edges)
                tr.vehicle.add(vid, f"r{v}", depart=str(T0 + scn.t_dep[v]))
                if k != OPT_OUT:
                    p = Probe(v, k, edges, i_psi, scn.t_dep[v])
                    probes[v] = p; awaiting[vid] = p
                    wR[v] = float(np.min(g.R))
                    a = A[i, k]
                    fc["pred"] = fc.get("pred", []) + [(v, float(a.mean()), float(np.percentile(a, 10)), float(np.percentile(a, 90)), float(scn.cost[v, k]))]
        t0, widx = t1, widx + 1
    advance(min(sw.END - 1, T0 + horizon + 900.0))
    st["never"] = len(awaiting)
    tr.close()
    # realized outcomes
    ties = sum(len(d) - len(set(d)) for d in done)   # samples sharing a timestamp with another at the same PsI
    pay = {}
    for k in range(scn.M):   # gap to the latest sample STRICTLY before a (simultaneous arrivals share the same predecessor)
        s = done[k]
        for a in set(s[1:]):
            j = bisect.bisect_left(s, a)
            if j > 0:
                pay[(k, a)] = R * (1 - math.exp(-(a - s[j - 1]) / scn.tau[k]))
    payout = payout_nominal = cost = 0.0
    for v, p in probes.items():
        if p.arr is not None:
            payout_nominal += pay.get((p.k, p.arr), 0.0)
            payout += pay.get((p.k, p.arr), 0.0) * wR[v] / R      # paid at the window's announced reward
        cost += float(scn.cost[v, p.k])
    for v, k, t, rw in zero:                                   # zero-edge visits are paid like any other sample
        payout_nominal += pay.get((k, t), 0.0); payout += pay.get((k, t), 0.0) * rw / R
        cost += float(scn.cost[v, k])
    for (v, m, lo, hi, c) in fc.get("pred", []):
        p = probes[v]
        if p.arr is not None:
            fc["err"].append(p.arr - m); fc["cover"].append(int(lo <= p.arr <= hi))
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
        if p.arr is not None:
            by_k.setdefault(p.k, []).append((float(scn.arr_mean[v, p.k]), p.arr))
    for lst in by_k.values():
        lst.sort()
        for (m1, a1), (m2, a2) in zip(lst[:-1], lst[1:]):
            tot += 1; inv += int(a2 < a1)
    err = np.array(fc["err"]) if fc["err"] else np.zeros(1)
    out = dict(nonconv=st.get("nonconv", 0), ties=ties, payout_nominal=payout_nominal, net_nominal=(payout_nominal - cost) / scn.n, seed=seed, est=est, deadline_violation=viol / (scn.M * scn.T), worst_gap=float(np.mean(worst)),
               detours=len(probes) + len(zero), zero_edge=len(zero), vehicles=scn.n, payout=payout, net_per_vehicle=(payout - cost) / scn.n,
               inversion_rate=inv / max(tot, 1), breaks=st["breaks"], games=st["games"], teleports=st["teleported"],
               never_arrived=st["never"], forecast_bias=float(err.mean()), forecast_mae=float(np.abs(err).mean()),
               forecast_cover80=float(np.mean(fc["cover"])) if fc["cover"] else float("nan"),
               final_scale=float(fc["scale"][-1]) if fc["scale"] else 1.0)
    out["_diag"] = [(v, p.depart, p.actual_depart, p.arr, float(scn.arr_mean[v, p.k]), len(p.route), p.i_psi) for v, p in probes.items()]
    return out, reg
