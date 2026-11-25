"""SUMO world for the London grid (ANTE-Mobility london-seg4.100.net.xml).

Ground truth = SUMO microsimulation:
  * background traffic: randomTrips (validated routes), one demand file per seed;
  * a time-varying speed field: every 60 s each edge's speed limit is set to L_e / T_e(t) with
        T_e(t) = exp(m_e + sqrt(RS) * sigma_e * Z_e(t)),  m_e = log(mean_e) - RS * sigma_e^2 / 2,
    where (mean_e, sigma_e) come from the edge's Uber statistics (lognormal by moments) and Z_e is a
    stationary AR(1) standard-normal process (correlation exp(-dt/ell)), so the shared part of each edge's
    travel time follows the Uber data; individual variation and delays come from SUMO itself
    (driver speed factors, car following, junction yielding, queues);
  * speeds are clipped to [2, 25] m/s.
The forecast model used by the decision makers is NOT this field: it is learned from separate warm-up runs.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.signal import lfilter

import sumo as _sumo
import sumolib

from . import london_mission as lm

SUMO_HOME = _sumo.SUMO_HOME
os.environ.setdefault("SUMO_HOME", SUMO_HOME)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NET = os.path.join(DATA, "london.net.xml")
WORK = os.path.join(HERE, "..", "sumo_runs")
RS, ELL, DT_FIELD = 1.0, 300.0, 60.0
CENTER, FIELD_HALF = (2550.0, 2550.0), 1500.0   # speed field applied to edges whose midpoint is in this square
VMIN, VMAX = 1.0, 25.0
GAIN = float(os.environ.get('SPA_FIELD_GAIN', '1.0'))   # multiplies the field's log-sd (calibrated so SUMO edge cv matches Uber)
END = 3300.0                       # background demand and simulation end (s)

_W = None


def world():
    """Edge bookkeeping shared by every run: map the London graph's edges (lm order) to SUMO edge ids."""
    global _W
    if _W is None:
        G = lm.graph()
        net = sumolib.net.readNet(NET)
        by_nodes = {(e.getFromNode().getID(), e.getToNode().getID()): e for e in net.getEdges() if not e.isSpecial()}
        sid, length = [], []
        for (u, v) in G["edges"]:
            e = by_nodes.get((u, v))
            sid.append(e.getID() if e is not None else None)
            length.append(e.getLength() if e is not None else float("nan"))
        g = G["g"]
        umean = np.array([g.edges[e]["mean"] for e in G["edges"]])
        ustd = np.array([g.edges[e]["std"] for e in G["edges"]])
        _W = dict(sid=sid, length=np.array(length), umean=umean, ustd=ustd,
                  idx={s: i for i, s in enumerate(sid) if s is not None})
        os.makedirs(WORK, exist_ok=True)
    return _W


def background(seed: int, period: float = 0.8) -> str:
    """Validated random background trips over the whole grid (cached per seed)."""
    out = os.path.join(WORK, f"bg_{seed}.rou.xml")
    if not os.path.exists(out):
        trips = os.path.join(WORK, f"bg_{seed}.trips.xml")
        subprocess.run([sys.executable, os.path.join(SUMO_HOME, "tools", "randomTrips.py"), "-n", NET, "-o", trips,
                        "-r", out, "-b", "0", "-e", str(END), "-p", str(period), "--fringe-factor", "5",
                        "--min-distance", "1000", "--seed", str(seed), "--validate", "--prefix", "bg"],
                       check=True, capture_output=True)
    return out


class SpeedField:
    """Per-edge AR(1) congestion processes on a 60 s grid; speed limit L / T as in the module docstring."""

    def __init__(self, seed: int):
        W = world()
        rng = np.random.default_rng(10_000 + seed)
        xy = lm.graph()["xy"]
        ok = np.array([s is not None for s in W["sid"]]) & (np.abs(xy[:, 0] - CENTER[0]) <= FIELD_HALF) \
            & (np.abs(xy[:, 1] - CENTER[1]) <= FIELD_HALF)
        self.ids = [W["sid"][i] for i in np.nonzero(ok)[0]]
        m, sd, L = W["umean"][ok], W["ustd"][ok], W["length"][ok]
        s2 = np.log1p((sd / m) ** 2)
        self.scale = GAIN * np.sqrt(RS * s2)
        self.base = np.log(m) - self.scale ** 2 / 2
        self.L = L
        B = int(END // DT_FIELD) + 2
        phi = math.exp(-DT_FIELD / ELL); q = math.sqrt(1 - phi * phi)
        xi = rng.standard_normal((B, len(self.ids))); xi[0] /= q
        self.Z = lfilter([q], [1.0, -phi], xi, axis=0)

    def speeds(self, t: float) -> np.ndarray:
        b = min(int(t // DT_FIELD), len(self.Z) - 1)
        T = np.exp(self.base + self.scale * self.Z[b])
        return np.clip(self.L / T, VMIN, VMAX)

    def apply(self, traci, t: float):
        for sid, v in zip(self.ids, self.speeds(t)):
            traci.edge.setMaxSpeed(sid, float(v))


def sumo_cmd(seed: int, extra=()):
    return ["sumo", "-n", NET, "-r", background(seed), "--begin", "0", "--end", str(END), "--seed", str(seed),
            "--no-step-log", "true", "--no-warnings", "true", "--time-to-teleport", "300", "--step-length", "1",
            "--default.speeddev", "0.1", *extra]


def warmup(seed: int):
    """Run one warm-up simulation with the speed field; return per-edge lists of (entry time, traversal time)
    (interior edges of every vehicle route, entries between 600 s and END - 600 s)."""
    import traci
    out = os.path.join(WORK, f"warm_{seed}.vehroutes.xml")
    traci.start(sumo_cmd(seed, ["--vehroute-output", out, "--vehroute-output.exit-times", "true"]))
    field = SpeedField(seed)
    t = 0.0
    while t < END:
        if t % DT_FIELD == 0:
            field.apply(traci, t)
        traci.simulationStep()
        t = traci.simulation.getTime()
    traci.close()
    W = world()
    samples = {}
    for veh in ET.parse(out).getroot().iter("vehicle"):
        r = veh.find("route")
        if r is None or "exitTimes" not in r.attrib:
            continue
        edges = r.attrib["edges"].split(); ex = [float(x) for x in r.attrib["exitTimes"].split()]
        for i in range(1, len(edges) - 1):
            if 600 <= ex[i - 1] <= END - 600 and edges[i] in W["idx"]:
                samples.setdefault(W["idx"][edges[i]], []).append((ex[i - 1], ex[i] - ex[i - 1]))
    os.remove(out)
    return samples


def fit_forecast(sample_sets, n0: float = 10.0, pair_window: float = 60.0):
    """Forecast model from pooled warm-up samples.
    Per edge: empirical mean and std of traversal times, shrunk toward a prior (Uber mean and cv scaled by the
    pooled SUMO/Uber ratios) with weight n / (n + n0). Also estimates the shared fraction of log-variance as
    the correlation of de-meaned log traversal times between vehicles entering the same edge within
    `pair_window` seconds. Returns (mean, std, rho_hat, diagnostics)."""
    W = world()
    pooled = {}
    for S in sample_sets:
        for k, v in S.items():
            pooled.setdefault(k, []).extend(v)
    um, us = W["umean"], W["ustd"]
    ks = [k for k, v in pooled.items() if len(v) >= 20]
    r_mean = np.median([np.mean([x for _, x in pooled[k]]) / um[k] for k in ks])
    r_cv = np.median([np.std([x for _, x in pooled[k]]) / np.mean([x for _, x in pooled[k]]) / (us[k] / um[k]) for k in ks])
    mean = um * r_mean; std = us * r_mean * r_cv            # prior for every edge
    for k, v in pooled.items():
        x = np.array([y for _, y in v]); n = len(x)
        w = n / (n + n0)
        m_emp = x.mean(); sd_emp = x.std() if n > 1 else std[k]
        mean[k] = w * m_emp + (1 - w) * mean[k]
        std[k] = w * sd_emp + (1 - w) * std[k]
    # shared fraction of log-variance: pairs on the same edge, same run, entering within pair_window seconds
    a, b = [], []
    for S in sample_sets:
        for k, v in S.items():
            if len(v) < 4:
                continue
            v_sorted = sorted(v); lt = np.log([y for _, y in v_sorted]); lt = lt - lt.mean()
            ts = [t for t, _ in v_sorted]
            for i in range(len(v_sorted)):
                j = i + 1
                while j < len(v_sorted) and ts[j] - ts[i] <= pair_window:
                    a.append(lt[i]); b.append(lt[j]); j += 1
    std = np.maximum(std, 0.05 * mean)
    rho = float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else 0.5
    diag = dict(edges_sampled=len(pooled), samples=sum(map(len, pooled.values())), sumo_over_uber_mean=float(r_mean),
                sumo_over_uber_cv=float(r_cv), pairs=len(a), rho_hat=rho)
    return mean, std, rho, diag


def warmup_probes(seed: int, n_probe: int = 1200, t_start: float = 600.0, t_end: float = 2400.0):
    """Warm-up with calibration vehicles that drive probe-like routes: origin -> random PsI-like node ->
    destination inside the study area, routed by the same shortest-path rule as the probes. Returns per-edge
    lists of (entry time, traversal time) for the calibration vehicles only (first edge excluded, PsI-to-
    destination part included; the last edge excluded)."""
    import traci
    import networkx as nx
    G = lm.graph(); W = world()
    rng = np.random.default_rng(20_000 + seed)
    nodes = sorted(v for v, (x, y) in G["coords"].items() if v in G["g"] and abs(x - CENTER[0]) <= 1000 and abs(y - CENTER[1]) <= 1000)
    out = os.path.join(WORK, f"warmp_{seed}.vehroutes.xml")
    label = f"warmp{seed}_{os.getpid()}"
    traci.start(sumo_cmd(seed, ["--vehroute-output", out, "--vehroute-output.exit-times", "true"]), label=label)
    tr = traci.getConnection(label)
    deps = np.sort(rng.uniform(t_start, t_end, n_probe))
    for i, t in enumerate(deps):
        o, p, d = (nodes[j] for j in rng.integers(0, len(nodes), 3))
        if len({o, p, d}) < 3:
            continue
        path = nx.shortest_path(G["g"], o, p, weight="mean") + nx.shortest_path(G["g"], p, d, weight="mean")[1:]
        edges = [W["sid"][G["eid"][(a, b)]] for a, b in zip(path, path[1:])]
        tr.route.add(f"cr{i}", edges); tr.vehicle.add(f"cal{i}", f"cr{i}", depart=str(float(t)))
    field = SpeedField(seed)
    t = 0.0
    while t < END:
        if t % DT_FIELD == 0:
            field.apply(tr, t)
        tr.simulationStep()
        t = tr.simulation.getTime()
    tr.close()
    samples = {}
    for veh in ET.parse(out).getroot().iter("vehicle"):
        if not veh.attrib["id"].startswith("cal"):
            continue
        r = veh.find("route")
        if r is None or "exitTimes" not in r.attrib:
            continue
        edges = r.attrib["edges"].split(); ex = [float(x) for x in r.attrib["exitTimes"].split()]
        for i in range(1, len(edges) - 1):
            if edges[i] in W["idx"]:
                samples.setdefault(W["idx"][edges[i]], []).append((ex[i - 1], ex[i] - ex[i - 1]))
    os.remove(out)
    return samples


def fit_probe_forecast(probe_sets, prior_mean, prior_std, n0: float = 5.0, pair_window: float = 60.0):
    """Per-edge forecast from probe-like calibration traversals, shrunk toward a prior (the background-traffic
    fit scaled by the pooled probe/background ratio) with weight n / (n + n0); rho as in fit_forecast but
    from calibration pairs. Returns (mean, std, rho_hat, diagnostics)."""
    pooled = {}
    for S in probe_sets:
        for k, v in S.items():
            pooled.setdefault(k, []).extend(v)
    ks = [k for k, v in pooled.items() if len(v) >= 10]
    r_m = np.median([np.mean([y for _, y in pooled[k]]) / prior_mean[k] for k in ks])
    r_s = np.median([np.std([y for _, y in pooled[k]]) / prior_std[k] for k in ks])
    mean, std = prior_mean * r_m, prior_std * r_s
    for k, v in pooled.items():
        x = np.array([y for _, y in v]); n = len(x); w = n / (n + n0)
        mean[k] = w * x.mean() + (1 - w) * mean[k]
        if n > 1:
            std[k] = w * x.std() + (1 - w) * std[k]
    a, b = [], []
    for S in probe_sets:
        for k, v in S.items():
            if len(v) < 4:
                continue
            vs = sorted(v); lt = np.log([y for _, y in vs]); lt = lt - lt.mean(); ts = [t for t, _ in vs]
            for i in range(len(vs)):
                j = i + 1
                while j < len(vs) and ts[j] - ts[i] <= pair_window:
                    a.append(lt[i]); b.append(lt[j]); j += 1
    std = np.maximum(std, 0.05 * mean)
    rho = float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else 0.5
    return mean, std, rho, dict(edges=len(pooled), samples=sum(map(len, pooled.values())), probe_over_bg_mean=float(r_m),
                                probe_over_bg_std=float(r_s), pairs=len(a), rho_hat=rho)
