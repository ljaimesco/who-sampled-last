"""Real travel-time model built from ANTE-Mobility's London grid data.

ANTE-Mobility/uberspeed.xml gives, for 9119 real road segments of a London
grid, the real per-edge travel-time statistics (mean, std, p50, p85) derived
from Uber movement data (not simulated). This module turns that into a
routable graph and lets us draw route-level travel-time samples for an
arbitrary origin/destination pair, calibrated edge by edge from real data,
in place of the synthetic lognormal(cv) speed-factor model used elsewhere
in this package.

Each edge's travel time is modeled as lognormal, matched by moments to its
real (mean, std); a route's time is the sum of its edges' samples, drawn
independently per edge. This is an approximation (ignores spatial
correlation between adjacent edges, which the synthetic model also doesn't
have beyond the zone-level component), but it uses real measured variability
instead of an assumed cv, which is the point of this check.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class LondonGraph:
    graph: "nx.DiGraph"
    coords: dict  # node id -> (x, y)


def load(nod_path: str, uberspeed_path: str) -> LondonGraph:
    coords = {}
    for node in ET.parse(nod_path).getroot().findall("node"):
        coords[node.attrib["id"]] = (float(node.attrib["x"]), float(node.attrib["y"]))

    g = nx.DiGraph()
    for edge in ET.parse(uberspeed_path).getroot().findall("edge"):
        u, v = edge.attrib["from"], edge.attrib["to"]
        if u == v:
            continue  # self-loop edges (u-turn / same-cell) carry no routing information
        mean = float(edge.attrib["mean"])
        std = float(edge.attrib["std"])
        if mean <= 0:
            continue
        s2 = math.log1p((std / mean) ** 2)
        sigma = math.sqrt(s2)
        mu = math.log(mean) - s2 / 2.0
        g.add_edge(u, v, mean=mean, std=std, mu=mu, sigma=sigma, id=edge.attrib["id"])
    return LondonGraph(graph=g, coords=coords)


def shortest_route(lg: LondonGraph, origin: str, dest: str):
    """List of (mu, sigma) for the edges on the mean-time shortest path, or None."""
    try:
        path = nx.shortest_path(lg.graph, origin, dest, weight="mean")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if len(path) < 2:
        return []
    return [(lg.graph[a][b]["mu"], lg.graph[a][b]["sigma"]) for a, b in zip(path, path[1:])]


def route_mean(route) -> float:
    return sum(math.exp(mu + sigma * sigma / 2.0) for mu, sigma in route)


def sample_route_time(route, rng, size: int) -> np.ndarray:
    """Route travel time = sum of independently sampled real edge times."""
    if not route:
        return np.zeros(size)
    total = np.zeros(size)
    for mu, sigma in route:
        total += np.exp(mu + sigma * rng.standard_normal(size)) if sigma > 0 else math.exp(mu)
    return total


def largest_component(lg: LondonGraph) -> LondonGraph:
    """Restrict to the largest weakly-connected component, so random origin/
    destination picks are routable."""
    keep = [n for n in lg.graph.nodes() if n in lg.coords]
    g = lg.graph.subgraph(keep)
    comp = max(nx.strongly_connected_components(g), key=len)
    sub = g.subgraph(comp).copy()
    return LondonGraph(graph=sub, coords={k: v for k, v in lg.coords.items() if k in comp})
