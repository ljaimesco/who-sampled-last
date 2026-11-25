"""Per-decision runtime. One process, no other load. Wraps corr_mission.build
(preprocessing: drawing the joint arrival samples and the rival-independent
part of each predecessor) and spa (equilibrium search) with timers."""
import os, platform, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import spa_uav.corr_mission as cm
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_build, _spa = cm.build, cm.spa
LOG = []
def tbuild(*a, **k):
    t = time.perf_counter(); g = _build(*a, **k); LOG.append(dict(build=time.perf_counter() - t, g=g)); return g
def tspa(g, *a, **k):
    t = time.perf_counter(); r = _spa(g, *a, **k)
    LOG[-1].update(solve=time.perf_counter() - t, N=g.N, cand=np.mean([len(o) - 1 for o in g.options]),
                   switches=r.switches, checks=r.checks, evals=g.evals); return r
cm.build, cm.spa = tbuild, tspa

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cases = [("mean", 1), ("fixorder", 256), ("ordermean", 256), ("dist", 64), ("dist", 256), ("dist", 1024)]
    lines = ["# Runtime per decision window", "",
             f"Hardware: {platform.processor() or 'Intel Xeon @ 2.10GHz'}, 2 vCPU, one process; Python {platform.python_version()}, NumPy {np.__version__}.",
             f"Scenario: cv = 0.30, rho = 0.5, M = 20 PsIs, {n} missions (same window states for every row). SPA stops at a verified equilibrium of its sample game (strict improvement tolerance 1e-9).", "",
             "| Estimator | Draws S | Windows | Players per window (median / max) | Candidate PsIs per player (mean) | Utility evaluations (median) | Preprocessing ms (median / p95) | Equilibrium search ms (median / p95 / max) | Total ms (median / p95) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for est, S in cases:
        LOG.clear()
        for m in range(n):
            cm.run(cm.CorrScenario(3000 + m, 0.3, 0.5), est, 400.0, S=S, seed=m + 1)
        b = np.array([x["build"] for x in LOG]) * 1e3; s = np.array([x["solve"] for x in LOG]) * 1e3
        N = np.array([x["N"] for x in LOG]); c = np.array([x["cand"] for x in LOG]); ev = np.array([x["evals"] for x in LOG])
        q = lambda x, p: np.percentile(x, p)
        lines.append(f"| {est} | {S} | {len(LOG)} | {int(np.median(N))} / {N.max()} | {c.mean():.1f} | {int(np.median(ev))} | "
                     f"{q(b,50):.2f} / {q(b,95):.2f} | {q(s,50):.2f} / {q(s,95):.2f} / {s.max():.1f} | {q(b+s,50):.2f} / {q(b+s,95):.2f} |")
    open(f"{OUT}/runtime.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
