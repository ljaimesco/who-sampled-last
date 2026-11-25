"""Runtime at measured solution quality, on identical window states.
States come from the dist missions (cv = 0.30, rho = 0.5). For every state and
estimator: time preprocessing (build) and equilibrium search (SPA) separately,
then score the resulting profile with the SAME independent evaluation
(best deviation chosen on 512 fresh draws, gain evaluated on 1024 other draws,
seeded per state). Failure = SPA did not end at a verified equilibrium of its
own sample game; reward cuts are also counted."""
import os, platform, random, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import spa_uav.corr_mission as cm
from spa_uav.policies import spa
from spa_uav.stochastic import profile_regret
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
R = 400.0
STATES = []
_build = cm.build
def capture(scn, players, t0, pend, last, Rr, S, rng, est, rivals=True):
    STATES.append((scn, list(players), t0, list(pend), last.copy()))
    return _build(scn, players, t0, pend, last, Rr, S, rng, est, rivals)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cm.build = capture
    for m in range(n):
        cm.run(cm.CorrScenario(3000 + m, 0.3, 0.5), "dist", R, S=256, seed=m + 1)
    cm.build = _build
    cases = [("mean", 256), ("fixorder", 256), ("ordermean", 256), ("dist", 64), ("dist", 256), ("dist", 1024)]
    res = {c: dict(b=[], s=[], reg=[], fail=0, cuts=0) for c in cases}
    for w, (scn, players, t0, pend, last) in enumerate(STATES):
        erng = np.random.default_rng(10_000 + w)
        gs = _build(scn, players, t0, pend, last, R, 512, erng, "dist")
        ge = _build(scn, players, t0, pend, last, R, 1024, erng, "dist")
        for est, S in cases:
            t = time.perf_counter(); g = _build(scn, players, t0, pend, last, R, S, np.random.default_rng(w), est)
            t1 = time.perf_counter(); r = spa(g, random.Random(w), adapt="global"); t2 = time.perf_counter()
            d = res[(est, S)]; d["b"].append(t1 - t); d["s"].append(t2 - t1)
            d["fail"] += int(not (r.converged and g.is_nash(r.assign))); d["cuts"] += int(r.breaks > 0)
            if r.breaks: gs.R[:] = g.R; ge.R[:] = g.R          # score against the reward actually offered
            d["reg"].append(max(profile_regret(gs, ge, r.assign)) / R)
            gs.R[:] = R; ge.R[:] = R
    N = [len(s[1]) for s in STATES]
    lines = ["# Runtime at measured solution quality (identical window states)", "",
             f"Hardware: Intel Xeon @ 2.10 GHz, 2 vCPU, single process; Python {platform.python_version()}, NumPy {np.__version__}. "
             f"Workload: {len(STATES)} decision windows from {n} missions, cv = 0.30, rho = 0.5, M = 20 PsIs, players per window median {int(np.median(N))}, max {max(N)}. "
             "SPA stop rule: verified equilibrium of its own sample game, strict-improvement tolerance 1e-9. "
             "Quality: ex-ante regret under one common independent evaluation (512 selection draws, 1024 evaluation draws per state).", "",
             "| Estimator | Draws S (mean: averaged) | Preprocessing ms (median / p95) | Search ms (median / p95) | Total ms (median / p95) | Stable windows (regret <= 1% R) | Mean regret | P95 regret | Failures | Windows with reward cut |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (est, S), d in res.items():
        b, s, rg = np.array(d["b"]) * 1e3, np.array(d["s"]) * 1e3, np.array(d["reg"])
        q = np.percentile
        lines.append(f"| {est} | {S} | {q(b,50):.2f} / {q(b,95):.2f} | {q(s,50):.2f} / {q(s,95):.2f} | {q(b+s,50):.2f} / {q(b+s,95):.2f} | "
                     f"{int((rg<=0.01).sum())}/{len(rg)} | {rg.mean()*100:.2f}% | {q(rg,95)*100:.1f}% | {d['fail']} | {d['cuts']} |")
    open(f"{OUT}/runtime_quality.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
