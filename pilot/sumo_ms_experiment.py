"""SUMO closed-loop experiment on the London grid with multi-stop routes (up to two PsIs per vehicle).
Same worlds, forecast and calibration as sumo_experiment.py; decisions with S=1024 draws (multi-stop games need more
draws than single-stop ones for the same sampling error).

Ground truth: SUMO with background traffic and the Uber-calibrated speed field (sumo_world.py).
Forecast: per-edge lognormal learned from warm-up runs with probe-like calibration vehicles, shared fraction
estimated from same-run pairs; prior travel-time scale 1.285 and spread inflation 1.5 fixed on calibration
missions (seeds 100-103, never evaluated); online rescaling from completed trips during each mission.
30 paired seeds (same demand, background traffic, speed field and SUMO seed for every estimator).
dist missions also score every estimator's profile at each window state (ex-ante regret in the forecast model,
independent 512 / 1024 draws)."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from spa_uav import sumo_multistop as sms
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
R, TOL, ESTS = 400.0, 0.01, ("dist", "ordermean", "fixorder", "mean")
KEYS = ("deadline_violation", "worst_gap", "net_per_vehicle", "payout", "inversion_rate", "teleports",
        "never_sampled", "pair_routes", "samples", "forecast_bias", "forecast_mae", "forecast_cover80", "detours")

def job(a):
    seed, est = a
    try:
        o, reg = sms.run(seed, est, R, S=1024, regret=(512, 1024) if est == "dist" else None)
    except Exception as ex:                       # keep the experiment alive; failures are reported
        return a, {"error": repr(ex)[:300]}, None
    return a, {k: float(o[k]) for k in KEYS}, ({e: reg[e]["win"] for e in ESTS} if reg else None)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    t = time.time(); res = []
    from spa_uav import sumo_world as sw
    with Pool(2) as pool:                         # pre-generate background traffic (one file per seed, no races)
        pool.map(sw.background, [300 + s for s in range(n)])
    print("background traffic ready", round(time.time() - t), flush=True)
    with Pool(2) as pool:
        done = {}
        if os.path.exists(f"{OUT}/sumo_ms_raw.json"):
            for a, o, w in json.load(open(f"{OUT}/sumo_ms_raw.json")):
                if "error" not in o: done[tuple(a)] = (tuple(a), o, w)
        todo = [(s, e) for s in range(n) for e in ESTS if (s, e) not in done]
        res = list(done.values())
        for r in pool.imap_unordered(job, todo):
            res.append(r)
            json.dump([[list(a), o, w] for a, o, w in res], open(f"{OUT}/sumo_ms_raw.json", "w"))
            if len(res) % 10 == 0: print("missions", len(res), round(time.time() - t), flush=True)
    failed = [a for a, o, w in res if "error" in o]
    print("failed missions:", failed, flush=True)
    res = [r for r in res if "error" not in r[1]]
    ok_seeds = sorted({a[0] for a, o, w in res if all(any(b == (a[0], e) for b, _, _ in res) for e in ESTS)})
    res = [r for r in res if r[0][0] in ok_seeds]
    n = len(ok_seeds)
    rng = np.random.default_rng(0)
    out = {(a[0], a[1]): o for a, o, w in res}
    regs = [w for a, o, w in res if w]
    L = ["# SUMO closed-loop experiment (London grid, multi-stop)", "",
         f"{n} paired seeds. Forecast quality over all estimators' missions: bias {np.mean([o['forecast_bias'] for o in out.values()]):.1f} s, "
         f"MAE {np.mean([o['forecast_mae'] for o in out.values()]):.1f} s, 80% interval coverage {np.mean([o['forecast_cover80'] for o in out.values()]):.2f}; "
         f"teleports per mission {np.mean([o['teleports'] for o in out.values()]):.0f}; stops never sampled {np.mean([o['never_sampled'] for o in out.values()]):.1f} of {np.mean([o['detours'] for o in out.values()]):.0f}.", "",
         "## Ex-ante window regret (forecast model, matched states)", "",
         "| Estimator | Stable windows | Stable share (95% CI) | Mean regret | P95 regret |", "|---|---|---|---|---|"]
    for e in ESTS:
        per = [np.array(w[e]) for w in regs]; x = np.concatenate(per)
        bs = [(np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))]) <= TOL).mean() for _ in range(3000)]
        L.append(f"| {e} | {int((x<=TOL).sum())}/{len(x)} | {np.percentile(bs,2.5)*100:.1f} to {np.percentile(bs,97.5)*100:.1f}% | {x.mean()*100:.2f}% | {np.percentile(x,95)*100:.1f}% |")
    L += ["", "## Realized outcomes in SUMO (mean over seeds, 95% bootstrap CI)", "",
          "| Estimator | Time past deadline | Longest gap (s) | Net per vehicle | Payout | Order inversions |", "|---|---|---|---|---|---|"]
    col = lambda e, k: np.array([out[(s, e)][k] for s in ok_seeds])
    for e in ESTS:
        cells = []
        for k, f in (("deadline_violation", "{:.3f}"), ("worst_gap", "{:.0f}"), ("net_per_vehicle", "{:.0f}"), ("payout", "{:.0f}"), ("inversion_rate", "{:.1%}")):
            x = col(e, k); b = [rng.choice(x, len(x)).mean() for _ in range(3000)]
            cells.append(f"{f.format(x.mean())} ({f.format(np.percentile(b,2.5))} to {f.format(np.percentile(b,97.5))})")
        L.append(f"| {e} | " + " | ".join(cells) + " |")
    L += ["", "Paired Wilcoxon, dist vs mean:", "", "| Metric | Change | p |", "|---|---|---|"]
    for k in ("deadline_violation", "worst_gap", "net_per_vehicle", "payout", "inversion_rate"):
        d_, m_ = col("dist", k), col("mean", k)
        L.append(f"| {k} | {(d_.mean()-m_.mean())/m_.mean()*100:+.1f}% | {stats.wilcoxon(d_, m_).pvalue:.2g} |")
    open(f"{OUT}/sumo_ms_summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("elapsed", round(time.time() - t))
