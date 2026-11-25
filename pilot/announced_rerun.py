"""Rerun of the main missions with the reward-announcement rule made explicit.

Mechanism: in each window the platform solves the game, lowers the reward if best responses cycle, and announces
the final reward R_w before vehicles commit; each committed vehicle is paid at its window's R_w, also when it samples
in a later window or after the horizon. Decisions are unchanged (same seeds, same code path), so freshness outcomes
reproduce the original runs; payouts now use R_w (payout) and the nominal R is kept for reference (payout_nominal).
Regret missions (est = dist) also record, for every estimator's profile, the regret under R_w (win) and under the
nominal R = 400 (win400), whether the window needed a cut (cut), and R_w / R (rfinal).

Settings and seeds as in the main experiments: synthetic (cv 0.3, rho 0.5), London single stop (rho 0.5), SUMO single
stop, SUMO two stops (30 seeds each), London two stops (20 seeds). Raw: results/announced_raw.json (resumable).
"""
import json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PATH = f"{OUT}/announced_raw.json"
R = 400.0
ESTS = ("dist", "ordermean", "fixorder", "mean")
KEYS = ("deadline_violation", "worst_gap", "net_per_vehicle", "net_nominal", "payout", "payout_nominal", "detours",
        "samples", "breaks")


def job(a):
    setting, m, est = a
    rg = (512, 1024) if est == "dist" else None
    try:
        if setting == "syn":
            from spa_uav.corr_mission import CorrScenario, run
            o, reg = run(CorrScenario(3000 + m, 0.3, 0.5), est, R, S=256, seed=m + 1, regret=rg, ests=ESTS)
            if reg:
                for e in reg:
                    reg[e]["win"] = [x / R for x in reg[e]["win"]]; reg[e]["win400"] = [x / R for x in reg[e]["win400"]]
        elif setting == "london1":
            from spa_uav import london_mission as lm
            o, reg, _ = lm.run(lm.LondonScenario(5000 + m, 0.5), est, R, S=256, seed=m + 1, regret=rg, ests=ESTS)
        elif setting == "london2":
            from spa_uav import london_mission as lm, multistop as ms
            o, reg = ms.run_london(ms.MultiLondon(lm.LondonScenario(5000 + m, 0.5), R, max_stops=2), est, R, S=1024,
                                   seed=m + 1, regret=rg, ests=ESTS)
        elif setting == "sumo1":
            from spa_uav import sumo_mission as smm
            o, reg = smm.run(m, est, R, regret=rg, ests=ESTS)
        else:
            from spa_uav import sumo_multistop as sms
            o, reg = sms.run(m, est, R, S=1024, regret=rg, ests=ESTS)
    except Exception as ex:
        return [list(a), {"error": repr(ex)[:300]}, None]
    o = {k: float(o[k]) for k in KEYS if k in o}
    if reg:
        reg = {e: {k: [float(x) for x in reg[e][k]] for k in ("win", "win400", "cut", "rfinal")} for e in reg}
    return [list(a), o, reg]


if __name__ == "__main__":
    t = time.time()
    done = {}
    if os.path.exists(PATH):
        done = {tuple(a): (o, r) for a, o, r in json.load(open(PATH)) if "error" not in o}
    plan = (("london1", 30), ("syn", 30), ("sumo1", 30), ("sumo2", 30), ("london2", 20))
    jobs = [(s, m, e) for s, n in plan for m in range(n) for e in ESTS]
    todo = [j for j in jobs if j not in done]
    if any(j[0].startswith("sumo") for j in todo):
        from spa_uav import sumo_world as sw
        for s in range(30):
            sw.background(300 + s)
    with Pool(2, maxtasksperchild=1) as pool:   # fresh process per mission (SUMO runs change global edge statistics)
        for a, o, r in pool.imap_unordered(job, todo):
            if "error" in o:
                print("failed", a, o, flush=True); continue
            done[tuple(a)] = (o, r)
            json.dump([[list(k), v[0], v[1]] for k, v in done.items()], open(PATH, "w"))
            print("done", len(done), "of", len(jobs), a, round(time.time() - t), flush=True)
    print("finished", round(time.time() - t), flush=True)
