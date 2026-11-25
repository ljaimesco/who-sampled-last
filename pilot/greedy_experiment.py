"""Platform-oriented greedy baseline (spa_uav.fresh_greedy) in the four main settings, same seeds as the main runs.

Per setting and seed:
  greedy     the greedy profile followed as is (full compliance); regret of the greedy and greedy_br profiles is
             scored at these window states on independent draws (selection 512, evaluation 1024), as for the
             estimators in the main experiments;
  greedy_seq drivers react once, in random order, each best-responding to the decisions so far (operational baseline);
  greedy_br  drivers react until no one wants to change (best-response dynamics from the greedy profile).
o["changed"] / o["players"]: share of drivers whose final action differs from the greedy recommendation.
Results are compared with the estimators' outcomes from the main raw files (same seeds, R = 400).
Raw: results/greedy_raw.json (resumable); summary: results/greedy_summary.md.
"""
import json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PATH = f"{OUT}/greedy_raw.json"
R = 400.0
KEYS = ("deadline_violation", "worst_gap", "net_per_vehicle", "payout", "detours", "samples")
GE = ("greedy", "greedy_seq", "greedy_br")


def job(a):
    setting, seed, est = a
    from spa_uav import fresh_greedy as fg
    fg.COUNT.clear()
    reg_arg = (512, 1024) if est == "greedy" else None
    try:
        if setting == "london1":
            from spa_uav import london_mission as lm
            o, reg, _ = lm.run(lm.LondonScenario(5000 + seed, 0.5), est, R, S=256, seed=seed + 1, regret=reg_arg, ests=GE)
        elif setting == "london2":
            from spa_uav import london_mission as lm, multistop as ms
            ml = ms.MultiLondon(lm.LondonScenario(5000 + seed, 0.5), R, max_stops=2)
            o, reg = ms.run_london(ml, est, R, S=1024, seed=seed + 1, regret=reg_arg, ests=GE)
        elif setting == "sumo1":
            from spa_uav import sumo_mission as smm
            o, reg = smm.run(seed, est, R, regret=reg_arg, ests=GE)
        else:
            from spa_uav import sumo_multistop as sms
            o, reg = sms.run(seed, est, R, S=1024, regret=reg_arg, ests=GE)
    except Exception as ex:
        return [list(a), {"error": repr(ex)[:300]}, None]
    o = {k: float(o[k]) for k in KEYS if k in o}
    c = fg.COUNT.get(est, [0, 0]); o["players"], o["changed"] = c
    reg = {e: {"win": [float(x) for x in reg[e]["win"]], "cut": reg[e]["cut"]} for e in reg} if reg else None
    return [list(a), o, reg]


if __name__ == "__main__":
    t = time.time()
    done = {}
    if os.path.exists(PATH):
        done = {tuple(a): (o, r) for a, o, r in json.load(open(PATH)) if "error" not in o}
    jobs = [(s, m, e) for s, n in (("london1", 30), ("sumo2", 30), ("sumo1", 30), ("london2", 20)) for m in range(n) for e in GE]
    todo = [j for j in jobs if j not in done]
    if any(j[0].startswith("sumo") for j in todo):
        from spa_uav import sumo_world as sw
        for s in range(30):
            sw.background(300 + s)
    with Pool(2, maxtasksperchild=1) as pool:   # fresh process per mission: SUMO runs reset global edge statistics
        for a, o, r in pool.imap_unordered(job, todo):
            if "error" in o:
                print("failed", a, o, flush=True); continue
            done[tuple(a)] = (o, r)
            json.dump([[list(k), v[0], v[1]] for k, v in done.items()], open(PATH, "w"))
            print("done", len(done), "of", len(jobs), a, round(time.time() - t), flush=True)
    print("finished", round(time.time() - t), flush=True)
