# Who Sampled Last?

Code and results for the paper

> L. G. Jaimes et al., "Who Sampled Last? Equilibria of Freshness Incentives Under Uncertain Arrival Order in
> Vehicular Crowdsensing," submitted to *IEEE Transactions on Intelligent Transportation Systems*.

A platform pays vehicles for samples at places of interest (PsIs), and each payment grows with the time since the
previous sample at the same PsI. A driver's payoff therefore depends on which vehicle arrives first, which is
uncertain when travel times are random. The code builds the drivers' route-choice game in 60 s decision windows,
estimates expected payoffs with four estimators (`dist`, `ordermean`, `fixorder`, `mean`), runs best-response
dynamics with reward cuts when choices cycle, and measures held-out regret, stability, and freshness on a synthetic
network, a London road network model, and closed-loop SUMO microsimulation, for single-stop and two-stop routes.

## Install

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Python 3.10 or later. The `eclipse-sumo`, `traci`, and `sumolib` packages are needed only for the SUMO
experiments; the `eclipse-sumo` wheel ships the SUMO 1.28 binaries. Quick check:

    python -m pytest -q tests

## Reproducing the paper

All experiment scripts run from `pilot/` (`cd pilot && python <script>.py`) and write to `pilot/results/`. Every
script uses fixed seeds, pairs missions across estimators, and saves after each mission, so an interrupted run
resumes where it stopped. The raw results used in the paper are included, so every summary and figure can be
regenerated without rerunning the simulations. Times are approximate, on 2 cores.

| Paper item | Script(s) | Output | Time |
|---|---|---|---|
| Fig. 1 (A/B example) | `figures/fig_ab.py` | `figures/fig_ab.pdf` | seconds |
| Fig. 2 (stable windows), outcomes table | `announced_rerun.py`, then `announced_summary.py`; figure: `figures/fig_stability.py` | `announced_raw.json`, `announced_summary.md` | several hours |
| Fig. 3 (regret distributions) | `london_experiment.py 30`, `sumo_ms_experiment.py`; figure: `figures/fig_regret.py` | `london_raw.json`, `sumo_ms_raw.json` | about 2.5 h (60 min London, 80 min SUMO) |
| Theory check table | `theory_check.py`, `theory_summary.py`, `theory_mc.py` | `theory_check.md`, `theory_mc.md` | under 1 h |
| Matched spending table | `multistop_matched_announced.py`, `multistop_accounting_announced.py` | `multistop_matched_announced.md`, `multistop_accounting_announced.md` | about 3 h |
| Greedy baseline table | `greedy_experiment.py`, `greedy_summary.py` | `greedy_raw.json`, `greedy_summary.md` | several hours |
| Supplement: synthetic worlds and ablation | `corr_experiment.py 30`, `ablation_experiment.py 30` | `corr_summary.md`, `ablation_summary.md` | 7 min |
| Supplement: certificates | `certificate.py`, `certificate_bounds.py` | `certificates.md`, `certificate_bounds.md`, `CERTIFICATE.md` | 3 min |
| Supplement: London extras | `london_stress.py`, `london_matched.py` | `london_stress.md`, `london_matched.md` | 75 min |
| Supplement: SUMO single stop and forecast follow-up | `sumo_experiment.py 30`, `sumo_forecast_quality.py`, `sumo_forecast_quality_summary.py` | `sumo_summary.md`, `sumo_followup.md` | 45 min plus follow-up |
| Supplement: two-stop details | `multistop_experiment.py`, `multistop_summary.py`, `sumo_ms_compare.py` | `multistop_summary.md`, `sumo_ms_compare.md` | 1 to 2 h |
| Supplement: runtime | `runtime_quality.py 10`, `runtime_benchmark.py` | `runtime_quality.md`, `runtime.md` | minutes |

Figures: `python figures/fig_ab.py` (and likewise `fig_stability.py`, `fig_regret.py`) writes PDF and PNG files
into `figures/`.

Scripts not listed above are exploratory or earlier versions kept for provenance: `matched_budget.py`,
`multistop_matched.py`, and `multistop_accounting.py` use nominal rather than announced payments;
`order_control.py`, `fixorder_variants.py`, `small_games*.py`, `exact_small.py`, and `uncertainty_pilot.py` are
diagnostics.

## Layout

| Path | Content |
|---|---|
| `spa_uav/stochastic.py` | Expected-utility game, the four payoff estimators, held-out regret |
| `spa_uav/policies.py`, `spa_uav/model.py` | Best-response dynamics, opting out, reward cuts on cycles |
| `spa_uav/corr_mission.py` | Synthetic correlated-traffic missions |
| `spa_uav/london_mission.py`, `spa_uav/real_london.py` | London road network model with edge-and-time congestion |
| `spa_uav/multistop.py` | Two-stop routes (ordered pairs of PsIs) |
| `spa_uav/sumo_world.py`, `spa_uav/sumo_mission.py`, `spa_uav/sumo_multistop.py` | Closed-loop SUMO missions via TraCI, learned travel-time forecast |
| `spa_uav/fresh_greedy.py` | Freshness-oriented greedy baseline |
| `spa_uav/theory.py` | Order, curvature, and truncation terms of the payoff error |
| `pilot/` | One script per experiment and its summary; raw results in `pilot/results/` |
| `figures/` | Scripts for the paper's figures |
| `data/` | Road network and travel-time data |
| `tests/` | Equilibrium, cycle-breaking, reproducibility, and two-stop checks |

The package name `spa_uav` comes from the shared Smart PsI Allocation core (best response with reward cuts), which
this code extends to uncertain arrival order.

## Data

`data/london.net.xml`, `data/london-seg4.100.nod.xml`, and `data/uberspeed.xml` are copies from the ANTE-Mobility
project (London road network and per-segment travel-time means and standard deviations derived from Uber Movement
data); see that project for the original sources and their terms of use. `data/sumo_forecast*.npz` are travel-time
forecasts learned from SUMO warm-up runs; they can be regenerated with the `warmup`, `warmup_probes`,
`fit_forecast`, and `fit_probe_forecast` functions in `spa_uav/sumo_world.py`. SUMO background traffic is
generated per seed into `sumo_runs/` on first use. The SUMO calibration constants (prior travel-time scale 1.285,
spread inflation 1.5) were fixed on calibration seeds and can be overridden with the environment variables
`SPA_SCALE_PRIOR` and `SPA_SIGMA_GAIN`.

## License

MIT; see `LICENSE`. The files in `data/` keep the terms of their original sources.
