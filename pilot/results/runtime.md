# Runtime per decision window

Hardware: x86_64, 2 vCPU, one process; Python 3.13.18, NumPy 2.5.3.
Scenario: cv = 0.30, rho = 0.5, M = 20 PsIs, 10 missions (same window states for every row). SPA stops at a verified equilibrium of its sample game (strict improvement tolerance 1e-9).

| Estimator | Draws S | Windows | Players per window (median / max) | Candidate PsIs per player (mean) | Utility evaluations (median) | Preprocessing ms (median / p95) | Equilibrium search ms (median / p95 / max) | Total ms (median / p95) |
|---|---|---|---|---|---|---|---|---|
| mean | 1 | 300 | 10 / 19 | 10.5 | 324 | 0.41 / 0.61 | 1.45 / 3.81 / 7.4 | 1.91 / 4.41 |
| fixorder | 256 | 300 | 10 / 19 | 10.5 | 321 | 0.90 / 1.41 | 1.78 / 4.21 / 6.8 | 2.74 / 5.46 |
| ordermean | 256 | 300 | 10 / 19 | 10.5 | 296 | 0.80 / 1.19 | 3.98 / 8.98 / 20.9 | 4.79 / 9.95 |
| dist | 64 | 300 | 10 / 19 | 10.5 | 293 | 0.53 / 0.76 | 1.35 / 3.28 / 5.2 | 1.90 / 3.94 |
| dist | 256 | 300 | 10 / 19 | 10.5 | 303 | 1.11 / 1.85 | 1.57 / 3.59 / 6.7 | 2.71 / 5.23 |
| dist | 1024 | 300 | 10 / 19 | 10.5 | 304 | 4.20 / 7.69 | 2.08 / 4.53 / 6.7 | 6.28 / 11.53 |
