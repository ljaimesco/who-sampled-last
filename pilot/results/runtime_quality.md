# Runtime at measured solution quality (identical window states)

Hardware: Intel Xeon @ 2.10 GHz, 2 vCPU, single process; Python 3.13.18, NumPy 2.5.3. Workload: 300 decision windows from 10 missions, cv = 0.30, rho = 0.5, M = 20 PsIs, players per window median 10, max 19. SPA stop rule: verified equilibrium of its own sample game, strict-improvement tolerance 1e-9. Quality: ex-ante regret under one common independent evaluation (512 selection draws, 1024 evaluation draws per state).

| Estimator | Draws S (mean: averaged) | Preprocessing ms (median / p95) | Search ms (median / p95) | Total ms (median / p95) | Stable windows (regret <= 1% R) | Mean regret | P95 regret | Failures | Windows with reward cut |
|---|---|---|---|---|---|---|---|---|---|
| mean | 256 | 1.08 / 1.65 | 1.57 / 3.38 | 2.71 / 4.82 | 115/300 | 8.75% | 28.4% | 0 | 1 |
| fixorder | 256 | 0.98 / 1.44 | 1.98 / 4.18 | 3.04 / 5.55 | 110/300 | 9.12% | 28.9% | 0 | 1 |
| ordermean | 256 | 0.91 / 1.32 | 4.15 / 9.81 | 5.08 / 11.03 | 242/300 | 0.60% | 3.9% | 0 | 0 |
| dist | 64 | 0.65 / 0.99 | 1.45 / 3.29 | 2.13 / 4.06 | 252/300 | 0.52% | 3.5% | 0 | 0 |
| dist | 256 | 1.14 / 1.73 | 1.71 / 3.76 | 2.90 / 5.48 | 277/300 | 0.19% | 1.5% | 0 | 0 |
| dist | 1024 | 2.98 / 4.63 | 2.31 / 5.00 | 5.33 / 9.63 | 294/300 | 0.05% | 0.0% | 0 | 0 |
