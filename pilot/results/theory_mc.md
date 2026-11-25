# Monte Carlo uncertainty of the reversal analysis

Paired standard error of u(true best) - u(estimator's best) on the 4096 evaluation draws, percent of R (decisions where the two differ).

| Estimator | Decisions | Differing choices | Paired SE: median / 95th pct | Reversals, 95% interval excludes 0 | Rate (mission-clustered 95% CI) | Reversals > 2% of R, interval excludes 2% |
|---|---|---|---|---|---|---|
| dist | 2918 | 30 | 0.34 / 0.61 | 7 | 0.2% (0.1% to 0.4%) | 0 (0.0%) |
| ordermean | 2918 | 116 | 0.27 / 0.54 | 94 | 3.2% (2.4% to 4.1%) | 18 (0.6%) |
| fixorder | 2918 | 391 | 0.49 / 0.64 | 364 | 12.5% (11.7% to 13.3%) | 302 (10.3%) |
| mean | 2918 | 418 | 0.48 / 0.64 | 389 | 13.3% (12.5% to 14.2%) | 306 (10.5%) |
