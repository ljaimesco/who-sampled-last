# London stress test: stronger dependence vs matched variability (rho = 0.8)

| Setting | Median route cv | Pair arrival correlation (median, p90) | Var(Ti - Tj)/(Var Ti + Var Tj) (median, p10) |
|---|---|---|---|
| area | 0.312 | 0.402, 0.875 | 0.686, 0.156 |
| spread | 0.306 | 0.058, 0.387 | 0.952, 0.626 |

| Setting | Estimator | Stable windows | Stable share (95% CI) | Mean regret | P95 regret |
|---|---|---|---|---|---|
| area | dist | 798/900 | 86.3 to 90.8% | 0.28% | 1.9% |
| area | ordermean | 508/900 | 52.8 to 60.1% | 1.54% | 6.2% |
| area | fixorder | 163/900 | 16.0 to 20.2% | 12.80% | 30.2% |
| area | mean | 147/900 | 14.0 to 18.7% | 12.93% | 30.1% |
| spread | dist | 808/900 | 87.7 to 91.9% | 0.27% | 2.0% |
| spread | ordermean | 469/900 | 48.9 to 55.1% | 1.87% | 6.5% |
| spread | fixorder | 154/900 | 15.0 to 19.3% | 12.79% | 29.5% |
| spread | mean | 144/900 | 13.7 to 18.4% | 12.67% | 29.4% |

| Setting | Time past deadline dist vs mean | p | Net per vehicle | p | Payout | p | Inversions dist / mean |
|---|---|---|---|---|---|---|---|
| area | 0.150 vs 0.169 (-11.6%) | 1.9e-09 | +4.1% | 3.7e-09 | +2.9% | 5.6e-09 | 7.7% / 9.2% |
| spread | 0.153 vs 0.166 (-8.2%) | 6.9e-07 | +3.6% | 1.9e-09 | +2.3% | 1e-07 | 7.6% / 10.4% |
