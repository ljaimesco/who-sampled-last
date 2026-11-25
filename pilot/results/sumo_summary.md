# SUMO closed-loop experiment (London grid, single-stop)

30 paired seeds. Forecast quality over all estimators' missions: bias 43.2 s, MAE 82.5 s, 80% interval coverage 0.80; teleports per mission 63; probes never reaching their PsI 16.2 of 291.

## Ex-ante window regret (forecast model, matched states)

| Estimator | Stable windows | Stable share (95% CI) | Mean regret | P95 regret |
|---|---|---|---|---|
| dist | 778/900 | 84.3 to 88.3% | 0.33% | 2.3% |
| ordermean | 256/900 | 25.0 to 31.9% | 5.31% | 15.0% |
| fixorder | 82/900 | 7.2 to 11.1% | 17.20% | 39.8% |
| mean | 92/900 | 7.6 to 12.9% | 16.06% | 38.3% |

## Realized outcomes in SUMO (mean over seeds, 95% bootstrap CI)

| Estimator | Time past deadline | Longest gap (s) | Net per vehicle | Payout | Order inversions |
|---|---|---|---|---|---|
| dist | 0.295 (0.279 to 0.310) | 473 (461 to 486) | 222 (217 to 227) | 78432 (76800 to 80116) | 21.5% (20.4% to 22.7%) |
| ordermean | 0.315 (0.299 to 0.330) | 499 (482 to 516) | 217 (212 to 222) | 76949 (75169 to 78610) | 20.9% (19.8% to 22.1%) |
| fixorder | 0.288 (0.272 to 0.304) | 466 (453 to 481) | 213 (208 to 218) | 77960 (76331 to 79614) | 21.9% (20.8% to 23.0%) |
| mean | 0.306 (0.290 to 0.320) | 491 (476 to 507) | 213 (208 to 217) | 77655 (75964 to 79459) | 22.5% (21.4% to 23.5%) |

Paired Wilcoxon, dist vs mean:

| Metric | Change | p |
|---|---|---|
| deadline_violation | -3.6% | 2.1e-05 |
| worst_gap | -3.7% | 7.1e-05 |
| net_per_vehicle | +4.4% | 3.5e-08 |
| payout | +1.0% | 0.019 |
| inversion_rate | -4.4% | 0.028 |
