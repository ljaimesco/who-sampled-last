# SUMO closed-loop experiment (London grid, multi-stop)

30 paired seeds. Forecast quality over all estimators' missions: bias 49.7 s, MAE 89.4 s, 80% interval coverage 0.80; teleports per mission 65; stops never sampled 36.6 of 289.

## Ex-ante window regret (forecast model, matched states)

| Estimator | Stable windows | Stable share (95% CI) | Mean regret | P95 regret |
|---|---|---|---|---|
| dist | 800/900 | 86.6 to 91.2% | 0.29% | 2.0% |
| ordermean | 145/900 | 13.6 to 18.8% | 8.61% | 21.3% |
| fixorder | 26/900 | 1.8 to 4.0% | 26.03% | 54.5% |
| mean | 23/900 | 1.6 to 3.8% | 26.77% | 55.0% |

## Realized outcomes in SUMO (mean over seeds, 95% bootstrap CI)

| Estimator | Time past deadline | Longest gap (s) | Net per vehicle | Payout | Order inversions |
|---|---|---|---|---|---|
| dist | 0.159 (0.147 to 0.171) | 367 (355 to 379) | 276 (268 to 284) | 113326 (110790 to 115802) | 29.7% (28.9% to 30.3%) |
| ordermean | 0.169 (0.157 to 0.182) | 378 (366 to 390) | 262 (254 to 271) | 112552 (110181 to 114855) | 30.5% (29.6% to 31.3%) |
| fixorder | 0.165 (0.152 to 0.176) | 371 (359 to 384) | 266 (258 to 275) | 111166 (108891 to 113481) | 29.5% (28.7% to 30.2%) |
| mean | 0.170 (0.158 to 0.182) | 384 (371 to 398) | 259 (251 to 267) | 111970 (109358 to 114577) | 30.2% (29.5% to 31.0%) |

Paired Wilcoxon, dist vs mean:

| Metric | Change | p |
|---|---|---|
| deadline_violation | -6.7% | 9.2e-06 |
| worst_gap | -4.5% | 1.4e-05 |
| net_per_vehicle | +6.5% | 1.9e-09 |
| payout | +1.2% | 0.0058 |
| inversion_rate | -1.9% | 0.08 |
