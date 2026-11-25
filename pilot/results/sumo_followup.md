# SUMO follow-up: forecast quality and matched spending

30 paired seeds. Original = global warm-up forecast (scale prior 1.285, spread x1.5). Per-seed = forecast refit from a calibration day with the same PsIs and demand pattern (different traffic), no inflation.

## Forecast accuracy (all estimators' missions)

| Metric | Original | Per-seed |
|---|---|---|
| forecast_bias | 43.2 s | 38.3 s |
| forecast_mae | 82.5 s | 81.5 s |
| forecast_cover80 | 0.80 | 0.55 |
| inversion_rate | 21.7% | 15.7% |

## Ex-ante stability (forecast model)

| Estimator | Original stable | Per-seed stable |
|---|---|---|
| dist | 778/900 (86%) | 821/900 (91%) |
| ordermean | 256/900 (28%) | 524/900 (58%) |
| fixorder | 82/900 (9%) | 216/900 (24%) |
| mean | 92/900 (10%) | 187/900 (21%) |

## Realized outcomes: paired Wilcoxon, dist vs each estimator

| Forecast | Comparison | Time past deadline | p | Longest gap | p | Net per vehicle | p | Payout | p |
|---|---|---|---|---|---|---|---|---|---|
| original | dist vs mean | -3.6% | 2.1e-05 | -3.7% | 7.1e-05 | +4.4% | 3.5e-08 | +1.0% | 0.019 |
| original | dist vs fixorder | +2.4% | 0.015 | +1.5% | 0.077 | +4.2% | 3.7e-09 | +0.6% | 0.24 |
| original | dist vs ordermean | -6.3% | 2.6e-08 | -5.1% | 8.3e-07 | +2.4% | 2e-06 | +1.9% | 3e-05 |
| per-seed | dist vs mean | -3.0% | 0.0024 | -2.8% | 7.1e-05 | +4.2% | 1.6e-07 | +1.9% | 0.00023 |
| per-seed | dist vs fixorder | -1.2% | 0.22 | -0.6% | 0.45 | +3.2% | 8.3e-07 | +1.2% | 0.003 |
| per-seed | dist vs ordermean | -1.8% | 0.0099 | -1.9% | 0.0022 | +0.1% | 0.73 | +0.3% | 0.44 |

## Mean outcomes per estimator

| Forecast | Estimator | Time past deadline | Longest gap | Net per vehicle | Payout |
|---|---|---|---|---|---|
| original | dist | 0.295 | 473 | 222 | 78432 |
| original | ordermean | 0.315 | 499 | 217 | 76949 |
| original | fixorder | 0.288 | 466 | 213 | 77960 |
| original | mean | 0.306 | 491 | 213 | 77655 |
| per-seed | dist | 0.306 | 489 | 212 | 77508 |
| per-seed | ordermean | 0.311 | 498 | 212 | 77266 |
| per-seed | fixorder | 0.310 | 492 | 206 | 76581 |
| per-seed | mean | 0.315 | 503 | 204 | 76100 |

## Matched spending (original forecast): dist reward 398.3, calibrated on seeds 100-103

| Comparison | Payout | p | Time past deadline | p | Longest gap | p | Net per vehicle | p |
|---|---|---|---|---|---|---|---|---|
| dist (matched) vs mean | +1.0% | 0.043 | -3.9% | 0.00038 | -3.8% | 0.00014 | +4.2% | 2e-07 |
| dist (matched) vs fixorder | +0.6% | 0.26 | +2.1% | 0.017 | +1.4% | 0.16 | +4.0% | 1.9e-08 |
