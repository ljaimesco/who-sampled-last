# Multi-stop: expenditure approximately matched (held-out)

dist's reward is calibrated on seeds never evaluated so its payout matches mean-time's at R = 400; other estimators keep R = 400.

## LONDON (calibrated dist reward 295.0, 20 evaluation seeds)

| Comparison | Payout difference | p | Time past deadline | p | Longest gap | p | Net per vehicle | p | Samples |
|---|---|---|---|---|---|---|---|---|---|
| dist (R = 295) vs mean (R = 400) | +4.1% | 0.076 | -17.0% | 3.6e-05 | -4.5% | 0.0023 | +5.8% | 0.04 | 570 vs 546 |
| dist (R = 295) vs fixorder (R = 400) | +3.3% | 0.036 | -14.5% | 6.3e-05 | -3.6% | 0.0032 | +4.5% | 0.0094 | 570 vs 546 |
| dist (R = 295) vs ordermean (R = 400) | -28.0% | 1.9e-06 | +10.5% | 0.0023 | +1.0% | 0.37 | -29.7% | 1.9e-06 | 570 vs 584 |

## SUMO (calibrated dist reward 341.4, 30 evaluation seeds)

| Comparison | Payout difference | p | Time past deadline | p | Longest gap | p | Net per vehicle | p | Samples |
|---|---|---|---|---|---|---|---|---|---|
| dist (R = 341) vs mean (R = 400) | -4.7% | 0.00046 | -2.6% | 0.047 | -2.5% | 0.0099 | +4.8% | 0.0022 | 557 vs 557 |
| dist (R = 341) vs fixorder (R = 400) | -4.9% | 0.0011 | +0.8% | 0.75 | +1.0% | 0.26 | +0.1% | 0.81 | 557 vs 554 |
| dist (R = 341) vs ordermean (R = 400) | -14.9% | 1.9e-09 | -2.0% | 0.088 | -0.7% | 0.17 | -11.9% | 1.9e-09 | 557 vs 576 |

## SUMO1 (calibrated dist reward 400.6, 30 evaluation seeds)

| Comparison | Payout difference | p | Time past deadline | p | Longest gap | p | Net per vehicle | p | Samples |
|---|---|---|---|---|---|---|---|---|---|
| dist (R = 401) vs mean (R = 400) | +0.7% | 0.058 | -2.6% | 0.001 | -3.4% | 0.00017 | +4.0% | 2.6e-08 | 291 vs 292 |
| dist (R = 401) vs fixorder (R = 400) | +0.3% | 0.31 | +3.4% | 0.00067 | +1.7% | 0.096 | +3.7% | 4.7e-08 | 291 vs 291 |
| dist (R = 401) vs ordermean (R = 400) | +1.5% | 0.00061 | -5.4% | 8e-08 | -4.9% | 8.3e-07 | +1.9% | 0.00011 | 291 vs 292 |

