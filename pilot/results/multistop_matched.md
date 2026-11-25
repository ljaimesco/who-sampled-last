# Multi-stop: expenditure approximately matched (held-out)

dist's reward is calibrated on seeds never evaluated so its payout matches mean-time's at R = 400; other estimators keep R = 400.

## LONDON (calibrated dist reward 381.1, 20 evaluation seeds)

| Comparison | Payout difference | p | Time past deadline | p | Longest gap | p | Net per vehicle | p | Samples |
|---|---|---|---|---|---|---|---|---|---|
| dist (R = 381) vs mean (R = 400) | +0.7% | 0.19 | -21.7% | 1.9e-06 | -5.6% | 0.00017 | -4.3% | 5.7e-06 | 579 vs 546 |
| dist (R = 381) vs fixorder (R = 400) | +0.6% | 0.41 | -19.4% | 1.9e-06 | -4.7% | 0.00013 | -4.7% | 1.9e-06 | 579 vs 546 |
| dist (R = 381) vs ordermean (R = 400) | -5.5% | 1.9e-06 | +4.2% | 0.014 | -0.2% | 0.9 | -4.8% | 1.9e-06 | 579 vs 584 |

## SUMO (calibrated dist reward 393.5, 30 evaluation seeds)

| Comparison | Payout difference | p | Time past deadline | p | Longest gap | p | Net per vehicle | p | Samples |
|---|---|---|---|---|---|---|---|---|---|
| dist (R = 394) vs mean (R = 400) | -0.4% | 0.44 | -7.3% | 7.8e-05 | -4.7% | 7.9e-05 | +4.5% | 1.6e-07 | 568 vs 557 |
| dist (R = 394) vs fixorder (R = 400) | +0.3% | 0.82 | -4.1% | 0.055 | -1.3% | 0.35 | +1.7% | 0.038 | 568 vs 554 |
| dist (R = 394) vs ordermean (R = 400) | -1.0% | 0.0099 | -6.7% | 0.00017 | -3.0% | 0.00023 | +3.2% | 0.00011 | 568 vs 576 |

