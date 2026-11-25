# Multi-stop routes (up to two PsIs), London-calibrated model, rho = 0.5

20 paired seeds (scenarios 5000+). Multi-stop decisions use 1024 draws; the single-stop comparison is the London experiment on the same seeds (256 draws). Stable = ex-ante regret <= 1% of R; intervals are mission-clustered bootstrap.

## Stability

| Estimator | Single-stop stable | Multi-stop stable (95% CI) | Multi-stop mean regret | P95 |
|---|---|---|---|---|
| dist | 560/600 (93%) | 546/600 (89 to 93%) | 0.25% | 1.9% |
| ordermean | 464/600 (77%) | 414/600 (65 to 73%) | 0.97% | 4.2% |
| fixorder | 169/600 (28%) | 68/600 (9 to 14%) | 12.07% | 31.6% |
| mean | 170/600 (28%) | 59/600 (8 to 12%) | 12.40% | 29.4% |

## Outcomes (mean over seeds)

| Estimator | Routes with two stops | Samples per mission | Time past deadline (single -> multi) | Longest gap | Net per vehicle | Payout |
|---|---|---|---|---|---|---|
| dist | 98% | 581 | 0.133 -> 0.052 | 322 -> 254 | 240 -> 298 | 83802 -> 116319 |
| ordermean | 99% | 584 | 0.137 -> 0.053 | 327 -> 255 | 241 -> 295 | 83697 -> 116629 |
| fixorder | 93% | 546 | 0.148 -> 0.068 | 332 -> 267 | 232 -> 294 | 81203 -> 109631 |
| mean | 93% | 546 | 0.152 -> 0.070 | 335 -> 269 | 232 -> 293 | 81146 -> 109495 |

## Paired Wilcoxon, multi-stop, dist vs each estimator

| Comparison | Time past deadline | p | Longest gap | p | Net per vehicle | p | Payout | p |
|---|---|---|---|---|---|---|---|---|
| dist vs mean | -25.1% | 1.9e-06 | -5.9% | 4.8e-05 | +1.5% | 0.017 | +6.2% | 1.9e-06 |
| dist vs fixorder | -22.9% | 1.9e-06 | -5.0% | 2.7e-05 | +1.1% | 0.011 | +6.1% | 1.9e-06 |
| dist vs ordermean | -0.3% | 0.93 | -0.5% | 0.52 | +1.0% | 0.011 | -0.3% | 0.29 |

Reward cuts (cycle breaking) per mission, dist: 8.2; mean: 37.9.
