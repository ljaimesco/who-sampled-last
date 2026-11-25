# SUMO: single-stop vs multi-stop

30 paired seeds.

| Estimator | Time past deadline (single to multi) | Longest gap | Net per vehicle | Payout | Samples (multi) |
|---|---|---|---|---|---|
| dist | 0.295 to 0.159 | 473 to 367 | 222 to 276 | 78432 to 113326 | 569 |
| ordermean | 0.315 to 0.169 | 499 to 378 | 217 to 262 | 76949 to 112552 | 576 |
| fixorder | 0.288 to 0.165 | 466 to 371 | 213 to 266 | 77960 to 111166 | 554 |
| mean | 0.306 to 0.170 | 491 to 384 | 213 to 259 | 77655 to 111970 | 557 |

Paired Wilcoxon within multi-stop:

| Comparison | Deadline | p | Longest gap | p | Net per vehicle | p | Payout | p |
|---|---|---|---|---|---|---|---|---|
| dist vs mean | -6.7% | 9.2e-06 | -4.5% | 1.4e-05 | +6.5% | 1.9e-09 | +1.2% | 0.0058 |
| dist vs fixorder | -3.5% | 0.055 | -1.1% | 0.36 | +3.6% | 9.2e-06 | +1.9% | 0.0012 |
| dist vs ordermean | -6.2% | 3.5e-05 | -2.8% | 0.00015 | +5.1% | 9.3e-09 | +0.7% | 0.12 |
