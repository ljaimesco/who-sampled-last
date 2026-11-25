# Multi-stop matched spending: cost accounting and paired intervals

## LONDON (dist at R = 381.1, others at R = 400, 20 seeds; N = all vehicles, mean 296)

| Estimator | Payout | Detour cost | Samples | Pay per sample | Cost per sample | Net per vehicle |
|---|---|---|---|---|---|---|
| dist | 110237 | 27385 | 579 | 190.7 | 47.5 | 280.7 |
| mean | 109495 | 22907 | 546 | 200.7 | 42.0 | 293.4 |
| fixorder | 109631 | 22723 | 546 | 200.9 | 41.7 | 294.5 |

| dist vs | Payout diff (90% CI) | Within +/-2%? | Detour cost diff (95% CI) | Deadline diff (95% CI) | Holm p | Longest gap (95% CI) | Holm p | Net per vehicle (95% CI) | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| mean | +0.7% (-0.0 to +1.4) | yes | +19.5% (+12.9 to +25.8) | -21.7% (-26.8 to -17.0) | 1.1e-05 | -5.6% (-7.7 to -3.5) | 0.00027 | -4.3% (-5.3 to -3.3) | 1.7e-05 |
| fixorder | +0.6% (-0.0 to +1.2) | yes | +20.5% (+16.1 to +25.9) | -19.4% (-22.6 to -16.2) | 1.1e-05 | -4.7% (-6.6 to -2.9) | 0.00027 | -4.7% (-5.4 to -4.0) | 1.1e-05 |

## SUMO (dist at R = 393.5, others at R = 400, 30 seeds; N = all vehicles, mean 298)

| Estimator | Payout | Detour cost | Samples | Pay per sample | Cost per sample | Net per vehicle |
|---|---|---|---|---|---|---|
| dist | 111479 | 30766 | 568 | 196.4 | 54.4 | 270.6 |
| mean | 111970 | 34801 | 557 | 201.4 | 62.6 | 258.9 |
| fixorder | 111166 | 31903 | 554 | 201.0 | 57.8 | 266.1 |

| dist vs | Payout diff (90% CI) | Within +/-2%? | Detour cost diff (95% CI) | Deadline diff (95% CI) | Holm p | Longest gap (95% CI) | Holm p | Net per vehicle (95% CI) | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| mean | -0.4% (-1.2 to +0.3) | yes | -11.6% (-13.9 to -9.1) | -7.3% (-9.9 to -4.7) | 0.00039 | -4.7% (-6.7 to -2.7) | 0.00039 | +4.5% (+3.3 to +5.8) | 9.8e-07 |
| fixorder | +0.3% (-0.6 to +1.2) | yes | -3.6% (-6.3 to -0.7) | -4.1% (-7.4 to -0.7) | 0.12 | -1.3% (-3.3 to +0.7) | 0.35 | +1.7% (+0.3 to +3.2) | 0.12 |

