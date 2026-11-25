# Multi-stop matched spending (payments at the announced reward): cost accounting and paired intervals

## LONDON (dist at R = 295.0, others at R = 400, 20 seeds; N = all vehicles, mean 296)

| Estimator | Payout | Detour cost | Samples | Pay per sample | Cost per sample | Net per vehicle |
|---|---|---|---|---|---|---|
| dist | 81056 | 22728 | 570 | 142.5 | 40.0 | 197.7 |
| mean | 77872 | 22907 | 546 | 142.9 | 42.0 | 186.8 |
| fixorder | 78472 | 22723 | 546 | 143.9 | 41.7 | 189.2 |

| dist vs | Payout diff (90% CI) | Within +/-2%? | Detour cost diff (95% CI) | Deadline diff (95% CI) | Holm p | Longest gap (95% CI) | Holm p | Net per vehicle (95% CI) | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| mean | +4.1% (+0.2 to +8.1) | no | -0.8% (-7.2 to +4.9) | -17.0% (-22.9 to -11.1) | 0.00022 | -4.5% (-6.7 to -2.2) | 0.0093 | +5.8% (+1.3 to +10.7) | 0.04 |
| fixorder | +3.3% (+1.1 to +5.7) | no | +0.0% (-3.9 to +4.3) | -14.5% (-19.7 to -9.5) | 0.00031 | -3.6% (-5.7 to -1.5) | 0.0095 | +4.5% (+1.5 to +7.5) | 0.019 |

## SUMO (dist at R = 341.4, others at R = 400, 30 seeds; N = all vehicles, mean 298)

| Estimator | Payout | Detour cost | Samples | Pay per sample | Cost per sample | Net per vehicle |
|---|---|---|---|---|---|---|
| dist | 95945 | 26840 | 557 | 172.4 | 48.4 | 231.7 |
| mean | 100683 | 34801 | 557 | 181.0 | 62.6 | 221.1 |
| fixorder | 100852 | 31903 | 554 | 182.3 | 57.8 | 231.4 |

| dist vs | Payout diff (90% CI) | Within +/-2%? | Detour cost diff (95% CI) | Deadline diff (95% CI) | Holm p | Longest gap (95% CI) | Holm p | Net per vehicle (95% CI) | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| mean | -4.7% (-6.4 to -2.9) | no | -22.9% (-25.1 to -20.6) | -2.6% (-5.3 to +0.2) | 0.19 | -2.5% (-4.5 to -0.6) | 0.05 | +4.8% (+2.1 to +7.7) | 0.013 |
| fixorder | -4.9% (-6.8 to -2.7) | no | -15.9% (-18.4 to -13.2) | +0.8% (-2.3 to +4.0) | 1 | +1.0% (-1.2 to +3.1) | 0.79 | +0.1% (-2.6 to +3.1) | 1 |

