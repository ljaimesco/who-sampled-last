# London-calibrated correlation sensitivity experiment

ANTE-Mobility London grid (Uber per-edge travel-time means and stds), central 2 km x 2 km, 20 PsIs, single-stop decisions, 30 paired scenario seeds per rho, congestion correlation length 300 s. Intervals: 95% mission-clustered bootstrap.

## 1. Route travel-time marginals (same routes, 4000 draws each)

| rho | Routes | Median relative change in mean vs rho = 0 | in sd | in p10 | in p90 | Median route cv |
|---|---|---|---|---|---|---|
| 0.0 | 1385 | +0.00% | +0.00% | +0.00% | +0.00% | 0.159 |
| 0.5 | 1385 | +0.01% | -0.12% | +0.02% | +0.02% | 0.160 |
| 0.8 | 1385 | +0.02% | +0.07% | -0.01% | +0.04% | 0.159 |

## 2. Dependence relevant to order flips (co-located pairs in each window, decision draws)

| rho | Pairs | Arrival correlation (median, p90) | Var(Ti - Tj) / (Var Ti + Var Tj) (median, p10) |
|---|---|---|---|
| 0.0 | 453771 | -0.000, 0.081 | 1.000, 0.928 |
| 0.5 | 453771 | 0.048, 0.263 | 0.959, 0.744 |
| 0.8 | 453771 | 0.064, 0.434 | 0.945, 0.577 |

## 3. Ex-ante window regret (matched states)

| rho | Estimator | Stable windows | Stable share (95% CI) | Mean regret (95% CI) | P95 regret |
|---|---|---|---|---|---|
| 0.0 | dist | 845/900 | 92.8 to 95.1% | 0.18% (0.15 to 0.21) | 1.3% |
| 0.0 | ordermean | 689/900 | 73.7 to 79.6% | 0.72% (0.62 to 0.81) | 4.0% |
| 0.0 | fixorder | 258/900 | 25.3 to 32.0% | 10.58% (9.89 to 11.31) | 28.4% |
| 0.0 | mean | 241/900 | 23.9 to 29.9% | 10.80% (10.08 to 11.60) | 28.7% |
| 0.5 | dist | 848/900 | 92.7 to 95.7% | 0.15% (0.12 to 0.19) | 1.2% |
| 0.5 | ordermean | 702/900 | 74.9 to 80.9% | 0.64% (0.55 to 0.72) | 3.6% |
| 0.5 | fixorder | 253/900 | 25.1 to 31.0% | 10.40% (9.76 to 11.02) | 28.3% |
| 0.5 | mean | 251/900 | 24.8 to 31.0% | 10.41% (9.81 to 11.02) | 27.7% |
| 0.8 | dist | 842/900 | 92.0 to 95.2% | 0.18% (0.14 to 0.22) | 1.3% |
| 0.8 | ordermean | 705/900 | 75.3 to 81.2% | 0.62% (0.54 to 0.71) | 3.4% |
| 0.8 | fixorder | 256/900 | 24.9 to 32.0% | 10.85% (9.97 to 11.73) | 28.7% |
| 0.8 | mean | 247/900 | 24.6 to 30.6% | 10.85% (10.03 to 11.64) | 28.9% |

## 4. Mission outcomes (paired seeds; mean over missions, 95% bootstrap CI)

| rho | Estimator | Time past deadline | Longest gap (s) | Net per vehicle | Payout | Order inversions |
|---|---|---|---|---|---|---|
| 0.0 | dist | 0.141 (0.129 to 0.153) | 321 (313 to 330) | 242 (237 to 248) | 85116 (83428 to 86795) | 3.7% (3.3% to 4.1%) |
| 0.0 | ordermean | 0.143 (0.132 to 0.156) | 325 (317 to 334) | 243 (237 to 248) | 84945 (83252 to 86639) | 4.1% (3.6% to 4.6%) |
| 0.0 | fixorder | 0.157 (0.143 to 0.170) | 333 (325 to 342) | 234 (229 to 239) | 82735 (81069 to 84285) | 6.0% (5.5% to 6.6%) |
| 0.0 | mean | 0.157 (0.145 to 0.170) | 332 (324 to 340) | 235 (230 to 240) | 82941 (81349 to 84416) | 5.9% (5.4% to 6.3%) |
| 0.5 | dist | 0.139 (0.128 to 0.151) | 321 (313 to 330) | 242 (237 to 248) | 85273 (83635 to 86948) | 3.6% (3.2% to 4.0%) |
| 0.5 | ordermean | 0.144 (0.132 to 0.156) | 327 (320 to 334) | 243 (238 to 248) | 85084 (83478 to 86759) | 3.8% (3.3% to 4.4%) |
| 0.5 | fixorder | 0.156 (0.144 to 0.169) | 331 (323 to 338) | 234 (229 to 239) | 82599 (80977 to 84242) | 6.0% (5.5% to 6.5%) |
| 0.5 | mean | 0.159 (0.145 to 0.172) | 335 (327 to 344) | 234 (229 to 239) | 82566 (81001 to 84103) | 6.0% (5.6% to 6.5%) |
| 0.8 | dist | 0.140 (0.129 to 0.151) | 322 (314 to 331) | 242 (237 to 247) | 85164 (83537 to 86730) | 3.6% (3.1% to 4.0%) |
| 0.8 | ordermean | 0.145 (0.134 to 0.156) | 326 (318 to 334) | 243 (237 to 248) | 85058 (83438 to 86621) | 3.9% (3.5% to 4.4%) |
| 0.8 | fixorder | 0.155 (0.141 to 0.167) | 330 (321 to 338) | 234 (229 to 239) | 82746 (81203 to 84294) | 6.0% (5.5% to 6.5%) |
| 0.8 | mean | 0.160 (0.147 to 0.174) | 336 (327 to 346) | 234 (229 to 239) | 82493 (80931 to 84026) | 6.3% (5.7% to 7.0%) |

Paired Wilcoxon, dist vs mean:

| rho | Time past deadline | p | Net per vehicle | p | Payout | p |
|---|---|---|---|---|---|---|
| 0.0 | -10.1% | 3.7e-09 | +3.2% | 1.9e-09 | +2.6% | 3.7e-09 |
| 0.5 | -12.4% | 3.1e-07 | +3.5% | 3.7e-09 | +3.3% | 3.7e-09 |
| 0.8 | -12.6% | 1.9e-09 | +3.6% | 1.9e-09 | +3.2% | 1.9e-09 |
