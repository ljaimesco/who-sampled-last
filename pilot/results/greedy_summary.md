# Greedy freshness baseline against the estimators

greedy: predicted reduction in time past deadline per unit of detour, followed as is (hypothetical full compliance at the same payment rule); one-pass reaction: each driver, in random order, best-responds once (dist evaluation) and commits; reaction to equilibrium: best-response dynamics from the greedy profile. Stability of the greedy profiles is scored at the greedy mission's window states; estimators' stability is in the main summaries.

## London model, single stop (30 paired seeds)

| Policy | Stable windows | Median window regret, % of R | Drivers changing | Time past deadline | Longest gap, s | Net per vehicle | Payout | Detours |
|---|---|---|---|---|---|---|---|---|
| dist | see main |  |  | 0.139 | 321 | 242 | 85273 | 297 |
| ordermean | see main |  |  | 0.144 | 327 | 243 | 85084 | 297 |
| fixorder | see main |  |  | 0.156 | 331 | 233 | 82503 | 297 |
| mean | see main |  |  | 0.159 | 335 | 234 | 82538 | 297 |
| greedy (compliance) | 0% | 47.5 | 0% | 0.129 | 329 | 194 | 76628 | 265 |
| greedy + one-pass reaction | 41% | 4.5 | 56% | 0.150 | 332 | 241 | 84152 | 297 |
| greedy + reaction to equilibrium | 93% | 0.0 | 59% | 0.138 | 321 | 243 | 85329 | 297 |

| Paired comparison | Time past deadline | p | Net per vehicle | p | Payout |
|---|---|---|---|---|---|
| greedy vs dist | -7.2% | 0.004 | -20.1% | 1.9e-09 | -10.1% |
| greedy_seq vs dist | +7.8% | 1.9e-08 | -0.6% | 0.0093 | -1.3% |
| greedy_seq vs greedy | +16.2% | 1.6e-07 | +24.3% | 1.9e-09 | +9.8% |
| greedy_br vs dist | -1.3% | 0.1 | +0.4% | 0.1 | +0.1% |
| greedy vs mean | -18.7% | 2.6e-08 | -17.3% | 1.9e-09 | -7.2% |
| greedy_seq vs mean | -5.6% | 0.00051 | +2.8% | 4.7e-08 | +2.0% |

## SUMO, two stops (30 paired seeds)

| Policy | Stable windows | Median window regret, % of R | Drivers changing | Time past deadline | Longest gap, s | Net per vehicle | Payout | Detours |
|---|---|---|---|---|---|---|---|---|
| dist | see main |  |  | 0.159 | 367 | 276 | 113295 | 290 |
| ordermean | see main |  |  | 0.169 | 378 | 263 | 112705 | 292 |
| fixorder | see main |  |  | 0.165 | 371 | 231 | 100852 | 287 |
| mean | see main |  |  | 0.170 | 384 | 221 | 100683 | 287 |
| greedy (compliance) | 0% | 110.3 | 0% | 0.185 | 404 | 178 | 85905 | 240 |
| greedy + one-pass reaction | 49% | 1.2 | 83% | 0.164 | 375 | 275 | 113212 | 290 |
| greedy + reaction to equilibrium | 91% | 0.0 | 83% | 0.160 | 367 | 277 | 113736 | 291 |

| Paired comparison | Time past deadline | p | Net per vehicle | p | Payout |
|---|---|---|---|---|---|
| greedy vs dist | +16.4% | 1.3e-07 | -35.5% | 1.9e-09 | -24.2% |
| greedy_seq vs dist | +3.1% | 0.014 | -0.2% | 0.87 | -0.1% |
| greedy_seq vs greedy | -11.4% | 1.7e-06 | +54.6% | 1.9e-09 | +31.8% |
| greedy_br vs dist | +0.7% | 0.52 | +0.5% | 0.24 | +0.4% |
| greedy vs mean | +8.6% | 0.00017 | -19.6% | 1.9e-09 | -14.7% |
| greedy_seq vs mean | -3.8% | 0.0013 | +24.4% | 1.9e-09 | +12.4% |

## London model, two stops (20 paired seeds)

| Policy | Stable windows | Median window regret, % of R | Drivers changing | Time past deadline | Longest gap, s | Net per vehicle | Payout | Detours |
|---|---|---|---|---|---|---|---|---|
| dist | see main |  |  | 0.052 | 254 | 277 | 110177 | 293 |
| ordermean | see main |  |  | 0.053 | 255 | 281 | 112586 | 294 |
| fixorder | see main |  |  | 0.068 | 267 | 189 | 78472 | 283 |
| mean | see main |  |  | 0.070 | 269 | 187 | 77872 | 282 |
| greedy (compliance) | 0% | 101.4 | 0% | 0.061 | 275 | 171 | 89879 | 245 |
| greedy + one-pass reaction | 16% | 17.2 | 87% | 0.054 | 256 | 292 | 115756 | 294 |
| greedy + reaction to equilibrium | 92% | 0.0 | 89% | 0.052 | 252 | 282 | 112009 | 293 |

| Paired comparison | Time past deadline | p | Net per vehicle | p | Payout |
|---|---|---|---|---|---|
| greedy vs dist | +15.9% | 0.00059 | -38.1% | 1.9e-06 | -18.4% |
| greedy_seq vs dist | +3.2% | 0.11 | +5.4% | 1.9e-06 | +5.1% |
| greedy_seq vs greedy | -10.9% | 0.0042 | +70.4% | 1.9e-06 | +28.8% |
| greedy_br vs dist | -1.0% | 0.7 | +1.9% | 0.048 | +1.7% |
| greedy vs mean | -13.2% | 0.0037 | -8.2% | 0.0094 | +15.4% |
| greedy_seq vs mean | -22.7% | 1.9e-06 | +56.3% | 1.9e-06 | +48.6% |

## SUMO, single stop (30 paired seeds)

| Policy | Stable windows | Median window regret, % of R | Drivers changing | Time past deadline | Longest gap, s | Net per vehicle | Payout | Detours |
|---|---|---|---|---|---|---|---|---|
| dist | see main |  |  | 0.295 | 473 | 223 | 78629 | 291 |
| ordermean | see main |  |  | 0.315 | 499 | 218 | 77147 | 292 |
| fixorder | see main |  |  | 0.288 | 466 | 214 | 78106 | 291 |
| mean | see main |  |  | 0.306 | 491 | 213 | 77785 | 292 |
| greedy (compliance) | 1% | 53.9 | 0% | 0.263 | 482 | 164 | 68467 | 247 |
| greedy + one-pass reaction | 68% | 0.0 | 53% | 0.301 | 476 | 221 | 78092 | 290 |
| greedy + reaction to equilibrium | 89% | 0.0 | 54% | 0.296 | 474 | 223 | 78632 | 291 |

| Paired comparison | Time past deadline | p | Net per vehicle | p | Payout |
|---|---|---|---|---|---|
| greedy vs dist | -10.9% | 2.6e-08 | -26.2% | 1.9e-09 | -12.9% |
| greedy_seq vs dist | +2.0% | 0.013 | -0.8% | 0.15 | -0.7% |
| greedy_seq vs greedy | +14.5% | 1.9e-08 | +34.5% | 1.9e-09 | +14.1% |
| greedy_br vs dist | +0.3% | 0.78 | +0.1% | 0.86 | +0.0% |
| greedy vs mean | -14.1% | 3.7e-09 | -22.9% | 1.9e-09 | -12.0% |
| greedy_seq vs mean | -1.6% | 0.18 | +3.7% | 6.9e-07 | +0.4% |

