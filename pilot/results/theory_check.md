# Theory check on London single-stop windows (rho = 0.5)

10 missions, 300 windows; 165224 (estimator, player, PsI) evaluations on 4096 independent draws.

## 1. The moment bounds hold

| Statement | Holds in |
|---|---|
| order term <= predecessor-change probability | 100.00% |
| predecessor-change probability <= Cantelli sum | 100.00% |
| curvature term <= Taylor + Scarf bound | 100.00% |
| 0 <= ordermean error <= conditional-variance bound | 100.00% |

## 2. Size of the terms (mean / 95th percentile, percent of R; dist profiles)

| Term | Exact | Bound |
|---|---|---|
| order (predecessor change), fixorder and mean | 6.16% / 28.4% | 37.41% / 100.0% |
| curvature (gap randomness at fixed order), mean | 1.90% / 6.3% | 12.25% / 36.3% |
| conditional curvature, ordermean | 1.56% / 4.4% | 7.15% / 23.9% |
| predecessor-change probability | 10.94% / 44.7% | |

Estimator error |u_hat - u| (credits as percent of R, each estimator's own profiles):

| Estimator | Mean | 95th pct | Share of evaluations above 2% of R |
|---|---|---|---|
| dist | 0.39% | 1.3% | 1.1% |
| ordermean | 1.60% | 4.6% | 31.5% |
| fixorder | 6.04% | 28.1% | 41.1% |
| mean | 6.97% | 31.4% | 48.6% |

## 3. Best responses (Lemma 1): reversal needs bound(true best) + bound(chosen) >= true margin

A reversal: the estimator's best option is worth at least 0.5% of R less than the true best (opt-out has utility 0).

| Estimator | Player decisions | Reversals | Necessary condition holds (exact terms) | Holds (bounds) | Sufficient condition certifies no reversal | Dominant term in reversals |
|---|---|---|---|---|---|---|
| dist | 2918 | 12 (0.4%) | 0/12 | 0/12 | 100.0% | sampling |
| ordermean | 2918 | 82 (2.8%) | 75/82 | 81/82 | 12.1% | curvature |
| fixorder | 2918 | 367 (12.6%) | 361/367 | 367/367 | 0.1% | order |
| mean | 2918 | 384 (13.2%) | 384/384 | 384/384 | 0.1% | order 349, curvature 35 |

## 4. Windows: larger approximation terms go with instability

Per window and estimator, two indices over all the players' options under the estimator's own profile: the mean exact error term and the mean moment bound (percent of R). Thirds are by rank.

| Estimator | Spearman with regret: exact / bound | Unstable share by third of exact index (low / middle / high) | By third of bound index |
|---|---|---|---|
| ordermean | 0.16 / -0.03 | 9% / 15% / 23% | 12% / 17% / 18% |
| fixorder | 0.26 / 0.34 | 50% / 73% / 82% | 43% / 79% / 83% |
| mean | 0.24 / 0.31 | 55% / 76% / 78% | 49% / 80% / 80% |

## 5. Why predecessor uncertainty is small or large

Exact predecessor-change probability by Var(T_i - T_p) relative to the expected gap (dist profiles, PsIs with a predicted predecessor):

| sd(T_i - T_p) / E(T_i - T_p) | Evaluations | Mean predecessor-change probability | Mean order bound |
|---|---|---|---|
| 0 to 0.25 | 9015 | 0.06 | 0.22 |
| 0.25 to 0.5 | 10305 | 0.07 | 0.33 |
| 0.5 to 1 | 5881 | 0.12 | 0.51 |
| 1 to 2 | 2876 | 0.25 | 0.77 |
| 2 to inf | 2763 | 0.41 | 0.97 |

