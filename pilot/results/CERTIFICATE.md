# Certificate of non-existence of a pure-strategy equilibrium

Code: `pilot/certificate_bounds.py`. Per-profile output: `results/certificate_bounds_seed271.csv`, `results/certificate_bounds_seed276.csv`.

## Claim

For the two small games below, every action profile in the modeled finite action sets admits a feasible unilateral deviation whose gain in expected utility has a strictly positive guaranteed lower bound. Hence neither game has a pure-strategy Nash equilibrium within those action sets. Nothing is claimed about mixed strategies (which exist, since the games are finite) or about larger action sets.

| Game | N vehicles | M PsIs | cv | R | Profiles | Certified | Smallest gain lower bound |
|---|---|---|---|---|---|---|---|
| seed 271 | 6 | 2 | 0.3 | 400 | 486 | 486 | 0.069 credits (6000 x 3000 grid); 0.384 on a 20000 x 6000 grid |
| seed 276 | 5 | 3 | 0.5 | 800 | 1024 | 1024 | 1.661 credits (1500 x 1500 grid) |

## Model (as in `pilot/small_games.py` and `pilot/exact_small.py`)

Vehicle i departs at t_i and reaches PsI k at a_i = t_i + c_ik / F_i, with F_i lognormal (mean 1, coefficient of variation cv), independent across vehicles; write F_i = exp(mu + s Z_i), Z_i standard normal, so a_i is strictly decreasing in Z_i. The last realized sample at k is at L_k < 0 <= t_i, so L_k < a_i always. Rival j reaches k by time x with probability F_j(x) = P(a_j <= x), which is 0 for x <= t_j and nondecreasing in x.

Action set of vehicle i: opt out (utility 0) or any PsI k with c_ik' < R, where c_ik' is the detour cost. A PsI with cost at least R yields utility at most R - cost <= 0 with equality impossible unless delta = 1, so it is weakly dominated by opting out and strictly dominated whenever delta < 1 (always the case here, since a_i - prev is finite); removing strictly dominated actions does not change the set of pure equilibria.

The payment for a sample is R (1 - exp(-(a_i - prev_i)/tau_k)), where prev_i is the latest earlier sample at k: L_k or a rival who chose k and arrived in (L_k, a_i). Given a_i = a, the rivals' arrivals are independent, so

    P(prev_i <= x | a) = G(x; a) = prod_{j in J} [ F_j(x) + 1 - F_j(a) ],   L_k <= x < a,

and the expected multiplier is delta = E_a[ h(a) ] with

    h(a) = integral_{L_k}^{a} f(x; a) dx,   f(x; a) = (1/tau_k) exp(-(a - x)/tau_k) G(x; a).

(Integration by parts of E[1 - exp(-(a - prev)/tau)] over the law of prev given a.) Expected utility: U = R delta - cost (opt out: 0). Also 0 <= h(a) <= 1.

## Monotonicity used (for this exact integrand)

1. In x, for fixed a: exp(-(a - x)/tau) is positive and increasing; each factor F_j(x) + 1 - F_j(a) is nonnegative and nondecreasing because F_j is a CDF. A product of nonnegative nondecreasing functions is nondecreasing. So f(., a) is nonnegative and nondecreasing on [L_k, a].
2. In a, for fixed x: exp(-(a - x)/tau) is decreasing in a; 1 - F_j(a) is nonincreasing in a. So f(x; .) is nonincreasing in a for every x.

## Enclosure of h on a cell a in [a1, a2]

Upper bound. For a in [a1, a2] and x in [L_k, a]: f(x; a) <= f(x; a1) by (2), and the integration range [L_k, a] is contained in [L_k, a2] with a nonnegative integrand, so

    h(a) <= integral_{L_k}^{a2} f(x; a1) dx  <=  right Riemann sum of f(.; a1) on [L_k, a2],

the last step because f(.; a1) is nondecreasing on [L_k, a2] (1 still applies: each factor F_j(x) + 1 - F_j(a1) is nondecreasing in x for all x). The bound is then capped at 1.

Lower bound. For a in [a1, a2]: f(x; a) >= f(x; a2) by (2), and [L_k, a1] is contained in [L_k, a], so

    h(a) >= integral_{L_k}^{a1} f(x; a2) dx  >=  left Riemann sum of f(.; a2) on [L_k, a1].

Both sums use nx equal subintervals.

## Outer expectation

The standard-normal variable Z is cut into nz equal cells on [-6.5, 6.5]. Each cell maps to an interval [a1, a2] of own-arrival times (a is monotone in Z), and its probability p_m = Phi(z_{m+1}) - Phi(z_m) is computed exactly up to rounding. With h_lo,m and h_hi,m the cell bounds above,

    sum_m p_m h_lo,m  <=  delta  <=  sum_m p_m h_hi,m + P(|Z| > 6.5),

using h in [0, 1] on the two omitted tails. The tail mass is computed directly as 2 Phi(-6.5) = 8.03e-11, not as one minus the sum of cell probabilities.

## Floating-point rounding

Each term f is a product of at most 6 factors from exp, log and the normal CDF (library accuracy of a few units in the last place, u = 1.1e-16), so its relative error is below about 20u. The inner sum has nx <= 6000 nonnegative terms and the outer sum at most nz <= 20000; recursive summation of n nonnegative terms has relative error below n u. The total relative error of delta is therefore below (20 + 6000 + 20000) u, about 3e-12, and delta <= 1, so the absolute error of R delta is below 3e-12 R. The code subtracts 1e-9 R (more than 300 times that) from every lower bound and adds it to every upper bound. This relies on the documented accuracy of NumPy's exp/log and SciPy's ndtr; a fully machine-checked certificate would use interval arithmetic (for example Arb), which we have not done.

## Certification rule

For each profile, the candidate deviations are ranked by exact-integration point estimates; for the best few, the guaranteed gain lower bound

    Delta_lo = U_lo(deviation) - U_hi(current)

is computed. The profile is certified as soon as one Delta_lo > 0. Profiles not certified on the 1500 x 1500 grid are retried on 6000 x 3000 and then 20000 x 6000. All 1510 profiles were certified; the CSV files list, for every profile, the deviating vehicle, its deviation, the point estimate of the gain and its guaranteed lower bound.
