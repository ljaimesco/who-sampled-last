"""Certificates of non-existence of a pure-strategy equilibrium, with guaranteed bounds.

See results/CERTIFICATE.md for the derivation. Summary:
  * every expected utility is enclosed in [U_lo, U_hi] by monotone Riemann sums
    over exact normal-probability cells (no sampling, no extrapolation);
  * the omitted normal tails |z| > Z are bounded by 0 (lower) and 1 (upper),
    with their mass computed directly as 2 * Phi(-Z), not by subtraction;
  * an explicit floating-point allowance FP_REL * R is subtracted from every
    lower bound and added to every upper bound;
  * a profile is certified if some feasible unilateral deviation has
    U_lo(deviation) - U_hi(current) > 0; uncertified profiles are retried on
    finer grids.
The claim covers pure strategies within the modeled finite action sets."""
import csv, itertools, os, sys, time
import numpy as np
from scipy.special import ndtr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exact_small import ExactGame
from small_games import make
from spa_uav import OPT_OUT
from spa_uav.stochastic import lognorm_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
Z = 6.5
FP_REL = 1e-9          # >= 100x the rounding bound derived in CERTIFICATE.md (about 1e-11 relative)
GRIDS = ((1500, 1500), (6000, 3000), (20000, 6000))
CHUNK = 1000


class Encloser:
    def __init__(self, inst, cv, R, nz, nx, v=10.0):
        psi, self.tau, self.last, self.tdep, dop, self.cost = inst
        self.c, self.R, self.nx = dop / v, float(R), nx
        self.s, self.mu = lognorm_params(cv)
        self.zg = np.linspace(-Z, Z, nz + 1)
        self.p = ndtr(self.zg[1:]) - ndtr(self.zg[:-1])     # exact cell probabilities (to rounding)
        self.tail = 2.0 * ndtr(-Z)                           # omitted mass, computed directly
        self.cache = {}

    def F(self, j, k, x):
        """P(rival j has reached k by time x): lognormal travel time, 0 before departure."""
        t = x - self.tdep[j]
        out = np.zeros_like(t)
        ok = t > 0
        out[ok] = ndtr((np.log(t[ok] / self.c[j, k]) + self.mu) / self.s)
        return out

    def _sum(self, k, rivals, a_end, a_ref, right):
        """Monotone Riemann sum of f(x) = (1/tau) e^{-(a_ref-x)/tau} prod_j [F_j(x) + 1 - F_j(a_ref)] on [L, a_end].
        f is nonnegative and nondecreasing in x, so the right sum bounds the integral above, the left sum below."""
        L, tau, n = self.last[k], self.tau[k], self.nx
        u = (np.arange(1, n + 1) if right else np.arange(0, n)) / n
        x = L + (a_end - L)[:, None] * u[None, :]
        G = np.ones_like(x)
        for j in rivals:
            G *= self.F(j, k, x) + 1.0 - self.F(j, k, a_ref)[:, None]
        f = np.exp(-(a_ref[:, None] - x) / tau) * G / tau
        return (a_end - L) / n * f.sum(1)

    def delta(self, i, k, rivals):
        key = (i, k, rivals)
        if key in self.cache:
            return self.cache[key]
        lo = hi = 0.0
        for s in range(0, len(self.p), CHUNK):
            zg = self.zg[s:s + CHUNK + 1]
            a = self.tdep[i] + self.c[i, k] * np.exp(-self.mu - self.s * zg)   # own arrival, decreasing in z
            a1, a2 = a[1:], a[:-1]                                             # cell [a1, a2]
            hu = np.minimum(self._sum(k, rivals, a2, a1, right=True), 1.0)     # h(a) <= 1 always
            hl = self._sum(k, rivals, a1, a2, right=False)
            p = self.p[s:s + CHUNK]
            lo += float((p * hl).sum()); hi += float((p * hu).sum())
        hi += self.tail
        self.cache[key] = (lo, hi)
        return lo, hi

    def U(self, i, k, prof):
        if k == OPT_OUT:
            return 0.0, 0.0
        rivals = tuple(j for j, kk in enumerate(prof) if kk == k and j != i)
        lo, hi = self.delta(i, k, rivals)
        fp = FP_REL * self.R
        return self.R * lo - self.cost[i, k] - fp, self.R * hi - self.cost[i, k] + fp


def certify(p, g, enc, top=4):
    cands = []
    for i in range(g.N):
        cur = g.utility(i, p[i], p)
        for k in g.options[i]:
            if k != p[i]:
                b = p.copy(); b[i] = k
                cands.append((g.utility(i, k, b) - cur, i, k))
    cands.sort(reverse=True)
    for est, i, k in cands[:top]:
        b = p.copy(); b[i] = k
        lb = enc.U(i, k, b)[0] - enc.U(i, p[i], p)[1]
        if lb > 0:
            return i, k, est, lb
    return None, None, cands[0][0], float("nan")


if __name__ == "__main__":
    lines = []
    for (N, M, cv, R, seed) in ((6, 2, 0.3, 400.0, 271), (5, 3, 0.5, 800.0, 276)):
        t = time.time()
        inst = make(seed, N, M, cv, R, 256)
        g = ExactGame(inst, cv, R, 200, 400)          # point estimates only choose which deviation to certify
        profs = [list(p) for p in itertools.product(*g.options)]
        result = {}
        todo = profs
        for nz, nx in GRIDS:
            enc = Encloser(inst, cv, R, nz, nx)
            left = []
            for p in todo:
                r = certify(p, g, enc)
                if r[0] is None: left.append(p)
                else: result[tuple(p)] = r + (f"{nz}x{nx}",)
            todo = left
            if not todo: break
        for p in todo:
            result[tuple(p)] = (None, None, float("nan"), float("nan"), "none")
        with open(f"{OUT}/certificate_bounds_seed{seed}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["profile (PsI per vehicle, -1 = opt out)", "deviating vehicle", "deviation", "gain estimate", "gain lower bound", "grid"])
            for p in profs:
                i, k, est, lb, gr = result[tuple(p)]
                w.writerow([p, i, k, f"{est:.4f}", f"{lb:.4f}", gr])
        lbs = [r[3] for r in result.values() if r[0] is not None]
        tight = min(result.items(), key=lambda kv: kv[1][3] if kv[1][0] is not None else np.inf)
        lines.append(f"seed {seed} (N={N}, M={M}, cv={cv}, R={R:.0f}): {len(profs)} profiles, {len(lbs)} certified, {len(todo)} uncertified. "
                     f"Smallest guaranteed gain lower bound {min(lbs):.4f} credits at profile {list(tight[0])} (vehicle {tight[1][0]} -> {tight[1][1]}, "
                     f"estimate {tight[1][2]:.4f}, grid {tight[1][4]}). Tail mass {2*ndtr(-Z):.2e}; floating-point allowance {FP_REL*R:.1e} credits per utility. {time.time()-t:.0f} s.")
        print(lines[-1], flush=True)
    open(f"{OUT}/certificate_bounds.md", "w").write("\n\n".join(lines) + "\n")
