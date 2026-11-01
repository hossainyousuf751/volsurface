"""Raw SVI parameterisation of a single-expiry total-variance smile (Gatheral 2004).

w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2)),  k = log(K / F)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k):
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2))

    def dw(self, k):
        k = np.asarray(k, dtype=float)
        return self.b * (self.rho + (k - self.m) / np.sqrt((k - self.m) ** 2 + self.sigma**2))

    def d2w(self, k):
        k = np.asarray(k, dtype=float)
        return self.b * self.sigma**2 / ((k - self.m) ** 2 + self.sigma**2) ** 1.5

    def g(self, k):
        """Gatheral's density function g(k); g < 0 means butterfly arbitrage."""
        k = np.asarray(k, dtype=float)
        w, w1, w2 = self.total_variance(k), self.dw(k), self.d2w(k)
        return (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2

    @classmethod
    def from_ssvi(cls, theta: float, rho: float, phi: float) -> SVIParams:
        """The raw SVI slice equal to an SSVI slice (Gatheral & Jacquier 2014, Lemma 3.2)."""
        return cls(0.5 * theta * (1 - rho**2), 0.5 * theta * phi, rho, -rho / phi, np.sqrt(1 - rho**2) / phi)

    @property
    def is_valid(self) -> bool:
        """Parameter constraints: positive total variance and Roger Lee's wing bound."""
        return (
            self.b >= 0
            and abs(self.rho) < 1
            and self.sigma > 0
            and self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2) >= 0
            and self.b * (1 + abs(self.rho)) <= 2
        )


def _svi(x, k):
    a, b, rho, m, sigma = x
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def fit_svi(k, w, weights=None) -> SVIParams:
    """Weighted least-squares SVI fit with multiple starting points.

    Constraint violations (negative minimum variance, Lee's bound) are
    penalised so the optimiser stays inside the admissible region.
    """
    k, w = np.asarray(k, dtype=float), np.asarray(w, dtype=float)
    if len(k) < 5:
        raise ValueError("need at least 5 quotes to fit 5 SVI parameters")
    weights = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)
    if not weights.mean() > 0:  # all-zero or NaN weights (e.g. vanishing vega): fall back to equal
        weights = np.ones_like(w)
    sw = np.sqrt(weights / weights.mean())

    def residuals(x):
        a, b, rho, _, sigma = x
        min_var = a + b * sigma * np.sqrt(1 - rho**2)
        penalties = [1e3 * max(0.0, -min_var), 1e3 * max(0.0, b * (1 + abs(rho)) - 2)]
        return np.concatenate([sw * (_svi(x, k) - w), penalties])

    kmin, kmax, wmax = k.min(), k.max(), w.max()
    lb = np.array([-wmax, 1e-8, -0.999, min(2 * kmin, -0.05), 1e-4])
    ub = np.array([wmax, 2.0, 0.999, max(2 * kmax, 0.05), 2.0])

    best = None
    for m0, sigma0, rho0 in product((0.0, 0.5 * kmin, 0.5 * kmax), (0.05, 0.2), (-0.6, 0.0)):
        b0 = 0.1
        a0 = w.min() - b0 * sigma0 * np.sqrt(1 - rho0**2)
        x0 = np.clip([a0, b0, rho0, m0, sigma0], lb + 1e-10, ub - 1e-10)
        sol = least_squares(residuals, x0, bounds=(lb, ub), method="trf", x_scale="jac")
        if best is None or sol.cost < best.cost:
            best = sol
    return SVIParams(*map(float, best.x))


Z_DOMAIN = np.linspace(-6.0, 4.0, 201)  # standard deviations k / sqrt(theta) where arbitrage is excluded


def fit_svi_arbitrage_free(slices: dict, starts: dict, thetas: dict, z=Z_DOMAIN,
                           penalty: float = 1e2, max_escalations: int = 3) -> dict[float, SVIParams]:
    """SVI slices free of static arbitrage on k = z * sqrt(theta), after Gatheral & Jacquier (2014), §5.

    slices maps T -> (k, w, weights); residuals are weights * (w_model - w), so
    weights = 1 / (2 * iv * T) makes them implied-vol errors. starts maps T to an
    arbitrage-free first guess (the SSVI slices). Slices are fitted from the longest
    expiry down, so each shorter slice only has to stay below the next longer one,
    which constrains its wings where it has no quotes. Butterfly (g >= margin) and
    calendar (w <= w_next - margin) violations are penalised; a slice still
    infeasible on a fine grid is refitted with a 10x penalty, up to max_escalations times.
    """
    fitted, nxt = {}, None
    for T in sorted(slices, reverse=True):
        k, w, sw = (np.asarray(x, dtype=float) for x in slices[T])
        root = 1.1 * np.sqrt(thetas[T])  # 10% padding: checks scale z by the fitted ATM variance, not theta
        kg, kf = root * z, root * np.linspace(z[0], z[-1], 10 * len(z))
        w_next, w_next_f = (None, None) if nxt is None else (nxt.total_variance(kg), nxt.total_variance(kf))

        def residuals(x, pen):
            p = SVIParams(*x)
            wg = p.total_variance(kg)
            lee = max(0.0, p.b * (1 + abs(p.rho)) - 2)
            parts = [np.maximum(0, 1e-3 - p.g(kg)), np.maximum(0, -wg), [lee]]
            if w_next is not None:
                parts.append(np.maximum(0, wg - w_next + 1e-3 * thetas[T]))
            return np.concatenate([sw * (p.total_variance(k) - w), pen * np.concatenate(parts)])

        def feasible(p):
            return p.g(kf).min() >= 0 and (w_next_f is None or (p.total_variance(kf) <= w_next_f).all())

        lb, ub = np.array([-1.0, 1e-8, -0.999, -5.0, 1e-4]), np.array([1.0, 5.0, 0.999, 5.0, 5.0])
        s = starts[T]
        x = np.clip([s.a, s.b, s.rho, s.m, s.sigma], lb + 1e-9, ub - 1e-9)
        for i in range(max_escalations + 1):
            x = least_squares(residuals, x, bounds=(lb, ub), x_scale="jac", args=(penalty * 10**i,)).x
            if feasible(SVIParams(*x)):
                break
        fitted[T] = nxt = SVIParams(*map(float, x))
    return dict(sorted(fitted.items()))
