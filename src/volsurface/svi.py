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
