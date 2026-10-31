"""Surface SVI (Gatheral & Jacquier 2014) with power-law curvature.

w(k, theta) = theta / 2 * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2))
phi(theta)  = eta / (theta^gamma * (1 + theta)^(1 - gamma))

With eta * (1 + |rho|) <= 2, 0 < gamma <= 1/2 and theta non-decreasing in T,
the surface is free of static arbitrage (Gatheral & Jacquier, Remark 4.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import least_squares


def ssvi_total_variance(k, theta, rho, eta, gamma):
    k, theta = np.asarray(k, dtype=float), np.asarray(theta, dtype=float)
    phi = eta / (theta**gamma * (1 + theta) ** (1 - gamma))
    return 0.5 * theta * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))


@dataclass(frozen=True)
class SSVIParams:
    rho: float
    eta: float
    gamma: float

    def total_variance(self, k, theta):
        return ssvi_total_variance(k, theta, self.rho, self.eta, self.gamma)

    @property
    def is_arbitrage_free(self) -> bool:
        return abs(self.rho) < 1 and self.eta > 0 and 0 < self.gamma <= 0.5 \
            and self.eta * (1 + abs(self.rho)) <= 2 + 1e-12


def fit_ssvi(k, theta, w, weights=None) -> SSVIParams:
    """Fit (rho, eta, gamma) given each quote's ATM total variance theta.

    eta is reparameterised as eta = s * 2 / (1 + |rho|) with s in (0, 1], so
    every candidate the optimiser tries satisfies the no-arbitrage bound.
    """
    k, theta, w = (np.asarray(x, dtype=float) for x in (k, theta, w))
    weights = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)
    if not weights.mean() > 0:  # all-zero or NaN weights (e.g. vanishing vega): fall back to equal
        weights = np.ones_like(w)
    sw = np.sqrt(weights / weights.mean())

    def to_params(x):
        rho, s, gamma = x
        return rho, s * 2 / (1 + abs(rho)), gamma

    def residuals(x):
        return sw * (ssvi_total_variance(k, theta, *to_params(x)) - w)

    lb, ub = np.array([-0.999, 1e-3, 0.01]), np.array([0.999, 1.0, 0.5])
    best = None
    for rho0, s0, g0 in product((-0.7, -0.3, 0.0), (0.3, 0.7), (0.25, 0.45)):
        sol = least_squares(residuals, [rho0, s0, g0], bounds=(lb, ub), method="trf")
        if best is None or sol.cost < best.cost:
            best = sol
    return SSVIParams(*map(float, to_params(best.x)))


@dataclass(frozen=True)
class ESSVIParams:
    """Extended SSVI (Hendriks & Martini 2019): the skew rho varies with theta.

    rho(theta) = rho0 + (rho1 - rho0) * sqrt(min(theta / theta_ref, 1))

    Each slice is butterfly-free when eta * (1 + max(|rho0|, |rho1|)) <= 2 and
    0 < gamma <= 1/2. Calendar arbitrage is no longer excluded by construction
    once rho varies; the wings need psi * (1 +- rho) non-decreasing in theta,
    psi = theta * phi(theta). fit_essvi penalises it; check_surface reports it.
    """

    rho0: float
    rho1: float
    eta: float
    gamma: float
    theta_ref: float

    def rho(self, theta):
        u = np.sqrt(np.clip(np.asarray(theta, dtype=float) / self.theta_ref, 0.0, 1.0))
        return self.rho0 + (self.rho1 - self.rho0) * u

    def total_variance(self, k, theta):
        return ssvi_total_variance(k, theta, self.rho(theta), self.eta, self.gamma)

    @property
    def is_butterfly_free(self) -> bool:
        rho_max = max(abs(self.rho0), abs(self.rho1))
        return rho_max < 1 and self.eta > 0 and 0 < self.gamma <= 0.5 \
            and self.eta * (1 + rho_max) <= 2 + 1e-12


def fit_essvi(k, theta, w, weights=None, start: SSVIParams | None = None) -> ESSVIParams:
    """Fit (rho0, rho1, eta, gamma) given each quote's ATM total variance theta.

    eta = s * 2 / (1 + max(|rho0|, |rho1|)), s in (0, 1], keeps every slice
    butterfly-free. A decrease of w in theta on a (k, theta) grid is penalised.
    The fit starts from the SSVI solution (rho0 = rho1, zero penalty) and the
    optimiser only accepts cost-reducing steps, so it never fits worse than SSVI.
    """
    k, theta, w = (np.asarray(x, dtype=float) for x in (k, theta, w))
    weights = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)
    if not weights.mean() > 0:  # all-zero or NaN weights (e.g. vanishing vega): fall back to equal
        weights = np.ones_like(w)
    sw = np.sqrt(weights / weights.mean())
    start = start or fit_ssvi(k, theta, w, weights)
    theta_ref = float(theta.max())
    k_grid = np.linspace(-3.0, 3.0, 61)[:, None]
    theta_grid = np.linspace(theta_ref / 40, theta_ref, 40)  # beyond theta_ref rho is constant (SSVI)

    def to_params(x):
        rho0, rho1, s, gamma = map(float, x)
        return ESSVIParams(rho0, rho1, s * 2 / (1 + max(abs(rho0), abs(rho1))), gamma, theta_ref)

    def residuals(x):
        p = to_params(x)
        dw = np.diff(p.total_variance(k_grid, theta_grid), axis=1)
        return np.append(sw * (p.total_variance(k, theta) - w), 1e3 * max(0.0, -dw.min()))

    s0 = np.clip(start.eta * (1 + abs(start.rho)) / 2, 1e-3, 1.0)
    x0 = [start.rho, start.rho, s0, np.clip(start.gamma, 0.01, 0.5)]
    lb, ub = np.array([-0.999, -0.999, 1e-3, 0.01]), np.array([0.999, 0.999, 1.0, 0.5])
    return to_params(least_squares(residuals, x0, bounds=(lb, ub), method="trf").x)
