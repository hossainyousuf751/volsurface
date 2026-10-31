"""Static-arbitrage diagnostics for implied volatility surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from volsurface.surface import VolSurface
from volsurface.svi import SVIParams


def svi_g(params: SVIParams, k):
    """Gatheral's density function g(k); g < 0 means butterfly arbitrage."""
    return params.g(k)


def _numeric_g(surface: VolSurface, k, T, dk=1e-3):
    tv = surface.total_variance
    w = tv(k, T)
    w1 = (tv(k + dk, T) - tv(k - dk, T)) / (2 * dk)
    w2 = (tv(k + dk, T) - 2 * w + tv(k - dk, T)) / dk**2
    return (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2


@dataclass
class ArbitrageReport:
    min_g_by_expiry: dict[float, float] = field(default_factory=dict)
    calendar_violations: list[tuple[float, float, float]] = field(default_factory=list)
    tol: float = 1e-8

    @property
    def butterfly_free(self) -> bool:
        return all(g >= -self.tol for g in self.min_g_by_expiry.values())

    @property
    def calendar_free(self) -> bool:
        return not self.calendar_violations

    @property
    def is_free(self) -> bool:
        return self.butterfly_free and self.calendar_free

    def summary(self) -> str:
        worst_g = min(self.min_g_by_expiry.values()) if self.min_g_by_expiry else float("nan")
        return (f"butterfly-free: {self.butterfly_free} (min g = {worst_g:.4g}); "
                f"calendar-free: {self.calendar_free} ({len(self.calendar_violations)} violations)")


def check_surface(surface: VolSurface, expiries, k_grid=None, tol: float = 1e-8,
                  z_grid=None) -> ArbitrageReport:
    """Check butterfly (g >= 0 per expiry) and calendar (w non-decreasing in T) arbitrage.

    With z_grid, each expiry is checked on k = z * sqrt(w(0, T)), i.e. in standard
    deviations, and each calendar pair on the shorter expiry's grid.
    """
    k_grid = np.linspace(-1.0, 1.0, 401) if k_grid is None else np.asarray(k_grid, dtype=float)
    expiries = np.sort(np.asarray(expiries, dtype=float))

    def grid(T):
        if z_grid is None:
            return k_grid
        return np.asarray(z_grid, dtype=float) * np.sqrt(float(surface.total_variance(0.0, T)))

    report = ArbitrageReport(tol=tol)
    for T in expiries:
        report.min_g_by_expiry[float(T)] = float(np.min(_numeric_g(surface, grid(T), T)))
    for t0, t1 in zip(expiries[:-1], expiries[1:]):
        kg = grid(t0)
        dw = surface.total_variance(kg, t1) - surface.total_variance(kg, t0)
        for k in kg[dw < -tol]:
            report.calendar_violations.append((float(t0), float(t1), float(k)))
    return report
