"""Static-arbitrage diagnostics for implied volatility surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from volsurface.surface import VolSurface
from volsurface.svi import SVIParams


def svi_g(params: SVIParams, k):
    """Gatheral's density function g(k); g < 0 means butterfly arbitrage."""
    k = np.asarray(k, dtype=float)
    w, w1, w2 = params.total_variance(k), params.dw(k), params.d2w(k)
    return (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2


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


def check_surface(surface: VolSurface, expiries, k_grid=None, tol: float = 1e-8) -> ArbitrageReport:
    """Check butterfly (g >= 0 per expiry) and calendar (w non-decreasing in T) arbitrage."""
    k_grid = np.linspace(-1.0, 1.0, 401) if k_grid is None else np.asarray(k_grid, dtype=float)
    expiries = np.sort(np.asarray(expiries, dtype=float))
    report = ArbitrageReport(tol=tol)
    for T in expiries:
        report.min_g_by_expiry[float(T)] = float(np.min(_numeric_g(surface, k_grid, T)))
    for t0, t1 in zip(expiries[:-1], expiries[1:]):
        dw = surface.total_variance(k_grid, t1) - surface.total_variance(k_grid, t0)
        for k in k_grid[dw < -tol]:
            report.calendar_violations.append((float(t0), float(t1), float(k)))
    return report
