"""Volatility surface objects: forward curve, interpolation, Dupire local vol."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from volsurface.ssvi import SSVIParams
from volsurface.svi import SVIParams


def _interp_extrap(x, xp, fp):
    """Linear interpolation, extending the last segment beyond the final node."""
    x = np.asarray(x, dtype=float)
    inside = np.interp(x, xp, fp)
    slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return np.where(x > xp[-1], fp[-1] + slope * (x - xp[-1]), inside)


class ForwardCurve:
    """Forwards and discount factors implied by the option market, log-linear in T."""

    def __init__(self, spot: float, expiries, forwards, discounts):
        self.spot = float(spot)
        self._t = np.concatenate([[0.0], np.asarray(expiries, dtype=float)])
        self._ln_f = np.concatenate([[np.log(spot)], np.log(forwards)])
        self._ln_df = np.concatenate([[0.0], np.log(discounts)])

    def forward(self, T):
        return np.exp(_interp_extrap(T, self._t, self._ln_f))

    def discount(self, T):
        return np.exp(_interp_extrap(T, self._t, self._ln_df))


class VolSurface(ABC):
    """A surface defined by total implied variance w(k, T), k = log(K / F(T))."""

    def __init__(self, curve: ForwardCurve):
        self.curve = curve

    @abstractmethod
    def total_variance(self, k, T): ...

    def implied_vol(self, K, T):
        T = np.asarray(T, dtype=float)
        k = np.log(np.asarray(K, dtype=float) / self.curve.forward(T))
        return np.sqrt(self.total_variance(k, T) / T)

    def local_vol(self, K, T, dk: float = 1e-3, dT: float = 1e-4):
        """Dupire local volatility via Gatheral's formula in total variance.

        sigma_loc^2 = (dw/dT) / [1 - (k/w) w_k + 1/4 (-1/4 - 1/w + k^2/w^2) w_k^2 + 1/2 w_kk]

        Returns NaN where the numerator or denominator is non-positive, which
        is exactly where the surface admits calendar or butterfly arbitrage.
        """
        T = np.asarray(T, dtype=float)
        k = np.log(np.asarray(K, dtype=float) / self.curve.forward(T))
        tv = self.total_variance
        w = tv(k, T)
        w_k = (tv(k + dk, T) - tv(k - dk, T)) / (2 * dk)
        w_kk = (tv(k + dk, T) - 2 * w + tv(k - dk, T)) / dk**2
        t_lo = np.maximum(T - dT, 1e-8)
        w_t = (tv(k, T + dT) - tv(k, t_lo)) / (T + dT - t_lo)
        den = 1 - k / w * w_k + 0.25 * (-0.25 - 1 / w + k**2 / w**2) * w_k**2 + 0.5 * w_kk
        ok = (w_t > 0) & (den > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(ok, np.sqrt(np.where(ok, w_t / den, 1.0)), np.nan)


class FlatVolSurface(VolSurface):
    """Constant volatility: a baseline and a test case (local vol equals implied vol)."""

    def __init__(self, curve: ForwardCurve, vol: float):
        super().__init__(curve)
        self.vol = vol

    def total_variance(self, k, T):
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        return self.vol**2 * T


class SVISliceSurface(VolSurface):
    """Per-expiry SVI smiles, linear in total variance between expiries at fixed k.

    Linear interpolation preserves the absence of calendar arbitrage when the
    input slices are free of it. Before the first expiry and after the last,
    implied vol is held constant in T at fixed k.
    """

    def __init__(self, curve: ForwardCurve, expiries, slices: list[SVIParams]):
        super().__init__(curve)
        order = np.argsort(expiries)
        self.expiries = np.asarray(expiries, dtype=float)[order]
        self.slices = [slices[i] for i in order]

    def total_variance(self, k, T):
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        t = self.expiries
        stacked = np.stack([s.total_variance(k) for s in self.slices])  # (n_exp, *shape)
        idx = np.clip(np.searchsorted(t, T) - 1, 0, len(t) - 2)
        t0, t1 = t[idx], t[idx + 1]
        w0 = np.take_along_axis(stacked, idx[None, ...], 0)[0]
        w1 = np.take_along_axis(stacked, (idx + 1)[None, ...], 0)[0]
        inner = w0 + (T - t0) / (t1 - t0) * (w1 - w0)
        before = stacked[0] * T / t[0]
        after = stacked[-1] * T / t[-1]
        return np.where(T < t[0], before, np.where(T > t[-1], after, inner))


class SSVISurface(VolSurface):
    """SSVI surface with ATM total variance theta(T) linear between expiries."""

    def __init__(self, curve: ForwardCurve, params: SSVIParams, expiries, thetas):
        super().__init__(curve)
        self.params = params
        self._t = np.concatenate([[0.0], np.asarray(expiries, dtype=float)])
        self._theta = np.concatenate([[0.0], np.maximum.accumulate(np.asarray(thetas, dtype=float))])

    def theta(self, T):
        return np.maximum(_interp_extrap(T, self._t, self._theta), 1e-12)

    def total_variance(self, k, T):
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        return self.params.total_variance(k, self.theta(T))
