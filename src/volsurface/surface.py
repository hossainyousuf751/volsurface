"""Volatility surface objects: forward curve, interpolation, Dupire local vol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict

import numpy as np
from scipy.interpolate import PchipInterpolator

from volsurface.ssvi import ESSVIParams, SSVIParams
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
        nodes = {"expiries": expiries, "forwards": forwards, "discounts": discounts}
        self._nodes = {name: [float(x) for x in np.atleast_1d(v)] for name, v in nodes.items()}
        self._t = np.concatenate([[0.0], np.asarray(expiries, dtype=float)])
        self._ln_f = np.concatenate([[np.log(spot)], np.log(forwards)])
        self._ln_df = np.concatenate([[0.0], np.log(discounts)])

    def forward(self, T):
        return np.exp(_interp_extrap(T, self._t, self._ln_f))

    def discount(self, T):
        return np.exp(_interp_extrap(T, self._t, self._ln_df))

    def to_dict(self) -> dict:
        """JSON-safe nodes; from_dict rebuilds an identical curve."""
        return {"spot": self.spot, **self._nodes}

    @classmethod
    def from_dict(cls, d: dict) -> ForwardCurve:
        return cls(d["spot"], d["expiries"], d["forwards"], d["discounts"])


class VolSurface(ABC):
    """A surface defined by total implied variance w(k, T), k = log(K / F(T)).

    k_range is the log-moneyness range of the quotes the surface was fitted to
    (set by fit_surface); local_vol(extrapolate="flat") holds local vol flat beyond it.
    """

    k_range: tuple[float, float] | None = None

    def __init__(self, curve: ForwardCurve):
        self.curve = curve
        self.n_local_vol = 0  # evaluations through local_vol's safe path (floor / cap / extrapolate)
        self.n_clamped = 0  # ... of which extrapolated, NaN-replaced or clipped

    @abstractmethod
    def total_variance(self, k, T): ...

    def implied_vol(self, K, T):
        T = np.asarray(T, dtype=float)
        k = np.log(np.asarray(K, dtype=float) / self.curve.forward(T))
        return np.sqrt(self.total_variance(k, T) / T)

    def local_vol(self, K, T, dk: float = 1e-3, dT: float = 1e-4, *, floor: float | None = None,
                  cap: float | None = None, extrapolate: str | None = None):
        """Dupire local volatility via Gatheral's formula in total variance.

        sigma_loc^2 = (dw/dT) / [1 - (k/w) w_k + 1/4 (-1/4 - 1/w + k^2/w^2) w_k^2 + 1/2 w_kk]

        By default returns NaN where the numerator or denominator is non-positive,
        which is exactly where the surface admits calendar or butterfly arbitrage.

        Safe evaluation for simulation, used when any keyword is given: with
        extrapolate="flat", k is clamped to k_range so local vol is flat in k
        beyond the fitted quotes (the far wings are extrapolation, and Dupire
        there can explode); NaN falls back to the implied vol at that point; the
        result is clipped to [floor, cap]. Every evaluation is counted in
        n_local_vol and every one any of these steps changed in n_clamped, so a
        simulation can report how much of it ran on the guard rails.
        """
        if floor is None and cap is None and extrapolate is None:
            return self._dupire(K, T, dk, dT)
        if extrapolate not in (None, "flat"):
            raise ValueError(f"extrapolate must be None or 'flat', got {extrapolate!r}")
        K, T = np.broadcast_arrays(np.asarray(K, dtype=float), np.asarray(T, dtype=float))
        changed = np.zeros(K.shape, dtype=bool)
        if extrapolate == "flat":
            if self.k_range is None:
                raise ValueError("extrapolate='flat' needs k_range (set by fit_surface, or assign it)")
            F = self.curve.forward(T)
            k = np.log(K / F)
            k_in = np.clip(k, *self.k_range)
            changed |= k_in != k
            K = F * np.exp(k_in)
        lv = self._dupire(K, T, dk, dT)
        bad = ~np.isfinite(lv)
        lv = np.where(bad, self.implied_vol(K, T), lv)
        clipped = np.clip(lv, -np.inf if floor is None else floor, np.inf if cap is None else cap)
        changed |= bad | (clipped != lv)
        self.n_local_vol += changed.size
        self.n_clamped += int(changed.sum())
        return clipped

    def shifted(self, parallel: float = 0.0, skew: float = 0.0, expiries=None) -> ShiftedSurface:
        """Scenario surface with implied vol sigma(k, T) + parallel + skew * k.

        parallel is in vol units (0.05 = +5 vol points); skew < 0 raises put-wing
        vols (k < 0) and lowers call-wing vols. Raises ValueError if the result has
        a non-positive vol or static arbitrage at the expiries (default: the
        surface's own) on k in [-1, 1].
        """
        return ShiftedSurface(self, parallel, skew, expiries)

    def _dupire(self, K, T, dk, dT):
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
    """Per-expiry SVI smiles, a monotone cubic (PCHIP) in T at fixed k between expiries.

    PCHIP keeps w(k, .) non-decreasing wherever the input slices are, so it adds
    no calendar arbitrage, and makes w continuously differentiable in T, so the
    local vol has no jumps at the expiries. Before the first expiry and after the
    last, implied vol is held constant in T at fixed k.
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
        coef = PchipInterpolator(t, stacked, axis=0).c  # (4, n_exp - 1, *shape), powers of (T - t[idx])
        c = np.take_along_axis(coef, idx[None, None, ...], 1)[:, 0]
        x = T - t[idx]
        inner = ((c[0] * x + c[1]) * x + c[2]) * x + c[3]
        before = stacked[0] * T / t[0]
        after = stacked[-1] * T / t[-1]
        return np.where(T < t[0], before, np.where(T > t[-1], after, inner))


class SSVISurface(VolSurface):
    """SSVI surface with ATM total variance theta(T) a monotone cubic (PCHIP) in T.

    PCHIP keeps theta non-decreasing, so the surface stays free of calendar
    arbitrage, and makes it continuously differentiable, so dw/dT and the
    Dupire local vol have no jumps at the expiries. Beyond the last expiry
    theta continues along its end tangent. Also serves eSSVI: params only needs
    total_variance(k, theta).
    """

    def __init__(self, curve: ForwardCurve, params: SSVIParams | ESSVIParams, expiries, thetas):
        super().__init__(curve)
        self.params = params
        self.expiries = np.asarray(expiries, dtype=float)
        self._thetas = [float(x) for x in thetas]
        t = np.concatenate([[0.0], self.expiries])
        theta = np.concatenate([[0.0], np.maximum.accumulate(np.asarray(thetas, dtype=float))])
        self._pchip = PchipInterpolator(t, theta, extrapolate=False)
        self._t_end, self._theta_end = t[-1], theta[-1]
        self._slope_end = float(self._pchip.derivative()(t[-1]))

    def theta(self, T):
        T = np.asarray(T, dtype=float)
        inside = self._pchip(np.clip(T, 0.0, self._t_end))
        after = self._theta_end + self._slope_end * (T - self._t_end)
        return np.maximum(np.where(T > self._t_end, after, inside), 1e-12)

    def total_variance(self, k, T):
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        return self.params.total_variance(k, self.theta(T))

    def to_dict(self) -> dict:
        """JSON-safe form (plain floats and lists), e.g. to save a fitted SPX snapshot."""
        return {"curve": self.curve.to_dict(), "params_type": type(self.params).__name__,
                "params": asdict(self.params), "expiries": self.expiries.tolist(), "thetas": self._thetas,
                "k_range": None if self.k_range is None else list(self.k_range)}

    @classmethod
    def from_dict(cls, d: dict) -> SSVISurface:
        params_cls = {"SSVIParams": SSVIParams, "ESSVIParams": ESSVIParams}[d["params_type"]]
        surf = cls(ForwardCurve.from_dict(d["curve"]), params_cls(**d["params"]), d["expiries"], d["thetas"])
        if d.get("k_range") is not None:
            surf.k_range = tuple(d["k_range"])
        return surf


class ShiftedSurface(VolSurface):
    """A scenario surface: implied vol of `base` plus parallel + skew * k (see VolSurface.shifted).

    Validated at construction on k in [-1, 1] at `expiries`. Beyond that range a
    steep skew can drive the vol to zero; it is floored at zero there rather than
    going negative.
    """

    def __init__(self, base: VolSurface, parallel: float = 0.0, skew: float = 0.0, expiries=None):
        super().__init__(base.curve)
        self.base, self.parallel, self.skew = base, float(parallel), float(skew)
        self.k_range = base.k_range
        if expiries is None:
            if not hasattr(base, "expiries"):
                raise ValueError("base surface has no expiries; pass expiries to validate the shift on")
            expiries = base.expiries
        self.expiries = np.sort(np.asarray(expiries, dtype=float))

        from volsurface.arbitrage import check_surface  # arbitrage imports this module

        k = np.linspace(-1.0, 1.0, 401)[:, None]
        if np.any(self._vol(k, self.expiries[None, :]) <= 0):
            raise ValueError(f"shift (parallel={parallel}, skew={skew}) gives a non-positive vol on |k| <= 1")
        report = check_surface(self, self.expiries)
        if not report.is_free:
            raise ValueError(f"shifted surface admits static arbitrage: {report.summary()}")

    def _vol(self, k, T):
        T = np.maximum(T, 1e-12)
        return np.sqrt(self.base.total_variance(k, T) / T) + self.parallel + self.skew * k

    def total_variance(self, k, T):
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        return np.maximum(self._vol(k, T), 0.0) ** 2 * T
