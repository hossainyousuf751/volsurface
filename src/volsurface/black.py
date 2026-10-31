"""Black-76 pricing on the forward and robust implied-volatility inversion.

Working on the forward F and discount factor DF (rather than spot, rate and
dividend yield) means the dividend and funding assumptions come from the
option market itself via put-call parity.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

VOL_LO, VOL_HI = 1e-4, 5.0


def _d1(F, K, T, vol):
    return (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))


def black_price(F, K, T, vol, df=1.0, is_call=True):
    """Undiscounted-forward Black-76 price, times the discount factor."""
    F, K, T, vol, df = (np.asarray(x, dtype=float) for x in (F, K, T, vol, df))
    is_call = np.asarray(is_call, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = _d1(F, K, T, vol)
        d2 = d1 - vol * np.sqrt(T)
    call = df * (F * ndtr(d1) - K * ndtr(d2))
    put = df * (K * ndtr(-d2) - F * ndtr(-d1))
    return np.where(is_call, call, put)


def black_vega(F, K, T, vol, df=1.0):
    """dPrice/dVol (per 1.00 of vol)."""
    F, K, T, vol, df = (np.asarray(x, dtype=float) for x in (F, K, T, vol, df))
    d1 = _d1(F, K, T, vol)
    return df * F * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * np.sqrt(T)


def implied_vol(price: float, F: float, K: float, T: float, df: float = 1.0,
                is_call: bool = True) -> float:
    """Implied vol by Brent's method; NaN if the price violates no-arbitrage bounds."""
    intrinsic = df * max(F - K, 0.0) if is_call else df * max(K - F, 0.0)
    upper = df * F if is_call else df * K
    if not np.isfinite(price) or not (intrinsic < price < upper):
        return float("nan")

    def objective(v: float) -> float:
        return float(black_price(F, K, T, v, df, is_call)) - price

    if objective(VOL_LO) > 0 or objective(VOL_HI) < 0:
        return float("nan")
    return brentq(objective, VOL_LO, VOL_HI, xtol=1e-12)


def implied_vols(prices, F, K, T, df=1.0, is_call=True) -> np.ndarray:
    """Element-wise implied vols for broadcastable inputs."""
    arrays = np.broadcast_arrays(*(np.asarray(x) for x in (prices, F, K, T, df, is_call)))
    out = np.empty(arrays[0].shape, dtype=float)
    for idx in np.ndindex(out.shape):
        p, f, k, t, d, c = (a[idx] for a in arrays)
        out[idx] = implied_vol(float(p), float(f), float(k), float(t), float(d), bool(c))
    return out
