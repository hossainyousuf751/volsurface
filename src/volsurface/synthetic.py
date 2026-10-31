"""Synthetic option chains generated from a known SSVI surface.

Used for tests and offline demos: because the true surface is known, every
stage of the pipeline (forwards, implied vols, fitted parameters) can be
checked against the truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface.black import black_price
from volsurface.ssvi import ESSVIParams, SSVIParams

DEFAULT_EXPIRIES = (0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def default_atm_vol(T):
    """Downward-sloping ATM term structure, typical of equity indices in calm markets."""
    return 0.16 + 0.04 * np.exp(-2.0 * np.asarray(T, dtype=float))


def synthetic_chain(spot: float = 100.0, rate: float = 0.04, div_yield: float = 0.015,
                    expiries=DEFAULT_EXPIRIES,
                    params: SSVIParams | ESSVIParams = SSVIParams(-0.7, 1.0, 0.4),
                    atm_vol=default_atm_vol, strike_range=(0.6, 1.4), strike_step: float = 2.5,
                    rel_spread: float = 0.04, tick: float = 0.01, noise: float = 0.0,
                    seed: int | None = 0) -> tuple[pd.DataFrame, dict]:
    """Return (quotes, truth). quotes has columns T, strike, option_type, bid, ask."""
    rng = np.random.default_rng(seed)
    strikes = np.arange(spot * strike_range[0], spot * strike_range[1] + 1e-9, strike_step)
    rows, truth = [], {"spot": spot, "params": params, "forwards": {}, "discounts": {}, "thetas": {}}
    for T in expiries:
        F, DF = spot * np.exp((rate - div_yield) * T), np.exp(-rate * T)
        theta = float(atm_vol(T)) ** 2 * T
        truth["forwards"][T], truth["discounts"][T], truth["thetas"][T] = F, DF, theta
        vols = np.sqrt(params.total_variance(np.log(strikes / F), theta) / T)
        for opt_type, is_call in (("C", True), ("P", False)):
            mid = black_price(F, strikes, T, vols, DF, is_call)
            mid = mid * (1 + noise * rng.standard_normal(mid.shape))
            half = 0.5 * np.maximum(tick, rel_spread * mid)
            bid, ask = mid - half, mid + half
            if tick > 0:
                bid, ask = np.floor(bid / tick) * tick, np.ceil(ask / tick) * tick
            keep = mid >= tick
            rows.append(pd.DataFrame({
                "T": T, "strike": strikes[keep], "option_type": opt_type,
                "bid": np.maximum(bid[keep], 0.0), "ask": ask[keep],
            }))
    return pd.concat(rows, ignore_index=True), truth
