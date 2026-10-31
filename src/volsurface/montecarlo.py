"""Correlated Monte Carlo paths under Dupire local volatility.

Each asset is simulated as S(t) = F(t) * X(t), with F the surface's market-implied
forward curve and X a driftless martingale:

    d ln X = -0.5 * sigma^2 dt + sigma dW,    sigma = local_vol(S(t), t)

so forwards are matched by construction. Paths come back as
(n_paths, n_times, n_assets), the layout sppricer products take.
"""

from __future__ import annotations

import numpy as np

from volsurface.surface import VolSurface

MAX_VOL = 5.0
K_GRID = np.linspace(-4.0, 2.0, 1201)  # log-moneyness nodes for local vol, step 0.005


def simulate_local_vol(surfaces: list[VolSurface], times, n_paths: int, correlation=None,
                       steps_per_year: int = 252, seed: int | None = None, antithetic: bool = True,
                       flat_vols=None) -> np.ndarray:
    """Simulate correlated local-vol paths and sample them at `times`.

    Log-Euler steps of at most 1 / steps_per_year, with every time in `times`
    on the grid; sigma is frozen over each step at the step's mid-time. Weekly steps
    bias at-the-money vanillas by about 0.05 per 100 (4 SE at 200k paths); daily is unbiased.
    flat_vols (one per asset) replaces local vol by constants: GBM on the same
    forward curves and the same random numbers, the flat-vol benchmark.
    With antithetic=True the second half of the paths uses -Z.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) == 0 or times[0] <= 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must be positive and strictly increasing")
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even when antithetic=True")
    n_assets = len(surfaces)
    corr = np.eye(n_assets) if correlation is None else np.asarray(correlation, dtype=float)
    chol = np.linalg.cholesky(corr)

    n_grid = np.ceil(np.diff(np.concatenate([[0.0], times])) * steps_per_year).astype(int)
    grid = np.concatenate([[0.0]] + [np.linspace(a, b, n + 1)[1:]
                                     for a, b, n in zip(np.concatenate([[0.0], times[:-1]]), times, n_grid)])
    sample_at = np.cumsum(n_grid)

    rng = np.random.default_rng(seed)
    n_draw = n_paths // 2 if antithetic else n_paths
    fwd = [s.curve.forward for s in surfaces]
    log_x = np.zeros((n_paths, n_assets))
    out = np.empty((n_paths, len(times), n_assets))
    j = 0
    for i, (t0, t1) in enumerate(zip(grid[:-1], grid[1:]), start=1):
        dt, t_mid = t1 - t0, 0.5 * (t0 + t1)
        z = rng.standard_normal((n_draw, n_assets))
        z = (np.concatenate([z, -z]) if antithetic else z) @ chol.T
        if flat_vols is not None:
            sigma = np.broadcast_to(np.asarray(flat_vols, dtype=float), log_x.shape)
        else:
            sigma = np.empty_like(log_x)
            for a, surf in enumerate(surfaces):
                # Local vol at fixed t depends only on k = ln(S / F(t)): evaluate on K_GRID, interpolate.
                # ponytail: flat beyond K_GRID's ends (|k| > 4, 2); NaN (arbitrage corners) falls back to
                # implied vol. Count NaNs if a surface with real arbitrage is ever used.
                strikes = fwd[a](t_mid) * np.exp(K_GRID)
                lv = surf.local_vol(strikes, t_mid)
                lv = np.clip(np.where(np.isfinite(lv), lv, surf.implied_vol(strikes, t_mid)), 0, MAX_VOL)
                k = log_x[:, a] + np.log(fwd[a](t0) / fwd[a](t_mid))
                sigma[:, a] = np.interp(k, K_GRID, lv)
        log_x += -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z
        if j < len(times) and i == sample_at[j]:
            out[:, j] = np.exp(log_x) * np.array([f(times[j]) for f in fwd])
            j += 1
    return out
