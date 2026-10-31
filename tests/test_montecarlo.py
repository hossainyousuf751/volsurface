import numpy as np

from volsurface import FlatVolSurface, ForwardCurve, SSVIParams, SSVISurface, black_price
from volsurface.montecarlo import simulate_local_vol
from volsurface.synthetic import default_atm_vol

CURVE = ForwardCurve(100.0, [0.5, 1.0, 2.0], [101.0, 102.0, 104.0], [0.98, 0.96, 0.92])
EXPIRIES = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
SSVI = SSVISurface(CURVE, SSVIParams(-0.7, 1.0, 0.4), EXPIRIES, default_atm_vol(EXPIRIES) ** 2 * EXPIRIES)
K = np.array([70.0, 85.0, 100.0, 115.0, 130.0])


def mc_prices(terminal, K, is_call, df):
    """Undiscounted-then-discounted option prices and antithetic-paired standard errors."""
    payoff = np.maximum(terminal[:, None] - K, 0.0) if is_call else np.maximum(K - terminal[:, None], 0.0)
    half = len(terminal) // 2
    pv = df * 0.5 * (payoff[:half] + payoff[half:])
    return pv.mean(axis=0), pv.std(axis=0, ddof=1) / np.sqrt(half)


def test_local_vol_reprices_the_surface_vanillas():
    times = np.array([0.5, 1.0])
    paths = simulate_local_vol([SSVI], times, 200_000, seed=1)
    for j, T in enumerate(times):
        F, df = CURVE.forward(T), CURVE.discount(T)
        for is_call in (True, False):
            mc, se = mc_prices(paths[:, j, 0], K, is_call, df)
            exact = black_price(F, K, T, SSVI.implied_vol(K, T), df, is_call)
            assert np.all(np.abs(mc - exact) < 4 * se), (T, is_call, mc - exact, se)


def test_flat_surface_is_gbm_and_matches_forwards():
    paths = simulate_local_vol([FlatVolSurface(CURVE, 0.2)], [1.0], 200_000, seed=2)
    mc, se = mc_prices(paths[:, 0, 0], K, True, CURVE.discount(1.0))
    assert np.all(np.abs(mc - black_price(CURVE.forward(1.0), K, 1.0, 0.2, CURVE.discount(1.0))) < 4 * se)
    assert abs(paths[:, 0, 0].mean() / CURVE.forward(1.0) - 1) < 4 * paths[:, 0, 0].std() / 102 / np.sqrt(2e5)


def test_flat_vols_share_random_numbers_and_correlation():
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    kw = dict(times=[0.5, 1.0], n_paths=20_000, correlation=corr, seed=3)
    flat = simulate_local_vol([SSVI, SSVI], flat_vols=[0.2, 0.3], **kw)
    gbm = simulate_local_vol([FlatVolSurface(CURVE, 0.2), FlatVolSurface(CURVE, 0.3)], **kw)
    np.testing.assert_allclose(flat, gbm, rtol=1e-12)
    log_ret = np.log(flat[:, 1] / flat[:, 0])
    assert abs(np.corrcoef(log_ret.T)[0, 1] - 0.8) < 0.02
