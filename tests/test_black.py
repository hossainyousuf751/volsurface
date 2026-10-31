import numpy as np
import pytest

from volsurface import black_price, implied_vol, implied_vols


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("K", [60.0, 95.0, 100.0, 105.0, 150.0])
def test_implied_vol_round_trip(is_call, K):
    F, T, vol, df = 102.0, 0.75, 0.23, 0.97
    price = float(black_price(F, K, T, vol, df, is_call))
    assert implied_vol(price, F, K, T, df, is_call) == pytest.approx(vol, abs=1e-8)


def test_price_outside_bounds_is_nan():
    F, K, T, df = 100.0, 90.0, 1.0, 0.96
    assert np.isnan(implied_vol(df * (F - K) * 0.99, F, K, T, df, True))  # below intrinsic
    assert np.isnan(implied_vol(df * F * 1.01, F, K, T, df, True))        # above upper bound


def test_vectorised_matches_scalar():
    K = np.array([80.0, 100.0, 120.0])
    prices = black_price(100.0, K, 0.5, 0.2, 0.98, False)
    np.testing.assert_allclose(implied_vols(prices, 100.0, K, 0.5, 0.98, False), 0.2, atol=1e-8)
