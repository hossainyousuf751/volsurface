import numpy as np
import pytest

from volsurface import (
    FlatVolSurface, ForwardCurve, SVIParams, SVISliceSurface, check_surface, fit_errors,
    fit_surface, synthetic_chain,
)

CURVE = ForwardCurve(100.0, [0.5, 1.0, 2.0], [101.0, 102.0, 104.0], [0.98, 0.96, 0.92])


def test_forward_curve_hits_nodes_and_spot():
    assert CURVE.forward(0.0) == pytest.approx(100.0)
    assert CURVE.forward(1.0) == pytest.approx(102.0)
    assert CURVE.discount(2.0) == pytest.approx(0.92)


def test_flat_surface_local_vol_equals_implied_vol():
    surf = FlatVolSurface(CURVE, 0.2)
    K, T = np.meshgrid(np.linspace(70, 140, 15), np.linspace(0.2, 1.8, 9))
    np.testing.assert_allclose(surf.implied_vol(K, T), 0.2, atol=1e-12)
    np.testing.assert_allclose(surf.local_vol(K, T), 0.2, atol=1e-5)


def test_crossed_slices_flag_calendar_arbitrage():
    low = SVIParams(0.01, 0.1, -0.5, 0.0, 0.1)
    high = SVIParams(0.03, 0.1, -0.5, 0.0, 0.1)
    surf = SVISliceSurface(CURVE, [0.5, 1.0], [high, low])  # variance falls with maturity
    report = check_surface(surf, [0.5, 1.0])
    assert not report.calendar_free


def test_end_to_end_on_noisy_synthetic_chain():
    quotes, truth = synthetic_chain(noise=0.003, seed=7)
    fit = fit_surface(quotes, truth["spot"])
    assert fit.ssvi_arbitrage.is_free
    assert fit.ssvi_params.rho == pytest.approx(truth["params"].rho, abs=0.05)
    errors = fit_errors(fit)
    assert (errors["ssvi_rmse_volpts"] < 0.5).all()
    for T, F in truth["forwards"].items():
        if T in fit.svi_slices:
            assert fit.curve.forward(T) == pytest.approx(F, rel=1e-3)
    lv = fit.ssvi_surface.local_vol(np.linspace(80, 120, 9), 1.0)
    assert np.all(np.isfinite(lv)) and np.all(lv > 0)
    assert fit.skipped == {}


def test_fit_records_skipped_expiries():
    quotes, truth = synthetic_chain(noise=0.003, seed=7)
    quotes.loc[quotes["T"] == 0.5, "bid"] = 0.0          # whole expiry removed by cleaning
    quotes = quotes[~((quotes["T"] == 1.0) & (quotes["option_type"] == "P"))]  # no parity
    fit = fit_surface(quotes, truth["spot"])
    assert set(fit.skipped) == {0.5, 1.0}
    assert "cleaning" in fit.skipped[0.5] and "parity" in fit.skipped[1.0]
