import numpy as np
import pytest

from volsurface import (
    ESSVIParams, FlatVolSurface, ForwardCurve, SSVIParams, SSVISurface, SVIParams, SVISliceSurface,
    check_surface, fit_errors, fit_surface, synthetic_chain,
)
from volsurface.synthetic import default_atm_vol

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


def test_ssvi_theta_monotone_and_local_vol_continuous_across_expiries():
    expiries = np.array([0.25, 0.5, 1.0, 2.0])
    thetas = default_atm_vol(expiries) ** 2 * expiries
    surf = SSVISurface(CURVE, SSVIParams(-0.7, 1.0, 0.4), expiries, thetas)
    np.testing.assert_allclose(surf.theta(expiries), thetas, rtol=1e-12)
    assert np.all(np.diff(surf.theta(np.linspace(0.0, 3.0, 601))) >= 0)
    K = np.array([90.0, 100.0, 110.0])
    for T in expiries[1:-1]:  # piecewise-linear theta jumps here (the local-vol bands)
        np.testing.assert_allclose(surf.local_vol(K, T - 1e-3), surf.local_vol(K, T + 1e-3), rtol=5e-3)


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
    assert fit.essvi_arbitrage.is_free
    assert (errors["essvi_rmse_volpts"] < 0.5).all()
    for rho in (fit.essvi_params.rho0, fit.essvi_params.rho1):  # no spurious skew term structure
        assert rho == pytest.approx(truth["params"].rho, abs=0.05)
    assert fit.svi_af_arbitrage.is_free
    assert (errors["svi_af_rmse_volpts"] < 0.5).all()
    pillars = fit.local_vol_surface()
    assert np.all(pillars.expiries[1:-1] / pillars.expiries[:-2] >= 1.5)
    assert check_surface(pillars, pillars.expiries, z_grid=np.linspace(-6, 4, 201)).is_free


def test_svi_slice_surface_hits_slices_and_is_c1_in_T():
    slices = [SVIParams(0.002 * i, 0.1, -0.5, 0.0, 0.2) for i in (1, 2, 4, 8)]
    expiries = [0.25, 0.5, 1.0, 2.0]
    surf = SVISliceSurface(CURVE, expiries, slices)
    k = np.linspace(-0.5, 0.3, 9)
    for T, s in zip(expiries, slices):
        np.testing.assert_allclose(surf.total_variance(k, T), s.total_variance(k), rtol=1e-12)
    K = CURVE.forward(1.0) * np.exp(k)
    np.testing.assert_allclose(surf.local_vol(K, 1.0 - 1e-3), surf.local_vol(K, 1.0 + 1e-3), rtol=5e-3)


def test_essvi_fits_varying_skew_chain_better_than_ssvi():
    true = ESSVIParams(rho0=-0.85, rho1=-0.45, eta=1.0, gamma=0.4, theta_ref=0.05)
    quotes, truth = synthetic_chain(params=true, noise=0.003, seed=7)
    fit = fit_surface(quotes, truth["spot"])
    assert fit.essvi_arbitrage.is_free and fit.essvi_params.is_butterfly_free
    assert fit.essvi_params.rho0 == pytest.approx(true.rho0, abs=0.05)
    assert fit.essvi_params.rho1 == pytest.approx(true.rho1, abs=0.05)
    errors = fit_errors(fit)
    assert errors["essvi_rmse_volpts"].mean() < 0.5 * errors["ssvi_rmse_volpts"].mean()


def test_fit_records_skipped_expiries():
    quotes, truth = synthetic_chain(noise=0.003, seed=7)
    quotes.loc[quotes["T"] == 0.5, "bid"] = 0.0          # whole expiry removed by cleaning
    quotes = quotes[~((quotes["T"] == 1.0) & (quotes["option_type"] == "P"))]  # no parity
    fit = fit_surface(quotes, truth["spot"])
    assert set(fit.skipped) == {0.5, 1.0}
    assert "cleaning" in fit.skipped[0.5] and "parity" in fit.skipped[1.0]
