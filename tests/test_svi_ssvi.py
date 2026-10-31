import numpy as np
import pytest

from volsurface import ESSVIParams, SSVIParams, SVIParams, fit_essvi, fit_ssvi, fit_svi, svi_g
from volsurface.svi import fit_svi_arbitrage_free

K_GRID = np.linspace(-1.0, 0.6, 41)


def test_svi_fit_recovers_known_smile():
    true = SVIParams(a=0.02, b=0.12, rho=-0.55, m=0.05, sigma=0.15)
    fitted = fit_svi(K_GRID, true.total_variance(K_GRID))
    np.testing.assert_allclose(fitted.total_variance(K_GRID), true.total_variance(K_GRID), atol=1e-6)
    assert fitted.is_valid


def test_well_behaved_svi_has_no_butterfly_arbitrage():
    p = SVIParams(a=0.02, b=0.12, rho=-0.55, m=0.05, sigma=0.15)
    assert svi_g(p, np.linspace(-1.5, 1.5, 601)).min() >= 0


def test_vogt_example_has_butterfly_arbitrage():
    # Axel Vogt's counterexample from Gatheral & Jacquier (2014)
    p = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    assert svi_g(p, np.linspace(-1.5, 1.5, 601)).min() < 0


def test_ssvi_fit_recovers_parameters():
    true = SSVIParams(rho=-0.65, eta=1.1, gamma=0.35)
    thetas = np.array([0.003, 0.01, 0.02, 0.04, 0.08])
    k = np.tile(K_GRID, len(thetas))
    theta = np.repeat(thetas, len(K_GRID))
    fitted = fit_ssvi(k, theta, true.total_variance(k, theta))
    assert fitted.rho == pytest.approx(true.rho, abs=1e-3)
    assert fitted.eta == pytest.approx(true.eta, abs=1e-3)
    assert fitted.gamma == pytest.approx(true.gamma, abs=1e-3)
    assert fitted.is_arbitrage_free


def test_zero_weights_fall_back_to_equal_weights():
    true = SVIParams(a=0.02, b=0.12, rho=-0.55, m=0.05, sigma=0.15)
    w = true.total_variance(K_GRID)
    assert fit_svi(K_GRID, w, weights=np.zeros_like(w)) == fit_svi(K_GRID, w)
    theta = np.full_like(K_GRID, 0.02)
    w_ssvi = SSVIParams(-0.65, 1.1, 0.35).total_variance(K_GRID, theta)
    assert fit_ssvi(K_GRID, theta, w_ssvi, weights=np.zeros_like(w_ssvi)) == fit_ssvi(K_GRID, theta, w_ssvi)


def test_essvi_fit_recovers_varying_skew_and_beats_ssvi():
    true = ESSVIParams(rho0=-0.85, rho1=-0.45, eta=1.0, gamma=0.4, theta_ref=0.05)
    thetas = np.array([0.003, 0.006, 0.01, 0.02, 0.03, 0.05])
    k = np.tile(K_GRID, len(thetas))
    theta = np.repeat(thetas, len(K_GRID))
    w = true.total_variance(k, theta)
    fitted = fit_essvi(k, theta, w)
    for name in ("rho0", "rho1", "eta", "gamma"):
        assert getattr(fitted, name) == pytest.approx(getattr(true, name), abs=1e-3)
    assert fitted.is_butterfly_free
    ssvi_err = np.abs(fit_ssvi(k, theta, w).total_variance(k, theta) - w).max()
    assert np.abs(fitted.total_variance(k, theta) - w).max() < 1e-3 * ssvi_err


def test_essvi_with_constant_rho_is_ssvi():
    k, theta = np.tile(K_GRID, 3), np.repeat([0.01, 0.02, 0.04], len(K_GRID))
    np.testing.assert_allclose(ESSVIParams(-0.6, -0.6, 1.1, 0.35, 0.04).total_variance(k, theta),
                               SSVIParams(-0.6, 1.1, 0.35).total_variance(k, theta), rtol=1e-14)


def test_svi_from_ssvi_is_exact():
    theta, rho, phi = 0.04, -0.6, 3.0
    p = SVIParams.from_ssvi(theta, rho, phi)
    expected = 0.5 * theta * (1 + rho * phi * K_GRID + np.sqrt((phi * K_GRID + rho) ** 2 + 1 - rho**2))
    np.testing.assert_allclose(p.total_variance(K_GRID), expected, rtol=1e-13)


def test_arbitrage_free_svi_uncrosses_slices():
    # Short smile fitted alone would cross the long one: same data, much steeper wings.
    long_true = SVIParams(a=0.03, b=0.10, rho=-0.5, m=0.0, sigma=0.2)
    short_true = SVIParams(a=0.002, b=0.30, rho=-0.5, m=0.0, sigma=0.05)
    k_long, k_short = np.linspace(-0.8, 0.4, 25), np.linspace(-0.25, 0.1, 15)
    slices = {1.0: (k_long, long_true.total_variance(k_long), np.ones(25)),
              0.1: (k_short, short_true.total_variance(k_short), np.ones(15))}
    thetas = {T: float(p.total_variance(0.0)) for T, p in ((1.0, long_true), (0.1, short_true))}
    free_short = fit_svi(*slices[0.1][:2])
    kz = np.sqrt(thetas[0.1]) * np.linspace(-6, 4, 201)
    assert (free_short.total_variance(kz) > long_true.total_variance(kz)).any()   # the problem
    fitted = fit_svi_arbitrage_free(slices, {0.1: free_short, 1.0: long_true}, thetas)
    assert (fitted[0.1].total_variance(kz) <= fitted[1.0].total_variance(kz)).all()
    assert all(fitted[T].g(np.sqrt(thetas[T]) * np.linspace(-6, 4, 201)).min() >= 0 for T in fitted)
    assert np.abs(fitted[1.0].total_variance(k_long) - slices[1.0][1]).max() < 1e-4


def test_ssvi_constraint_flags_violation():
    assert not SSVIParams(rho=-0.7, eta=1.5, gamma=0.4).is_arbitrage_free
