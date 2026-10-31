import numpy as np
import pytest

from volsurface import SSVIParams, SVIParams, fit_ssvi, fit_svi, svi_g

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


def test_ssvi_constraint_flags_violation():
    assert not SSVIParams(rho=-0.7, eta=1.5, gamma=0.4).is_arbitrage_free
