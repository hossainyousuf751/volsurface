"""Pin the README headline numbers to the examples that produce them (same seeds, same inputs).

Fitted parameters come from least-squares optimisers, whose results move by up to ~1e-5 relative
between supported scipy versions (1.10 to 1.17), so pins use rel=1e-4: still far tighter than any
README digit. The penalised multi-start arbitrage-free SVI fit can settle on a slightly different
optimum (NDX median RMSE 0.1790 on scipy 1.17, 0.1815 on 1.10; the README prints 0.18), so that one
figure is pinned to 2%. On the README's environment every value matches to better than 1e-9.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volsurface import ESSVIParams, fit_errors, fit_surface, synthetic_chain

DATA = Path(__file__).resolve().parent.parent / "data"
REL = 1e-4
AF_SVI_REL = 2e-2


def test_synthetic_demo_numbers():
    """examples/synthetic_demo.py: the validation and eSSVI tables."""
    quotes, truth = synthetic_chain(noise=0.003, seed=7)
    fit = fit_surface(quotes, truth["spot"])
    p, err = fit.ssvi_params, fit_errors(fit)
    want = (-0.699293324001155, 1.0019425545571596, 0.39936143625558285)
    assert (p.rho, p.eta, p.gamma) == pytest.approx(want, rel=REL)
    assert err["ssvi_rmse_volpts"].min() == pytest.approx(0.024388033205921564, rel=REL)
    assert err["ssvi_rmse_volpts"].max() == pytest.approx(0.11461133656532353, rel=REL)
    assert err["ssvi_in_spread_pct"].mean() == 100.0
    assert fit.svi_af_arbitrage.is_free and fit.ssvi_arbitrage.is_free and fit.essvi_arbitrage.is_free

    skew = ESSVIParams(rho0=-0.85, rho1=-0.45, eta=1.0, gamma=0.4, theta_ref=0.05)
    sq, st = synthetic_chain(params=skew, noise=0.003, seed=7)
    sfit = fit_surface(sq, st["spot"])
    e, serr = sfit.essvi_params, fit_errors(sfit)
    assert (e.rho0, e.rho1) == pytest.approx((-0.8372976304697973, -0.44715746887568353), rel=REL)
    assert serr["ssvi_rmse_volpts"].mean() == pytest.approx(0.5379891264357894, rel=REL)
    assert serr["essvi_rmse_volpts"].mean() == pytest.approx(0.06260142150707759, rel=REL)
    assert serr["ssvi_in_spread_pct"].mean() == pytest.approx(53.80821754999386, rel=REL)


# examples/real_data_study.py on the saved CBOE snapshots (git-ignored data/, so skipped on CI)
REAL = {
    "SPX": dict(n_iv=10132, n_expiries=47, atm_vol_1m=0.149244461026, atm_vol_1y=0.170937821289,
                ssvi=(-0.547581423679, 1.239809013134, 0.5), essvi=(-0.679479160876, -0.529693862602),
                rmse=(0.198596713241, 2.250536242913, 2.273980318645, 0.201975856298)),
    "NDX": dict(n_iv=3380, n_expiries=36, atm_vol_1m=0.242339106347, atm_vol_1y=0.230770173046,
                ssvi=(-0.463365121551, 1.25908028737, 0.456893228804),
                essvi=(-0.533072973208, -0.439337181311),
                rmse=(0.143245994692, 0.840499873011, 0.73081497633, 0.179014295069)),
}


@pytest.mark.slow
@pytest.mark.parametrize("ticker", sorted(REAL))
def test_real_data_study_numbers(ticker):
    path = DATA / f"{ticker}_20261030.csv"
    if not path.exists():
        pytest.skip(f"{path.name} not saved locally")
    quotes = pd.read_csv(path)
    fit = fit_surface(quotes, float(quotes["spot"].iloc[0]), max_rel_spread=0.3)
    want, err, s, e = REAL[ticker], fit_errors(fit), fit.ssvi_params, fit.essvi_params
    assert (len(fit.table), len(fit.expiries)) == (want["n_iv"], want["n_expiries"])
    atm = [float(fit.ssvi_surface.implied_vol(fit.curve.forward(T), T)) for T in (1 / 12, 1.0)]
    np.testing.assert_allclose(atm, [want["atm_vol_1m"], want["atm_vol_1y"]], rtol=REL)
    np.testing.assert_allclose([s.rho, s.eta, s.gamma], want["ssvi"], rtol=REL)
    np.testing.assert_allclose([e.rho0, e.rho1], want["essvi"], rtol=REL)
    rmse = [err[f"{m}_rmse_volpts"].median() for m in ("svi", "ssvi", "essvi", "svi_af")]
    np.testing.assert_allclose(rmse[:3], want["rmse"][:3], rtol=REL)
    np.testing.assert_allclose(rmse[3], want["rmse"][3], rtol=AF_SVI_REL)
