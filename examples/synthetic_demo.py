"""Fit the pipeline to a noisy synthetic chain where the true surface is known."""

from pathlib import Path

import pandas as pd

from volsurface import ESSVIParams, fit_errors, fit_surface, synthetic_chain
from volsurface.plotting import plot_smiles, plot_vol_heatmap

out = Path(__file__).resolve().parent.parent / "docs"
out.mkdir(exist_ok=True)

quotes, truth = synthetic_chain(noise=0.003, seed=7)
fit = fit_surface(quotes, truth["spot"])

print(f"Quotes: {len(quotes)} raw -> {len(fit.table)} OTM implied vols across {len(fit.expiries)} expiries\n")
p, t = fit.ssvi_params, truth["params"]
print("SSVI parameters (fitted vs true):")
print(f"  rho   {p.rho:+.4f}  vs {t.rho:+.4f}")
print(f"  eta   {p.eta:.4f}   vs {t.eta:.4f}")
print(f"  gamma {p.gamma:.4f}   vs {t.gamma:.4f}")
print(f"  arbitrage-free by construction: {p.is_arbitrage_free}\n")
print("Static arbitrage checks:")
print(f"  SVI slices: {fit.svi_arbitrage.summary()}")
print(f"  SSVI:       {fit.ssvi_arbitrage.summary()}")
print(f"  eSSVI:      {fit.essvi_arbitrage.summary()}\n")
with pd.option_context("display.float_format", "{:.3f}".format):
    print(fit_errors(fit).to_string(index=False))

# eSSVI: a chain whose skew flattens with maturity, which constant-rho SSVI cannot fit
skew_true = ESSVIParams(rho0=-0.85, rho1=-0.45, eta=1.0, gamma=0.4, theta_ref=0.05)
skew_quotes, skew_truth = synthetic_chain(params=skew_true, noise=0.003, seed=7)
skew_fit = fit_surface(skew_quotes, skew_truth["spot"])
e, err = skew_fit.essvi_params, fit_errors(skew_fit)
print("\nMaturity-dependent skew, eSSVI (fitted vs true):")
print(f"  rho0  {e.rho0:+.4f}  vs {skew_true.rho0:+.4f}")
print(f"  rho1  {e.rho1:+.4f}  vs {skew_true.rho1:+.4f}")
print(f"  eSSVI: {skew_fit.essvi_arbitrage.summary()}")
for name in ("ssvi", "essvi"):
    print(f"  {name:5s} mean RMSE {err[f'{name}_rmse_volpts'].mean():.3f} vol pts, "
          f"{err[f'{name}_in_spread_pct'].mean():.0f}% inside bid-ask")

plot_smiles(fit, out / "smiles.png", title="SVI and SSVI fits to a noisy synthetic chain")
plot_vol_heatmap(fit, out / "implied_vol.png", kind="implied")
plot_vol_heatmap(fit, out / "local_vol.png", kind="local", title="Dupire local volatility from the SSVI surface")
print(f"\nPlots written to {out}")
