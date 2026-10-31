"""Fit the pipeline to a noisy synthetic chain where the true surface is known."""

from pathlib import Path

import pandas as pd

from volsurface import fit_errors, fit_surface, synthetic_chain
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
print(f"  SSVI:       {fit.ssvi_arbitrage.summary()}\n")
with pd.option_context("display.float_format", "{:.3f}".format):
    print(fit_errors(fit).to_string(index=False))

plot_smiles(fit, out / "smiles.png", title="SVI and SSVI fits to a noisy synthetic chain")
plot_vol_heatmap(fit, out / "implied_vol.png", kind="implied")
plot_vol_heatmap(fit, out / "local_vol.png", kind="local", title="Dupire local volatility from the SSVI surface")
print(f"\nPlots written to {out}")
