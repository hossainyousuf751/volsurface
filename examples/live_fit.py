"""Fit a surface to a live Yahoo Finance chain.

Run during US market hours; outside them many bids and asks are zero.
Usage: python examples/live_fit.py ^SPX
"""

import sys
from datetime import date
from pathlib import Path

from volsurface import fit_errors, fit_surface
from volsurface.data import load_yfinance_chain
from volsurface.plotting import plot_smiles, plot_vol_heatmap

ticker = sys.argv[1] if len(sys.argv) > 1 else "^SPX"
quotes, spot = load_yfinance_chain(ticker, max_expiries=10)

root = Path(__file__).resolve().parent.parent
data_dir, docs_dir = root / "data", root / "docs"
data_dir.mkdir(exist_ok=True)
stamp = f"{ticker.strip('^')}_{date.today():%Y%m%d}"
quotes.to_csv(data_dir / f"{stamp}.csv", index=False)  # keep snapshots for reproducibility

fit = fit_surface(quotes, spot, max_rel_spread=0.3)
print(f"{ticker} spot {spot:.2f}: {len(fit.table)} implied vols, {len(fit.expiries)} expiries")
for T, reason in fit.skipped.items():
    print(f"  skipped T={T:.3f}: {reason}")
print(f"SSVI: {fit.ssvi_params}")
print(f"SVI slices: {fit.svi_arbitrage.summary()}")
print(f"SSVI:       {fit.ssvi_arbitrage.summary()}")
print(f"eSSVI: {fit.essvi_params}")
print(f"eSSVI:      {fit.essvi_arbitrage.summary()}")
print(fit_errors(fit).round(3).to_string(index=False))

plot_smiles(fit, docs_dir / f"{stamp}_smiles.png", title=f"{ticker} smiles")
plot_vol_heatmap(fit, docs_dir / f"{stamp}_local_vol.png", kind="local")
