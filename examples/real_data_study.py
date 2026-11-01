"""M3: fit every saved snapshot in data/ and track the surface over time.

Writes data/study.csv (one row per ticker and day), docs/<TICKER>_smiles.png for
the latest SPX snapshot, and docs/study_params.png once there are two or more days.
Usage: python examples/snapshot.py   (daily)   then   python examples/real_data_study.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from volsurface import fit_errors, fit_surface
from volsurface.plotting import plot_smiles

root = Path(__file__).resolve().parent.parent
data_dir, docs_dir = root / "data", root / "docs"
rows, latest, n_raw = [], {}, {}
for path in sorted(data_dir.glob("*_20[0-9][0-9][0-1][0-9][0-3][0-9].csv")):
    ticker, day = path.stem.split("_")
    quotes = pd.read_csv(path)
    fit = fit_surface(quotes, float(quotes["spot"].iloc[0]), max_rel_spread=0.3)
    err = fit_errors(fit)
    s, e = fit.ssvi_params, fit.essvi_params
    row = {"date": pd.Timestamp(day).date(), "ticker": ticker, "spot": quotes["spot"].iloc[0],
           "n_iv": len(fit.table), "n_expiries": len(fit.expiries),
           "atm_vol_1m": float(fit.ssvi_surface.implied_vol(fit.curve.forward(1 / 12), 1 / 12)),
           "atm_vol_1y": float(fit.ssvi_surface.implied_vol(fit.curve.forward(1.0), 1.0)),
           "ssvi_rho": s.rho, "ssvi_eta": s.eta, "ssvi_gamma": s.gamma,
           "essvi_rho0": e.rho0, "essvi_rho1": e.rho1}
    for name in ("svi", "ssvi", "essvi", "svi_af"):
        row[f"{name}_rmse"] = err[f"{name}_rmse_volpts"].median()
        row[f"{name}_in_spread"] = err[f"{name}_in_spread_pct"].mean()
        row[f"{name}_arb_free"] = getattr(fit, f"{name}_arbitrage").is_free
    rows.append(row)
    latest[ticker], n_raw[ticker] = fit, len(quotes)

study = pd.DataFrame(rows).sort_values(["ticker", "date"])
study.to_csv(data_dir / "study.csv", index=False)
with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
    print(study.to_string(index=False))

# RMSE by moneyness band, latest snapshot per ticker. z = k / sqrt(w) is each quote's distance
# from the money in its own standard deviations, so bands mean the same thing at every expiry.
for ticker, fit in sorted(latest.items()):
    t = fit.table
    z = t["k"] / np.sqrt(t["w"])
    band = pd.cut(z, [-np.inf, -2, -1, 1, np.inf], labels=["< -2 sd", "-2..-1 sd", "+-1 sd", "> 1 sd"])
    T = t["T"].to_numpy()
    out = {"quotes": t.groupby(band, observed=True).size()}
    for name in ("ssvi", "essvi", "svi_af"):
        model = np.sqrt(getattr(fit, f"{name}_surface").total_variance(t["k"].to_numpy(), T) / T)
        sq_err = (100 * (model - t["iv_mid"])) ** 2
        out[f"{name}_rmse"] = sq_err.groupby(band, observed=True).mean() ** 0.5
    s = fit.ssvi_params
    print(f"\n{ticker}: {n_raw[ticker]:,} raw quotes; free SVI slices: "
          f"{len(fit.svi_arbitrage.calendar_violations)} calendar violations, "
          f"butterfly-free {fit.svi_arbitrage.butterfly_free}; "
          f"SSVI eta(1+|rho|) = {s.eta * (1 + abs(s.rho)):.3f}")
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(pd.DataFrame(out).to_string())

if "SPX" in latest:
    fit = latest["SPX"]
    picks = fit.expiries[np.unique(np.linspace(0, len(fit.expiries) - 1, 8).round().astype(int))]
    fit.svi_slices = {T: fit.svi_slices[T] for T in picks}  # plot_smiles shows these expiries
    plot_smiles(fit, docs_dir / "spx_smiles.png", title="SPX smiles, CBOE close")

if study["date"].nunique() > 1:
    import matplotlib.pyplot as plt

    cols = ["atm_vol_1y", "ssvi_rho", "essvi_rho0", "essvi_rho1", "ssvi_eta", "ssvi_rmse"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for ax, col in zip(axes.flat, cols):
        for ticker, s in study.groupby("ticker"):
            ax.plot(s["date"], s[col], marker="o", ms=3, label=ticker)
        ax.set_title(col)
    axes.flat[0].legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(docs_dir / "study_params.png", dpi=150)
