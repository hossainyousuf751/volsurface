"""Diagnostic plots (requires matplotlib)."""

from __future__ import annotations

import numpy as np

from volsurface.pipeline import SurfaceFit

COLORS = {"quote": "#4C72B0", "svi_af": "#DD8452", "ssvi": "#55A868"}


def plot_smiles(fit: SurfaceFit, path: str, max_panels: int = 8, title: str | None = None):
    import matplotlib.pyplot as plt

    expiries = fit.expiries[:max_panels]
    n_cols = min(4, len(expiries))
    n_rows = int(np.ceil(len(expiries) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows), squeeze=False)
    for ax, T in zip(axes.flat, expiries):
        s = fit.table[fit.table["T"] == T]
        lo = (s["iv_mid"] - s["iv_bid"]).clip(lower=0).fillna(0) * 100
        hi = (s["iv_ask"] - s["iv_mid"]).clip(lower=0).fillna(0) * 100
        ax.errorbar(s["k"], s["iv_mid"] * 100, yerr=[lo, hi], fmt="o", ms=3, lw=0.8,
                    color=COLORS["quote"], label="bid/ask")
        k = np.linspace(s["k"].min(), s["k"].max(), 200)
        for key, name, surf, ls in (("svi_af", "SVI (arbitrage-free)", fit.svi_af_surface, "-"),
                                    ("ssvi", "SSVI", fit.ssvi_surface, "--")):
            ax.plot(k, np.sqrt(surf.total_variance(k, T) / T) * 100, ls, lw=1.6,
                    color=COLORS[key], label=name)
        ax.set_title(f"T = {T:.2f}y", fontsize=11)
        ax.set_xlabel("log-moneyness k = ln(K/F)")
        ax.set_ylabel("implied vol (%)")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(expiries):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title or "Fitted smiles vs. market quotes", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vol_heatmap(fit: SurfaceFit, path: str, kind: str = "implied", model: str = "ssvi",
                     moneyness=(0.7, 1.3), title: str | None = None):
    """Heatmap of implied or local vol over (strike % of spot, maturity)."""
    import matplotlib.pyplot as plt

    surf = fit.ssvi_surface if model == "ssvi" else fit.svi_surface
    spot = fit.curve.spot
    m = np.linspace(*moneyness, 121)
    T = np.linspace(fit.expiries[0], fit.expiries[-1], 80)
    MM, TT = np.meshgrid(m, T)
    fn = surf.local_vol if kind == "local" else surf.implied_vol
    Z = fn(MM * spot, TT) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    mesh = ax.pcolormesh(MM * 100, TT, Z, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label=f"{kind} vol (%)")
    ax.set_xlabel("strike (% of spot)")
    ax.set_ylabel("maturity (years)")
    ax.set_title(title or f"{model.upper()} {kind} volatility surface", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
