"""End-to-end: raw quotes -> IV table -> SVI, SSVI, eSSVI and arbitrage-free SVI surfaces -> diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from volsurface.arbitrage import ArbitrageReport, check_surface
from volsurface.chain import build_iv_table, clean_quotes
from volsurface.ssvi import ESSVIParams, SSVIParams, fit_essvi, fit_ssvi
from volsurface.surface import ForwardCurve, SSVISurface, SVISliceSurface
from volsurface.svi import Z_DOMAIN, SVIParams, fit_svi, fit_svi_arbitrage_free


@dataclass
class SurfaceFit:
    table: pd.DataFrame
    curve: ForwardCurve
    svi_slices: dict[float, SVIParams]
    thetas: dict[float, float]
    ssvi_params: SSVIParams
    svi_surface: SVISliceSurface
    ssvi_surface: SSVISurface
    svi_arbitrage: ArbitrageReport
    ssvi_arbitrage: ArbitrageReport
    essvi_params: ESSVIParams
    essvi_surface: SSVISurface
    essvi_arbitrage: ArbitrageReport
    svi_af_slices: dict[float, SVIParams]
    svi_af_surface: SVISliceSurface
    svi_af_arbitrage: ArbitrageReport
    skipped: dict[float, str] = field(default_factory=dict)  # expiry -> why it was dropped

    @property
    def expiries(self) -> np.ndarray:
        return np.array(sorted(self.svi_slices))

    def local_vol_surface(self, min_ratio: float = 1.5) -> SVISliceSurface:
        """Arbitrage-free SVI on pillar expiries, each >= min_ratio x the previous, plus the last.

        Interpolating exactly through expiries a few days apart turns tiny noise in w
        into large swings in dw/dT, so Dupire local vol oscillates in T. Geometrically
        spaced pillars keep it smooth. A subset of calendar-ordered slices stays ordered.
        """
        keep = [self.expiries[0]]
        for T in self.expiries[1:]:
            if T >= min_ratio * keep[-1] or T == self.expiries[-1]:
                keep.append(T)
        surface = SVISliceSurface(self.curve, keep, [self.svi_af_slices[T] for T in keep])
        surface.k_range = self.svi_af_surface.k_range
        return surface


def fit_surface(quotes: pd.DataFrame, spot: float, min_quotes_per_expiry: int = 8,
                **clean_kwargs) -> SurfaceFit:
    cleaned = clean_quotes(quotes, **clean_kwargs)
    skipped = {float(T): "no quotes left after cleaning" for T in set(quotes["T"]) - set(cleaned["T"])}
    table = build_iv_table(cleaned)
    skipped.update(table.attrs["skipped"])
    counts = table.groupby("T").size()
    for T, n in counts[counts < min_quotes_per_expiry].items():
        skipped[float(T)] = f"only {n} implied vols (< {min_quotes_per_expiry})"
    table = table[table["T"].isin(counts[counts >= min_quotes_per_expiry].index)].reset_index(drop=True)
    if table["T"].nunique() < 2:
        raise ValueError(f"need at least two usable expiries to build a surface; skipped: {skipped}")

    slices, thetas = {}, {}
    for T, s in table.groupby("T"):
        slices[float(T)] = fit_svi(s["k"], s["w"], weights=s["vega"])
    expiries = np.array(sorted(slices))
    raw_theta = np.array([float(slices[T].total_variance(0.0)) for T in expiries])
    for T, th in zip(expiries, np.maximum.accumulate(raw_theta)):  # enforce monotone ATM variance
        thetas[float(T)] = float(th)

    theta_per_quote = table["T"].map(thetas).to_numpy()
    ssvi = fit_ssvi(table["k"], theta_per_quote, table["w"], weights=table["vega"])
    essvi = fit_essvi(table["k"], theta_per_quote, table["w"], weights=table["vega"], start=ssvi)
    starts = {T: SVIParams.from_ssvi(th, float(essvi.rho(th)), float(essvi.phi(th)))
              for T, th in thetas.items()}
    vol_weighted = {float(T): (s["k"], s["w"], 1 / (2 * s["iv_mid"] * T)) for T, s in table.groupby("T")}
    af_slices = fit_svi_arbitrage_free(vol_weighted, starts, thetas)

    per_exp = table.groupby("T")[["forward", "discount"]].first().loc[expiries]
    curve = ForwardCurve(spot, expiries, per_exp["forward"].to_numpy(), per_exp["discount"].to_numpy())
    svi_surface = SVISliceSurface(curve, expiries, [slices[T] for T in expiries])
    ssvi_surface = SSVISurface(curve, ssvi, expiries, [thetas[T] for T in expiries])
    essvi_surface = SSVISurface(curve, essvi, expiries, [thetas[T] for T in expiries])
    svi_af_surface = SVISliceSurface(curve, expiries, [af_slices[T] for T in expiries])
    k_range = (float(table["k"].min()), float(table["k"].max()))
    for surf in (svi_surface, ssvi_surface, essvi_surface, svi_af_surface):
        surf.k_range = k_range

    return SurfaceFit(
        table=table, curve=curve, svi_slices=slices, thetas=thetas, ssvi_params=ssvi,
        svi_surface=svi_surface, ssvi_surface=ssvi_surface,
        svi_arbitrage=check_surface(svi_surface, expiries, z_grid=Z_DOMAIN),
        ssvi_arbitrage=check_surface(ssvi_surface, expiries, z_grid=Z_DOMAIN),
        essvi_params=essvi, essvi_surface=essvi_surface,
        essvi_arbitrage=check_surface(essvi_surface, expiries, z_grid=Z_DOMAIN),
        svi_af_slices=af_slices, svi_af_surface=svi_af_surface,
        svi_af_arbitrage=check_surface(svi_af_surface, expiries, z_grid=Z_DOMAIN),
        skipped=dict(sorted(skipped.items())),
    )


def fit_errors(fit: SurfaceFit) -> pd.DataFrame:
    """Per-expiry RMSE (vol points) and share of model vols inside the bid-ask band."""
    t = fit.table
    rows = []
    for T, s in t.groupby("T"):
        row = {"T": T, "n_quotes": len(s)}
        for name, surf in (("svi", fit.svi_surface), ("ssvi", fit.ssvi_surface),
                           ("essvi", fit.essvi_surface), ("svi_af", fit.svi_af_surface)):
            model = np.sqrt(surf.total_variance(s["k"].to_numpy(), T) / T)
            row[f"{name}_rmse_volpts"] = 100 * np.sqrt(np.mean((model - s["iv_mid"]) ** 2))
            band = s["iv_bid"].notna() & s["iv_ask"].notna()
            inside = (model[band] >= s["iv_bid"][band]) & (model[band] <= s["iv_ask"][band])
            row[f"{name}_in_spread_pct"] = 100 * inside.mean() if band.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
