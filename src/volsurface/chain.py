"""Option chain cleaning, implied forwards, and the implied-vol table.

Expected quote columns: T (years), strike, option_type ('C'/'P'), bid, ask.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface.black import black_vega, implied_vols

REQUIRED = ("T", "strike", "option_type", "bid", "ask")


def clean_quotes(quotes: pd.DataFrame, min_bid: float = 0.0, max_rel_spread: float = 0.5,
                 min_T: float = 7 / 365) -> pd.DataFrame:
    """Drop quotes that carry no reliable price information."""
    missing = set(REQUIRED) - set(quotes.columns)
    if missing:
        raise ValueError(f"quotes missing columns: {sorted(missing)}")
    q = quotes.copy()
    q["mid"] = 0.5 * (q["bid"] + q["ask"])
    keep = (
        (q["bid"] > min_bid)
        & (q["ask"] > q["bid"])
        & (q["T"] >= min_T)
        & ((q["ask"] - q["bid"]) / q["mid"] <= max_rel_spread)
    )
    return q.loc[keep].reset_index(drop=True)


def implied_forward_and_discount(slice_quotes: pd.DataFrame, n_strikes: int = 8) -> tuple[float, float]:
    """Forward and discount factor for one expiry from put-call parity.

    C - P = DF * (F - K) is linear in K, so regressing (C - P) on K over the
    strikes nearest the money gives slope -DF and intercept DF * F.
    """
    q = slice_quotes
    if "mid" not in q:
        q = q.assign(mid=0.5 * (q["bid"] + q["ask"]))
    calls = q[q["option_type"] == "C"].groupby("strike")["mid"].mean()
    puts = q[q["option_type"] == "P"].groupby("strike")["mid"].mean()
    both = pd.concat([calls.rename("C"), puts.rename("P")], axis=1, join="inner")
    if len(both) < 2:
        raise ValueError("need at least two strikes quoted on both sides")
    diff = both["C"] - both["P"]
    near = diff.abs().nsmallest(min(n_strikes, len(both))).index
    slope, intercept = np.polyfit(np.asarray(near, dtype=float), diff.loc[near].to_numpy(), 1)
    df = -slope
    if not 0.0 < df <= 1.5:
        raise ValueError(f"implausible discount factor {df:.4f} from parity regression")
    return float(intercept / df), float(df)


def build_iv_table(quotes: pd.DataFrame, n_parity_strikes: int = 8) -> pd.DataFrame:
    """Implied vols of out-of-the-money options, one row per quote.

    OTM options are used because they are the most liquid and, for American
    options, carry the least early-exercise premium.

    Expiries that cannot be used are listed in ``table.attrs["skipped"]``
    as {T: reason}.
    """
    q = quotes if "mid" in quotes else quotes.assign(mid=0.5 * (quotes["bid"] + quotes["ask"]))
    frames, skipped = [], {}
    for T, s in q.groupby("T"):
        try:
            F, DF = implied_forward_and_discount(s, n_parity_strikes)
        except ValueError as exc:
            skipped[float(T)] = f"parity: {exc}"
            continue
        otm = s[((s["option_type"] == "C") & (s["strike"] >= F))
                | ((s["option_type"] == "P") & (s["strike"] < F))]
        if otm.empty:
            skipped[float(T)] = "no OTM quotes"
            continue
        K = otm["strike"].to_numpy(dtype=float)
        is_call = (otm["option_type"] == "C").to_numpy()
        iv_mid = implied_vols(otm["mid"].to_numpy(), F, K, T, DF, is_call)
        frames.append(pd.DataFrame({
            "T": T, "strike": K, "option_type": otm["option_type"].to_numpy(),
            "forward": F, "discount": DF, "k": np.log(K / F),
            "iv_mid": iv_mid,
            "iv_bid": implied_vols(otm["bid"].to_numpy(), F, K, T, DF, is_call),
            "iv_ask": implied_vols(otm["ask"].to_numpy(), F, K, T, DF, is_call),
        }))
    if not frames:
        raise ValueError("no expiry produced a usable implied-vol slice")
    table = pd.concat(frames, ignore_index=True).dropna(subset=["iv_mid"])
    table["w"] = table["iv_mid"] ** 2 * table["T"]
    table["vega"] = black_vega(table["forward"], table["strike"], table["T"], table["iv_mid"], table["discount"])
    table = table.sort_values(["T", "k"]).reset_index(drop=True)
    table.attrs["skipped"] = skipped
    return table
