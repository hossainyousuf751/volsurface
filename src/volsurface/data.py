"""Live option chains from Yahoo Finance (requires `pip install yfinance`).

Prefer index options such as ^SPX: they are European, so put-call parity and
Black-76 hold exactly. ETF and single-stock options are American, which biases
in-the-money quotes; the pipeline uses OTM options only to limit this.
Yahoo quotes are delayed and bids/asks are often zero outside US market hours.

On index monthly expiries Yahoo mixes AM-settled (SPX) and PM-settled (SPXW)
contracts under one date. Only the root with the most contracts is kept, and
an index's AM root (root == ticker without '^') expires at the 09:30 open.
"""

from __future__ import annotations

import pandas as pd


def pick_root(chain: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, pd.Timedelta]:
    """Keep one settlement root of a calls+puts chain; return it with its expiry time of day."""
    roots = chain["contractSymbol"].str.extract(r"^([A-Z]+)\d", expand=False)
    root = roots.value_counts().idxmax()
    am = ticker.startswith("^") and root == ticker.lstrip("^")
    return chain[roots == root], pd.Timedelta(hours=9, minutes=30) if am else pd.Timedelta(hours=16)


def load_yfinance_chain(ticker: str = "^SPX", max_expiries: int = 10,
                        now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, float]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("install the data extra: pip install -e '.[data]'") from exc

    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    now = now or pd.Timestamp.now(tz="UTC")
    frames = []
    for expiry in tk.options[:max_expiries]:
        chain = tk.option_chain(expiry)
        both = pd.concat([chain.calls.assign(option_type="C"), chain.puts.assign(option_type="P")])
        both, time_of_day = pick_root(both, ticker)
        expiry_ts = pd.Timestamp(expiry).tz_localize("America/New_York") + time_of_day
        T = (expiry_ts - now).total_seconds() / (365.25 * 86400)
        if T <= 0:
            continue
        f = both[["strike", "option_type", "bid", "ask", "volume", "openInterest"]].copy()
        f["T"], f["expiry"] = T, expiry
        frames.append(f)
    quotes = pd.concat(frames, ignore_index=True).rename(columns={"openInterest": "open_interest"})
    return quotes, spot
