"""Option chains from CBOE delayed quotes (no extra dependency) or Yahoo Finance.

Prefer index options such as ^SPX: they are European, so put-call parity and
Black-76 hold exactly. ETF and single-stock options are American, which biases
in-the-money quotes; the pipeline uses OTM options only to limit this.

CBOE's delayed feed keeps the last session's closing bids and asks, so it also
works outside market hours. Yahoo quotes are delayed and its bids/asks are
often zero outside US market hours (requires `pip install yfinance`).

On index monthly expiries both sources mix AM-settled (SPX) and PM-settled
(SPXW) contracts under one date. Only the root with the most contracts is
kept, and an index's AM root (root == ticker without '^') expires at the
09:30 open.
"""

from __future__ import annotations

import json
import urllib.request

import pandas as pd

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{}.json"
NY = "America/New_York"


def load_cboe_chain(ticker: str = "^SPX", max_T: float = 3.5) -> tuple[pd.DataFrame, float, pd.Timestamp]:
    """Delayed CBOE chain as (quotes, spot, quote time). Indices use '^' (^SPX, ^NDX, ^RUT)."""
    req = urllib.request.Request(CBOE_URL.format(ticker.replace("^", "_")),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)["data"]
    return parse_cboe_chain(data, ticker, max_T)


def parse_cboe_chain(data: dict, ticker: str, max_T: float = 3.5) -> tuple[pd.DataFrame, float, pd.Timestamp]:
    """Turn a CBOE delayed-quotes payload into the pipeline's quote columns."""
    now = pd.Timestamp(data["last_trade_time"]).tz_localize(NY)
    opts = pd.DataFrame(data["options"]).rename(columns={"option": "contractSymbol"})
    parts = opts["contractSymbol"].str.extract(r"^[A-Z]+(\d{6})([CP])(\d{8})$")
    opts["expiry"] = pd.to_datetime(parts[0], format="%y%m%d")
    opts["option_type"] = parts[1]
    opts["strike"] = parts[2].astype(float) / 1000
    frames = []
    for expiry, chain in opts.groupby("expiry"):
        chain, time_of_day = pick_root(chain, ticker)
        T = (expiry.tz_localize(NY) + time_of_day - now).total_seconds() / (365.25 * 86400)
        if 0 < T <= max_T:
            f = chain[["strike", "option_type", "bid", "ask", "volume", "open_interest"]].copy()
            f["T"], f["expiry"] = T, f"{expiry:%Y-%m-%d}"
            frames.append(f)
    return pd.concat(frames, ignore_index=True), float(data["current_price"]), now


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
