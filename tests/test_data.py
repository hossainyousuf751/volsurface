import pandas as pd
import pytest

from volsurface.data import parse_cboe_chain, pick_root


def test_parse_cboe_chain_settlement_and_year_fractions():
    opt = {"bid": 1.0, "ask": 1.2, "volume": 5, "open_interest": 10}
    data = {"last_trade_time": "2026-10-30T16:00:00", "current_price": 7580.0, "options": [
        {**opt, "option": "SPX261120C07600000"}, {**opt, "option": "SPX261120P07500000"},
        {**opt, "option": "SPXW261120C07650000"},                  # minority PM root: dropped
        {**opt, "option": "SPXW261102P07000000"},                  # weekly, PM settled
        {**opt, "option": "SPXW261030C07600000"},                  # already expired
    ]}
    quotes, spot, now = parse_cboe_chain(data, "^SPX")
    assert spot == 7580.0 and now == pd.Timestamp("2026-10-30 16:00", tz="America/New_York")
    assert sorted(quotes["strike"]) == [7000.0, 7500.0, 7600.0]
    monthly = quotes[quotes["expiry"] == "2026-11-20"]
    year = 365.25 * 24  # hours; +1 h because US daylight saving time ends on 2026-11-01
    assert monthly["T"].iloc[0] == pytest.approx((20 * 24 + 17.5 + 1) / year)   # 09:30 AM settlement
    assert quotes.loc[quotes["expiry"] == "2026-11-02", "T"].iloc[0] == pytest.approx((3 * 24 + 1) / year)


def test_pick_root_keeps_majority_root_and_am_settlement():
    chain = pd.DataFrame({"contractSymbol": ["SPX261120C05000000"] * 3 + ["SPXW261120C05010000"] * 2})
    kept, time_of_day = pick_root(chain, "^SPX")
    assert (kept["contractSymbol"].str.startswith("SPX2")).all() and len(kept) == 3
    assert time_of_day == pd.Timedelta(hours=9, minutes=30)


def test_pick_root_weekly_and_single_stock_settle_pm():
    weekly = pd.DataFrame({"contractSymbol": ["SPXW261102C05000000"] * 2})
    assert pick_root(weekly, "^SPX")[1] == pd.Timedelta(hours=16)
    stock = pd.DataFrame({"contractSymbol": ["AAPL261120C00200000"] * 2})
    assert pick_root(stock, "AAPL")[1] == pd.Timedelta(hours=16)
