import pandas as pd

from volsurface.data import pick_root


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
