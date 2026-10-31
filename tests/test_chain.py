import numpy as np
import pytest

from volsurface import build_iv_table, clean_quotes, implied_forward_and_discount, synthetic_chain


def test_parity_recovers_forward_and_discount():
    quotes, truth = synthetic_chain(rel_spread=0.0, tick=0.0)
    for T, s in quotes.groupby("T"):
        F, DF = implied_forward_and_discount(s)
        assert F == pytest.approx(truth["forwards"][T], rel=1e-6)
        assert DF == pytest.approx(truth["discounts"][T], rel=1e-6)


def test_iv_table_recovers_true_vols():
    quotes, truth = synthetic_chain(rel_spread=0.0, tick=0.0)
    table = build_iv_table(quotes)
    p = truth["params"]
    true_w = p.total_variance(table["k"], table["T"].map(truth["thetas"]))
    np.testing.assert_allclose(table["w"], true_w, rtol=1e-4)


def test_iv_table_uses_otm_options_only():
    table = build_iv_table(clean_quotes(synthetic_chain()[0]))
    calls, puts = table[table["option_type"] == "C"], table[table["option_type"] == "P"]
    assert (calls["strike"] >= calls["forward"]).all()
    assert (puts["strike"] < puts["forward"]).all()


def test_clean_drops_bad_quotes():
    quotes, _ = synthetic_chain()
    quotes.loc[0, "bid"] = 0.0
    quotes.loc[1, "ask"] = quotes.loc[1, "bid"]
    cleaned = clean_quotes(quotes)
    assert len(cleaned) <= len(quotes) - 2
    assert (cleaned["bid"] > 0).all() and (cleaned["ask"] > cleaned["bid"]).all()
