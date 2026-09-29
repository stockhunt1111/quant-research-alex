"""Buy & hold of a list: equal money when it starts, then the units are held; money moves only when the list's
composition changes, and only those trades pay costs, at spot rates. Currency pairs and crude are not held."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_lab.config import COSTS
from strategy_lab.engine.hold import buy_and_hold, nothing_to_hold
from tests.conftest import make_panel, quote_spreads

EQUITY = (COSTS["us_equity"].commission_bps + COSTS["us_equity"].half_spread_bps) / 1e4
SPOT = (COSTS["crypto_spot"].commission_bps + COSTS["crypto_spot"].half_spread_bps) / 1e4


def _hours_of(p) -> pd.DatetimeIndex:
    """The hours inside a daily panel's bars, by their closes."""
    return pd.date_range(p.index[0] - pd.Timedelta(hours=23), p.index[-1], freq="h")


def _grown(r: pd.Series) -> float:
    return float((1 + r).prod())


def test_one_instrument_earns_its_price_move_less_one_entry_cost():
    p = make_panel(ids=("td:AAA",))
    live = pd.DataFrame(True, index=p.index, columns=p.ids)
    c, o = p.close["td:AAA"], p.open["td:AAA"]
    # decided at the first close: bought at the next bar's close, or at its open
    assert np.isclose(_grown(buy_and_hold(p, live, "next_close")), c.iloc[-1] / c.iloc[1] * (1 - EQUITY))
    assert np.isclose(_grown(buy_and_hold(p, live, "next_open")), c.iloc[-1] / o.iloc[1] * (1 - EQUITY))


def test_a_list_that_never_changes_is_never_rebalanced():
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), seed=3)
    live = pd.DataFrame(True, index=p.index, columns=p.ids)
    held = (p.close.iloc[-1] / p.close.iloc[1]).mean() * (1 - EQUITY)          # equal money, units never touched
    rebalanced = float((1 + p.close.pct_change().iloc[2:].mean(axis=1)).prod())  # equal weights every bar
    assert np.isclose(_grown(buy_and_hold(p, live, "next_close")), held)
    assert not np.isclose(held, rebalanced * (1 - EQUITY))


def test_a_leavers_money_buys_the_newcomer_and_the_holding_that_stays_is_not_traded():
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), seed=5)
    live = pd.DataFrame(False, index=p.index, columns=p.ids)
    k = 200                                                   # BBB leaves the list at bar k's close, CCC takes its seat
    live.loc[:p.index[k - 1], ["td:AAA", "td:BBB"]] = True
    live.loc[p.index[k]:, ["td:AAA", "td:CCC"]] = True
    c = p.close
    cash = 1 - EQUITY                                         # bought at bar 1's close: half each, after the entry cost
    units_a, units_b = cash / 2 / c["td:AAA"].iloc[1], cash / 2 / c["td:BBB"].iloc[1]
    proceeds = units_b * c["td:BBB"].iloc[k + 1] * (1 - EQUITY)   # sold at bar k+1's close
    units_c = proceeds * (1 - EQUITY) / c["td:CCC"].iloc[k + 1]
    expected = units_a * c["td:AAA"].iloc[-1] + units_c * c["td:CCC"].iloc[-1]
    assert np.isclose(_grown(buy_and_hold(p, live, "next_close")), expected)


def test_a_seat_left_empty_keeps_its_money_for_its_newcomer_and_a_fixed_list_spreads_it_over_the_rest():
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), seed=5)
    live = pd.DataFrame(False, index=p.index, columns=p.ids)
    # BBB leaves at bar k's close and CCC takes its seat two bars later, as a delisted name's is taken at the end of
    # its last day
    k = 200
    live.loc[:p.index[k - 1], "td:BBB"] = True
    live["td:AAA"] = True
    live.loc[p.index[k + 2]:, "td:CCC"] = True
    c = p.close
    cash = 1 - EQUITY
    units_a, units_b = cash / 2 / c["td:AAA"].iloc[1], cash / 2 / c["td:BBB"].iloc[1]
    proceeds = units_b * c["td:BBB"].iloc[k + 1] * (1 - EQUITY)   # sold at bar k+1's close
    # a ranked list: the money waits in cash and buys CCC at bar k+3's close; AAA is not touched
    units_c = proceeds * (1 - EQUITY) / c["td:CCC"].iloc[k + 3]
    waited = units_a * c["td:AAA"].iloc[-1] + units_c * c["td:CCC"].iloc[-1]
    assert np.isclose(_grown(buy_and_hold(p, live, "next_close", refills=True)), waited)
    # a fixed list: AAA takes the money, then gives CCC half of itself, an equal share of a list of two
    units_a += proceeds * (1 - EQUITY) / c["td:AAA"].iloc[k + 1]
    units_c = units_a / 2 * c["td:AAA"].iloc[k + 3] * (1 - EQUITY) ** 2 / c["td:CCC"].iloc[k + 3]
    spread = units_a / 2 * c["td:AAA"].iloc[-1] + units_c * c["td:CCC"].iloc[-1]
    assert np.isclose(_grown(buy_and_hold(p, live, "next_close")), spread)
    assert not np.isclose(waited, spread)


def test_crude_and_currency_pairs_are_not_held_and_a_list_of_nothing_else_is_cash():
    p = make_panel(ids=("td:XAU/USD", "td:WTI/USD"), seed=7)
    quote_spreads(p.ids, _hours_of(p), opening=0.3, closing=0.5)
    live = pd.DataFrame(True, index=p.index, columns=p.ids)
    c = p.close["td:XAU/USD"]
    assert not nothing_to_hold(p)
    # gold alone, bought at its second bar's close for half the spread its broker quoted there
    assert np.isclose(_grown(buy_and_hold(p, live, "next_close")), c.iloc[-1] / c.iloc[1] * (1 - 0.25 / c.iloc[1]))
    for ids in (("td:EUR/USD", "td:USD/JPY"), ("td:WTI/USD",)):
        q = make_panel(ids=ids, seed=7)
        assert nothing_to_hold(q)
        assert (buy_and_hold(q, pd.DataFrame(True, index=q.index, columns=q.ids), "next_open") == 0.0).all()


def test_a_perp_is_held_on_spot_terms():
    p = make_panel(ids=("perp:XUSDT",), freq="h", n=300)
    live = pd.DataFrame(True, index=p.index, columns=p.ids)
    c = p.close["perp:XUSDT"]
    assert np.isclose(_grown(buy_and_hold(p, live, "next_close")), c.iloc[-1] / c.iloc[1] * (1 - SPOT))


def test_a_comparison_buys_at_its_own_start_not_with_weights_drifted_since():
    from strategy_lab.evaluate import _buy_and_hold
    p = make_panel(ids=("td:AAA", "td:BBB"), seed=7)
    start = p.index[150]
    k = 151                                   # decided at the first bar from `start` on, bought at the next close
    expected = (p.close.iloc[-1] / p.close.iloc[k]).mean() * (1 - EQUITY)
    assert np.isclose(_grown(_buy_and_hold(p, None, "next_close", start)), expected)


def test_a_dividend_is_reinvested_in_what_paid_it(tmp_path, monkeypatch):
    from tests.test_engine import _new_york_date, _store_dividends
    p = make_panel(ids=("td:AAA",), seed=2)
    k = 200
    _store_dividends(tmp_path, monkeypatch, "AAA", [_new_york_date(p.index[k])], 2.0)
    live = pd.DataFrame(True, index=p.index, columns=p.ids)
    c = p.close["td:AAA"]
    expected = c.iloc[-1] / c.iloc[1] * (1 + 2.0 / c.iloc[k]) * (1 - EQUITY)     # the cash buys more at the ex-date's close
    assert np.isclose(_grown(buy_and_hold(p, live, "next_close")), expected)


def test_a_perp_is_held_at_its_spot_pairs_prices_and_on_their_scale_before_the_pair_lists(tmp_path, monkeypatch):
    from strategy_lab.data import bars, store
    from strategy_lab.data.bars import FIELDS
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(bars, "STORE_DIR", tmp_path)
    p = make_panel(ids=("perp:1000XUSDT",), freq="h", n=300, seed=6)          # the perp quotes 1000 coins
    k = 100                                                                    # the spot pair lists at bar k
    other = make_panel(ids=("spot:XUSDT",), freq="h", n=300, seed=9)           # the coin on spot: another path
    spot = pd.DataFrame({f: other.close["spot:XUSDT"] if f != "volume" else 1.0 for f in FIELDS}).iloc[k:] / 1000
    spot["open"] = other.open["spot:XUSDT"].iloc[k:] / 1000
    store.write_bars("spot", p.timeframe, "XUSDT", spot)
    perp = p.close["perp:1000XUSDT"]
    scale = (spot["close"] / perp.iloc[k:]).median()
    held = spot["close"].reindex(p.index).combine_first(perp * scale)
    live = pd.DataFrame(True, index=p.index, columns=p.ids)
    assert np.isclose(_grown(buy_and_hold(p, live, "next_close")), held.iloc[-1] / held.iloc[1] * (1 - SPOT))
