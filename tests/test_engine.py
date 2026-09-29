import numpy as np
import pandas as pd
import pytest

from strategy_lab.config import COSTS
from strategy_lab.engine import backtest as bt
from strategy_lab.engine.trades import ledger
from tests.conftest import make_minutes, make_panel

RATE = (COSTS["us_equity"].commission_bps + COSTS["us_equity"].half_spread_bps) / 1e4
BORROW = COSTS["us_equity"].borrow_bps_annual / 1e4


def _units_account(fill_px, close, T, book=False, rate=None):
    """An independent account kept in cash and units at the bars' prices, not in weights, without margin (the targets
    are long): at a fill whose target moved (a position's own target, or any target of a book rebalanced whole) the
    position is bought or sold to its target times the equity at the fill price, and otherwise its units are held; the
    bar's sales come first, and its purchases are paid out of the cash then, cut pro rata to it when they want more.
    A fill's cost is paid out of every holding pro rata, as the engine's (1 - cost) factor does; no carry. `fill_px[k]`
    is the price a fill before bar k's session takes place at: its open (next_open), or the close before
    (next_close); `rate[k]` what such a fill costs a side, by instrument (RATE when not given)."""
    n, m = close.shape
    cash, units, last = 1.0, np.zeros(m), np.zeros(m)
    equity, out = 1.0, np.zeros(n)
    for k in range(1, n):
        tgt = T[k - 1]
        at_fill = cash + (units * fill_px[k]).sum()
        held = units * fill_px[k] / at_fill
        move = np.full(m, (tgt != last).any()) if book else tgt != last
        new = np.where(move, tgt, held)
        sold = np.where(move, np.maximum(held - new, 0.0), 0.0)
        bought = np.where(move, np.maximum(new - held, 0.0), 0.0)
        free = cash / at_fill + sold.sum()                  # the cash the bar's sales leave, a share of the equity
        if bought.sum() > free:
            new = np.where(bought > 0, held + bought * max(free, 0.0) / bought.sum(), new)
        after = at_fill * (1 - (np.abs(new - held) * (RATE if rate is None else rate[k])).sum())
        units = new * after / fill_px[k]
        cash = after * (1 - new.sum())
        last = np.where(move, tgt, last)
        closed = cash + (units * close[k]).sum()
        out[k] = closed / equity - 1
        equity = closed
    return out


def _random_target(p, seed=1, values=(-0.5, 0.0, 0.25, 0.5)):
    rng = np.random.default_rng(seed)
    raw = rng.choice(values, size=p.close.shape)
    raw = np.where(rng.random(p.close.shape) < 0.8, np.nan, raw)          # sparse changes, held in between
    return pd.DataFrame(raw, index=p.index, columns=p.ids).ffill().fillna(0.0)


@pytest.mark.parametrize("book", [False, True])
def test_next_open_holds_units_as_an_account_in_cash_and_units(book):
    p = make_panel()
    T = _random_target(p, values=(0.0, 0.25, 0.5))
    res = bt.run(p, T, fill="next_open", book=book)
    ref = _units_account(p.open.to_numpy(), p.close.to_numpy(), T.to_numpy(), book)
    assert np.allclose(res.returns.to_numpy(), ref, atol=1e-12)


@pytest.mark.parametrize("book", [False, True])
def test_next_close_fills_at_the_close_a_bar_later_and_holds_units(book):
    p = make_panel(seed=3)
    T = _random_target(p, seed=4, values=(0.0, 0.25, 0.5))
    res = bt.run(p, T, fill="next_close", book=book)
    c = p.close.to_numpy()
    before = np.vstack([c[:1], c[:-1]])                                      # a fill before bar k: at k-1's close
    shifted = np.vstack([np.zeros((1, 2)), T.to_numpy()[:-1]])              # of the target decided a bar earlier
    assert np.allclose(res.returns.to_numpy(), _units_account(before, c, shifted, book), atol=1e-12)


def test_a_held_position_drifts_with_its_price_and_a_book_is_brought_back_to_its_weights():
    p = make_panel(n=60, seed=8)
    half = pd.DataFrame({"td:AAA": 0.5, "td:BBB": 0.0}, index=p.index)
    grew = (1 + bt.run(p, half).returns).prod()
    o, c = p.open["td:AAA"].iloc[1], p.close["td:AAA"].iloc[-1]
    assert grew == pytest.approx((1 - 0.5 * RATE) * (0.5 + 0.5 * c / o), rel=1e-12)     # half the equity's units, held
    both = pd.DataFrame(0.5, index=p.index, columns=p.ids)
    both.iloc[30:] = [0.5, 0.49]                                            # one weight moves at bar 30
    rule, book = bt.run(p, both), bt.run(p, both, book=True)
    assert rule.turnover.iloc[31] == pytest.approx(abs(0.49 - _held_before(p, 31, "td:BBB")), rel=1e-9)
    assert book.turnover.iloc[31] > rule.turnover.iloc[31]                  # the book trades AAA back to 0.5 as well


def _held_before(p, k, inst):
    """The weight of a half-and-half book's `inst` at bar k's open, after holding its units since bar 1's open."""
    g = (p.close.iloc[k - 1] / p.open.iloc[1]) * (p.open.iloc[k] / p.close.iloc[k - 1])
    return float(0.5 * g[inst] / (0.5 * g).sum())


def test_signal_cannot_earn_the_bar_it_was_computed_on():
    p = make_panel(ids=("td:AAA",), n=3000, seed=5)
    sess = (p.close / p.open - 1)["td:AAA"]
    knows_now = pd.DataFrame({"td:AAA": np.sign(sess)})                     # known at close t -> held in t+1
    knows_next = pd.DataFrame({"td:AAA": np.sign(sess.shift(-1)).fillna(0)})  # peeks one bar ahead
    fair = bt.run(p, knows_now).gross.mean()
    cheat = bt.run(p, knows_next).gross.mean()
    assert abs(fair) < 5e-4 < cheat


def test_gross_above_capital_is_refused():
    p = make_panel()
    T = pd.DataFrame(0.8, index=p.index, columns=p.ids)
    with pytest.raises(ValueError, match="gross"):
        bt.run(p, T)


def _bars(rows):
    """Tiny single-instrument panel from (open, high, low, close) rows."""
    from strategy_lab.data.bars import Panel
    from strategy_lab.data.instruments import parse
    idx = pd.date_range("2024-01-01 21:00", periods=len(rows), freq="D", tz="UTC")
    a = np.array(rows, dtype=float)
    cols = {"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3],
            "volume": np.ones(len(a)), "dollar_volume": a[:, 3]}
    return Panel("1d", {"td:X": parse("td:X")}, **{k: pd.DataFrame({"td:X": v}, index=idx) for k, v in cols.items()})


def test_stop_fills_at_level_and_stays_flat_until_signal_changes():
    p = _bars([(100, 100, 100, 100), (100, 101, 97, 99), (99, 104, 98, 103), (103, 104, 102, 103)])
    T = pd.DataFrame({"td:X": [1.0, 1.0, 1.0, 1.0]}, index=p.index)
    res = bt.run(p, T, exits=bt.Exits(stop=0.02))
    assert res.exits["td:X"].iloc[1] == pytest.approx(98.0)                 # entry 100 at bar 1 open, stop 98
    assert res.weights["td:X"].iloc[2] == 0.0                              # target unchanged -> stays out
    assert res.gross.iloc[1] == pytest.approx(98 / 100 - 1)


def test_gap_through_stop_fills_at_open_and_stop_beats_take():
    p = _bars([(100, 100, 100, 100), (100, 101, 99, 100), (95, 96, 94, 95)])
    T = pd.DataFrame({"td:X": [1.0, 1.0, 1.0]}, index=p.index)
    res = bt.run(p, T, exits=bt.Exits(stop=0.02))
    assert res.exits["td:X"].iloc[2] == pytest.approx(95.0)
    both = _bars([(100, 100, 100, 100), (100, 106, 97, 101)])
    r2 = bt.run(both, pd.DataFrame({"td:X": [1.0, 1.0]}, index=both.index), exits=bt.Exits(stop=0.02, take=0.05))
    assert r2.exits["td:X"].iloc[1] == pytest.approx(98.0)                  # both touched: the stop is assumed first


def test_trailing_stop_follows_the_best_price_of_previous_bars():
    p = _bars([(100, 100, 100, 100), (100, 110, 99, 109), (109, 110, 104, 105)])
    res = bt.run(p, pd.DataFrame({"td:X": [1.0, 1.0, 1.0]}, index=p.index), exits=bt.Exits(trail=0.05))
    assert res.exits["td:X"].iloc[2] == pytest.approx(110 * 0.95)


def test_ledger_counts_a_flip_as_exit_plus_entry_and_a_resize_as_one_trade():
    p = make_panel(ids=("td:AAA",), n=10, seed=9)
    T = pd.DataFrame({"td:AAA": [0.5, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, index=p.index)
    res = bt.run(p, T)
    tr = ledger(p, res)
    assert list(tr["side"]) == [1, -1]
    assert tr["entry_time"].iloc[0] == p.index[1] and tr["exit_time"].iloc[0] == p.index[4]
    assert tr["entry_px"].iloc[1] == pytest.approx(p.open["td:AAA"].iloc[4])


def _miss(p, inst, rows):
    """The panel with `inst` not printing the given bars."""
    for f in ("open", "high", "low", "close", "volume", "dollar_volume"):
        getattr(p, f).loc[p.index[rows], inst] = np.nan
    return p


def test_a_missing_bar_is_ridden_through_not_traded():
    p = _miss(make_panel(ids=("td:AAA",), n=30, seed=2), "td:AAA", [10])
    res = bt.run(p, pd.DataFrame({"td:AAA": 1.0}, index=p.index))
    c, o = p.close["td:AAA"], p.open["td:AAA"]
    assert res.weights["td:AAA"].iloc[10] == 1.0 and res.returns.iloc[10] == 0.0
    assert res.returns.iloc[11] == pytest.approx(c.iloc[11] / c.iloc[9] - 1, abs=1e-12)
    assert (1 + res.returns).prod() == pytest.approx((1 - RATE) * c.iloc[-1] / o.iloc[1], rel=1e-12)


def test_a_fill_due_on_a_missing_bar_waits_for_the_next_bar():
    p = _miss(make_panel(ids=("td:AAA",), n=30, seed=2), "td:AAA", [10])
    T = pd.DataFrame({"td:AAA": [0.0] * 9 + [1.0] * 21}, index=p.index)             # decided at close 9
    nxt = bt.run(p, T)
    assert nxt.weights["td:AAA"].iloc[10] == 0.0 and nxt.weights["td:AAA"].iloc[11] == 1.0
    c, o = p.close["td:AAA"], p.open["td:AAA"]
    assert nxt.returns.iloc[11] == pytest.approx((1 - RATE) * c.iloc[11] / o.iloc[11] - 1, abs=1e-12)
    late = bt.run(p, T, fill="next_close")
    assert late.weights["td:AAA"].iloc[11] == 0.0 and late.weights["td:AAA"].iloc[12] == 1.0


def test_a_position_is_closed_at_the_last_close_when_the_data_ends():
    p = _miss(make_panel(ids=("td:AAA", "td:BBB"), n=40, seed=6), "td:BBB", list(range(25, 40)))
    T = pd.DataFrame(0.5, index=p.index, columns=p.ids)
    for exits in (None, bt.Exits(trail=0.5)):
        res = bt.run(p, T, exits=exits)
        assert (res.weights["td:BBB"].iloc[25:] == 0.0).all()
        tr = ledger(p, res)
        last = tr[tr["instrument"] == "td:BBB"].iloc[-1]
        assert last["exit_reason"] == "delisted" and last["exit_px"] == p.close["td:BBB"].iloc[24]
        assert res.cost.iloc[25] == pytest.approx(0.5 * RATE * res.weights["td:BBB"].iloc[24] / 0.5, rel=0.05)


def test_a_series_ending_a_few_days_early_is_a_vendor_lag_and_stays_open():
    p = _miss(make_panel(ids=("td:AAA", "td:BBB"), n=40, seed=6), "td:BBB", [38, 39])       # two days short
    res = bt.run(p, pd.DataFrame(0.5, index=p.index, columns=p.ids))
    assert res.weights["td:BBB"].iloc[-1] > 0.4
    last = ledger(p, res).query("instrument == 'td:BBB'").iloc[-1]
    assert last["exit_reason"] == "open_at_end" and last["exit_px"] == p.close["td:BBB"].iloc[37]


def test_a_perp_the_exchange_delisted_ends_at_its_last_bar_even_close_to_the_data_end(tmp_path, monkeypatch):
    from strategy_lab.data import store
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    p = _miss(make_panel(ids=("perp:AAAUSDT", "perp:BBBUSDT"), n=40, seed=6), "perp:BBBUSDT", [38, 39])
    assert not bt.ended(p)["perp:BBBUSDT"].any()                     # two days short: a vendor lag, by the bars alone
    store.write_meta("perp", "1d", "BBBUSDT", {"delisted": str(p.index[37])})       # the exchange delivered it
    stopped = bt.ended(p)
    assert stopped["perp:BBBUSDT"].tolist()[-3:] == [False, True, True] and not stopped["perp:AAAUSDT"].any()


def test_a_perp_delisted_and_listed_anew_holds_nothing_between_and_trades_again_after(tmp_path, monkeypatch):
    from strategy_lab.data import bars, store
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(bars, "STORE_DIR", tmp_path)
    p = _miss(make_panel(ids=("perp:AAAUSDT", "perp:BBBUSDT"), n=60, seed=9), "perp:BBBUSDT", list(range(20, 36)))
    for sym in ("AAAUSDT", "BBBUSDT"):                              # no carry: funding of zero at every settlement
        (tmp_path / "perp" / "funding").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"rate": 0.0}, index=pd.DatetimeIndex(p.index, name="settle_time")) \
            .to_parquet(tmp_path / "perp" / "funding" / f"{sym}.parquet")
    store.write_meta("perp", "1d", "BBBUSDT", {"delisted_periods": [[str(p.index[19]), str(p.index[36])]]})
    res = bt.run(p, pd.DataFrame(0.5, index=p.index, columns=p.ids))
    w = res.weights["perp:BBBUSDT"]
    assert (w.iloc[20:36] == 0.0).all() and (w.iloc[38:] > 0.4).all()
    bbb = ledger(p, res).query("instrument == 'perp:BBBUSDT'")
    assert bbb["exit_reason"].tolist() == ["delisted", "open_at_end"]
    assert bbb["exit_px"].iloc[0] == p.close["perp:BBBUSDT"].iloc[19]


def test_a_position_closed_because_its_instrument_left_the_list_is_marked_so():
    p = make_panel(ids=("td:AAA", "td:BBB"), n=120, seed=8)
    k = 60
    live = pd.DataFrame(True, index=p.index, columns=p.ids)
    live.loc[p.index[k]:, "td:BBB"] = False                          # BBB leaves the list at bar k's close
    res = bt.run(p, live.astype(float) * 0.5, fill="next_open")      # always long while listed
    marked, plain = ledger(p, res, live), ledger(p, res)
    bbb = marked[marked["instrument"] == "td:BBB"]
    assert bbb["exit_reason"].tolist() == ["left_list"] and bbb["exit_time"].iloc[0] == p.index[k + 1]
    assert plain.loc[plain["instrument"] == "td:BBB", "exit_reason"].tolist() == ["signal"]
    assert marked.loc[marked["instrument"] == "td:AAA", "exit_reason"].tolist() == ["open_at_end"]


def test_a_panel_backtested_again_after_its_funding_was_rewritten_pays_the_new_funding(tmp_path, monkeypatch):
    import os
    from strategy_lab.data import bars, store
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(bars, "STORE_DIR", tmp_path)
    p = make_panel(ids=("perp:AAAUSDT",), n=30, seed=3)
    path = tmp_path / "perp" / "funding" / "AAAUSDT.parquet"
    path.parent.mkdir(parents=True)
    long = pd.DataFrame(1.0, index=p.index, columns=p.ids)
    pd.DataFrame({"rate": 0.0}, index=pd.DatetimeIndex(p.index, name="settle_time")).to_parquet(path)
    assert bt.run(p, long).carry.sum() == 0.0
    pd.DataFrame({"rate": 0.001}, index=pd.DatetimeIndex(p.index, name="settle_time")).to_parquet(path)
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))     # a later write, however quick
    assert bt.run(p, long).carry.sum() > 0.0


def _store_dividends(tmp_path, monkeypatch, symbol, ex_dates, amount):
    from strategy_lab.data import bars, store
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(bars, "STORE_DIR", tmp_path)
    path = tmp_path / "td" / "dividends" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"amount": amount}, index=pd.DatetimeIndex(ex_dates, name="ex_date")).to_parquet(path)


def _new_york_date(t):
    return t.tz_convert("America/New_York").normalize().tz_localize(None)


def test_a_dividend_is_paid_on_its_ex_date_to_the_position_held_at_the_close_before(tmp_path, monkeypatch):
    p = make_panel(ids=("td:AAA",), n=60, seed=4)            # daily bars closing at 21:00 UTC: one session each
    k = 30
    p.open.iloc[k - 1] = p.close.iloc[k - 1]                  # a flat session before: the short is still -1 at its close
    p.open.iloc[k] = p.close.iloc[k - 1]                      # and no gap after it: -1 again at the ex-date's open
    _store_dividends(tmp_path, monkeypatch, "AAA", [_new_york_date(p.index[k])], 1.5)
    paid = 1.5 / p.close["td:AAA"].iloc[k - 1]
    long = pd.DataFrame(1.0, index=p.index, columns=p.ids)
    assert bt.run(p, long).carry.iloc[k] == pytest.approx(-paid)                 # the long is paid
    a_day = (p.index[k] - p.index[k - 1]) / bt.YEAR
    short = -long
    short.iloc[: k - 2] = 0.0                                 # sold short at k-1's open: -1 of the equity then
    assert bt.run(p, short).carry.iloc[k] == pytest.approx(paid + BORROW * a_day)   # the short pays it, and its borrow
    bought_on_the_ex_date = long.copy()
    bought_on_the_ex_date.iloc[: k - 1] = 0.0                                     # decided at k-1's close, bought at k's open
    assert bt.run(p, bought_on_the_ex_date).carry.iloc[k] == pytest.approx(0.0)


def _still(p, inst, rows):
    """The panel with `inst` printing no trade on the given bars: no volume, every price at the last close."""
    j = p.close.columns.get_loc(inst)
    for r in rows:
        last = p.close.iloc[r - 1, j]
        for f in ("open", "high", "low", "close"):
            getattr(p, f).iloc[r, j] = last
        p.volume.iloc[r, j] = p.dollar_volume.iloc[r, j] = 0.0
    return p


def test_a_fill_due_on_a_bar_without_a_trade_waits_for_the_next_one_that_trades():
    p = _still(make_panel(ids=("td:AAA",), n=30, seed=2), "td:AAA", [10])
    T = pd.DataFrame({"td:AAA": [0.0] * 9 + [1.0] * 21}, index=p.index)             # decided at close 9
    nxt = bt.run(p, T)
    assert nxt.weights["td:AAA"].iloc[10] == 0.0 and nxt.weights["td:AAA"].iloc[11] == 1.0
    quotes = make_panel(ids=("td:AAA",), n=30, seed=2)                 # a series with no volume at all: quotes
    quotes.volume.iloc[:, :] = 0.0
    quotes = _still(quotes, "td:AAA", [10])
    assert bt.run(quotes, T).weights["td:AAA"].iloc[10] == 1.0


def test_an_exit_walk_fills_on_the_next_bar_that_trades_as_the_plain_walk_does():
    p = _still(make_panel(ids=("td:AAA",), n=30, seed=2), "td:AAA", [10])
    T = pd.DataFrame({"td:AAA": [0.0] * 9 + [1.0] * 21}, index=p.index)             # decided at close 9
    walked = bt.run(p, T, exits=bt.Exits(stop=0.5))
    assert walked.weights["td:AAA"].iloc[10] == 0.0 and walked.weights["td:AAA"].iloc[11] == 1.0
    assert walked.weights.equals(bt.run(p, T).weights)


def test_a_stopped_out_position_stays_flat_when_its_target_is_only_resized():
    p = _bars([(100, 100, 100, 100), (100, 101, 97, 99), (99, 104, 98, 103), (103, 104, 102, 103), (103, 104, 102, 103)])
    # the list's share of the name halves after the stop (a second name joined): not a new signal of the rule
    res = bt.run(p, pd.DataFrame({"td:X": [1.0, 1.0, 0.5, 0.5, 0.5]}, index=p.index), exits=bt.Exits(stop=0.02))
    assert res.exits["td:X"].iloc[1] == pytest.approx(98.0)
    assert (res.weights["td:X"].iloc[2:] == 0.0).all()
    again = bt.run(p, pd.DataFrame({"td:X": [1.0, 1.0, 0.0, 0.5, 0.5]}, index=p.index), exits=bt.Exits(stop=0.02))
    assert again.weights["td:X"].iloc[4] == 0.5                             # after the rule turned flat, a new trade


def test_a_short_whose_price_doubles_is_liquidated_there_and_loses_the_money_it_was_sold_for():
    # sold at 100 with half the equity; the price reaches 250 inside bar 2: closed at 200, the account keeps half
    p = _bars([(100, 100, 100, 100), (100, 120, 99, 110), (110, 250, 105, 240), (240, 245, 230, 240)])
    res = bt.run(p, pd.DataFrame({"td:X": [-0.5] * 4}, index=p.index))
    assert res.exits["td:X"].iloc[2] == pytest.approx(200.0) and res.liquidated["td:X"].iloc[2]
    assert (1 + res.gross).prod() == pytest.approx(0.5)
    assert res.weights["td:X"].iloc[3] == 0.0 and res.gross.iloc[3] == 0.0
    trades = ledger(p, res)
    assert trades["exit_reason"].tolist() == ["liquidated"] and trades["gross_return"].iloc[0] == pytest.approx(-1.0)


def test_a_short_whose_price_gaps_past_twice_its_entry_loses_no_more_than_the_money_it_was_sold_for():
    p = _bars([(100, 100, 100, 100), (100, 120, 99, 110), (300, 310, 290, 305), (305, 310, 300, 305)])
    res = bt.run(p, pd.DataFrame({"td:X": [-0.5] * 4}, index=p.index))
    assert (1 + res.gross).prod() == pytest.approx(0.5)                   # at 305 unliquidated it would be below zero
    assert res.exits["td:X"].iloc[1] == pytest.approx(200.0) and res.liquidated["td:X"].iloc[1]
    assert (res.weights["td:X"].iloc[2:] == 0.0).all()
    assert ledger(p, res)["exit_reason"].tolist() == ["liquidated"]


def test_a_short_short_of_twice_its_entry_is_held_and_a_long_is_never_liquidated():
    p = _bars([(100, 100, 100, 100), (100, 190, 99, 180), (180, 195, 170, 190)])
    short = bt.run(p, pd.DataFrame({"td:X": [-0.5] * 3}, index=p.index))
    assert not short.liquidated.to_numpy().any() and short.exits["td:X"].isna().all()
    assert short.gross.iloc[2] != 0.0
    up = _bars([(100, 100, 100, 100), (100, 400, 99, 390), (390, 900, 380, 850)])
    assert not bt.run(up, pd.DataFrame({"td:X": [0.5] * 3}, index=up.index)).liquidated.to_numpy().any()


def test_a_liquidated_short_stays_out_until_its_target_leaves_the_short_side():
    p = _bars([(100, 100, 100, 100), (100, 120, 99, 110), (110, 250, 105, 240), (240, 245, 230, 240),
               (240, 245, 230, 240), (240, 245, 230, 240)])
    # after the liquidation the rule's short is only resized (-0.25): still out; it turns flat, then short again
    res = bt.run(p, pd.DataFrame({"td:X": [-0.5, -0.5, -0.25, 0.0, -0.5, -0.5]}, index=p.index))
    assert res.liquidated["td:X"].iloc[2]
    assert (res.weights["td:X"].iloc[3:5] == 0.0).all() and res.weights["td:X"].iloc[5] == -0.5
    again = ledger(p, res)
    assert again["exit_reason"].tolist() == ["liquidated", "open_at_end"]
    assert again["entry_px"].iloc[1] == 240.0


def _quiet_then(rows):
    """Twenty bars at 100 whose true range is 2 (an ATR of 2 by bar 14), then the given (open, high, low, close)."""
    return _bars([(100, 101, 99, 100)] * 20 + rows)


def test_an_atr_stop_and_target_sit_their_multiples_of_the_average_true_range_from_the_entry():
    T = lambda p: pd.DataFrame({"td:X": [0.0] * 19 + [1.0] * (len(p.index) - 19)}, index=p.index)   # noqa: E731
    stop = _quiet_then([(100, 101, 99, 100), (100, 100.5, 95, 97)])        # entry at bar 20's open, 100
    got = bt.run(stop, T(stop), exits=bt.Exits(stop_atr=2.0))
    assert np.isnan(got.exits["td:X"].iloc[20]) and got.exits["td:X"].iloc[21] == pytest.approx(96.0)
    take = _quiet_then([(100, 101, 99, 100), (100, 107, 99.5, 105)])
    assert bt.run(take, T(take), exits=bt.Exits(take_atr=3.0)).exits["td:X"].iloc[21] == pytest.approx(106.0)
    # a chandelier: 2 ATRs under the best price so far, the ATR as of the close before the bar (Wilder's average)
    trail = _quiet_then([(100, 101, 99, 100), (100, 110, 99.5, 109), (109, 109.5, 104, 105)])
    got = bt.run(trail, T(trail), exits=bt.Exits(trail_atr=2.0))
    atr = (2.0 * 13 + (110 - 99.5)) / 14                                    # bar 21's true range 10.5
    assert np.isnan(got.exits["td:X"].iloc[21]) and got.exits["td:X"].iloc[22] == pytest.approx(110 - 2 * atr)
    both = bt.run(stop, T(stop), exits=bt.Exits(stop=0.05, stop_atr=2.0))  # 95 and 96: the one nearer the price
    assert both.exits["td:X"].iloc[21] == pytest.approx(96.0)


def _hourly_panel(hours: pd.DataFrame, instrument: str):
    from strategy_lab.data.bars import Panel
    from strategy_lab.data.instruments import parse
    cols = {**{f: hours[f] for f in ("open", "high", "low", "close")}, "volume": pd.Series(1e6, index=hours.index),
            "dollar_volume": hours["close"] * 1e6}
    return Panel("1h", {instrument: parse(instrument)}, **{f: pd.DataFrame({instrument: v}) for f, v in cols.items()})


def _stop_reset_every_minute(hours, mins, target, trail):
    """An independent account of a long trailing stop a bot re-sets at every minute's close: filled at the hour's open
    after its target turns long, the stop `trail` below the best price since, checked against each minute (a gap
    through it at the minute's open); an hour without minutes counts as one minute of its own prices. After an exit
    the position waits for its target to leave long. The exit price by hour."""
    exits, held, out, best = {}, False, False, None
    for k in range(1, len(hours)):
        want = target[k - 1] > 0
        if out and want:
            want = False
        elif not want:
            out = False
        if want and not held:
            held, best = True, hours["open"].iloc[k]
        elif not want:
            held = False
        if not held:
            continue
        inside = mins[(mins.index > hours.index[k - 1]) & (mins.index <= hours.index[k])]
        if inside.empty:
            inside = hours.iloc[[k]]
        for _, bar in inside.iterrows():
            level = best * (1 - trail)
            px = bar["open"] if bar["open"] <= level else level if bar["low"] <= level else None
            if px is not None:
                exits[hours.index[k]] = px
                held, out = False, True
                break
            best = max(best, bar["high"])
    return exits


def test_a_trailing_stop_walked_through_minutes_is_the_stop_a_bot_resets_at_every_minute():
    """Re-set every minute where the store has an instrument's minutes, a trailing stop's level moves after each minute
    inside the bar and is checked against each: the same exits, at the same prices, as an account of a stop re-set at
    every minute's close; an hour whose minutes are missing is walked on its own prices."""
    from strategy_lab.data import minutes
    mins, hours = make_minutes(seed=5)
    kept = mins.drop(mins.index[(mins.index > "2024-01-02 06:00") & (mins.index <= "2024-01-02 18:00")])
    minutes.write("td:AAA", kept, {"source": "test"})
    p = _hourly_panel(hours, "td:AAA")
    target = np.where(np.arange(len(hours)) % 4 == 3, 0.0, 1.0)
    T = pd.DataFrame({"td:AAA": target}, index=p.index)
    got = bt.run(p, T, exits=bt.Exits(trail=0.004, trail_every="minute")).exits["td:AAA"].dropna()
    want = _stop_reset_every_minute(hours, kept, target, 0.004)
    assert len(want) > 10
    assert list(got.index) == list(want)
    np.testing.assert_allclose(got.to_numpy(), list(want.values()), rtol=1e-6)
    on_hours = bt.run(_hourly_panel(hours, "td:BBB"), T.rename(columns={"td:AAA": "td:BBB"}),
                      exits=bt.Exits(trail=0.004, trail_every="minute")).exits["td:BBB"].dropna()
    assert list(on_hours.index) != list(got.index)          # the hours' own prices alone give other exits


def _stop_reset_every_hour(hours, mins, target, trail):
    """As `_stop_reset_every_minute`, the level re-set at each hour's close from the best high of the hours since the
    fill, checked against each of the hour's minutes (a gap through it at the minute's open)."""
    exits, held, out, best = {}, False, False, None
    for k in range(1, len(hours)):
        want = target[k - 1] > 0
        if out and want:
            want = False
        elif not want:
            out = False
        if want and not held:
            held, best = True, hours["open"].iloc[k]
        elif not want:
            held = False
        if not held:
            continue
        level = best * (1 - trail)
        inside = mins[(mins.index > hours.index[k - 1]) & (mins.index <= hours.index[k])]
        if inside.empty:
            inside = hours.iloc[[k]]
        for _, bar in inside.iterrows():
            px = bar["open"] if bar["open"] <= level else level if bar["low"] <= level else None
            if px is not None:
                exits[hours.index[k]] = px
                held, out = False, True
                break
        else:
            best = max(best, hours["high"].iloc[k])
    return exits


def test_a_trailing_stop_re_set_once_a_bar_is_checked_against_its_minutes():
    """Re-set once a bar, the level holds through the bar, from the best high of the bars before; its minutes decide
    only whether and where it is hit: at the level, or at the open of a minute that gaps through it."""
    from strategy_lab.data import minutes
    mins, hours = make_minutes(seed=9)
    minutes.write("td:AAA", mins, {"source": "test"})
    p = _hourly_panel(hours, "td:AAA")
    target = np.where(np.arange(len(hours)) % 4 == 3, 0.0, 1.0)
    T = pd.DataFrame({"td:AAA": target}, index=p.index)
    got = bt.run(p, T, exits=bt.Exits(trail=0.006)).exits["td:AAA"].dropna()
    want = _stop_reset_every_hour(hours, mins, target, 0.006)
    assert len(want) > 10
    assert list(got.index) == list(want)
    np.testing.assert_allclose(got.to_numpy(), list(want.values()), rtol=1e-6)
    with pytest.raises(ValueError, match="needs a trailing stop"):
        bt.Exits(stop=0.01, trail_every="minute")


def _closes(**series):
    """A daily panel of US listings from their closes: each bar opens at the close before it (no gap), its range the
    two."""
    from strategy_lab.data.bars import Panel
    from strategy_lab.data.instruments import parse
    idx = pd.date_range("2024-01-01 21:00", periods=len(next(iter(series.values()))), freq="D", tz="UTC")
    close = pd.DataFrame({f"td:{k}": v for k, v in series.items()}, index=idx, dtype=float)
    opn = close.shift(1).fillna(close)
    vol = pd.DataFrame(1e6, index=idx, columns=close.columns)
    return Panel("1d", {i: parse(i) for i in close.columns}, opn, np.maximum(opn, close), np.minimum(opn, close), close,
                 vol, vol * close)


def test_a_trade_entered_while_the_positions_held_have_grown_buys_only_the_cash_left():
    # two seats of half: A bought at 100 doubles, then B's half of the equity is wanted with a third of it in cash
    p = _closes(AAA=[100, 100, 150, 200, 200, 200, 200, 200], BBB=[100, 100, 100, 100, 100, 110, 121, 133.1])
    T = pd.DataFrame(0.0, index=p.index, columns=p.ids)
    T.loc[p.index[0]:, "td:AAA"] = 0.5
    T.loc[p.index[3]:, "td:BBB"] = 0.5
    res = bt.run(p, T)
    assert res.weights["td:BBB"].iloc[4] == pytest.approx(1 / 3)          # all the cash, not half of the equity
    assert res.unfunded.iloc[4] == pytest.approx(1 / 6) and res.unfunded.drop(p.index[4]).eq(0.0).all()
    assert res.exposure.max() <= 1.0 + 1e-12
    # by hand, in money: A's half bought (cost on half), grown to 1.5 of equity; B bought with the third in cash
    # (cost on a third, taken from both); then B rises 33.1% and A stays
    after = (1 - 0.5 * RATE) * 1.5 * (1 - RATE / 3)
    assert (1 + res.returns).prod() == pytest.approx(after * (2 / 3 + 1.331 / 3), rel=1e-12)


def test_a_sale_goes_through_while_the_positions_hold_more_than_the_capital_and_nothing_is_added_to_them():
    # a short gone against it (short of its liquidation at twice its entry) holds 1.7 times the equity: the long sold
    # goes through whole, the new long gets nothing
    p = _closes(XXX=[100, 100, 150, 190, 190, 190], YYY=[100] * 6, ZZZ=[100] * 6)
    T = pd.DataFrame(0.0, index=p.index, columns=p.ids)
    T["td:XXX"] = -0.5
    T.loc[:p.index[2], "td:YYY"] = 0.5
    T.loc[p.index[3]:, "td:ZZZ"] = 0.5
    res = bt.run(p, T)
    k = 4                                                                  # the fills decided at close 3
    assert res.weights["td:YYY"].iloc[k] == 0.0 and res.weights["td:ZZZ"].iloc[k] == 0.0
    assert res.unfunded.iloc[k] == pytest.approx(0.5)
    held_short = abs(res.weights["td:XXX"].iloc[k])                        # its fill: half the equity back then
    assert res.exposure.iloc[k] > 1.5 > held_short                         # drifted: a short's own risk, not cut
    # by hand: sold at 100 for half the equity, the short lost 0.45 of it at 190, so the equity is 0.55 and the long's
    # half of the start is 10/11 of it: the long is sold whole, and nothing is bought
    assert res.turnover.iloc[k] == pytest.approx(0.5 / 0.55, rel=1e-9)


def test_a_settlement_stamped_after_its_hour_is_paid_by_the_position_held_into_it(tmp_path, monkeypatch):
    from strategy_lab.data import bars, store
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(bars, "STORE_DIR", tmp_path)
    p = make_panel(ids=("perp:AAAUSDT",), n=48, seed=5, freq="h", start="2024-01-01 01:00")
    k = int(np.flatnonzero(p.index == pd.Timestamp("2024-01-01 16:00", tz="UTC"))[0])    # the bar closing at 16:00
    path = tmp_path / "perp" / "funding" / "AAAUSDT.parquet"
    path.parent.mkdir(parents=True)
    settled = pd.DatetimeIndex([pd.Timestamp("2024-01-01 16:00:00.002", tz="UTC")], name="settle_time")
    pd.DataFrame({"rate": [0.001]}, index=settled).to_parquet(path)
    into = pd.DataFrame(0.0, index=p.index, columns=p.ids)
    into.iloc[: k] = 1.0                                   # held through the bar closing at 16:00, sold at its end
    held = bt.run(p, into)
    assert held.carry.iloc[k] == pytest.approx(0.001 * held.weights.iloc[k, 0]) and held.carry.iloc[k + 1] == 0.0
    after = pd.DataFrame(0.0, index=p.index, columns=p.ids)
    after.iloc[k:] = 1.0                                   # bought at 16:00: not held when the funding settled
    assert bt.run(p, after).carry.abs().sum() == 0.0


def test_a_bar_with_a_close_and_no_open_is_not_traded_and_its_move_is_held_from_the_close_before():
    p = make_panel(ids=("td:AAA",), n=30, seed=6)
    k = 12
    p.open.iloc[k] = np.nan                                # the vendor gave only its close
    late = pd.DataFrame({"td:AAA": [0.0] * (k - 1) + [1.0] * (30 - k + 1)}, index=p.index)   # decided at close k-1
    res = bt.run(p, late)
    assert res.weights["td:AAA"].iloc[k] == 0.0 and res.weights["td:AAA"].iloc[k + 1] == 1.0
    held = bt.run(p, pd.DataFrame(1.0, index=p.index, columns=p.ids))
    c = p.close["td:AAA"]
    assert held.gross.iloc[k] == pytest.approx(c.iloc[k] / c.iloc[k - 1] - 1, rel=1e-12)   # nothing of it lost
