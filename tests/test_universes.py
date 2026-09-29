import numpy as np
import pandas as pd

from strategy_lab import universes
from strategy_lab.data import store
from strategy_lab.data.bars import FIELDS, Panel
from strategy_lab.universes import top_liquid
from tests.conftest import make_panel


def test_metals_and_crude_spelled_like_pairs_are_not_currencies(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    p = make_panel(ids=("td:EUR/USD",), n=3)
    for s in ("EUR/USD", "XAU/USD", "WTI/USD"):
        store.write_bars("td", "1h", s, p.one("td:EUR/USD"))
    assert universes.fx_all("1h") == ["EUR/USD"]


def _scaled(p, dollar):
    """Same bars, dollar volume set per instrument (a constant per day)."""
    dv = pd.DataFrame({i: np.full(len(p.index), dollar[i]) for i in p.ids}, index=p.index).where(p.close.notna())
    return Panel(p.timeframe, p.instruments, **{f: (dv if f == "dollar_volume" else getattr(p, f)) for f in FIELDS})


def test_the_liquidity_window_is_in_calendar_days_on_any_timeframe():
    weekly = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), n=60, seed=1, freq="7D")
    for f in FIELDS:
        getattr(weekly, f).loc[:weekly.index[-11], "td:CCC"] = np.nan              # lists ten weeks before the end
    m = top_liquid(_scaled(weekly, {"td:AAA": 1e6, "td:BBB": 2e6, "td:CCC": 3e6}), 2, band=1.0)
    assert m.iloc[-1].tolist() == [False, True, True]                             # 30 days old, not 30 bars


def test_a_missed_bar_keeps_the_slot_and_a_name_that_stops_trading_leaves():
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), n=300, seed=3)
    for f in FIELDS:
        getattr(p, f).loc[p.index[100], "td:AAA"] = np.nan                          # one missing bar
        getattr(p, f).loc[p.index[150]:, "td:BBB"] = np.nan                         # stops trading
    m = top_liquid(_scaled(p, {"td:AAA": 3e6, "td:BBB": 2e6, "td:CCC": 1e6}), 2)
    assert m["td:AAA"].iloc[100]
    assert not m["td:BBB"].iloc[-1] and m["td:CCC"].iloc[-1]


def test_a_member_keeps_its_seat_inside_the_buffer_and_gives_it_up_below_it():
    names = [f"td:I{k}" for k in range(10)]
    p = make_panel(ids=tuple(names), n=700, seed=2)
    level = {name: 100e6 - 10e6 * k for k, name in enumerate(names)}          # I0 the most traded ... I9 the least
    dv = pd.DataFrame({i: np.full(len(p.index), level[i]) for i in names}, index=p.index)
    dv.loc[p.index[300]:, "td:I2"] = 45e6        # a member falls to rank 6: inside 1.5 x 4, it keeps its seat
    dv.loc[p.index[500]:, "td:I2"] = 25e6        # ... then to rank 8: outside, the seat goes to I4
    p = Panel(p.timeframe, p.instruments, **{f: (dv if f == "dollar_volume" else getattr(p, f)) for f in FIELDS})

    buffered, plain = top_liquid(p, 4), top_liquid(p, 4, band=1.0)
    at_rank6, at_rank8 = p.index[450], p.index[-1]
    assert buffered.loc[at_rank6, "td:I2"] and not buffered.loc[at_rank6, "td:I4"]
    assert not plain.loc[at_rank6, "td:I2"] and plain.loc[at_rank6, "td:I4"]
    assert not buffered.loc[at_rank8, "td:I2"] and buffered.loc[at_rank8, "td:I4"]
    assert (buffered.loc[p.index[100]:].sum(axis=1) == 4).all()             # always four seats, all filled


def test_a_name_that_stopped_trading_never_fills_an_empty_seat():
    p = make_panel(ids=("perp:AAAUSDT", "perp:BBBUSDT", "perp:CCCUSDT"), n=300, seed=2)
    for f in FIELDS:
        getattr(p, f).loc[p.index[150]:, "perp:CCCUSDT"] = np.nan         # its market stops: no bars from day 150
    m = top_liquid(p, 3, 30)
    assert m["perp:CCCUSDT"].iloc[60:140].all()                            # held while it traded
    assert not m["perp:CCCUSDT"].iloc[200:].any()                          # then its seat stays empty


def test_a_market_that_stops_gives_its_seat_at_once_to_the_best_name_outside():
    p = make_panel(ids=("perp:AAAUSDT", "perp:BBBUSDT", "perp:CCCUSDT"), n=300, seed=3)
    p = _scaled(p, {"perp:AAAUSDT": 3e6, "perp:BBBUSDT": 2e6, "perp:CCCUSDT": 1e6})
    for f in FIELDS:
        getattr(p, f).loc[p.index[150]:, "perp:BBBUSDT"] = np.nan          # delisted: its last bar is 2021-06-02
    m = top_liquid(p, 2, 30)
    assert m.loc["2021-05-20", "perp:BBBUSDT"].all() and not m.loc["2021-05-20", "perp:CCCUSDT"].any()
    assert not m.loc["2021-06-03":, "perp:BBBUSDT"].any()                   # nothing held after its last bar
    assert m.loc["2021-06-03":"2021-06-30", "perp:CCCUSDT"].all()          # the seat is taken at once, not in July
    assert m.loc["2021-06-03":, "perp:AAAUSDT"].all()                       # the other seat is not touched


def test_a_market_delisted_for_a_while_gives_its_seat_at_once_and_holds_nothing_until_listed_anew(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    p = make_panel(ids=("perp:AAAUSDT", "perp:BBBUSDT", "perp:CCCUSDT"), n=300, seed=3)
    for f in FIELDS:
        getattr(p, f).loc[p.index[150]:p.index[199], "perp:BBBUSDT"] = np.nan   # between two listings
    p = _scaled(p, {"perp:AAAUSDT": 3e6, "perp:BBBUSDT": 2e6, "perp:CCCUSDT": 1e6})
    store.write_meta("perp", "1d", "BBBUSDT", {"delisted_periods": [[str(p.index[149]), str(p.index[200])]]})
    m = top_liquid(p, 2, 30)
    assert not m.loc["2021-06-03":"2021-07-22", "perp:BBBUSDT"].any()
    assert m.loc["2021-06-03":"2021-06-30", "perp:CCCUSDT"].all()          # the seat taken at once


def test_the_etf_top_n_holds_the_most_traded_of_the_34_by_the_stock_lists_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    ids = ("td:DIA", "td:IWM", "td:QQQ", "td:SPY")
    p = _scaled(make_panel(ids=ids, n=200, seed=4), {"td:SPY": 4e9, "td:QQQ": 3e9, "td:IWM": 2e9, "td:DIA": 1e9})
    for i in ids:
        store.write_bars("td", "1d", i.removeprefix("td:"), p.one(i))
    store.write_bars("td", "1d", "AAPL", p.one("td:SPY"))                 # a stock is not one of the 34 ETFs
    u = universes.resolve("etf_top3", "1d")
    assert sorted(u.ids) == sorted(ids) and u.size == 3
    assert u.member(p).iloc[-1].to_dict() == {"td:DIA": False, "td:IWM": True, "td:QQQ": True, "td:SPY": True}
