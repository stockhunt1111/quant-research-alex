import numpy as np
import pandas as pd
import pytest

from strategy_lab import lookahead, market
from strategy_lab.data.bars import Panel
from strategy_lab.strategy import fill_signals, load
from tests.conftest import MARKET_INDEX, make_panel

READERS = ["market_regime", "regime_ema_trail"]


def _hourly(ids=("td:AAA", "td:BBB"), days=500, seed=7) -> Panel:
    """Seven hourly bars a weekday closing 14:00..20:00 UTC, before the synthetic index's daily close at 21:00."""
    p = make_panel(ids=ids, n=days * 24, seed=seed, freq="h", start="2021-01-04 00:00")
    keep = (p.index.dayofweek < 5) & (p.index.hour >= 14) & (p.index.hour <= 20)
    return Panel("1h", p.instruments, **{f: getattr(p, f)[keep] for f in ("open", "high", "low", "close", "volume",
                                                                            "dollar_volume")})


def _index(monkeypatch, closes: pd.Series, tag: str) -> None:
    monkeypatch.setattr(market, "_index_closes", lambda index_id: (0, closes, tag))


def test_a_bar_sees_the_markets_close_of_a_day_only_once_that_day_has_closed():
    s = load("market_regime")
    p = _hourly()
    w = s.target(p, {"n_days": 50, "decay": False, "vol_parity": False, "long_only": True})["td:AAA"]
    moved = w.ne(w.shift()).to_numpy() & (np.arange(len(w)) > 0)
    first_of_day = ~pd.Series(w.index.normalize()).duplicated().to_numpy()
    # every change is on a day's first bar: the index's close of the day before, stamped 21:00, decides the whole day
    assert moved.any() and (~moved | first_of_day).all()


@pytest.mark.parametrize("name", READERS)
def test_positions_do_not_change_when_the_markets_future_is_cut_off(name, monkeypatch):
    s = load(name)
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), n=900, seed=5)
    cut = p.index[600]
    configs = s.configs(len(p.ids))
    full = [s.target(p, c) for c in configs]
    _index(monkeypatch, MARKET_INDEX.loc[:cut], "cut")
    held = 0
    for c, whole in zip(configs, full):
        part = s.target(p, c)
        pd.testing.assert_frame_equal(part.loc[:cut], whole.loc[:cut])
        held += int((part.loc[:cut] != 0).to_numpy().sum())
    assert held > 0


@pytest.mark.parametrize("name, params", [
    ("market_regime", {"n_days": 50, "decay": True, "vol_parity": False}),
    ("regime_ema_trail", {"n_days": 50, "ema_days": 10, "high_days": 20})])
def test_the_short_side_is_the_long_side_on_the_name_and_the_market_turned_upside_down(name, params, monkeypatch):
    s = load(name)
    p = make_panel(ids=("td:AAA",), n=1200, seed=4)
    k = 2.0 * max(float(p.high.max().max()), float(MARKET_INDEX.max()))
    turned = Panel(p.timeframe, p.instruments, open=k - p.open, high=k - p.low, low=k - p.high, close=k - p.close,
                   volume=p.volume, dollar_volume=p.dollar_volume)
    both = s.fn(p.one("td:AAA"), **params, long_only=False)
    _index(monkeypatch, k - MARKET_INDEX, "turned")
    short = s.fn(turned.one("td:AAA"), **params, long_only=True)
    assert (both < 0).any() and (both > 0).any()
    pd.testing.assert_series_equal(both.clip(upper=0.0), -short, check_names=False)


def test_decay_holds_a_trend_whole_for_ninety_days_then_less_each_day_down_to_thirty_percent(monkeypatch):
    days = pd.date_range("2020-01-01 21:00", periods=600, freq="D", tz="UTC")
    _index(monkeypatch, pd.Series(100 * np.exp(0.001 * np.arange(600)), index=days), "rising")
    bars = make_panel(ids=("td:AAA",), n=600, start="2020-01-01 21:00").one("td:AAA")
    w = load("market_regime").fn(bars, n_days=50, decay=True, vol_parity=False, long_only=True)
    trend = w[w > 0]
    assert w.index[w > 0][0] == days[49]                   # on from the day the 50-day average has its 50 days
    assert (trend.iloc[:90] == 1.0).all()
    assert (trend.iloc[90:270].diff().iloc[1:] < 0).all() and trend.iloc[90] < 1.0
    assert np.allclose(trend.iloc[269:], 0.3) and (trend >= 0.3 - 1e-12).all()


def test_vol_parity_holds_a_name_at_most_whole_and_less_when_it_swings_more_than_its_market():
    s = load("market_regime")
    bars = make_panel(ids=("td:AAA",), n=900, seed=6).one("td:AAA")
    plain = s.fn(bars, n_days=50, decay=False, vol_parity=False, long_only=True)
    sized = s.fn(bars, n_days=50, decay=False, vol_parity=True, long_only=True)
    on = (plain > 0) & (sized > 0)
    assert ((sized >= 0) & (sized <= plain)).all()
    # the synthetic name's daily swings (1%) and the index's (1%) are alike: days either way
    assert (sized[on] < 1.0).any() and (sized[on] == 1.0).any()


@pytest.mark.parametrize("name", READERS)
def test_a_rule_reading_the_market_refuses_a_market_without_an_index(name):
    s = load(name)
    for pair in ("td:EUR/USD", "td:XAU/USD", "cme:GC"):
        p = make_panel(ids=(pair,), n=300)
        with pytest.raises(ValueError, match="no market index"):
            s.fn(p.one(pair), **s.signal_params(s.configs(1)[0]))


def test_positions_kept_on_disk_are_kept_under_the_index_they_were_worked_out_on(monkeypatch):
    s = load("market_regime")
    p = make_panel(ids=("td:AAA", "td:BBB"), n=700, seed=9)
    configs = s.configs(2)[:4]
    first: dict = {}
    fill_signals(s, p, configs, first)                     # worked out and kept on disk
    _index(monkeypatch, MARKET_INDEX * np.exp(0.3 * np.sin(np.arange(len(MARKET_INDEX)) / 40.0)), "restated")
    again: dict = {}
    fill_signals(s, p, configs, again)
    moved = False
    for c in configs:
        sp = s.signal_params(c)
        for i in p.ids:
            key = (repr(sorted(sp.items())), i)
            assert (again[key].astype(float) == s.fn(p.one(i), **sp).to_numpy()).all()   # none read back from before
            moved |= bool((again[key] != first[key]).any())
    assert moved                                           # the restated index moves the positions


@pytest.mark.parametrize("name", READERS)
def test_rules_reading_the_market_pass_the_truncation_test_on_hourly_bars(name):
    s = load(name)
    assert lookahead.check(s, _hourly(days=420), s.configs(2)[0], cuts=(0.6, 0.9)) > 0
