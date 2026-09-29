"""The market's index beside a result's buy & hold: each list's market's but FX's, an instrument alone's only where an
index measures its own market (crude's quote has its future), bought on the record's first day and held as buy & hold
holds one instrument, with its dividends; SPY's comes out at its published total return."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_lab import db, lists, universes
from strategy_lab.config import COSTS
from strategy_lab.data import bars, store
from strategy_lab.data.bars import FIELDS

from server import market_index
from tests.conftest import make_panel

EQUITY = (COSTS["us_equity"].commission_bps + COSTS["us_equity"].half_spread_bps) / 1e4
# SPY's calendar-year total returns, dividends reinvested (totalrealreturns.com/n/SPY, read 2026-09-28; another
# source gives -36.81% and -18.17%): without its dividends SPY lost 38.3% and 19.4% in those years
SPY_TOTAL_RETURN = {2008: -0.3679, 2022: -0.1818}


def test_every_list_but_fx_has_its_markets_index():
    for x in lists.OURS:
        assert (market_index.of(x.universe, None) is None) == (x.market == "FX"), x.universe
    assert market_index.of("an_ad_hoc_list", None) is None


def test_an_instrument_alone_has_an_index_only_where_one_measures_its_own_market():
    spy, btc = market_index.BY_MARKET["Stocks"], market_index.BY_MARKET["Crypto"]
    for universe in ("us_stocks_top100", "us_stocks_mcap10"):
        assert market_index.of(universe, "sh:AAPL") == spy
    for universe in ("crypto_top100", "crypto_mcap10"):
        assert market_index.of(universe, "perp:ETHUSDT") == btc
        assert market_index.of(universe, "perp:BTCUSDT") is None           # a coin's perp is held on its spot pair
    etfs = {f"td:{s}" for s in universes.ETF_CORE}
    assert market_index.US_STOCK_ETFS <= etfs
    for universe in ("etf_core", "stockhunt_etfs"):
        for i in etfs:
            want = None if i == spy.id or i not in market_index.US_STOCK_ETFS else spy
            assert market_index.of(universe, i) == want, i
    # a metal, a future or a currency pair is its own market: nothing beside it but its own buy & hold, or cash; crude's
    # spot quote, which cannot be held, has crude held through its future
    for i in [f"td:{s}" for s in universes.STOCKHUNT_COMMODITIES]:
        got = market_index.of("stockhunt_commodities", i)
        assert (got is None) != (i == "td:WTI/USD"), i
    assert market_index.of("stockhunt_commodities", "td:WTI/USD").id == "cme:CL"
    for i in universes.CME_FUTURES:
        assert market_index.of("cme_futures", i) is None, i
    for s in universes.FX_MAJORS:
        assert market_index.of("fx_majors", f"td:{s}") is None, s


@pytest.mark.skipif(not all(store.path(src, "1d", s).exists() for src, s in
                            (("td", "WTI/USD"), ("cme", "CL"), ("td", "XAU/USD"))),
                    reason="crude's quote, its future and gold's quote are not in this checkout's store")
def test_what_stands_beside_crude_alone_moves_with_crude_and_gold_did_not(monkeypatch):
    from strategy_lab.data import spreads
    monkeypatch.setattr(spreads, "STORE_DIR", store.STORE_DIR)          # the quotes are bought at their broker's spread
    days = pd.date_range("2016-01-01", "2026-08-31", freq="D", tz="UTC")
    close = store.read_bars("td", "1d", "WTI/USD", ["close"])["close"]          # buy & hold holds no crude
    crude = close.pct_change().set_axis(close.index.normalize()).reindex(days)
    beside = market_index.daily(market_index.of("stockhunt_commodities", "td:WTI/USD"), days, "next_open")
    gold = market_index.daily(market_index.BY_MARKET["Commodities"], days, "next_open")
    traded = crude.notna() & (beside != 0) & (gold != 0)
    assert crude[traded].corr(beside[traded]) > 0.8
    assert abs(crude[traded].corr(gold[traded])) < 0.2


def _stored(monkeypatch, tmp_path, panel, symbol: str) -> None:
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(bars, "STORE_DIR", tmp_path)
    monkeypatch.setattr(market_index, "_panels", {})
    store.write_bars("td", "1d", symbol, pd.DataFrame({f: getattr(panel, f).iloc[:, 0] for f in FIELDS}))


def test_the_index_is_bought_on_the_records_first_day_and_held_with_its_dividends(tmp_path, monkeypatch):
    p = make_panel(ids=("td:SPY",), n=300, seed=4, freq="B")                # weekdays: a weekend has no bar
    _stored(monkeypatch, tmp_path, p, "SPY")
    ex = p.index[150]
    path = bars.dividends_path("td", "SPY")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"amount": [1.5]}, index=pd.DatetimeIndex([ex.tz_localize(None).normalize()], name="ex_date")) \
        .to_parquet(path)
    days = pd.date_range(p.index[100].normalize(), p.index[-1].normalize(), freq="D", tz="UTC")   # a record's days
    got = market_index.daily(market_index.BY_MARKET["Stocks"], days, "next_open")
    c, o = p.close["td:SPY"].to_numpy(), p.open["td:SPY"].to_numpy()
    paid = np.where(p.index == ex, 1.5, 0.0)
    # decided at the close before the first day, bought at that day's open for one purchase's cost, as the record's
    # positions held that day were; then its closes and the dividend on its ex-date
    held = c[100] / o[100] * np.prod((c[101:] + paid[101:]) / c[100:-1]) * (1 - EQUITY)
    assert got.index.equals(days)
    assert np.isclose(float((1 + got).prod()), held)
    assert got.iloc[0] != 0.0 and (got[got.index.dayofweek >= 5] == 0.0).all()
    assert market_index.figures(got)["start"] == str(days[0].date())
    assert set(market_index.figures(got)) == set(db.BENCHMARK_FIGURES)      # every figure a buy & hold shows


@pytest.mark.skipif(not store.path("td", "1d", "SPY").exists() or not bars.dividends_path("td", "SPY").exists(),
                    reason="SPY's daily bars and dividends are not in this checkout's store")
def test_spy_held_as_the_index_earns_its_published_total_return():
    close = store.read_bars("td", "1d", "SPY", ["close"])["close"]
    for year, published in SPY_TOTAL_RETURN.items():
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D", tz="UTC")       # a record of the year
        got = float((1 + market_index.daily(market_index.BY_MARKET["Stocks"], days, "next_open")).prod() - 1)
        # bought at the year's first open, not at the close before it: that gap and one purchase's cost apart
        assert abs(got - published) < 0.004, (year, got)
        yearly = close.groupby(close.index.year).last()
        assert abs(yearly[year] / yearly[year - 1] - 1 - published) > 0.01      # the price alone misses by more
