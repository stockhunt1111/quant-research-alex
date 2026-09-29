"""The data check reads series written to a store and finds what a run would otherwise meet in silence: a name held
without bars, a hole against the market's calendar, a dead market, a history shorter than the vendor's, a token of
gold or a stablecoin in a crypto list."""
import numpy as np
import pandas as pd
import pytest

from strategy_lab import universes
from strategy_lab.data import calendars as cal
from strategy_lab.data import check, store

COLS = ["open", "high", "low", "close", "volume", "dollar_volume"]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(check, "REFERENCE_DIR", tmp_path / "reference")
    (tmp_path / "reference").mkdir()
    for f in (check.series, check._vendor_first, check._perp_contracts, check._warn_once):
        f.cache_clear()
    yield
    for f in (check.series, check._vendor_first, check._perp_contracts, check._warn_once):
        f.cache_clear()


def _bars(index, price, volume=1e6) -> pd.DataFrame:
    close = pd.Series(price, index=index, dtype=float)
    vol = pd.Series(volume, index=index, dtype=float)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": vol,
                         "dollar_volume": close * vol})[COLS]


def _funding(sym: str) -> None:
    """One funding settlement stored for a perp: the engine refuses a perp with none."""
    folder = store.STORE_DIR / "perp" / "funding"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"rate": [0.0001]}, index=pd.DatetimeIndex(["2024-01-01 08:00"], tz="UTC", name="settle_time")) \
        .to_parquet(folder / f"{sym}.parquet")


def _walk(n, start=100.0, seed=0):
    return start * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.02, n)))


def test_a_hole_and_a_dead_tail_in_a_coins_hourly_bars_are_found():
    idx = pd.date_range("2024-01-01 01:00", periods=24 * 20, freq="h", tz="UTC")
    bars = _bars(idx, _walk(len(idx)))
    bars = bars.drop(idx[100:130])                                  # 30 hourly bars the exchange never gave
    tail = idx[-24 * 4:]                                            # the last four days: settled, still printing
    bars.loc[tail, "close"] = bars.loc[tail[0], "close"]
    bars.loc[tail, ["volume", "dollar_volume"]] = 0.0
    store.write_bars("perp", "1h", "DEADUSDT", bars)
    s = check.series("perp:DEADUSDT", "1h")
    assert [h[2] for h in s.holes] == [30]
    assert len(s.dead) == 1 and s.dead[0][1] == check._day(idx[-1])


def test_a_session_missing_from_a_stocks_daily_bars_is_a_hole_of_one_nyse_session():
    sessions = cal.nyse_sessions(pd.Timestamp("2025-06-02", tz="UTC"), pd.Timestamp("2025-07-31", tz="UTC"))
    closes = pd.DatetimeIndex(sessions["market_close"])             # July 3 closes early, July 4 is a holiday
    gone = closes[20]
    store.write_bars("td", "1d", "AAA", _bars(closes.drop(gone), _walk(len(closes) - 1)))
    s = check.series("td:AAA", "1d")
    assert s.holes == [(sessions.index[20], sessions.index[20], 1)]


def test_a_name_the_list_holds_with_no_hourly_bars_is_missing_for_the_days_it_is_held(monkeypatch):
    days = pd.date_range("2024-01-01", periods=60, freq="D")
    for sym in ("AAAUSDT", "BBBUSDT"):
        store.write_bars("perp", "1d", sym, _bars(days.tz_localize("UTC") + pd.Timedelta(days=1), _walk(60, seed=1)))
    hourly = pd.date_range("2024-01-01 01:00", periods=24 * 60, freq="h", tz="UTC")
    store.write_bars("perp", "1h", "AAAUSDT", _bars(hourly, _walk(len(hourly), seed=2)))
    mask = pd.DataFrame({"perp:AAAUSDT": True, "perp:BBBUSDT": days >= days[-30]}, index=days)
    monkeypatch.setattr(check, "held", lambda u: mask)
    for sym in ("AAAUSDT", "BBBUSDT"):
        _funding(sym)
    found, summary = check.check_list("crypto_top10", ("1h",))
    missing = [f for f in found if f.kind == "missing"]
    assert [(f.instrument, f.days, f.detail) for f in missing] == [("perp:BBBUSDT", 30, "no 1h bars")]
    assert summary["1h"]["recent_share"] == pytest.approx(30 / 90)


def test_a_token_priced_as_gold_and_a_stablecoin_are_not_crypto_and_a_coin_is():
    days = pd.date_range("2025-01-01", periods=90, freq="D", tz="UTC")
    gold = _walk(90, start=2600.0, seed=3)
    store.write_bars("td", "1d", "XAU/USD", _bars(days + pd.Timedelta(hours=22), gold))
    store.write_bars("perp", "1d", "PAXGUSDT", _bars(days + pd.Timedelta(days=1), gold * 1.004))
    store.write_bars("perp", "1d", "USDCUSDT", _bars(days + pd.Timedelta(days=1), 1.0 + np.linspace(-.002, .002, 90)))
    store.write_bars("perp", "1d", "BTCUSDT", _bars(days + pd.Timedelta(days=1), _walk(90, start=60000.0, seed=4)))
    assert check.not_crypto("perp:PAXGUSDT").startswith("a token of gold")
    assert check.not_crypto("perp:USDCUSDT").startswith("a stablecoin")
    assert check.not_crypto("perp:BTCUSDT") is None


def test_a_contract_binance_does_not_class_as_a_coin_is_not_crypto():
    pd.DataFrame({"symbol": ["TSLAUSDT"], "underlying_type": ["EQUITY"], "contract_type": ["TRADIFI_PERPETUAL"],
                  "status": ["TRADING"], "onboard_date": [0]}).to_csv(check.REFERENCE_DIR / "binance_perp_contracts.csv",
                                                                         index=False)
    assert check.not_crypto("perp:TSLAUSDT") == "Binance classes the contract as EQUITY, not a coin"


def test_hourly_bars_starting_after_the_listing_or_the_vendors_first_bar_are_short():
    days = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
    store.write_bars("perp", "1d", "NEWUSDT", _bars(days + pd.Timedelta(days=1), _walk(120)))
    hourly = pd.date_range("2024-03-01 01:00", periods=24 * 30, freq="h", tz="UTC")
    store.write_bars("perp", "1h", "NEWUSDT", _bars(hourly, _walk(len(hourly))))
    assert check._short("perp:NEWUSDT", "1h") == ("1h bars from 2024-03-01, daily bars from 2024-01-02 "
                                                  "(both start at the listing)")

    sessions = cal.nyse_sessions(pd.Timestamp("2024-03-01", tz="UTC"), pd.Timestamp("2024-03-29", tz="UTC"))
    closes = pd.DatetimeIndex(sessions["market_close"])
    listed = cal.nyse_sessions(pd.Timestamp("2018-06-01", tz="UTC"), pd.Timestamp("2024-03-29", tz="UTC"))
    for source, symbol in (("td", "BBB"), ("sh", "CCC"), ("sh", "NEW")):
        daily = closes if symbol == "NEW" else pd.DatetimeIndex(listed["market_close"])   # NEW: listed 2024-03-01
        store.write_bars(source, "1d", symbol, _bars(daily, _walk(len(daily))))
        store.write_bars(source, "1h", symbol, _bars(closes, _walk(len(closes))))
    # a US listing's hourly sessions go back to 2019-01-07, where Databento's minutes begin, whatever Twelve Data has
    assert check._short("td:BBB", "1h") == "1h bars from 2024-03-01, its sessions from 2019-01-07"
    assert check._short("sh:CCC", "1h") == "1h bars from 2024-03-01, its sessions from 2019-01-07"
    assert check._short("sh:NEW", "1h") is None
    store.write_meta("sh", "1h", "CCC", {"daily_basis": {"dropped_through": "2024-02-28"}})
    assert "dropped: they did not match the daily bars (another company)" in check._short("sh:CCC", "1h")

    fx = pd.date_range("2024-03-01 01:00", periods=24 * 20, freq="h", tz="UTC")       # FX: Twelve Data's first bar
    store.write_bars("td", "1h", "EUR/USD", _bars(fx, _walk(len(fx))))
    pd.DataFrame({"symbol": ["EUR/USD"], "interval": ["1h"], "first": ["2020-01-30 00:00:00+00:00"],
                  "asked": ["2026-09-24"]}).to_csv(check.REFERENCE_DIR / "twelvedata_first_bars.csv", index=False)
    assert check._short("td:EUR/USD", "1h") == "1h bars from 2024-03-01, the vendor's from 2020-01-30"


def test_a_hole_most_of_the_list_shares_is_named_once_and_a_names_own_hole_on_its_own():
    days = pd.date_range("2024-01-01", periods=30, freq="D")
    shared = (days[10], days[11], 2)
    spans = {f"td:S{k}": (check.Series(days[0], days[-1], [shared]), pd.Series(True, index=days)) for k in range(4)}
    spans["td:S0"][0].holes.append((days[20], days[22], 3))
    found = check._holes("us_stocks_top10", "1d", spans, days[-1])
    assert [(f.instrument, f.detail) for f in found] == [
        ("4 of 4 names", "the same hole in most of the list: 2024-01-11..2024-01-12, 2 sessions"),
        ("td:S0", "1 hole(s) while held; the longest 2024-01-21..2024-01-23, 3 sessions")]
    # a hole every name shares has no row in the list's daily panel: it is still held as the day before
    gap = days.delete([10, 11])
    spans = {f"td:S{k}": (check.Series(days[0], days[-1], [shared]), pd.Series(True, index=gap)) for k in range(4)}
    assert [f.instrument for f in check._holes("us_stocks_top10", "1d", spans, days[-1])] == ["4 of 4 names"]


def test_the_crypto_lists_take_no_name_the_check_finds_priced_as_gold_or_a_dollar(monkeypatch):
    """The lists' filter, by name and by Binance's class, against the check's own test, by price."""
    monkeypatch.setattr(universes, "REFERENCE_DIR", check.REFERENCE_DIR)
    days = pd.date_range("2025-01-01", periods=90, freq="D", tz="UTC")
    gold = _walk(90, start=2600.0, seed=5)
    store.write_bars("td", "1d", "XAU/USD", _bars(days + pd.Timedelta(hours=22), gold))
    prices = {"PAXGUSDT": gold * 1.003, "XAUTUSDT": gold * 0.998, "USDCUSDT": np.full(90, 1.0001),
              "BTCUSDT": _walk(90, start=60000.0, seed=6), "SOLUSDT": _walk(90, start=150.0, seed=7)}
    for sym, px in prices.items():
        store.write_bars("perp", "1d", sym, _bars(days + pd.Timedelta(days=1), px))
    pd.DataFrame({"symbol": list(prices), "underlying_type": "COIN", "contract_type": "PERPETUAL",
                  "status": "TRADING", "onboard_date": 0}).to_csv(check.REFERENCE_DIR / "binance_perp_contracts.csv",
                                                                   index=False)
    flagged = {s for s in prices if check.not_crypto(f"perp:{s}")}
    assert flagged == {"PAXGUSDT", "XAUTUSDT", "USDCUSDT"}
    assert set(universes.crypto_candidates("1d")) == set(prices) - flagged


def test_a_price_that_resumes_far_from_its_last_trade_after_a_halt_is_two_markets_joined():
    days = pd.date_range("2025-06-01", periods=60, freq="D", tz="UTC") + pd.Timedelta(days=1)
    old = _bars(days[:30], _walk(30, start=0.05, seed=8))
    dead = _bars(days[30:40], np.full(10, old["close"].iloc[-1]), volume=0.0)    # delisted, still printing
    new = _bars(days[40:], _walk(20, start=0.006, seed=9))                        # a later coin under the same name
    store.write_bars("perp", "1d", "PUMPUSDT", pd.concat([old, dead, new]))
    s = check.series("perp:PUMPUSDT", "1d")
    assert [(b.date(), a.date()) for b, _, a, _ in s.joins] == [(check._day(days[29]).date(), check._day(days[40]).date())]

    calm = pd.concat([old, dead, _bars(days[40:], _walk(20, start=float(old["close"].iloc[-1]) * 1.1, seed=9))])
    store.write_bars("perp", "1d", "CALMUSDT", calm)                              # resumes near where it stopped
    assert check.series("perp:CALMUSDT", "1d").joins == []


def test_a_perp_with_no_funding_stored_is_missing_before_a_run_fails_on_it(monkeypatch):
    days = pd.date_range("2024-01-01", periods=20, freq="D")
    store.write_bars("perp", "1d", "NOFUNDUSDT", _bars(days.tz_localize("UTC") + pd.Timedelta(days=1), _walk(20)))
    monkeypatch.setattr(check, "held", lambda u: pd.DataFrame({"perp:NOFUNDUSDT": True}, index=days))
    found, _ = check.check_list("crypto_top10", ("1d",))
    assert [(f.kind, f.days) for f in found if "funding" in f.detail] == [("missing", 20)]
