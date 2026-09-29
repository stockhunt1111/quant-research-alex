"""Vendor refresh rules on mocked responses (no network): 429 stops, empty answers are remembered, a non-JSON
error is reported, the request ceiling holds, and a split on the overlap triggers a full re-fetch."""
import json

import numpy as np
import pandas as pd
import pytest

from strategy_lab import config
from strategy_lab.data import bars, refresh, store
from strategy_lab.data import calendars as cal


class _Resp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(bars, "STORE_DIR", tmp_path / "store")     # where a backtest reads funding from
    monkeypatch.setattr(refresh, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(config, "REFERENCE_DIR", tmp_path / "reference")      # the saved contract list
    monkeypatch.setattr(refresh, "env", lambda k: {"TWELVEDATA_API_KEY": "k", "TWELVEDATA_REQUESTS_PER_MINUTE": "1000"}.get(k))


def test_ceiling_stops_the_run():
    b = refresh.Budget("x", per_minute=None, max_requests=2)
    b.acquire()
    b.acquire()
    with pytest.raises(refresh.BudgetExceeded):
        b.acquire()


def test_twelvedata_429_stops_empty_is_empty_non_json_is_reported(tmp_path):
    b = refresh.Budget("twelvedata", per_minute=None, max_requests=None)
    with pytest.raises(refresh.RateLimited):
        refresh._td_get(_Session([_Resp(429, {"status": "error", "code": 429, "message": "limit"})]), {}, b, {})
    assert refresh._td_get(_Session([_Resp(200, {"status": "error", "code": 400, "message": "No data is available on the specified dates"})]), {}, b, {}) == []
    with pytest.raises(RuntimeError, match="twelvedata 502"):
        refresh._td_get(_Session([_Resp(502, "<html>bad gateway</html>")]), {}, b, {})
    ledger = (tmp_path / "logs" / "requests" / "twelvedata.jsonl").read_text()
    assert "apikey" not in ledger and "k" not in json.loads(ledger.splitlines()[0]).values()


def test_binance_429_stops():
    b = refresh.Budget("binance", per_minute=None, max_requests=None)
    with pytest.raises(refresh.RateLimited):
        refresh._binance_get(_Session([_Resp(429, [])]), "https://x/klines", {}, b, {})


def _daily(dates, closes):
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="America/New_York").replace(hour=16).tz_convert("UTC") for d in dates])
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0, "dollar_volume": c})


def _td_values(dates, closes):
    return [{"datetime": d, "open": str(c), "high": str(c), "low": str(c), "close": str(c), "volume": "1"}
            for d, c in zip(dates, closes)]


def test_split_on_the_overlap_refetches_the_whole_series(monkeypatch):
    old_dates = ["2026-08-03", "2026-08-04", "2026-08-05"]
    store.write_bars("td", "1d", "ACME", _daily(old_dates, [100.0, 102.0, 104.0]))
    tail = _td_values(["2026-08-04", "2026-08-05", "2026-08-06"], [51.0, 52.0, 53.0])            # 2:1 split restated
    full = _td_values(["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06"], [50.0, 51.0, 52.0, 53.0])
    sess = _Session([_Resp(200, {"status": "ok", "values": tail}), _Resp(200, {"status": "ok", "values": full})])
    monkeypatch.setattr(refresh.requests, "Session", lambda: sess)
    monkeypatch.setattr(refresh, "td_series_to_refresh", lambda syms: [("ACME", "1d")])
    rep = refresh.refresh_twelvedata(["ACME"], dry_run=False)
    bars = store.read_bars("td", "1d", "ACME")
    assert rep["series_refetched"] == 1 and len(sess.calls) == 2
    assert list(bars["close"]) == [50.0, 51.0, 52.0, 53.0]


def test_matching_overlap_appends_only_the_new_bars(monkeypatch):
    store.write_bars("td", "1d", "ACME", _daily(["2026-08-03", "2026-08-04", "2026-08-05"], [100.0, 102.0, 104.0]))
    tail = _td_values(["2026-08-04", "2026-08-05", "2026-08-06"], [102.0, 104.0, 106.0])
    sess = _Session([_Resp(200, {"status": "ok", "values": tail})])
    monkeypatch.setattr(refresh.requests, "Session", lambda: sess)
    monkeypatch.setattr(refresh, "td_series_to_refresh", lambda syms: [("ACME", "1d")])
    rep = refresh.refresh_twelvedata(["ACME"], dry_run=False)
    assert rep["bars_added"] == 1 and list(store.read_bars("td", "1d", "ACME")["close"]) == [100.0, 102.0, 104.0, 106.0]


def test_a_commodity_day_closes_17_new_york_and_its_weekend_stubs_are_dropped():
    rows = [{"datetime": d, "open": "1", "high": "2", "low": "0.5", "close": "1.5"}
            for d in ["2026-01-09", "2026-01-10", "2026-01-11", "2026-01-12",       # Fri, Sat, Sun, Mon in winter
                      "2026-09-19", "2026-09-20", "2026-09-21", "2026-09-24"]]      # stubs, Mon, a day still open
    bars = refresh._td_normalise("XAU/USD", "1d", refresh._td_frame(rows), pd.Timestamp("2026-09-24 12:00", tz="UTC"))
    assert list(bars.index) == [pd.Timestamp("2026-01-09 22:00", tz="UTC"), pd.Timestamp("2026-01-12 22:00", tz="UTC"),
                                pd.Timestamp("2026-09-21 21:00", tz="UTC")]
    # the vendor's open beyond its own day's range (platinum's 2019-04-03: 846.5 under a low of 848.375) is held in it
    off = [{"datetime": "2019-04-03", "open": "846.5", "high": "881.905", "low": "848.375", "close": "874.78"}]
    bar = refresh._td_normalise("XPT/USD", "1d", refresh._td_frame(off), pd.Timestamp("2026-09-24", tz="UTC"))
    assert bar[["open", "high", "low", "close"]].iloc[0].tolist() == [848.375, 881.905, 848.375, 874.78]


def test_a_commoditys_days_are_built_from_its_hours_as_an_fx_pairs_and_the_vendors_stand_where_hours_are_missing():
    hours = pd.date_range("2024-06-09 22:00", "2024-06-11 21:00", freq="h", tz="UTC")     # Sun 18:00 -> Tue 17:00 NY
    v = pd.Series(range(1, len(hours) + 1), index=hours, dtype=float)
    h1 = pd.DataFrame({"open": v, "high": v + 1, "low": v - 1, "close": v, "volume": 0.0, "dollar_volume": 0.0})
    monday = h1.iloc[[23]].assign(open=5.0, high=50.0, low=0.5, close=7.0)       # the vendor's Monday, a fixing's close
    wednesday = monday.set_axis([pd.Timestamp("2024-06-12 21:00", tz="UTC")]).assign(close=9.0)   # no hours that day
    for s in ("XAU/USD", "EUR/USD"):
        store.write_bars("td", "1h", s, h1)
        store.write_bars("td", "1d", s, pd.concat([monday, wednesday]))
        store.write_meta("td", "1h", s, {"last": str(hours[-1])})
        store.write_meta("td", "1d", s, {"last": str(hours[-1])})
        refresh._hourly_written(s)
    assert store.read_bars("td", "1d", "XAU/USD")["close"].tolist() == [24.0, 48.0, 9.0]   # built, built, the vendor's
    assert store.read_bars("td", "1d", "EUR/USD")["close"].tolist() == [24.0, 48.0]
    assert set(refresh.td_series_to_refresh(["XAU/USD", "EUR/USD"])) == {("XAU/USD", "1h"), ("XAU/USD", "1d"),
                                                                         ("EUR/USD", "1h")}


def _hours(first: str, last: str) -> pd.DatetimeIndex:
    """A spot quote's hours in winter between two closes: every hour but the one of the daily pause (22:00-23:00 UTC)."""
    h = pd.date_range(first, last, freq="h", tz="UTC")
    return h[h.hour != 23]


def _vendor_bars(hours: pd.DatetimeIndex, level) -> pd.DataFrame:
    close = pd.Series(level, index=hours, dtype=float)
    return pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 0.0,
                         "dollar_volume": 0.0})


def _exness_month(tmp_path, monkeypatch, symbol: str, mids: pd.Series) -> None:
    """Exness's ticks of a month on disk: two quotes an hour around each hour's mid, 50 and 10 minutes before its
    close, the first 0.1% under the mid, the second at it."""
    import io
    import zipfile
    from strategy_lab.data import spread_refresh
    monkeypatch.setattr(spread_refresh, "EXNESS_DIR", tmp_path / "exness")
    rows = ['"Exness","Symbol","Timestamp","Bid","Ask"']
    for t, m in mids.items():
        for before, mid in ((50, m * 0.999), (10, m)):
            at = (t - pd.Timedelta(minutes=before)).strftime("%Y-%m-%d %H:%M:%S.000Z")
            rows.append(f'"exness","{symbol}","{at}",{mid - 0.01},{mid + 0.01}')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"Exness_{symbol}.csv", "\n".join(rows) + "\n")
    d = tmp_path / "exness" / symbol
    d.mkdir(parents=True)
    (d / f"{mids.index[0]:%Y-%m}.zip").write_bytes(buf.getvalue())


def test_crudes_hours_are_its_brokers_where_it_quotes_its_holes_and_short_days_its_own_and_a_rebuild_is_the_same(
        tmp_path, monkeypatch):
    week = _hours("2024-01-08 00:00", "2024-01-12 22:00")                # Mon..Fri, 23 hours a day
    tail = _hours("2024-01-15 00:00", "2024-01-15 22:00")                # the next Monday: the vendor's only
    vendor = _vendor_bars(week.append(tail), np.linspace(70.0, 75.0, len(week) + len(tail)))
    days = [pd.Timestamp(f"2024-01-{d} 22:00", tz="UTC") for d in ("05", "08", "09", "10", "11", "12", "15")]
    vendor_days = pd.DataFrame({"open": 60.0, "high": 90.0, "low": 50.0, "close": 60.0, "volume": 0.0,
                                "dollar_volume": 0.0}, index=pd.DatetimeIndex(days))      # another level: a settlement
    store.write_bars("td", "1h", "WTI/USD", vendor)
    store.write_bars("td", "1d", "WTI/USD", vendor_days)
    # Exness from Tuesday: two percent above the vendor, none of Wednesday's 05:00-08:00, Thursday a short session
    quoted = week[(week >= "2024-01-09 00:00") & ~((week >= "2024-01-10 05:00") & (week <= "2024-01-10 08:00"))
                  & ~((week > "2024-01-11 10:00") & (week <= "2024-01-11 22:00"))]
    _exness_month(tmp_path, monkeypatch, "USOIL", vendor["close"].reindex(quoted) * 1.02)
    monkeypatch.setattr(refresh, "EXNESS_HOLES", {"WTI/USD": (("2024-01-10 05:00", "2024-01-10 08:00"),)})
    refresh._hourly_written("WTI/USD")
    h1, d1 = store.read_bars("td", "1h", "WTI/USD"), store.read_bars("td", "1d", "WTI/USD")
    ex = h1.loc[quoted]
    assert np.allclose(ex["close"], vendor["close"].reindex(quoted) * 1.02)            # the broker's hours, its mid
    assert np.allclose(ex["open"], ex["close"] * 0.999) and np.allclose(ex["low"], ex["open"])
    assert h1.loc[:"2024-01-08 22:00"].index.equals(week[week <= "2024-01-08 22:00"])  # the vendor's before them
    assert h1.loc["2024-01-15"].index.equals(tail)                                      # and after them
    hole = h1.loc["2024-01-10 04:00":"2024-01-10 09:00", "close"]     # the vendor's returns on the broker's level: no
    step = vendor["close"].loc["2024-01-10 04:00":"2024-01-10 09:00"].pct_change()    # jump into the hole or out of it
    assert np.allclose(hole.pct_change().iloc[1:], step.iloc[1:])
    assert not h1.loc["2024-01-11 11:00":"2024-01-11 22:00"].size                       # the short session's end
    # days: the vendor's before the broker's, built from the hours where whole, the short session of its own hours,
    # never the vendor's day at another level inside the broker's span
    assert d1.index.normalize().strftime("%m-%d").tolist() == ["01-05", "01-08", "01-09", "01-10", "01-11", "01-12",
                                                                "01-15"]
    assert d1["close"].iloc[0] == 60.0 and d1["close"].iloc[1] == vendor["close"].loc["2024-01-08 22:00"]
    assert d1.index[4] == pd.Timestamp("2024-01-11 10:00", tz="UTC")
    assert d1["close"].iloc[4] == h1.at[pd.Timestamp("2024-01-11 10:00", tz="UTC"), "close"]
    assert (d1["close"].iloc[2:6] > 70).all() and d1["close"].iloc[6] == vendor["close"].iloc[-1]
    again = {tf: store.read_bars("td", tf, "WTI/USD") for tf in ("1h", "4h", "1d")}
    refresh._hourly_written("WTI/USD")
    assert all(store.read_bars("td", tf, "WTI/USD").equals(v) for tf, v in again.items())
    # its minutes, which the exits are walked through, are the broker's too: they make every hour of the broker's
    from strategy_lab.data import minute_refresh, minutes
    monkeypatch.setattr(minute_refresh, "TD_RAW", tmp_path / "td_minutes")
    rep = minute_refresh.store_twelvedata({"td:WTI/USD": []})["td:WTI/USD"]
    assert rep == {"rows": 2 * len(quoted) + 4, "hours_on_own_prices": 0, "hours_without_minutes": 4}   # the hole's
    t, p = minutes.read("td:WTI/USD")
    assert np.isclose(p[:, 1].max(), ex["high"].max(), rtol=1e-6) and np.isclose(p[:, 2].min(), ex["low"].min(), rtol=1e-6)


def test_a_metals_hours_that_are_not_the_markets_carry_its_brokers_returns_on_the_vendors_level_or_have_no_bar(
        tmp_path, monkeypatch):
    hours = _hours("2025-06-27 00:00", "2025-06-27 22:00")
    level = pd.Series(np.linspace(1100.0, 1150.0, len(hours)), index=hours)
    wrong = (hours >= "2025-06-27 18:00") & (hours <= "2025-06-27 20:00")
    shut = (hours >= "2025-06-27 06:00") & (hours <= "2025-06-27 07:00")       # a price printed while no one quoted
    vendor = _vendor_bars(hours, np.where(wrong, 972.0, np.where(shut, 1061.0, level)))
    store.write_bars("td", "1h", "XPD/USD", vendor)
    store.write_bars("td", "1d", "XPD/USD", _vendor_bars(hours[-1:], [1150.0]).assign(high=1200.0, low=900.0))
    _exness_month(tmp_path, monkeypatch, "XPDUSD", level[~shut] * 1.01)
    monkeypatch.setattr(refresh, "VENDOR_HOURS_WRONG", {"XPD/USD": (("2025-06-27 06:00", "2025-06-27 07:00"),
                                                                    ("2025-06-27 18:00", "2025-06-27 20:00"))})
    refresh._hourly_written("XPD/USD")
    h1 = store.read_bars("td", "1h", "XPD/USD")
    assert h1.index.equals(hours[~shut])                    # the hours Exness did not quote have no bar
    assert np.allclose(h1["close"], level[~shut])           # Exness's returns, on the vendor's level: no jump either way
    assert h1.loc[hours[~wrong & ~shut]].equals(vendor.loc[~wrong & ~shut])     # the vendor's other hours as they were
    again = store.read_bars("td", "1h", "XPD/USD")
    refresh._hourly_written("XPD/USD")
    assert store.read_bars("td", "1h", "XPD/USD").equals(again)


def test_crudes_refresh_fetches_its_brokers_ticks_and_spreads_before_it_writes_the_vendors_hours(monkeypatch):
    from strategy_lab.data import spread_refresh
    # the stored hours are Exness's: the vendor's disagree on the overlap, which is no restatement to fetch it all for
    store.write_bars("td", "1h", "WTI/USD", _vendor_bars(_hours("2026-08-03 00:00", "2026-08-03 05:00"), 69.0))
    tail = _td_values([f"2026-08-03 {h:02d}:00:00" for h in range(4, 9)], [70.0, 70.0, 70.5, 71.0, 71.5])
    session = _Session([_Resp(200, {"status": "ok", "values": tail})])
    monkeypatch.setattr(refresh.requests, "Session", lambda: session)
    monkeypatch.setattr(refresh, "td_series_to_refresh", lambda syms: [("WTI/USD", "1h")])
    calls = []
    monkeypatch.setattr(spread_refresh, "fetch_exness", lambda quotes, dry_run=True, max_requests=None: calls.append(
        ("ticks", tuple(quotes), dry_run)) or {"planned_requests": 2})
    monkeypatch.setattr(spread_refresh, "build", lambda quotes: calls.append(("spreads", tuple(quotes))))
    monkeypatch.setattr(refresh, "_hourly_written", lambda s: calls.append(("hours", s)))
    assert refresh.refresh_twelvedata(["WTI/USD"], dry_run=True)["exness_planned_requests"] == 2
    assert calls == [("ticks", ("WTI/USD",), True)]                                      # a dry run asks nothing
    calls.clear()
    refresh.refresh_twelvedata(["WTI/USD"], dry_run=False)
    assert calls == [("ticks", ("WTI/USD",), True), ("ticks", ("WTI/USD",), False), ("spreads", ("WTI/USD",)),
                     ("hours", "WTI/USD")]
    assert len(session.calls) == 1                                  # the tail only, no whole series fetched again
    h1 = store.read_bars("td", "1h", "WTI/USD")
    assert h1.loc[:"2026-08-03 05:00", "close"].eq(69.0).all()      # the stored hours kept; the vendor's after them
    assert h1.loc["2026-08-03 06:00":, "close"].tolist() == [70.0, 70.5, 71.0, 71.5]      # labelled at their close


def test_a_commodity_hour_is_held_to_its_days_bar_and_a_day_short_at_its_end_keeps_its_last_close():
    ny = lambda t: pd.Timestamp(t, tz="America/New_York").tz_convert("UTC")         # noqa: E731
    hours = pd.date_range(ny("2022-05-15 19:00"), ny("2022-05-16 17:00"), freq="h")   # 18:00 -> 17:00, 23 hours
    h1 = pd.DataFrame({"open": 21.2, "high": 21.3, "low": 21.1, "close": 21.2, "volume": 0.0, "dollar_volume": 0.0},
                      index=hours)
    h1.iloc[0, :4] = [25.24, 25.24, 21.10, 21.15]                     # a first hour opening at a print far off
    h1.iloc[-1, :4] = [21.20, 25.48, 21.10, 25.48]                    # and a last hour closing at one
    short = pd.date_range(ny("2023-03-16 19:00"), ny("2023-03-17 04:00"), freq="h")   # ten hours, none after 04:00
    h2 = pd.DataFrame({"open": 68.5, "high": 69.0, "low": 68.4, "close": 68.97, "volume": 0.0, "dollar_volume": 0.0},
                      index=short)
    daily = pd.DataFrame({"open": [21.105, 68.28], "high": [21.715, 69.25], "low": [20.835, 66.36],
                          "close": [21.599, 66.36], "volume": 0.0, "dollar_volume": 0.0},
                         index=[ny("2022-05-16 17:00"), ny("2023-03-17 17:00")])
    out, rep = refresh.conform_commodity_hourly(pd.concat([h1, h2]), daily)
    day1 = out.loc[:ny("2022-05-16 17:00")]
    assert day1["high"].max() <= 21.715 and day1["open"].iloc[0] == 21.105        # held to the day, its open
    assert day1["close"].iloc[-1] == 21.599                                        # its last hour ends the day
    assert day1["close"].iloc[5] == 21.2 and day1["open"].iloc[1] == 21.2          # inside the range: as quoted
    assert out["close"].iloc[-1] == 68.97                                           # ends at 04:00: its own close
    assert rep == {"bars_held_within_the_days_range": 2, "days_opened_or_closed_at_the_days": 2}


def _next_session() -> str:
    """The next NYSE session after today: its bar cannot have closed yet."""
    from strategy_lab.data import calendars as cal
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    sched = cal.nyse_sessions(pd.Timestamp(today + pd.Timedelta(days=1), tz="UTC"), pd.Timestamp(today + pd.Timedelta(days=12), tz="UTC"))
    return pd.DatetimeIndex(sched.index)[0].strftime("%Y-%m-%d")


def test_a_bar_still_open_when_fetched_is_not_stored(monkeypatch):
    store.write_bars("td", "1d", "ACME", _daily(["2026-08-03", "2026-08-04", "2026-08-05"], [100.0, 102.0, 104.0]))
    tail = _td_values(["2026-08-04", "2026-08-05", _next_session()], [102.0, 104.0, 105.0])      # vendor's bar in progress
    sess = _Session([_Resp(200, {"status": "ok", "values": tail})])
    monkeypatch.setattr(refresh.requests, "Session", lambda: sess)
    monkeypatch.setattr(refresh, "td_series_to_refresh", lambda syms: [("ACME", "1d")])
    rep = refresh.refresh_twelvedata(["ACME"], dry_run=False)
    assert rep["bars_added"] == 0 and list(store.read_bars("td", "1d", "ACME")["close"]) == [100.0, 102.0, 104.0]


def test_bars_stored_before_they_closed_are_dropped_by_the_next_refresh(monkeypatch):
    store.write_bars("td", "1d", "ACME", _daily(["2026-08-04", "2026-08-05", _next_session()], [102.0, 104.0, 105.0]))
    tail = _td_values(["2026-08-04", "2026-08-05"], [102.0, 104.0])
    sess = _Session([_Resp(200, {"status": "ok", "values": tail})])
    monkeypatch.setattr(refresh.requests, "Session", lambda: sess)
    monkeypatch.setattr(refresh, "td_series_to_refresh", lambda syms: [("ACME", "1d")])
    refresh.refresh_twelvedata(["ACME"], dry_run=False)
    bars = store.read_bars("td", "1d", "ACME")
    assert list(bars["close"]) == [102.0, 104.0] and bars.index.max() <= pd.Timestamp.now(tz="UTC")


class _Vendor:
    """TwelveData's paging as observed: of the bars inside [start_date, end_date] it returns the latest `outputsize`."""
    def __init__(self, dates, closes, reject=()):
        self.bars = list(zip(dates, closes))
        self.reject = set(reject)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        if params["symbol"] in self.reject:
            return _Resp(404, {"status": "error", "code": 404, "message": "**symbol** or **figi** parameter is missing or invalid."})
        lo = params.get("start_date", "0000")[:10]
        hi = params.get("end_date", "9999")[:10]
        inside = [(d, c) for d, c in self.bars if lo <= d <= hi][-int(params.get("outputsize", 30)):]
        if not inside:
            return _Resp(400, {"status": "error", "code": 400, "message": "No data is available on the specified dates."})
        return _Resp(200, {"status": "ok", "values": _td_values([d for d, _ in inside], [c for _, c in inside])})


def test_a_full_refetch_pages_backwards_to_the_start_of_the_history(monkeypatch):
    dates = ["2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31", "2026-08-03", "2026-08-04"]
    vendor = _Vendor(dates, [float(k) for k in range(1, 8)])
    monkeypatch.setattr(refresh, "TD_PAGE", 3)
    monkeypatch.setattr(refresh, "TD_HISTORY_START", {**refresh.TD_HISTORY_START, "1d": "2026-07-01"})
    bars = refresh._td_full(vendor, "ACME", "1d", refresh.Budget("x", None, None), pd.Timestamp.now(tz="UTC"))
    assert list(bars["close"]) == [float(k) for k in range(1, 8)] and len(vendor.calls) == 3


def test_a_shorter_history_never_replaces_the_stored_one():
    store.write_bars("td", "1d", "ACME", _daily(["2026-07-27", "2026-07-28", "2026-08-04"], [1.0, 2.0, 3.0]))
    old = store.read_bars("td", "1d", "ACME")
    assert not refresh._replace_whole_series("ACME", "1d", _daily(["2026-08-04", "2026-08-05"], [3.0, 4.0]), old)
    assert list(store.read_bars("td", "1d", "ACME")["close"]) == [1.0, 2.0, 3.0]


def test_a_rejected_symbol_is_remembered_and_the_refresh_goes_on(monkeypatch):
    for sym in ("GONE", "ACME"):
        store.write_bars("td", "1d", sym, _daily(["2026-08-03", "2026-08-04"], [100.0, 102.0]))
    vendor = _Vendor(["2026-08-03", "2026-08-04", "2026-08-05"], [100.0, 102.0, 104.0], reject={"GONE"})
    monkeypatch.setattr(refresh.requests, "Session", lambda: vendor)
    monkeypatch.setattr(refresh, "td_series_to_refresh", lambda syms: [("GONE", "1d"), ("ACME", "1d")])
    rep = refresh.refresh_twelvedata(["GONE", "ACME"], dry_run=False)
    assert rep["rejected_now"] == ["GONE:1d"] and rep["bars_added"] == 1
    assert store.read_meta("td", "1d", "GONE")["vendor_rejected"]
    assert refresh.refresh_twelvedata(["GONE", "ACME"], dry_run=True)["rejected_earlier_skipped"] == 1


def test_a_partial_refresh_does_not_make_the_rest_look_delisted():
    for sym in ("AAA", "BBB", "CCC"):
        store.write_bars("td", "1d", sym, _daily(["2026-08-03", "2026-08-04"], [1.0, 2.0]))
        store.write_meta("td", "1d", sym, {"last": "2026-08-04 20:00:00+00:00"})       # where the seed ended
    store.write_bars("td", "1d", "AAA", _daily(["2026-08-03", "2026-08-04", "2026-09-22"], [1.0, 2.0, 3.0]))
    assert sorted(s for s, _ in refresh.td_series_to_refresh(["AAA", "BBB", "CCC"])) == ["AAA", "BBB", "CCC"]


def test_a_series_seeded_today_does_not_make_an_older_seed_look_delisted():
    for sym, seeded_end, end, checked in (("OLD", "2026-08-04", "2026-09-23", "2026-09-23 15:00"),   # seeded, refreshed
                                          ("NEW", "2026-09-25", "2026-09-25", "2026-09-25 23:00"),   # seeded today
                                          ("GONE", "2026-08-04", "2026-08-04", "2026-09-23 15:00")):  # nothing since
        store.write_bars("td", "1d", sym, _daily(["2026-08-03", end], [1.0, 2.0]))
        store.write_meta("td", "1d", sym, {"last": f"{seeded_end} 20:00:00+00:00",
                                           "checked_through": f"{checked}:00+00:00"})
    assert sorted(s for s, _ in refresh.td_series_to_refresh(["OLD", "NEW", "GONE"])) == ["NEW", "OLD"]


def _hourly_bars(closes_at, price=10.0):
    idx = pd.DatetimeIndex(closes_at)
    c = pd.Series(price, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0, "dollar_volume": c})


def _kline(close_at, price):
    open_ms = int((pd.Timestamp(close_at) - pd.Timedelta(hours=1)).timestamp() * 1000)
    return [open_ms, str(price), str(price), str(price), str(price), "1", open_ms + 3_599_999, str(price)]


def test_binance_gaps_are_filled_from_the_exchange_and_an_empty_answer_is_remembered(monkeypatch):
    t = pd.date_range("2022-02-27 00:00", periods=6, freq="h", tz="UTC")
    store.write_bars("perp", "1h", "AAAUSDT", _hourly_bars(t.delete([2, 3])))           # two hours missing
    store.write_bars("perp", "1h", "BBBUSDT", _hourly_bars(t.delete([4])))              # the exchange has none either
    answers = {"AAAUSDT": [_kline(t[2], 11.0), _kline(t[3], 12.0)], "BBBUSDT": []}

    class _Bin:
        calls = []

        def get(self, url, params=None, timeout=None):
            self.calls.append(params["symbol"])
            return _Resp(200, answers[params["symbol"]])
    sess = _Bin()
    monkeypatch.setattr(refresh.requests, "Session", lambda: sess)
    rep = refresh.fill_binance_gaps(markets=("perp",), tfs=("1h",), dry_run=False)
    assert rep["perp"]["bars_filled"] == 2 and len(sess.calls) == 2
    assert list(store.read_bars("perp", "1h", "AAAUSDT")["close"]) == [10.0, 10.0, 11.0, 12.0, 10.0, 10.0]
    again = refresh.fill_binance_gaps(markets=("perp",), tfs=("1h",), dry_run=True)
    assert again["perp"]["series_with_gaps"] == 0                         # BBB's hole is known to be the exchange's


def test_a_new_binance_series_is_seeded_from_its_first_daily_bar(monkeypatch):
    store.write_bars("perp", "1d", "NEWUSDT", _hourly_bars(pd.DatetimeIndex(["2026-01-02", "2026-01-03"], tz="UTC")))
    hours = pd.date_range("2026-01-01 01:00", periods=5, freq="h", tz="UTC")

    settled = [{"symbol": "NEWUSDT", "fundingTime": int(pd.Timestamp(t, tz="UTC").timestamp() * 1000),
                "fundingRate": r} for t, r in (("2026-01-01 08:00", "0.0001"), ("2026-01-01 16:00", "-0.0002"))]

    class _Bin:
        calls = []

        def get(self, url, params=None, timeout=None):
            if url.endswith("exchangeInfo"):
                return _Resp(200, {"symbols": [{"symbol": "NEWUSDT", "status": "TRADING", "deliveryDate": 4133404800000}]})
            self.calls.append(dict(params))
            if url.endswith("fundingRate"):
                return _Resp(200, [f for f in settled if f["fundingTime"] >= params["startTime"]])
            later = [h for h in hours if (h - pd.Timedelta(hours=1)).timestamp() * 1000 >= params["startTime"]]
            return _Resp(200, [_kline(h, 10.0 + k) for k, h in enumerate(later[: params["limit"]])])
    sess = _Bin()
    monkeypatch.setattr(refresh.requests, "Session", lambda: sess)
    monkeypatch.setitem(refresh.TF_DELTA, "1h", pd.Timedelta(hours=1))
    rep = refresh.seed_binance_series(["NEWUSDT"], tfs=("1h",), dry_run=False)
    seeded = store.read_bars("perp", "1h", "NEWUSDT")
    assert rep["series_written"] == 1 and list(seeded.index) == list(hours)
    assert sess.calls[0]["startTime"] == int(pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000)
    # a perp's funding comes with it: a backtest refuses a perp without its settlements
    assert bars.load_funding("perp:NEWUSDT").tolist() == [0.0001, -0.0002]


def _fixture_rows(kind):
    from pathlib import Path
    return json.loads((Path(__file__).parent / "fixtures" / "td_aapl_hourly_grids.json").read_text())[kind]


def _ny(ts):
    return pd.Timestamp(ts).tz_convert("America/New_York")


def _hourly_by_day():
    bars = refresh._td_normalise("AAPL", "1h", refresh._td_frame(_fixture_rows("1h")), pd.Timestamp.now(tz="UTC"))
    starts = refresh.td_equity_hourly_starts(refresh._td_frame(_fixture_rows("1h")).index).dropna().sort_values()
    return bars, pd.DatetimeIndex(starts)


def test_hourly_bars_of_2019_and_2020_are_stamped_where_the_vendors_15_minute_bars_put_them():
    """Before 2020-06-29 the labels are not the bars' starts; each re-stamped bar must hold exactly the vendor's own
    15-minute bars of its [start, close) window: a half-hour grid with shifted labels in 2019, clock hours in 2020."""
    bars, starts = _hourly_by_day()
    m15 = refresh._td_frame(_fixture_rows("15min"))
    for day in ("2019-12-03", "2020-03-16"):
        sel = bars[bars.index.tz_convert("America/New_York").strftime("%Y-%m-%d") == day]
        assert len(sel) == 7
        for close, row in sel.iterrows():
            start = starts[starts < close][-1]
            window = m15[(m15.index >= start) & (m15.index < close)]
            assert abs(window["volume"].sum() / row["volume"] - 1) < 0.03      # a window 30 minutes off: 13-80%
            assert abs(window["high"].max() / row["high"] - 1) < 1e-3 and abs(window["low"].min() / row["low"] - 1) < 1e-3
    assert [_ny(t).strftime("%H:%M") for t in bars.index if _ny(t).strftime("%Y-%m-%d") == "2020-03-16"] == \
        ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"]


def test_every_2019_session_opens_and_closes_where_the_vendors_daily_bar_does():
    """Winter, a week of US summer time before Europe's, European summer time, and the week after it: after the
    re-stamping each session runs 09:30-16:00 New York, its first hourly open is the day's open and its last hourly
    close the day's close, and no hourly price leaves the day's range."""
    bars, _ = _hourly_by_day()
    daily = {r["date"]: r for r in _fixture_rows("1day")}
    for day in ("2019-01-15", "2019-03-20", "2019-07-16", "2019-10-30"):
        sel = bars[bars.index.tz_convert("America/New_York").strftime("%Y-%m-%d") == day]
        d = daily[day]
        assert [_ny(t).strftime("%H:%M") for t in sel.index] == ["10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00"]
        assert abs(sel["open"].iloc[0] / d["open"] - 1) < 1e-3 and abs(sel["close"].iloc[-1] / d["close"] - 1) < 1e-3
        assert sel["high"].max() <= d["high"] * (1 + 1e-4) and sel["low"].min() >= d["low"] * (1 - 1e-4)


def test_the_off_grid_bar_of_2020_06_26_is_dropped_and_a_clock_hour_session_splits_at_13():
    bars, _ = _hourly_by_day()
    day = bars[bars.index.tz_convert("America/New_York").strftime("%Y-%m-%d") == "2020-06-26"]
    assert "15:30" not in [_ny(t).strftime("%H:%M") for t in day.index]
    from strategy_lab.data import resample
    clock = bars[bars.index.tz_convert("America/New_York").strftime("%Y-%m-%d") == "2020-03-16"]
    h4 = resample.equity_4h_from_1h(clock)
    assert [_ny(t).strftime("%H:%M") for t in h4.index] == ["13:00", "16:00"]
    assert h4["volume"].tolist() == [clock["volume"].iloc[:4].sum(), clock["volume"].iloc[4:].sum()]


def test_history_before_the_first_stored_bar_is_joined_at_a_matching_seam_and_not_asked_again(monkeypatch):
    store.write_bars("td", "1d", "ACME", _daily(["2026-07-30", "2026-07-31", "2026-08-03"], [3.0, 4.0, 5.0]))
    vendor = _Vendor(["2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31", "2026-08-03"],
                     [1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    monkeypatch.setattr(refresh.requests, "Session", lambda: vendor)
    monkeypatch.setattr(refresh, "TD_PAGE", 2)
    monkeypatch.setattr(refresh, "TD_HISTORY_START", {**refresh.TD_HISTORY_START, "1d": "2026-07-01"})
    rep = refresh.extend_twelvedata([("ACME", "1d")], dry_run=False)
    assert rep["bars_added"] == 3 and list(store.read_bars("td", "1d", "ACME")["close"]) == [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    assert all(c["end_date"] <= "2026-07-31 20:59:59" for c in vendor.calls)          # up to the bar after the seam
    assert refresh.extend_twelvedata([("ACME", "1d")], dry_run=True)["asked_before_skipped"] == 1


def test_a_seam_that_disagrees_refetches_the_whole_series(monkeypatch):
    store.write_bars("td", "1d", "ACME", _daily(["2026-07-30", "2026-07-31"], [6.0, 8.0]))       # before a 2:1 split
    vendor = _Vendor(["2026-07-29", "2026-07-30", "2026-07-31"], [2.0, 3.0, 4.0])
    monkeypatch.setattr(refresh.requests, "Session", lambda: vendor)
    monkeypatch.setattr(refresh, "TD_HISTORY_START", {**refresh.TD_HISTORY_START, "1d": "2026-07-01"})
    rep = refresh.extend_twelvedata([("ACME", "1d")], dry_run=False)
    assert rep["series_refetched"] == ["ACME:1d"] and list(store.read_bars("td", "1d", "ACME")["close"]) == [2.0, 3.0, 4.0]


def test_a_vendor_without_earlier_history_is_asked_once(monkeypatch):
    store.write_bars("td", "1d", "ACME", _daily(["2026-07-30", "2026-07-31"], [3.0, 4.0]))
    vendor = _Vendor(["2026-07-30", "2026-07-31"], [3.0, 4.0])
    monkeypatch.setattr(refresh.requests, "Session", lambda: vendor)
    monkeypatch.setattr(refresh, "TD_HISTORY_START", {**refresh.TD_HISTORY_START, "1d": "2026-07-01"})
    rep = refresh.extend_twelvedata([("ACME", "1d")], dry_run=False)
    assert rep["vendor_has_nothing_earlier"] == 1 and len(vendor.calls) == 1
    assert refresh.extend_twelvedata([("ACME", "1d")], dry_run=True)["series"] == 0


def test_an_hourly_seam_is_checked_at_its_true_close(monkeypatch):
    """The stored first hourly bar closes where the next bar starts: the window must reach that next bar, or the seam
    bar is stamped at the session close and the seam goes unchecked."""
    rows = [r for r in _fixture_rows("1h") if r["datetime"][:10] == "2020-07-15"]
    day = refresh._td_normalise("AAPL", "1h", refresh._td_frame(rows), pd.Timestamp.now(tz="UTC"))
    store.write_bars("td", "1h", "AAPL", day.iloc[3:])                                  # stored from the 13:30 bar
    requested = []

    class _HourlyVendor:
        def get(self, url, params=None, timeout=None):
            requested.append(dict(params))
            inside = [r for r in rows if params["start_date"] <= r["datetime"] <= params.get("end_date", "9999")]
            return _Resp(200, {"status": "ok", "values": inside})
    monkeypatch.setattr(refresh.requests, "Session", lambda: _HourlyVendor())
    monkeypatch.setattr(refresh, "TD_HISTORY_START", {**refresh.TD_HISTORY_START, "1h": "2020-07-15"})
    rep = refresh.extend_twelvedata([("AAPL", "1h")], dry_run=False)
    assert rep["bars_added"] == 3 and rep["series_refetched"] == []
    assert store.read_bars("td", "1h", "AAPL")["close"].tolist() == day["close"].tolist()
    store.write_bars("td", "1h", "AAPL", day.iloc[3:].assign(close=day["close"].iloc[3:] * 2))   # a split since stored
    store.write_meta("td", "1h", "AAPL", {})
    assert refresh.extend_twelvedata([("AAPL", "1h")], dry_run=False)["series_refetched"] == ["AAPL:1h"]


def test_a_coin_new_to_the_store_is_seeded_from_the_exchanges_opening(monkeypatch):
    days = pd.date_range("2025-05-01", periods=3, freq="D", tz="UTC")                 # the pair listed on 2025-04-30
    asked = []

    class _Bin:
        def get(self, url, params=None, timeout=None):
            asked.append(params["startTime"])
            later = [d for d in days if (d - pd.Timedelta(days=1)).timestamp() * 1000 >= params["startTime"]]
            return _Resp(200, [[int((d - pd.Timedelta(days=1)).timestamp() * 1000), "1", "1", "1", "1", "1",
                                int(d.timestamp() * 1000) - 1, "1"] for d in later[: params["limit"]]])
    monkeypatch.setattr(refresh.requests, "Session", lambda: _Bin())
    rep = refresh.seed_binance_series(["NEWUSDT"], tfs=("1d",), market="spot", dry_run=False)
    assert rep["series_written"] == 1 and asked[0] == int(refresh.BINANCE_OPENED.timestamp() * 1000)
    assert list(store.read_bars("spot", "1d", "NEWUSDT").index) == list(days)


def test_earlier_history_is_taken_only_back_to_its_last_hole(monkeypatch):
    """The vendor's early history is sparse in places; a position held across a years-long hole would book the whole
    move as one bar, so the joined history starts after the last hole longer than MAX_HOLE."""
    store.write_bars("td", "1d", "ACME", _daily(["1990-01-10", "1990-01-11"], [10.0, 11.0]))
    vendor = _Vendor(["1975-04-11", "1975-04-14", "1990-01-02", "1990-01-03", "1990-01-04", "1990-01-05", "1990-01-08",
                      "1990-01-09", "1990-01-10", "1990-01-11"], [1.0, 1.1, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 10.0, 11.0])
    monkeypatch.setattr(refresh.requests, "Session", lambda: vendor)
    monkeypatch.setattr(refresh, "TD_HISTORY_START", {**refresh.TD_HISTORY_START, "1d": "1970-01-01"})
    rep = refresh.extend_twelvedata([("ACME", "1d")], dry_run=False)
    bars = store.read_bars("td", "1d", "ACME")
    assert rep["bars_added"] == 6 and bars.index.min().date() == pd.Timestamp("1990-01-02").date()


def _sessions(n, start="2024-01-02"):
    from strategy_lab.data import calendars as cal
    sched = cal.nyse_sessions(pd.Timestamp(start, tz="UTC"), pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=2 * n))
    return pd.DatetimeIndex(sched["market_close"].iloc[:n]).tz_convert("UTC")


def _bars_at(idx, close, volume=100.0):
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": volume, "dollar_volume": c * volume})


def test_an_hourly_series_missing_a_split_is_put_on_the_daily_basis():
    """The vendor's hourly bars of APH were not adjusted for its 2024 2:1 split while the daily bars were: the hourly
    prices before the split must come out halved, their volume doubled, and the fake -50% night gone."""
    idx = _sessions(80)
    daily = 20 + np.cumsum(np.random.default_rng(3).normal(0, 0.2, 80))
    hourly = np.where(np.arange(80) < 50, 2.0 * daily, daily)
    h1, d1 = _bars_at(idx, hourly), _bars_at(idx, daily)
    fixed, rep = refresh.td_hourly_on_daily_basis(h1, d1)
    assert np.allclose(fixed["close"].to_numpy(), daily) and np.allclose(fixed["volume"].iloc[:50], 200.0)
    assert list(rep["rescaled"].values()) == [0.5] and "dropped_through" not in rep


def test_hourly_bars_of_another_company_under_the_same_ticker_are_dropped():
    """GEN's hourly bars before 2022 are Genesis Healthcare's, a penny stock that drifts against Gen Digital's daily
    bars: everything before the ticker's current owner is dropped; a single bad print is not a new owner."""
    idx = _sessions(80)
    rng = np.random.default_rng(5)
    daily = 20 + np.cumsum(rng.normal(0, 0.2, 80))
    other = 0.7 * np.exp(np.cumsum(rng.normal(0, 0.05, 80)))
    hourly = np.where(np.arange(80) < 40, other, daily)
    hourly[60] = daily[60] * 1.3                                                      # one bad print
    fixed, rep = refresh.td_hourly_on_daily_basis(_bars_at(idx, hourly), _bars_at(idx, daily))
    assert rep["dropped_through"] == str(idx[39].tz_convert("America/New_York").date())
    assert fixed.index.min() == idx[40] and fixed["close"].iloc[20] == hourly[60] and "rescaled" not in rep


def test_an_hourly_series_that_never_matches_its_daily_bars_is_quarantined():
    """BNY's hourly bars are a municipal bond fund's while its daily bars are the bank's: the hourly and 4h series
    leave the store for data/quarantine, with the reason, and no universe sees them."""
    idx = _sessions(60)
    rng = np.random.default_rng(7)
    store.write_bars("td", "1d", "BNY", _bars_at(idx, 60 + np.cumsum(rng.normal(0, 0.5, 60))))
    store.write_bars("td", "1h", "BNY", _bars_at(idx, 13 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))))
    refresh._hourly_written("BNY")
    assert "BNY" not in store.symbols("td", "1h") and "BNY" not in store.symbols("td", "4h")
    moved = store.STORE_DIR.parent / "quarantine" / "td" / "1h"
    assert (moved / "BNY.parquet").exists() and json.loads((moved / "BNY.meta.json").read_text())["quarantined"]


def test_a_few_steady_sessions_of_another_company_are_dropped_not_rescaled():
    """ADEA's hourly bars had five sessions of another company's prices at a steady ratio to the daily bars: too few to
    prove the same instrument, so they go with everything before them."""
    idx = _sessions(80)
    rng = np.random.default_rng(9)
    daily = 20 + np.cumsum(rng.normal(0, 0.2, 80))
    hourly = daily.copy()
    hourly[:35] = 0.7 * np.exp(np.cumsum(rng.normal(0, 0.05, 35)))
    hourly[35:40] = 57.0 * daily[35:40]                                              # steady, but five sessions
    fixed, rep = refresh.td_hourly_on_daily_basis(_bars_at(idx, hourly), _bars_at(idx, daily))
    assert fixed.index.min() == idx[40] and "rescaled" not in rep


def test_a_contract_binance_no_longer_serves_is_skipped_remembered_and_not_asked_again(monkeypatch):
    for sym in ("GONEUSDT", "NEWUSDT"):
        store.write_bars("perp", "1d", sym, _hourly_bars(pd.DatetimeIndex(["2026-01-02", "2026-01-03"], tz="UTC")))
    hours = pd.date_range("2026-01-01 01:00", periods=5, freq="h", tz="UTC")

    class _Bin:
        calls = []

        def get(self, url, params=None, timeout=None):
            if url.endswith("exchangeInfo"):
                return _Resp(200, {"symbols": [{"symbol": "NEWUSDT", "status": "TRADING", "deliveryDate": 4133404800000}]})
            self.calls.append(params["symbol"])
            if params["symbol"] == "GONEUSDT":
                return _Resp(400, {"code": -1121, "msg": "Invalid symbol."})
            if url.endswith("fundingRate"):
                return _Resp(200, [])
            later = [h for h in hours if (h - pd.Timedelta(hours=1)).timestamp() * 1000 >= params["startTime"]]
            return _Resp(200, [_kline(h, 10.0 + k) for k, h in enumerate(later[: params["limit"]])])
    sess = _Bin()
    monkeypatch.setattr(refresh.requests, "Session", lambda: sess)
    monkeypatch.setitem(refresh.TF_DELTA, "1h", pd.Timedelta(hours=1))
    rep = refresh.seed_binance_series(["GONEUSDT", "NEWUSDT"], tfs=("1h",), dry_run=False)
    assert rep["rejected"] == ["GONEUSDT:1h"] and rep["series_written"] == 1
    assert store.read_meta("perp", "1h", "GONEUSDT")["vendor_rejected"]["message"] == "binance: Invalid symbol."
    assert refresh.seed_binance_series(["GONEUSDT"], tfs=("1h",), dry_run=True)["series"] == 0


def test_funding_stored_from_later_than_the_listing_is_filled_back_to_it(monkeypatch):
    store.write_bars("perp", "1d", "LATEUSDT", _hourly_bars(pd.DatetimeIndex(["2026-01-02", "2026-03-01"], tz="UTC")))
    folder = store.STORE_DIR / "perp" / "funding"
    folder.mkdir(parents=True)
    pd.DataFrame({"rate": [0.0003]}, index=pd.DatetimeIndex(["2026-02-20"], tz="UTC", name="settle_time")) \
        .to_parquet(folder / "LATEUSDT.parquet")                  # what a request from time 0 had brought back
    ms = {t: int(pd.Timestamp(t, tz="UTC").timestamp() * 1000) for t in ("2026-01-01 08:00", "2026-01-15 08:00",
                                                                         "2026-02-20", "2026-02-25")}
    rates = dict(zip(ms, ("0.0001", "0.0002", "0.0003", "0.0004")))

    class _Bin:
        def get(self, url, params=None, timeout=None):
            end = params.get("endTime", float("inf"))
            return _Resp(200, [{"symbol": "LATEUSDT", "fundingTime": v, "fundingRate": rates[k]} for k, v in ms.items()
                               if params["startTime"] <= v <= end])
    monkeypatch.setattr(refresh.requests, "Session", lambda: _Bin())
    assert refresh._funding_short("LATEUSDT")
    refresh.seed_binance_series(["LATEUSDT"], tfs=(), dry_run=False)
    assert bars.load_funding("perp:LATEUSDT").tolist() == [0.0001, 0.0002, 0.0003, 0.0004]
    assert not refresh._funding_short("LATEUSDT")


def test_a_delisted_perp_ends_at_the_exchanges_delivery_time_or_else_at_its_last_trade(monkeypatch):
    closes = pd.date_range("2026-06-23 01:00", periods=48, freq="h", tz="UTC")
    printing = _hourly_bars(closes)
    printing.loc[closes[9]:, ["volume", "dollar_volume"]] = 0.0     # the exchange goes on printing after 09:00
    for sym in ("TONUSDT", "EOSUSDT", "BTCUSDT"):
        store.write_bars("perp", "1h", sym, printing)
    delivery = int(pd.Timestamp("2026-06-23 09:00", tz="UTC").timestamp() * 1000)
    info = {"symbols": [{"symbol": "TONUSDT", "status": "SETTLING", "deliveryDate": delivery},
                        {"symbol": "BTCUSDT", "status": "TRADING", "deliveryDate": 4133404800000}]}  # EOS: long gone

    class _Bin:
        def get(self, url, params=None, timeout=None):
            return _Resp(200, info)
    monkeypatch.setattr(refresh.requests, "Session", lambda: _Bin())
    assert refresh.mark_delistings(("1h",), dry_run=True)["1h"] == {"delisted_series": 2, "bars_dropped": 78}
    assert store.read_bars("perp", "1h", "TONUSDT").index[-1] == closes[-1]          # a dry run writes nothing
    refresh.mark_delistings(("1h",), dry_run=False)
    assert store.read_bars("perp", "1h", "TONUSDT").index[-1] == closes[8]          # the 08:00-09:00 bar is its last
    assert store.read_bars("perp", "1h", "EOSUSDT").index[-1] == closes[8]          # its last bar with trades
    assert store.read_bars("perp", "1h", "BTCUSDT").index[-1] == closes[-1]         # still trading: untouched
    assert store.read_meta("perp", "1h", "TONUSDT")["delisted"] == "2026-06-23 09:00:00+00:00"
    assert refresh.perp_delistings(["TONUSDT", "EOSUSDT", "BTCUSDT"]) == {
        "TONUSDT": pd.Timestamp("2026-06-23 09:00", tz="UTC"), "EOSUSDT": None}
    assert refresh.mark_delistings(("1h",), dry_run=True)["1h"] == {"delisted_series": 0, "bars_dropped": 0}


def test_a_stretch_the_exchange_printed_without_trades_is_a_delisting_and_a_stretch_without_bars_is_not(monkeypatch):
    days = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    relisted = _hourly_bars(days)
    relisted.loc[days[10]:days[24], ["volume", "dollar_volume"]] = 0.0      # delisted, then listed anew on day 25
    assert refresh.delisted_between(relisted, "1d") == [(days[9], days[25])]
    holey = _hourly_bars(days.delete(range(10, 25)))                         # the data simply misses those days
    assert refresh.delisted_between(holey, "1d") == []
    store.write_bars("perp", "1d", "TLMUSDT", relisted)
    info = {"symbols": [{"symbol": "TLMUSDT", "status": "TRADING", "deliveryDate": 4133404800000}]}

    class _Bin:
        def get(self, url, params=None, timeout=None):
            return _Resp(200, info)
    monkeypatch.setattr(refresh.requests, "Session", lambda: _Bin())
    refresh.mark_delistings(("1d",), dry_run=False)
    kept = store.read_bars("perp", "1d", "TLMUSDT")
    assert len(kept) == 25 and kept.index[9] == days[9] and kept.index[10] == days[25]
    assert store.read_meta("perp", "1d", "TLMUSDT")["delisted_periods"] == [[str(days[9]), str(days[25])]]


def _archive_month(rows, header):
    """A monthly archive file as the exchange publishes it: one CSV in a zip, its header row in the newer files."""
    import io
    import zipfile
    cols = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("X-1h-2024-01.csv", "\n".join(([cols] if header else []) + [",".join(map(str, r)) for r in rows]))
    return buf.getvalue()


def _archive_row(opens_at, unit, price=1.5, volume=10.0):
    scale = 1_000_000 if unit == "us" else 1000
    return [int(pd.Timestamp(opens_at).timestamp() * scale), price, price + 0.5, price - 0.5, price, volume, 0,
            price * volume, 3, 0, 0, 0]


def test_an_archive_month_is_read_with_or_without_its_header_in_milli_or_microseconds():
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    hour = pd.Timedelta(hours=1)
    for header, unit in ((False, "ms"), (True, "ms"), (True, "us")):     # spot files give microseconds from 2025
        got = refresh.archive_klines(_archive_month([_archive_row(t0, unit), _archive_row(t0 + hour, unit)], header), "1h")
        assert list(got.index) == [t0 + hour, t0 + 2 * hour]                                   # stamped at the close
        assert list(got["dollar_volume"]) == [15.0, 15.0] and list(got.columns) == store.BAR_COLUMNS


def test_a_delisted_pair_is_seeded_from_the_exchanges_archive_up_to_its_last_trade(monkeypatch):
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    hour = pd.Timedelta(hours=1)
    store.write_bars("perp", "1d", "GONEUSDT", _hourly_bars(pd.DatetimeIndex([t0 + pd.Timedelta(days=1)])))
    month = _archive_month([_archive_row(t0, "ms"), _archive_row(t0 + hour, "ms"),
                            _archive_row(t0 + 2 * hour, "ms", volume=0.0)], header=True)
    listing = ("<ListBucketResult><Contents><Key>data/futures/um/monthly/klines/GONEUSDT/1h/GONEUSDT-1h-2024-01.zip"
               "</Key></Contents><Contents><Key>data/futures/um/monthly/klines/GONEUSDT/1h/"
               "GONEUSDT-1h-2024-01.zip.CHECKSUM</Key></Contents></ListBucketResult>")

    class _Archive:
        calls = []

        def get(self, url, params=None, timeout=None):
            self.calls.append(url)
            r = _Resp(200, listing if "s3" in url else "")
            r.content = month
            return r
    sess = _Archive()
    monkeypatch.setattr(refresh.requests, "Session", lambda: sess)
    assert refresh.seed_binance_from_archive(["GONEUSDT"], ("1h",), "perp", dry_run=True)["planned_requests"] == 2
    rep = refresh.seed_binance_from_archive(["GONEUSDT"], ("1h",), "perp", dry_run=False)
    assert rep["requests_made"] == 2 and len(sess.calls) == 2               # the listing, then the one zip file
    bars = store.read_bars("perp", "1h", "GONEUSDT")
    assert list(bars.index) == [t0 + hour, t0 + 2 * hour]                    # the hour without trades is not kept


def test_a_perp_listed_before_the_archive_is_extended_back_from_the_exchange(monkeypatch):
    days = pd.date_range("2020-01-02", periods=3, freq="D", tz="UTC")                   # the archive's first days
    store.write_bars("perp", "1d", "OLDUSDT", _hourly_bars(days, price=10.0))
    store.write_bars("perp", "1d", "NEWUSDT", _hourly_bars(days + pd.Timedelta(days=30)))  # listed later: nothing to ask
    before = [[int((days[0] - pd.Timedelta(days=k + 1)).timestamp() * 1000), "9", "9", "9", "9", "1", 0, "9"]
              for k in (2, 1)]                                                           # the two days before its first

    class _Bin:
        calls = []

        def get(self, url, params=None, timeout=None):
            self.calls.append(url.rsplit("/", 1)[-1])
            return _Resp(200, before if url.endswith("klines") else [])
    sess = _Bin()
    monkeypatch.setattr(refresh.requests, "Session", lambda: sess)
    assert refresh.extend_binance(("1d",), dry_run=True)["perps"] == ["OLDUSDT"]
    refresh.extend_binance(("1d",), dry_run=False)
    got = store.read_bars("perp", "1d", "OLDUSDT")
    assert list(got.index) == [days[0] - pd.Timedelta(days=2), days[0] - pd.Timedelta(days=1), *days]
    assert list(got["close"]) == [9.0, 9.0, 10.0, 10.0, 10.0] and sess.calls == ["klines", "fundingRate"]


def _bi5(records):
    """A Dukascopy candle file: LZMA-compressed 24-byte records (seconds from the month's start, open, close, low,
    high in integer points, tick volume)."""
    import lzma
    import struct
    return lzma.compress(b"".join(struct.pack(">5if", *r) for r in records), format=lzma.FORMAT_ALONE)


def test_a_dukascopy_month_is_read_from_its_bid_candles_and_an_hour_without_ticks_is_no_bar():
    month = pd.Timestamp("2020-01-01", tz="UTC")
    got = refresh.dukascopy_hours(_bi5([(3600, 110000, 110010, 109990, 110020, 5.0), (7200, 1, 1, 1, 1, 0.0)]), month)
    assert list(got.index) == [month + pd.Timedelta(hours=1)]                               # stamped at its start
    assert got.iloc[0][["open", "high", "low", "close"]].tolist() == [110000, 110020, 109990, 110010]


def test_an_fx_pair_is_extended_back_from_dukascopy_only_when_its_closes_match_the_vendors(monkeypatch, tmp_path):
    monkeypatch.setattr(refresh, "DUKASCOPY_FIRST_YEAR", 2020)
    monkeypatch.setattr(refresh, "DUKASCOPY_DIR", tmp_path / "dukascopy")
    monkeypatch.setattr(refresh.time, "sleep", lambda s: None)
    closes = pd.date_range("2020-02-03 11:00", periods=3, freq="h", tz="UTC")               # the vendor's first hours
    for pair in ("EUR/USD", "GBP/USD"):
        store.write_bars("td", "1h", pair, _hourly_bars(closes, price=1.1))
    day = pd.date_range("2020-01-06 22:00", periods=24, freq="h")       # Tuesday's FX day: 17:00 to 17:00 New York
    jan = [(int((t - pd.Timestamp("2020-01-01")).total_seconds()), 110100, 110100, 110100, 110100, 3.0) for t in day]
    feb = [(int((t - pd.Timedelta(hours=1) - pd.Timestamp("2020-02-01", tz="UTC")).total_seconds()), 110000, 110000,
            110000, 110000, 3.0) for t in closes]                                            # the vendor's hours, 1.1000
    off = [(r[0], 130000, 130000, 130000, 130000, 3.0) for r in feb]                         # another price basis

    class _Duka:
        calls = 0

        def get(self, url, timeout=None, headers=None):
            _Duka.calls += 1
            r = _Resp(200, "")
            r.content = _bi5(jan if "/2020/00/" in url else (feb if "EURUSD" in url else off))
            return r
    monkeypatch.setattr(refresh.requests, "Session", lambda: _Duka())
    rep = refresh.extend_from_dukascopy(["EUR/USD", "GBP/USD"], dry_run=False)
    assert list(rep["extended"]) == ["EUR/USD"] and "GBP/USD" in rep["skipped"]
    eur = store.read_bars("td", "1h", "EUR/USD")
    assert eur.index[0] == pd.Timestamp("2020-01-06 23:00", tz="UTC") and eur["close"].iloc[0] == pytest.approx(1.101)
    assert len(eur) == 27 and list(eur.index[24:]) == list(closes)                          # the vendor's hours kept
    daily = store.read_bars("td", "1d", "EUR/USD")                                         # rebuilt from the hourly
    assert list(daily.index) == [pd.Timestamp("2020-01-07 22:00", tz="UTC")] and daily["close"].iloc[0] == pytest.approx(1.101)
    assert store.read_bars("td", "1h", "GBP/USD").index.min() == closes[0]                  # left as it was
    asked = _Duka.calls
    refresh.extend_from_dukascopy(["EUR/USD", "GBP/USD"], dry_run=False)
    assert _Duka.calls == asked                                    # every month is on disk: a new run asks nothing



def test_dukascopy_is_matched_on_the_mid_of_its_quote_and_an_extended_series_is_left_as_it_is(monkeypatch, tmp_path):
    monkeypatch.setattr(refresh, "DUKASCOPY_FIRST_YEAR", 2020)
    monkeypatch.setattr(refresh, "DUKASCOPY_FIRST_MONTH", {})
    monkeypatch.setattr(refresh, "DUKASCOPY_DIR", tmp_path / "dukascopy")
    monkeypatch.setattr(refresh.time, "sleep", lambda s: None)
    closes = pd.date_range("2020-02-03 11:00", periods=3, freq="h", tz="UTC")
    store.write_bars("td", "1h", "XAG/USD", _hourly_bars(closes, price=17.0))
    day = pd.date_range("2020-01-06 22:00", periods=24, freq="h")
    seconds = lambda t, month: int((t - pd.Timestamp(month)).total_seconds())               # noqa: E731
    jan = [(seconds(t, "2020-01-01"), 16990, 16990, 16990, 16990, 3.0) for t in day]
    feb = lambda price: [(seconds(t.tz_localize(None) - pd.Timedelta(hours=1), "2020-02-01"), price, price, price,  # noqa: E731
                          price, 3.0) for t in closes]

    class _Duka:
        def get(self, url, timeout=None, headers=None):
            r = _Resp(200, "")
            ask = "ASK_" in url
            r.content = _bi5(jan if "/2020/00/" in url else feb(17010 if ask else 16990))     # bid 10 bp under, ask over
            return r
    monkeypatch.setattr(refresh.requests, "Session", lambda: _Duka())
    rep = refresh.extend_from_dukascopy(["XAG/USD"], dry_run=False)
    assert list(rep["extended"]) == ["XAG/USD"]                         # the bid alone would be 10 bp from the vendor
    assert store.read_meta("td", "1h", "XAG/USD")["dukascopy_overlap_bp"] == 0.0
    store.write_meta("td", "1h", "XAG/USD", {**store.read_meta("td", "1h", "XAG/USD"), "dukascopy_overlap_bp": 0.9})
    again = refresh.extend_from_dukascopy(["XAG/USD"], dry_run=False)
    assert again["unchanged"] == ["XAG/USD"] and not again["extended"]
    assert store.read_meta("td", "1h", "XAG/USD")["dukascopy_overlap_bp"] == 0.9                 # untouched


def _sharadar_zip(path, name, frame):
    import zipfile
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{name}.csv", frame.to_csv(index=False))


def test_sharadar_membership_spans_pair_additions_and_removals(monkeypatch, tmp_path):
    from strategy_lab.data import sharadar
    monkeypatch.setattr(sharadar, "DIR", tmp_path)
    rows = [("1957-03-04", "added", "OLD1"), ("2008-12-31", "removed", "OLD1"),            # in, then out
            ("2001-11-30", "removed", "ENRNQ"),                                  # its addition missing from the table
            ("2010-01-04", "added", "NEW"), ("2020-06-22", "removed", "NEW"), ("2024-03-18", "added", "NEW"),
            ("2026-06-30", "historical", "NEW")]
    _sharadar_zip(tmp_path / "sp500.zip", "sp500", pd.DataFrame(rows, columns=["date", "action", "ticker"]))
    got = sharadar.sp500_spans().sort_values(["ticker", "start"], na_position="first")
    assert [(r.ticker, str(r.start)[:10], str(r.end)[:10]) for r in got.itertuples()] == [
        ("ENRNQ", "1957-03-04", "2001-11-30"), ("NEW", "2010-01-04", "2020-06-22"), ("NEW", "2024-03-18", "NaT"),
        ("OLD1", "1957-03-04", "2008-12-31")]


def test_sharadar_bars_are_stamped_at_the_nyse_close_and_a_spin_off_is_paid_like_a_dividend(monkeypatch, tmp_path):
    from strategy_lab.data import sharadar
    monkeypatch.setattr(sharadar, "DIR", tmp_path)
    monkeypatch.setattr(refresh, "SHARADAR_DIR", tmp_path)
    monkeypatch.setattr(bars, "STORE_DIR", tmp_path / "store")
    csv = ("ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated\n"
           "HPQ,2015-10-30,27.0,27.5,26.8,27.2,100,20.0,27.2,2020-01-01\n"
           "HPQ,2015-11-02,12.9,13.3,12.8,13.1,300,20.1,13.1,2020-01-01\n"            # the day HPE was spun off
           "HPQ,2015-11-01,13.0,13.0,13.0,13.0,1,13.0,13.0,2020-01-01\n")             # a Sunday: no session
    (tmp_path / "stocks").mkdir()
    (tmp_path / "stocks" / "HPQ.csv").write_text(csv)
    _sharadar_zip(tmp_path / "actions.zip", "actions", pd.DataFrame(
        [("2015-11-02", "spinoffdividend", "HPQ", 16.62), ("2015-09-09", "dividend", "HPQ", 0.1771),
         ("2015-11-02", "spinoff", "HPQ", 1.0)], columns=["date", "action", "ticker", "value"]))
    rep = refresh.store_sharadar_stocks(["HPQ", "GONE"])
    assert rep == {"stored": 1, "no_bars": ["GONE"], "rows_from_twelvedata": []}
    got = store.read_bars("sh", "1d", "HPQ")
    assert list(got.index) == [pd.Timestamp("2015-10-30 20:00", tz="UTC"), pd.Timestamp("2015-11-02 21:00", tz="UTC")]
    assert got["dollar_volume"].tolist() == [2720.0, 3930.0]
    paid = bars.load_dividends("sh:HPQ")
    assert paid.to_dict() == {pd.Timestamp("2015-09-09"): 0.1771, pd.Timestamp("2015-11-02"): 16.62}


def test_a_members_hourly_bars_are_put_on_its_sharadar_daily_basis_unless_they_are_another_companys(monkeypatch):
    sessions = pd.bdate_range("2024-01-02", periods=30)
    hours = pd.DatetimeIndex([pd.Timestamp(d, tz="America/New_York") + pd.Timedelta(hours=h, minutes=30)
                              for d in sessions for h in (10, 11, 12, 13, 14, 15)]).tz_convert("UTC")
    hours = hours.append(pd.DatetimeIndex([pd.Timestamp(d, tz="America/New_York").replace(hour=16).tz_convert("UTC")
                                          for d in sessions])).sort_values()
    daily = _daily([d.strftime("%Y-%m-%d") for d in sessions], [100.0 + k for k in range(len(sessions))])
    day_of = (hours - pd.Timedelta(microseconds=1)).tz_convert("America/New_York").normalize().tz_localize(None)
    c = pd.Series(pd.Series(daily["close"].to_numpy(), index=list(sessions)).reindex(day_of).to_numpy(), index=hours)
    later = pd.DatetimeIndex([pd.Timestamp("2024-03-04 16:00", tz="America/New_York").tz_convert("UTC")])
    c = pd.concat([c, pd.Series(3.0, index=later)])                        # the ticker's next company, after this one
    for t, scale in (("SAME", 1.0), ("OTHER", 5.0)):                     # OTHER: another company's prices
        store.write_bars("sh", "1d", t, daily)
        store.write_bars("td", "1h", t, pd.DataFrame({"open": c * scale, "high": c * scale, "low": c * scale,
                                                      "close": c * scale, "volume": 1.0, "dollar_volume": c}))
    store.write_bars("sh", "1h", "OTHER", daily)                         # written before from the other company's bars
    basis = refresh.td_hourly_on_daily_basis                               # its heuristics have tests of their own

    def matched(h1, d1):
        if h1["close"].iloc[0] / d1["close"].iloc[0] > 2:
            return h1.iloc[:0], {"dropped_all": True}
        return basis(h1, d1)
    monkeypatch.setattr(refresh, "td_hourly_on_daily_basis", matched)
    rep = refresh.store_sharadar_hourly(["SAME", "OTHER", "NONE"])
    assert rep == {"hourly_stored": 1, "another_company": ["OTHER"], "no_hourly": 1}
    assert store.read_bars("sh", "1h", "SAME")["close"].iloc[-1] == daily["close"].iloc[-1]
    assert store.read_bars("sh", "4h", "SAME")["close"].iloc[-1] == daily["close"].iloc[-1]      # its 4h built too
    assert later[0] not in store.read_bars("sh", "1h", "SAME").index
    assert not store.path("sh", "1h", "OTHER").exists()                  # moved to data/quarantine


def test_hourly_bars_from_the_minutes_of_every_venue_sit_on_the_stores_grid_with_both_crosses():
    ny = "America/New_York"
    minute = lambda hh, mm, day="2023-06-01": pd.Timestamp(f"{day} {hh:02d}:{mm:02d}", tz=ny).tz_convert("UTC")  # noqa: E731
    rows = [  # (minute, venue volume, open, high, low, close)
        (minute(9, 29), 10, 80.0, 80.0, 80.0, 80.0),        # before the open: not a session bar
        (minute(9, 30), 45000, 80.3, 80.35, 80.0, 80.02),   # the opening cross, on the listing venue
        (minute(9, 30), 500, 80.25, 80.3, 80.1, 80.2),      # the same minute on another venue
        (minute(10, 29), 100, 80.4, 80.5, 79.9, 80.45),
        (minute(10, 30), 200, 80.45, 80.6, 80.4, 80.5),     # the second hour begins
        (minute(15, 59), 900, 80.18, 80.2, 80.1, 80.19),
        (minute(16, 0), 1400000, 80.2, 80.3, 80.2, 80.27),  # the closing cross at 80.20, then a trade after hours
        (minute(16, 1), 50, 80.3, 80.3, 80.3, 80.3),        # after hours: not a session bar
    ]
    idx = pd.DatetimeIndex([r[0] for r in rows])
    m = pd.DataFrame([r[1:] for r in rows], columns=["volume", "open", "high", "low", "close"], index=idx)
    h = refresh.hourly_from_minutes(m[["open", "high", "low", "close", "volume"]])
    assert list(h.index) == [minute(10, 30), minute(11, 30), minute(16, 0)]
    first, second, last = h.iloc[0], h.iloc[1], h.iloc[2]
    assert (first["open"], first["high"], first["low"], first["close"]) == (80.3, 80.5, 79.9, 80.45)
    assert first["volume"] == 45000 + 500 + 100 and second["open"] == 80.45
    assert last["close"] == 80.2 and last["volume"] == 900 + 1400000       # the cross's price and volume
    quiet = pd.DataFrame([(20.0, 20.1, 19.9, 20.05, 5000), (20.8, 20.8, 20.8, 20.8, 10)],   # no auction in the feed:
                         columns=["open", "high", "low", "close", "volume"],             # a print after the close
                         index=pd.DatetimeIndex([minute(15, 59, "2023-06-02"), minute(16, 0, "2023-06-02")]))
    q = refresh.hourly_from_minutes(quiet).iloc[-1]
    assert (q["close"], q["high"], q["volume"]) == (20.05, 20.1, 5000)
    early = pd.DataFrame({"open": [50.0, 51.0], "high": [50.0, 51.0], "low": [50.0, 51.0], "close": [50.0, 51.0],
                          "volume": [1.0, 2.0]}, index=pd.DatetimeIndex([minute(12, 59, "2023-11-24"),
                                                                        minute(13, 0, "2023-11-24")]))
    assert list(refresh.hourly_from_minutes(early).index) == [minute(13, 0, "2023-11-24")]   # a half day's cross


def test_a_securitys_traded_tickers_come_from_its_ticker_changes(monkeypatch, tmp_path):
    from strategy_lab.data import sharadar
    monkeypatch.setattr(sharadar, "DIR", tmp_path)
    _sharadar_zip(tmp_path / "actions.zip", "actions", pd.DataFrame(
        [("2016-09-13", "tickerchangefrom", "INFO1", "MRKT"), ("2022-02-25", "tickerchangefrom", "INFO1", "INFO"),
         ("2024-08-22", "tickerchangefrom", "GAP", "GPS")], columns=["date", "action", "ticker", "contraticker"]))
    d = lambda s: pd.Timestamp(s, tz="UTC")                                               # noqa: E731
    assert refresh.traded_periods("INFO1", d("2022-02-25 21:00")) == [("MRKT", None, d("2016-09-13")),
                                                                      ("INFO", d("2016-09-13"), d("2022-02-25"))]
    assert refresh.traded_periods("GAP", d("2026-09-25 20:00")) == [("GPS", None, d("2024-08-22")),
                                                                    ("GAP", d("2024-08-22"), d("2026-09-26 20:00"))]
    assert refresh.traded_periods("ATVI", d("2023-10-12 20:00")) == [("ATVI", None, d("2023-10-13 20:00"))]


def _databento_dirs(monkeypatch, tmp_path):
    from strategy_lab.data import sharadar
    monkeypatch.setattr(sharadar, "DIR", tmp_path / "sharadar")
    monkeypatch.setattr(refresh, "DATABENTO_DIR", tmp_path / "databento")
    monkeypatch.setattr(refresh, "DATABENTO_VENUES", ("XNAS.ITCH", "XNYS.PILLAR"))


def _minutes(rows):
    """(New York minute 'YYYY-MM-DD HH:MM', price, volume) as a venue's one-minute bars."""
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="America/New_York").tz_convert("UTC") for t, _, _ in rows],
                           name="ts_event")
    p = [r[1] for r in rows]
    return pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": [r[2] for r in rows]}, index=idx)


def test_hourly_holes_are_asked_per_venue_under_the_ticker_traded_then_and_never_twice(monkeypatch, tmp_path):
    _databento_dirs(monkeypatch, tmp_path)
    _sharadar_zip(tmp_path / "sharadar" / "actions.zip", "actions", pd.DataFrame(
        [("2024-08-22", "tickerchangefrom", "GAP", "GPS")], columns=["date", "action", "ticker", "contraticker"]))
    sessions = cal.nyse_sessions(pd.Timestamp("2024-07-15", tz="UTC"), pd.Timestamp("2024-09-30", tz="UTC"))
    days = [d.strftime("%Y-%m-%d") for d in sessions.index]
    store.write_bars("sh", "1d", "GAP", _daily(days, [20.0] * len(days)))
    lacking = {"2024-07-16", "2024-08-20", "2024-08-21", "2024-08-23", "2024-09-25"}
    had = [d for d in days if d not in lacking]
    store.write_bars("sh", "1h", "GAP", _daily(had, [20.0] * len(had)))
    done = refresh._minutes_path("XNAS.ITCH", "GPS", pd.Timestamp("2024-07-16"), pd.Timestamp("2024-07-17"))
    done.parent.mkdir(parents=True)
    refresh._no_minutes().to_parquet(done)                                   # asked before: GPS did not trade there
    holes = refresh.hourly_holes("sh", "GAP")
    assert [str(d.date()) for d in holes] == sorted(lacking)
    got = [(v, raw, str(a.date()), str(b.date())) for v, raw, a, b in refresh.databento_windows("sh", "GAP", holes)]
    # GPS until the change and GAP after it, both within five sessions of it (Sharadar's date can be a week off);
    # a day a venue was asked for already is not asked again
    assert got == [("XNAS.ITCH", "GPS", "2024-08-20", "2024-08-22"), ("XNAS.ITCH", "GPS", "2024-08-23", "2024-08-24"),
                   ("XNYS.PILLAR", "GPS", "2024-07-16", "2024-07-17"),
                   ("XNYS.PILLAR", "GPS", "2024-08-20", "2024-08-22"), ("XNYS.PILLAR", "GPS", "2024-08-23", "2024-08-24"),
                   ("XNAS.ITCH", "GAP", "2024-08-20", "2024-08-22"), ("XNAS.ITCH", "GAP", "2024-08-23", "2024-08-24"),
                   ("XNAS.ITCH", "GAP", "2024-09-25", "2024-09-26"),
                   ("XNYS.PILLAR", "GAP", "2024-08-20", "2024-08-22"), ("XNYS.PILLAR", "GAP", "2024-08-23", "2024-08-24"),
                   ("XNYS.PILLAR", "GAP", "2024-09-25", "2024-09-26")]


def test_minutes_are_bought_a_request_per_venue_and_span_and_only_within_the_cap(monkeypatch, tmp_path):
    import databento
    _databento_dirs(monkeypatch, tmp_path)
    asked = []

    class _Data:
        def __init__(self, df):
            self.df = df

        def to_df(self):
            return self.df

    class _Client:
        def __init__(self, key):
            self.metadata, self.timeseries = self, self

        def get_cost(self, **ask):
            if ask["symbols"] == ["ZZZ"]:
                raise databento.BentoClientError(422, json_body={"detail": {        # as Databento answers it
                    "case": "symbology_invalid_request", "message": "None of the symbols could be resolved"}})
            return 0.01 * len(ask["symbols"])

        def get_range(self, **ask):
            asked.append((ask["dataset"], ask["symbols"]))
            return _Data(_minutes([("2024-08-21 09:30", 10.0, 5)]).assign(symbol="AAA"))   # BBB did not trade

    monkeypatch.setattr(databento, "Historical", _Client)
    d0, d1 = pd.Timestamp("2024-08-21"), pd.Timestamp("2024-08-22")
    windows = [("XNAS.ITCH", "AAA", d0, d1), ("XNAS.ITCH", "BBB", d0, d1), ("XNYS.PILLAR", "AAA", d0, d1),
               ("XNYS.PILLAR", "ZZZ", d1, d1 + pd.Timedelta(days=1))]
    assert refresh.fetch_databento_minutes(windows) == {"windows": 4, "requests": 3, "none_traded": 1,
                                                        "cost_usd": 0.03}
    with pytest.raises(RuntimeError, match="over the --max-usd"):
        refresh.fetch_databento_minutes(windows, dry_run=False, max_usd=0.02)
    assert not asked and not (tmp_path / "databento").exists()
    refresh.fetch_databento_minutes(windows, dry_run=False, max_usd=1.0)
    assert sorted(asked) == [("XNAS.ITCH", ["AAA", "BBB"]), ("XNYS.PILLAR", ["AAA"])]    # nothing asked for ZZZ
    read = lambda v, s, a, b: pd.read_parquet(refresh._minutes_path(v, s, a, b))           # noqa: E731
    assert len(read("XNAS.ITCH", "AAA", d0, d1)) == 1 and len(read("XNYS.PILLAR", "AAA", d0, d1)) == 1
    assert read("XNAS.ITCH", "BBB", d0, d1).empty                            # remembered as not traded there
    assert read("XNYS.PILLAR", "ZZZ", d1, d1 + pd.Timedelta(days=1)).empty
    assert refresh._asked("XNAS.ITCH", "BBB") == [(d0, d1)]


def test_hourly_bars_from_databento_keep_a_tickers_minutes_to_its_own_company(monkeypatch, tmp_path):
    _databento_dirs(monkeypatch, tmp_path)
    _sharadar_zip(tmp_path / "sharadar" / "actions.zip", "actions", pd.DataFrame(
        [("2024-08-22", "tickerchangefrom", "GAP", "GPS")], columns=["date", "action", "ticker", "contraticker"]))
    store.write_bars("sh", "1d", "GAP", _daily(["2024-08-21", "2024-08-22", "2024-08-23"], [20.0, 21.0, 21.0]))
    for venue, raw, day, price in (("XNAS.ITCH", "GPS", "2024-08-21", 20.0), ("XNYS.PILLAR", "GAP", "2024-08-22", 21.0),
                                   ("XNAS.ITCH", "GPS", "2024-08-22", 77.0),     # GPS printed too: GAP's day
                                   ("XNAS.ITCH", "GPS", "2024-08-23", 21.5),     # only GPS printed: borrowed
                                   ("XNAS.ITCH", "GPS", "2024-09-03", 99.0)):    # GPS as another company since
        a = pd.Timestamp(day)
        path = refresh._minutes_path(venue, raw, a, a + pd.Timedelta(days=1))
        path.parent.mkdir(parents=True, exist_ok=True)
        _minutes([(f"{day} 09:30", price, 100), (f"{day} 16:00", price, 1000)]).to_parquet(path)
    assert refresh.build_databento_hourly("sh", "GAP") == 3
    got = pd.read_parquet(refresh._databento_hourly_path("sh", "GAP"))
    assert sorted(set(got["close"])) == [20.0, 21.0, 21.5]
    assert got.groupby(refresh._session_days(got.index))["borrowed"].first().tolist() == [False, False, True]


def test_databento_sessions_fill_only_what_the_hourly_bars_lack_on_the_daily_split_basis(monkeypatch, tmp_path):
    _databento_dirs(monkeypatch, tmp_path)
    _sharadar_zip(tmp_path / "sharadar" / "actions.zip", "actions", pd.DataFrame(
        [("2024-06-10", "split", "XYZ", 10.0)], columns=["date", "action", "ticker", "value"]))
    daily = _daily(["2024-06-06", "2024-06-07", "2024-06-10", "2024-06-11", "2024-06-12", "2024-06-13"],
                   [120.0, 121.0, 123.0, 123.0, 124.0, 125.0])
    daily["open"] = [119.5, 121.0, 121.8, 123.0, 123.5, 125.5]
    daily["volume"] = [1000.0, 1100.0, 900.0, 800.0, 500.0, 500.0]            # consolidated, split-adjusted
    ny = lambda t: pd.Timestamp(t, tz="America/New_York").tz_convert("UTC")                 # noqa: E731
    raw = pd.DataFrame({"open": [1195.0, 1200.0, 999.0, 121.5, 122.0, 150.0, 123.8, 90.0],
                        "high": [1201.0, 1202.0, 999.0, 122.5, 122.5, 150.0, 124.2, 90.0],
                        "low": [1190.0, 1195.0, 999.0, 121.0, 121.5, 150.0, 123.4, 90.0],
                        "close": [1198.0, 1200.0, 999.0, 122.0, 122.0, 150.0, 124.0, 90.0],
                        "volume": [30.0, 70.0, 5.0, 40.0, 10.0, 7.0, 9.0, 9.0], "dollar_volume": 0.0,
                        "borrowed": [False] * 6 + [True, True]},     # 06-12 and 06-13: the other ticker's prints
                       index=pd.DatetimeIndex([ny("2024-06-06 15:30"), ny("2024-06-06 16:00"), ny("2024-06-07 16:00"),
                                               ny("2024-06-10 15:30"), ny("2024-06-10 16:00"), ny("2024-06-11 16:00"),
                                               ny("2024-06-12 16:00"), ny("2024-06-13 16:00")]))
    path = refresh._databento_hourly_path("sh", "XYZ")
    path.parent.mkdir(parents=True)
    raw.to_parquet(path)
    h1 = _daily(["2024-06-07"], [121.0])                                   # the session Twelve Data has
    out, rep = refresh.fill_from_databento("sh", "XYZ", h1, daily)
    assert rep == {"sessions": 4, "opens_from_the_daily_bar": 3, "daily_bar_repeats_the_close_before": ["2024-06-11"],
                   "borrowed_left_out": ["2024-06-13"]}                   # 90 is another company's price: not 125
    assert out.loc[ny("2024-06-12 16:00"), "close"] == 124.0 and ny("2024-06-13 16:00") not in out.index
    assert out.loc[ny("2024-06-07 16:00"), "close"] == 121.0                  # its own session is kept
    before = out.loc[[ny("2024-06-06 15:30"), ny("2024-06-06 16:00")]]
    assert before["open"].iloc[0] == 119.5                                  # the daily open, inside the first hour
    assert before["close"].tolist() == pytest.approx([119.8, 120.0])        # a tenth: printed before the 10:1 split
    assert before["volume"].tolist() == pytest.approx([300.0, 700.0])       # the day's 1000 in the lit shares' shape
    after = out.loc[[ny("2024-06-10 15:30"), ny("2024-06-10 16:00")]]
    assert after["close"].tolist() == [122.0, 123.0] and after["high"].iloc[1] == 123.0   # the auction's price
    assert after["volume"].tolist() == pytest.approx([720.0, 180.0])
    stale = out.loc[ny("2024-06-11 16:00")]                  # the daily bar repeats 06-10's close: no price of its own
    assert (stale["close"], stale["volume"]) == (150.0, 7.0)
    assert (out["dollar_volume"] == out["close"] * out["volume"]).loc[before.index].all()


def test_an_fx_hole_is_filled_from_the_mid_of_dukascopys_quote_and_a_metals_daily_pause_is_no_hole(monkeypatch,
                                                                                                   tmp_path):
    monkeypatch.setattr(refresh, "DUKASCOPY_DIR", tmp_path / "dukascopy")
    monkeypatch.setattr(refresh, "REFERENCE_DIR", tmp_path / "reference")
    monkeypatch.setattr(refresh.time, "sleep", lambda s: None)
    (tmp_path / "reference").mkdir()
    pd.DataFrame({"symbol": ["EUR/USD", "XAG/USD"], "interval": ["1h", "1h"],
                  "first": ["2021-07-01 00:00:00+00:00"] * 2}).to_csv(tmp_path / "reference" / refresh.TD_FIRST_BARS,
                                                                       index=False)
    closes = pd.date_range("2021-07-06 12:00", periods=10, freq="h", tz="UTC")           # a Tuesday, 08:00-17:00 NY
    hole = closes[4]
    for s in ("EUR/USD", "XAG/USD"):
        store.write_bars("td", "1h", s, _hourly_bars(closes.delete(4), price=1.2))
        store.write_meta("td", "1h", s, {"extended_from": "dukascopy"})
    assert list(refresh.fx_hourly_holes("EUR/USD")) == [hole]
    pause = pd.Timestamp("2021-07-06 22:00", tz="UTC")               # 17:00-18:00 New York: a metal's daily pause
    store.write_bars("td", "1h", "XAG/USD", _hourly_bars(closes.delete(4).append(
        pd.DatetimeIndex([pause + pd.Timedelta(hours=1)])), price=1.2))
    assert list(refresh.fx_hourly_holes("XAG/USD")) == [hole]                         # the pause is not a hole
    seconds = lambda t: int((t - pd.Timedelta(hours=1) - pd.Timestamp("2021-07-01", tz="UTC")).total_seconds())  # noqa: E731

    class _Duka:
        def get(self, url, timeout=None, headers=None):
            r = _Resp(200, "")
            ask = "ASK_" in url
            price = (120010 if ask else 119990) if "EURUSD" in url else (140010 if ask else 139990)   # XAG: off
            r.content = _bi5([(seconds(t), price, price, price, price, 3.0) for t in closes])
            return r
    monkeypatch.setattr(refresh.requests, "Session", lambda: _Duka())
    rep = refresh.fill_from_dukascopy(["EUR/USD", "XAG/USD"], dry_run=False)
    assert list(rep["filled"]) == ["EUR/USD"] and "XAG/USD" in rep["skipped"]
    eur = store.read_bars("td", "1h", "EUR/USD")
    assert list(eur.index) == list(closes) and eur.loc[hole, "close"] == pytest.approx(1.2)   # the mid, not the bid
    assert store.read_meta("td", "1h", "EUR/USD")["dukascopy_filled_hours"] == 1
    assert hole not in store.read_bars("td", "1h", "XAG/USD").index


def test_sharadars_rows_for_days_it_lost_the_ticker_come_from_twelve_data_on_sharadars_basis():
    days = ["2020-11-12", "2020-11-13", "2020-11-16", "2020-11-17", "2020-11-18", "2020-11-19"]

    def bars(o, h, lo, c, v):
        idx = pd.DatetimeIndex([pd.Timestamp(d, tz="America/New_York").replace(hour=16).tz_convert("UTC") for d in days])
        return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v}, index=idx, dtype=float).assign(
            dollar_volume=lambda f: f["close"] * f["volume"])
    # Sharadar carries 11.0 forward on 11-16 and 11-17 (a day it lost the ticker); the last row repeats too
    sh = bars([10, 11, 11, 11, 12, 12], [10, 11, 11, 11, 12, 12], [10, 11, 11, 11, 12, 12],
              [10, 11, 11, 11, 12, 12], [100, 100, 0, 0, 100, 0])
    # Twelve Data on another basis (twice Sharadar's prices, half its shares): 11-16 traded, 11-17 was halted
    td = bars([20, 22, 22.2, 22.6, 24, 24.5], [20, 22, 23, 22.6, 24, 25], [20, 22, 22, 22.6, 24, 24],
              [20, 22, 22.6, 22.6, 24, 24.8], [50, 50, 200, 0, 50, 60])
    store.write_bars("td", "1d", "VTRS", td)
    got, fixed = refresh.repair_sharadar_rows("VTRS", sh)
    assert fixed == ["2020-11-16"]
    assert got.iloc[2][["open", "high", "low", "close", "volume"]].tolist() == pytest.approx([11.1, 11.5, 11.0, 11.3, 400])
    assert got.iloc[3]["close"] == 11.0 and got.iloc[5]["close"] == 12.0      # the halt and the last row stay
    split = td.copy()
    split.iloc[4:, :4] *= 0.5                                             # a split between: the basis moved
    store.write_bars("td", "1d", "VTRS", split)
    assert refresh.repair_sharadar_rows("VTRS", sh)[1] == []
    tiny = td.copy()
    tiny.iloc[2, :4] = [22.0, 22.0014, 21.9996, 22.0006]                 # a range Sharadar's rounding flattens
    store.write_bars("td", "1d", "VTRS", tiny)
    assert refresh.repair_sharadar_rows("VTRS", sh)[1] == []


def test_an_hourly_session_is_made_the_parts_of_its_daily_bar():
    ny = lambda d, hm: pd.Timestamp(f"{d} {hm}", tz="America/New_York").tz_convert("UTC")          # noqa: E731
    stamps = ["10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00"]
    days = ["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06"]
    rows = []
    for d in days:
        for k, hm in enumerate(stamps):
            price = {"2024-06-04": 200.0}.get(d, 100.0 + k)                     # 06-04: the other basis
            rows.append((ny(d, hm), price, price + 0.5, price - 0.5, price, 10.0))
    h1 = pd.DataFrame([r[1:] for r in rows], columns=["open", "high", "low", "close", "volume"],
                      index=pd.DatetimeIndex([r[0] for r in rows])).assign(dollar_volume=0.0)
    h1.loc[ny("2024-06-03", "12:30"), "low"] = 40.0                           # a print far below the day
    h1 = h1.drop(index=ny("2024-06-06", "11:30"))
    h1.loc[ny("2024-06-06", "11:00")] = [101.0, 101.5, 100.5, 101.0, 10.0, 0.0]   # 06-06: a bar off the grid
    h1 = h1.sort_index()
    h1["dollar_volume"] = h1["close"] * h1["volume"]
    daily = _daily(days, [106.2, 100.0, 100.0, 106.0])
    daily["open"], daily["high"], daily["low"], daily["volume"] = [99.8, 99.0, 100.0, 100.0], \
        [106.8, 101.0, 100.0, 107.0], [99.3, 98.0, 100.0, 99.0], [700.0, 700.0, 0.0, 700.0]
    daily.iloc[2, :4] = 100.0                                                 # 06-05 repeats 06-04's close: no price
    out, rep = refresh.conform_to_daily(h1, daily)
    assert rep["sessions_dropped_not_the_days_prices"] == ["2024-06-04"]
    assert rep["sessions_dropped_off_the_grid"] == ["2024-06-06"]
    day1 = out[refresh._session_days(out.index) == pd.Timestamp("2024-06-03")]
    assert day1["low"].min() == 99.3 and day1["high"].max() <= 106.8          # held within the day's range
    assert (day1["open"].iloc[0], day1["close"].iloc[-1]) == (99.8, 106.2)     # the day's open and close
    assert day1["volume"].sum() == pytest.approx(700.0)
    assert ((day1["low"] <= day1[["open", "close"]].min(axis=1)) & (day1["high"] >= day1[["open", "close"]].max(axis=1))).all()
    day3 = out[refresh._session_days(out.index) == pd.Timestamp("2024-06-05")]
    pd.testing.assert_frame_equal(day3, h1[refresh._session_days(h1.index) == pd.Timestamp("2024-06-05")])
    quiet = h1.copy()                                     # 06-03's bars carry no volume: the shape of the other days
    quiet.loc[refresh._session_days(quiet.index) == pd.Timestamp("2024-06-03"), "volume"] = 0.0
    other = _daily(["2024-06-07"], [100.0]).assign(open=100.0, high=110.0, low=90.0, volume=700.0)
    rows7 = [(ny("2024-06-07", hm), 100.0, 101.0, 99.0, 100.0, float(k + 1)) for k, hm in enumerate(stamps)]
    day7 = pd.DataFrame([r[1:] for r in rows7], columns=["open", "high", "low", "close", "volume"],
                        index=pd.DatetimeIndex([r[0] for r in rows7])).assign(dollar_volume=0.0)
    out2, rep2 = refresh.conform_to_daily(pd.concat([quiet, day7]).sort_index(), pd.concat([daily, other]))
    got = out2[refresh._session_days(out2.index) == pd.Timestamp("2024-06-03")]["volume"]
    assert rep2["sessions_given_the_days_volume"] == 1 and got.sum() == pytest.approx(700.0)
    assert got.tolist() == pytest.approx([700.0 * k / 28 for k in range(1, 8)])     # 06-07's shape: 1, 2 ... 7
    lumped = quiet.copy()                                 # 06-03's volume all in its last bar, none in its others
    lumped.loc[lumped.index[refresh._session_days(lumped.index) == pd.Timestamp("2024-06-03")][-1], "volume"] = 700.0
    out3, rep3 = refresh.conform_to_daily(pd.concat([lumped, day7]).sort_index(), pd.concat([daily, other]))
    got3 = out3[refresh._session_days(out3.index) == pd.Timestamp("2024-06-03")]["volume"]
    assert rep3["sessions_given_the_days_volume"] == 1
    assert got3.tolist() == pytest.approx([700.0 * k / 28 for k in range(1, 8)])


def test_a_reviewed_sharadar_split_row_a_day_off_is_put_on_the_adjusted_basis_and_a_new_one_is_named(monkeypatch,
                                                                                                     tmp_path, caplog):
    from strategy_lab.data import sharadar
    monkeypatch.setattr(sharadar, "DIR", tmp_path)
    _sharadar_zip(tmp_path / "actions.zip", "actions", pd.DataFrame(
        [("2018-02-28", "split", "BF.B", 1.25), ("2020-01-02", "split", "XYZ", 2.0)],
        columns=["date", "action", "ticker", "value"]))
    days = ["2018-02-27", "2018-02-28", "2018-03-01"]
    bars = _daily(days, [55.896, 69.79, 55.37]).assign(volume=[1510000.0, 1606000.0, 4122000.0])
    raw = pd.DataFrame({"date": days, "closeunadj": [69.87, 69.79, 55.37]})
    got, fixed = refresh._split_rows("BF.B", bars, raw)
    assert fixed == ["2018-02-28"] and got["close"].tolist() == pytest.approx([55.896, 55.832, 55.37])
    assert got["volume"].iloc[1] == pytest.approx(1606000 * 1.25)
    xyz_days = ["2019-12-31", "2020-01-02", "2020-01-03"]
    xyz = _daily(xyz_days, [50.0, 50.25, 50.3])
    with caplog.at_level("WARNING"):
        same, none = refresh._split_rows("XYZ", xyz, pd.DataFrame({"date": xyz_days, "closeunadj": [100.0, 100.5, 50.3]}))
    assert none == [] and same["close"].tolist() == [50.0, 50.25, 50.3]
    assert "not reviewed yet" in caplog.text


def test_a_cme_future_is_joined_across_its_rolls_on_the_latest_contracts_basis(monkeypatch, tmp_path):
    monkeypatch.setattr(refresh, "CME_DIR", tmp_path)
    monkeypatch.setitem(refresh.CME_CONTRACT_SIZE, "XX", 1_000)
    starts = pd.DatetimeIndex(["2024-01-18 14:00", "2024-01-18 15:00", "2024-01-19 14:00", "2024-01-19 15:00",   # Thu, Fri
                               "2024-01-20 12:00",                                                            # a Saturday
                               "2024-01-21 23:00", "2024-01-22 00:00"], tz="UTC")                            # Sunday's open
    v0 = pd.DataFrame({"instrument_id": [1, 1, 1, 1, 1, 2, 2], "open": 100.0, "high": 101.0, "low": 99.0,
                       "close": [100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 111.0], "volume": 5.0}, index=starts)
    v1 = pd.DataFrame({"instrument_id": 2, "open": 110.0, "high": 110.0, "low": 110.0, "close": 110.0, "volume": 1.0},
                      index=starts[:4])                                # the next contract, last printed on Friday
    v0.loc[starts[5:], ["open", "high", "low"]] = [[110.0, 111.5, 109.5], [110.5, 111.5, 110.0]]
    v0.to_parquet(tmp_path / "XX.v.0.ohlcv-1h.parquet")
    v1.to_parquet(tmp_path / "XX.v.1.ohlcv-1h.parquet")
    got, rolls = refresh.cme_back_adjusted("XX")
    assert rolls == {"rolls": 1, "rolls_without_a_common_hour": 0}         # matched on Friday, across the weekend
    assert list(got.index) == list(starts.delete(4) + pd.Timedelta(hours=1))   # Saturday is no trading hour
    assert got["close"].tolist() == pytest.approx([110.0, 110.0, 110.0, 110.0, 110.0, 111.0])   # no jump at the roll
    # a bar's dollar volume is what its contracts hold: 5 contracts of 1,000 units at the (adjusted) close
    assert got["dollar_volume"].tolist() == pytest.approx([550_000.0] * 5 + [555_000.0])


def test_a_coins_4h_and_daily_bars_are_carried_to_the_end_of_its_hourly_ones():
    hours = pd.date_range("2026-08-01 01:00", periods=24 * 3 + 5, freq="h", tz="UTC")   # three days and five hours
    h1 = pd.DataFrame({"open": np.arange(len(hours), dtype=float), "high": np.arange(len(hours)) + 0.5,
                       "low": np.arange(len(hours)) - 0.5, "close": np.arange(len(hours)) + 0.25, "volume": 1.0,
                       "dollar_volume": 2.0}, index=hours)
    store.write_bars("spot", "1h", "OLDUSDT", h1)
    store.write_bars("spot", "1d", "OLDUSDT", _hourly_bars([pd.Timestamp("2026-08-01", tz="UTC")]))   # stopped early
    store.write_bars("spot", "4h", "OLDUSDT", _hourly_bars([pd.Timestamp("2026-08-01", tz="UTC")]))
    rep = refresh.extend_crypto_from_hourly("spot", ["OLDUSDT"])
    d = store.read_bars("spot", "1d", "OLDUSDT")
    assert list(d.index[1:]) == list(pd.date_range("2026-08-02", periods=4, freq="D", tz="UTC"))   # the last in part
    assert (d["open"].iloc[1], d["close"].iloc[1], d["volume"].iloc[1]) == (0.0, 23.25, 24.0)
    assert d["volume"].iloc[-1] == 5.0 and "OLDUSDT 4h" in rep


def test_an_etfs_daily_bars_are_sharadars_but_a_day_the_venues_session_contradicts(monkeypatch, tmp_path):
    monkeypatch.setattr(refresh, "SHARADAR_DIR", tmp_path / "sharadar")
    monkeypatch.setattr(refresh, "DATABENTO_DIR", tmp_path / "databento")
    monkeypatch.setattr(refresh, "_hourly_written", lambda s: None)
    monkeypatch.setattr(refresh.sharadar, "split_factor", lambda t, days: np.ones(len(days)))
    days = pd.DatetimeIndex(["2020-06-29", "2020-06-30", "2020-07-01", "2020-07-02"])
    before = pd.DataFrame({"open": 86.0, "high": 86.1, "low": 85.9, "close": 86.0, "volume": 10.0, "dollar_volume": 0.0},
                          index=cal.equity_daily_close(pd.DatetimeIndex(["2020-06-26"])).to_numpy())
    store.write_bars("td", "1d", "SHY", before)                           # the vendor's day before Sharadar's first
    (tmp_path / "sharadar" / "funds").mkdir(parents=True)
    sh = [86.60, 86.61, 86.47, 86.51]                                    # 07-01 sits under the venues
    rows = "\n".join(f"SHY,{d.date()},{c},{c + 0.02},{c - 0.02},{c},2000,80.0,{c},2026-01-01" for d, c in zip(days, sh))
    (tmp_path / "sharadar" / "funds" / "SHY.csv").write_text(
        "ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated\n" + rows + "\n")
    ny = lambda d, hm: pd.Timestamp(f"{d} {hm}", tz="America/New_York").tz_convert("UTC")   # noqa: E731
    hours = ["10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00"]
    session = [(ny("2020-06-30", h), 86.61) for h in hours] + [(ny("2020-07-01", h), 86.52 + 0.005 * k)
                                                               for k, h in enumerate(hours)]
    layer = pd.DataFrame({"open": [p for _, p in session], "high": [p + 0.01 for _, p in session],
                          "low": [p - 0.01 for _, p in session], "close": [p for _, p in session], "volume": 100.0,
                          "dollar_volume": 0.0, "borrowed": False}, index=pd.DatetimeIndex([t for t, _ in session]))
    (tmp_path / "databento" / "hourly" / "td").mkdir(parents=True)
    layer.to_parquet(tmp_path / "databento" / "hourly" / "td" / "SHY.parquet")
    rep = refresh.etf_daily_reconciled(["SHY"])["SHY"]
    out = store.read_bars("td", "1d", "SHY")
    assert rep["days_from_the_venues"] == ["2020-07-01"]
    assert out["close"].round(3).tolist() == [86.0, 86.60, 86.61, 86.55, 86.51]   # 07-01: the session's last close
    assert out[["open", "high", "low"]].iloc[3].tolist() == pytest.approx([86.52, 86.56, 86.51])
    assert out["volume"].iloc[3] == 2000.0                               # Sharadar's volume counts every venue
    assert store.read_meta("td", "1d", "SHY")["daily_from"] == "sharadar"
    monkeypatch.setattr(refresh, "td_series_to_refresh", lambda syms: [("SHY", "1d")])
    assert refresh.refresh_twelvedata(["SHY"], dry_run=True)["daily_reconciled_skipped"] == 1


def test_dukascopys_first_months_are_put_an_hour_earlier_and_its_mixed_week_left_out():
    """Dukascopy stamped its first months of the pairs an hour late (the US payrolls of June 2003 in its hour closing at
    14:00, not 13:00): those bars move an hour earlier; the week whose hours are late and on time in turn is left out,
    and later bars stay where they are."""
    from strategy_lab.data.refresh import dukascopy_on_utc
    stamps = pd.DatetimeIndex(["2003-06-06 12:00", "2003-06-06 13:00", "2003-07-29 12:00", "2003-08-04 12:00"], tz="UTC")
    raw = pd.DataFrame({"open": [1.0, 2.0, 3.0, 4.0]}, index=stamps)
    out = dukascopy_on_utc("EUR/USD", raw)
    assert list(out.index) == list(pd.DatetimeIndex(["2003-06-06 11:00", "2003-06-06 12:00", "2003-08-04 12:00"], tz="UTC"))
    assert list(out["open"]) == [1.0, 2.0, 4.0]
    gold = dukascopy_on_utc("XAU/USD", raw)                    # gold is late through its week of 2003-07-28 too
    assert list(gold.index) == list(pd.DatetimeIndex(["2003-06-06 11:00", "2003-06-06 12:00", "2003-07-29 11:00",
                                                      "2003-08-04 12:00"], tz="UTC"))
    assert dukascopy_on_utc("AUD/USD", raw).index.equals(stamps)     # its history begins in 2003-08
