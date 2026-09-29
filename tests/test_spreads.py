"""A spot quote of a commodity pays the spread its broker quoted at each fill: the hours of spreads as Exness's ticks
give them, a bar's spreads from its hours, and the engine, its trades and a switch of configuration charged them."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from strategy_lab.config import COSTS
from strategy_lab.data import spread_refresh, spreads
from strategy_lab.engine import backtest as bt
from strategy_lab.engine import costs
from strategy_lab.engine.trades import ledger
from tests.conftest import make_panel, quote_spreads
from tests.test_engine import _random_target, _units_account

FX = (COSTS["fx"].commission_bps + COSTS["fx"].half_spread_bps) / 1e4


def _hours_of(p) -> pd.DatetimeIndex:
    """The hours inside a daily panel's bars (24 a bar: make_panel's days follow one another), by their closes."""
    return pd.date_range(p.index[0] - pd.Timedelta(hours=23), p.index[-1], freq="h")


def test_an_hour_of_exness_ticks_keeps_the_spread_of_its_first_and_last_quote():
    rows = ['"Exness","Symbol","Timestamp","Bid","Ask"',
            '"exness","XPTUSD","2024-01-02 10:00:01.500Z",1000.0,1002.0',
            '"exness","XPTUSD","2024-01-02 10:59:59.900Z",1001.0,1004.0',
            '"exness","XPTUSD","2024-01-02 11:30:00.000Z",1003.0,1003.5']
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Exness_XPTUSD_2024_01.csv", "\n".join(rows) + "\n")
    h = spread_refresh.exness_hours(buf.getvalue())
    assert list(h.index) == [pd.Timestamp("2024-01-02 11:00", tz="UTC"), pd.Timestamp("2024-01-02 12:00", tz="UTC")]
    assert h["open"].tolist() == [2.0, 0.5] and h["close"].tolist() == [3.0, 0.5]
    assert h["mid"].tolist() == [1002.5, 1003.25] and h["ticks"].tolist() == [2.0, 1.0]
    assert spread_refresh.exness_hours(b"").empty                          # a month the archive does not have


def _exness_zip(symbol: str, ticks: list[tuple[str, float, float]]) -> bytes:
    rows = ['"Exness","Symbol","Timestamp","Bid","Ask"'] + [f'"exness","{symbol}","{t}",{b},{a}' for t, b, a in ticks]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"Exness_{symbol}.csv", "\n".join(rows) + "\n")
    return buf.getvalue()


def test_a_minute_of_exness_ticks_is_the_first_highest_lowest_and_last_of_their_mids(tmp_path, monkeypatch):
    ticks = [("2024-01-02 10:00:01.500Z", 80.00, 80.02), ("2024-01-02 10:00:30.000Z", 80.10, 80.12),
             ("2024-01-02 10:00:59.900Z", 79.90, 79.94), ("2024-01-02 10:02:10.000Z", 80.20, 80.20)]
    m = spread_refresh.exness_minutes(_exness_zip("USOIL", ticks[::-1]))            # the archive's order is not time's
    assert list(m.index) == [pd.Timestamp("2024-01-02 10:01", tz="UTC"), pd.Timestamp("2024-01-02 10:03", tz="UTC")]
    assert m.loc[m.index[0]].round(6).tolist() == [80.01, 80.11, 79.92, 79.92]        # no tick in 10:01-10:02: no bar
    assert m.loc[m.index[1]].tolist() == [80.2] * 4
    # a quote's months kept once built, again when the month's zip is newer
    monkeypatch.setattr(spread_refresh, "EXNESS_DIR", tmp_path)
    d = tmp_path / "USOIL"
    d.mkdir()
    (d / "2024-01.zip").write_bytes(_exness_zip("USOIL", ticks))
    assert spread_refresh.exness_mid_minutes("WTI/USD").equals(m.sort_index())
    kept = d / "minutes" / "2024-01.parquet"
    assert kept.exists()
    (d / "2024-01.zip").write_bytes(_exness_zip("USOIL", ticks + [("2024-01-02 10:05:00.000Z", 81.0, 81.0)]))
    import os
    os.utime(d / "2024-01.zip", (kept.stat().st_mtime + 5, kept.stat().st_mtime + 5))
    assert len(spread_refresh.exness_mid_minutes("WTI/USD")) == 3


def test_a_bar_takes_its_first_hours_opening_spread_its_last_hours_closing_one_and_their_median_inside():
    hours = pd.date_range("2024-01-01 01:00", periods=6, freq="h", tz="UTC")
    quote_spreads(["td:XPT/USD"], hours, opening=[1.0, 2, 3, 4, 5, 6], closing=[10.0, 20, 30, 40, 50, 60])
    closes = pd.DatetimeIndex(["2024-01-01 04:00", "2024-01-01 08:00", "2024-01-01 12:00"], tz="UTC")
    at_open, inside, at_close = spreads.per_bar("td:XPT/USD", closes, "4h")
    assert at_open.tolist() == [1.0, 5.0, 60.0]                # the last bar has no hour: the last spread quoted
    assert at_close.tolist() == [40.0, 60.0, 60.0]
    assert inside.tolist() == [13.75, 30.25, 60.0]             # median of (1+10)/2 .. (4+40)/2; of 27.5 and 33
    # before the first hour on record: what the first year quoted at the same hours of the day, at the median; an
    # hour of the day it never quoted takes the year's median
    before = pd.DatetimeIndex(["2023-12-30 04:00"], tz="UTC")
    assert [x[0] for x in spreads.per_bar("td:XPT/USD", before, "4h")] == [1.0, 13.75, 40.0]
    night = pd.DatetimeIndex(["2023-12-30 20:00"], tz="UTC")
    assert [x[0] for x in spreads.per_bar("td:XPT/USD", night, "4h")] == [3.5, 19.25, 35.0]


def test_a_spot_quote_without_its_brokers_spreads_is_not_filled_for_nothing():
    p = make_panel(ids=("td:XPT/USD",))
    with pytest.raises(FileNotFoundError, match="spread_refresh"):
        bt.run(p, pd.DataFrame(0.5, index=p.index, columns=p.ids))


@pytest.mark.parametrize("book", [False, True])
def test_each_fill_pays_half_the_spread_quoted_at_its_bars_open_as_an_account_in_cash_and_units(book):
    p = make_panel(ids=("td:XPT/USD", "td:XPD/USD", "td:EUR/USD"), seed=4)
    hours = _hours_of(p)
    rng = np.random.default_rng(9)
    opening = {i: rng.uniform(0.5, 4.0, len(hours)) for i in p.ids[:2]}
    for i in p.ids[:2]:
        quote_spreads([i], hours, opening=opening[i], closing=rng.uniform(0.5, 4.0, len(hours)))
    T = _random_target(p, values=(0.0, 0.15, 0.3))
    res = bt.run(p, T, fill="next_open", book=book)
    first_hour = 24 * np.arange(len(p.index))                  # bar k's first hour
    rate = np.column_stack([0.5 * opening[i][first_hour] / p.open[i].to_numpy() for i in p.ids[:2]]
                           + [np.full(len(p.index), FX)])
    ref = _units_account(p.open.to_numpy(), p.close.to_numpy(), T.to_numpy(), book, rate=rate)
    assert np.allclose(res.returns.to_numpy(), ref, atol=1e-12)
    assert (res.cost > 0).any()


def test_a_trade_nets_the_spreads_at_its_entrys_and_its_exits_opens():
    p = make_panel(ids=("td:XPT/USD",), seed=2)
    hours = _hours_of(p)
    opening = np.random.default_rng(3).uniform(1.0, 3.0, len(hours))
    quote_spreads(p.ids, hours, opening=opening, closing=5.0)
    T = pd.DataFrame(0.0, index=p.index, columns=p.ids)
    T.iloc[10:20] = 0.5                                       # decided at bar 10's close: in at 11's open, out at 21's
    t = ledger(p, bt.run(p, T, fill="next_open")).iloc[0]
    o = p.open["td:XPT/USD"]
    paid = 0.5 * opening[24 * 11] / o.iloc[11] + 0.5 * opening[24 * 21] / o.iloc[21]
    assert t["net_return"] == pytest.approx(t["gross_return"] - paid, abs=1e-15)


def test_a_stop_filled_inside_a_bar_pays_the_median_spread_its_hours_quoted():
    p = make_panel(ids=("td:XPT/USD",), seed=5)
    hours = _hours_of(p)
    rng = np.random.default_rng(6)
    opening, closing = rng.uniform(1.0, 3.0, len(hours)), rng.uniform(1.0, 3.0, len(hours))
    quote_spreads(p.ids, hours, opening=opening, closing=closing)
    T = pd.DataFrame(0.5, index=p.index, columns=p.ids)
    res = bt.run(p, T, fill="next_open", exits=bt.Exits(stop=0.01))
    stopped = np.flatnonzero(res.exits["td:XPT/USD"].notna().to_numpy())
    assert len(stopped)
    for k in stopped[:5]:
        inside = np.median((opening[24 * k:24 * k + 24] + closing[24 * k:24 * k + 24]) / 2)
        filled = abs(res.weights["td:XPT/USD"].iloc[k] - res.weights["td:XPT/USD"].iloc[k - 1]) * (
            0.5 * opening[24 * k] / p.open["td:XPT/USD"].iloc[k])
        assert res.cost.iloc[k] == pytest.approx(filled + res.exposure.iloc[k] * 0.5 * inside
                                                 / p.close["td:XPT/USD"].iloc[k], rel=1e-12)


def test_a_fill_at_the_close_pays_the_spread_quoted_at_that_close():
    p = make_panel(ids=("td:XPT/USD",), seed=8)
    hours = _hours_of(p)
    closing = np.random.default_rng(1).uniform(1.0, 3.0, len(hours))
    quote_spreads(p.ids, hours, opening=9.0, closing=closing)
    T = pd.DataFrame(0.0, index=p.index, columns=p.ids)
    T.iloc[30:] = 0.5                                         # decided at bar 30's close, filled at bar 31's close
    res = bt.run(p, T, fill="next_close")
    assert res.cost.iloc[32] == pytest.approx(0.5 * 0.5 * closing[24 * 31 + 23] / p.close["td:XPT/USD"].iloc[31])


def test_a_lists_figures_show_a_quotes_median_cost_of_its_last_year(monkeypatch):
    p = make_panel(ids=("td:XPT/USD",), n=800, seed=1)
    hours = pd.date_range(p.index[0] - pd.Timedelta(hours=23), p.index[-1], freq="h")
    quote_spreads(p.ids, hours, opening=2.0)
    monkeypatch.setattr(costs, "load_panel", lambda ids, tf, fields: p)
    last = p.index >= p.index[-1] - pd.Timedelta(days=365)
    first_hour = 24 * np.arange(len(p.index))
    want = np.median(0.5 * 2.0 / p.open["td:XPT/USD"].to_numpy()[last]) * 1e4
    assert first_hour[-1] < len(hours) and costs.typical_bps("td:XPT/USD") == pytest.approx(want)
    assert costs.typical_bps("td:EUR/USD") == pytest.approx(FX * 1e4)


def test_dukascopys_spreads_before_exness_are_put_on_exness_level_over_its_first_year(monkeypatch):
    ex_hours = pd.date_range("2015-08-03 01:00", periods=24 * 400, freq="h", tz="UTC")
    du_hours = pd.date_range(pd.Timestamp("2014-01-01 01:00", tz="UTC"), ex_hours[24 * 30], freq="h")
    exness = pd.DataFrame({"open": 0.4, "close": 0.5, "mid": 1100.0, "ticks": 10.0}, index=ex_hours)
    duka = pd.DataFrame({"open": 1.0, "close": 2.0, "mid": 1100.0, "ticks": np.nan}, index=du_hours)
    duka.iloc[5:8, :2] = 0.0                            # the ask's file served as the bid's: no quote's other side
    monkeypatch.setattr(spread_refresh, "exness_on_disk", lambda s: exness)
    monkeypatch.setattr(spread_refresh, "dukascopy_spreads", lambda s: duka)
    rep = spread_refresh.build(["XAU/USD"])
    kept = pd.read_parquet(spreads.spread_path("td:XAU/USD"))
    assert rep["XAU/USD"]["dukascopy_scale"] == pytest.approx(0.25)          # 0.5 / 2.0 over the hours of both
    early = kept[kept.index < ex_hours[0]]
    assert (early["source"] == "dukascopy").all() and len(early) == (du_hours < ex_hours[0]).sum() - 3
    assert np.allclose(early["open"], 0.25) and np.allclose(early["close"], 0.5)
    assert (kept.loc[ex_hours, "source"] == "exness").all() and np.allclose(kept.loc[ex_hours, "close"], 0.5)
