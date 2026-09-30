import pkgutil

import numpy as np
import pandas as pd
import pytest

import strategies
from strategy_lab.data.bars import Panel
from strategy_lab.strategy import load, rule, seats
from tests.conftest import make_panel

BARS = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")


def _frame(**columns):
    return pd.DataFrame({k: np.asarray(v) for k, v in columns.items()}, index=BARS)


def _steps(*spans):
    """A per-bar series from (value, first bar, last bar) spans, zero elsewhere."""
    out = np.zeros(len(BARS))
    for value, a, b in spans:
        out[a:b + 1] = value
    return out


def test_a_leaver_keeps_its_seat_until_its_trade_ends_and_the_newcomer_waits_for_it():
    listed = _frame(A=_steps((1, 0, 29)), B=_steps((1, 0, 9)), C=_steps((1, 10, 29))).astype(bool)
    wanted = _frame(A=_steps((1, 0, 29)), B=_steps((1, 5, 14)), C=_steps((1, 8, 29)))
    s = seats(listed, wanted)
    assert s["A"].all()
    assert s["B"].iloc[:15].all() and not s["B"].iloc[15:].any()        # kept through its trade, gone when it ends
    assert not s["C"].iloc[:15].any() and s["C"].iloc[15:].all()        # waits for the seat, then takes it
    assert (s.sum(axis=1) <= listed.sum(axis=1)).all()


def test_a_kept_trade_that_turns_to_the_other_side_gives_the_seat_up_without_opening_it():
    listed = _frame(A=_steps((1, 0, 29)), B=_steps((1, 0, 9)), C=_steps((1, 10, 29))).astype(bool)
    wanted = _frame(A=_steps((1, 0, 29)), B=_steps((1, 5, 12), (-1, 13, 20)), C=_steps((1, 0, 29)))
    s = seats(listed, wanted)
    assert s["B"].iloc[10:13].all() and not s["B"].iloc[13:].any()
    assert s["C"].iloc[13:].all()


def test_a_name_back_in_the_universe_while_its_trade_is_kept_stays_after_the_trade():
    listed = _frame(A=_steps((1, 0, 29)), B=_steps((1, 0, 9), (1, 20, 29)), C=_steps((1, 10, 19))).astype(bool)
    wanted = _frame(A=_steps((1, 0, 29)), B=_steps((1, 5, 25)), C=_steps((1, 0, 29)))
    s = seats(listed, wanted)
    assert s["B"].all()                  # kept 10-19, a member again from 20, still one after its trade ends at 26
    assert not s["C"].any()              # listed only while B's kept trade held the seat


def test_a_rule_holding_an_exposure_follows_the_list_where_a_rule_of_trades_keeps_its_seat():
    panel = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), n=60)
    member = pd.DataFrame(True, index=panel.index, columns=panel.ids)
    member.iloc[30:, 0] = False             # AAA leaves the list at bar 30
    member.iloc[:30, 2] = False             # CCC joins it there

    def always(bars):                       # held on every bar, never closed by the rule
        return pd.Series(1.0, index=bars.index)

    held = rule(exposure=True)(always).target(panel, {}, member)
    kept = rule()(always).target(panel, {}, member)
    assert (held.iloc[30:, 0] == 0).all() and (held.iloc[30:, 2] == 0.5).all()      # sold at the re-pick, bought at once
    assert (kept.iloc[30:, 0] == 0.5).all() and (kept.iloc[30:, 2] == 0).all()      # the trade keeps the seat for ever
    assert np.allclose(held.sum(axis=1), 1.0) and np.allclose(kept.sum(axis=1), 1.0)


@rule(grid={"slots": [1, 2]})
def wants_its_volume(bars):                 # a rule holding what its instrument's `volume` says: a test's positions
    return bars.volume


def _on(n, first, last):
    out = np.zeros(n)
    out[first:last + 1] = 1.0
    return out


def _wanting(positions, liquidity=None, n=80):
    """A panel whose instruments' volumes are the positions `wants_its_volume` takes, their dollar volumes (their
    liquidity) given or make_panel's."""
    p = make_panel(ids=tuple(positions), n=n)
    for i, pos in positions.items():
        p.volume[i] = pos
        if liquidity is not None:
            p.dollar_volume[i] = liquidity[i]
    return p


def test_a_trade_keeps_its_slot_to_its_end_and_a_trade_that_finds_none_is_not_traded():
    n = 80
    p = _wanting({"td:AAA": _on(n, 10, 60), "td:BBB": _on(n, 20, 40), "td:CCC": _on(n, 30, 70),
                  "td:DDD": _on(n, 45, 55)})
    w = wants_its_volume.target(p, {"slots": 2})
    assert (w["td:AAA"].iloc[10:61] == 0.5).all()           # its share throughout, whoever comes and goes
    assert (w["td:BBB"].iloc[20:41] == 0.5).all()
    assert (w["td:CCC"] == 0).all()                         # both slots taken on its first bar: never traded
    assert (w["td:DDD"].iloc[45:56] == 0.5).all()           # BBB's slot, free again
    assert ((w != 0).sum(axis=1) <= 2).all()


def test_a_trade_its_stop_ends_gives_its_slot_back_at_once():
    n = 80
    p = _wanting({"td:AAA": _on(n, 10, 60), "td:BBB": _on(n, 30, 50)})
    p.low.iloc[20, 0] = p.open.iloc[20, 0] * 0.5                     # AAA's stop is hit inside bar 20
    stopped = rule(grid={"slots": [1], "stop": [0.2]})(lambda bars: bars.volume)
    w = stopped.target(p, {"slots": 1, "stop": 0.2})
    assert (w["td:AAA"].iloc[10:20] == 1.0).all() and (w["td:AAA"].iloc[20:] == 0.0).all()
    assert (w["td:BBB"].iloc[30:51] == 1.0).all()                    # the slot AAA's trade gave back
    assert wants_its_volume.target(p, {"slots": 1})["td:BBB"].eq(0.0).all()     # without the stop it had none


def test_of_trades_starting_on_one_bar_the_more_liquid_instrument_takes_the_last_slot():
    n = 80
    wants = {"td:AAA": _on(n, 50, 60), "td:BBB": _on(n, 50, 65)}
    for liquid, other in (("td:AAA", "td:BBB"), ("td:BBB", "td:AAA")):
        p = _wanting(wants, liquidity={liquid: 5e7, other: 1e7}, n=n)
        w = wants_its_volume.target(p, {"slots": 1})
        assert (w[liquid] == wants[liquid]).all() and (w[other] == 0).all()


def test_currency_pairs_whose_quotes_carry_no_volume_take_slots_by_their_markets_turnover():
    n = 80
    wants = {"td:NZD/USD": _on(n, 50, 60), "td:EUR/USD": _on(n, 50, 60)}     # NZD/USD first in the panel
    p = _wanting(wants, liquidity={i: 0.0 for i in wants}, n=n)
    w = wants_its_volume.target(p, {"slots": 1})
    assert (w["td:EUR/USD"] == wants["td:EUR/USD"]).all() and (w["td:NZD/USD"] == 0).all()


def _upside_down(p: Panel) -> Panel:
    """The panel's prices turned upside down around a level above them all: where a price rises, here it falls."""
    k = 2.0 * float(p.high.max().max())
    return Panel(p.timeframe, p.instruments, open=k - p.open, high=k - p.low, low=k - p.high, close=k - p.close,
                 volume=p.volume, dollar_volume=p.dollar_volume)


# rules whose indicators turn with the prices (moving averages, channels, bands, RSI, IBS): their short side on the
# prices is their long side on the prices turned upside down
MIRRORED = [("sma_cross", {"fast": 10, "slow": 100}), ("breakout_trail", {"n_in": 20, "n_out": 50}),
            ("ibs_reversion", {"entry": 0.2})]


@pytest.mark.parametrize("name, params", MIRRORED, ids=[m[0] for m in MIRRORED])
def test_a_rules_short_side_is_its_long_side_on_the_prices_turned_upside_down(name, params):
    s = load(name)
    p = make_panel(ids=("td:AAA",), n=900, seed=4)
    bars, turned = p.one("td:AAA"), _upside_down(p).one("td:AAA")
    both = s.fn(bars, **params, long_only=False)
    long, short = s.fn(bars, **params, long_only=True), s.fn(turned, **params, long_only=True)
    one = (long != 0) != (short != 0)
    assert (both[one] == (long - short)[one]).all()                                 # one side holding: that side
    assert ((both > 0) <= (long != 0)).all() and ((both < 0) <= (short != 0)).all()
    if name == "breakout_trail":            # stop and reverse: while both sides hold, the latest entry's side
        assert (both[(long != 0) & (short != 0)] != 0).all() and ((long != 0) & (short != 0)).any()
    else:                                   # the two sides exclude each other
        assert (both == long - short).all()
    assert (both > 0).any() and (both < 0).any()                # both sides trade on this walk


def test_every_strategy_file_holds_one_strategy_named_after_it():
    names = [m.name for m in pkgutil.iter_modules(strategies.__path__)]
    assert names and all(load(n).name == n for n in names)


def test_positions_kept_for_one_membership_give_another_membership_of_the_same_bars_its_own_positions():
    from strategies.ibs_ml_filter import ibs_ml_filter
    from strategies.rsi2_connors import rsi2_connors
    from tests.conftest import make_panel
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC", "td:DDD"), n=700, seed=5)
    one = pd.DataFrame(True, index=p.index, columns=p.ids)
    one.iloc[:, 3] = False
    other = pd.DataFrame(True, index=p.index, columns=p.ids)
    other.iloc[300:, 0] = False
    for s in (rsi2_connors, ibs_ml_filter):
        signals: dict = {}
        for cfg in s.configs()[:4]:
            s.target(p, cfg, one, signals=signals)                 # kept while evaluating one list
            pd.testing.assert_frame_equal(s.target(p, cfg, other, signals=signals), s.target(p, cfg, other))


def test_positions_kept_for_other_bars_are_refused():
    from strategies.ibs import ibs
    from tests.conftest import make_panel
    signals: dict = {}
    ibs.target(make_panel(n=300, seed=1), ibs.configs()[0], signals=signals)
    try:
        ibs.target(make_panel(n=301, seed=1), ibs.configs()[0], signals=signals)
    except ValueError as e:
        assert "other bars" in str(e)
    else:
        raise AssertionError("positions kept for other bars were used")


def _counted_rsi2(calls: list):
    """RSI(2) with its calls counted; its code is this file's, a version of its own."""
    import dataclasses

    from strategies.rsi2_connors import rsi2_connors

    def counted(bars, **sp):
        calls.append(float(bars["close"].iloc[-1]))       # which instrument's bars: its last close
        return rsi2_connors.fn(bars, **sp)
    return dataclasses.replace(rsi2_connors, fn=counted)


def test_positions_kept_on_disk_are_read_back_by_a_later_evaluation_as_the_rule_computed_them():
    from strategy_lab.strategy import fill_signals
    calls: list = []
    s = _counted_rsi2(calls)
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), n=500, seed=7)
    configs = s.configs()
    computed: dict = {}
    fill_signals(s, p, configs, computed)
    signal_configs = {repr(sorted(s.signal_params(c).items())) for c in configs}
    assert len(calls) == len(p.ids) * len(signal_configs)
    read: dict = {}
    fill_signals(s, p, configs, read)                  # another evaluation: another process of a batch, another list
    assert len(calls) == len(p.ids) * len(signal_configs)       # nothing worked out again
    keys = [k for k in computed if k != "index"]
    assert keys and sorted(keys) == sorted(k for k in read if k != "index")
    for k in keys:
        assert computed[k].dtype == read[k].dtype and np.array_equal(computed[k], read[k]), k


def test_a_bar_written_anew_is_other_bars_and_its_rule_computes_them_again():
    from strategy_lab.data.bars import FIELDS
    from strategy_lab.strategy import fill_signals
    calls: list = []
    s = _counted_rsi2(calls)
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), n=500, seed=7)
    fill_signals(s, p, s.configs(), {})
    before = len(calls)
    rewritten = Panel(p.timeframe, dict(p.instruments), **{f: getattr(p, f).copy() for f in FIELDS})
    rewritten.close.iloc[250, 1] *= 1.01              # one close of BBB corrected by its vendor
    fresh: dict = {}
    fill_signals(s, rewritten, s.configs(), fresh)
    signal_configs = {repr(sorted(s.signal_params(c).items())) for c in s.configs()}
    assert len(calls) - before == len(signal_configs)           # BBB's alone, once per signal configuration
    alone: dict = {}
    for sig in signal_configs:                         # and its positions are the ones its new bars give
        sp = s.signal_params(next(c for c in s.configs() if repr(sorted(s.signal_params(c).items())) == sig))
        alone[sig] = s.fn(rewritten.one("td:BBB"), **sp)
    from strategy_lab.strategy import _aligned, _packed
    for sig, out in alone.items():
        assert np.array_equal(fresh[(sig, "td:BBB")], _packed(_aligned(out, rewritten.index)))


def _series_panel(closes: dict, timeframe="1d", index=None) -> Panel:
    """A panel of the given close paths (open at the close before, a 0.5% range around both)."""
    from strategy_lab.data.instruments import parse
    idx = index if index is not None else pd.date_range("2020-01-01", periods=len(next(iter(closes.values()))),
                                                       freq="D", tz="UTC")
    c = pd.DataFrame(closes, index=idx, dtype=float)
    o = c.shift(1).fillna(c.iloc[0])
    return Panel(timeframe, {i: parse(i) for i in closes}, open=o, high=np.maximum(o, c) * 1.005,
                 low=np.minimum(o, c) * 0.995, close=c, volume=c * 0 + 1e6, dollar_volume=c * 1e6)


def test_a_period_opens_at_the_first_close_at_or_after_its_turn():
    from strategy_lab.strategy import period_starts
    hours = pd.date_range("2024-01-31 21:00", "2024-02-01 03:00", freq="h", tz="UTC")     # a coin's hourly closes
    assert list(hours[period_starts(hours, "M")]) == [hours[0], pd.Timestamp("2024-02-01 00:00", tz="UTC")]
    sessions = pd.DatetimeIndex(["2024-01-30 21:00", "2024-01-31 21:00", "2024-02-01 21:00", "2024-02-02 21:00",
                                 "2024-02-05 21:00"], tz="UTC")                            # a stock's daily closes
    assert list(period_starts(sessions, "M")) == [True, False, True, False, False]
    assert list(period_starts(sessions, "W")) == [True, False, False, False, True]         # 2024-02-05 is a Monday
    assert list(period_starts(sessions, "3D")) == [True, False, False, True, True]         # days 19752-4, 19755-7, ...


def _stock_hours(start: str, end: str) -> pd.DatetimeIndex:
    """A US listing's hourly closes (10:30 .. 15:30 and the 16:00 close, New York) on the NYSE's full sessions."""
    from strategy_lab.data.calendars import nyse_sessions
    s = nyse_sessions(pd.Timestamp(start), pd.Timestamp(end))
    full = s[(s["market_close"] - s["market_open"]) == pd.Timedelta(hours=6, minutes=30)]
    return pd.DatetimeIndex([o + pd.Timedelta(minutes=60 * h + 60) for o in full["market_open"] for h in range(6)]
                            + list(full["market_close"])).sort_values()


def test_a_span_in_months_is_the_same_time_on_every_timeframe():
    from strategy_lab.strategy import bars_in
    coin_hours = pd.date_range("2021-01-01", periods=24 * 800, freq="h", tz="UTC")
    coin_days = pd.date_range("2021-01-02", periods=800, freq="D", tz="UTC")
    stock_hours = _stock_hours("2021-01-01", "2023-06-30")
    stock_days = pd.DatetimeIndex(sorted({t.normalize() + pd.Timedelta(hours=21) for t in stock_hours}))
    for ids, idx, tf in ((("perp:BTCUSDT",), coin_hours, "1h"), (("perp:BTCUSDT",), coin_days, "1d"),
                         (("td:AAPL",), stock_hours, "1h"), (("td:AAPL",), stock_days, "1d")):
        p = _series_panel({ids[0]: np.ones(len(idx))}, tf, idx)
        n = bars_in(p, months=12)
        assert bars_in(p.one(ids[0]), months=12) == n                     # a rule's bars count as their panel's
        spans = (idx[n:] - idx[:-n]).days
        assert 359 <= np.median(spans) <= 368, (ids, tf, n, np.median(spans))


def test_a_monthly_rule_ignores_the_moves_inside_a_month():
    days = pd.date_range("2021-01-01", "2023-12-31", freq="D", tz="UTC")
    close = pd.Series(100 * np.exp(0.001 * np.arange(len(days))), index=days)
    dip = (days >= "2023-06-08") & (days <= "2023-06-20")
    close[dip] *= 0.6                                    # a crash inside June, over before its end
    p = _series_panel({"perp:BTCUSDT": close.to_numpy()}, "1d", days)
    faber = load("gtaa_faber").fn(p.one("perp:BTCUSDT"), months=10)
    assert (faber["2022-01-01":] == 1.0).all()           # Faber looks at the month's end only: held through the dip
    daily = p.close["perp:BTCUSDT"] > p.close["perp:BTCUSDT"].rolling(210).mean()
    assert not daily["2023-06-08":"2023-06-20"].any()    # a daily 210-day average would have sold it
    from strategy_lab.strategy import period_starts
    for name, params in (("tsmom", {"months": 3, "long_only": False}), ("vol_managed", {"vol_months": 1, "target_vol": 0.1}),
                         ("gtaa_faber", {"months": 10})):
        pos = load(name).fn(p.one("perp:BTCUSDT"), **params)
        moved = pos.ne(pos.shift()).to_numpy()[1:]
        assert not (moved & ~period_starts(days, "M")[1:]).any(), name     # it changes at a month's turn only


def test_dual_momentum_holds_an_asset_only_while_its_year_beats_treasury_bills():
    # the tests' T-bill rate is 4% a year (conftest.CASH_RATE)
    days = pd.date_range("2021-01-01", periods=900, freq="D", tz="UTC")
    t = np.arange(len(days)) / 365.0
    p = _series_panel({"perp:AAAUSDT": 100 * np.exp(np.log(1.03) * t), "perp:BBBUSDT": 100 * np.exp(np.log(1.06) * t)},
                      "1d", days)
    w = load("dual_momentum").target(p, {"months": 12, "top_k": 2, "rebalance": "M"})
    late = w.loc["2022-03-01":]
    assert (late["perp:BBBUSDT"] == 0.5).all() and (late["perp:AAAUSDT"] == 0.0).all()


def test_connors_rsi2_sells_on_the_first_close_above_the_five_bar_average():
    days = pd.date_range("2021-01-01", periods=400, freq="D", tz="UTC")
    close = 100 * np.exp(0.002 * np.arange(len(days)))                       # well above its 200-day average
    close[300:303] *= [0.97, 0.94, 0.91]                                     # three closes down: RSI(2) near 0
    close[303:] *= 0.91 * np.exp(0.01 * np.arange(1, len(days) - 302))       # then up again
    p = _series_panel({"td:AAA": close}, "1d", days)
    bars = p.one("td:AAA")
    pos = load("rsi2_connors").fn(bars, rsi_entry=5)
    first_above = next(k for k in range(303, 400) if bars.close.iloc[k] > bars.close.iloc[k - 4:k + 1].mean())
    assert pos.iloc[301] == 1.0 and (pos.iloc[301:first_above] == 1.0).all() and pos.iloc[first_above] == 0.0


def _priced(n: int, **closes) -> Panel:
    """A daily panel of US listings from their closes: each bar opens at the close before it, its range the two
    widened a little."""
    from strategy_lab.data.instruments import parse
    idx = pd.date_range("2024-01-01 21:00", periods=n, freq="D", tz="UTC")
    close = pd.DataFrame({f"td:{k}": np.asarray(v, dtype=float) for k, v in closes.items()}, index=idx)
    opn = close.shift(1).fillna(close)
    vol = pd.DataFrame(1e6, index=idx, columns=close.columns)
    return Panel("1d", {i: parse(i) for i in close.columns}, opn, np.maximum(opn, close) * 1.001,
                 np.minimum(opn, close) * 0.999, close, vol, vol * close)


def test_a_name_seated_in_the_middle_of_its_rules_trade_leaves_with_the_trade_at_its_stop():
    from strategy_lab import evaluate as ev
    n = 40
    rise = np.r_[np.linspace(100, 130, 21), np.linspace(130, 80, n - 21)]
    p = _priced(n, AAA=np.full(n, 50.0), BBB=rise)
    member = pd.DataFrame(True, index=p.index, columns=p.ids)
    member.iloc[:20, 1] = False                               # BBB joins the list at bar 20, its rule long since 5
    long_from_5 = rule(grid={"stop": [0.1]})(lambda bars: pd.Series((np.arange(len(bars)) >= 5).astype(float),
                                                                     index=bars.index))
    res = ev._run(long_from_5, p, {"stop": 0.1}, member, "next_open")
    # the rule's trade: in at bar 6's open, out where a low first reaches 10% under that; the seat's position, bought
    # at bar 21's open at 130, leaves with it there at that level, not 10% under its own fill
    level = 0.9 * p.open["td:BBB"].iloc[6]
    k = 21 + int(np.flatnonzero(p.low["td:BBB"].iloc[21:].to_numpy() <= level)[0])
    exits = res.exits["td:BBB"].dropna()
    assert list(exits.index) == [p.index[k]] and exits.iloc[0] == pytest.approx(level)
    held = res.weights["td:BBB"]
    assert (held.iloc[21:k + 1] > 0).all() and (held.iloc[k + 1:] == 0).all() and (held.iloc[:21] == 0).all()
    target = long_from_5.target(p, {"stop": 0.1}, member)
    assert (target["td:BBB"].iloc[20:k] == 0.5).all() and (target["td:BBB"].iloc[k:] == 0).all()   # cut at the exit


def test_a_panel_of_positions_trades_a_new_one_and_leaves_those_held_as_they_are():
    from strategy_lab import evaluate as ev
    from strategy_lab.strategy import panel as panel_strategy
    n = 30
    p = _priced(n, AAA=np.r_[np.full(10, 100.0), np.full(n - 10, 120.0)], BBB=np.full(n, 100.0))

    def events(q):                                           # AAA from bar 5, BBB from bar 20, half the capital each
        w = pd.DataFrame(0.0, index=q.close.index, columns=q.close.columns)
        w.iloc[5:, 0] = 0.5
        w.iloc[20:, 1] = 0.5
        return w
    own = ev._run(panel_strategy(book=False)(events), p, {}, None, "next_open")
    book = ev._run(panel_strategy()(events), p, {}, None, "next_open")
    grown = 0.5 * 1.2 / (1 + 0.5 * 0.2)                      # AAA's weight at bar 21's open: half, up 20%
    assert own.weights["td:AAA"].iloc[21] == 0.5              # left as it is (its last fill)
    assert own.turnover.iloc[21] == pytest.approx(1 - grown, rel=1e-9)       # BBB bought with the cash there is
    assert book.turnover.iloc[21] == pytest.approx(grown - 0.5 + 0.5, rel=1e-9)   # AAA sold back to half, BBB half


def test_an_instrument_that_stopped_trading_leaves_a_fixed_list_and_its_share_goes_to_the_others():
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), n=60)
    for f in ("open", "high", "low", "close", "volume", "dollar_volume"):
        getattr(p, f).iloc[20:, 2] = np.nan                    # CCC's data ends at bar 19, weeks before the others'

    def always(bars):
        return pd.Series(1.0, index=bars.index)
    w = rule()(always).target(p, {})
    assert np.allclose(w.iloc[:20].to_numpy(), 1 / 3)
    assert np.allclose(w.iloc[25:, :2].to_numpy(), 0.5) and (w.iloc[25:, 2] == 0).all()


def test_of_names_joining_on_one_bar_the_more_liquid_takes_the_first_free_seat():
    listed = _frame(A=_steps((1, 0, 29)), B=_steps((1, 0, 9)), C=_steps((1, 10, 29)), D=_steps((1, 10, 29))).astype(bool)
    wanted = _frame(A=_steps((1, 0, 29)), B=_steps((1, 5, 14)), C=_steps((1, 0, 29)), D=_steps((1, 0, 29)))
    liquid = _frame(A=np.full(30, 5.0), B=np.full(30, 5.0), C=np.full(30, 1.0), D=np.full(30, 3.0))
    s = seats(listed, wanted, liquid)                        # one seat free at bar 10 (B's kept trade holds the other)
    assert s["D"].iloc[10:].all() and not s["C"].iloc[:15].any() and s["C"].iloc[15:].all()
    s = seats(listed, wanted)                                # no liquidity given: the columns' order
    assert s["C"].iloc[10:].all() and not s["D"].iloc[:15].any()


def test_positions_kept_on_disk_are_keyed_by_the_model_library_as_by_the_code(monkeypatch):
    import strategies.ml_feature_search as mfs
    from strategy_lab import strategy
    path = mfs.__file__
    strategy._code_of.cache_clear()
    monkeypatch.setattr(strategy, "_version_of", lambda package: "4.7.0")
    before = strategy._code_of(path)
    strategy._code_of.cache_clear()
    monkeypatch.setattr(strategy, "_version_of", lambda package: "4.8.0")
    assert strategy._code_of(path) != before
    strategy._code_of.cache_clear()


def test_a_member_that_stops_trading_in_its_trade_gives_its_seat_to_the_newcomer_at_once():
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), n=60)
    for f in ("open", "high", "low", "close", "volume", "dollar_volume"):
        getattr(p, f).iloc[20:, 1] = np.nan                    # BBB delisted after bar 19, in the middle of its trade
    member = pd.DataFrame(True, index=p.index, columns=p.ids)
    member.iloc[20:, 1] = False                                # the list gives its seat up...
    member.iloc[:20, 2] = False                                # ... to CCC

    def always(bars):                                          # a trade that never ends by the rule
        return pd.Series(1.0, index=bars.index)
    w = rule()(always).target(p, {}, member)
    assert np.allclose(w.iloc[:20, :2].to_numpy(), 0.5) and (w.iloc[20:, 1] == 0).all()
    assert np.allclose(w.iloc[20:, [0, 2]].to_numpy(), 0.5)    # the newcomer seated at once, not after a trade that never ends
