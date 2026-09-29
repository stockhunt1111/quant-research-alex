import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from strategy_lab import db
from strategy_lab import evaluate as ev
from strategy_lab.data.bars import FIELDS, Panel
from strategy_lab.universes import Universe
from strategies.donchian_breakout import donchian_breakout
from strategies.dual_momentum import dual_momentum
from strategies.ibs_reversion import ibs_reversion
from tests.conftest import make_panel, store_of, top_lists


def test_a_universe_that_holds_names_late_is_scored_from_then_as_if_its_bars_began_there(monkeypatch):
    # ibs_reversion decides each bar from that bar alone, so bars before the universe holds anything change nothing
    full = make_panel(n=900, seed=5)
    k = 250
    joined = full.index[k]
    cut = Panel(full.timeframe, dict(full.instruments), **{f: getattr(full, f).iloc[k:] for f in FIELDS})

    def held_from_k(p):
        return pd.DataFrame(np.repeat((p.index >= joined)[:, None], len(p.ids), axis=1), index=p.index, columns=p.ids)

    monkeypatch.setattr(ev, "resolve", lambda name, tf: Universe(name, list(full.ids), held_from_k))
    monkeypatch.setattr(ev, "load_panel", lambda ids, tf, start=None, end=None, **_: Panel(
        full.timeframe, {i: full.instruments[i] for i in ids}, **{f: getattr(full, f)[ids] for f in FIELDS}))
    late = ev.evaluate(ibs_reversion, "late", "1d", monte_carlo=False, robustness=False, save=False)

    monkeypatch.setattr(ev, "resolve", lambda name, tf: Universe(name, list(cut.ids)))
    monkeypatch.setattr(ev, "load_panel", lambda ids, tf, start=None, end=None: cut)
    trimmed = ev.evaluate(ibs_reversion, "cut", "1d", monte_carlo=False, robustness=False, save=False)

    assert late.oos["start"] == trimmed.oos["start"] > str(full.index[0].date())
    pd.testing.assert_series_equal(late.oos_daily, trimmed.oos_daily, check_names=False)
    assert late.oos["sharpe"] == trimmed.oos["sharpe"]


def test_a_list_runs_only_the_slot_counts_below_its_number_of_names(monkeypatch):
    p = make_panel(ids=tuple(f"td:S{k:02d}" for k in range(12)), n=500, seed=3)
    monkeypatch.setattr(ev, "load_panel", store_of(p))
    monkeypatch.setattr(ev, "stored_version", lambda ids, tf: 1)
    monkeypatch.setattr(ev, "resolve", top_lists(p))           # x_topN: the N most liquid of the twelve
    top3 = ev.evaluate(ibs_reversion, "x_top3", "1d", monte_carlo=False, robustness=False, save=False)
    top10 = ev.evaluate(ibs_reversion, "x_top10", "1d", monte_carlo=False, robustness=False, save=False)
    monkeypatch.setattr(ev, "resolve", lambda name, tf: Universe(name, list(p.ids)))
    whole = ev.evaluate(ibs_reversion, "all12", "1d", monte_carlo=False, robustness=False, save=False)
    assert [top3.grid["slots"], top10.grid["slots"], whole.grid["slots"]] == [[None], [None, 5], [None, 5, 10]]
    assert {json.loads(x)["slots"] for x in top10.folds["params"]} <= {None, 5}
    monkeypatch.setattr(ev, "resolve", top_lists(p))
    held = ev.evaluate(dual_momentum, "x_top3", "1d", monte_carlo=False, robustness=False, save=False)
    assert held.grid["top_k"] == [1, 2, 3]                         # holding four or five of three names: idle capital


def test_a_list_is_scored_from_the_first_day_it_holds_half_of_its_names(monkeypatch):
    base = make_panel(ids=("td:AAA", "td:BBB", "td:CCC", "td:DDD"), n=900, seed=6)
    k = 300
    early = {f: getattr(base, f).copy() for f in FIELDS}
    for f in FIELDS:                                   # a list of four that is one name until bar k
        early[f].iloc[:k, 1:] = np.nan
    grown = Panel(base.timeframe, dict(base.instruments), **early)
    cut = Panel(base.timeframe, dict(base.instruments), **{f: getattr(base, f).iloc[k:] for f in FIELDS})
    monkeypatch.setattr(ev, "resolve", lambda name, tf: Universe(name, list(base.ids)))

    monkeypatch.setattr(ev, "load_panel", lambda ids, tf, start=None, end=None: grown)
    late = ev.evaluate(ibs_reversion, "grown", "1d", monte_carlo=False, robustness=False, save=False)
    monkeypatch.setattr(ev, "load_panel", lambda ids, tf, start=None, end=None: cut)
    whole = ev.evaluate(ibs_reversion, "whole", "1d", monte_carlo=False, robustness=False, save=False)

    assert late.oos["start"] == whole.oos["start"] > str(base.index[0].date())
    pd.testing.assert_series_equal(late.oos_daily, whole.oos_daily, check_names=False)


def test_a_list_is_read_once_for_the_evaluations_on_it_and_again_when_its_stored_bars_change(monkeypatch):
    p = make_panel(n=300, seed=4)
    reads, version = [], [1]
    monkeypatch.setattr(ev, "resolve", lambda name, tf: Universe(name, list(p.ids)))
    monkeypatch.setattr(ev, "load_panel", lambda ids, tf, start=None, end=None: reads.append(1) or p)
    monkeypatch.setattr(ev, "stored_version", lambda ids, tf: version[0])
    first = ev._load("list", "1d", None, None)
    assert ev._load("list", "1d", None, None) is first and len(reads) == 1
    version[0] = 2                                    # a refresh wrote one of its instruments' bars
    assert ev._load("list", "1d", None, None) is not first and len(reads) == 2
    ev._load("other", "1d", None, None)
    assert len(ev._LOADED) == 1                       # only the last list is kept


def test_stored_version_moves_when_an_instruments_bars_are_written(tmp_path, monkeypatch):
    import os
    from strategy_lab.data import bars, store
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    p = make_panel(ids=("td:AAA",), n=10, seed=1)
    df = pd.DataFrame({f: getattr(p, f)["td:AAA"] for f in FIELDS})
    store.write_bars("td", "1d", "AAA", df)
    before = bars.stored_version(["td:AAA", "td:BBB"], "1d")          # a name with nothing stored counts nothing
    store.write_bars("td", "1d", "AAA", df)
    path = store.path("td", "1d", "AAA")
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))     # a later write, however quick
    assert bars.stored_version(["td:AAA"], "1d") > before > 0


def test_a_grid_that_keeps_only_the_backtests_it_needs_scores_exactly_as_one_that_keeps_all():
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), n=900, seed=7)
    configs = donchian_breakout.configs()
    all_runs, daily, rows, choices, oos, best = ev._walk_forward(donchian_breakout, p, None, "next_open", configs,
                                                                 "1d", keep_all=True)
    runs, daily2, rows2, choices2, oos2, best2 = ev._walk_forward(donchian_breakout, p, None, "next_open", configs,
                                                                  "1d", keep_all=False)
    assert len(all_runs) == len(configs) and set(runs) == set(choices) | {best} and len(runs) < len(configs)
    pd.testing.assert_frame_equal(daily, daily2)
    pd.testing.assert_series_equal(oos, oos2)
    assert (rows, choices, best) == (rows2, choices2, best2)
    for c, r in runs.items():
        pd.testing.assert_frame_equal(r.weights, all_runs[c].weights)
        pd.testing.assert_frame_equal(r.exits, all_runs[c].exits)


def test_a_result_saved_again_keeps_nothing_an_earlier_evaluation_had_that_this_one_has_not(tmp_path, monkeypatch):
    p = make_panel(ids=("td:AAA", "td:BBB"), n=900, seed=5)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.sqlite")
    monkeypatch.setattr(ev, "resolve", lambda name, tf: Universe(name, list(p.ids)))
    monkeypatch.setattr(ev, "load_panel", lambda ids, tf, start=None, end=None: p)
    ev.evaluate(donchian_breakout, "pair", "1d", monte_carlo=False, robustness=False)
    conn = db.connect()
    rid = db.result_id(conn, "donchian_breakout", "pair", "1d")
    assert db.windows(conn, rid)
    one = dataclasses.replace(donchian_breakout, grid={k: v[:1] for k, v in donchian_breakout.grid.items()})
    ev.evaluate(one, "pair", "1d", monte_carlo=False, robustness=False)        # one configuration: no windows chosen
    assert db.result_id(conn, "donchian_breakout", "pair", "1d") == rid and db.windows(conn, rid) == []
    conn.close()


def test_a_list_read_in_two_steps_holds_the_same_bars_and_seats_as_one_read_of_all_its_candidates(tmp_path, monkeypatch):
    from strategy_lab.data import bars, store
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    full = make_panel(ids=("td:AAA", "td:BBB", "td:CCC", "td:DDD"), n=300, seed=9)
    for k, i in enumerate(full.ids):                  # histories of different lengths
        df = pd.DataFrame({f: getattr(full, f)[i] for f in FIELDS}).iloc[40 * k:300 - 25 * k]
        if i == "td:DDD":                             # never liquid enough for the list, and alone on one bar
            df = df.assign(dollar_volume=df["dollar_volume"] * 1e-6)
            df.index = df.index.where(df.index != df.index[100], df.index[100] + pd.Timedelta(hours=1))
        store.write_bars("td", "1d", i.split(":")[1], df)

    def top_two(p):                                   # seats from the close and the dollar volume only
        rank = p.dollar_volume.rank(axis=1, ascending=False)
        return (rank <= 2) & (p.dollar_volume > 1e4) & p.close.notna() & (p.close.index >= p.close.index[60])[:, None]

    monkeypatch.setattr(ev, "resolve", lambda name, tf: Universe(name, list(full.ids), top_two, size=2))
    uni, panel, member = ev._load("pair", "1d", None, None)
    everything = bars.load_panel(list(full.ids), "1d")
    seats = top_two(everything)
    ever = [i for i in everything.ids if seats[i].any()]
    assert panel.ids == ever and "td:DDD" not in ever
    assert len(panel.index) == len(everything.index) > len(bars.load_panel(ever, "1d").index)   # DDD's own bar stays
    for f in FIELDS:
        pd.testing.assert_frame_equal(getattr(panel, f), getattr(everything, f)[ever])
    live = seats[ever].copy()
    live.loc[live.index < ev.half_held_from(everything, seats[ever], 2)] = False
    pd.testing.assert_frame_equal(member, live)


def test_a_neighbouring_list_read_around_a_list_is_that_list_read_on_its_own(monkeypatch):
    p = make_panel(ids=tuple(f"td:S{k:02d}" for k in range(9)), n=500, seed=21)
    for k, scale in enumerate([10, 10, 10, 3, 3]):          # the three always the most liquid, two next: Top-5 holds more
        p.dollar_volume.iloc[:, k] *= scale
    reads, load = [], store_of(p)
    monkeypatch.setattr(ev, "load_panel", lambda ids, *a, **k: reads.append((list(ids), k.get("fields", FIELDS)))
                        or load(ids, *a, **k))
    monkeypatch.setattr(ev, "resolve", top_lists(p))
    monkeypatch.setattr(ev, "stored_version", lambda ids, tf: 1)
    _, main, _ = ev._load("demo_top3", "1d", None, None)
    reads.clear()
    _, around, member_around = ev._load("demo_top5", "1d", None, None, keep=False, base=main)
    assert reads and all(fields == FIELDS and not set(ids) & set(main.ids) for ids, fields in reads)
    assert [k[0] for k in ev._LOADED] == ["demo_top3"]         # the list around it is still the one kept
    ev._LOADED.clear()
    ev._SEATING.clear()
    _, alone, member_alone = ev._load("demo_top5", "1d", None, None)
    for f in FIELDS:
        pd.testing.assert_frame_equal(getattr(around, f), getattr(alone, f))
    pd.testing.assert_frame_equal(member_around, member_alone)


def test_a_ranked_list_holds_on_its_intraday_bars_the_names_its_daily_bars_rank_first(tmp_path, monkeypatch):
    from strategy_lab.data import store
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    days = make_panel(ids=("td:SPY",), n=200, seed=31).index
    # the daily bars rank SPY and QQQ first; the hourly bars, which miss the auctions, would rank IWM first
    for sym, daily_dollars, hourly_dollars in (("SPY", 3e9, 1e9), ("QQQ", 2e9, 2e9), ("IWM", 1e9, 5e9)):
        for tf, idx, dollars in (("1d", days, daily_dollars), ("1h", days - pd.Timedelta(hours=6), hourly_dollars)):
            store.write_bars("td", tf, sym, pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                                                          "volume": dollars / 100, "dollar_volume": dollars},
                                                         index=idx))
    _, panel, member = ev._load("etf_top2", "1h", None, None)
    assert member.iloc[-1].to_dict() == {"td:QQQ": True, "td:SPY": True} and set(panel.ids) == {"td:QQQ", "td:SPY"}


def test_a_window_nothing_could_be_chosen_for_holds_nothing_and_the_first_choice_pays_for_its_positions():
    from strategy_lab.config import COSTS
    from strategy_lab.strategy import rule
    rate = (COSTS["us_equity"].commission_bps + COSTS["us_equity"].half_spread_bps) / 1e4
    p = make_panel(n=900, seed=7)
    late = rule(grid={"k": [1, 2]})(lambda bars, k: pd.Series((np.arange(len(bars)) >= 420).astype(float),
                                                               index=bars.index))
    runs, daily, rows, choices, oos, _ = ev._walk_forward(late, p, None, "next_open", late.configs(), "1d")
    assert choices[:2] == [None, None] and choices[2] is not None
    assert all(r["config"] is None and json.loads(r["params"]) == {} for r in rows[:2])
    for r, c in zip(rows, choices):
        days = (oos.index >= pd.Timestamp(r["test_start"], tz="UTC")) & (oos.index < pd.Timestamp(r["test_end"], tz="UTC"))
        assert c is not None or (oos[days] == 0).all()
    # the first window with a choice takes its positions from nothing: that trade is paid on its first day
    day = pd.Timestamp(rows[2]["test_start"], tz="UTC")
    held = runs[choices[2]].weights.iloc[p.index.searchsorted(day)].abs().sum()
    assert held > 0
    assert oos[day] == pytest.approx((1 + daily[choices[2]][day]) * (1 - held * rate) - 1, rel=1e-12)


def test_the_records_trades_are_the_positions_its_windows_held_one_after_another():
    from strategy_lab.engine import backtest as bt
    p = make_panel(ids=("td:AAA",), n=900, seed=8)

    def held(a, b, side=1.0):
        t = pd.DataFrame(0.0, index=p.index, columns=p.ids)
        t.iloc[a:b] = side
        return t
    day = lambda k: (p.index[k] - pd.Timedelta(microseconds=1)).normalize().date()      # noqa: E731
    end = (p.index[-1] + pd.Timedelta(days=1)).date()
    rows = [{"test_start": day(0), "test_end": day(480)}, {"test_start": day(480), "test_end": end}]
    days = pd.date_range(pd.Timestamp(day(0), tz="UTC"), pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=1), freq="D")
    # long in both configurations across the switch: one trade, from the first one's entry to the second one's exit
    runs = {0: bt.run(p, held(10, 500)), 1: bt.run(p, held(450, 800))}
    t = ev._oos_held(p, runs, rows, [0, 1], days).trades
    assert len(t) == 1
    assert t["entry_time"].iloc[0] == p.index[11] and t["exit_time"].iloc[0] == p.index[801]
    assert t["entry_px"].iloc[0] == p.open["td:AAA"].iloc[11] and t["exit_px"].iloc[0] == p.open["td:AAA"].iloc[801]
    # flat in the second one: the trade ends at the switch, at the open of the window's first bar
    runs = {0: bt.run(p, held(10, 500)), 1: bt.run(p, held(600, 800, -1.0))}
    t = ev._oos_held(p, runs, rows, [0, 1], days).trades
    first = int(np.flatnonzero((p.index - pd.Timedelta(microseconds=1)).normalize().date >= day(480))[0])
    assert t["exit_time"].iloc[0] == p.index[first] and t["exit_px"].iloc[0] == p.open["td:AAA"].iloc[first]
    assert t["side"].tolist() == [1, -1] and t["entry_time"].iloc[1] == p.index[601]


def test_a_fixed_list_scored_from_its_half_is_held_as_a_fixed_list(monkeypatch):
    from strategy_lab.engine.hold import bought_on, buy_and_hold
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC", "td:DDD"), n=900, seed=9)
    for f in FIELDS:
        getattr(p, f).iloc[:100, 1:] = np.nan                  # three names start at bar 100: half held from there
        getattr(p, f).iloc[600:, 1] = np.nan                   # BBB stops trading at bar 599
    monkeypatch.setattr(ev, "resolve", lambda name, tf: Universe(name, list(p.ids)))
    monkeypatch.setattr(ev, "load_panel", store_of(p))
    monkeypatch.setattr(ev, "stored_version", lambda ids, tf: 1)
    e = ev.evaluate(ibs_reversion, "four", "1d", monte_carlo=False, robustness=False, save=False)
    _, panel, member = ev._load("four", "1d", None, None)
    assert member is not None                                  # a mask: the list is scored from its half only
    live = bought_on(member, e.oos_daily.index[0])            # bought at the record's first open
    fixed = ev.metrics.daily_returns(buy_and_hold(panel, live, "next_open", refills=False))
    waits = ev.metrics.daily_returns(buy_and_hold(panel, live, "next_open", refills=True))
    days = e.oos_daily.index
    # BBB's money goes to the three left, as a fixed list's does; kept for a newcomer, it would wait for ever
    assert np.allclose(e.bh_daily.to_numpy(), fixed.reindex(days).fillna(0.0).to_numpy(), atol=1e-15)
    assert not np.allclose(fixed.reindex(days).to_numpy(), waits.reindex(days).to_numpy())
