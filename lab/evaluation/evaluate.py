"""One call that measures a strategy against the firm's target.

    res = evaluate(sma_cross, "etf_core", "1d")

Steps:
  1. every configuration of the strategy's grid is run through the engine on the universe;
  2. IN-SAMPLE: the configuration that is best over the whole history — the view a product shows when it fits on all
     the data — scored over the out-of-sample days, so that the two records cover the same span;
  3. OUT-OF-SAMPLE: walk-forward — each test window uses the configuration that was best on the window before it (a
     window before which no configuration traded enough to be judged holds no position); switching configuration at
     a window boundary pays the trade it causes; the record starts on the first day the universe holds at least half
     of its instruments (the bars before it are only indicator history: a basket of one or two names is not the list,
     and years in which the list holds nothing would dilute every figure); its trades are those of the positions the
     windows held, one after another, as the account holds them across a switch;
  4. buy-and-hold of the same universe over the same span, bought at its start and held on spot terms
     (`engine.hold`); a universe of FX pairs or crude, which buy-and-hold does not hold, is compared with cash;
  5. a record that makes money out-of-sample (positive Sharpe and compounded return) is checked for robustness
     (`lab.evaluation.robustness`): Monte Carlo (block bootstrap) of the OOS record, random timing (the OOS positions
     against the same positions moved in time), and the other checks, measured on this evaluation's own backtests; a
     record that loses money is not checked.
A strategy without a grid has nothing to fit: its whole history is reported as out-of-sample.
A saved evaluation goes to the app's database (lab.evaluation.db) in one transaction.
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lab.data.bars import Panel
from lab.engine import backtest as bt
from lab.engine.hold import bought_on, buy_and_hold, margined, nothing_to_hold
from lab.engine.strategy import Strategy
from lab.engine.trades import ledger, trades_of
from lab.evaluation import db, metrics, montecarlo, significance, walk
from lab.evaluation import robustness as rb
from lab.foundation import log, provenance

LOG = log.get("evaluate")


@dataclass
class Evaluation:
    strategy: str
    universe: str
    timeframe: str
    oos: dict
    in_sample: dict
    monte_carlo: dict
    folds: pd.DataFrame
    oos_daily: pd.Series
    trades: pd.DataFrame
    params_in_sample: dict
    notes: list[str] = field(default_factory=list)
    is_daily: pd.Series | None = None
    bh_daily: pd.Series | None = None
    code: dict = field(default_factory=dict)
    random_timing: dict = field(default_factory=dict)
    robustness: dict | None = None        # measured by lab.evaluation.robustness; None: it loses money, nothing checked
    grid: dict = field(default_factory=dict)      # the grid the list ran (`Strategy.grid_for` its number of names)

    def summary(self) -> str:
        o, i = self.oos, self.in_sample
        pct = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v * 100:+.2f}%"   # noqa: E731
        return (f"{self.strategy} | {self.universe} {self.timeframe} | OOS {o['start']}..{o['end']}: "
                f"avg/month {pct(o['avg_monthly'])}, green {o['pct_green_active'] * 100:.0f}% of "
                f"{o['pct_months_active'] * 100:.0f}% months in market, maxDD {pct(o['max_dd'])}, "
                f"Sharpe {o['sharpe']:.2f}, trades {o['n_trades']}, beats B&H {o.get('beats_bh')}, "
                f"targets {o['targets_met']}/5 | IS Sharpe {i['sharpe']:.2f}")


@dataclass
class Held:
    """What the out-of-sample record held: its trades, each position's weight at its last fill on the record's bars,
    the exposure held through each of them, and what each position made on each bar (on the record's own path)."""
    trades: pd.DataFrame
    weights: pd.DataFrame
    exposure: pd.Series
    parts: bt.Parts


def _oos_held(panel: Panel, runs: dict[int, bt.Result], fold_rows: list[dict], choices: list[int | None],
              days: pd.DatetimeIndex, member=None) -> Held:
    """The out-of-sample record's positions on the bars of its `days`: each window's chosen configuration's, one window
    after another (none in a window without a choice), and their trades, as one account holds them. A position a
    switch keeps on its side runs on as one trade, one it closes ends at the switch, and a trade is the record's from
    its first bar in it to its last: taken by configuration, a trade entered in one window would count whole to its end
    and one carried into the next window not at all. What each trade made is the windows' runs' (`backtest.Parts`),
    on the record's own path: a switch's cost (`walk.switch_costs`) is the trades' it closes or opens, and a window's
    first bar is its run's, the gap of what that run held into it going to what it holds from there."""
    idx = panel.index
    bar_day = (idx - pd.Timedelta(microseconds=1)).normalize()      # a bar's UTC day, as the daily record counts it
    if not fold_rows:
        r = runs[0]
        span = (bar_day >= days.min()) & (bar_day <= days.max())
        return Held(ledger(panel, r, member), r.weights[span], r.exposure[span], r.parts)
    n, m = len(idx), len(panel.ids)
    weights = np.zeros((m, n)).T
    exit_px = np.full((m, n), np.nan).T
    liquidated = np.zeros((m, n), dtype=bool).T
    exposure = np.zeros(n)
    span = np.zeros(n, dtype=bool)
    net, opened = np.zeros(n), np.ones(n)
    pieces = []                                                    # (bars, instruments, held into, held from the open)
    firsts = []                                                    # each window's first bar, with its switch
    for i, (row, c) in enumerate(zip(fold_rows, choices)):
        lo, hi = pd.Timestamp(row["test_start"], tz="UTC"), pd.Timestamp(row["test_end"], tz="UTC")
        rows = (bar_day >= lo) & (bar_day < hi)
        span |= rows
        if not rows.any():
            continue
        first = int(np.argmax(rows))
        prev = choices[i - 1] if i else c
        if prev != c:
            firsts.append((first, prev, c, lo))
        if c is None:
            continue
        r = runs[c]
        weights[rows] = r.weights.to_numpy()[rows]
        exit_px[rows] = r.exits.to_numpy()[rows]
        liquidated[rows] = r.liquidated.to_numpy()[rows]
        exposure[rows] = r.exposure.to_numpy()[rows]
        net[rows], opened[rows] = r.parts.net[rows], r.parts.opened[rows]
        p = r.parts
        sel = rows[p.bar]
        bar, held_in, held_on = p.bar[sel], p.held_in[sel].copy(), p.held_on[sel].copy()
        if i == 0 or prev != c:                                    # what this run held into its window's first bar
            at = bar == first
            held_on[at] += held_in[at]
            held_in[at] = 0.0
        pieces.append((bar, p.inst[sel], held_in, held_on))
    for first, prev, c, lo in firsts:
        paid = walk.switch_costs(panel, runs.get(prev), runs.get(c), lo)
        if not paid.any():
            continue
        whole = 1.0 + net[first]                                   # the record pays it on its day's growth
        net[first] = whole * (1.0 - paid.sum()) - 1.0
        was, now = weights[first - 1] if first else np.zeros(m), weights[first]
        closes = (was != 0.0) & ((now == 0.0) | (np.sign(was) != np.sign(now)))
        into = np.where(closes, paid * np.abs(was) / np.maximum(np.abs(was) + np.abs(now), 1e-300), 0.0)
        cols = np.flatnonzero(paid)
        pieces.append((np.full(len(cols), first), cols, -whole * into[cols], -whole * (paid - into)[cols]))
    bar, inst, held_in, held_on = (np.concatenate([q[k] for q in pieces]) if pieces else np.zeros(0)
                                   for k in range(4))
    order = np.argsort(bar, kind="stable")
    parts = bt.Parts(bar[order].astype(np.int64), inst[order].astype(np.int64), held_in[order], held_on[order], net,
                     opened)
    held = pd.DataFrame(weights, index=idx, columns=panel.ids)
    fill = next(iter(runs.values())).fill                          # every configuration's: the evaluation's
    trades = trades_of(panel, held, pd.DataFrame(exit_px, index=idx, columns=panel.ids), fill, parts, member,
                       pd.DataFrame(liquidated, index=idx, columns=panel.ids), stitched=True)
    return Held(trades, held[span], pd.Series(exposure, index=idx)[span], parts)


def _log_unfunded(name: str, universe: str, timeframe: str, runs: dict[int, bt.Result], fold_rows: list[dict],
                  choices: list[int | None], span: pd.DatetimeIndex) -> None:
    """Say how often the out-of-sample record's fills found less capital free than they asked for (the engine buys
    no more than the positions held leave free: no leverage), and how much of the equity went unbought at the most."""
    parts = []
    for row, c in zip(fold_rows or [None], choices):
        if c is None:
            continue
        u = runs[c].unfunded
        day = (u.index - pd.Timedelta(microseconds=1)).normalize()
        lo = pd.Timestamp(row["test_start"], tz="UTC") if row else span.min()
        hi = pd.Timestamp(row["test_end"], tz="UTC") if row else span.max() + pd.Timedelta(days=1)
        parts.append(u[(day >= lo) & (day < hi)])
    short = pd.concat(parts) if parts else pd.Series(dtype=float)
    if (short > 0).any():
        LOG.info("%s on %s %s: on %d bars of the out-of-sample record the fills asked for more than the capital the "
                 "positions held left free, and were cut to it (up to %.1f%% of the equity unbought)", name, universe,
                 timeframe, int((short > 0).sum()), 100 * float(short.max()))


def _buy_and_hold(panel: Panel, member, fill: str, start=None, ranked: bool = False) -> pd.Series:
    """Daily returns of holding the universe (`engine.hold`), bought at the first open of the UTC day `start` (the
    first bar when None; `engine.hold.bought_on`): a comparison starts where the record it is compared with starts, not
    with weights drifted since. Zero
    when there is nothing to hold (`engine.hold.nothing_to_hold`): cash. `ranked`: the universe is a ranked list, whose
    empty seats newcomers take; else a fixed list (`member` then only says from when it is held), whose leaver's money
    goes to the rest."""
    live = (panel.started if member is None else member).copy()          # `member` is kept for the next job
    if start is not None:
        live = bought_on(live, start)
    return metrics.daily_returns(buy_and_hold(panel, live, fill, refills=ranked))


def evaluate(strategy: Strategy, universe: str, timeframe: str, *, start=None, end=None, fill: str = "next_open",
             monte_carlo: bool = True, robustness: bool = True, save: bool = True) -> Evaluation:
    """`robustness` off leaves out the checks of `lab.evaluation.robustness` but those Monte Carlo and random timing
    measure anyway (a quick run; the report shows the others as not computed)."""
    t0 = time.perf_counter()
    code = provenance.stamp(inspect.getsourcefile(strategy.fn))
    uni, panel, member = walk.load(universe, timeframe, start, end)
    grid, configs = strategy.grid_for(uni.capacity), strategy.configs(uni.capacity)
    notes = list(uni.notes)
    LOG.info("%s on %s %s: %d instruments, %d bars, %d configurations", strategy.name, universe, timeframe,
             len(panel.ids), len(panel.index), len(configs))
    undo = strategy.prepare(panel, configs, member) if strategy.prepare is not None else None
    try:
        ev = _scored(strategy, universe, timeframe, start, end, panel, member, configs, notes, fill, monte_carlo,
                     robustness, code, ranked=uni.member is not None)
    finally:
        if undo is not None:                # what `prepare` kept for this list's membership serves no other evaluation
            undo()
    ev.grid = grid
    if save:
        _save(ev, strategy.description, fill, time.perf_counter() - t0)
    LOG.info(ev.summary())
    return ev


def _scored(strategy, universe, timeframe, start, end, panel, member, configs, notes, fill, monte_carlo, robustness,
            code, ranked: bool) -> Evaluation:
    memo: dict = {}
    # a rule's positions on each instrument, kept for its checks: not a rule whose function reads what `prepare` keeps
    signals = {} if strategy.kind == "rule" and strategy.prepare is None else None
    runs, daily, fold_rows, choices, oos, best_is = walk.walk_forward(strategy, panel, member, fill, configs, timeframe,
                                                                  keep_all=False, memo=memo, signals=signals)
    if len(configs) == 1:
        notes.append("single configuration: nothing is fitted, the whole history is out-of-sample")
    span = oos.index
    # a rule's trades end by the rule (a name leaving the universe keeps its seat until then); a panel strategy, and a
    # rule that holds an exposure, drops a name that leaves, and the ledger marks those exits
    cut_by_list = member if strategy.kind == "panel" or strategy.exposure else None
    held = _oos_held(panel, runs, fold_rows, choices, span, cut_by_list)
    _log_unfunded(strategy.name, universe, timeframe, runs, fold_rows, choices, span)
    gone = held.trades[held.trades["exit_reason"] == "liquidated"]
    if len(gone):
        LOG.info("%s on %s %s: %d shorts of the out-of-sample record were liquidated, their price at %g times their "
                 "entry (%s)", strategy.name, universe, timeframe, len(gone), bt.LIQUIDATION,
                 ", ".join(sorted(set(gone["instrument"]))))
    bh = _buy_and_hold(panel, member, fill, span[0], ranked=ranked).reindex(span).fillna(0.0)
    cash, futures = nothing_to_hold(panel), margined(panel)
    oos_card = metrics.scorecard(oos, trades=held.trades, exposure=held.exposure, benchmark=bh, cash=cash,
                                 margined=futures)
    is_run = runs[best_is]
    is_daily = daily[best_is].reindex(span).fillna(0.0)
    is_trades = ledger(panel, is_run, cut_by_list)
    is_trades = is_trades[is_trades["entry_time"] >= span.min()]
    is_exposure = is_run.exposure[is_run.exposure.index > span.min()]
    is_card = metrics.scorecard(is_daily, trades=is_trades, exposure=is_exposure, benchmark=bh, cash=cash,
                                margined=futures)
    # a record that loses money out-of-sample is not checked: holding up under costs, a delay or other parameters means
    # nothing for it, and Monte Carlo and random timing were a third of an evaluation
    earns = bool(oos_card["sharpe"] > 0 and oos_card["cagr"] > 0)
    mc = montecarlo.bootstrap(oos) if monte_carlo and earns else {}
    timing = _random_timing(panel, held.weights, span, fill, member) if earns else {}
    checks = None
    trades = held.trades
    if earns:
        fit = rb.Fit(strategy, universe, timeframe, start, end, fill, panel, member, configs, daily, choices,
                     walk.folds(daily, timeframe) if len(configs) > 1 else [], oos, runs, memo, signals, trades, bh)
        del runs, is_run, held                            # the checks' own backtests follow: these are the fit's now
        checks = rb.measure(fit, mc, timing, only=None if robustness else ("luck", "timing"))
    else:
        LOG.info("%s on %s %s loses money out-of-sample (Sharpe %.2f, compounded %s a year): no robustness checks",
                 strategy.name, universe, timeframe, oos_card["sharpe"], oos_card["cagr"])
    return Evaluation(strategy.name, universe, timeframe, oos_card, is_card, mc, pd.DataFrame(fold_rows), oos, trades,
                      configs[best_is], notes, is_daily=is_daily, bh_daily=bh, code=code, random_timing=timing,
                      robustness=checks)


def _random_timing(panel: Panel, weights: pd.DataFrame, span: pd.DatetimeIndex, fill: str,
                   member: pd.DataFrame | None = None) -> dict:
    """`significance.random_timing` of the OOS record, its rotations held to the bars a name is in the list or held
    (`member`: the list's names; None, an instrument alone: every bar it printed); its valuation follows next-open
    fills only."""
    if fill != "next_open":
        LOG.info("random timing is valued for next-open fills: not measured on a %s run", fill)
        return {}
    return significance.random_timing(panel, weights, span, member=member)


def _save(ev: Evaluation, description: str, fill: str, seconds: float) -> None:
    """The evaluation into the app's database, in one transaction: a reader sees all of it or none of it."""
    conn = db.connect()
    try:
        db.save_list_result(conn, ev, description, fill=fill, seconds=seconds)
    finally:
        conn.close()
