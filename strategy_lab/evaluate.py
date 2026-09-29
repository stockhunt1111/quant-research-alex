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
     (`strategy_lab.robustness`): Monte Carlo (block bootstrap) of the OOS record, random timing (the OOS positions
     against the same positions moved in time), and the other checks, measured on this evaluation's own backtests; a
     record that loses money is not checked.
A strategy without a grid has nothing to fit: its whole history is reported as out-of-sample.
A saved evaluation goes to the app's database (strategy_lab.db) in one transaction.
"""
from __future__ import annotations

import inspect
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy_lab import db, log, metrics, montecarlo, provenance, significance, walkforward
from strategy_lab.config import WF_SCHEMES
from strategy_lab.data.bars import Panel, load_panel, stored_version
from strategy_lab.engine import backtest as bt
from strategy_lab.engine import costs
from strategy_lab.engine.hold import buy_and_hold, margined, nothing_to_hold
from strategy_lab.engine.trades import ledger, trades_of
from strategy_lab.strategy import Strategy, fill_signals
from strategy_lab.universes import resolve

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
    robustness: dict | None = None        # measured by strategy_lab.robustness; None: it loses money, nothing checked
    grid: dict = field(default_factory=dict)      # the grid the list ran (`Strategy.grid_for` its number of names)

    def summary(self) -> str:
        o, i = self.oos, self.in_sample
        pct = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v * 100:+.2f}%"   # noqa: E731
        return (f"{self.strategy} | {self.universe} {self.timeframe} | OOS {o['start']}..{o['end']}: "
                f"avg/month {pct(o['avg_monthly'])}, green {o['pct_green_active'] * 100:.0f}% of "
                f"{o['pct_months_active'] * 100:.0f}% months in market, maxDD {pct(o['max_dd'])}, "
                f"Sharpe {o['sharpe']:.2f}, trades {o['n_trades']}, beats B&H {o.get('beats_bh')}, "
                f"targets {o['targets_met']}/5 | IS Sharpe {i['sharpe']:.2f}")


def _run(strategy: Strategy, panel: Panel, cfg: dict, member, fill: str, memo: dict | None = None,
         delay: int = 0, signals: dict | None = None) -> bt.Result:
    """`delay` = k decides every position k bars later than the strategy does: its entries and its rule's own exits k
    bars later, and a stop or target walked from the fill the delayed order gets (it rests in the market from there,
    not from the rule's decision)."""
    target, fills = strategy.target_and_exits(panel, cfg, member, memo, signals)
    if delay:
        target, fills = target.shift(delay).fillna(0.0), None
    return bt.run(panel, target, fill=fill, exits=strategy.exits(cfg), book=strategy.rebalanced_whole,
                  exit_fills=fills)


def _switch_cost(panel: Panel, a: bt.Result | None, b: bt.Result | None, when: pd.Timestamp) -> float:
    """Cost of moving from config a's positions to config b's at the first bar of a new window (None: no position,
    a window no configuration could be chosen for), paid where that bar's fills are: at its open, or at the close
    before it for next_close fills."""
    k = panel.index.searchsorted(when)
    if k >= len(panel.index) or (a is None and b is None):
        return 0.0
    rates = costs.rates(panel)
    rate = rates.row(k, "at_open") if (a or b).fill == "next_open" else rates.row(max(k - 1, 0), "at_close")
    held = lambda r: np.zeros(len(panel.ids)) if r is None else r.weights.iloc[k].to_numpy()     # noqa: E731
    return float((np.abs(held(a) - held(b)) * rate).sum())


def first_held_day(member: pd.DataFrame | None) -> pd.Timestamp | None:
    """The UTC day of the first bar on which the universe holds any instrument (a bar's day as metrics.daily_returns
    counts it); None without a membership rule or when it never holds one."""
    if member is None:
        return None
    held = member.any(axis=1)
    return (held.idxmax() - pd.Timedelta(microseconds=1)).normalize() if held.any() else None


def half_held_from(panel: Panel, member: pd.DataFrame | None, size: int) -> pd.Timestamp | None:
    """The first bar on which the universe holds at least half of its `size` instruments (members, or instruments
    trading then when it has no membership rule); None when it never does."""
    held = ((panel.started & ~bt.ended(panel)) if member is None else member).sum(axis=1)
    enough = held >= math.ceil(size / 2)
    return enough.idxmax() if enough.any() else None


def _walk_forward(strategy, panel, member, fill, configs, timeframe, delay: int = 0, keep_all: bool = True,
                  memo: dict | None = None, signals: dict | None = None):
    """Every configuration run over the whole panel, one of them chosen for each window on the windows before it.

    Returns {configuration: its backtest}, the daily returns of every configuration, the windows' rows, the choices,
    the stitched out-of-sample record and the best in-sample configuration. With `keep_all` off only the backtests of
    the configurations chosen and of the best in-sample one are returned, run again once the choice is made: a list's
    backtest holds a weight and an exit price per bar and instrument, and a whole grid's took gigabytes. `memo` (the
    positions of each configuration on this membership) is filled for the caller's own runs of the same
    configurations; `signals`: see `Strategy.target`."""
    memo = {} if memo is None else memo
    fill_signals(strategy, panel, configs, signals)
    kept: dict[int, bt.Result] = {}
    daily = {}
    for ci, cfg in enumerate(configs):
        r = _run(strategy, panel, cfg, member, fill, memo, delay, signals)
        daily[ci] = metrics.daily_returns(r.returns)
        if keep_all or len(configs) == 1:
            kept[ci] = r
    daily = pd.DataFrame(daily).fillna(0.0)
    # the record starts when the universe first holds a name: the bars before it (a stock's history older than the
    # index membership data) trade nothing, and empty years would dilute every figure and fit the first window on zeros
    first = first_held_day(member)
    if first is not None:
        daily = daily[daily.index >= first]
    best_is = int(np.argmax([walkforward.sharpe(daily[c]) for c in daily.columns]))

    def runs_of(needed) -> dict[int, bt.Result]:
        if keep_all:
            return kept
        return {c: kept[c] if c in kept else _run(strategy, panel, configs[c], member, fill, memo, delay, signals)
                for c in sorted({c for c in needed if c is not None})}

    if len(configs) == 1:
        return runs_of([0]), daily, [], [0], daily[0], best_is
    fl = _folds(daily, timeframe)
    picks = walkforward.choose(daily, fl)
    choices = [c for c, _ in picks]
    runs = runs_of(choices + [best_is])
    oos = _stitched(panel, runs, daily, fl, choices)
    # a window no configuration could be chosen for trades nothing: its parameters are none
    fold_rows = [{"train_start": f.train_start.date(), "train_end": f.train_end.date(), "test_start": f.test_start.date(),
                  "test_end": f.test_end.date(), "config": c, "train_sharpe_daily": s,
                  "params": json.dumps({} if c is None else configs[c])}
                 for f, (c, s) in zip(fl, picks)]
    blind = sum(c is None for c in choices)
    if blind:
        LOG.info("%s: %d of %d windows hold no position, no configuration having traded on %d days of the history "
                 "before them", strategy.name, blind, len(choices), walkforward.MIN_TRAIN_DAYS_WITH_RETURNS)
    return runs, daily, fold_rows, choices, oos, best_is


def _folds(daily: pd.DataFrame, timeframe: str) -> list[walkforward.Fold]:
    """The walk-forward windows over the record of `daily`."""
    scheme = WF_SCHEMES[timeframe]
    return walkforward.folds(daily.index.min(), daily.index.max() + pd.Timedelta(days=1), scheme["first_train"],
                             scheme["test"], scheme["train"])


def _stitched(panel: Panel, runs: dict[int, bt.Result], daily: pd.DataFrame, fl: list, choices: list[int | None],
              cost_scale: float = 1.0) -> pd.Series:
    """Each window's chosen configuration's daily returns, one window after another (a window without a choice:
    none); a switch of configuration at a window's first bar pays the trade it causes, at `cost_scale` times the
    modelled cost."""
    oos = walkforward.stitch(daily, fl, choices).copy()
    for prev, cur, f in zip(choices[:-1], choices[1:], fl[1:]):
        if prev != cur:
            day = f.test_start.normalize()
            if day in oos.index:
                paid = cost_scale * _switch_cost(panel, runs.get(prev), runs.get(cur), f.test_start)
                oos.loc[day] = (1 + oos.loc[day]) * (1 - paid) - 1
    return oos


@dataclass
class Held:
    """What the out-of-sample record held: its trades, each position's weight at its last fill on the record's bars,
    and the exposure held through each of them."""
    trades: pd.DataFrame
    weights: pd.DataFrame
    exposure: pd.Series


def _oos_held(panel: Panel, runs: dict[int, bt.Result], fold_rows: list[dict], choices: list[int | None],
              days: pd.DatetimeIndex, member=None) -> Held:
    """The out-of-sample record's positions on the bars of its `days`: each window's chosen configuration's, one window
    after another (none in a window without a choice), and their trades, as one account holds them. A position a
    switch keeps on its side runs on as one trade, one it closes ends at the switch, and a trade is the record's from
    its first bar in it to its last: taken by configuration, a trade entered in one window would count whole to its end
    and one carried into the next window not at all."""
    idx = panel.index
    bar_day = (idx - pd.Timedelta(microseconds=1)).normalize()      # a bar's UTC day, as the daily record counts it
    if not fold_rows:
        r = runs[0]
        span = (bar_day >= days.min()) & (bar_day <= days.max())
        return Held(ledger(panel, r, member), r.weights[span], r.exposure[span])
    n, m = len(idx), len(panel.ids)
    weights = np.zeros((m, n)).T
    exit_px = np.full((m, n), np.nan).T
    liquidated = np.zeros((m, n), dtype=bool).T
    exposure = np.zeros(n)
    span = np.zeros(n, dtype=bool)
    for row, c in zip(fold_rows, choices):
        lo, hi = pd.Timestamp(row["test_start"], tz="UTC"), pd.Timestamp(row["test_end"], tz="UTC")
        rows = (bar_day >= lo) & (bar_day < hi)
        span |= rows
        if c is None:
            continue
        r = runs[c]
        weights[rows] = r.weights.to_numpy()[rows]
        exit_px[rows] = r.exits.to_numpy()[rows]
        liquidated[rows] = r.liquidated.to_numpy()[rows]
        exposure[rows] = r.exposure.to_numpy()[rows]
    held = pd.DataFrame(weights, index=idx, columns=panel.ids)
    fill = next(iter(runs.values())).fill                          # every configuration's: the evaluation's
    trades = trades_of(panel, held, pd.DataFrame(exit_px, index=idx, columns=panel.ids), fill, member,
                       pd.DataFrame(liquidated, index=idx, columns=panel.ids))
    return Held(trades, held[span], pd.Series(exposure, index=idx)[span])


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
    """Daily returns of holding the universe (`engine.hold`), bought on the first bar from `start` on (the first bar
    when None): a comparison starts where the record it is compared with starts, not with weights drifted since. Zero
    when there is nothing to hold (`engine.hold.nothing_to_hold`): cash. `ranked`: the universe is a ranked list, whose
    empty seats newcomers take; else a fixed list (`member` then only says from when it is held), whose leaver's money
    goes to the rest."""
    live = (panel.started if member is None else member).copy()          # `member` is kept for the next job
    if start is not None:
        live.loc[live.index < start] = False
    return metrics.daily_returns(buy_and_hold(panel, live, fill, refills=ranked))


# the last list `_load` read, for the next evaluation on it: a batch runs a list's jobs one after another in one process
_LOADED: dict[tuple, tuple] = {}
# the closes and dollar volumes of the candidates the last list's seats were worked out from: the lists next to it in
# its market (Top-80 and Top-120 around Top-100) have the same candidates
_SEATING: dict[tuple, Panel] = {}


def _seating(ids: list[str], timeframe: str, start, end, version) -> Panel:
    key = (tuple(ids), timeframe, start, end, version)
    if key not in _SEATING:
        _SEATING.clear()
        _SEATING[key] = load_panel(ids, timeframe, start, end, fields=("close", "dollar_volume"))
    return _SEATING[key]


def _bars(ids: list[str], timeframe: str, start, end, index: pd.DatetimeIndex, base: Panel | None) -> Panel:
    """The instruments' full bars on `index`, those `base` holds on the same bars taken from it and only the others
    read (a column read alone on `index` is the same as read with any others)."""
    have = [] if base is None or not base.index.equals(index) else [i for i in ids if i in base.instruments]
    if not have:
        return load_panel(ids, timeframe, start, end, index=index)
    missing = [i for i in ids if i not in have]
    read = load_panel(missing, timeframe, start, end, index=index) if missing else None
    fields = {}
    for f in ("open", "high", "low", "close", "volume", "dollar_volume"):
        parts = [getattr(base, f)[have]] + ([getattr(read, f)] if read is not None else [])
        fields[f] = pd.concat(parts, axis=1)[ids]
    instruments = {i: base.instruments[i] if i in have else read.instruments[i] for i in ids}
    return Panel(timeframe, instruments, **fields)


# the closes and dollar volumes of the daily bars a ranked list's intraday timeframes are ranked on (`_seats`)
_DAILY_SEATING: dict[tuple, Panel] = {}


def _seats(uni, universe: str, timeframe: str, start, end, seating: Panel) -> pd.DataFrame:
    """A ranked list's seats on the bars of `seating`, ranked on its daily bars whatever the timeframe, so a list holds
    the same names on 1h and 4h as on 1d: a session's intraday bars miss a varying share of its volume (its auctions:
    Twelve Data's hourly bars carry 74-89% of AAPL's and MSFT's daily volume, 91-97% of TSLA's and AMD's), and ranked
    on them the stocks' Top-3 held other names than on its daily bars on a third of the days since 2021. A name holds
    its daily seat on the intraday bars of the same day while it trades there.

    Worked out once per list and candidates' bars, and kept on the panel of those bars (`seating` and the daily one):
    a batch runs a list's jobs one after another, and every job ranks the lists next to it again for its checks
    (`robustness.neighbour_lists`), from the same candidates. Each caller gets its own copy."""
    key = ("seats", universe)
    if key not in seating.memo:
        if timeframe == "1d":
            seating.memo[key] = uni.member(seating)
        else:
            daily = resolve(universe, "1d")
            ranked_on = _daily_seating(daily.ids, start, end)
            if key not in ranked_on.memo:
                ranked_on.memo[key] = daily.member(ranked_on)
            seats = ranked_on.memo[key]
            day = lambda idx: (idx - pd.Timedelta(microseconds=1)).normalize()            # noqa: E731
            on_bars = (seats.groupby(day(seats.index)).max().reindex(day(seating.index)).reindex(columns=seating.ids)
                       .eq(True))
            on_bars.index = seating.index
            seating.memo[key] = on_bars & seating.started & ~bt.ended(seating)
    return seating.memo[key].copy()


def _daily_seating(ids: list[str], start, end) -> Panel:
    """The closes and dollar volumes of the daily bars of a list's candidates, kept for the next list ranked on them."""
    key = (tuple(ids), start, end, stored_version(ids, "1d"))
    if key not in _DAILY_SEATING:
        _DAILY_SEATING.clear()
        _DAILY_SEATING[key] = load_panel(ids, "1d", start, end, fields=("close", "dollar_volume"))
    return _DAILY_SEATING[key]


def _load(universe: str, timeframe: str, start, end, *, keep: bool = True, base: Panel | None = None):
    """The list's universe, its panel (only the instruments it ever holds) and its membership, from the day it holds
    half of its names. Kept for the next call on the same list while the store holds the same bars for it: reading a
    list's bars and working out its seats was half of a short evaluation.

    With `keep` off (a neighbouring list evaluated inside another list's evaluation) the list is neither taken from
    nor put in that cache, so the list around it stays there; its seats come from the kept candidates' closes when they
    are the same, and of its full bars only the names `base` (the panel of the list around it) does not hold are read."""
    uni = resolve(universe, timeframe)
    version = stored_version(uni.ids, timeframe)
    key = (universe, timeframe, start, end, tuple(uni.ids), version)
    if keep:
        if key in _LOADED:
            return _LOADED[key]
        _LOADED.clear()                         # the last list's bars go before the next one's are read
    if uni.member is None:
        panel, member = load_panel(uni.ids, timeframe, start, end), None
    else:
        # the seats need only the candidates' closes and dollar volumes (their daily bars', `_seats`); instruments never
        # in the universe carry zero weight in every configuration, so only those it ever holds are read in full, on the
        # bars of all candidates (a whole list's bars at once took over 5 GB for crypto 1h, in every process of a batch)
        seating = _seating(uni.ids, timeframe, start, end, version)
        member = _seats(uni, universe, timeframe, start, end, seating)
        ever = [i for i in seating.ids if bool(member[i].any())]
        if not ever:
            raise ValueError(f"{universe} never holds an instrument on {timeframe} bars")
        panel = _bars(ever, timeframe, start, end, seating.index, base)
        member = member[ever]
    listed = half_held_from(panel, member, uni.capacity)
    if listed is None:
        LOG.warning("%s never holds half of its %d instruments on %s bars: its whole history is scored", universe,
                    uni.capacity, timeframe)
    elif listed > panel.index.min():
        # the universe is not traded before it holds half of its names: a mask, so the record, the buy-and-hold and
        # its checks all start there, and the bars before it still feed the indicators
        live = (panel.started if member is None else member).copy()
        live.loc[live.index < listed] = False
        member = live
        LOG.info("%s holds half of its %d instruments from %s: the record starts there, older bars are indicator "
                 "history", universe, uni.capacity, listed.date())
    out = uni, panel, member
    if keep:
        _LOADED[key] = out
    return out


def evaluate(strategy: Strategy, universe: str, timeframe: str, *, start=None, end=None, fill: str = "next_open",
             monte_carlo: bool = True, robustness: bool = True, save: bool = True) -> Evaluation:
    """`robustness` off leaves out the checks of `strategy_lab.robustness` but those Monte Carlo and random timing
    measure anyway (a quick run; the report shows the others as not computed)."""
    t0 = time.perf_counter()
    code = provenance.stamp(inspect.getsourcefile(strategy.fn))
    uni, panel, member = _load(universe, timeframe, start, end)
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
    runs, daily, fold_rows, choices, oos, best_is = _walk_forward(strategy, panel, member, fill, configs, timeframe,
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
    timing = _random_timing(panel, held.weights, span, fill) if earns else {}
    checks = None
    trades = held.trades
    if earns:
        from strategy_lab import robustness as rb           # it evaluates neighbouring lists through this module
        fit = rb.Fit(strategy, universe, timeframe, start, end, fill, panel, member, configs, daily, choices,
                     _folds(daily, timeframe) if len(configs) > 1 else [], oos, runs, memo, signals, trades, bh)
        del runs, is_run, held                            # the checks' own backtests follow: these are the fit's now
        checks = rb.measure(fit, mc, timing, only=None if robustness else ("luck", "timing"))
    else:
        LOG.info("%s on %s %s loses money out-of-sample (Sharpe %.2f, compounded %s a year): no robustness checks",
                 strategy.name, universe, timeframe, oos_card["sharpe"], oos_card["cagr"])
    return Evaluation(strategy.name, universe, timeframe, oos_card, is_card, mc, pd.DataFrame(fold_rows), oos, trades,
                      configs[best_is], notes, is_daily=is_daily, bh_daily=bh, code=code, random_timing=timing,
                      robustness=checks)


def _random_timing(panel: Panel, weights: pd.DataFrame, span: pd.DatetimeIndex, fill: str) -> dict:
    """`significance.random_timing` of the OOS record; its valuation follows next-open fills only."""
    if fill != "next_open":
        LOG.info("random timing is valued for next-open fills: not measured on a %s run", fill)
        return {}
    return significance.random_timing(panel, weights, span)


def _save(ev: Evaluation, description: str, fill: str, seconds: float) -> None:
    """The evaluation into the app's database, in one transaction: a reader sees all of it or none of it."""
    conn = db.connect()
    try:
        db.save_list_result(conn, ev, description, fill=fill, seconds=seconds)
    finally:
        conn.close()
