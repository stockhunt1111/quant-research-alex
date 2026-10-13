"""The walk-forward of a strategy on a list, as an evaluation and its robustness checks both run it: the list's panel
and members loaded (`load`), a configuration run through the engine (`run`), every configuration of the grid run and
each window's choice stitched into the out-of-sample record (`walk_forward`, `folds`, `stitched`), and what a switch of
configuration at a window's first bar costs (`switch_costs`)."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from lab.data.bars import Panel, load_panel, stored_version
from lab.data.delisting import ended
from lab.data.universes import resolve
from lab.engine import backtest as bt
from lab.engine import costs
from lab.engine.strategy import Strategy, fill_signals
from lab.evaluation import metrics, walkforward
from lab.foundation import log
from lab.foundation.config import WF_SCHEMES

LOG = log.get("walk")


def run(strategy: Strategy, panel: Panel, cfg: dict, member, fill: str, memo: dict | None = None,
         delay: int = 0, signals: dict | None = None, parts: bool = True) -> bt.Result:
    """`delay` = k decides every position k bars later than the strategy does: its entries and its rule's own exits k
    bars later, and a stop or target walked from the fill the delayed order gets (it rests in the market from there,
    not from the rule's decision). `parts` off: a run whose record alone is read, no trades (`backtest.run`)."""
    target, fills = strategy.target_and_exits(panel, cfg, member, memo, signals)
    decided = strategy.decisions(panel, cfg)
    if delay:
        target, fills = target.shift(delay).fillna(0.0), None
        decided = None if decided is None else np.r_[np.zeros(delay, dtype=bool), decided[:-delay]]
    return bt.run(panel, target, fill=fill, exits=strategy.exits(cfg), book=strategy.rebalanced_whole,
                  exit_fills=fills, decided=decided, parts=parts)


def _switch_cost(panel: Panel, a: bt.Result | None, b: bt.Result | None, when: pd.Timestamp) -> float:
    """Cost of moving from config a's positions to config b's at the first bar of a new window (None: no position,
    a window no configuration could be chosen for), paid where that bar's fills are: at its open, or at the close
    before it for next_close fills. The positions are those held into that bar, each configuration's last fills
    before it: what b fills on the bar itself is in b's own returns, and a's fills there never happen."""
    return float(switch_costs(panel, a, b, when).sum())


def switch_costs(panel: Panel, a: bt.Result | None, b: bt.Result | None, when: pd.Timestamp) -> np.ndarray:
    """`_switch_cost` by instrument."""
    k = panel.index.searchsorted(when)
    if k >= len(panel.index) or (a is None and b is None):
        return np.zeros(len(panel.ids))
    rates = costs.rates(panel)
    rate = rates.row(k, "at_open") if (a or b).fill == "next_open" else rates.row(max(k - 1, 0), "at_close")
    held = lambda r: (np.zeros(len(panel.ids)) if r is None or k == 0          # noqa: E731
                      else r.weights.iloc[k - 1].to_numpy())
    return np.abs(held(a) - held(b)) * rate


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
    held = ((panel.started & ~ended(panel)) if member is None else member).sum(axis=1)
    enough = held >= math.ceil(size / 2)
    return enough.idxmax() if enough.any() else None


def walk_forward(strategy, panel, member, fill, configs, timeframe, delay: int = 0, keep_all: bool = True,
                  memo: dict | None = None, signals: dict | None = None, parts: bool = True):
    """Every configuration run over the whole panel, one of them chosen for each window on the windows before it.

    Returns {configuration: its backtest}, the daily returns of every configuration, the windows' rows, the choices,
    the stitched out-of-sample record and the best in-sample configuration. With `keep_all` off only the backtests of
    the configurations chosen and of the best in-sample one are returned, run again once the choice is made: a list's
    backtest holds a weight and an exit price per bar and instrument, and a whole grid's took gigabytes. `memo` (the
    positions of each configuration on this membership) is filled for the caller's own runs of the same
    configurations; `signals`: see `Strategy.target`. `parts` off: the backtests returned carry no trades (a list whose
    record alone is read: a neighbouring list's), and a grid's first pass carries none unless it keeps them all."""
    memo = {} if memo is None else memo
    fill_signals(strategy, panel, configs, signals)
    kept: dict[int, bt.Result] = {}
    daily = {}
    for ci, cfg in enumerate(configs):
        r = run(strategy, panel, cfg, member, fill, memo, delay, signals, parts and (keep_all or len(configs) == 1))
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
        return {c: kept[c] if c in kept else run(strategy, panel, configs[c], member, fill, memo, delay, signals,
                                                  parts)
                for c in sorted({c for c in needed if c is not None})}

    if len(configs) == 1:
        return runs_of([0]), daily, [], [0], daily[0], best_is
    fl = folds(daily, timeframe)
    picks = walkforward.choose(daily, fl)
    choices = [c for c, _ in picks]
    runs = runs_of(choices + [best_is])
    oos = stitched(panel, runs, daily, fl, choices)
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


def folds(daily: pd.DataFrame, timeframe: str) -> list[walkforward.Fold]:
    """The walk-forward windows over the record of `daily`."""
    scheme = WF_SCHEMES[timeframe]
    return walkforward.folds(daily.index.min(), daily.index.max() + pd.Timedelta(days=1), scheme["first_train"],
                             scheme["test"], scheme["train"])


def stitched(panel: Panel, runs: dict[int, bt.Result], daily: pd.DataFrame, fl: list, choices: list[int | None],
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


# the last list `load` read, for the next evaluation on it: a batch runs a list's jobs one after another in one process
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
    its daily seat on the intraday bars that close on the day the seat is in force, while it trades there.

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
            # the names in force on each day, from the daily bar that closes on it (a coin's closes at the day's first
            # instant, a stock's in its evening), held on the intraday bars that close on it, and on the days after the
            # last daily bar until a daily bar says otherwise (a day's daily bar is stored once the day is over)
            on_day = seats.groupby(seats.index.normalize()).max()
            days = seating.index.normalize()
            on_bars = (on_day.reindex(on_day.index.union(days.unique())).ffill().reindex(days)
                       .reindex(columns=seating.ids).eq(True))
            on_bars.index = seating.index
            seating.memo[key] = on_bars & seating.started & ~ended(seating)
    return seating.memo[key].copy()


def _daily_seating(ids: list[str], start, end) -> Panel:
    """The closes and dollar volumes of the daily bars of a list's candidates, kept for the next list ranked on them."""
    key = (tuple(ids), start, end, stored_version(ids, "1d"))
    if key not in _DAILY_SEATING:
        _DAILY_SEATING.clear()
        _DAILY_SEATING[key] = load_panel(ids, "1d", start, end, fields=("close", "dollar_volume"))
    return _DAILY_SEATING[key]


def load(universe: str, timeframe: str, start, end, *, keep: bool = True, base: Panel | None = None):
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
