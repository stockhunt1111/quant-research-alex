"""Each strategy on each instrument alone — how the firm's products, dashboard and simulator run a strategy: one
strategy on one symbol with its whole capital (e.g. $10,000), parameters picked for that symbol.

For every (strategy, instrument, timeframe):
  * parameters come from a walk-forward on the instrument's OWN past (config.WF_SCHEMES windows), as the firm's
    engine fits a model per symbol; the out-of-sample record is the chosen configurations stitched together;
  * the scorecard compares that record with the target and with buy-and-hold of the same instrument over the same
    span (cash for a currency pair or crude, which buy-and-hold does not hold); the in-sample figures (best
    configuration on the whole history) are kept beside it;
  * random timing (`significance.random_timing`): the record's positions against the same positions moved in time;
  * a record that makes money is checked for robustness as a list's is (`strategy_lab.robustness`): Monte Carlo and
    the other checks, measured on this evaluation's own backtests, all but a list's own (the names it traded, the lists
    next to it); how many of the other instruments of its market the strategy makes money on is judged when the result
    is read (`board.peers`), from their results. A record that loses money is not checked;
  * grid key `slots` is dropped: it splits capital among instruments, and one instrument takes all of it.
An instrument with fewer than MIN_MONTHS out-of-sample months is listed as too short and not scored. Only `rule`
strategies apply; a `panel` strategy trades a basket and is evaluated on its universe (strategy_lab.evaluate).

    outcomes = evaluate(rsi2_connors, "crypto_top100", "1d")    -> the app's database (strategy_lab.db)

A run of a list's instruments is saved in one transaction and replaces that run's earlier results: an instrument no
longer in the list goes with them. Each instrument's result keeps its windows, its record, its position at each day's
end (what strategy_pick charges a switch by) and its buy-and-hold, as a list's result does.
"""
from __future__ import annotations

import inspect
import json
import time

import numpy as np
import pandas as pd

from strategy_lab import db, log, metrics, montecarlo, provenance
from strategy_lab import robustness as rb
from strategy_lab.data.bars import load_panel
from strategy_lab.engine.hold import margined, nothing_to_hold
from strategy_lab.evaluate import _buy_and_hold, _folds, _oos_held, _random_timing, _walk_forward
from strategy_lab.strategy import SIZING_KEYS, Strategy
from strategy_lab.universes import instruments_now

LOG = log.get("per_asset")
MIN_MONTHS = 6


def single_asset_configs(strategy: Strategy) -> list[dict]:
    """The strategy's grid without sizing keys, duplicates removed."""
    out, seen = [], set()
    for cfg in strategy.configs():
        cfg = {k: v for k, v in cfg.items() if k not in SIZING_KEYS}
        key = json.dumps(cfg, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def single_asset_grid(strategy: Strategy) -> dict:
    """The grid a single-asset run chooses from: the strategy's own without sizing keys."""
    return {k: list(v) for k, v in strategy.grid.items() if k not in SIZING_KEYS}


def daily_position(weights: pd.DataFrame, instrument: str, days: pd.DatetimeIndex) -> pd.Series:
    """An instrument's position at each UTC day's end (the weight after the day's last bar; a day without a bar keeps
    the one before), on `days`: what `strategy_pick` charges a switch of strategy by."""
    w = weights[instrument]
    by_day = w.groupby((w.index - pd.Timedelta(microseconds=1)).tz_convert("UTC").normalize()).last()
    return by_day.reindex(by_day.index.union(days)).ffill().reindex(days).fillna(0.0)


def best_days_share(daily: pd.Series, k: int = 5) -> float:
    lg = np.log1p(daily)
    total = lg.sum()
    return float(lg.nlargest(k).sum() / total) if total > 0 else np.nan


def evaluate_instrument(strategy: Strategy, instrument: str, timeframe: str, configs: list[dict], universe: str,
                        fill: str = "next_open") -> tuple[db.Outcome | None, int]:
    """The strategy alone on one instrument: its outcome and its out-of-sample months; no outcome when the record is
    too short to score (fewer than 100 bars: 0 months)."""
    t0 = time.perf_counter()
    panel = load_panel([instrument], timeframe)
    if panel.close[instrument].notna().sum() < 100:
        return None, 0
    # a rule's positions kept on disk serve the lists and this instrument alone alike (`strategy.fill_signals`); not a
    # rule whose function reads what a `prepare` step keeps
    signals = {} if strategy.prepare is None else None
    memo: dict = {}
    runs, daily, fold_rows, choices, oos, best_is = _walk_forward(strategy, panel, None, fill, configs, timeframe,
                                                                  memo=memo, signals=signals)
    months = len(metrics.monthly_returns(oos)) if len(oos) else 0
    if months < MIN_MONTHS:
        return None, months
    held = _oos_held(panel, runs, fold_rows, choices, oos.index)
    cash = nothing_to_hold(panel)
    bh = _buy_and_hold(panel, None, fill, oos.index[0]).reindex(oos.index).fillna(0.0)
    card = metrics.scorecard(oos, trades=held.trades, exposure=held.exposure, benchmark=bh, cash=cash,
                             margined=margined(panel))
    is_card = metrics.core(daily[best_is].reindex(oos.index).fillna(0.0))
    timing = _random_timing(panel, held.weights, oos.index, fill)
    mc, checks = {}, None
    if card["sharpe"] > 0 and (card["cagr"] or 0) > 0:
        mc = montecarlo.bootstrap(oos)
        fit = rb.Fit(strategy, universe, timeframe, None, None, fill, panel, None, configs, daily, choices,
                     _folds(daily, timeframe) if len(configs) > 1 else [], oos, runs, memo, signals, held.trades, bh,
                     instrument=instrument)
        checks = rb.measure(fit, mc, timing)
    else:
        LOG.debug("%s on %s %s loses money out-of-sample (Sharpe %.2f, compounded %s a year): no robustness checks",
                  strategy.name, instrument, timeframe, card["sharpe"], card["cagr"])
    return db.Outcome(
        strategy=strategy.name, list_id=universe, timeframe=timeframe, instrument_id=instrument, card=card, oos=oos,
        in_sample=db.Figures(is_card), oos_deals=db.deals(held.trades),
        position=daily_position(held.weights, instrument, oos.index), benchmark=None if cash else bh,
        windows=fold_rows, monte_carlo=mc, random_timing=timing, robustness=checks, params_in_sample=configs[best_is],
        params_now={} if choices[-1] is None else configs[choices[-1]], best5_share=best_days_share(oos),
        seconds=time.perf_counter() - t0), months


def evaluate(strategy: Strategy, universe: str, timeframe: str, *, fill: str = "next_open",
             save: bool = True) -> list[db.Outcome]:
    if strategy.kind != "rule":
        raise ValueError(f"{strategy.name} is a panel strategy: it trades a basket, evaluate it on its universe")
    code = provenance.stamp(inspect.getsourcefile(strategy.fn))
    ids, notes = instruments_now(universe, timeframe)
    configs, grid = single_asset_configs(strategy), single_asset_grid(strategy)
    LOG.info("%s on each of %d instruments of %s %s, %d configurations", strategy.name, len(ids), universe, timeframe,
             len(configs))
    outcomes, too_short = [], []
    for i in ids:
        o, months = evaluate_instrument(strategy, i, timeframe, configs, universe, fill)
        if o is None:
            too_short.append((i, months))
            continue
        o.notes, o.grid = list(notes), grid
        outcomes.append(o)
    if save:
        conn = db.connect()
        try:
            db.save_outcomes(conn, outcomes, description=strategy.description, code=code, fill=fill,
                             replace_run=(strategy.name, universe, timeframe), too_short=too_short)
        finally:
            conn.close()
    money = [o for o in outcomes if o.card["sharpe"] > 0 and (o.card["cagr"] or 0) > 0]
    LOG.info("%s | %s %s: %d scored, %d too short; making money on %d, beating buy & hold on %d", strategy.name,
             universe, timeframe, len(outcomes), len(too_short), len(money),
             sum(bool(o.card.get("beats_bh")) for o in outcomes))
    return outcomes
