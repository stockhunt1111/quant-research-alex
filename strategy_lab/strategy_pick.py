"""A strategy chosen for each instrument of the ML task: the task the firm set (for a given asset, the
best strategy with as little overfitting as possible), measured on history the way everything here is.

Every strategy run on the instrument alone (strategy_lab.per_asset: the rules and the firm's engine, ml_feature_search)
is a candidate, with its own walk-forward record. The choice among them is the same walk-forward that chooses a
strategy's parameters (`walkforward.choose`), one level up: every 91 days, from the day the candidates have a year of
records, the candidate with the best Sharpe on all the days before it (a candidate with a position on fewer than
MIN_TRAIN_DAYS_WITH_RETURNS of them is not chosen; none left: no position) is held until the next choice. Both
choices — the parameters inside a strategy and the strategy for the instrument — are made on the past only, so the
stitched record is what choosing the best strategy would have earned. A switch of strategy pays the trade it causes:
the instrument's cost a side at the window's first open (`engine.costs`) times the difference of the two strategies'
positions on the day before the window, as a switch of configuration pays inside an evaluation.

The result is saved as strategy "strategy_pick" on the instrument (a list of the ML task, a timeframe), with the
windows' choices as its windows and the candidates it chose among (`pick_candidate`), stamped with the code of the
evaluations, the candidates' strategies and `lists.py` too, whose order of the lists decides which list's run of a
strategy is a candidate. Looking at many such records and
singling out the best is a choice again: the luck bar over them is the tries of family picks (`board.luck_of`).

    pick_all(conn)      # every instrument of the ML task's lists on each timeframe (the picks stage of a run)
"""
from __future__ import annotations

import inspect
import sys

import numpy as np
import pandas as pd

from strategy_lab import db, lists, log, metrics, provenance, walkforward
from strategy_lab.config import TIMEFRAMES, WF_SCHEMES
from strategy_lab.data.bars import load_panel
from strategy_lab.engine import costs
from strategy_lab.engine.hold import margined, nothing_to_hold
from strategy_lab.evaluate import _buy_and_hold
from strategy_lab.universes import instruments_now

LOG = log.get("strategy_pick")
NAME = "strategy_pick"
CANDIDATES = ("sma_cross", "ibs_reversion", "donchian_breakout", "ema_trend", "tsmom", "breakout_trail", "calm_trend",
              "trend_or_revert", "late_entry_trend", "ml_direction", "rsi2_connors", "gtaa_faber", "vol_managed",
              "bollinger_reversion", "keltner_breakout", "ibs", "ibs_ml_filter", "ibs_ml_sized", "rsi2_ml_filter",
              "rsi2_ml_sized", "bollinger_ml_filter", "bollinger_ml_sized", "trend_or_revert_ml_filter",
              "trend_or_revert_ml_sized", "ml_feature_search")
DESCRIPTION = ("Every 91 days, the strategy with the best Sharpe so far on this instrument alone, among every strategy "
               "run on it, held until the next choice: the best-strategy-for-an-asset engine, walk-forward.")


def candidates(conn, instrument: str, timeframe: str) -> list[dict]:
    """Every candidate strategy on the instrument alone with its record and its daily position; a strategy run from
    two lists is taken from the list earlier in `lists.PER_INSTRUMENT` (as the Assets view shows it). A record saved
    without its positions (before the database kept them) is left out: its switches could not be charged."""
    rows = conn.execute(f"SELECT id, strategy, list_id FROM result WHERE instrument_id = ? AND timeframe = ? AND strategy "
                        f"IN ({', '.join('?' * len(CANDIDATES))})", (instrument, timeframe, *CANDIDATES)).fetchall()
    order = {u: k for k, u in enumerate(lists.PER_INSTRUMENT)}
    best: dict = {}
    for r in sorted(rows, key=lambda r: order.get(r["list_id"], len(order))):
        best.setdefault(r["strategy"], r)
    out = []
    for name, r in sorted(best.items()):
        oos, position = db.series(conn, r["id"]), db.series(conn, r["id"], "position")
        if position is None:
            LOG.info("%s %s %s: saved without its positions, not a candidate (re-run it)", name, instrument, timeframe)
            continue
        out.append({"id": r["id"], "strategy": name, "oos": oos, "position": position})
    return out


def choose(cands: list[dict], timeframe: str,
           cost: float | pd.Series) -> tuple[pd.Series, pd.Series, list[dict], list[int | None]]:
    """The chosen record (net of the switches' cost), its daily position, its windows and each window's candidate.
    `cost`: a side, the same on every day, or on each bar (by its close) where the rate changes bar by bar."""
    daily = pd.DataFrame({k: c["oos"] for k, c in enumerate(cands)}).sort_index()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC"))
    held = pd.DataFrame({k: c["position"] for k, c in enumerate(cands)}).reindex(daily.index)
    scheme = WF_SCHEMES[timeframe]
    folds = walkforward.folds(daily.index.min(), daily.index.max() + pd.Timedelta(days=1), scheme["first_train"],
                              scheme["test"], scheme["train"])
    picks = walkforward.choose(daily, folds)
    choices = [c for c, _ in picks]
    record = walkforward.stitch(daily.fillna(0.0), folds, choices).copy()
    position = walkforward.stitch(held.fillna(0.0), folds, choices).copy()
    for f, prev, cur in zip(folds[1:], choices[:-1], choices[1:]):
        if prev == cur:
            continue
        before = f.test_start - pd.Timedelta(days=1)
        was = 0.0 if prev is None else float(held.at[before, prev]) if before in held.index else 0.0
        now = 0.0 if cur is None else float(held.at[before, cur]) if before in held.index else 0.0
        day = f.test_start.normalize()
        if day in record.index and not (np.isnan(was) or np.isnan(now)):
            at = min(cost.index.searchsorted(f.test_start), len(cost) - 1) if not np.isscalar(cost) else 0
            rate = cost if np.isscalar(cost) else float(cost.iloc[at])
            record.loc[day] = (1 + record.loc[day]) * (1 - rate * abs(now - was)) - 1
    windows = [{"train_start": f.train_start.date(), "train_end": f.train_end.date(), "test_start": f.test_start.date(),
                "test_end": f.test_end.date(), "train_sharpe_daily": s if np.isfinite(s) else None,
                "params": {"strategy": None if c is None else cands[c]["strategy"]}}
               for f, (_, s), c in zip(folds, picks, choices)]
    return record, position, windows, choices


def pick(conn, instrument: str, universe: str, timeframe: str, fill: str = "next_open") -> db.Outcome | None:
    """The choice on one instrument and timeframe; None when fewer than two candidates have records to choose on."""
    cands = candidates(conn, instrument, timeframe)
    if len(cands) < 2:
        LOG.info("%s %s: %d candidates, nothing to choose between", instrument, timeframe, len(cands))
        return None
    panel = load_panel([instrument], timeframe)
    rate = pd.Series(costs.rates(panel).of(0, "at_open"), index=panel.index)
    record, position, windows, choices = choose(cands, timeframe, rate)
    if len(record) == 0 or len(metrics.monthly_returns(record)) < 1:
        LOG.info("%s %s: the candidates' records are too short for a first choice", instrument, timeframe)
        return None
    cash = nothing_to_hold(panel)
    bh = _buy_and_hold(panel, None, fill, record.index[0]).reindex(record.index).fillna(0.0)   # as per_asset holds
    exposure = position.abs()
    exposure.index = exposure.index + pd.Timedelta(days=1)   # a day's position, as the bar that closes it would carry
    card = metrics.scorecard(record, exposure=exposure, benchmark=bh, cash=cash, margined=margined(panel))
    card |= {"n_trades": None, "trades_per_month": None}      # its trades are the candidates', not kept: not counted
    counts = pd.Series([c for c in choices if c is not None]).value_counts()
    whole = int(np.argmax([walkforward.sharpe(c["oos"]) for c in cands]))
    return db.Outcome(
        strategy=NAME, list_id=universe, timeframe=timeframe, instrument_id=instrument, card=card, oos=record,
        position=position, benchmark=None if cash else bh, windows=windows,
        notes=[f"chosen every 91 days among {len(cands)} strategies run on {instrument.split(':', 1)[1]} alone, each "
               "on its record before the choice"],
        grid={"strategy": [c["strategy"] for c in cands]}, params_in_sample={"strategy": cands[whole]["strategy"]},
        params_now={"strategy": windows[-1]["params"]["strategy"]},
        candidates=[(cands[k]["id"], int(counts.get(k, 0))) for k in range(len(cands))])


def pick_all(conn) -> int:
    """Every instrument of the ML task's lists, on each timeframe: chosen, saved (replacing the earlier choices), and
    the luck bar over them kept. Returns how many were saved."""
    # the candidates come from the lists in the order `lists.PER_INSTRUMENT` gives: a result reads that order too, and
    # the candidates' records, which a change of their strategies' files makes stale
    code = provenance.stamp(inspect.getsourcefile(sys.modules[__name__]),
                            also=("strategy_lab/lists.py", *(f"strategies/{name}.py" for name in CANDIDATES)))
    saved = []
    for universe in lists.ML_TASK:
        for tf in TIMEFRAMES:
            instruments = instruments_now(universe, tf)[0]          # the ML task's instruments today
            outcomes = [o for o in (pick(conn, i, universe, tf) for i in sorted(instruments)) if o is not None]
            db.save_outcomes(conn, outcomes, description=DESCRIPTION, code=code, fill="next_open",
                             replace_run=(NAME, universe, tf))
            saved += outcomes
    if len(saved) >= 2:
        from strategy_lab import board
        daily = pd.DataFrame({k: o.oos for k, o in enumerate(saved)})
        independent, best_95 = board.luck_of(daily)
        db.save_tries(conn, "picks", db.results_fingerprint(conn, "picks"), len(saved), independent, best_95,
                      provenance.sha_of(board.TRY_CODE))
    LOG.info("strategy picks: %d instrument-timeframes", len(saved))
    return len(saved)
