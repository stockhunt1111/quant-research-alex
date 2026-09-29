"""Is an out-of-sample result robust? Twelve checks, measured inside the evaluation of a result on a list
(`evaluate._scored`) or of a strategy on an instrument alone (`per_asset.evaluate_instrument`) and judged against their
thresholds when a report is built (`board.robustness`), so a threshold can change without a re-run. A result that loses
money out-of-sample is not checked: holding up under costs, a delay or other parameters means nothing for a record that
does not make money in the first place.

  luck             Monte Carlo's 5th-percentile Sharpe; the deflated Sharpe needs the number of evaluations made,
                   known when the report is built
  vs_hold          t of the Sharpe above holding the same list over the same days (paired block bootstrap)
  timing           random timing's p (`significance.random_timing`)
  pbo              the probability that the walk-forward's way of choosing a configuration overfits (Bailey,
                   Borwein, Lopez de Prado and Zhu, "The Probability of Backtest Overfitting": combinatorially
                   symmetric cross-validation of the grid's records)
  plateau          the Sharpe of the configurations next to the one the windows chose most
  eras             steadiness over two-year windows: the share of windows with a positive Sharpe, their spread and
                   the worst one (the Minerva tester's consistency)
  delay            the same choices with every position taken a bar later
  costs            the same choices at three times the modelled costs, a CME future at 10bp a side
  names            the share of the list's names whose trades made money
  neighbour_lists  the walk-forward on the Top-N lists next to a Top-N list
  seeds            the same choices with other seeds of the strategy's model
  vs_rule          t of the Sharpe above the same rule without its model

A check's entry holds what it measured; {"na": why} where it does not apply, {"too_short": why} where the record is
too short to judge, and no entry where it was not measured. An instrument alone has one name and no list around it:
`names` and `neighbour_lists` are a list's (LIST_ONLY) and are not measured for it; whether the strategy makes money on
most of the other instruments of its market is judged instead when it is read (`board.peers`), from their results.
"""
from __future__ import annotations

import dataclasses
import gc
import itertools
import math
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap
from scipy import stats

from strategy_lab import evaluate as ev
from strategy_lab import log, metrics, significance, trade_model
from strategy_lab.config import COSTS, SEED
from strategy_lab.data import spreads
from strategy_lab.data.bars import Panel
from strategy_lab.strategy import Strategy

LOG = log.get("robustness")
SEEDS = (1, 2, 3, 4)                # other draws of a model than config.SEED
BOOTSTRAP_REPS = 1000
BLOCK_DAYS = 20
PBO_BLOCKS = 10                     # 252 splits of the record into two halves
ERA_DAYS = 730                      # the Minerva tester's two-year windows
ERA_TAIL_DAYS = 365                 # a last window at least this long counts as one
COST_MULTIPLE = 3.0                 # the research desk's "survives 3x the fee schedule"
# a side, for an instrument charged its class's flat cost: a CME future's thin intraday quotes, whose spread its flat
# 2bp understates (a spot quote pays its broker's spread, bar by bar: `engine.costs`)
STRESS_BPS = {"commodity": 10.0}
TOP_N = re.compile(r"^(?P<base>.+_top)(?P<n>\d+)$")
LIST_ONLY = ("names", "neighbour_lists")    # a list's own: an instrument alone has one name and no list around it


@dataclass
class Fit:
    """What an evaluation's walk-forward leaves for its checks."""
    strategy: Strategy
    universe: str
    timeframe: str
    start: object
    end: object
    fill: str
    panel: Panel
    member: pd.DataFrame | None
    configs: list[dict]
    daily: pd.DataFrame             # every configuration's daily returns from the first day the list holds a name
    choices: list[int | None]       # the configuration of each window ([0] for a single configuration; None:
                                    # a window no configuration could be chosen for, holding nothing)
    folds: list                     # the windows ([] for a single configuration)
    oos: pd.Series
    runs: dict                      # the chosen configurations' backtests
    memo: dict                      # the positions of each configuration on this membership
    signals: dict | None            # a rule's positions on each instrument before its grade and seats
    trades: pd.DataFrame
    bh: pd.Series
    instrument: str | None = None   # the instrument a strategy runs on alone (`per_asset`); None: a list


def measure(fit: Fit, mc: dict, timing: dict, only: tuple[str, ...] | None = None) -> dict:
    """Every check of a result that makes money, the cheap ones first; `only` limits them (a run that skips the
    checks keeps what Monte Carlo and random timing measured anyway). An instrument alone has no LIST_ONLY check."""
    checks = (("luck", lambda: _luck(mc)), ("timing", lambda: _timing(fit, timing)),
              ("vs_hold", lambda: paired_t(fit.oos, fit.bh)), ("pbo", lambda: pbo(fit)), ("eras", lambda: eras(fit.oos)),
              ("names", lambda: names(fit.trades)), ("costs", lambda: costs(fit)), ("delay", lambda: delay(fit)),
              ("plateau", lambda: plateau(fit)), ("vs_rule", lambda: vs_rule(fit)), ("seeds", lambda: seeds(fit)),
              ("neighbour_lists", lambda: neighbour_lists(fit)))
    out, seconds = {}, {}
    for name, check in checks:
        if name == "plateau":
            # the chosen configurations' backtests and positions served the costs and the delay only: a list's take
            # gigabytes on 1h bars, and the checks that follow run their own
            fit.runs, fit.memo = {}, {}
            gc.collect()
        if (only is not None and name not in only) or (fit.instrument is not None and name in LIST_ONLY):
            continue
        t = time.perf_counter()
        got = check()
        seconds[name] = round(time.perf_counter() - t, 3)
        if got is not None:
            out[name] = got
            if "too_short" in got:
                LOG.info("%s on %s %s: %s not judged, %s", fit.strategy.name, fit.instrument or fit.universe,
                         fit.timeframe, name, got["too_short"])
    out["seconds"] = seconds
    return out


def _sr(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(metrics.DAYS)) if len(x) > 1 and sd > 0 else 0.0


def _figures(daily: pd.Series) -> dict:
    c = metrics.core(daily)
    return {"sharpe": _num(c["sharpe"]), "cagr": _num(c["cagr"])}


def _num(v) -> float | None:
    return float(v) if v is not None and np.isfinite(v) else None


def _luck(mc: dict) -> dict | None:
    if not mc:
        return None                                     # Monte Carlo was not run
    if not mc.get("reps"):
        return {"too_short": mc.get("note", "too short for Monte Carlo")}
    return {"mc_sharpe_p5": _num(mc["sharpe"]["p5"])}


def _timing(fit: Fit, timing: dict) -> dict | None:
    if fit.fill != "next_open":
        return None                                     # random timing is valued for next-open fills only
    if not timing:
        return {"too_short": f"fewer than {significance.MIN_DAYS} days, or nothing held"}
    return {"p": _num(timing["p"]), "null_median": _num(timing["null_median"])}


def paired_t(a: pd.Series, b: pd.Series) -> dict:
    """How far a's Sharpe is above b's over the same days, in standard errors of the difference: the two records
    resampled together in blocks (a stationary bootstrap of BLOCK_DAYS days), so what they share cancels."""
    both = pd.concat([a, b], axis=1).fillna(0.0).to_numpy()
    if len(both) < significance.MIN_DAYS:
        return {"too_short": f"fewer than {significance.MIN_DAYS} days"}
    bs = StationaryBootstrap(BLOCK_DAYS, both, seed=np.random.default_rng(SEED))
    diffs = np.array([_sr(x[:, 0]) - _sr(x[:, 1]) for (x,), _ in bs.bootstrap(BOOTSTRAP_REPS)])
    edge, sd = _sr(both[:, 0]) - _sr(both[:, 1]), diffs.std(ddof=1)
    return {"t": _num(edge / sd) if sd > 0 else None, "sharpe": _sr(both[:, 0]), "sharpe_other": _sr(both[:, 1]),
            "reps": BOOTSTRAP_REPS, "block_days": BLOCK_DAYS}


def pbo(fit: Fit) -> dict:
    """The share of the ways to split the record into two halves in which the configuration best on one half ranks in
    the lower half of the grid on the other: the probability that choosing the best configuration picks one that is
    worse than a typical one on new data. Each configuration's record over the out-of-sample days, cut into
    PBO_BLOCKS blocks; every choice of half of them is one split."""
    if len(fit.configs) < 2:
        return {"na": "one configuration: nothing is chosen"}
    m = fit.daily.reindex(fit.oos.index).fillna(0.0).to_numpy()
    blocks = np.array_split(np.arange(len(m)), PBO_BLOCKS)
    sums = np.array([m[b].sum(axis=0) for b in blocks])
    squares = np.array([(m[b] ** 2).sum(axis=0) for b in blocks])
    counts = np.array([len(b) for b in blocks])

    def sharpe(chosen):
        n = counts[list(chosen)].sum()
        s, q = sums[list(chosen)].sum(axis=0), squares[list(chosen)].sum(axis=0)
        var = (q - s * s / n) / (n - 1)
        return np.where(var > 0, (s / n) / np.sqrt(np.where(var > 0, var, 1.0)), 0.0)

    below, splits = 0, 0
    everything = set(range(PBO_BLOCKS))
    for half in itertools.combinations(range(PBO_BLOCKS), PBO_BLOCKS // 2):
        best = int(np.argmax(sharpe(half)))
        rank = stats.rankdata(sharpe(sorted(everything - set(half))))[best]          # 1: the worst
        below += rank / (len(fit.configs) + 1) < 0.5
        splits += 1
    return {"pbo": below / splits, "configs": len(fit.configs), "splits": splits, "blocks": PBO_BLOCKS}


def eras(oos: pd.Series) -> dict:
    """Minerva's consistency of the record over two-year windows: rho = 0.5 x the share of windows with a positive
    Sharpe + 0.3 x clip(1 - their spread / 2) + 0.2 x logistic(the worst window's Sharpe). The windows start on the
    record's first day; a last one of at least ERA_TAIL_DAYS counts."""
    x = oos.to_numpy()
    windows = [x[k:k + ERA_DAYS] for k in range(0, len(x), ERA_DAYS)]
    windows = [w for w in windows if len(w) == ERA_DAYS or len(w) >= ERA_TAIL_DAYS]
    sharpes = [_sr(w) for w in windows]
    if len(sharpes) < 3:
        return {"too_short": f"{len(sharpes)} two-year window(s) out-of-sample, 3 needed", "sharpes": sharpes}
    s = np.array(sharpes)
    rho = 0.5 * (s > 0).mean() + 0.3 * min(max(1 - s.std(ddof=1) / 2, 0.0), 1.0) + 0.2 / (1 + math.exp(-s.min()))
    return {"rho": float(min(max(rho, 0.0), 1.0)), "sharpes": sharpes, "window_days": ERA_DAYS}


def names(trades: pd.DataFrame) -> dict:
    """The share of the names the strategy traded out-of-sample whose trades made money, compounded."""
    if trades.empty:
        return {"too_short": "no trade out-of-sample"}
    each = np.log1p(trades["net_return"].clip(lower=-0.999999)).groupby(trades["instrument"]).sum()
    return {"instruments": int(len(each)), "positive": int((each > 0).sum()), "share": float((each > 0).mean())}


def cost_multiple(panel: Panel) -> float:
    """How many times its modelled costs a result is charged in the costs check: COST_MULTIPLE, or more where the
    class of an instrument charged its flat cost has a stress cost a side (STRESS_BPS) above it."""
    k = COST_MULTIPLE
    for cls in {ins.asset_class for i, ins in panel.instruments.items() if spreads.spread_path(i) is None}:
        modelled = COSTS[cls].commission_bps + COSTS[cls].half_spread_bps
        if cls in STRESS_BPS and modelled > 0:
            k = max(k, STRESS_BPS[cls] / modelled)
    return k


def _chosen(fit: Fit) -> set[int]:
    """The configurations the windows chose (a window without a choice holds nothing, and has none to replay)."""
    return {c for c in fit.choices if c is not None}


def _replayed(fit: Fit, results: dict[int, object], cost_scale: float = 1.0,
              returns: dict[int, pd.Series] | None = None) -> pd.Series:
    """The out-of-sample record of other backtests of the chosen configurations (`results`, by configuration), on the
    same windows and choices; `returns` replaces a backtest's net returns (the costs check)."""
    daily = pd.DataFrame({c: metrics.daily_returns(returns[c] if returns else r.returns)
                          for c, r in results.items()}).fillna(0.0)
    daily = daily[daily.index >= fit.daily.index.min()]
    if not fit.folds:
        return daily[fit.choices[0]].reindex(fit.oos.index).fillna(0.0)
    return ev._stitched(fit.panel, results, daily, fit.folds, fit.choices, cost_scale)


def costs(fit: Fit) -> dict:
    k = cost_multiple(fit.panel)
    chosen = {c: fit.runs[c] for c in _chosen(fit)}
    stressed = {c: (1 + r.gross) * (1 - k * r.cost) - 1 - r.carry for c, r in chosen.items()}
    return {"multiple": k, **_figures(_replayed(fit, chosen, k, stressed))}


def delay(fit: Fit) -> dict:
    late = {c: ev._run(fit.strategy, fit.panel, fit.configs[c], fit.member, fit.fill, fit.memo, delay=1,
                       signals=fit.signals) for c in _chosen(fit)}
    return _figures(_replayed(fit, late))


def _sharpe_over(fit: Fit, daily: pd.Series) -> float:
    return _sr(daily.reindex(fit.oos.index).fillna(0.0).to_numpy())


def _over(fit: Fit, daily: pd.Series) -> dict:
    """Sharpe and compounded return of a configuration's record over the out-of-sample days."""
    return _figures(daily.reindex(fit.oos.index).fillna(0.0))


def plateau(fit: Fit) -> dict:
    """The configuration the windows chose most (the latest of those chosen as often) and the Sharpe over the
    out-of-sample days of each configuration one grid step away along one parameter."""
    if len(fit.configs) < 2:
        return {"na": "one configuration: no neighbours"}
    chosen = [c for c in fit.choices if c is not None]
    counts = pd.Series(chosen).value_counts()
    top = [c for c in reversed(chosen) if counts[c] == counts.max()][0]
    here = fit.configs[top]
    near = []
    for j, cfg in enumerate(fit.configs):
        moved = [k for k in here if cfg[k] != here[k]]
        if len(moved) == 1:
            values = list(fit.strategy.grid[moved[0]])
            if abs(values.index(cfg[moved[0]]) - values.index(here[moved[0]])) == 1:
                near.append({"params": {moved[0]: cfg[moved[0]]}, **_over(fit, fit.daily[j])})
    if not near:
        return {"na": "no configuration one step away"}
    return {"config": here, "sharpe": _sharpe_over(fit, fit.daily[top]), "neighbours": near}


def _has_model(s: Strategy) -> bool:
    return s.grade is not None or s.prepare is not None


@contextmanager
def _model_seed(s: Strategy, seed: int):
    """The strategy's model draws with `seed` inside the block: a grade's model (`trade_model`), or the model the
    strategy's own module fits (ml_direction)."""
    mod = trade_model if s.grade is not None else sys.modules[s.fn.__module__]
    before = mod.SEED
    mod.SEED = seed
    try:
        yield
    finally:
        mod.SEED = before


def seeds(fit: Fit) -> dict:
    s = fit.strategy
    if not _has_model(s):
        return {"na": "no model"}
    sharpes, cagrs = [], []
    for seed in SEEDS:
        with _model_seed(s, seed):
            undo = s.prepare(fit.panel, fit.configs, fit.member) if s.prepare is not None else None
            try:
                memo: dict = {}                 # the positions kept by configuration do not know the seed
                runs = {c: ev._run(s, fit.panel, fit.configs[c], fit.member, fit.fill, memo, signals=fit.signals)
                        for c in _chosen(fit)}
            finally:
                if undo is not None:
                    undo()
        got = _figures(_replayed(fit, runs))
        sharpes.append(got["sharpe"])
        cagrs.append(got["cagr"])
    return {"seeds": list(SEEDS), "sharpes": sharpes, "cagrs": cagrs}


def _chosen_days(fit: Fit) -> pd.DatetimeIndex:
    """The out-of-sample days of the windows a configuration was chosen for (all of them for a single one)."""
    if not fit.folds:
        return fit.oos.index
    days = fit.oos.index
    keep = np.zeros(len(days), dtype=bool)
    for f, c in zip(fit.folds, fit.choices):
        if c is not None:
            keep |= (days >= f.test_start) & (days < f.test_end)
    return days[keep]


def vs_rule(fit: Fit) -> dict:
    """The graded rule against the same rule without its model, on the same windows' choices, over the windows a
    configuration was chosen for: before its model has trades to learn from (its first windows), the graded rule
    has nothing to choose on and holds nothing, and the model's worth is measured where it grades."""
    s = fit.strategy
    if s.grade is None:
        return {"na": "no rule under the model" if s.prepare is not None else "no model"}
    keys = s.grade_keys()
    plain = dataclasses.replace(s, grade=None, grid={k: v for k, v in s.grid.items() if k not in keys})
    memo: dict = {}                             # not the graded rule's: a sizing grade keeps the same keys
    runs = {c: ev._run(plain, fit.panel, {k: v for k, v in fit.configs[c].items() if k not in keys}, fit.member,
                       fit.fill, memo, signals=fit.signals) for c in _chosen(fit)}
    days = _chosen_days(fit)
    out = paired_t(fit.oos.reindex(days), _replayed(fit, runs).reindex(days))
    if "sharpe_other" in out:
        out["sharpe_plain"] = out.pop("sharpe_other")
    return out


def neighbour_lists(fit: Fit) -> dict:
    """The walk-forward of the same strategy on the Top-N lists next to this one, N +- d with d = 20% of N rounded to
    5, at least 5 and at most half of N (Top-2 and Top-4 around Top-3): the result must not depend on which names sit
    at the edge of the list. Read around this list's panel and evaluated here, never saved: they are not research
    results."""
    m = TOP_N.match(fit.universe)
    if not m:
        return {"na": "not a Top-N list"}
    n = int(m["n"])
    d = min(max(5, int(round(n * 0.2 / 5)) * 5), n // 2)
    s, out = fit.strategy, []
    for u in (f"{m['base']}{n - d}", f"{m['base']}{n + d}"):
        near, panel, member = ev._load(u, fit.timeframe, fit.start, fit.end, keep=False, base=fit.panel)
        configs = s.configs(near.capacity)          # the grid that list runs on its own: its own slot counts
        undo = s.prepare(panel, configs, member) if s.prepare is not None else None
        try:
            *_, oos, _ = ev._walk_forward(s, panel, member, fit.fill, configs, fit.timeframe, keep_all=False,
                                          memo={}, signals=fit.signals)
        finally:
            if undo is not None:
                undo()
        out.append({"universe": u, "start": str(oos.index.min().date()), **_figures(oos)})
        del panel, member, oos                  # a Top-120 list's bars on 1h are gigabytes: gone before the next
        gc.collect()
    return {"lists": out}
