"""Is an out-of-sample record more than luck? Two checks, measured on the record itself.

* `deflated` — a record picked from many tries (strategies x lists x timeframes, or strategy x instrument x timeframe
  pairs) looks good by chance alone: the best of N worthless tries has a positive Sharpe. `noise_bar` is the Sharpe
  that best would reach on a record of the same length (the expected maximum of N draws, Bailey & Lopez de Prado,
  "The Deflated Sharpe Ratio", 2014), and `prob` is the probability that the record's true Sharpe is above it, with
  the record's skew and kurtosis (their deflated Sharpe ratio). N counts every evaluation made; tries that resemble
  each other are fewer independent chances, so with N counted in full the bar is the strict one.
* `random_timing` — the record's positions against the same positions moved in time. Each instrument's position path
  is rotated within the bars it printed, every instrument by the same fraction of its own path, so the time in the
  market, the holding periods and the turnover stay as they were and only the timing becomes random. The p-value is
  the share of rotations whose Sharpe is at least the record's. The record and its rotations are valued bar by bar the
  same simplified way (the gap on the previous bar's weight, the session on the new one, the trading cost on the change
  of weight, short borrow and perp funding; no intrabar exits and no drift between fills), so the comparison is like for
  like even where that valuation differs slightly from the engine's.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy import stats

from strategy_lab import log
from strategy_lab.config import SEED
from strategy_lab.data.bars import Panel
from strategy_lab.engine import backtest as bt
from strategy_lab.metrics import DAYS

LOG = log.get("significance")
PASS_PROB = 0.95           # a record counts as more than luck when its deflated Sharpe probability reaches this
PASS_P = 0.05              # ... and when random timing does as well in fewer than this share of rotations
N_ROTATIONS = 500
MIN_DAYS = 30


@functools.lru_cache(maxsize=4096)
def expected_max(n_trials: int) -> float:
    """Expected maximum of `n_trials` independent standard normal draws (0 for a single try)."""
    if n_trials <= 1:
        return 0.0
    g = np.euler_gamma
    return float((1 - g) * stats.norm.ppf(1 - 1 / n_trials) + g * stats.norm.ppf(1 - 1 / (n_trials * np.e)))


def deflated(daily: pd.Series, n_trials: int) -> dict:
    """The Sharpe the best of `n_trials` worthless tries reaches on a record this long (annualised, as
    `metrics.core`), and the probability that this record's true Sharpe is above it."""
    return deflated_each(daily, [n_trials])[0]


def deflated_each(daily: pd.Series, trials: list[float]) -> list[dict]:
    """`deflated` of one record against each count of tries in `trials`, its moments worked out once (its skew and
    kurtosis were most of the time of judging a record twice)."""
    d = daily.dropna().to_numpy(dtype=np.float64)
    n = len(d)
    if n < MIN_DAYS:
        LOG.info("deflated Sharpe: %d days are too few to judge (at least %d)", n, MIN_DAYS)
        return [{"noise_bar": np.nan, "prob": np.nan, "n_trials": k} for k in trials]
    sd = d.std(ddof=1)
    bars = [expected_max(k) / np.sqrt(n - 1) for k in trials]    # daily Sharpe of the best null try: sd x max z
    if not sd > 0:
        return [{"noise_bar": float(bar * np.sqrt(DAYS)), "prob": 0.0, "n_trials": k} for k, bar in zip(trials, bars)]
    sr = d.mean() / sd
    skew = float(stats.skew(d, bias=False))
    kurt = float(stats.kurtosis(d, fisher=True, bias=False))
    var = (1 - skew * sr + (kurt + 2) / 4 * sr ** 2) / (n - 1)
    return [{"noise_bar": float(bar * np.sqrt(DAYS)),
             "prob": float(stats.norm.cdf((sr - bar) / np.sqrt(var))) if var > 0 else float(sr > bar), "n_trials": k}
            for k, bar in zip(trials, bars)]


def random_timing(panel: Panel, weights: pd.DataFrame, days: pd.DatetimeIndex, *, n: int = N_ROTATIONS,
                  seed: int = SEED) -> dict:
    """The record held `weights` (the weight of each bar's session, as the engine reports it) over the UTC days
    `days`; returns its simply-valued Sharpe, the median and 95th percentile of the rotations' Sharpe, and the
    p-value. Empty when it held nothing on those days."""
    if len(days) < MIN_DAYS:
        LOG.info("random timing: %d days are too few to judge (at least %d)", len(days), MIN_DAYS)
        return {}
    idx = panel.index
    bar_day = (idx - pd.Timedelta(microseconds=1)).tz_convert("UTC").normalize()
    lo, hi = int(bar_day.searchsorted(days.min(), "left")), int(bar_day.searchsorted(days.max(), "right"))
    w = weights.reindex(index=idx[lo:hi], columns=panel.ids).fillna(0.0)
    held = [i for i in panel.ids if (w[i] != 0).any()]
    if not held:
        LOG.info("random timing: the record held no position on its days")
        return {}
    col = [panel.ids.index(i) for i in held]
    terms = bt._terms(panel)
    gap, intra, funding = terms.gap, terms.intra, bt.funding_held(panel)
    rates, borrow_rate = terms.rates, terms.borrow_rate
    dt = terms.dt_years[lo:hi].astype(np.float64)
    printed = panel.close[held].iloc[lo:hi].notna().to_numpy()
    starts, lens = np.zeros(len(held), dtype=np.int64), np.zeros(len(held), dtype=np.int64)
    for k in range(len(held)):
        rows = np.flatnonzero(printed[:, k])
        starts[k], lens[k] = rows[0], rows[-1] - rows[0] + 1
    day_of = days.searchsorted(bar_day[lo:hi]).astype(np.int64)
    by_instrument = lambda f: np.ascontiguousarray(f.to_numpy(dtype=np.float64).T)      # noqa: E731
    args = (by_instrument(w[held]), by_instrument(gap[held].iloc[lo:hi]), by_instrument(intra[held].iloc[lo:hi]),
            by_instrument(funding[held].iloc[lo:hi]), rates.flat[col].astype(np.float64),
            rates.column[col].astype(np.int64), np.ascontiguousarray(rates.at_open[lo:hi].T),
            borrow_rate[col].astype(np.float64), dt, day_of, len(days), starts, lens)
    observed = float(_valued(*args, np.zeros(1), False)[0])
    null = _valued(*args, np.random.default_rng(seed).random(n), True)
    return {"sharpe": observed, "null_median": float(np.median(null)), "null_p95": float(np.quantile(null, 0.95)),
            "p": float((1 + np.sum(null >= observed)) / (1 + n)), "rotations": n}


@njit(parallel=True, cache=True)
def _valued(w, gap, intra, funding, flat_rate, rate_column, quoted_rate, borrow_rate, dt, day_of, n_days, starts, lens,
            fractions, rotate):
    """Annualised Sharpe of the daily returns of each rotation (one per fraction; unrotated when `rotate` is off). The
    arrays are laid out instrument by instrument (instruments x bars), so that a rotation walks each instrument's path
    through contiguous memory: laid out bar by bar, the same walk took 2 to 11 times as long. A change of weight pays
    its bar's rate at the open where the instrument has a column of quoted rates (`rate_column`), its flat rate
    otherwise."""
    n_bars = w.shape[1]
    out = np.empty(len(fractions))
    for p in prange(len(fractions)):
        port = np.zeros(n_bars)
        for k in range(w.shape[0]):
            a, length = starts[k], lens[k]
            shift = 0
            if rotate and length > 1:
                shift = 1 + int(fractions[p] * (length - 1))
            prev = 0.0
            q = rate_column[k]
            for t in range(a, a + length):
                x = w[k, a + (t - a + shift) % length]
                rate = quoted_rate[q, t] if q >= 0 else flat_rate[k]
                port[t] += (prev * gap[k, t] + x * intra[k, t] - abs(x - prev) * rate - x * funding[k, t]
                            - max(-x, 0.0) * borrow_rate[k] * dt[t])
                prev = x
        logs = np.zeros(n_days)
        for t in range(n_bars):
            logs[day_of[t]] += np.log1p(max(port[t], -0.999999))
        daily = np.expm1(logs)
        mean = daily.sum() / n_days
        sd = np.sqrt(((daily - mean) ** 2).sum() / (n_days - 1))
        out[p] = mean / sd * np.sqrt(DAYS) if sd > 0 else 0.0
    return out
