"""Monte Carlo of the out-of-sample record: stationary block bootstrap of daily returns.

Blocks keep the serial dependence (volatility clustering, streaks) that a day-by-day shuffle would destroy;
the block length is chosen from the data (Politis-White). Each resampled path has the original length and is
scored with the same figures as the real one: `metrics.core`'s arithmetic, compiled (`_figures`), since a thousand
resamples through pandas took seconds a record.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from numba import njit

from strategy_lab import metrics
from strategy_lab.config import FIRM_TARGETS, MC_REPS, SEED

FIGURES = ("avg_monthly", "pct_green_active", "max_dd", "sharpe", "worst_month")


def bootstrap(daily: pd.Series, reps: int = MC_REPS, seed: int = SEED) -> dict:
    d = daily.dropna()
    if len(d) < 90:
        return {"reps": 0, "note": "fewer than 90 days: no Monte Carlo"}
    block = float(optimal_block_length(d.to_numpy())["stationary"].iloc[0])
    block = max(1.0, min(block, len(d) / 4))
    bs = StationaryBootstrap(block, d.to_numpy(), seed=np.random.default_rng(seed))
    first, last = _complete_months(d.index)
    rows = np.empty((reps, len(FIGURES)))
    for k, ((arr,), _) in enumerate(bs.bootstrap(reps)):
        rows[k] = _figures(np.ascontiguousarray(arr, dtype=np.float64), first, last, metrics.DAYS)
    a = pd.DataFrame(rows, columns=FIGURES)
    out = {"reps": reps, "block_days": round(block, 1)}
    for f in FIGURES:
        out[f] = {"p5": float(a[f].quantile(0.05)), "p50": float(a[f].quantile(0.5)), "p95": float(a[f].quantile(0.95))}
    out["prob_avg_monthly_below_target"] = float((a["avg_monthly"] < FIRM_TARGETS["avg_monthly"]).mean())
    out["prob_max_dd_beyond_target"] = float((a["max_dd"] < FIRM_TARGETS["max_dd"]).mean())
    return out


def _complete_months(days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """The first and one-past-last position of each complete calendar month of `days` (`metrics.monthly_returns`'
    months: a partial month at either end left out)."""
    codes = days.tz_localize(None).to_period("M").asi8
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)]
    if days.min().day != 1:
        starts, ends = starts[1:], ends[1:]
    last = days.max()
    if last.day != last.days_in_month:
        starts, ends = starts[:-1], ends[:-1]
    return starts.astype(np.int64), ends.astype(np.int64)


@njit(cache=True, error_model="numpy")
def _figures(x, first, last, days_per_year):
    """FIGURES of the daily returns `x` as `metrics.core` gives them, to the last bit: its mean and standard deviation
    are pandas', whose sums are numpy's (`_numpy_sum`); a product runs day after day as pandas' does; the months are the
    complete ones, from `first` to `last` (`_complete_months`)."""
    n = len(x)
    mean = _numpy_sum(x) / n
    dev = np.empty(n)
    for i in range(n):
        dev[i] = (mean - x[i]) * (mean - x[i])
    sd = np.sqrt(_numpy_sum(dev) / (n - 1)) if n > 1 else np.nan
    sharpe = mean / sd * np.sqrt(days_per_year) if n > 1 and sd > 0 else 0.0
    equity, peak, deepest = 1.0, -np.inf, 0.0
    for i in range(n):
        equity *= 1.0 + x[i]
        peak = max(peak, equity)
        deepest = min(deepest, equity / peak - 1.0)
    worst, green, active = np.inf, 0, 0
    for k in range(len(first)):
        grew = 1.0
        for i in range(first[k], last[k]):
            grew *= 1.0 + x[i]
        month = grew - 1.0
        worst = min(worst, month)
        if month != 0.0:
            active += 1
            if month > 0.0:
                green += 1
    months = n / days_per_year * 12.0
    if len(first) == 0 or months <= 0:
        average = np.nan
    elif equity > 0:
        average = equity ** (1.0 / months) - 1.0
    else:
        average = -1.0
    out = np.empty(5)
    out[0] = average
    out[1] = green / active if active > 0 else np.nan
    out[2] = deepest
    out[3] = sharpe
    out[4] = worst if len(first) > 0 else np.nan
    return out


@njit(cache=True)
def _numpy_sum(a):
    """numpy's (1.26) sum of a contiguous float64 array, to the last bit: blocks of 8192 added in turn, each summed
    pairwise (`_pairwise`)."""
    total = 0.0
    for lo in range(0, len(a), 8192):
        total += _pairwise(a, lo, min(8192, len(a) - lo))
    return total


@njit(cache=True)
def _pairwise(a, lo, n):
    """numpy's pairwise sum of a[lo:lo + n]: eight running sums below 129 values, halves (at a multiple of 8) above."""
    if n < 8:
        res = -0.0
        for i in range(lo, lo + n):
            res += a[i]
        return res
    if n <= 128:
        r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
        r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
        i = 8
        while i < n - n % 8:
            r0 += a[lo + i]
            r1 += a[lo + i + 1]
            r2 += a[lo + i + 2]
            r3 += a[lo + i + 3]
            r4 += a[lo + i + 4]
            r5 += a[lo + i + 5]
            r6 += a[lo + i + 6]
            r7 += a[lo + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[lo + i]
            i += 1
        return res
    half = n // 2
    half -= half % 8
    return _pairwise(a, lo, half) + _pairwise(a, lo + half, n - half)
