"""The spread a broker quoted on the five spot quotes of commodities, hour by hour (`spread_refresh` keeps it): what the
engine charges a fill on them (`engine.costs`) in place of a flat cost of their class.

A quote's hours hold the spread (ask - bid, in the quote's currency) at the hour's first and last quote. A bar of any
timeframe takes its first hour's opening spread as its open's, its last hour's closing spread as its close's, and the
median of its hours' spreads (each hour's opening and closing ones averaged) as what was quoted inside it, where a stop
or a target fills. A bar without an hour of its own (the broker's daily pause, where the vendor prints one) takes the
last spread quoted before it. A bar before the first hour on record takes what the first year quoted at the same hours
of the day (UTC), at the median: the brokers set these spreads in dollars and seldom move them (Dukascopy's platinum
$2.66 in both 2024-06 and 2026-08, crude's $0.05 at $40 and at $110), so a dollar spread carries back where one in
basis points would shrink with the price.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from strategy_lab.config import STORE_DIR
from strategy_lab.data import store
from strategy_lab.data.instruments import parse

PROFILE_DAYS = 365                  # the first year on record: its median spread at each hour of the day carries back
HOUR = pd.Timedelta(hours=1).value
BAR = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}


def spread_path(instrument_id: str) -> Path | None:
    """Where a quote's hours of spreads are kept (None: an instrument charged its class's flat cost)."""
    ins = parse(instrument_id)
    if ins.source == "td" and ins.asset_class == "commodity":
        return STORE_DIR / "td" / "spread" / f"{store.safe_name(ins.symbol)}.parquet"
    return None


def load_spreads(instrument_id: str) -> pd.DataFrame | None:
    """A quote's hours of spreads (`open`, `close`, stamped at the hour's close), None for an instrument charged its
    class's flat cost. Read once per process and version of the file."""
    p = spread_path(instrument_id)
    if p is None:
        return None
    if not p.exists():
        raise FileNotFoundError(f"no spreads stored for {instrument_id} (spread_refresh build): a spot quote without "
                                "its broker's spreads would be filled at no cost")
    return _read(str(p), p.stat().st_mtime_ns)


def version(instrument_id: str) -> int | None:
    """When a quote's spreads were last written (None: an instrument charged its class's flat cost, or none stored)."""
    p = spread_path(instrument_id)
    return p.stat().st_mtime_ns if p is not None and p.exists() else None


@functools.lru_cache(maxsize=None)
def _read(path: str, version: int) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["open", "close"]).sort_index()


def per_bar(instrument_id: str, closes: pd.DatetimeIndex, timeframe: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The spread quoted at the open of each bar closing at `closes`, inside it and at its close (the module's rules),
    in the quote's currency. A bar opens where the one before it closes; the first one a bar of `timeframe` earlier."""
    hours = load_spreads(instrument_id)
    if hours.empty:
        raise ValueError(f"{instrument_id}: its spreads file holds no hour (spread_refresh build)")
    h = hours.index.asi8
    o, c = hours["open"].to_numpy(float), hours["close"].to_numpy(float)
    ends = closes.asi8
    starts = np.r_[ends[0] - BAR[timeframe].value, ends[:-1]] if len(ends) else ends
    first_year = h < h[0] + pd.Timedelta(days=PROFILE_DAYS).value
    of_day = ((h[first_year] - HOUR) // HOUR) % 24
    prof_o = _by_hour_of_day(o[first_year], of_day)
    prof_c = _by_hour_of_day(c[first_year], of_day)
    return _per_bar(h, o, c, starts, ends, prof_o, prof_c)


def _by_hour_of_day(values: np.ndarray, of_day: np.ndarray) -> np.ndarray:
    """The median of `values` at each hour of the day (UTC); an hour the year never quoted takes the year's median."""
    out = np.full(24, np.nanmedian(values) if len(values) else np.nan)
    for k in range(24):
        v = values[of_day == k]
        if len(v):
            out[k] = np.median(v)
    return out


@njit(cache=True)
def _per_bar(h, o, c, starts, ends, prof_o, prof_c):
    n = len(ends)
    at_open, during, at_close = np.empty(n), np.empty(n), np.empty(n)
    for k in range(n):
        j = np.searchsorted(h, starts[k], side="right")          # the first hour closing inside the bar
        i = np.searchsorted(h, ends[k], side="right") - 1         # the last one
        if j < len(h) and i >= j:
            at_open[k], at_close[k] = o[j], c[i]
            during[k] = np.median((o[j:i + 1] + c[j:i + 1]) / 2.0)
        elif j > 0:                                               # none: the last spread quoted before it
            at_open[k] = at_close[k] = during[k] = c[j - 1]
        else:                                                     # before the record: the first year's, by hour of day
            first = ((starts[k] // 3_600_000_000_000) % 24 + 24) % 24
            last = (((ends[k] - 3_600_000_000_000) // 3_600_000_000_000) % 24 + 24) % 24
            at_open[k], at_close[k] = prof_o[first], prof_c[last]
            span = min(24, max(1, (ends[k] - starts[k]) // 3_600_000_000_000))
            mids = np.empty(span)
            for x in range(span):
                hr = (first + x) % 24
                mids[x] = (prof_o[hr] + prof_c[hr]) / 2.0
            during[k] = np.median(mids)
    return at_open, during, at_close
