"""One-minute bars, kept for the engine to walk intrabar exits through: a stop, a target or a trailing stop is checked at
every minute inside a bar, and a trailing stop's best price moves after each minute, where the store has the
instrument's minutes (`engine.backtest`).

A series is two arrays beside its meta, memory-mapped when read, so that every process of a batch shares one copy of a
list's minutes in the page cache: the minutes' close times (int64 nanoseconds, UTC, strictly increasing) and their
open, high and low (float32, one row a minute). Only minutes with all three prices are kept.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_lab.config import STORE_DIR
from strategy_lab.data.instruments import parse
from strategy_lab.data.store import safe_name

ROOT = STORE_DIR                   # tests point it elsewhere
TIMEFRAME = "1m"


def _base(instrument_id: str) -> Path:
    ins = parse(instrument_id)
    return ROOT / ins.source / TIMEFRAME / safe_name(ins.symbol)


def paths(instrument_id: str) -> tuple[Path, Path, Path]:
    """The close times, the prices and the meta of an instrument's minutes."""
    base = _base(instrument_id)
    return base.with_suffix(".time.npy"), base.with_suffix(".ohl.npy"), base.with_suffix(".meta.json")


def write(instrument_id: str, bars: pd.DataFrame, meta: dict) -> int:
    """Keep an instrument's minutes (open, high, low; indexed by close time, tz-aware), replacing what was kept; the
    two arrays are written atomically, the prices before the times, so a reader never pairs new times with old
    prices of another length. Returns the minutes kept."""
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError(f"{instrument_id}: minutes must be indexed by tz-aware close times")
    bars = bars[["open", "high", "low"]].dropna()
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError(f"{instrument_id}: minutes must be strictly increasing in time")
    if ((bars["high"] < bars[["open", "low"]].max(axis=1)) | (bars["low"] > bars["open"])).any():
        raise ValueError(f"{instrument_id}: a minute's high is below its open or low, or its low above its open")
    t_path, p_path, m_path = paths(instrument_id)
    t_path.parent.mkdir(parents=True, exist_ok=True)
    times = bars.index.tz_convert("UTC").asi8.astype(np.int64)
    prices = np.ascontiguousarray(bars.to_numpy(dtype=np.float32))
    for target, array in ((p_path, prices), (t_path, times)):
        tmp = target.with_name(target.name + f".tmp{os.getpid()}")
        with open(tmp, "wb") as fh:
            np.save(fh, array)
        os.replace(tmp, target)
    body = {**meta, "minutes": len(bars), "first": str(bars.index[0]) if len(bars) else None,
            "last": str(bars.index[-1]) if len(bars) else None}
    m_tmp = m_path.with_name(m_path.name + f".tmp{os.getpid()}")
    m_tmp.write_text(json.dumps(body, indent=1, sort_keys=True, default=str))
    os.replace(m_tmp, m_path)
    return len(bars)


def read(instrument_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    """An instrument's minutes as kept, memory-mapped read-only: (close times, prices of shape (n, 3): open, high,
    low); None when the store has none."""
    t_path, p_path, _ = paths(instrument_id)
    if not (t_path.exists() and p_path.exists()):
        return None
    times = np.load(t_path, mmap_mode="r")
    prices = np.load(p_path, mmap_mode="r")
    if len(times) != len(prices):
        raise ValueError(f"{instrument_id}: {len(times)} minute close times against {len(prices)} rows of prices")
    return times, prices


def read_meta(instrument_id: str) -> dict:
    p = paths(instrument_id)[2]
    return json.loads(p.read_text()) if p.exists() else {}
