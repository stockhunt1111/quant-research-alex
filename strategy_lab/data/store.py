"""On-disk bar store: one parquet per (source, timeframe, symbol), indexed by bar close time (UTC).

Next to each parquet sits a small JSON with what is known about coverage (first/last bar, the last moment the
vendor was asked, and any recorded absence), so a refresh never asks twice for something that is not there.
Writes are atomic (temp file + rename).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from strategy_lab.config import STORE_DIR

BAR_COLUMNS = ["open", "high", "low", "close", "volume", "dollar_volume"]


def safe_name(symbol: str) -> str:
    return symbol.replace("/", "-")


def path(source: str, timeframe: str, symbol: str) -> Path:
    return STORE_DIR / source / timeframe / f"{safe_name(symbol)}.parquet"


def meta_path(source: str, timeframe: str, symbol: str) -> Path:
    return path(source, timeframe, symbol).with_suffix(".meta.json")


def _atomic_write_bytes(target: Path, write) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".tmp{os.getpid()}")
    write(tmp)
    os.replace(tmp, target)


def write_bars(source: str, timeframe: str, symbol: str, df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError(f"{source}:{symbol} {timeframe}: index must be tz-aware close times")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError(f"{source}:{symbol} {timeframe}: index must be strictly increasing")
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source}:{symbol} {timeframe}: missing columns {missing}")
    out = df[BAR_COLUMNS].astype("float64")
    out.index.name = "close_time"
    _atomic_write_bytes(path(source, timeframe, symbol), lambda p: out.to_parquet(p))


def read_bars(source: str, timeframe: str, symbol: str, columns: list[str] | None = None) -> pd.DataFrame:
    p = path(source, timeframe, symbol)
    if not p.exists():
        raise FileNotFoundError(f"no stored bars for {source}:{symbol} {timeframe} ({p})")
    return pd.read_parquet(p, columns=columns)


def read_meta(source: str, timeframe: str, symbol: str) -> dict:
    p = meta_path(source, timeframe, symbol)
    return json.loads(p.read_text()) if p.exists() else {}


def write_meta(source: str, timeframe: str, symbol: str, meta: dict) -> None:
    body = json.dumps(meta, indent=1, sort_keys=True, default=str)
    _atomic_write_bytes(meta_path(source, timeframe, symbol), lambda p: p.write_text(body))


def quarantine(source: str, timeframe: str, symbol: str, reason: dict) -> Path:
    """Move a series out of the store, with its meta and the reason, into data/quarantine: it is no longer offered to
    any universe, and nothing is lost."""
    target = STORE_DIR.parent / "quarantine" / source / timeframe
    target.mkdir(parents=True, exist_ok=True)
    os.replace(path(source, timeframe, symbol), target / path(source, timeframe, symbol).name)
    meta = {**read_meta(source, timeframe, symbol), "quarantined": reason}
    (target / meta_path(source, timeframe, symbol).name).write_text(json.dumps(meta, indent=1, sort_keys=True, default=str))
    meta_path(source, timeframe, symbol).unlink(missing_ok=True)
    return target


def symbols(source: str, timeframe: str) -> list[str]:
    folder = STORE_DIR / source / timeframe
    if not folder.exists():
        return []
    return sorted(p.stem for p in folder.glob("*.parquet"))
