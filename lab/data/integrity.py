"""Bar-level quality checks. Invalid bars are removed and counted, never repaired."""
from __future__ import annotations

import pandas as pd


def clean_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop impossible bars. Returns the clean frame and the count removed per reason."""
    counts: dict[str, int] = {}
    ohlc = df[["open", "high", "low", "close"]]
    bad_nan = ohlc.isna().any(axis=1)
    bad_price = (ohlc <= 0).any(axis=1) & ~bad_nan
    bad_range = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    bad_range &= ~bad_nan & ~bad_price
    for name, mask in (("nan_price", bad_nan), ("non_positive_price", bad_price), ("high_low_inconsistent", bad_range)):
        if mask.any():
            counts[name] = int(mask.sum())
    keep = ~(bad_nan | bad_price | bad_range)
    out = df[keep]
    if out.index.has_duplicates:
        counts["duplicate_close_time"] = int(out.index.duplicated().sum())
        out = out[~out.index.duplicated(keep="last")]
    return out.sort_index(), counts


def stale_tail_days(df: pd.DataFrame, asof: pd.Timestamp) -> float:
    """Days between the last bar and `asof` — how far behind the store is."""
    if df.empty:
        return float("inf")
    return (asof - df.index.max()) / pd.Timedelta(days=1)
