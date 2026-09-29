"""Sharadar's tables as the store keeps them (data/raw/sharadar, `refresh sharadar-tables`): the S&P 500's membership,
every security it covers, and the corporate actions. No network.

Sharadar keys a security by its ticker today; a company that no longer trades keeps its last ticker, with a suffix
when the ticker was taken up again (WB1 is Wachovia, WB is Weibo; LEHMQ, ENRNQ, WCOEQ), so its membership spans and
its prices (`sh:<ticker>`) name the same company.
"""
from __future__ import annotations

import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd

from strategy_lab.config import DATA_DIR

DIR = DATA_DIR / "raw" / "sharadar"


def table(name: str) -> pd.DataFrame:
    path = DIR / f"{name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m strategy_lab.data.refresh sharadar-tables`")
    with zipfile.ZipFile(path) as z:
        return pd.read_csv(z.open(z.namelist()[0]), low_memory=False)


def actions() -> pd.DataFrame:
    """Sharadar's corporate actions, read once per version of the file (the frame is shared: filter it, do not change
    it)."""
    path = DIR / "actions.zip"
    return _table(str(path), "actions", path.stat().st_mtime_ns if path.exists() else 0)


@lru_cache(maxsize=2)
def _table(path: str, name: str, version: int) -> pd.DataFrame:
    return table(name)


def split_factor(ticker: str, days: pd.DatetimeIndex) -> np.ndarray:
    """What puts a price printed on each of the days (New York dates) on the split-adjusted basis of Sharadar's prices:
    one over every split of the ticker after that day, from its `split` actions, the new shares per old one (AAPL 4.0
    on 2020-08-31, USO 0.125 on 2020-04-29)."""
    a = actions()
    splits = a[(a["action"] == "split") & (a["ticker"] == ticker)]
    f = np.ones(len(days))
    for d, v in zip(pd.to_datetime(splits["date"]), splits["value"]):
        f[np.asarray(days < d)] /= float(v)
    return f


def sp500_spans() -> pd.DataFrame:
    """The S&P 500's membership spans (ticker, start, end; end NaT: a member today): each ticker's added and removed
    events paired, a removal with no addition on record (ENRNQ) a member from the table's first date (1957-03-04). Its
    quarterly snapshots agree with the spans at all 114 quarter-ends (1998-03-31 to 2026-06-30, checked 2026-09-26)."""
    path = DIR / "sp500.zip"
    return _spans(str(path), path.stat().st_mtime_ns if path.exists() else 0).copy()


@lru_cache(maxsize=2)
def _spans(path: str, version: int) -> pd.DataFrame:
    sp = table("sp500")
    sp["date"] = pd.to_datetime(sp["date"], utc=True)
    spans, first = [], sp["date"].min()
    for t, g in sp[sp["action"].isin(["added", "removed"])].sort_values(["ticker", "date"]).groupby("ticker"):
        start = None
        for d, action in zip(g["date"], g["action"]):
            if action == "added":
                start = d if start is None else start
            else:
                spans.append((t, start if start is not None else first, d))
                start = None
        if start is not None:
            spans.append((t, start, pd.NaT))
    return pd.DataFrame(spans, columns=["ticker", "start", "end"])
