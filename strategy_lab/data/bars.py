"""Read bars from the store: one instrument (`load_bars`) or an aligned panel (`load_panel`).

Every index is the bar close time in UTC.
"""
from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_lab import log
from strategy_lab.config import STORE_DIR, TIMEFRAMES
from strategy_lab.data import store
from strategy_lab.data.instruments import Instrument, parse

FIELDS = ("open", "high", "low", "close", "volume", "dollar_volume")
LOG = log.get("bars")


def load_bars(instrument_id: str, timeframe: str, start=None, end=None, fields=FIELDS) -> pd.DataFrame:
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {TIMEFRAMES}, got {timeframe!r}")
    ins = parse(instrument_id)
    df = store.read_bars(ins.source, timeframe, ins.symbol, list(fields))
    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df


def stored_version(instrument_ids: list[str], timeframe: str) -> int:
    """When these instruments' stored bars or their meta were last written: a panel read before then is out of date."""
    latest = 0
    for i in instrument_ids:
        ins = parse(i)
        for p in (store.path(ins.source, timeframe, ins.symbol), store.meta_path(ins.source, timeframe, ins.symbol)):
            if p.exists():
                latest = max(latest, p.stat().st_mtime_ns)
    return latest


def funding_path(instrument_id: str) -> Path | None:
    """Where a perp's funding settlements are kept (None: not a perp)."""
    ins = parse(instrument_id)
    return STORE_DIR / "perp" / "funding" / f"{store.safe_name(ins.symbol)}.parquet" if ins.source == "perp" else None


def load_funding(instrument_id: str) -> pd.Series | None:
    """Perp funding settlements (rate per settlement, positive = longs pay). None for non-perps.

    A backtest asks for every perp's funding on every run: the file is read once per process and version of it, and
    each call gets its own copy."""
    p = funding_path(instrument_id)
    if p is None:
        return None
    if not p.exists():
        raise FileNotFoundError(f"no funding stored for {instrument_id}; a perp without funding would be free carry")
    return _read_funding(str(p), p.stat().st_mtime_ns).copy()


@functools.lru_cache(maxsize=None)
def _read_funding(path: str, version: int) -> pd.Series:
    s = pd.read_parquet(path)["rate"]
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC") if s.index.tz is not None else pd.DatetimeIndex(s.index).tz_localize("UTC")
    return s.sort_index()


def dividends_path(source: str, symbol: str) -> Path:
    """Where a US stock's or ETF's cash dividends are kept, by the source of its bars (`refresh dividends` for
    Twelve Data's, `refresh sharadar-stocks` for Sharadar's)."""
    return STORE_DIR / source / "dividends" / f"{store.safe_name(symbol)}.parquet"


def _pays_dividends(instrument_id: str) -> bool:
    return parse(instrument_id).asset_class == "us_equity"


def dividends_version(instrument_id: str) -> int | None:
    """When a stock's or ETF's dividends were last written (None: not one, or none stored)."""
    if not _pays_dividends(instrument_id):
        return None
    ins = parse(instrument_id)
    p = dividends_path(ins.source, ins.symbol)
    return p.stat().st_mtime_ns if p.exists() else None


def load_dividends(instrument_id: str) -> pd.Series | None:
    """A US stock's or ETF's cash dividends: amount per share by ex-date (a date), on the split-adjusted basis of its
    bars. None for other instruments. One whose dividends were never fetched is paid none, with a warning: its bars
    are not adjusted for them, so a holder of it earns less here than on the market."""
    if not _pays_dividends(instrument_id):
        return None
    ins = parse(instrument_id)
    p = dividends_path(ins.source, ins.symbol)
    if not p.exists():
        _warn_no_dividends(instrument_id)
        return None
    return _read_dividends(str(p), p.stat().st_mtime_ns).copy()


@functools.lru_cache(maxsize=None)
def _warn_no_dividends(instrument_id: str) -> None:
    LOG.warning("%s: no dividends stored (refresh dividends): its holder is paid none", instrument_id)


@functools.lru_cache(maxsize=None)
def _read_dividends(path: str, version: int) -> pd.Series:
    s = pd.read_parquet(path)["amount"].astype(float)
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize() if len(s) else pd.DatetimeIndex([])
    return s.sort_index()


@dataclass
class Panel:
    """Aligned bars of several instruments of ONE calendar. Wide frames: index = close time, columns = ids. A panel read
    only to work out a list's seats holds the close and the dollar volume, its other fields None."""
    timeframe: str
    instruments: dict[str, Instrument]
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    dollar_volume: pd.DataFrame
    # what the engine works out from these bars alone, kept for its next use on this panel: a panel's bars are not
    # changed once it is built (a truncated or filtered panel is a new one), so nothing kept here goes stale
    memo: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def ids(self) -> list[str]:
        return list(self.close.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    def truncate(self, end: pd.Timestamp) -> "Panel":
        """The panel's bars up to `end`, its fields not read left out (a panel of closes and dollar volumes only)."""
        cut = {f: None if getattr(self, f) is None else getattr(self, f).loc[:end] for f in FIELDS}
        return Panel(self.timeframe, self.instruments, **cut)

    def one(self, instrument_id: str) -> pd.DataFrame:
        """One instrument's bars, those it printed; `attrs` name its timeframe and asset class, which turn a span in days
        or months into its number of bars (`strategy.bars_in`)."""
        bars = pd.DataFrame({f: getattr(self, f)[instrument_id] for f in FIELDS}).dropna(subset=["close"])
        bars.attrs = {"timeframe": self.timeframe, "asset_class": self.instruments[instrument_id].asset_class}
        return bars

    def digest(self, instrument_id: str, bars: pd.DataFrame | None = None) -> str:
        """`digest` of the instrument's bars (`one`; `bars` are they, when the caller has them already), worked out once
        per panel: what is kept for an instrument's bars beyond one panel (a rule's positions, a model's features) is
        kept under it."""
        got = self.memo.setdefault("digests", {})
        if instrument_id not in got:
            got[instrument_id] = digest(self.one(instrument_id) if bars is None else bars)
        return got[instrument_id]

    @property
    def started(self) -> pd.DataFrame:
        """True from an instrument's first bar on, including bars it missed later: a missing bar is not a delisting
        (whether the instrument ever prints again is not known at that bar). Worked out once per panel (every
        configuration of a grid asks for it); each caller gets its own copy."""
        got = self.memo.get("started")
        if got is None:
            got = self.memo["started"] = self.close.notna().cumsum() > 0
        return got.copy()

    def as_known(self) -> "Panel":
        """The panel as a strategy knows it at each bar: a bar an instrument missed repeats its last close, with no
        volume. Prices are never filled before an instrument's first bar."""
        missed = self.started & self.close.isna()
        last = self.close.ffill()
        price = {f: getattr(self, f).mask(missed, last) for f in ("open", "high", "low", "close")}
        flow = {f: getattr(self, f).mask(missed, 0.0) for f in ("volume", "dollar_volume")}
        return Panel(self.timeframe, self.instruments, **price, **flow)


def digest(bars: pd.DataFrame) -> str:
    """A hash of one instrument's bars: their times, every field of every bar and their `attrs`. Bars written anew, or
    another first bar (a panel starting on another day), hash otherwise."""
    h = hashlib.blake2b(digest_size=16)
    h.update(bars.index.asi8.tobytes())
    for f in bars.columns:
        h.update(f.encode() + np.ascontiguousarray(bars[f].to_numpy(dtype=np.float64)).tobytes())
    h.update(repr(sorted(bars.attrs.items())).encode())
    return h.hexdigest()


def liquidity(panel: Panel, lookback_days: int = 60) -> pd.DataFrame:
    """Each instrument's median dollar volume a day over the last `lookback_days` calendar days, by day, as it is known
    when the day starts (the days of the panel's bars): a day on which other instruments traded and it did not counts
    as zero, and an instrument has none until its first bar is half that window old. Worked out once per panel and
    window: the lists next to a list (Top-80 and Top-120 around Top-100) rank the same candidates, and a batch ranks
    them again for every job on the list; each caller gets its own copy."""
    key = ("liquidity", lookback_days)
    got = panel.memo.get(key)
    if got is None:
        dv = panel.dollar_volume
        day = (dv.index - pd.Timedelta(microseconds=1)).normalize()
        daily = dv.groupby(day).sum(min_count=1)
        first = daily.notna().idxmax().where(daily.notna().any())
        daily = daily.fillna(0.0).where(daily.notna().cumsum() > 0)
        seasoned = pd.DataFrame(np.greater_equal.outer(daily.index.values,
                                                       (first + pd.Timedelta(days=lookback_days // 2)).values),
                                index=daily.index, columns=daily.columns)
        got = panel.memo[key] = daily.rolling(f"{lookback_days}D", min_periods=1).median().where(seasoned).shift(1)
    return got.copy()


def load_panel(instrument_ids: list[str], timeframe: str, start=None, end=None, fields=FIELDS,
               index: pd.DatetimeIndex | None = None) -> Panel:
    """The instruments' bars, aligned: only `fields` are read (the others None), on the bars any of them printed or,
    with `index`, on those bars (a list's panel stays on the bars of all its candidates)."""
    instruments = {i: parse(i) for i in instrument_ids}
    calendars = {ins.calendar for ins in instruments.values()}
    if len(calendars) != 1:
        raise ValueError(f"a panel must share one trading calendar, got {sorted(calendars)}: evaluate classes "
                         "separately and combine their daily returns")
    frames = {i: load_bars(i, timeframe, start, end, fields) for i in instrument_ids}
    wide = {f: pd.DataFrame({i: df[f] for i, df in frames.items()}).sort_index() for f in fields}
    if index is not None:
        wide = {f: w.reindex(index) for f, w in wide.items()}
    return Panel(timeframe, instruments, **{f: wide.get(f) for f in FIELDS})
