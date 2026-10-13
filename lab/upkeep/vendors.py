"""What the refreshes share: a request budget and the ledger of every request asked, a vendor's rate limit,
and Dukascopy's hourly quotes with its clock put right."""
from __future__ import annotations

import json
import lzma
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from lab.data import calendars as cal
from lab.data import store
from lab.foundation import log
from lab.foundation.config import DATA_DIR, LOGS_DIR

LOG = log.get("refresh")

DUKASCOPY = "https://datafeed.dukascopy.com/datafeed"         # its public hourly bid candles of FX pairs

# its names of the metals and crude; an FX pair is named by its pair without the slash
DUKASCOPY_NAMES = {"XAU/USD": "XAUUSD", "XAG/USD": "XAGUSD", "XPT/USD": "XPTCMDUSD", "XPD/USD": "XPDCMDUSD",
                   "WTI/USD": "LIGHTCMDUSD"}

DUKASCOPY_DIR = DATA_DIR / "raw" / "dukascopy"                # its monthly files as they arrive, kept to resume
# seconds after each request: at 1.5 it refused one month four times in a row within 20 minutes (2026-09-26)
DUKASCOPY_PAUSE = 5.0
# spans of an instrument's Dukascopy candles (their stamps, as UTC) that carry New York's wall clock written as UTC,
# four hours early in summer and five in winter: AUD/USD's moves line up with EUR/USD's hour by hour only once put on
# that clock (a return correlation of 0.35-0.73 in each month of both spans, 0.0-0.2 as stamped; 99 of their 106 weeks
# line up best at exactly New York's offset), and Forexite's minutes of September 2007 match them to 2 bp so, 8 bp and
# more otherwise. As stamped, their Friday afternoons looked missing. Every other series of theirs lines up as stamped
# (each month of 2003-2020 against EUR/USD, EUR/USD's against GBP/USD; checked 2026-09-27).
DUKASCOPY_NEW_YORK_CLOCK = {"AUD/USD": (("2007-04-01", "2008-09-21"), ("2009-04-05", "2009-09-20"))}
# Dukascopy's first months stamp their hours an hour late (checked 2026-09-28 against Forexite's minutes, whose clock the
# US payrolls confirm: their 12:30 UTC release is Forexite's widest minute, at 12:31, and falls in Dukascopy's hour that
# closes at 14:00 in June and July 2003): day by day, the four pairs it has from 2003-05 line up with Forexite an hour
# earlier on every day to 2003-07-25, and gold on every day to 2003-08-01; they are put an hour earlier. In the pairs'
# week of 2003-07-28 the hours are late and on time in turn within a day (USD/JPY's late to 05:00 and from 20:00 on the
# 29th, USD/CHF's from 16:00 on the 30th): not taken, so Forexite's minutes fill them (`forexite-fill`); gold's
# Forexite quote is 8-35 bp from Dukascopy's and cannot, and its days of that week are late throughout
DUKASCOPY_HOUR_LATE = {**{s: ("2003-05-01", "2003-07-27") for s in ("EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF")},
                       "XAU/USD": ("2003-05-01", "2003-08-02")}
DUKASCOPY_CLOCK_UNKNOWN = {s: ("2003-07-27", "2003-08-02") for s in ("EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF")}


class RateLimited(RuntimeError):
    """The vendor refused to answer; stop the run (a partial universe must never pass for a complete one)."""


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    vendor: str
    per_minute: int | None
    max_requests: int | None
    used: int = 0
    window: list = field(default_factory=list)

    def acquire(self) -> None:
        if self.max_requests is not None and self.used >= self.max_requests:
            raise BudgetExceeded(f"{self.vendor}: run ceiling of {self.max_requests} requests reached")
        if self.per_minute:
            now = time.monotonic()
            self.window = [t for t in self.window if now - t < 60.0]
            if len(self.window) >= self.per_minute:
                wait = 60.0 - (now - self.window[0]) + 0.1
                LOG.info("%s: %d requests in the last minute, waiting %.1fs", self.vendor, len(self.window), wait)
                time.sleep(wait)
            self.window.append(time.monotonic())
        self.used += 1


def _ledger(vendor: str, **entry) -> None:
    p = LOGS_DIR / "requests" / f"{vendor}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    entry["at"] = datetime.now(timezone.utc).isoformat()
    with p.open("a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def dukascopy_on_utc(s: str, raw: pd.DataFrame) -> pd.DataFrame:
    """An instrument's Dukascopy bars (stamped at their start) on UTC: the spans it stamped on New York's wall clock
    put on UTC (DUKASCOPY_NEW_YORK_CLOCK), those it stamped an hour late put an hour earlier (DUKASCOPY_HOUR_LATE),
    and those whose clock changes hour to hour left out (DUKASCOPY_CLOCK_UNKNOWN)."""
    if raw.empty:
        return raw
    unknown = DUKASCOPY_CLOCK_UNKNOWN.get(s)
    if unknown:
        out = np.asarray((raw.index >= pd.Timestamp(unknown[0], tz="UTC")) & (raw.index < pd.Timestamp(unknown[1], tz="UTC")))
        if out.any():
            LOG.info("%s: %d Dukascopy hours of %s..%s, stamped late and on time in turn, left out", s,
                     int(out.sum()), *unknown)
            raw = raw[~out]
    late = DUKASCOPY_HOUR_LATE.get(s)
    if late:
        moved = np.asarray((raw.index >= pd.Timestamp(late[0], tz="UTC")) & (raw.index < pd.Timestamp(late[1], tz="UTC")))
        if moved.any():
            stamps = raw.index.to_series()
            stamps.iloc[np.flatnonzero(moved)] -= pd.Timedelta(hours=1)
            raw = raw.set_axis(pd.DatetimeIndex(stamps))
            LOG.info("%s: %d Dukascopy hours stamped an hour late put an hour earlier", s, int(moved.sum()))
            raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    spans = DUKASCOPY_NEW_YORK_CLOCK.get(s)
    if not spans:
        return raw
    idx = raw.index
    wall = np.zeros(len(idx), dtype=bool)
    for lo, hi in spans:
        wall |= np.asarray((idx >= pd.Timestamp(lo, tz="UTC")) & (idx < pd.Timestamp(hi, tz="UTC")))
    if not wall.any():
        return raw
    moved = idx[wall].tz_localize(None).tz_localize(cal.NY, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    stamps = idx.to_series().copy()
    stamps.iloc[np.flatnonzero(wall)] = moved
    out = raw.set_axis(pd.DatetimeIndex(stamps))
    out = out[out.index.notna()]
    LOG.info("%s: %d Dukascopy hours stamped on New York's clock put on UTC", s, int(wall.sum()))
    return out[~out.index.duplicated(keep="last")].sort_index()


def dukascopy_hours(blob: bytes, month_start: pd.Timestamp) -> pd.DataFrame:
    """One month of Dukascopy's hourly bid candles as raw bars stamped at their start, prices in its integer points:
    its bi5 file is LZMA-compressed 24-byte records of the seconds from the month's start, open, close, low, high and
    the tick volume. An hour without ticks is not a bar."""
    raw = lzma.decompress(blob) if blob else b""
    rows = [r for r in struct.iter_unpack(">5if", raw) if r[5] > 0]
    df = pd.DataFrame(rows, columns=["t", "open", "close", "low", "high", "ticks"], dtype=float)
    df.index = month_start + pd.to_timedelta(df["t"], unit="s")
    df["volume"] = 0.0                                       # an FX pair has no traded volume, as the vendor's
    df["dollar_volume"] = 0.0
    return df[store.BAR_COLUMNS]


def _dukascopy_month(session, s: str, p: pd.Period, budget: Budget, side: str = "BID") -> bytes:
    """One month of a pair's hourly candles on one side of the quote, kept on disk as it arrives (an empty file: a
    month it has none of), so a
    run that stops resumes where it stopped and a month is never asked twice. Dukascopy answers slowly and refuses
    with 503 or a dropped connection when asked fast: a pause after each request, and a refusal waits 1, 2 and 4
    minutes before the run stops."""
    name = DUKASCOPY_NAMES.get(s, s.replace("/", ""))
    cached = DUKASCOPY_DIR / name / (f"{p}.bi5" if side == "BID" else f"{p}.{side}.bi5")
    if cached.exists():
        return cached.read_bytes()
    url = f"{DUKASCOPY}/{name}/{p.year}/{p.month - 1:02d}/{side}_candles_hour_1.bi5"
    for wait in (60, 120, 240, None):
        budget.acquire()
        try:
            r = session.get(url, timeout=120, headers={"User-Agent": "strategy-lab"})
            status, body = r.status_code, r.content
        except requests.ConnectionError as e:
            status, body = f"connection: {type(e).__name__}", b""
        _ledger("dukascopy", status=status, symbol=s, month=str(p), bytes=len(body))
        time.sleep(DUKASCOPY_PAUSE)
        if status in (200, 404):
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(body if status == 200 else b"")
            return body if status == 200 else b""
        if wait is None:
            raise RateLimited(f"dukascopy refused {s} {p} four times ({status}): stop; a new run resumes here")
        LOG.warning("dukascopy %s %s: %s, waiting %ds before asking again", s, p, status, wait)
        time.sleep(wait)
    raise AssertionError("unreachable")

