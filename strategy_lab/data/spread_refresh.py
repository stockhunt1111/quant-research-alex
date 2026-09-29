"""Fetch the bid and ask of the five spot quotes and keep, hour by hour, the spread a broker quoted at the hour's first
and last quote: what a fill at a bar's open or close pays (`data.spreads`, charged by `engine.costs`).

    python -m strategy_lab.data.spread_refresh exness --dry-run      # Exness's ticks, a zip a quote and month
    python -m strategy_lab.data.spread_refresh dukascopy --dry-run   # Dukascopy's hourly bid and ask before Exness's
    python -m strategy_lab.data.spread_refresh build                 # what the engine reads: data/store/td/spread

The rules are `refresh`'s: a dry run counts the requests and makes none, every request goes to the ledger, a refusal
stops the run, and each file is kept as it arrives (data/raw/exness, data/raw/dukascopy), so a stopped run resumes.

Exness, a retail CFD broker on MetaTrader 5, publishes every
tick of its quotes with bid and ask, free and with no key: its Pro account's series, whose spread is the whole cost of
a fill (the account charges no commission), from 2015-08 (gold, silver), 2016-01 (platinum, palladium) and 2019-02
(crude). Its mid is the spot Twelve Data quotes, to 1-2 bp. Dukascopy's hourly bid and ask candles reach further back
(gold 2003, silver 2011, crude 2011-11); another broker's spreads, wider on gold (2.1 bp against Exness's 0.35 in
2025-04) and narrower on palladium, they are put on Exness's level by the ratio of the two over Exness's first year
(`build`), so the history does not jump where one broker hands over to the other. Its crude is a CFD on the future,
off the spot by up to 1%: its spread is still a crude spread in dollars.
"""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import requests

from strategy_lab import log
from strategy_lab.config import DATA_DIR
from strategy_lab.data import spreads, store
from strategy_lab.data.refresh import (DUKASCOPY_NAMES, Budget, BudgetExceeded, RateLimited, _dukascopy_month,
                                       _ledger, dukascopy_hours, dukascopy_on_utc)

LOG = log.get("spread_refresh")
EXNESS = "https://ticks.ex2archive.com/ticks"
EXNESS_DIR = DATA_DIR / "raw" / "exness"          # a quote's month: its zip as it came, and its hours (`exness_hours`)
EXNESS_SYMBOLS = {"XAU/USD": "XAUUSD", "XAG/USD": "XAGUSD", "XPT/USD": "XPTUSD", "XPD/USD": "XPDUSD",
                  "WTI/USD": "USOIL"}              # the Pro account's series: no suffix
EXNESS_PAUSE = 0.5                                 # seconds after each request
# the first month of each series in Exness's archive (its listing, 2026-09-28)
EXNESS_FIRST_MONTH = {"XAU/USD": "2015-08", "XAG/USD": "2015-08", "XPT/USD": "2016-01", "XPD/USD": "2016-01",
                      "WTI/USD": "2019-02"}
# the months of Dukascopy's hourly bid and ask a quote takes before Exness's first, and the first year of Exness's that
# puts them on its level: gold's hourly history begins in 2003-05, silver's in 2011 (Dukascopy's silver before is not
# the market's, `refresh.DUKASCOPY_FIRST_MONTH`), and crude's CFD in 2013-10, for the daily bars' fills before Exness's
# of 2019: its ask before is not a quote's other side (checked 2026-09-28: the ask's file is the bid's, byte for byte,
# from its first month, 2011-11, to 2013-01; no candles in 2013-03 and 2013-04; in 2013-09 three quarters of the hours
# with ask and bid equal and the rest at $0.55, ten times the $0.056 of 2013-05..08 and the $0.02-0.06 from 2013-10);
# platinum's and palladium's Dukascopy CFDs begin after Exness's (2021)
DUKASCOPY_SPANS = {"XAU/USD": ("2003-05", "2016-07"), "XAG/USD": ("2011-01", "2016-07"),
                   "WTI/USD": ("2013-10", "2020-01")}
SCALE_DAYS = 365                                   # Exness's first year, whose level Dukascopy's spreads are put on


def _exness_ticks(zipped: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A month's ticks in time order: their times (ms since the epoch, UTC), bids and asks; none from an empty file (a
    month the archive does not have)."""
    if not zipped:
        return np.array([], dtype=np.int64), np.array([]), np.array([])
    with zipfile.ZipFile(io.BytesIO(zipped)) as z:
        raw = z.read(next(n for n in z.namelist() if n.endswith(".csv")))
    tab = pacsv.read_csv(io.BytesIO(raw), convert_options=pacsv.ConvertOptions(
        include_columns=["Timestamp", "Bid", "Ask"],
        column_types={"Timestamp": pa.timestamp("ms", tz="UTC"), "Bid": pa.float64(), "Ask": pa.float64()}))
    t = tab.column("Timestamp").to_numpy().astype("datetime64[ms]").astype(np.int64)
    bid, ask = tab.column("Bid").to_numpy(), tab.column("Ask").to_numpy()
    order = np.argsort(t, kind="stable")
    return t[order], bid[order], ask[order]


def exness_hours(zipped: bytes) -> pd.DataFrame:
    """A month of Exness's ticks as its hours: the spread (ask - bid, in the quote's currency) at the hour's first and
    last tick, the mid of the last and the number of ticks, stamped at the hour's close as the store stamps a bar. An
    empty file (a month the archive does not have) has none."""
    t, bid, ask = _exness_ticks(zipped)
    if len(t) == 0:
        return pd.DataFrame({k: pd.Series(dtype=float) for k in ("open", "close", "mid", "ticks")},
                            index=pd.DatetimeIndex([], tz="UTC", name="close_time"))
    hour = t // 3_600_000
    first = np.flatnonzero(np.r_[True, hour[1:] != hour[:-1]])
    last = np.r_[first[1:] - 1, len(t) - 1]
    spread = ask - bid
    crossed = int((spread < 0).sum())
    if crossed:
        LOG.warning("exness: %d of %d ticks quote the ask below the bid", crossed, len(t))
    index = pd.to_datetime((hour[first] + 1) * 3_600_000, unit="ms", utc=True)
    return pd.DataFrame({"open": spread[first], "close": spread[last], "mid": (bid[last] + ask[last]) / 2,
                         "ticks": (last - first + 1).astype(float)}, index=pd.DatetimeIndex(index, name="close_time"))


def exness_minutes(zipped: bytes) -> pd.DataFrame:
    """A month of Exness's ticks as one-minute bars of their mid ((bid + ask) / 2): its first, highest, lowest and last,
    stamped at the minute's close as the store stamps a bar; a minute without a tick has none. The quote's own prices,
    a broker's a client trades at, where a vendor's hourly bars mix two (`refresh.EXNESS_QUOTES`)."""
    t, bid, ask = _exness_ticks(zipped)
    if len(t) == 0:
        return pd.DataFrame({k: pd.Series(dtype=float) for k in ("open", "high", "low", "close")},
                            index=pd.DatetimeIndex([], tz="UTC", name="close_time"))
    mid = (bid + ask) / 2
    minute = t // 60_000
    first = np.flatnonzero(np.r_[True, minute[1:] != minute[:-1]])
    last = np.r_[first[1:] - 1, len(t) - 1]
    index = pd.to_datetime((minute[first] + 1) * 60_000, unit="ms", utc=True)
    return pd.DataFrame({"open": mid[first], "high": np.maximum.reduceat(mid, first),
                         "low": np.minimum.reduceat(mid, first), "close": mid[last]},
                        index=pd.DatetimeIndex(index, name="close_time"))


def exness_mid_minutes(s: str) -> pd.DataFrame:
    """A quote's Exness mid minutes over every month whose ticks are on disk (`exness_minutes`), each month's kept once
    built (<symbol>/minutes/<month>.parquet), again when its zip is newer (the month under way grows)."""
    d = EXNESS_DIR / EXNESS_SYMBOLS[s]
    zips = sorted(d.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"no Exness ticks of {s} on disk ({d}): spread_refresh exness first")
    parts = []
    for z in zips:
        kept = d / "minutes" / f"{z.stem}.parquet"
        if not kept.exists() or kept.stat().st_mtime < z.stat().st_mtime:
            kept.parent.mkdir(exist_ok=True)
            exness_minutes(z.read_bytes()).to_parquet(kept)
            LOG.info("exness %s %s: mid minutes built from its ticks", s, z.stem)
        parts.append(pd.read_parquet(kept))
    out = pd.concat([p for p in parts if len(p)])
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch_exness(quotes: list[str], dry_run: bool = True, max_requests: int | None = None) -> dict:
    """Every month of Exness's ticks of the quotes from its first (EXNESS_FIRST_MONTH) not yet on disk, each kept as
    its zip and its hours; a month the archive does not have is kept as an empty file, and is not asked again. A month
    kept is not asked again either, but for the month under way, whose file grows until it ends."""
    now = pd.Period(datetime.now(timezone.utc).strftime("%Y-%m"), freq="M")
    todo = [(EXNESS_SYMBOLS[s], p) for s in quotes for p in pd.period_range(EXNESS_FIRST_MONTH[s], now, freq="M")
            if p == now or not (EXNESS_DIR / EXNESS_SYMBOLS[s] / f"{p}.parquet").exists()]
    report = {"quotes": len(quotes), "planned_requests": len(todo)}
    LOG.info("exness ticks: %s", report)
    if dry_run:
        return report
    session = requests.Session()
    budget = Budget("exness", per_minute=None, max_requests=max_requests)
    fetched, missing, hours = 0, 0, 0
    try:
        for sym, p in todo:
            budget.acquire()
            y, m = p.year, f"{p.month:02d}"
            url = f"{EXNESS}/{sym}/{y}/{m}/Exness_{sym}_{y}_{m}.zip"
            r = session.get(url, timeout=600, headers={"User-Agent": "strategy-lab"})
            _ledger("exness", status=r.status_code, symbol=sym, month=str(p), bytes=len(r.content))
            time.sleep(EXNESS_PAUSE)
            if r.status_code in (403, 418, 429):
                raise RateLimited(f"exness refused {sym} {p} ({r.status_code}): stop; a new run resumes here")
            d = EXNESS_DIR / sym
            d.mkdir(parents=True, exist_ok=True)
            if r.status_code == 404:
                LOG.warning("exness %s %s: not in the archive, kept as an empty month", sym, p)
                (d / f"{p}.zip").write_bytes(b"")
                exness_hours(b"").to_parquet(d / f"{p}.parquet")
                missing += 1
                continue
            r.raise_for_status()
            (d / f"{p}.zip").write_bytes(r.content)
            h = exness_hours(r.content)
            h.to_parquet(d / f"{p}.parquet")
            fetched += 1
            hours += len(h)
            LOG.info("exness %s %s: %d ticks in %d hours", sym, p, int(h["ticks"].sum()), len(h))
    except BudgetExceeded as e:
        LOG.warning("%s: stopped; a new run resumes where this one stopped", e)
    return report | {"fetched": fetched, "not_in_archive": missing, "hours": hours}


def _dukascopy_months(s: str) -> pd.PeriodIndex:
    lo, hi = DUKASCOPY_SPANS[s]
    return pd.period_range(lo, hi, freq="M")


def fetch_dukascopy(quotes: list[str], dry_run: bool = True, max_requests: int | None = None) -> dict:
    """Dukascopy's hourly bid and ask candles of the months each quote takes from it (DUKASCOPY_SPANS), those not on
    disk yet (data/raw/dukascopy, shared with `refresh dukascopy-extend`, which keeps the bid's)."""
    todo = []
    for s in quotes:
        if s not in DUKASCOPY_SPANS:
            continue
        d = DATA_DIR / "raw" / "dukascopy" / DUKASCOPY_NAMES[s]
        for p in _dukascopy_months(s):
            for side, f in (("BID", d / f"{p}.bi5"), ("ASK", d / f"{p}.ASK.bi5")):
                if not f.exists():
                    todo.append((s, p, side))
    report = {"quotes": len(quotes), "planned_requests": len(todo)}
    LOG.info("dukascopy bid and ask: %s", report)
    if dry_run:
        return report
    session = requests.Session()
    budget = Budget("dukascopy", per_minute=None, max_requests=max_requests)
    try:
        for s, p, side in todo:
            _dukascopy_month(session, s, p, budget, side)
    except BudgetExceeded as e:
        LOG.warning("%s: stopped; a new run resumes where this one stopped", e)
    return report | {"fetched": budget.used}


def dukascopy_spreads(s: str) -> pd.DataFrame:
    """A quote's Dukascopy hours of DUKASCOPY_SPANS on disk as `exness_hours` gives Exness's: the spread at the hour's
    first and last quote (the bid's and ask's candles open and close on the same ticks) and the mid, in the quote's
    currency (Dukascopy's integer points on the price step that puts its mid on the vendor's closes: its hourly ones,
    or its daily ones where its hours begin later, as crude's), on UTC with
    Dukascopy's clock put right (`refresh.dukascopy_on_utc`), stamped at the hour's close."""
    d = DATA_DIR / "raw" / "dukascopy" / DUKASCOPY_NAMES[s]
    parts = []
    for p in _dukascopy_months(s):
        files = d / f"{p}.bi5", d / f"{p}.ASK.bi5"
        if not all(f.exists() for f in files) or not all(f.stat().st_size for f in files):
            continue
        start = pd.Timestamp(p.start_time, tz="UTC")
        bid, ask = (dukascopy_hours(f.read_bytes(), start) for f in files)
        both = bid[["open", "close"]].join(ask[["open", "close"]], lsuffix="_bid", rsuffix="_ask", how="inner")
        parts.append(both)
    if not parts:
        return pd.DataFrame(columns=["open", "close", "mid", "ticks"], dtype=float)
    raw = pd.concat(parts)
    raw = dukascopy_on_utc(s, raw[~raw.index.duplicated(keep="last")].sort_index())
    mid = (raw["close_bid"] + raw["close_ask"]) / 2
    stamps = raw.index + pd.Timedelta(hours=1)
    vendor = store.read_bars("td", "1h", s, ["close"])["close"]
    common = stamps.intersection(vendor.index)
    if len(common):
        ratio = np.median(mid.set_axis(stamps)[common] / vendor[common])
    else:
        daily = store.read_bars("td", "1d", s, ["close"])["close"]
        daily = daily[(daily.index >= stamps[0]) & (daily.index <= stamps[-1])]
        if daily.empty:
            raise ValueError(f"{s}: no bar of the vendor's over Dukascopy's hours to find its price step on")
        LOG.info("%s: the vendor's hours begin after Dukascopy's: its price step from the daily closes", s)
        ratio = np.median(mid) / daily.median()
    step = 10.0 ** np.round(np.log10(ratio))
    return pd.DataFrame({"open": (raw["open_ask"] - raw["open_bid"]).to_numpy() / step,
                         "close": (raw["close_ask"] - raw["close_bid"]).to_numpy() / step,
                         "mid": mid.to_numpy() / step, "ticks": np.nan},
                        index=pd.DatetimeIndex(stamps, name="close_time"))


def exness_on_disk(s: str) -> pd.DataFrame:
    d = EXNESS_DIR / EXNESS_SYMBOLS[s]
    parts = [x for x in (pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))) if len(x)]
    if not parts:
        raise FileNotFoundError(f"no Exness ticks of {s} on disk ({d}): spread_refresh exness first")
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="last")].sort_index()


def _quoted(hours: pd.DataFrame, s: str, source: str) -> pd.DataFrame:
    """The hours whose first and last quotes both hold an ask above the bid: an ask at the bid or below it is not the
    other side of a quote (a file of one side served for both), and would make a fill free."""
    one_sided = (hours["open"] <= 0) | (hours["close"] <= 0)
    if one_sided.any():
        LOG.warning("%s: %d of %s's %d hours quote no ask above the bid: left out", s, int(one_sided.sum()), source,
                    len(hours))
    return hours[~one_sided]


def build(quotes: list[str]) -> dict:
    """Each quote's hours of spreads, written where the engine reads them (`spreads.spread_path`): Exness's from its
    first tick, Dukascopy's before it put on Exness's level (the ratio of the two brokers' median spreads at the hours'
    close over Exness's first SCALE_DAYS), with the source of each hour; an hour without a two-sided quote is left out
    (`_quoted`)."""
    report = {}
    for s in quotes:
        ex = _quoted(exness_on_disk(s), s, "Exness")
        first = ex.index[0]
        parts, meta = [ex.assign(source="exness")], {"exness_from": str(first)}
        if s in DUKASCOPY_SPANS:
            du = _quoted(dukascopy_spreads(s), s, "Dukascopy")
            overlap = du.index.intersection(ex.index[ex.index < first + pd.Timedelta(days=SCALE_DAYS)])
            if len(overlap) == 0:
                raise ValueError(f"{s}: Dukascopy's hours do not reach Exness's first year: no level to put them on")
            scale = float(ex["close"][overlap].median() / du["close"][overlap].median())
            before = du[du.index < first]
            parts.insert(0, before.assign(open=before["open"] * scale, close=before["close"] * scale,
                                          source="dukascopy"))
            meta |= {"dukascopy_from": str(before.index[0]) if len(before) else None, "dukascopy_scale": scale,
                     "scale_hours": len(overlap)}
            LOG.info("%s: Dukascopy's %d hours before %s at %.3f of its spread (%d hours of both)", s, len(before),
                     first.date(), scale, len(overlap))
        out = pd.concat(parts)
        out.index.name = "close_time"
        path = spreads.spread_path(f"td:{s}")
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path)
        meta |= {"hours": len(out), "last": str(out.index[-1]), "built_at": datetime.now(timezone.utc).isoformat()}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        report[s] = meta
    return report


def main() -> None:
    from strategy_lab.universes import STOCKHUNT_COMMODITIES
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["exness", "dukascopy", "build"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--max-requests", type=int)
    args = ap.parse_args()
    log.setup(f"spread_refresh_{args.source}")
    quotes = args.symbols or STOCKHUNT_COMMODITIES
    if args.source == "exness":
        rep = fetch_exness(quotes, args.dry_run, args.max_requests)
    elif args.source == "dukascopy":
        rep = fetch_dukascopy(quotes, args.dry_run, args.max_requests)
    else:
        rep = build(quotes)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
