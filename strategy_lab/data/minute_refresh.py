"""Fetch the one-minute bars the engine walks intrabar exits through (`data.minutes`), and keep them only where they
make the stored hourly bars they sit in.

    python -m strategy_lab.data.minute_refresh binance --dry-run     # the crypto lists' perps, the months each is held
    python -m strategy_lab.data.minute_refresh forexite --dry-run    # the FX majors, 2003 on
    python -m strategy_lab.data.minute_refresh twelvedata --dry-run  # stocks, ETFs and spot metals, 2020-03 on

The rules are `refresh`'s: a dry run counts the requests and makes none, every request goes to the ledger, a refusal
stops the run, and each file is kept as it arrives (data/raw/binance/minutes, data/raw/forexite,
data/raw/twelvedata/minutes), so a stopped run resumes.

A perp's minutes and its hourly bars are the exchange's record of the same trades: an hour's minutes open at its open,
reach its high and its low and close at its close. An hour where they do not (2024-10-28 22:00: ETH's first minute
opens 60 bp under the hour's open and low) is kept as one bar of the hour's own prices, so it is walked on them as a bar
without minutes is, and every hour the minutes cover is still there. Twelve Data's minutes are the vendor's own record
of its stored hours once put on their basis (`on_stored_hours`); Forexite's are another feed, kept month by month
(`agreeing_months`).
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from strategy_lab import log
from strategy_lab.config import DATA_DIR, env
from strategy_lab.data import minutes
from strategy_lab.data.bars import load_bars
from strategy_lab.data.instruments import parse
from strategy_lab.data.refresh import (ARCHIVE, EXNESS_QUOTES, FOREXITE_DIR, FOREXITE_MATCH_BP, FOREXITE_ZONE, Budget,
                                       BudgetExceeded, RateLimited, _ledger)
from strategy_lab.data.spread_refresh import exness_mid_minutes

LOG = log.get("minute_refresh")
BINANCE_RAW = DATA_DIR / "raw" / "binance" / "minutes"          # a file a month (a day for the month under way)
# a perp's minutes and a spot pair's (the ML task's coins are held on spot): the archive's path, where they are kept
BINANCE_MINUTES = {"perp": ("futures/um", BINANCE_RAW), "spot": ("spot", BINANCE_RAW / "spot")}
MONTHLY = ARCHIVE + "/data/{market}/monthly/klines/{s}/1m/{s}-1m-{m}.zip"
DAILY = ARCHIVE + "/data/{market}/daily/klines/{s}/1m/{s}-1m-{d}.zip"
SAME = 1e-6              # prices of one record in single precision: the relative difference rounding leaves
HOUR = pd.Timedelta(hours=1).value


def crypto_months(universes: list[str] | None = None) -> dict[str, list[str]]:
    """The months each perp of the crypto lists is held in one of them, and the month after (a trade kept past the list
    runs on), up to the current one: the ML task's coins over their whole history."""
    from strategy_lab import evaluate as ev
    from strategy_lab import lists
    names = universes or [x.universe for x in lists.OURS if x.market == "Crypto"] + [
        u for u in lists.ML_TASK if u.startswith("crypto")]
    now = pd.Period(datetime.now(timezone.utc).strftime("%Y-%m"), freq="M")
    out: dict[str, set] = {}
    for u in names:
        _, panel, member = ev._load(u, "1d", lists.start(u), None)
        for i in panel.ids:
            held = member[i] if member is not None else panel.close[i].notna()
            if not held.any():
                continue
            per = pd.PeriodIndex(held.index[held.to_numpy()].tz_localize(None), freq="M").unique()
            out.setdefault(i, set()).update(str(p) for p in per.union(per + 1) if p <= now)
    return {i: sorted(m) for i, m in sorted(out.items())}


def _klines(zipped: bytes) -> pd.DataFrame:
    """An archive file's one-minute klines with a trade (open, high, low and close), stamped at the minute's close (its
    open plus a minute); the archive stamps in milliseconds, and in microseconds from 2025. A minute without a trade
    repeats the last close as its open, high, low and close and makes no price: the exchange's hour opens at its first
    trade, so such a minute would open the hour below or above it (DOGE on spot in 2019, half its hours)."""
    with zipfile.ZipFile(io.BytesIO(zipped)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]), header=None, dtype=str)
    if not raw.iat[0, 0].strip().isdigit():                     # a header row, in the newer files
        raw = raw.iloc[1:]
    t = raw[0].astype("int64").to_numpy()
    unit = "us" if t.max() > 10**14 else "ms"
    out = pd.DataFrame({c: raw[k].astype(float).to_numpy() for k, c in ((1, "open"), (2, "high"), (3, "low"),
                                                                        (4, "close"))})
    out.index = pd.to_datetime(t, unit=unit, utc=True) + pd.Timedelta(minutes=1)
    out = out[raw[5].astype(float).to_numpy() > 0]
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch_binance(months: dict[str, list[str]], dry_run: bool, max_requests: int | None) -> dict:
    """Each instrument's months not on disk yet, from the monthly archive of its market (a perp's, a spot pair's); the
    month under way day by day. A month the archive has no file of is noted beside the files (missing.txt) and not
    asked again."""
    now = datetime.now(timezone.utc)
    noted = set()
    for _, raw in BINANCE_MINUTES.values():
        if (raw / "missing.txt").exists():
            noted |= {f"{raw.name} {x}" for x in (raw / "missing.txt").read_text().split("\n")}
    todo = []
    for i, ms in months.items():
        ins = parse(i)
        market, raw = BINANCE_MINUTES[ins.source]
        for m in ms:
            if (raw / ins.symbol / f"{m}.parquet").exists() or f"{raw.name} {ins.symbol} {m}" in noted:
                continue
            if m == now.strftime("%Y-%m"):
                days = pd.date_range(f"{m}-01", now.date() - pd.Timedelta(days=1), freq="D").strftime("%Y-%m-%d")
                todo += [(market, raw, ins.symbol, d) for d in days if not (raw / ins.symbol / f"{d}.parquet").exists()]
            else:
                todo.append((market, raw, ins.symbol, m))
    report = {"instruments": len(months), "months": sum(len(v) for v in months.values()), "requests_planned": len(todo)}
    if dry_run:
        return report
    session, budget = requests.Session(), Budget("binance", per_minute=None, max_requests=max_requests)
    got = absent = 0
    for market, raw, sym, span in todo:
        budget.acquire()
        url = (DAILY if len(span) == 10 else MONTHLY).format(market=market, s=sym, m=span, d=span)
        r = session.get(url, timeout=180)
        _ledger("binance", status=r.status_code, what=f"1m archive, {market}", symbol=sym, span=span,
                bytes=len(r.content))
        if r.status_code in (418, 429):
            raise RateLimited(f"binance archive answered {r.status_code}: stop; a new run resumes here")
        if r.status_code != 200:
            if len(span) == 7:
                raw.mkdir(parents=True, exist_ok=True)
                with (raw / "missing.txt").open("a") as fh:
                    fh.write(f"{sym} {span}\n")
            absent += 1
            continue
        dest = raw / sym / f"{span}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        _klines(r.content).astype(np.float32).to_parquet(tmp)
        tmp.replace(dest)                           # a run stopped mid-write leaves no file that looks whole
        got += 1
    return {**report, "fetched": got, "not_in_archive": absent, "requests_made": budget.used}


def conformed(mins: pd.DataFrame, hours: pd.DataFrame, same: float | None,
              fields: tuple[str, ...] = ("open", "high", "low", "close")) -> tuple[pd.DataFrame, list[str], int]:
    """The minutes as the walk takes them, on the stored hours they sit in (those that close after the hour before and
    at or before the hour's close): an hour whose minutes do not make it (with `same`: one of `fields` further from the
    stored one than that, relative) is replaced by one bar of the hour's own prices stamped at its close, and so is an
    hour between the first minute and the last that has none, so that every stored hour the minutes span is walked on
    something; minutes of an hour the store does not have are left out. The minutes, the hours replaced and the number
    of hours filled."""
    cols = ["open", "high", "low", "close"]
    at = np.searchsorted(hours.index.asi8, mins.index.asi8, side="left")
    inside = at < len(hours)
    inside[inside] &= mins.index.asi8[inside] > hours.index.asi8[at[inside]] - HOUR   # not in a stored hour: left out
    mins, at = mins[inside], at[inside]
    if not len(mins):
        return mins[cols], [], 0
    made = mins.assign(bar=at).groupby("bar").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                                  close=("close", "last"))
    bad = np.array([], dtype=np.int64)
    if same is not None:
        stored = hours.iloc[made.index][list(fields)].to_numpy()
        off = (np.abs(made[list(fields)].to_numpy() / stored - 1) > same).any(axis=1)
        bad = made.index.to_numpy()[off]
    spanned = np.arange(at.min(), at.max() + 1)
    empty = np.setdiff1d(spanned, made.index.to_numpy())
    own = np.union1d(bad, empty)
    kept = mins[~np.isin(at, bad)][cols]
    path = pd.concat([kept, hours.iloc[own][cols]]).sort_index()
    return path, [str(t) for t in hours.index[bad]], len(empty)


def agreeing_months(mins: pd.DataFrame, hours: pd.DataFrame, within_bp: float) -> tuple[dict[str, float], dict]:
    """The months whose minutes, a second source's, make hours on the stored clock and at the stored prices but for a
    steady ratio: their hourly returns line up with the stored ones best without a shift (a shift of up to three hours
    either way is tried: a source that stamps another clock lines up best shifted), and their closes keep within
    `within_bp` of one ratio to the stored ones at the median (another feed may quote another side of the spread:
    Forexite's AUD/USD stands 2-4 bp above Dukascopy's bid through 2003-2005, steadily). The months kept, each with
    its ratio, and why each other one is not."""
    at = np.searchsorted(hours.index.asi8, mins.index.asi8, side="left")
    inside = at < len(hours)
    made = mins[inside].assign(bar=at[inside]).groupby("bar")["close"].last()
    stored = hours["close"].iloc[made.index]
    made.index = stored.index
    month = made.index.tz_localize(None).to_period("M")
    kept, refused = {}, {}
    for m in sorted(set(month)):
        a, b = made[month == m], stored[month == m]
        if len(a) < 24:
            refused[str(m)] = f"{len(a)} hours"
            continue
        ratio = a.to_numpy() / b.to_numpy()
        level = float(np.median(ratio))
        spread = float(np.median(np.abs(ratio / level - 1)) * 1e4)
        ra, rb = np.log(a).diff(), np.log(b).diff()
        with np.errstate(invalid="ignore", divide="ignore"):   # a series that never moves: no correlation, NaN
            lags = {k: ra.corr(rb.shift(k)) for k in range(-3, 4)}
        best = max(lags, key=lambda k: -np.inf if np.isnan(lags[k]) else lags[k])
        if best != 0 or spread > within_bp:
            refused[str(m)] = (f"best lined up {best:+d} h, closes {spread:.1f} bp from their median ratio "
                               f"({(level - 1) * 1e4:+.1f} bp)")
        else:
            kept[str(m)] = level
    return kept, refused


def store_binance(months: dict[str, list[str]]) -> dict:
    """Each perp's minutes on disk, held to its stored hourly bars (`conformed`: the exchange's minutes make its hours
    to single precision), kept in the store."""
    out = {}
    for i in months:
        ins = parse(i)
        files = sorted((BINANCE_MINUTES[ins.source][1] / ins.symbol).glob("*.parquet"))
        if not files:
            continue
        mins = pd.concat([pd.read_parquet(f).astype(float) for f in files]).sort_index()
        mins = mins[~mins.index.duplicated(keep="last")]
        hours = load_bars(i, "1h", fields=("open", "high", "low", "close"))
        path, replaced, filled = conformed(mins, hours, SAME)
        if replaced:
            LOG.warning("%s: %d hours whose minutes do not make the stored hour, walked on the hour's own prices: %s",
                        i, len(replaced), ", ".join(replaced[:5]) + (" ..." if len(replaced) > 5 else ""))
        n = minutes.write(i, path, {"source": f"binance {BINANCE_MINUTES[ins.source][0]} 1m archive",
                                    "hours_on_own_prices": len(replaced), "hours_on_own_prices_first": replaced[:20],
                                    "hours_without_minutes": filled, "files": len(files),
                                    "stored_at": datetime.now(timezone.utc).isoformat()})
        out[i] = {"rows": n, "hours_on_own_prices": len(replaced), "hours_without_minutes": filled}
    return out


def _forexite_minutes(pairs: list[str]) -> dict[str, pd.DataFrame]:
    """The pairs' minutes of every Forexite day file on disk (`refresh._forexite_day` keeps them), stamped at their
    close in UTC: its clock is Central European with summer time and stamps a minute at its end; the hour the clock
    goes back through twice cannot be placed and is left out."""
    want = {p.replace("/", ""): p for p in pairs}
    parts: dict[str, list] = {p: [] for p in pairs}
    for f in sorted(FOREXITE_DIR.glob("*.zip")):
        blob = f.read_bytes()
        if not blob:                                              # a day it has no file of
            continue
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            day = pd.read_csv(z.open(z.namelist()[0]))
        day.columns = [c.strip("<>").lower() for c in day.columns]
        day = day[day["ticker"].isin(want)]
        for t, g in day.groupby("ticker"):
            parts[want[t]].append(g)
    out = {}
    for p, frames in parts.items():
        if not frames:
            continue
        d = pd.concat(frames)
        wall = pd.to_datetime(d["dtyyyymmdd"].astype(str) + d["time"].astype(str).str.zfill(6), format="%Y%m%d%H%M%S")
        end = pd.DatetimeIndex(wall).tz_localize(FOREXITE_ZONE, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
        bars = pd.DataFrame({c: d[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close")}, index=end)
        bars = bars[end.notna()]
        out[p] = bars[~bars.index.duplicated(keep="last")].sort_index()
    return out


def store_forexite(pairs: list[str]) -> dict:
    """Each pair's Forexite minutes, the months that agree with its stored hourly bars (`agreeing_months`, within
    `refresh.FOREXITE_MATCH_BP` of their steady ratio) put on the stored quote by that ratio, held to those bars
    (`conformed`: another feed's minutes are not the stored hours to the bp, so only hours without a minute take the
    hour's own prices), kept in the store as `td:<pair>`."""
    out = {}
    for p, mins in _forexite_minutes(pairs).items():
        i = f"td:{p}"
        hours = load_bars(i, "1h", fields=("open", "high", "low", "close"))
        kept, refused = agreeing_months(mins, hours, FOREXITE_MATCH_BP)
        if refused:
            LOG.warning("%s: %d months of Forexite's minutes not kept: %s", i, len(refused),
                        "; ".join(f"{m} {why}" for m, why in list(refused.items())[:5]))
        month = mins.index.tz_localize(None).to_period("M").astype(str)
        mine = mins[month.isin(list(kept))].copy()
        level = month[month.isin(list(kept))].map(kept).to_numpy(dtype=float)
        mine[["open", "high", "low", "close"]] = mine[["open", "high", "low", "close"]].to_numpy() / level[:, None]
        path, _, filled = conformed(mine, hours, None)
        offsets = pd.Series(kept).sub(1).mul(1e4)
        n = minutes.write(i, path, {"source": "forexite free 1m, each month put on the stored quote by its median ratio",
                                    "months_kept": len(kept), "median_offset_bp": round(float(offsets.median()), 2),
                                    "months_refused": refused, "hours_without_minutes": filled,
                                    "stored_at": datetime.now(timezone.utc).isoformat()})
        out[i] = {"rows": n, "months_kept": len(kept), "months_refused": len(refused), "hours_without_minutes": filled}
    return out


FOREXITE_FROM = "2003-05-01"          # the first FX hourly history here (Dukascopy's, from 2003-05)


def fetch_forexite(days: pd.DatetimeIndex, max_requests: int | None) -> dict:
    """Every weekday's Forexite file not on disk yet, through `refresh._forexite_day` (2 s between requests, the day
    kept empty where it has no file)."""
    from strategy_lab.data.refresh import _forexite_day
    session, budget = requests.Session(), Budget("forexite", per_minute=None, max_requests=max_requests)
    for d in days:
        _forexite_day(session, d, budget)
    return {"requests_made": budget.used}


TD_RAW = DATA_DIR / "raw" / "twelvedata" / "minutes"             # a file a month; an empty one: the vendor has none
TD_FROM = pd.Timestamp("2020-03-24", tz="UTC")     # the vendor's first one-minute bars (MSFT, SPY; XAU/USD 2020-04-06)


def td_months(universes: list[str] | None = None) -> dict[str, list[str]]:
    """The months each stock, ETF and spot metal of the lists is held in one of them since the vendor's minutes begin,
    and the month after, up to the current one; the ML task's stocks over the whole span."""
    from strategy_lab import evaluate as ev
    from strategy_lab import lists
    names = universes or [x.universe for x in lists.OURS if x.market in ("Stocks", "ETFs", "Commodities")] + [
        u for u in lists.ML_TASK if u.startswith("us_stocks")]
    now = pd.Period(datetime.now(timezone.utc).strftime("%Y-%m"), freq="M")
    first = pd.Period(TD_FROM.strftime("%Y-%m"), freq="M")
    out: dict[str, set] = {}
    for u in names:
        _, panel, member = ev._load(u, "1d", lists.start(u), None)
        for i in panel.ids:
            held = member[i] if member is not None else panel.close[i].notna()
            held = held[held.index >= TD_FROM]
            if not held.any():
                continue
            per = pd.PeriodIndex(held.index[held.to_numpy()].tz_localize(None), freq="M").unique()
            out.setdefault(i, set()).update(str(p) for p in per.union(per + 1) if first <= p <= now)
    return {i: sorted(m) for i, m in sorted(out.items())}


def _td_symbol(instrument_id: str) -> str:
    """The vendor's symbol of a list's instrument: a stock's Sharadar ticker is its exchange ticker (another company's
    bars under a ticker taken up again do not make the stored hours, and are not kept)."""
    return parse(instrument_id).symbol


@dataclass
class _SharedBudget(Budget):
    """A request budget several threads draw on."""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def acquire(self) -> None:
        with self.lock:
            super().acquire()


def fetch_twelvedata(months: dict[str, list[str]], dry_run: bool, max_requests: int | None,
                     workers: int = 12) -> dict:
    """Each instrument's months not on disk yet, in pages of the vendor's 5000 minutes from the month's last minute
    back to its first (stamped in UTC at their start: kept at their close), `workers` months at once within the
    requests a minute `.env` allows (a request takes about a second to answer); a month it has no minute of is kept as
    an empty file, and a symbol it refuses is noted (refused.txt) and not asked again."""
    from strategy_lab.data.refresh import TD_PAGE
    refused_log = TD_RAW / "refused.txt"
    refused = set(refused_log.read_text().split("\n")) if refused_log.exists() else set()
    todo = [(i, m) for i, ms in months.items() for m in ms
            if _td_symbol(i) not in refused and not (TD_RAW / safe(i) / f"{m}.parquet").exists()]
    per_day = {"us_equity": 390, "commodity": 1380}
    planned = sum(-(-per_day.get(parse(i).asset_class, 390) * 23 // TD_PAGE) for i, _ in todo)
    report = {"instruments": len(months), "months": sum(len(v) for v in months.values()), "months_to_fetch": len(todo),
              "requests_planned_about": planned}
    if dry_run:
        return report
    per_min = env("TWELVEDATA_REQUESTS_PER_MINUTE")
    budget = _SharedBudget("twelvedata", per_minute=int(per_min) if per_min else None, max_requests=max_requests)
    now = pd.Timestamp.now(tz="UTC").floor("min")
    local = threading.local()
    stop = threading.Event()

    def month(job):
        i, m = job
        sym = _td_symbol(i)
        if stop.is_set() or sym in refused:
            return
        if not hasattr(local, "session"):
            local.session = requests.Session()
        got = _td_month(local.session, budget, i, sym, m, now, refused, refused_log)
        if got is not None:
            dest = TD_RAW / safe(i) / f"{m}.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + f".tmp{threading.get_ident()}")
            got.to_parquet(tmp)
            tmp.replace(dest)

    def guarded(job):
        try:
            month(job)
        except (RateLimited, BudgetExceeded):
            stop.set()
            raise

    with ThreadPoolExecutor(workers) as pool:
        for f in [pool.submit(guarded, job) for job in todo]:
            f.result()
    return {**report, "requests_made": budget.used}


def _td_month(session, budget, i: str, sym: str, m: str, now: pd.Timestamp, refused: set, refused_log) \
        -> pd.DataFrame | None:
    """One month of the vendor's minutes of `sym`, paged backwards from its last minute; None when it refuses the
    symbol (noted, not asked again)."""
    from strategy_lab.data.refresh import TD_PAGE, SymbolRejected, _td_get
    start = max(pd.Timestamp(f"{m}-01", tz="UTC"), TD_FROM)
    end = min(pd.Timestamp(pd.Period(m, freq="M").end_time).tz_localize("UTC").floor("min"), now)
    pages = []
    while start <= end:                        # the vendor answers a span with its last TD_PAGE minutes: page backwards
        try:
            vals = _td_get(session, {"symbol": sym, "interval": "1min", "start_date": f"{start:%Y-%m-%d %H:%M:%S}",
                                     "end_date": f"{end:%Y-%m-%d %H:%M:%S}", "outputsize": TD_PAGE, "order": "ASC",
                                     "timezone": "UTC"}, budget, {"symbol": sym, "what": "1min", "month": m})
        except SymbolRejected as e:
            LOG.warning("%s: the vendor refuses %s (%s): no minutes", i, sym, e)
            with refused_log.open("a") as fh:
                fh.write(f"{sym}\n")
            refused.add(sym)
            return None
        if not vals:
            break
        page = pd.DataFrame(vals)
        page.index = pd.to_datetime(page["datetime"], utc=True, format="ISO8601") + pd.Timedelta(minutes=1)
        pages.append(page[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce"))
        if len(vals) < TD_PAGE:
            break
        end = page.index[0] - pd.Timedelta(minutes=2)           # the minute before this page's first one starts
    got = pd.concat(pages) if pages else pd.DataFrame(columns=["open", "high", "low", "close"], dtype=float)
    return got[~got.index.duplicated(keep="last")].sort_index().astype(np.float64)


def safe(instrument_id: str) -> str:
    return instrument_id.replace(":", "_").replace("/", "-")


def on_stored_hours(mins: pd.DataFrame, hours: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The vendor's minutes put on the basis of the stored hourly bars they sit in, session by session: a stock's stored
    hours are the vendor's hourly bars rescaled to its daily bars where its intraday series misses a corporate action
    (FDX by 1.241 to 2026-05, HON's aerospace spin-off), or another source's where the vendor lacks the session or
    serves another company, so each New York session's minutes are scaled by the median ratio of its stored hours'
    closes to the closes the minutes make of them. Minutes in a session with no stored hour are not kept. The minutes,
    and how many sessions were scaled and by how much at the most."""
    at = np.searchsorted(hours.index.asi8, mins.index.asi8, side="left")
    inside = at < len(hours)
    mins, at = mins[inside], at[inside]
    made = mins.assign(bar=at).groupby("bar")["close"].last()
    ratio = hours["close"].to_numpy()[made.index] / made.to_numpy()
    session = (hours.index - pd.Timedelta(microseconds=1)).tz_convert("America/New_York").tz_localize(None).normalize()
    per_session = pd.Series(ratio, index=session[made.index]).groupby(level=0).median()
    factor = per_session.reindex(session[at]).to_numpy()
    keep = np.isfinite(factor)
    cols = ["open", "high", "low", "close"]
    out = mins[keep][cols].mul(factor[keep], axis=0)
    moved = per_session[(per_session - 1).abs() > 1e-9]
    return out, {"sessions_scaled": len(moved),
                 "farthest": round(float(moved.iloc[(moved - 1).abs().argmax()]), 4) if len(moved) else 1.0,
                 "minutes_without_a_stored_session": int((~keep).sum())}


def store_twelvedata(months: dict[str, list[str]]) -> dict:
    """Each instrument's vendor minutes on disk, put on its stored hourly bars' basis (`on_stored_hours`) and held to
    those bars on their highs and lows (`conformed`: an hour opens at its session's auction, which a minute's first
    trade is not), kept in the store."""
    out = {}
    for i in months:
        months_kept = [x for x in (pd.read_parquet(f) for f in sorted((TD_RAW / safe(i)).glob("*.parquet"))) if len(x)]
        source = "twelvedata 1min"
        if parse(i).symbol in EXNESS_QUOTES:    # its stored hours are Exness's where it quotes: so are its minutes
            ex = exness_mid_minutes(parse(i).symbol)
            months_kept = [ex] + [x[x.index > ex.index[-1]] for x in months_kept]
            source = f"exness mid minutes to {ex.index[-1]}, twelvedata 1min after"
        if not months_kept:                     # every month on disk empty: the vendor has no minute of it
            continue
        mins = pd.concat(months_kept).sort_index().dropna()
        mins = mins[~mins.index.duplicated(keep="last")]
        hours = load_bars(i, "1h", fields=("open", "high", "low", "close"))
        mins, basis = on_stored_hours(mins, hours)
        path, replaced, filled = conformed(mins, hours, SAME, fields=("high", "low"))
        n = minutes.write(i, path, {"source": source, "basis": basis,
                                    "hours_on_own_prices": len(replaced), "hours_on_own_prices_first": replaced[:20],
                                    "hours_without_minutes": filled, "stored_at": datetime.now(timezone.utc).isoformat()})
        out[i] = {"rows": n, "hours_on_own_prices": len(replaced), "hours_without_minutes": filled}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["binance", "forexite", "twelvedata"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--universes", nargs="*", help="binance, twelvedata: these lists only (every list of the market "
                    "by default)")
    ap.add_argument("--symbols", nargs="*", help="forexite: these pairs (the FX majors by default)")
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--workers", type=int, default=12, help="twelvedata: months fetched at once (an answer of 5000 "
                    "minutes takes the vendor 1-4 seconds; the requests a minute `.env` allows hold them all)")
    args = ap.parse_args()
    log.setup(f"minute_refresh_{args.source}")
    if args.source == "binance":
        months = crypto_months(args.universes)
        rep = fetch_binance(months, args.dry_run, args.max_requests)
        if not args.dry_run:
            rep["stored"] = store_binance(months)
    elif args.source == "twelvedata":
        months = td_months(args.universes)
        rep = fetch_twelvedata(months, args.dry_run, args.max_requests, args.workers)
        if not args.dry_run:
            rep["stored"] = store_twelvedata(months)
    else:
        from strategy_lab.universes import FX_MAJORS
        pairs = args.symbols or FX_MAJORS
        days = pd.bdate_range(FOREXITE_FROM, datetime.now(timezone.utc).date() - pd.Timedelta(days=1))
        on_disk = {f.stem for f in FOREXITE_DIR.glob("*.zip")}
        rep = {"pairs": len(pairs), "requests_planned": int(sum(f"{d:%Y-%m-%d}" not in on_disk for d in days))}
        if not args.dry_run:
            rep.update(fetch_forexite(days, args.max_requests))
            rep["stored"] = store_forexite(pairs)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
