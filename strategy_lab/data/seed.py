"""Seed the store from the previous project's on-disk bars. No network.

TwelveData (US stocks/ETFs and FX):
  * files for the same symbol/interval overlap; they are unioned, the file ending latest wins on a conflict;
  * stock hourly bars before 2020-06-29 are re-stamped where they really are (refresh.td_equity_hourly_starts: until
    2019 their labels are an hour or half an hour off, in the first half of 2020 they cover clock hours); a bar
    off the grid of its period, left by overlapping files of different grids, is dropped; a stock's hourly bars are
    then put on its daily bars' basis (refresh.td_hourly_on_daily_basis);
  * the vendor's FX daily bars carry open == close on most rows of 2022-2024 and its 4h bars used another
    grid before 2021 (with weekend bars), so FX daily and all 4h bars are built from 1h here;
  * bars outside the regular session (stocks) or the FX week are dropped.
Binance (USD-M perps and spot): monthly archives, bars restamped to their close; rows dated before 2017
(a timestamp-unit misparse in one archive) are dropped. Perp funding settlements are stored alongside.

    python -m strategy_lab.data.seed [--only td|perp|spot] [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from strategy_lab import log
from strategy_lab.config import LEGACY_PROJECT_DIR, LOGS_DIR
from strategy_lab.data import calendars as cal
from strategy_lab.data import integrity, resample, store
from strategy_lab.data.refresh import td_equity_hourly_starts, td_hourly_on_daily_basis

LOG = log.get("seed")

CRYPTO_FLOOR = pd.Timestamp("2017-01-01", tz="UTC")
TD_FILE = re.compile(r"^(?P<sym>.+)_(?P<interval>1h|1day)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.parquet$")
FX_LEGACY = re.compile(r"^[A-Z]{3}-[A-Z]{3}$")
BINANCE_TF = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}


def _td_symbol(legacy: str) -> str:
    return legacy.replace("-", "/") if FX_LEGACY.match(legacy) else legacy


def _td_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df["dollar_volume"] = df["close"] * df.get("volume", 0.0)
    if "volume" not in df:
        df["volume"] = 0.0
    return df[store.BAR_COLUMNS]


def _union(files: list[Path]) -> pd.DataFrame:
    frames = [_td_frame(pd.read_parquet(f)) for f in files]          # files sorted by end date ascending
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


def _report(rows: list, source: str, tf: str, symbol: str, n_in: int, df: pd.DataFrame, dropped: dict) -> None:
    rows.append({"source": source, "timeframe": tf, "symbol": symbol, "rows_in": n_in, "rows_out": len(df),
                 "first": df.index.min() if len(df) else "", "last": df.index.max() if len(df) else "",
                 **{f"dropped_{k}": v for k, v in dropped.items()}})


def _write(rows, source, tf, symbol, df, n_in, dropped) -> None:
    df, bad = integrity.clean_bars(df)
    dropped = {**dropped, **bad}
    if len(df):
        store.write_bars(source, tf, symbol, df)
        store.write_meta(source, tf, symbol, {"seeded_from": "legacy", "first": df.index.min(), "last": df.index.max()})
    _report(rows, source, tf, symbol, n_in, df, dropped)


def seed_td_symbol(symbol: str, files_1h: list[Path], files_1d: list[Path]) -> list[dict]:
    """A TwelveData symbol's daily bars, then its hourly bars (a stock's put on its daily bars' basis), then the 4h bars
    (and an FX pair's daily bars) built from the hourly ones."""
    rows: list[dict] = []
    fx = "/" in symbol
    d1 = None
    if files_1d and not fx:
        raw = _union(files_1d)
        n_in = len(raw)
        close = cal.equity_daily_close(raw.index)
        ok = close.notna().to_numpy()
        d1 = raw[ok].copy()
        d1.index = pd.DatetimeIndex(close[ok])
        _write(rows, "td", "1d", symbol, d1, n_in, {"not_a_session": int((~ok).sum())})
        d1 = store.read_bars("td", "1d", symbol) if store.path("td", "1d", symbol).exists() else None
    if files_1h:
        raw = _union(files_1h)
        n_in = len(raw)
        if fx:
            keep = cal.fx_in_session(raw.index)
            h1 = raw[keep.to_numpy()].copy()
            h1.index = h1.index + pd.Timedelta(hours=1)
            dropped = {"outside_fx_week": int((~keep).sum())}
        else:
            starts = td_equity_hourly_starts(raw.index)
            on_grid = starts.notna().to_numpy()
            raw = raw[on_grid].copy()
            raw.index = pd.DatetimeIndex(starts[on_grid])
            close = cal.equity_intraday_close(raw.index)
            ok = close.notna().to_numpy()
            h1 = raw[ok].copy()
            h1.index = pd.DatetimeIndex(close[ok])
            dropped = {"off_grid": int((~on_grid).sum()), "outside_session": int((~ok).sum())}
            if d1 is not None:
                h1, basis = td_hourly_on_daily_basis(integrity.clean_bars(h1)[0], d1)
                dropped.update({f"daily_basis_{k}": v for k, v in basis.items() if k != "sessions_compared"})
        _write(rows, "td", "1h", symbol, h1, n_in, dropped)
        h1 = store.read_bars("td", "1h", symbol) if store.path("td", "1h", symbol).exists() else h1.iloc[:0]
        h4 = resample.fx_4h_from_1h(h1) if fx else resample.equity_4h_from_1h(h1)
        _write(rows, "td", "4h", symbol, h4, len(h1), {})
        if fx:
            _write(rows, "td", "1d", symbol, resample.fx_1d_from_1h(h1), len(h1), {})
    return rows


def _binance_months(folder: Path) -> pd.DataFrame | None:
    frames = [pd.read_parquet(f) for f in sorted(folder.glob("*.parquet")) if f.stat().st_size > 0]
    frames = [f for f in frames if len(f)]
    if not frames:
        return None
    df = pd.concat(frames)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[~df.index.duplicated(keep="last")].sort_index()


def seed_binance_symbol(source: str, symbol: str, root: Path, funding_root: Path | None) -> list[dict]:
    rows: list[dict] = []
    for tf, width in BINANCE_TF.items():
        folder = root / symbol / tf
        if not folder.exists():
            continue
        raw = _binance_months(folder)
        if raw is None:
            continue
        bad_ts = raw.index < CRYPTO_FLOOR
        df = raw[~bad_ts].copy()
        df.index = df.index + width
        df["dollar_volume"] = df["quote_volume"] if "quote_volume" in df else df["close"] * df["volume"]
        _write(rows, source, tf, symbol, df[store.BAR_COLUMNS], len(raw), {"timestamp_before_2017": int(bad_ts.sum())})
    if funding_root is not None and (funding_root / symbol).exists():
        f = _binance_months(funding_root / symbol)
        if f is not None and "last_funding_rate" in f:
            out = f[["last_funding_rate"]].rename(columns={"last_funding_rate": "rate"}).astype("float64")
            out.index.name = "settle_time"
            p = store.STORE_DIR / "perp" / "funding" / f"{symbol}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(p)
    return rows


def _td_tasks() -> dict[str, dict[str, list[Path]]]:
    groups: dict[str, dict[str, list[tuple[str, Path]]]] = defaultdict(lambda: {"1h": [], "1day": []})
    for f in (LEGACY_PROJECT_DIR / "data/raw/twelvedata").glob("*.parquet"):
        m = TD_FILE.match(f.name)
        if m:
            groups[_td_symbol(m["sym"])][m["interval"]].append((m["end"], f))
    for f in (LEGACY_PROJECT_DIR / "data/raw/equity_td").glob("*_1d.parquet"):
        sym = f.name[: -len("_1d.parquet")]
        if "=" not in sym:                                            # FX there is the vendor daily: not used
            groups[sym]["1day"].append(("0000-00-00", f))             # oldest: twelvedata files override it
    return {s: {k: [p for _, p in sorted(v)] for k, v in d.items()} for s, d in groups.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["td", "perp", "spot"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    log.setup("seed")
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = []
        if args.only in (None, "td"):
            for sym, d in _td_tasks().items():
                futs.append(pool.submit(seed_td_symbol, sym, d["1h"], d["1day"]))
        for source, sub in (("perp", "futures/um/klines"), ("spot", "spot/klines")):
            if args.only not in (None, source):
                continue
            root = LEGACY_PROJECT_DIR / "data/raw" / sub
            funding = LEGACY_PROJECT_DIR / "data/raw/futures/um/fundingRate" if source == "perp" else None
            for d in sorted(p for p in root.iterdir() if p.is_dir()):
                futs.append(pool.submit(seed_binance_symbol, source, d.name, root, funding))
        LOG.info("seeding %d symbols with %d workers", len(futs), args.workers)
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.extend(fut.result())
            except Exception:
                LOG.exception("a seeding task failed")
            if i % 200 == 0:
                LOG.info("%d/%d done", i, len(futs))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOGS_DIR / "seed_report.csv"
    fields = sorted({k for r in rows for k in r}, key=lambda k: (not k.startswith(("source", "timeframe", "symbol", "rows")), k))
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(int)
    for r in rows:
        if r["rows_out"]:
            by[(r["source"], r["timeframe"])] += 1
    for (src, tf), n in sorted(by.items()):
        LOG.info("stored %-4s %-2s : %d symbols", src, tf, n)
    LOG.info("report: %s", out)


if __name__ == "__main__":
    main()
