"""The app's database: every evaluation with its figures and series, and the runs that produce them.

    conn = db.connect()                       # db/app.sqlite, created and brought to the current schema on first use
    db.save_list_result(conn, ev, description, fill="next_open", seconds=12.3)

One SQLite file in WAL mode (db_schema.sql beside this module says what each table holds). The batch processes write it,
one transaction per result (per run of a list's instruments alone), so a reader never sees half of one and never waits
for a writer; writers queue behind each other (`BUSY_MS`). The schema is applied in forward-only steps counted by
PRAGMA user_version, inside one IMMEDIATE transaction, so processes opening a new file at once apply it once.

What is stored is what an evaluation computed, and a few figures the writer works out from what it saves: a record's
K-ratio and its average month at the target's drawdown from its series, a buy & hold's from its own, the trades' average
win, average loss and profit factor from them; such a figure is not the evaluation's, so its code is not in the results' stamp, and a schema step gives it to the results saved before. What is
judged from it — a robustness check's pass or fail, the deflated Sharpe, whether a result is stale — is worked out when
it is read (board, the server), so a threshold can change without a re-run.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit

from strategy_lab import log, metrics
from strategy_lab.config import FIRM_TARGETS, ROOT_DIR

LOG = log.get("db")
# STRATEGY_LAB_DB points a process at another file: a test's, or the server's `--db` passed on to the runs it starts
DB_PATH = Path(os.environ.get("STRATEGY_LAB_DB") or ROOT_DIR / "db" / "app.sqlite")
SCHEMA = Path(__file__).with_name("db_schema.sql")
# (version, SQL): applied in order to an older file; a step's SQL can call k_ratio(days, series) and at_target_dd(...)
# (`migrate`)
MIGRATIONS: tuple[tuple[int, Path], ...] = ((1, SCHEMA), (2, SCHEMA.with_name("db_schema_2.sql")),
                                            (3, SCHEMA.with_name("db_schema_3.sql")),
                                            (4, SCHEMA.with_name("db_schema_4.sql")),
                                            (5, SCHEMA.with_name("db_schema_5.sql")),
                                            (6, SCHEMA.with_name("db_schema_6.sql")))
BUSY_MS = 60_000                   # how long a writer waits for another's transaction before it gives up
JOURNAL_LIMIT = 64 * 1024 * 1024   # the WAL file is cut back to this after a checkpoint
CHANGE_DAYS = 1                    # the change log the server reads keeps a day: a page that was away reloads everything

# a result's figures (result_figures): metrics.core, the record's K-ratio (`k_ratio`) and its average month at the
# target's drawdown (`at_target_dd`), metrics.trade_stats, metrics.exposure_stats, and the trades' average win, average
# loss and profit factor (`deals`); a buy & hold's (benchmark): metrics.core, its K-ratio and its average month at the
# target's drawdown, fully invested (HELD_GROSS). Tests hold these to the keys the functions return
CORE = ("start", "end", "months", "avg_monthly", "median_monthly", "pct_green", "pct_red", "months_active",
        "pct_months_active", "pct_green_active", "pct_red_active", "worst_month", "best_month", "longest_red_streak",
        "cagr", "ann_vol", "sharpe", "sortino", "max_dd", "max_dd_days")
STEADINESS = ("k_ratio",)
TRADE_STATS = ("n_trades", "trades_per_month", "win_rate", "avg_trade", "median_trade_bars")
AT_TARGET_DD = ("at_target_dd", "at_target_dd_multiple", "at_target_dd_5y")
EXPOSURE = ("time_in_market", "avg_gross")
DEALS = ("avg_win", "avg_loss", "profit_factor")
FIGURES = CORE + STEADINESS + AT_TARGET_DD + TRADE_STATS + EXPOSURE + DEALS
BENCHMARK_FIGURES = CORE + STEADINESS + AT_TARGET_DD
HELD_GROSS = 1.0                   # buy & hold's exposure: all its equity in what it holds, every day
TARGETS = {"avg_monthly>=1.5%": "target_avg_monthly", "green_months>=70%": "target_green_months",
           "max_dd>=-10%": "target_max_dd", "sharpe>=1": "target_sharpe", "beats_buy_and_hold": "target_beats_bh"}
BH = ("avg_monthly", "pct_green", "sharpe", "max_dd", "cagr")
MC_FIGURES = ("avg_monthly", "pct_green_active", "max_dd", "sharpe", "worst_month")
CHECKS = ("luck", "vs_hold", "timing", "pbo", "plateau", "eras", "delay", "costs", "names", "neighbour_lists", "seeds",
          "vs_rule")
TRADE_COLUMNS = ("instrument", "side", "entry_time", "exit_time", "entry_px", "exit_px", "bars", "gross_return",
                 "net_return", "exit_reason")


# ---------------------------------------------------------------------------------------------------- connection
def connect(path: Path | None = None, *, readonly: bool = False) -> sqlite3.Connection:
    """A connection in autocommit mode (transactions are opened explicitly, `write`), rows by column name. A writer's
    connection brings the file to the current schema first; a reader's refuses an older file and cannot write."""
    path = Path(path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=BUSY_MS / 1000, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    if readonly:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version < MIGRATIONS[-1][0]:
            conn.close()
            raise RuntimeError(f"{path} is at schema {version}, the code needs {MIGRATIONS[-1][0]}: "
                               "run `python -m strategy_lab db migrate`")
        conn.execute("PRAGMA query_only = ON")
        return conn
    if conn.execute("PRAGMA journal_mode = WAL").fetchone()[0] != "wal":
        LOG.warning("%s did not switch to WAL: readers and writers will wait for each other", path)
    conn.execute(f"PRAGMA journal_size_limit = {JOURNAL_LIMIT}")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    """Apply the schema steps the file has not had, in one IMMEDIATE transaction; returns the schema version. A step
    works out a new figure of the results already saved from the series kept with them: its SQL calls
    k_ratio(days, series), `k_ratio` of a series as this file keeps it, and at_target_dd(first_day, days, series,
    avg_gross, time_in_market), `at_target_dd` of one, as a JSON array of the figures in AT_TARGET_DD's order."""
    if conn.execute("PRAGMA user_version").fetchone()[0] >= MIGRATIONS[-1][0]:
        return MIGRATIONS[-1][0]
    conn.create_function("k_ratio", 2, lambda days, blob: number(k_ratio(_values(days, blob))), deterministic=True)
    conn.create_function("at_target_dd", 5, _at_target_dd_sql, deterministic=True)
    with write(conn):
        version = conn.execute("PRAGMA user_version").fetchone()[0]     # another process may have applied it meanwhile
        for step, sql in MIGRATIONS:
            if step > version:
                LOG.info("database schema %d -> %d (%s)", version, step, sql.name)
                for statement in _statements(sql.read_text()):
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {step}")
                version = step
    return version


def _statements(script: str) -> list[str]:
    """A script's statements one by one (execute() takes one; executescript() commits the open transaction)."""
    out, cur = [], []
    for line in script.splitlines():
        if line.strip().startswith("--") and not cur:
            continue
        cur.append(line)
        text = "\n".join(cur)
        if sqlite3.complete_statement(text):
            out.append(text.strip())
            cur = []
    if "".join(cur).strip():
        raise ValueError(f"an unfinished statement at the end of the schema: {''.join(cur)[:80]}")
    return out


@contextmanager
def write(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One IMMEDIATE transaction: it holds the write lock from its start, so it never fails half-way on another's."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------------------------------- values
def number(v) -> float | int | None:
    """A figure as SQLite keeps it: a finite int or float, else None (NaN, ±inf, missing)."""
    if v is None or isinstance(v, str):
        return None
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    f = float(v)
    return f if math.isfinite(f) else None


def plain(v):
    """A value as JSON writes it: numpy scalars as Python's, tuples as lists, non-finite floats as null; anything else
    JSON cannot hold is an error, not a string."""
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    if v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(v).date())
    raise TypeError(f"{type(v).__name__} has no JSON form here: {v!r}")


def to_json(v) -> str:
    return json.dumps(plain(v), allow_nan=False, sort_keys=True)


def day(ts) -> str:
    return str(pd.Timestamp(ts).date())


# ---------------------------------------------------------------------------------------------------- series
def encode_series(s: pd.Series) -> tuple[str, int, bytes]:
    """A daily series, one value a UTC day with none missing, as (first day, days, zstd of the float64 values)."""
    s = s.astype(np.float64)
    idx = pd.DatetimeIndex(s.index)
    if len(idx) == 0:
        raise ValueError("an empty series has nothing to store")
    first = idx[0].tz_convert("UTC") if idx.tz is not None else idx[0].tz_localize("UTC")
    expected = pd.date_range(first.normalize(), periods=len(idx), freq="D", tz="UTC")
    got = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    if not got.equals(expected):
        raise ValueError(f"a stored series has one value per UTC day from its first; this one runs {got[0]} .. "
                         f"{got[-1]} in {len(got)} values")
    raw = np.ascontiguousarray(s.to_numpy(), dtype="<f8").tobytes()
    return day(first), len(idx), pa.compress(raw, codec="zstd", asbytes=True)


def decode_series(first_day: str, days: int, blob: bytes, name: str | None = None) -> pd.Series:
    return pd.Series(_values(days, blob).copy(), index=pd.date_range(first_day, periods=days, freq="D", tz="UTC"),
                     name=name)


def _values(days: int, blob: bytes) -> np.ndarray:
    """A stored series' values, read-only (its days are worked out from the first by `decode_series`)."""
    return np.frombuffer(pa.decompress(blob, decompressed_size=days * 8, codec="zstd", asbytes=True), dtype="<f8")


def series_sha(first_day: str, s: pd.Series) -> str:
    """What makes two stored series the same: the first day and the values, not their compressed bytes."""
    h = hashlib.sha256(first_day.encode())
    h.update(np.ascontiguousarray(s.to_numpy(), dtype="<f8").tobytes())
    return h.hexdigest()[:32]


def encode_trades(trades: pd.DataFrame) -> bytes:
    missing = [c for c in TRADE_COLUMNS if c not in trades.columns]
    if missing:
        raise ValueError(f"trades without {missing}")
    t = trades[list(TRADE_COLUMNS)].copy()
    for c in ("entry_time", "exit_time"):
        t[c] = pd.to_datetime(t[c], utc=True)
    table = pa.Table.from_pandas(t, preserve_index=False)
    table = table.set_column(0, "instrument", table.column("instrument").dictionary_encode())
    table = table.set_column(len(TRADE_COLUMNS) - 1, "exit_reason", table.column("exit_reason").dictionary_encode())
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


def decode_trades(payload: bytes, columns: Sequence[str] | None = None) -> pd.DataFrame:
    return pq.read_table(io.BytesIO(payload), columns=list(columns) if columns else None).to_pandas()


# ---------------------------------------------------------------------------------------------------- what the record says
def k_ratio(daily: pd.Series | np.ndarray | None) -> float:
    """Kestner's K-ratio of a daily record, as he corrected it in 2013: the least-squares slope of the account's log
    equity against the day, over the slope's standard error, times the square root of the days in a year (365, the
    Sharpe's annualisation here) over the number of days. It measures how steadily the account grew: a record of
    independent days scores about 1.1 times its Sharpe, a steadier climb more, one made in a few jumps or broken by
    long flat or losing stretches less; the corrections of 2013 keep it the same whatever the record's length and
    however often it is sampled. A record's missing days are left out, as its Sharpe leaves them. NaN when it cannot
    be worked out: fewer than three days, or an account that lost everything (its log equity ends there)."""
    if daily is None:
        return np.nan
    v = np.asarray(daily, dtype=np.float64)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 3:
        LOG.debug("no K-ratio of a record of %d days", n)
        return np.nan
    if (v <= -1.0).any():
        LOG.info("no K-ratio of a record that lost everything on its day %d of %d", int(np.argmax(v <= -1.0)) + 1, n)
        return np.nan
    log_equity = np.cumsum(np.log1p(v))
    day = np.arange(n, dtype=np.float64) - (n - 1) / 2.0             # centred: the slope needs no intercept term
    sxx = float(day @ day)
    slope = float(day @ log_equity) / sxx
    residual = log_equity - log_equity.mean() - slope * day
    se = np.sqrt(float(residual @ residual) / (n - 2) / sxx)
    if se == 0.0:
        LOG.debug("a record of %d days on a straight line (slope %g): no spread to scale its slope by", n, slope)
        return 0.0 if slope == 0.0 else np.nan
    return slope / se * np.sqrt(metrics.DAYS) / n


RECENT_DAYS = 5 * 365      # the years every research list has out of sample: the crypto lists' records begin in 2021


def at_target_dd(daily: pd.Series | None, avg_gross: float | None, time_in_market: float | None) -> dict:
    """The record's average month over all its days with its positions scaled so that its max drawdown is the firm's
    target (FIRM_TARGETS["max_dd"]): how records whose returns come with different drawdowns compare, one that fell 25%
    held at 40% of its size, one that fell 5% at twice it. The money the positions leave idle earns nothing, as in the
    record's own average month (a multiple of 1 gives that month back); money borrowed past the equity pays T-bills +
    metrics.FINANCING_SPREAD, counted on the record's average exposure (`avg_gross`; for a record that did not keep it,
    an instrument alone saved before it was kept, its share of days in the market, which bounds it: a position never
    exceeds the capital). The multiple has no ceiling: the figure is what the record earns at the target's drawdown,
    and the borrowing is what the leverage costs; a record that never lost a day has no drawdown to size by and no
    figure. Beside it the multiple, and the same average month over the record's last RECENT_DAYS alone (None for a
    record no longer), the years every research list has: a record of 28 years had more years for its worst drawdown
    to come than one of 5. The multiple is chosen knowing the whole record's worst drawdown: a yardstick to compare
    records by, not a size a forward run could have known."""
    if daily is None:
        return dict.fromkeys(AT_TARGET_DD)
    d = daily.dropna()
    gross = next((g for g in (avg_gross, time_in_market) if g is not None and np.isfinite(g) and g > 0), None)
    if gross is None and (d != 0.0).any():
        LOG.warning("a record of %d days that held positions without an exposure kept (avg_gross %s, time in market "
                    "%s): its sizing to the target's drawdown takes it fully invested", len(d), avg_gross, time_in_market)
        gross = 1.0
    avg, multiple = _sized_to_target(d, gross)
    recent = _sized_to_target(d.iloc[-RECENT_DAYS:], gross)[0] if len(d) > RECENT_DAYS else None
    return {"at_target_dd": number(avg), "at_target_dd_multiple": number(multiple), "at_target_dd_5y": number(recent)}


def _sized_to_target(d: pd.Series, gross: float | None) -> tuple[float, float | None]:
    """(average month, multiple) of a daily record scaled to the target's drawdown (`at_target_dd`); a record without a
    position has nothing to scale: its average month, and no multiple; one that never lost a day, neither."""
    months = len(d) > 62 or len(metrics.monthly_returns(d)) > 0      # as metrics.core: no average without a whole month
    if not months:
        LOG.debug("a record of %d days holds no whole month: no average month to size", len(d))
        return np.nan, None
    values = d.to_numpy(dtype=np.float64)
    if gross is None or not (values != 0.0).any():
        LOG.debug("a record of %d days without a position: nothing to size to the target's drawdown", len(d))
        return metrics.average_month(float(np.prod(1.0 + values)), len(values)), None
    worst_day = float(values.min())
    if worst_day >= 0.0:
        LOG.debug("a record of %d days that never lost a day: no drawdown to size to the target's", len(d))
        return np.nan, None
    borrowing = (metrics.t_bills(d.index) + metrics.FINANCING_SPREAD / metrics.DAYS).to_numpy(dtype=np.float64)
    target = FIRM_TARGETS["max_dd"]
    # a larger multiple draws down deeper, and at 1 / |worst day| that day alone takes everything: the multiple the
    # target needs lies between 0 and that; halve the span until it is exact
    low, high = 0.0, 1.0 / -worst_day
    for _ in range(60):
        mid = 0.5 * (low + high)
        if _scaled(values, borrowing, mid, max(mid * gross - 1.0, 0.0))[0] >= target:
            low = mid
        else:
            high = mid
    multiple = 0.5 * (low + high)
    growth = _scaled(values, borrowing, multiple, max(multiple * gross - 1.0, 0.0))[1]
    return metrics.average_month(growth, len(values)), multiple


@njit(cache=True)
def _scaled(values, borrowing, multiple, borrowed):
    """(max drawdown, end value over start) of the account whose daily returns are `values` times `multiple`, less
    `borrowed` of the equity paying each day's `borrowing` rate; an account that loses everything stops there."""
    equity, peak, worst = 1.0, 1.0, 0.0
    for i in range(len(values)):
        equity *= 1.0 + multiple * values[i] - borrowed * borrowing[i]
        if equity <= 0.0:
            return -1.0, 0.0
        if equity > peak:
            peak = equity
        elif equity / peak - 1.0 < worst:
            worst = equity / peak - 1.0
    return worst, equity


def _at_target_dd_sql(first_day: str, days: int, blob: bytes, avg_gross: float | None,
                      time_in_market: float | None) -> str:
    """`at_target_dd` of a series as this file keeps it, for a schema step's SQL: its figures as a JSON array."""
    figures = at_target_dd(decode_series(first_day, days, blob), avg_gross, time_in_market)
    return json.dumps([figures[k] for k in AT_TARGET_DD])


# ---------------------------------------------------------------------------------------------------- what trades say
def deals(trades: pd.DataFrame | None) -> dict:
    """The trades' average win, average loss and profit factor, after costs (None without them)."""
    if trades is None or trades.empty:
        return {"avg_win": None, "avg_loss": None, "profit_factor": None}
    r = trades["net_return"]
    win, loss = r[r > 0], r[r < 0]
    return {"avg_win": number(win.mean()) if len(win) else None, "avg_loss": number(loss.mean()) if len(loss) else None,
            "profit_factor": number(win.sum() / -loss.sum()) if len(loss) else None}


def by_name(trades: pd.DataFrame | None) -> list[tuple[str, int, float, float | None]]:
    """Each instrument's trades in the book: (instrument, trades, share that made money, returns compounded)."""
    if trades is None or trades.empty:
        return []
    r = trades["net_return"]
    g = trades.assign(lr=np.log1p(r.clip(lower=-0.999999)), won=r > 0).groupby("instrument").agg(
        n=("lr", "size"), won=("won", "mean"), lr=("lr", "sum"))
    return [(str(i), int(x.n), float(x.won), number(np.expm1(x.lr))) for i, x in g.iterrows()]


# ---------------------------------------------------------------------------------------------------- writing results
@dataclass
class Figures:
    """One scope's figures: a scorecard (or the part of one an older file kept) and the trades' deals."""
    card: dict
    deals: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {k: (self.card.get(k) if k in ("start", "end") else number(self.card.get(k))) for k in CORE + TRADE_STATS
                + EXPOSURE} | {k: number(self.deals.get(k)) for k in DEALS}


@dataclass
class Outcome:
    """One result as the writer takes it: its key, what the evaluation computed, and the series behind it."""
    strategy: str
    list_id: str
    timeframe: str
    instrument_id: str | None
    card: dict                              # the out-of-sample scorecard (metrics.scorecard)
    oos: pd.Series
    in_sample: Figures | None = None
    oos_deals: dict = field(default_factory=dict)
    in_sample_daily: pd.Series | None = None
    position: pd.Series | None = None       # an instrument alone: its position at each day's end
    benchmark: pd.Series | None = None      # buy-and-hold over the record's days; None when it is cash
    windows: list[dict] = field(default_factory=list)
    names: list[tuple] = field(default_factory=list)
    trades: pd.DataFrame | None = None
    monte_carlo: dict = field(default_factory=dict)
    random_timing: dict = field(default_factory=dict)
    robustness: dict | None = None
    notes: list[str] = field(default_factory=list)
    grid: dict | None = None
    params_in_sample: dict | None = None
    params_now: dict | None = None
    best5_share: float | None = None
    seconds: float | None = None            # how long the evaluation took (`batch.expected` reads it)
    candidates: list[tuple[int, int]] = field(default_factory=list)       # strategy_pick: (candidate result id, windows)


@dataclass
class Encoded:
    """An outcome's rows and blobs, worked out before the write lock is taken."""
    row: dict
    figures: list[tuple[str, dict]]
    series: list[tuple[str, str, int, bytes]]
    benchmark: tuple[str, str, int, bytes, dict] | None
    windows: list[tuple]
    names: list[tuple]
    trades: tuple[int, bytes] | None
    monte_carlo: dict | None
    checks: list[tuple]
    candidates: list[tuple[int, int]]


def _encode(o: Outcome, *, code_sha: str, fill: str | None, evaluated_at: str) -> Encoded:
    card = o.card
    targets = card.get("targets") or {}
    row = {"strategy": o.strategy, "list_id": o.list_id, "instrument_id": o.instrument_id, "timeframe": o.timeframe,
           "evaluated_at": evaluated_at, "code_sha": code_sha, "fill": fill, "seconds": number(o.seconds),
           "notes": to_json(list(o.notes)), "grid": None if o.grid is None else to_json(o.grid),
           "params_in_sample": None if o.params_in_sample is None else to_json(o.params_in_sample),
           "params_now": None if o.params_now is None else to_json(o.params_now),
           "vs_bh": number(card.get("vs_bh")), "beats_bh": int(bool(card.get("beats_bh", False))),
           "bh_cash": None if card.get("bh_cash") is None else int(bool(card["bh_cash"])),
           **{f"bh_{k}": number(card.get(f"bh_{k}")) for k in BH},
           **{col: int(bool(targets.get(k, False))) for k, col in TARGETS.items()},
           "targets_met": int(card.get("targets_met", sum(bool(targets.get(k)) for k in TARGETS))),
           "rt_sharpe": number(o.random_timing.get("sharpe")), "rt_null_median": number(o.random_timing.get("null_median")),
           "rt_null_p95": number(o.random_timing.get("null_p95")), "rt_p": number(o.random_timing.get("p")),
           "rt_rotations": number(o.random_timing.get("rotations")), "best5_share": number(o.best5_share)}
    figures = [("out_of_sample", Figures(card, o.oos_deals).row() | {"k_ratio": number(k_ratio(o.oos))}
                | at_target_dd(o.oos, card.get("avg_gross"), card.get("time_in_market")))]
    if o.in_sample is not None:     # without its series kept (an instrument alone), the in-sample record has neither
        figures.append(("in_sample", o.in_sample.row() | {"k_ratio": number(k_ratio(o.in_sample_daily))}
                        | at_target_dd(o.in_sample_daily, o.in_sample.card.get("avg_gross"),
                                       o.in_sample.card.get("time_in_market"))))
    series = [("out_of_sample", *encode_series(o.oos))]
    if o.in_sample_daily is not None:
        series.append(("in_sample", *encode_series(o.in_sample_daily)))
    if o.position is not None:
        series.append(("position", *encode_series(o.position)))
    bench = None
    if o.benchmark is not None and not card.get("bh_cash"):
        first, days, blob = encode_series(o.benchmark)
        core = metrics.core(o.benchmark)            # the series' own figures, as the scorecard took its bh_* from it
        bench = (series_sha(first, o.benchmark), first, days, blob,
                 {k: (core[k] if k in ("start", "end") else number(core[k])) for k in CORE}
                 | {"k_ratio": number(k_ratio(o.benchmark))} | at_target_dd(o.benchmark, HELD_GROSS, HELD_GROSS))
    windows = [(n, day(w["train_start"]), day(w["train_end"]), day(w["test_start"]), day(w["test_end"]),
                number(w.get("train_sharpe_daily")),
                to_json(w["params"] if isinstance(w["params"], dict) else json.loads(w["params"])))
               for n, w in enumerate(o.windows)]
    trades = None if o.trades is None else (int(len(o.trades)), encode_trades(o.trades))
    mc = None
    if o.monte_carlo:
        m = o.monte_carlo
        mc = {"reps": int(m.get("reps", 0)), "block_days": number(m.get("block_days")), "note": m.get("note"),
              "prob_avg_monthly_below_target": number(m.get("prob_avg_monthly_below_target")),
              "prob_max_dd_beyond_target": number(m.get("prob_max_dd_beyond_target"))}
        for k in MC_FIGURES:
            q = m.get(k) or {}
            mc |= {f"{k}_{p}": number(q.get(p)) for p in ("p5", "p50", "p95")}
    checks = []
    if o.robustness is not None:
        seconds_of = o.robustness.get("seconds") or {}
        unknown = sorted(set(o.robustness) - set(CHECKS) - {"seconds"})
        if unknown:
            LOG.warning("%s on %s %s: robustness measured %s, which the database does not keep", o.strategy,
                        o.list_id, o.timeframe, unknown)
        for cid in CHECKS:
            m = o.robustness.get(cid)
            checks.append((cid, None if m is None else to_json(m), number(seconds_of.get(cid))))
    return Encoded(row, figures, series, bench, windows, [(i, n, w, c) for i, n, w, c in o.names], trades, mc, checks,
                   list(o.candidates))


def stamp_row(conn: sqlite3.Connection, code: dict) -> str:
    conn.execute("INSERT INTO code_stamp (sha, files) VALUES (?, ?) ON CONFLICT (sha) DO NOTHING",
                 (code["sha"], to_json(list(code["files"]))))
    return code["sha"]


def strategy_row(conn: sqlite3.Connection, name: str, description: str) -> None:
    conn.execute("INSERT INTO strategy (name, description, updated_at) VALUES (?, ?, ?) ON CONFLICT (name) DO UPDATE "
                 "SET description = excluded.description, updated_at = excluded.updated_at",
                 (name, description or "", now()))


_RESULT_COLUMNS = ("strategy", "list_id", "instrument_id", "timeframe", "evaluated_at", "code_sha", "fill",
                   "seconds", "notes", "grid", "params_in_sample", "params_now", "vs_bh", "beats_bh", "bh_cash",
                   *(f"bh_{k}" for k in BH), *TARGETS.values(), "targets_met", "rt_sharpe", "rt_null_median",
                   "rt_null_p95", "rt_p", "rt_rotations", "best5_share", "benchmark_id")
_UPSERT = (f"INSERT INTO result ({', '.join(_RESULT_COLUMNS)}) VALUES ({', '.join('?' * len(_RESULT_COLUMNS))}) "
           "ON CONFLICT (strategy, list_id, instrument_key, timeframe) DO UPDATE SET "
           + ", ".join(f"{c} = excluded.{c}" for c in _RESULT_COLUMNS[4:]) + " RETURNING id")


def _benchmark_id(conn: sqlite3.Connection, e: Encoded) -> int | None:
    if e.benchmark is None:
        return None
    sha, first, days, blob, figures = e.benchmark
    held = {"list_id": e.row["list_id"] if e.row["instrument_id"] is None else None,
            "instrument_id": e.row["instrument_id"]}
    cols = ("list_id", "instrument_id", "timeframe", "first_day", "days", "series_sha", "series", *BENCHMARK_FIGURES)
    vals = (held["list_id"], held["instrument_id"], e.row["timeframe"], first, days, sha, blob,
            *(figures[k] for k in BENCHMARK_FIGURES))
    conn.execute(f"INSERT INTO benchmark ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
                 "ON CONFLICT DO NOTHING", vals)
    return conn.execute("SELECT id FROM benchmark WHERE coalesce(list_id, '') = ? AND coalesce(instrument_id, '') = ? "
                        "AND timeframe = ? AND series_sha = ?",
                        (held["list_id"] or "", held["instrument_id"] or "", e.row["timeframe"], sha)).fetchone()[0]


def _put(conn: sqlite3.Connection, e: Encoded) -> int:
    """One encoded outcome into the open transaction: the result upserted (its id kept), its rows replaced."""
    before = conn.execute("SELECT id, benchmark_id FROM result WHERE strategy = ? AND list_id = ? AND instrument_key = ? "
                          "AND timeframe = ?", (e.row["strategy"], e.row["list_id"], e.row["instrument_id"] or "",
                                                e.row["timeframe"])).fetchone()
    row = e.row | {"benchmark_id": _benchmark_id(conn, e)}
    rid = conn.execute(_UPSERT, [row[c] for c in _RESULT_COLUMNS]).fetchone()[0]
    for table in ("result_figures", "result_series", "result_window", "result_name", "result_trades",
                  "result_monte_carlo", "result_check", "pick_candidate"):
        conn.execute(f"DELETE FROM {table} WHERE result_id = ?", (rid,))
    for scope, fig in e.figures:
        cols = ("result_id", "scope", *FIGURES)
        conn.execute(f"INSERT INTO result_figures ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                     (rid, scope, *(fig[k] for k in FIGURES)))
    conn.executemany("INSERT INTO result_series (result_id, kind, first_day, days, series) VALUES (?, ?, ?, ?, ?)",
                     [(rid, *s) for s in e.series])
    conn.executemany("INSERT INTO result_window (result_id, n, train_start, train_end, test_start, test_end, "
                     "train_sharpe_daily, params) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [(rid, *w) for w in e.windows])
    conn.executemany("INSERT INTO result_name (result_id, instrument_id, trades, won, compounded) VALUES (?, ?, ?, ?, ?)",
                     [(rid, *n) for n in e.names])
    if e.trades is not None:
        conn.execute("INSERT INTO result_trades (result_id, n_trades, payload) VALUES (?, ?, ?)", (rid, *e.trades))
    if e.monte_carlo is not None:
        cols = tuple(e.monte_carlo)
        conn.execute(f"INSERT INTO result_monte_carlo (result_id, {', '.join(cols)}) VALUES "
                     f"({', '.join('?' * (len(cols) + 1))})", (rid, *(e.monte_carlo[c] for c in cols)))
    conn.executemany("INSERT INTO result_check (result_id, check_id, measure, seconds) VALUES (?, ?, ?, ?)",
                     [(rid, *c) for c in e.checks])
    conn.executemany("INSERT INTO pick_candidate (result_id, candidate_id, windows) VALUES (?, ?, ?)",
                     [(rid, c, w) for c, w in e.candidates])
    if before is not None and before["benchmark_id"] is not None and before["benchmark_id"] != row["benchmark_id"]:
        _drop_unused_benchmark(conn, before["benchmark_id"])
    return rid


def _drop_unused_benchmark(conn: sqlite3.Connection, bid: int) -> None:
    if conn.execute("SELECT 1 FROM result WHERE benchmark_id = ? LIMIT 1", (bid,)).fetchone() is None:
        conn.execute("DELETE FROM benchmark WHERE id = ?", (bid,))


def _drop_results(conn: sqlite3.Connection, rows: Sequence[sqlite3.Row]) -> None:
    """Results (their id and benchmark_id) deleted inside the caller's transaction: their rows in the other tables go
    with them (ON DELETE CASCADE), and a buy & hold no other result reads."""
    for r in rows:
        conn.execute("DELETE FROM result WHERE id = ?", (r["id"],))
        if r["benchmark_id"] is not None:
            _drop_unused_benchmark(conn, r["benchmark_id"])


def drop_alone_runs(conn: sqlite3.Connection, runs: Sequence[tuple[str, str, str]]) -> int:
    """Runs of a strategy on each instrument of a list alone, (strategy, list, timeframe), deleted whole with their
    too-short instruments, in one transaction; returns how many results went."""
    n = 0
    with write(conn):
        for s, u, tf in runs:
            rows = conn.execute("SELECT id, benchmark_id FROM result WHERE strategy = ? AND list_id = ? AND timeframe = ? "
                                "AND instrument_id IS NOT NULL", (s, u, tf)).fetchall()
            _drop_results(conn, rows)
            conn.execute("DELETE FROM asset_too_short WHERE strategy = ? AND list_id = ? AND timeframe = ?", (s, u, tf))
            n += len(rows)
        _prune_changes(conn)
    return n


def _prune_changes(conn: sqlite3.Connection) -> None:
    cutoff = (datetime.now(timezone.utc) - pd.Timedelta(days=CHANGE_DAYS)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    conn.execute("DELETE FROM change WHERE at < ?", (cutoff,))


def save_outcomes(conn: sqlite3.Connection, outcomes: Sequence[Outcome], *, description: str, code: dict,
                  fill: str | None, evaluated_at: str | None = None, replace_run: tuple[str, str, str] | None = None,
                  too_short: Sequence[tuple[str, int]] = ()) -> list[int]:
    """Results of one strategy, saved in one transaction. `replace_run` (strategy, list, timeframe): the outcomes are a
    run of the list's instruments alone, which replaces that run's earlier results (an instrument no longer in the list
    goes with them) and its too-short instruments (`too_short`: (instrument, months))."""
    when = evaluated_at or now()
    encoded = [_encode(o, code_sha=code["sha"], fill=fill, evaluated_at=when) for o in outcomes]
    strategy = outcomes[0].strategy if outcomes else (replace_run[0] if replace_run else None)
    t0 = time.perf_counter()
    with write(conn):
        stamp_row(conn, code)
        if strategy is not None:
            strategy_row(conn, strategy, description)
        ids = [_put(conn, e) for e in encoded]
        if replace_run is not None:
            s, u, tf = replace_run
            kept = [e.row["instrument_id"] for e in encoded]
            others = f"AND instrument_id NOT IN ({', '.join('?' * len(kept))})" if kept else ""
            _drop_results(conn, conn.execute("SELECT id, benchmark_id FROM result WHERE strategy = ? AND list_id = ? AND "
                                             f"timeframe = ? AND instrument_id IS NOT NULL {others}",
                                             (s, u, tf, *kept)).fetchall())
            conn.execute("DELETE FROM asset_too_short WHERE strategy = ? AND list_id = ? AND timeframe = ?", (s, u, tf))
            conn.executemany("INSERT INTO asset_too_short (strategy, list_id, instrument_id, timeframe, months, "
                             "evaluated_at, code_sha) VALUES (?, ?, ?, ?, ?, ?, ?)",
                             [(s, u, i, tf, int(m), when, code["sha"]) for i, m in too_short])
        _prune_changes(conn)
    LOG.debug("saved %d results of %s in %.0f ms (write lock)", len(ids), strategy, (time.perf_counter() - t0) * 1e3)
    return ids


def outcome_of(ev, *, seconds: float | None = None) -> Outcome:
    """A list's evaluation (evaluate.Evaluation) as the writer takes it."""
    windows = ev.folds.to_dict(orient="records") if len(ev.folds) else []
    now_params = json.loads(windows[-1]["params"]) if windows else ev.params_in_sample
    return Outcome(
        strategy=ev.strategy, list_id=ev.universe, timeframe=ev.timeframe, instrument_id=None, card=ev.oos,
        oos=ev.oos_daily, in_sample=Figures(ev.in_sample, deals(None)), oos_deals=deals(ev.trades),
        in_sample_daily=ev.is_daily, benchmark=None if ev.oos.get("bh_cash") else ev.bh_daily, windows=windows,
        names=by_name(ev.trades), trades=ev.trades, monte_carlo=ev.monte_carlo, random_timing=ev.random_timing,
        robustness=ev.robustness, notes=list(ev.notes), grid=ev.grid or None, params_in_sample=ev.params_in_sample,
        params_now=now_params, seconds=seconds)


def save_list_result(conn: sqlite3.Connection, ev, description: str, *, fill: str, seconds: float | None = None) -> int:
    """A strategy's evaluation on a list as one book (evaluate.Evaluation), in one transaction; returns its id."""
    return save_outcomes(conn, [outcome_of(ev, seconds=seconds)], description=description, code=ev.code, fill=fill)[0]


# ---------------------------------------------------------------------------------------------------- reading
def result_id(conn: sqlite3.Connection, strategy: str, list_id: str, timeframe: str,
              instrument_id: str | None = None) -> int | None:
    r = conn.execute("SELECT id FROM result WHERE strategy = ? AND list_id = ? AND instrument_key = ? AND timeframe = ?",
                     (strategy, list_id, instrument_id or "", timeframe)).fetchone()
    return None if r is None else r[0]


def series(conn: sqlite3.Connection, rid: int, kind: str = "out_of_sample") -> pd.Series | None:
    r = conn.execute("SELECT first_day, days, series FROM result_series WHERE result_id = ? AND kind = ?",
                     (rid, kind)).fetchone()
    return None if r is None else decode_series(r["first_day"], r["days"], r["series"])


def benchmark_series(conn: sqlite3.Connection, bid: int) -> pd.Series:
    r = conn.execute("SELECT first_day, days, series FROM benchmark WHERE id = ?", (bid,)).fetchone()
    return decode_series(r["first_day"], r["days"], r["series"])


def figures(conn: sqlite3.Connection, rid: int, scope: str = "out_of_sample") -> dict | None:
    r = conn.execute("SELECT * FROM result_figures WHERE result_id = ? AND scope = ?", (rid, scope)).fetchone()
    return None if r is None else {k: r[k] for k in FIGURES}


def measures(conn: sqlite3.Connection, rid: int) -> dict | None:
    """A result's robustness as the evaluation measured it: {check: measurement} (a check not measured is absent),
    None for a result that loses money (nothing checked)."""
    rows = conn.execute("SELECT check_id, measure FROM result_check WHERE result_id = ?", (rid,)).fetchall()
    if not rows:
        return None
    return {r["check_id"]: json.loads(r["measure"]) for r in rows if r["measure"] is not None}


def monte_carlo(conn: sqlite3.Connection, rid: int) -> dict:
    """The Monte Carlo figures as montecarlo.bootstrap returns them; {} when it was not run."""
    r = conn.execute("SELECT * FROM result_monte_carlo WHERE result_id = ?", (rid,)).fetchone()
    if r is None:
        return {}
    out = {"reps": r["reps"], "block_days": r["block_days"]} | ({"note": r["note"]} if r["note"] else {})
    if r["reps"]:
        out |= {k: {p: r[f"{k}_{p}"] for p in ("p5", "p50", "p95")} for k in MC_FIGURES}
        out |= {"prob_avg_monthly_below_target": r["prob_avg_monthly_below_target"],
                "prob_max_dd_beyond_target": r["prob_max_dd_beyond_target"]}
    return out


def trades(conn: sqlite3.Connection, rid: int, columns: Sequence[str] | None = None) -> pd.DataFrame | None:
    r = conn.execute("SELECT payload FROM result_trades WHERE result_id = ?", (rid,)).fetchone()
    return None if r is None else decode_trades(r["payload"], columns)


def windows(conn: sqlite3.Connection, rid: int) -> list[dict]:
    return [dict(r) | {"params": json.loads(r["params"])} for r in conn.execute(
        "SELECT n, train_start, train_end, test_start, test_end, train_sharpe_daily, params FROM result_window "
        "WHERE result_id = ? ORDER BY n", (rid,))]


_JSON_COLUMNS = ("notes", "grid", "params_in_sample", "params_now", "code_files")


def results_frame(conn: sqlite3.Connection, kind: str, where: str = "", args: Sequence = ()) -> pd.DataFrame:
    """Results as one table: the result's own columns, its out-of-sample figures under their names, the in-sample
    Sharpe (is_sharpe), Monte Carlo's 5th-percentile Sharpe (mc_sharpe_p5), buy & hold's K-ratio (bh_k_ratio), its
    average month at the target's drawdown with its multiple (bh_at_target_dd, bh_at_target_dd_multiple) and the files
    its code stamp covers. `kind`: "list" (a list as one book) or "asset" (an instrument alone, strategy_pick's
    included); `where` narrows."""
    scope = "r.instrument_id IS NULL" if kind == "list" else "r.instrument_id IS NOT NULL"
    figures = ", ".join(f"o.{k}" for k in FIGURES)
    sql = (f"SELECT r.*, c.files AS code_files, {figures}, i.sharpe AS is_sharpe, mc.sharpe_p5 AS mc_sharpe_p5, "
           "b.k_ratio AS bh_k_ratio, b.at_target_dd AS bh_at_target_dd, "
           "b.at_target_dd_multiple AS bh_at_target_dd_multiple FROM result r JOIN code_stamp c ON c.sha = r.code_sha "
           "LEFT JOIN result_figures o ON o.result_id = r.id AND o.scope = 'out_of_sample' "
           "LEFT JOIN result_figures i ON i.result_id = r.id AND i.scope = 'in_sample' "
           "LEFT JOIN result_monte_carlo mc ON mc.result_id = r.id "
           "LEFT JOIN benchmark b ON b.id = r.benchmark_id "
           f"WHERE {scope} {('AND ' + where) if where else ''} ORDER BY r.id")
    cur = conn.execute(sql, tuple(args))
    names = [d[0] for d in cur.description]
    df = pd.DataFrame([tuple(r) for r in cur.fetchall()], columns=names)
    for c in _JSON_COLUMNS:
        df[c] = [None if v is None else json.loads(v) for v in df[c]]
    flags = df[list(TARGETS.values())].to_numpy(dtype=bool)
    df["targets"] = [dict(zip(TARGETS, row)) for row in flags.tolist()]
    df["beats_bh"] = df["beats_bh"].astype(bool)
    # a column with a NULL comes back as floats with NaN: bool(NaN) is True, so the yes/no/unknown one stays None or
    # bool, and counts stay whole numbers
    df["bh_cash"] = pd.Series([None if v is None or v != v else bool(v) for v in df["bh_cash"]], dtype=object,
                              index=df.index)
    for c in ("months", "n_trades", "targets_met", "max_dd_days", "longest_red_streak", "months_active"):
        df[c] = df[c].astype("Int64")
    return df


def measures_of(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, dict]:
    """`measures` of many results at once: {result id: {check: measurement}}; a result without checks is absent."""
    out: dict[int, dict] = {}
    ids = [int(i) for i in ids]
    for k in range(0, len(ids), 900):                     # SQLite's limit of bound values a statement
        chunk = ids[k:k + 900]
        for r in conn.execute(f"SELECT result_id, check_id, measure FROM result_check WHERE result_id IN "
                              f"({', '.join('?' * len(chunk))})", chunk):
            got = out.setdefault(r["result_id"], {})
            if r["measure"] is not None:
                got[r["check_id"]] = json.loads(r["measure"])
    return out


# the results a family of tries is worked out over (`result r`): a list's, a strategy_pick's, a strategy's on an
# instrument alone
FAMILIES = {"lists": "r.instrument_id IS NULL", "picks": "r.strategy = 'strategy_pick'",
            "assets": "r.instrument_id IS NOT NULL AND r.strategy != 'strategy_pick'"}


def results_fingerprint(conn: sqlite3.Connection, family: str = "lists") -> str:
    """What a family's tries were worked out on: how many results, the last id, the last evaluated_at."""
    r = conn.execute(f"SELECT count(*), max(r.id), max(r.evaluated_at) FROM result r WHERE {FAMILIES[family]}").fetchone()
    return f"{r[0]}:{r[1]}:{r[2]}"


def latest_tries(conn: sqlite3.Connection, family: str = "lists") -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM tries WHERE family = ? ORDER BY id DESC LIMIT 1", (family,)).fetchone()


def save_tries(conn: sqlite3.Connection, family: str, covers: str, saved: int, independent: float, best_95: float,
               code_sha: str, instruments: dict[str, tuple[int, float, float]] | None = None) -> int | None:
    """Keep a family's tries worked out on `covers`, unless the results changed while they were worked out (then a
    newer count is due and this one would be wrong the moment it landed); `instruments`: each instrument's own
    (saved, independent, best_95), the single assets'."""
    with write(conn):
        if results_fingerprint(conn, family) != covers:
            LOG.info("%s changed while its tries were worked out: not kept, a newer count is due", family)
            return None
        tid = conn.execute("INSERT INTO tries (family, computed_at, covers, saved, independent, best_95, code_sha) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id",
                           (family, now(), covers, saved, independent, best_95, code_sha)).fetchone()[0]
        conn.executemany("INSERT INTO instrument_tries (tries_id, instrument_id, saved, independent, best_95) "
                         "VALUES (?, ?, ?, ?, ?)", [(tid, i, *v) for i, v in (instruments or {}).items()])
        return tid


def instrument_tries(conn: sqlite3.Connection, tries_id: int) -> list[sqlite3.Row]:
    """Each instrument's own tries kept with a count of the single assets'."""
    return conn.execute("SELECT * FROM instrument_tries WHERE tries_id = ?", (tries_id,)).fetchall()


# ---------------------------------------------------------------------------------------------------- runs
def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


STARTING_SECONDS = 60             # a run created this long ago whose process never marked itself running did not start


def active_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """The run that is on now: running or stopping with its process alive, or created a moment ago and starting."""
    for r in conn.execute("SELECT * FROM run WHERE state IN ('starting', 'running', 'stopping') ORDER BY id DESC"):
        if r["state"] == "starting":
            age = datetime.now(timezone.utc) - datetime.strptime(r["requested_at"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
                tzinfo=timezone.utc)
            if age.total_seconds() < STARTING_SECONDS:
                return r
        elif pid_alive(r["pid"]):
            return r
    return None
