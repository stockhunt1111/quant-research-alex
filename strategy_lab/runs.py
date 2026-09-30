"""Re-runs of the research: what a run evaluates, and running it so that it can be stopped and resumed.

A run fixes its jobs when it starts (run_job): every strategy on the lists it applies to (stage lists) and, when asked,
each rule strategy on each instrument of a list alone (stage single_assets); unless the run is asked for everything, a
job whose result the current code already produced is marked skipped. The jobs run on the cores (strategy_lab.batch),
each marking its own state in the database as it starts and ends, so a page shows the run's progress as it goes. After
them: the choice of a strategy for each instrument of the ML task (stage picks, with single assets) and the lists' and
instruments' figures the pages show (stage summaries, `catalog`): everything a run makes is in the database. A run with
single assets also removes the kept runs of a strategy on each instrument alone that its plan no longer makes, within
what it covers (`unplanned_alone`: a list whose instruments its market's widest list holds today, a strategy no longer
run alone), as an instrument that leaves a list goes with the list's next run: kept, they would read as stale for ever.

Stopping sends SIGTERM to the run's process group: the running evaluations end at once (a result is saved in one
transaction, so none is left half-written), their jobs are marked stopped, and the run with them. Resuming the same run
starts its jobs that are not done — queued, stopped, failed — and the stages after them; a job cut off half-way starts
over. The run's process marks a heartbeat every HEARTBEAT seconds: a run whose process is gone without a stop is
interrupted, and resumes the same way.

    python scripts/rerun.py [--single-assets] [--everything] [--workers 10]     # a new run, from a terminal
    python scripts/rerun.py --single-assets --everything -u stockhunt_commodities  # some lists, strategies (--names)
                                                                                   # or timeframes (-t) only
    python scripts/rerun.py --run-id 12                                            # run or resume run 12
"""
from __future__ import annotations

import functools
import inspect
import json
import os
import resource
import shutil
import signal
import subprocess
import threading
import time
import traceback
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from strategy_lab import batch, db, lists, log, provenance
from strategy_lab.config import LOGS_DIR, TIMEFRAMES
from strategy_lab.strategy import load
from strategy_lab.universes import instruments_now

LOG = log.get("runs")
HEARTBEAT = 10                  # seconds between a run's marks that its process is alive
LOST_AFTER = 120                # a running run whose process is gone and has not marked for this long is interrupted
# open files a run's processes may hold: a list's exits walked on one-minute bars keep two memory-mapped files a name
# open (hundreds of names over a Top-100 list's history), where a process launchd starts, as the page's runs are, may
# open 256
OPEN_FILES = 65536

# ---------------------------------------------------------------------------------------------------- what runs on what
# every market's lists (strategy_lab.lists); ranking strategies need names to rank: the fifty- and hundred-name steps
OURS = lists.names(lists.OURS)
WIDE = ["us_stocks_top50", "us_stocks_top100", "crypto_top50", "crypto_top100"]
# a model refitted per instrument and window: the three- and ten-name steps and the small markets
ML = ["us_stocks_top3", "us_stocks_top10", "crypto_top3", "crypto_top10", "etf_top3", "stockhunt_etfs", "etf_core",
      "fx_majors", "stockhunt_commodities", "cme_futures"]
# the markets with an index a rule can time its names by (strategy_lab.market): not FX, not commodities
INDEXED_MARKETS = ("Stocks", "ETFs", "Crypto")
INDEXED = [u for u in OURS if lists.market(u) in INDEXED_MARKETS]
ON_EVERY_LIST = ["sma_cross", "ibs_reversion", "donchian_breakout", "ema_trend", "tsmom", "breakout_trail",
                 "calm_trend", "trend_or_revert", "late_entry_trend", "rsi2_connors", "gtaa_faber", "dual_momentum",
                 "sector_rotation", "vol_managed", "vol_managed_har", "bollinger_reversion", "keltner_breakout", "ibs",
                 "pocket_pivot",
                 # a model grading a rule's trades learns from every name of the list at once
                 "ibs_ml_filter", "ibs_ml_sized", "rsi2_ml_filter", "rsi2_ml_sized", "bollinger_ml_filter",
                 "bollinger_ml_sized", "trend_or_revert_ml_filter", "trend_or_revert_ml_sized"]
LIST_RUNS = [
    # (strategy, lists); a strategy is strategies/<name>.py, on every timeframe
    *[(name, OURS) for name in ON_EVERY_LIST],
    # rules reading their market's index (`rule(market=True)`)
    ("market_regime", INDEXED),
    ("regime_ema_trail", INDEXED),
    ("ml_direction", ML),
    ("big_move_follow", ["us_stocks_top100", "crypto_top100", "etf_core"]),
    ("xs_momentum", WIDE + ["etf_core"]),
    ("betting_against_beta", WIDE),
    ("funding_carry", ["crypto_top50", "crypto_top100"]),
]
# the rule strategies alone on each instrument (a panel strategy ranks a list: it has no one-instrument form)
ALONE_RULES = ["sma_cross", "ibs_reversion", "donchian_breakout", "ema_trend", "tsmom", "breakout_trail", "calm_trend",
               "trend_or_revert", "late_entry_trend", "ml_direction", "rsi2_connors", "gtaa_faber", "vol_managed",
               "bollinger_reversion", "keltner_breakout", "ibs", "ibs_ml_filter", "ibs_ml_sized", "rsi2_ml_filter",
               "rsi2_ml_sized", "bollinger_ml_filter", "bollinger_ml_sized", "trend_or_revert_ml_filter",
               "trend_or_revert_ml_sized", "market_regime", "regime_ema_trail", "pocket_pivot", "vol_managed_har"]
ENGINE = "ml_feature_search"    # the firm's per-symbol engine, measured: on the ML task's lists only


class Stopped(BaseException):
    """The run was asked to stop (SIGTERM, or ^C in a terminal): not an evaluation's error, nothing catches it on the way."""


def narrowed_to(only: dict | None) -> str | None:
    """What a run asked for some lists, strategies or timeframes only (`only_of`) covers, as the pages name them: the
    lists by their titles, then the strategies, then the timeframes; None for a run of everything. Such a run ends as a
    run of everything does: the ML task's picks read every result and the lists' figures every list's bars, so they are
    made again after any change."""
    parts = [", ".join(lists.title(u) for u in (only or {}).get("lists") or ()),
             ", ".join((only or {}).get("strategies") or ()), " ".join((only or {}).get("timeframes") or ())]
    return " · ".join(p for p in parts if p) or None


def only_of(names: Sequence[str] | None = None, universes: Sequence[str] | None = None,
            timeframes: Sequence[str] | None = None) -> dict | None:
    """What a run is narrowed to: these strategies, lists and timeframes (none given: every one), each one a run covers
    (a strategy of `LIST_RUNS`, `ALONE_RULES` or the firm's engine; a list of ours or of the ML task's; a research
    timeframe), or the run is refused. None when nothing narrows it."""
    known = {n for n, _ in LIST_RUNS} | set(ALONE_RULES) | {ENGINE}
    unknown = sorted(set(names or ()) - known)
    if unknown:
        raise SystemExit(f"no run evaluates {', '.join(unknown)}; its strategies: {', '.join(sorted(known))}")
    lists.refuse_outside(universes or (), lists.basket_lists() + lists.PER_INSTRUMENT, "a re-run")
    off = sorted(set(timeframes or ()) - set(TIMEFRAMES))
    if off:
        raise SystemExit(f"{', '.join(off)}: not a research timeframe ({', '.join(TIMEFRAMES)})")
    only = {k: list(dict.fromkeys(v)) for k, v in (("strategies", names), ("lists", universes),
                                                    ("timeframes", timeframes)) if v}
    return only or None


def _wanted(only: dict | None, strategy: str, universe: str, tf: str) -> bool:
    only = only or {}
    return (not only.get("strategies") or strategy in only["strategies"]) and \
        (not only.get("lists") or universe in only["lists"]) and (not only.get("timeframes") or tf in only["timeframes"])


def list_jobs(only: dict | None = None) -> list[tuple[str, str, str]]:
    """(strategy, list, timeframe) of every evaluation of a strategy on a list as one book."""
    for name, universes in LIST_RUNS:
        lists.refuse_outside(universes, lists.basket_lists(), f"{name} as a basket")
    return [(name, u, tf) for name, universes in LIST_RUNS for u in universes for tf in TIMEFRAMES
            if _wanted(only, name, u, tf)]


def _covered(inner: str, outer: str, tf: str) -> bool:
    """Every instrument `inner` holds today is also in `outer` today: a run of `inner` alone would repeat `outer`'s."""
    a, _ = instruments_now(inner, tf)
    b, _ = instruments_now(outer, tf)
    return set(a) <= set(b)


def runs_alone_on(name: str, universe: str) -> bool:
    """A rule runs alone on each instrument of the list: every rule does, but one reading its market's index
    (`rule(market=True)`) only where the list's market has one (`INDEXED_MARKETS`)."""
    return not load(name).market or lists.per_instrument_market(universe) in INDEXED_MARKETS


def alone_jobs(only: dict | None = None) -> list[tuple[str, str, str]]:
    """(strategy, list, timeframe) of every run of a strategy on each instrument of a list alone: the rules on the
    widest list of each market and on the ML task's lists, the firm's engine on the ML task's; a list whose instruments
    are all in its market's widest list today (the ML task's ten largest stocks, inside the hundred most liquid) is not
    run again for the rules."""
    out = []
    universes = (only or {}).get("lists") if (only or {}).get("outside_lists") else lists.PER_INSTRUMENT
    for tf in TIMEFRAMES:
        for u in universes:
            if not any(_wanted(only, name, u, tf) for name in (*ALONE_RULES, ENGINE)):
                continue                                    # a narrowed run's other lists: nothing of theirs to read
            wider = next((w for w in lists.PER_ASSET if w != u and lists.per_instrument_market(w)
                          == lists.per_instrument_market(u)), None)
            repeats = u not in lists.PER_ASSET and wider is not None and _covered(u, wider, tf)
            if repeats:
                LOG.info("%s %s: every instrument is in %s, whose run covers the rules", u, tf, wider)
            for name in ALONE_RULES:
                if not repeats and _wanted(only, name, u, tf) and runs_alone_on(name, u):
                    out.append((name, u, tf))
            if u in lists.ML_TASK and _wanted(only, ENGINE, u, tf):
                out.append((ENGINE, u, tf))
    return out


# ---------------------------------------------------------------------------------------------------- runs in the database
def log_path(run_id: int) -> Path:
    """A run's one log file, its resumes appended (a run the server starts writes its output there too)."""
    return LOGS_DIR / f"rerun_{run_id}.log"


def create(conn, *, requested_by: str, single_assets: bool, everything: bool, only: dict | None = None) -> int:
    with db.write(conn):
        return conn.execute(
            "INSERT INTO run (requested_at, requested_by, single_assets, everything, only, state, code_sha_at_start) "
            "VALUES (?, ?, ?, ?, ?, 'starting', ?) RETURNING id",
            (db.now(), requested_by, int(single_assets), int(everything), None if only is None else db.to_json(only),
             provenance.sha_of(provenance.all_evaluation_files()) or "")).fetchone()[0]


def _fresh(conn, stage: str, strategy: str, universe: str, tf: str) -> bool:
    """The job's results were produced by the current code (a list's result; every instrument of a run alone)."""
    if stage == "lists":
        rows = conn.execute("SELECT r.code_sha, c.files FROM result r JOIN code_stamp c ON c.sha = r.code_sha "
                            "WHERE r.strategy = ? AND r.list_id = ? AND r.instrument_id IS NULL AND r.timeframe = ?",
                            (strategy, universe, tf)).fetchall()
    else:
        rows = conn.execute("SELECT r.code_sha, c.files FROM result r JOIN code_stamp c ON c.sha = r.code_sha "
                            "WHERE r.strategy = ? AND r.list_id = ? AND r.instrument_id IS NOT NULL AND r.timeframe = ? "
                            "UNION SELECT t.code_sha, c.files FROM asset_too_short t JOIN code_stamp c ON "
                            "c.sha = t.code_sha WHERE t.strategy = ? AND t.list_id = ? AND t.timeframe = ?",
                            (strategy, universe, tf, strategy, universe, tf)).fetchall()
    return bool(rows) and all(provenance.is_current({"files": json.loads(r["files"]), "sha": r["code_sha"]})
                              for r in rows)


def _only(run) -> dict | None:
    return json.loads(run["only"]) if run["only"] else None


def _stages(only: dict | None) -> list[str]:
    return (only or {}).get("stages") or ["lists", "single_assets"]


def unplanned_alone(conn, run) -> list[tuple[str, str, str]]:
    """(strategy, list, timeframe) of every run of a strategy on each instrument of a list alone kept in the database
    that the run's plan does not make, within what the run covers (its lists, strategies and timeframes)."""
    only = _only(run)
    universes = (only or {}).get("lists") if (only or {}).get("outside_lists") else lists.PER_INSTRUMENT
    planned = {tuple(r) for r in conn.execute("SELECT strategy, list_id, timeframe FROM run_job WHERE run_id = ? AND "
                                              "stage = 'single_assets'", (run["id"],))}
    kept = conn.execute(f"SELECT DISTINCT r.strategy, r.list_id, r.timeframe FROM result r WHERE {db.FAMILIES['assets']} "
                        "UNION SELECT strategy, list_id, timeframe FROM asset_too_short").fetchall()
    return sorted(k for k in map(tuple, kept)
                  if k[1] in universes and _wanted(only, *k) and k not in planned)


def _enough_open_files() -> None:
    """This process's soft limit of open files, which its workers inherit, raised to OPEN_FILES (or the hard limit
    below it)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = OPEN_FILES if hard == resource.RLIM_INFINITY else min(OPEN_FILES, hard)
    if soft == resource.RLIM_INFINITY or soft >= want:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except (ValueError, OSError) as e:
        LOG.warning("open files: the soft limit stays at %d (%s); a list whose exits are walked on its minutes may "
                    "fail with too many open files", soft, e)
        return
    LOG.info("open files: the soft limit raised from %d to %d", soft, want)


def _plan(conn, run) -> None:
    """The run's jobs, fixed now: queued, or skipped when the current code already produced their result."""
    only = _only(run)
    stages = _stages(only)
    jobs = [("lists", *j) for j in list_jobs(only)] if "lists" in stages else []
    if run["single_assets"] and "single_assets" in stages:
        jobs += [("single_assets", *j) for j in alone_jobs(only)]
    times = {"lists": batch.recorded("list"), "single_assets": batch.recorded("asset") or batch.recorded("list")}
    rows = []
    for stage, name, u, tf in jobs:
        skipped = not run["everything"] and _fresh(conn, stage, name, u, tf)
        rows.append((run["id"], stage, name, u, tf, "skipped" if skipped else "queued",
                     batch.expected(times[stage], name, u, tf)))
    with db.write(conn):
        conn.executemany("INSERT INTO run_job (run_id, stage, strategy, list_id, timeframe, state, expected_seconds) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    LOG.info("run %d: %d jobs, %d of them already produced by the current code", run["id"], len(rows),
             sum(r[5] == "skipped" for r in rows))


_CONN = None


def _conn():
    """This process's own connection (a worker's jobs share one)."""
    global _CONN
    if _CONN is None:
        _CONN = db.connect()
    return _CONN


def _mark(job_id: int, sql: str, *args) -> None:
    conn = _conn()
    with db.write(conn):
        conn.execute(f"UPDATE run_job SET {sql} WHERE id = ?", (*args, job_id))


def _job(job: tuple[int, str, str, str, str]) -> None:
    """One job, run where batch puts it: its evaluation saved, its state marked as it starts and ends."""
    job_id, stage, name, universe, tf = job
    _mark(job_id, "state = 'running', attempts = attempts + 1, started_at = ?, finished_at = NULL, worker_pid = ?, "
                  "error = NULL", db.now(), os.getpid())
    t0 = time.perf_counter()
    try:
        if stage == "lists":
            from strategy_lab.evaluate import evaluate
            evaluate(load(name), universe, tf, start=lists.start(universe))
        else:
            from strategy_lab import per_asset
            per_asset.evaluate(load(name), universe, tf)
    except Exception as e:                      # noqa: BLE001 - kept on the job, listed at the end, never swallowed
        LOG.error("FAILED %s on %s %s: %s", name, universe, tf, e)
        _mark(job_id, "state = 'failed', finished_at = ?, seconds = ?, error = ?", db.now(), time.perf_counter() - t0,
              f"{e!r}\n{traceback.format_exc(limit=3)}")
        return
    _mark(job_id, "state = 'done', finished_at = ?, seconds = ?", db.now(), time.perf_counter() - t0)


def _load_all(names: Sequence[str]) -> None:
    """A worker's start: import every strategy of the run now, and pin the code its results are stamped with."""
    provenance.freeze([inspect.getsourcefile(load(n).fn) for n in names])


def _awake(run_id: int) -> subprocess.Popen | None:
    """Keep this Mac from sleeping while the run is on: caffeinate, holding off idle sleep (and system sleep on
    power) until the run's process ends, whatever ends it. A closed lid still sleeps a laptop. None where there is no
    such tool: a server does not sleep."""
    tool = shutil.which("caffeinate")
    if tool is None:
        LOG.info("run %d: no caffeinate here, nothing holds off sleep during the run", run_id)
        return None
    return subprocess.Popen([tool, "-i", "-s", "-w", str(os.getpid())], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _heartbeat(run_id: int, stop: threading.Event) -> None:
    conn = db.connect()
    try:
        while not stop.wait(HEARTBEAT):
            with db.write(conn):
                conn.execute("UPDATE run SET heartbeat_at = ? WHERE id = ?", (db.now(), run_id))
    finally:
        conn.close()


def _on_sigterm(signum, frame) -> None:
    raise Stopped()


def _stage(conn, run_id: int, stage: str) -> None:
    with db.write(conn):
        conn.execute("UPDATE run SET stage = ? WHERE id = ?", (stage, run_id))


def execute(run_id: int, *, workers: int = 10) -> int:
    """Run (or resume) run `run_id` in this process and its workers; returns the exit code (1: a job failed)."""
    conn = db.connect()
    run = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        raise SystemExit(f"no run {run_id}")
    if run["state"] == "done" and not conn.execute("SELECT 1 FROM run_job WHERE run_id = ? AND state = 'failed'",
                                                   (run_id,)).fetchone():
        LOG.info("run %d is done: nothing to resume", run_id)
        return 0
    other = db.active_run(conn)
    if other is not None and other["id"] != run_id:
        raise SystemExit(f"run {other['id']} is on (process {other['pid']}): one run at a time")
    _enough_open_files()
    planned = conn.execute("SELECT count(*) FROM run_job WHERE run_id = ?", (run_id,)).fetchone()[0]
    with db.write(conn):
        conn.execute("UPDATE run SET state = 'running', pid = ?, pgid = ?, started_at = coalesce(started_at, ?), "
                     "finished_at = NULL, heartbeat_at = ?, error = NULL, resumed = resumed + ? WHERE id = ?",
                     (os.getpid(), os.getpgid(0), db.now(), db.now(), int(planned > 0), run_id))
        conn.execute("UPDATE run_job SET state = 'queued' WHERE run_id = ? AND state IN ('running', 'stopped', 'failed')",
                     (run_id,))
    if not planned:
        _plan(conn, conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone())
    previous = signal.signal(signal.SIGTERM, _on_sigterm)
    beat = threading.Event()
    threading.Thread(target=_heartbeat, args=(run_id, beat), daemon=True).start()
    awake = _awake(run_id)
    cpu0, failed, state, error = batch._cpu(), 0, "done", None
    try:
        for stage in ("lists", "single_assets"):
            todo = [(r["id"], stage, r["strategy"], r["list_id"], r["timeframe"]) for r in conn.execute(
                "SELECT id, strategy, list_id, timeframe FROM run_job WHERE run_id = ? AND stage = ? AND state = 'queued'",
                (run_id, stage))]
            if not todo:
                continue
            _stage(conn, run_id, stage)
            names = sorted({j[2] for j in todo})
            provenance.freeze([inspect.getsourcefile(load(n).fn) for n in names])
            expected = {r["id"]: r["expected_seconds"] for r in conn.execute(
                "SELECT id, expected_seconds FROM run_job WHERE run_id = ?", (run_id,))}
            LOG.info("run %d, %s: %d jobs", run_id, stage, len(todo))
            batch.run(_job, todo, workers=workers, group=lambda j: (j[3], j[4]), seconds=lambda j: expected[j[0]],
                      in_main=(lambda j: load(j[2]).prepare is not None) if stage == "lists" else None,
                      setup=_setup_for(names))
        run = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
        if run["single_assets"] and "single_assets" in _stages(_only(run)):
            gone = unplanned_alone(conn, run)
            if gone:
                n = db.drop_alone_runs(conn, gone)
                LOG.info("run %d: %d results of %d runs alone its plan no longer makes removed (%s)", run_id, n,
                         len(gone), ", ".join(f"{s} on {u} {tf}" for s, u, tf in gone))
        if run["single_assets"]:
            _stage(conn, run_id, "picks")
            from strategy_lab import strategy_pick
            strategy_pick.pick_all(conn)
        _stage(conn, run_id, "summaries")
        from strategy_lab import catalog
        catalog.refresh()
        failed = conn.execute("SELECT count(*) FROM run_job WHERE run_id = ? AND state = 'failed'",
                              (run_id,)).fetchone()[0]
    except (Stopped, KeyboardInterrupt):
        state = "stopped"
        LOG.info("run %d stopped: its running evaluations were cut off and start over on resume", run_id)
    except BaseException as e:                              # noqa: BLE001 - the run is kept resumable, the error on it
        asked = conn.execute("SELECT state FROM run WHERE id = ?", (run_id,)).fetchone()[0] == "stopping"
        if asked:           # the workers the stop killed broke the pool before this process's own signal arrived
            state = "stopped"
            LOG.info("run %d stopped (its workers went first: %r)", run_id, e)
        else:
            state, error = "interrupted", f"{e!r}\n{traceback.format_exc(limit=5)}"
            LOG.exception("run %d broke off", run_id)
    finally:
        beat.set()
        if awake is not None:
            awake.terminate()
            awake.wait(10)
        signal.signal(signal.SIGTERM, previous)
        with db.write(conn):
            conn.execute("UPDATE run_job SET state = 'stopped' WHERE run_id = ? AND state = 'running'", (run_id,))
            conn.execute("UPDATE run SET state = ?, finished_at = ?, error = ?, pid = NULL, pgid = NULL, "
                         "cpu_seconds = coalesce(cpu_seconds, 0) + ? WHERE id = ?",
                         (state, db.now(), error, batch._cpu() - cpu0, run_id))
        conn.close()
    return 1 if failed or state != "done" else 0


def _setup_for(names: Sequence[str]):
    return functools.partial(_load_all, list(names))




def interrupted(conn) -> list[int]:
    """Mark interrupted every run that says it is on but whose process is gone and has not marked for LOST_AFTER
    seconds; returns their ids."""
    lost = []
    for r in conn.execute("SELECT id, pid, heartbeat_at FROM run WHERE state IN ('running', 'stopping')").fetchall():
        beat = r["heartbeat_at"]
        old = beat is None or (time.time() - _epoch(beat)) > LOST_AFTER
        if old and not db.pid_alive(r["pid"]):
            lost.append(r["id"])
    if lost:
        with db.write(conn):
            for rid in lost:
                conn.execute("UPDATE run_job SET state = 'stopped' WHERE run_id = ? AND state = 'running'", (rid,))
                conn.execute("UPDATE run SET state = 'interrupted', pid = NULL, pgid = NULL, finished_at = ? WHERE id = ?",
                             (db.now(), rid))
        LOG.warning("runs %s lost their process without a stop: interrupted, resumable", lost)
    return lost


def _epoch(stamp: str) -> float:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()
