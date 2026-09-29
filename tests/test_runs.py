"""A run fixes its jobs when it starts, stops at once when asked, and resumes where it stopped: the jobs done stay
done, the ones cut off start over, whatever became fresh meanwhile."""
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import dataclasses
import resource

import numpy as np
import pytest

from strategy_lab import db, runs

ONLY = {"strategies": ["sma_cross"], "lists": ["us_stocks_top10", "crypto_top10", "etf_core"], "timeframes": ["1d"]}


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.sqlite")
    monkeypatch.setattr(runs, "_CONN", None)
    c = db.connect()
    yield c
    c.close()


@pytest.fixture(autouse=True)
def later(monkeypatch):
    """The stages after the jobs, recorded rather than done: they read every result and every list's bars."""
    from strategy_lab import catalog, strategy_pick
    done: list = []
    monkeypatch.setattr(catalog, "refresh", lambda conn=None: done.append(("figures",)))
    monkeypatch.setattr(strategy_pick, "pick_all", lambda conn: done.append(("picks",)))
    return done


def _jobs(conn, run_id):
    return {(r["list_id"]): (r["state"], r["attempts"]) for r in conn.execute(
        "SELECT list_id, state, attempts FROM run_job WHERE run_id = ?", (run_id,))}


def _fake_jobs(monkeypatch, done: list, stop_on: str | None = None):
    """Jobs that only mark themselves, as the real one does, and one that is cut off by a stop."""
    def job(j):
        job_id, stage, name, universe, tf = j
        runs._mark(job_id, "state = 'running', attempts = attempts + 1, started_at = ?", db.now())
        if universe == stop_on:
            raise runs.Stopped()
        done.append(universe)
        runs._mark(job_id, "state = 'done', finished_at = ?", db.now())
    monkeypatch.setattr(runs, "_job", job)


def test_a_stopped_run_resumes_with_only_the_jobs_it_had_not_done(conn, monkeypatch):
    rid = runs.create(conn, requested_by="cli", single_assets=False, everything=True, only=ONLY)
    done: list = []
    _fake_jobs(monkeypatch, done, stop_on="crypto_top10")
    assert runs.execute(rid, workers=1) == 1
    run = conn.execute("SELECT * FROM run WHERE id = ?", (rid,)).fetchone()
    assert run["state"] == "stopped" and run["pid"] is None
    states = _jobs(conn, rid)
    assert states["crypto_top10"] == ("stopped", 1) and sum(s == "done" for s, _ in states.values()) == len(done)
    first = list(done)
    _fake_jobs(monkeypatch, done)
    assert runs.execute(rid, workers=1) == 0
    assert sorted(done[len(first):]) == sorted(set(ONLY["lists"]) - set(first))      # nothing done ran again
    run = conn.execute("SELECT * FROM run WHERE id = ?", (rid,)).fetchone()
    assert run["state"] == "done" and run["resumed"] == 1
    assert all(s == "done" for s, _ in _jobs(conn, rid).values()) and _jobs(conn, rid)["crypto_top10"][1] == 2


def test_a_run_of_one_market_evaluates_its_list_and_its_instruments_then_the_picks_and_figures(conn, monkeypatch,
                                                                                               later):
    only = runs.only_of(universes=["stockhunt_commodities"])
    rid = runs.create(conn, requested_by="cli", single_assets=True, everything=True, only=only)
    _fake_jobs(monkeypatch, [])
    assert runs.execute(rid, workers=1) == 0
    planned = {(r["stage"], r["strategy"], r["list_id"], r["timeframe"]) for r in conn.execute(
        "SELECT stage, strategy, list_id, timeframe FROM run_job WHERE run_id = ?", (rid,))}
    on_list = {("lists", name, "stockhunt_commodities", tf) for name, us in runs.LIST_RUNS
               if "stockhunt_commodities" in us for tf in ("1h", "4h", "1d")}
    alone = {("single_assets", name, "stockhunt_commodities", tf) for name in (*runs.ALONE_RULES, runs.ENGINE)
             for tf in ("1h", "4h", "1d")}
    assert planned == on_list | alone                 # nothing of another list: its bars are not even read
    assert later == [("picks",), ("figures",)]      # which read every result and every list's bars, as after all
    assert runs.narrowed_to(only) == "Commodities All 5" and runs.narrowed_to(None) is None


def test_a_run_is_refused_a_list_strategy_or_timeframe_no_run_covers():
    assert runs.only_of() is None
    assert runs.only_of(["ibs", "ibs"], ["etf_core"], ["1h"]) == {"strategies": ["ibs"], "lists": ["etf_core"],
                                                                   "timeframes": ["1h"]}
    for bad in ({"names": ["no_such_rule"]}, {"universes": ["stockhunt_stocks"]}, {"timeframes": ["1w"]}):
        with pytest.raises(SystemExit):
            runs.only_of(**bad)


def test_a_resumed_run_keeps_its_jobs_whatever_the_code_produced_meanwhile(conn, monkeypatch):
    """Resuming runs the run's own list of jobs: an everything run after a data refresh must not skip a job because
    its result looks fresh by the code's stamp."""
    rid = runs.create(conn, requested_by="ui", single_assets=False, everything=True, only=ONLY)
    done: list = []
    _fake_jobs(monkeypatch, done, stop_on="etf_core")
    runs.execute(rid, workers=1)
    monkeypatch.setattr(runs, "_fresh", lambda *a: True)          # everything looks produced by the current code now
    planned = conn.execute("SELECT count(*) FROM run_job WHERE run_id = ?", (rid,)).fetchone()[0]
    _fake_jobs(monkeypatch, done)
    runs.execute(rid, workers=1)
    assert "etf_core" in done and conn.execute("SELECT count(*) FROM run_job WHERE run_id = ?",
                                               (rid,)).fetchone()[0] == planned


def test_a_new_run_skips_what_the_current_code_produced_unless_asked_for_everything(conn, monkeypatch):
    monkeypatch.setattr(runs, "_fresh", lambda conn, stage, s, u, tf: u == "etf_core")
    lazy = runs.create(conn, requested_by="ui", single_assets=False, everything=False, only=ONLY)
    done: list = []
    _fake_jobs(monkeypatch, done)
    runs.execute(lazy, workers=1)
    assert _jobs(conn, lazy)["etf_core"] == ("skipped", 0) and "etf_core" not in done
    full = runs.create(conn, requested_by="ui", single_assets=False, everything=True, only=ONLY)
    runs.execute(full, workers=1)
    assert _jobs(conn, full)["etf_core"] == ("done", 1)


def test_a_run_whose_process_is_gone_without_a_stop_is_interrupted_and_resumable(conn, monkeypatch):
    rid = runs.create(conn, requested_by="ui", single_assets=False, everything=True, only=ONLY)
    with db.write(conn):
        conn.execute("UPDATE run SET state = 'running', pid = 999999, heartbeat_at = '2026-01-01T00:00:00.000000Z' "
                     "WHERE id = ?", (rid,))
        conn.execute("INSERT INTO run_job (run_id, stage, strategy, list_id, timeframe, state) VALUES "
                     "(?, 'lists', 'sma_cross', 'etf_core', '1d', 'running')", (rid,))
    assert runs.interrupted(conn) == [rid]
    assert conn.execute("SELECT state FROM run WHERE id = ?", (rid,)).fetchone()[0] == "interrupted"
    assert _jobs(conn, rid)["etf_core"][0] == "stopped"
    done: list = []
    _fake_jobs(monkeypatch, done)
    assert runs.execute(rid, workers=1) == 0 and done == ["etf_core"]


SLOW_RUN = """
import sys, time
from pathlib import Path
from strategy_lab import db, runs
db.DB_PATH = Path(sys.argv[1])
def job(j):
    runs._mark(j[0], "state = 'running', attempts = attempts + 1, started_at = ?", db.now())
    time.sleep(60)
runs._job = job
raise SystemExit(runs.execute(int(sys.argv[2]), workers=1))
"""


def test_sigterm_to_the_runs_process_group_stops_it_at_once_and_leaves_it_resumable(conn, tmp_path):
    rid = runs.create(conn, requested_by="ui", single_assets=False, everything=True, only=ONLY)
    root = Path(__file__).resolve().parent.parent
    p = subprocess.Popen([sys.executable, "-c", SLOW_RUN, str(db.DB_PATH), str(rid)], cwd=root, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 60
    while time.time() < deadline and conn.execute("SELECT count(*) FROM run_job WHERE run_id = ? AND state = 'running'",
                                                  (rid,)).fetchone()[0] == 0:
        time.sleep(0.2)
    assert db.active_run(conn)["id"] == rid
    t0 = time.time()
    os.killpg(p.pid, signal.SIGTERM)
    assert p.wait(30) == 1 and time.time() - t0 < 10
    run = conn.execute("SELECT * FROM run WHERE id = ?", (rid,)).fetchone()
    assert run["state"] == "stopped" and db.active_run(conn) is None
    assert {s for s, _ in _jobs(conn, rid).values()} <= {"stopped", "queued"}


def _asserting_for(pid: int) -> bool:
    """macOS holds off idle sleep on this process's behalf (pmset lists caffeinate's assertions and whom they serve)."""
    out = subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True, check=True).stdout.splitlines()
    return any("PreventUserIdleSystemSleep" in line and "caffeinate" in line
               and f"on behalf of Process ID {pid}" in (out[k + 1] if k + 1 < len(out) else "")
               for k, line in enumerate(out))


@pytest.mark.skipif(not (shutil.which("caffeinate") and shutil.which("pmset")), reason="macOS power management only")
def test_the_mac_stays_awake_while_a_run_is_on_and_not_after(conn, monkeypatch):
    rid = runs.create(conn, requested_by="ui", single_assets=False, everything=True, only=ONLY)
    seen = []

    def job(j):
        runs._mark(j[0], "state = 'running', attempts = attempts + 1, started_at = ?", db.now())
        seen.append(_asserting_for(os.getpid()))
        runs._mark(j[0], "state = 'done', finished_at = ?", db.now())
    monkeypatch.setattr(runs, "_job", job)
    assert runs.execute(rid, workers=1) == 0
    assert seen and all(seen)
    assert not _asserting_for(os.getpid())


def test_a_run_with_single_assets_removes_the_runs_alone_its_plan_no_longer_makes_within_what_it_covers(conn,
                                                                                                     monkeypatch):
    from tests.test_db import CODE, _outcome
    monkeypatch.setattr(runs, "_covered", lambda inner, outer, tf: inner == "crypto_mcap10")   # its rules: crypto_top100's
    kept = [("sma_cross", "crypto_mcap10", "4h"), ("ibs", "crypto_mcap10", "1d"),            # outside what the run covers
            ("sma_cross", "coiniq", "1d"),                                                   # a list not of ours
            ("sma_cross", "stockhunt_commodities", "1d")]                                    # the run makes it
    for k, (strategy, universe, tf) in enumerate([("sma_cross", "crypto_mcap10", "1d"), *kept]):
        o = dataclasses.replace(_outcome("spot:AAA", seed=k), strategy=strategy, list_id=universe, timeframe=tf)
        db.save_outcomes(conn, [o], description="", code=CODE, fill="next_open", replace_run=(strategy, universe, tf),
                         too_short=[("spot:ZZZ", 2)])
    pick = dataclasses.replace(_outcome("spot:AAA", seed=9), strategy="strategy_pick", timeframe="1d")
    db.save_outcomes(conn, [pick], description="", code=CODE, fill="next_open")
    assert conn.execute("SELECT count(*) FROM benchmark").fetchone()[0] == 6
    only = runs.only_of(["sma_cross"], ["crypto_mcap10", "stockhunt_commodities"], ["1d"])
    rid = runs.create(conn, requested_by="cli", single_assets=True, everything=True, only=only)
    _fake_jobs(monkeypatch, [])
    assert runs.execute(rid, workers=1) == 0
    left = {tuple(r) for r in conn.execute("SELECT strategy, list_id, timeframe FROM result")}
    assert left == {*kept, ("strategy_pick", "crypto_mcap10", "1d")}
    assert {tuple(r) for r in conn.execute("SELECT strategy, list_id, timeframe FROM asset_too_short")} == set(kept)
    assert conn.execute("SELECT count(*) FROM benchmark").fetchone()[0] == 5                # its buy & hold went too
    assert conn.execute("SELECT op FROM change ORDER BY seq DESC LIMIT 1").fetchone()[0] == "deleted"


def test_a_run_opens_as_many_files_as_a_lists_minutes_need_whatever_started_it(tmp_path):
    np.save(tmp_path / "minutes.npy", np.zeros(10))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    maps = []
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))        # what launchd gives the process of a page's run
        with pytest.raises(OSError):
            for _ in range(300):                                        # each memory map holds a file of its own open
                maps.append(np.load(tmp_path / "minutes.npy", mmap_mode="r"))
        maps.clear()
        runs._enough_open_files()
        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] >= min(runs.OPEN_FILES, hard)
        maps = [np.load(tmp_path / "minutes.npy", mmap_mode="r") for _ in range(1000)]
    finally:
        maps.clear()
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
