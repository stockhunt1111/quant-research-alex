"""Many evaluations spread over the cores of one machine.

Jobs on the same list and timeframe run one after another in one process, in pieces: `evaluate` keeps the last list
it loaded, so a piece reads the list's bars and works out its seats once, not once per job. The pieces start longest
first, each on the next free process, so the run does not end waiting on one process left with a queue of long jobs.
How long a job takes is how long its result took when it was last saved (the app's database: a list's result, or
the sum of a run's instruments alone); a job not on record counts as the median of its list and timeframe. A strategy that fits its models on every core itself (ml_direction)
runs in the main process meanwhile. The workers run at a lower priority (nice), which on macOS gave those fits little
edge: ten workers and ten fitting processes shared the ten cores about evenly.
"""
from __future__ import annotations

import os
import resource
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Hashable
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Any

import numba
import numpy as np

from strategy_lab import db, log

LOG = log.get("batch")
PIECES_PER_WORKER = 4       # a piece is at most a quarter of a process's share of the run: few loads, a short tail
NICE = 10

Key = tuple[str, str, str]  # strategy, universe, timeframe


def recorded(kind: str = "list") -> dict[Key, float]:
    """Seconds each (strategy, list, timeframe) evaluation took when it was last saved: a list's result (`kind` "list"),
    or the sum of a list's instruments alone ("asset")."""
    conn = db.connect()
    try:
        if kind == "list":
            rows = conn.execute("SELECT strategy, list_id, timeframe, seconds FROM result WHERE instrument_id IS NULL "
                                "AND seconds IS NOT NULL").fetchall()
        else:
            rows = conn.execute("SELECT strategy, list_id, timeframe, sum(seconds) FROM result WHERE instrument_id IS NOT "
                                "NULL AND seconds IS NOT NULL GROUP BY strategy, list_id, timeframe").fetchall()
    finally:
        conn.close()
    if not rows:
        LOG.warning("no %s evaluation times on record: jobs start in the order given", kind)
    return {(r[0], r[1], r[2]): float(r[3]) for r in rows}


def expected(times: dict[Key, float], strategy: str, universe: str, timeframe: str) -> float:
    """Seconds a job is expected to take: its own time on record, else the median of its list and timeframe, else of
    its timeframe; 0 with nothing on record."""
    if (strategy, universe, timeframe) in times:
        return times[strategy, universe, timeframe]
    for same in ([v for (_, u, tf), v in times.items() if u == universe and tf == timeframe],
                 [v for (_, _, tf), v in times.items() if tf == timeframe]):
        if same:
            return float(np.median(same))
    return 0.0


def pieces(jobs: list, group: Callable[[Any], Hashable], seconds: Callable[[Any], float],
           workers: int) -> list[tuple[float, list[tuple[int, Any]]]]:
    """The jobs cut into pieces of one group each, longest first: (expected seconds, [(job's position, job)])."""
    expect = [seconds(job) for job in jobs]
    if not sum(expect):                         # nothing on record: every job counts the same
        expect = [1.0] * len(jobs)
    by: dict[Hashable, list[tuple[float, int, Any]]] = defaultdict(list)
    for k, job in enumerate(jobs):
        by[group(job)].append((expect[k], k, job))
    cap = sum(expect) / (workers * PIECES_PER_WORKER)
    out = []
    for items in by.values():
        items.sort(key=lambda x: -x[0])
        cur, total = [], 0.0
        for s, k, job in items:
            if cur and total + s > cap:
                out.append((total, cur))
                cur, total = [], 0.0
            cur.append((k, job))
            total += s
        out.append((total, cur))
    out.sort(key=lambda p: -p[0])
    return out


def _start(setup: Callable[[], None] | None) -> None:
    log.setup()
    os.nice(NICE)
    # one of `workers` processes sharing the cores: its compiled parallel loops (the random-timing check) take one
    # thread, not one per core in every process
    numba.set_num_threads(1)
    if setup is not None:
        setup()


def _cpu() -> float:
    """CPU seconds of this process and of its workers that have ended."""
    own, ended = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + ended.ru_utime + ended.ru_stime


def _piece(fn: Callable, items: list[tuple[int, Any]]) -> list[tuple[int, Any, float]]:
    out = []
    for k, job in items:
        t = time.perf_counter()
        r = fn(job)
        out.append((k, r, time.perf_counter() - t))
    return out


def run(fn: Callable, jobs: list, *, workers: int, group: Callable[[Any], Hashable],
        seconds: Callable[[Any], float], in_main: Callable[[Any], bool] | None = None,
        setup: Callable[[], None] | None = None) -> list[tuple[Any, float]]:
    """`fn(job)` for every job on `workers` processes, each of which calls `setup()` first. The jobs `in_main` picks
    run in this process meanwhile, one after another, the longest first: a strategy that fits its models on every
    core itself does it only here (in a worker it would fit them one after another). With one worker everything runs
    in this process (a test's stand-ins and a debugger reach the jobs there). Returns (result, seconds taken) per job,
    in the jobs' order. A job that raises stops the run."""
    here = [k for k, job in enumerate(jobs) if in_main is not None and in_main(job)]
    rest = sorted(set(range(len(jobs))) - set(here))
    todo = [(s, [(rest[k], job) for k, job in items])
            for s, items in pieces([jobs[k] for k in rest], group, seconds, max(workers, 1))]
    here.sort(key=lambda k: -seconds(jobs[k]))
    out: list = [None] * len(jobs)
    t0, cpu0 = time.time(), _cpu()

    def run_here() -> None:
        for k, r, s in _piece(fn, [(k, jobs[k]) for k in here]):
            out[k] = (r, s)

    if workers <= 1:
        LOG.info("%d jobs in this process", len(jobs))
        if setup is not None:
            setup()
        run_here()
        for _, items in todo:
            for k, r, s in _piece(fn, items):
                out[k] = (r, s)
        _done(len(jobs), t0, cpu0)
        return out
    LOG.info("%d jobs in %d pieces on %d processes, the longest first; %d in this process meanwhile", len(rest),
             len(todo), workers, len(here))
    done, lock = 0, threading.Lock()

    def finished(f: Future) -> None:           # as each piece ends, even while this process runs its own jobs
        nonlocal done
        if f.cancelled() or f.exception() is not None:
            return
        with lock:
            for k, r, s in f.result():
                out[k] = (r, s)
                done += 1
            LOG.info("%d of %d jobs done, %.0f min", done, len(rest), (time.time() - t0) / 60)

    pool = ProcessPoolExecutor(workers, initializer=_start, initargs=(setup,))
    try:
        futures = [pool.submit(_piece, fn, items) for _, items in todo]
        for f in futures:
            f.add_done_callback(finished)
        run_here()
        for f in as_completed(futures):
            f.result()                          # the first job that raised stops the run
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    _done(len(jobs), t0, cpu0)
    return out


def _done(n: int, t0: float, cpu0: float) -> None:
    LOG.info("%d jobs done in %.0f min, %.0f CPU-minutes (this process, its workers and theirs)", n,
             (time.time() - t0) / 60, (_cpu() - cpu0) / 60)
