"""What the server does in the background, and the changes it pushes to the open pages (server-sent events).

Every POLL seconds it asks the database whether another connection committed (PRAGMA data_version). When one did:
the results saved or deleted since (the change log) go out as `results`, with their keys; a new luck bar as `judged`;
the run's progress as `run`, at most once a second. Every CODE_EVERY seconds it looks at the evaluation code's files:
an edit makes results stale, so the stamps are judged again and `code` goes out. The luck bars (the tries, minutes of
work over every list result and every single-asset result) are worked out in a child process when one no longer covers
its results and no run is on, once per state of the results; a run's own results are judged against the last bars
meanwhile.

A page that connects gets `hello` with where things stand; a page too slow to take its events gets `reset` and reloads
everything.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from strategy_lab import board, db, log, provenance
from strategy_lab.config import LOGS_DIR, ROOT_DIR

from server.research import Research, asset_key, list_key
from server.runner import Runner

LOG = log.get("server.live")
POLL = 0.5
CODE_EVERY = 5.0
RUN_EVERY = 1.0             # the run's progress goes out at most this often
QUEUE = 256                 # events a page may lag behind before it is told to reload
KEYS = 200                  # at most this many changed results named in one event; beyond that, every result


@dataclass(frozen=True)
class Event:
    name: str
    data: dict


def judge_log() -> Path:
    return LOGS_DIR / "judge.log"


def judge_command() -> list[str]:
    return [sys.executable, "-m", "strategy_lab", "db", "judge"]


class Live:
    def __init__(self, research: Research, runner: Runner, *, judge: list[str] | None = None,
                 root: Path = ROOT_DIR):
        self.research = research
        self.runner = runner
        self.judge = judge if judge is not None else judge_command()
        self.root = root
        self.clients: set[asyncio.Queue] = set()
        self.seq: int | None = None
        self.tries: tuple | None = None
        self.run_view: dict | None = None
        self.code_sha = ""
        self._conn = None
        self._data_version: int | None = None
        self._mtimes: dict[str, float] | None = None
        self._code_at = 0.0
        self._try_code = ""
        self._run_sent = ("", 0.0)
        self._run_due: dict | None = None
        self._judging: subprocess.Popen | None = None
        self._judge_failed: str | None = None      # what the last judge that failed worked on
        self._judged_on: str | None = None         # ... and the last that ended well
        self._judging_on: str | None = None
        self._left_uncovered: str | None = None    # what a judge that ended well left without a covering bar, told
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None

    # -------------------------------------------------------------------------------------------- the pages
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(QUEUE)
        self.clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.clients.discard(q)

    def hello(self) -> dict:
        return {"seq": self.seq, "tries": None if self.tries is None else self.tries[0],
                "code": self.research.code_version, "run": self.run_view}

    def publish(self, events: list[Event]) -> None:
        for q in list(self.clients):
            for ev in events:
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:           # a page that fell behind: it reloads everything
                    while not q.empty():
                        q.get_nowait()
                    q.put_nowait(Event("reset", {}))
                    break

    def close(self) -> None:
        """The server stops: every page's stream ends (a page reconnects to the next server)."""
        self._closed = True
        for q in list(self.clients):
            while not q.empty():
                q.get_nowait()
            q.put_nowait(None)
        if self._judging is not None and self._judging.poll() is None:
            LOG.info("the luck bar was being worked out: left to finish in its own process (%d)", self._judging.pid)

    def close_soon(self) -> None:
        """`close` from a signal handler or another thread, on the event loop's own turn."""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.close)
        else:
            self.close()

    # -------------------------------------------------------------------------------------------- the loop
    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        while not self._closed:
            try:
                events = await asyncio.to_thread(self.poll)
            except Exception:                        # noqa: BLE001 - the watcher keeps going; the error is logged
                LOG.exception("the database watcher failed a round")
                events = []
            if events:
                self.publish(events)
            await asyncio.sleep(POLL)

    def _reader(self):
        if self._conn is None:
            self._conn = db.connect(self.research.path, readonly=True)
        return self._conn

    def poll(self) -> list[Event]:
        """One round: what changed since the last one, as events; the background work that is due."""
        events: list[Event] = []
        self.runner.supervise()
        conn = self._reader()
        code_changed = self._look_at_code(events)
        version = conn.execute("PRAGMA data_version").fetchone()[0]
        changed = version != self._data_version or code_changed
        self._data_version = version
        if changed:
            self._results(conn, events)
            tries = tuple(_id(db.latest_tries(conn, f)) for f in ("lists", "picks", "assets"))
            if self.tries is not None and tries != self.tries:
                events.append(Event("judged", {"tries": tries[0], "picks": tries[1], "assets": tries[2]}))
            self.tries = tries
            self._run_due = self.runner.snapshot(conn, self.code_sha)
        self._send_run(events)
        if changed or (self._judging is not None and self._judging.poll() is not None):
            self._judge_if_due(conn)
        return events

    def _results(self, conn, events: list[Event]) -> None:
        seq = self.research.seq()
        if self.seq is not None and seq != self.seq:
            rows = conn.execute("SELECT c.kind, c.op, r.strategy, r.list_id, r.timeframe, r.instrument_id FROM change c "
                                "LEFT JOIN result r ON r.id = c.result_id WHERE c.seq > ? ORDER BY c.seq",
                                (self.seq,)).fetchall()
            keys = list(dict.fromkeys(
                (list_key(r["strategy"], r["list_id"], r["timeframe"]) if r["instrument_id"] is None else
                 asset_key(r["strategy"], r["list_id"], r["timeframe"], r["instrument_id"]))
                for r in rows if r["strategy"] is not None))
            first = conn.execute("SELECT min(seq) FROM change").fetchone()[0]
            gone = first is None or first > self.seq + 1         # the log was cut back past what this server saw
            every = gone or len(keys) > KEYS or any(r["op"] == "deleted" for r in rows)
            events.append(Event("results", {"seq": seq, "kinds": sorted({r["kind"] for r in rows}) or
                                            ["list", "asset", "pick"], "keys": [] if every else keys,
                                            "every": every}))
        self.seq = seq

    def _send_run(self, events: list[Event]) -> None:
        if self._run_due is None and self.run_view is None:
            return
        text = json.dumps(self._run_due, sort_keys=True)
        last, at = self._run_sent
        if text != last and time.time() - at >= RUN_EVERY:
            self.run_view = self._run_due
            self._run_sent = (text, time.time())
            events.append(Event("run", {"run": self._run_due}))

    # -------------------------------------------------------------------------------------------- code on disk
    def _look_at_code(self, events: list[Event]) -> bool:
        if time.time() - self._code_at < CODE_EVERY and self._mtimes is not None:
            return False
        self._code_at = time.time()
        files = provenance.all_evaluation_files()
        mtimes = {f: _mtime(self.root / f) for f in [*files, *board.TRY_CODE]}
        if mtimes == self._mtimes:
            return False
        first = self._mtimes is None
        self._mtimes = mtimes
        self.code_sha = provenance.sha_of(files) or ""
        self._try_code = provenance.sha_of(board.TRY_CODE) or ""
        if first:
            return False
        self.research.code_changed()
        LOG.info("the evaluation code changed on disk: results' stamps judged again")
        events.append(Event("code", {"version": self.research.code_version}))
        return True

    # -------------------------------------------------------------------------------------------- the luck bars
    def _judge_if_due(self, conn) -> None:
        if self._judging is not None:
            code = self._judging.poll()
            if code is None:
                return
            if code != 0:
                LOG.error("working out the luck bars failed (exit %s): see logs/judge.log; not tried again until the "
                          "results change", code)
                self._judge_failed = self._judging_on
            else:
                self._judged_on = self._judging_on
            self._judging = None
        if db.active_run(conn) is not None:
            return                      # the run's results are judged against the last bars; worked out at its end
        parts, due = [], False
        for family in ("lists", "assets"):
            if not conn.execute(f"SELECT 1 FROM result r WHERE {db.FAMILIES[family]} LIMIT 1").fetchone():
                continue                # no result of the family: no bar to work out
            parts.append(db.results_fingerprint(conn, family))
            kept = db.latest_tries(conn, family)
            due = due or kept is None or kept["covers"] != parts[-1] or kept["code_sha"] != self._try_code
        covers = " ".join([*parts, self._try_code])
        if not due or covers == self._judge_failed:
            return
        if covers == self._judged_on:          # it ended well on these results and code, and still no bar covers them
            if self._left_uncovered != covers:
                LOG.warning("the judge ended well and left a luck bar not covering its results: not started again "
                            "until the results or its code change")
                self._left_uncovered = covers
            return
        env = dict(os.environ) | {"STRATEGY_LAB_DB": str(self.research.path)}
        path = judge_log()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab") as out:
            self._judging = subprocess.Popen(self.judge, cwd=self.root, stdin=subprocess.DEVNULL, stdout=out,
                                             stderr=subprocess.STDOUT, env=env)
        self._judging_on = covers
        LOG.info("a luck bar no longer covers its results: worked out again (process %d)", self._judging.pid)


def _id(row) -> int | None:
    return None if row is None else row["id"]


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return -1.0
