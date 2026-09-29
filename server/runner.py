"""Re-runs started, stopped and resumed from the page; strategy_lab.runs does the running, in a process of its own.

A run the page starts is `scripts/rerun.py --run-id N` in a new session (its own process group, so a stop reaches its
workers too, and it outlives the server), its output appended to logs/rerun_<N>.log. One run at a time: a start or a
resume is refused while a run is on or a batch runs in a terminal on this checkout. Stopping sends SIGTERM to the run's
process group; a run still on STOP_GRACE seconds later is killed and marked stopped here, since its process can no
longer mark itself.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from strategy_lab import db, log, provenance, runs
from strategy_lab.config import ROOT_DIR

LOG = log.get("server.runner")
STOP_GRACE = 30                     # seconds a stopped run has to end by itself before it is killed
ERROR_CHARS = 4000                  # a failed job's error as the page shows it
SCRIPTS = ("scripts/week1.py", "scripts/per_asset.py", "scripts/rerun.py")
RECENT = 900                        # the pace of a run is measured on the jobs it ended in the last 15 minutes


class Refused(Exception):
    """A start, stop or resume the server will not do (409): its reason, and whether asking again with confirm goes
    ahead."""

    def __init__(self, message: str, confirm: bool = False):
        super().__init__(message)
        self.message = message
        self.confirm = confirm


class NotFound(Exception):
    pass


def rerun_command(run_id: int) -> list[str]:
    return [sys.executable, str(ROOT_DIR / "scripts" / "rerun.py"), "--run-id", str(run_id)]


def shown(path: Path) -> str:
    """A path as the page shows it: from the repo's root when it is inside it."""
    return str(path.relative_to(ROOT_DIR) if path.is_relative_to(ROOT_DIR) else path)


def _epoch(stamp: str | None) -> float | None:
    return None if stamp is None else runs._epoch(stamp)


def batches_in_terminals(root: Path = ROOT_DIR) -> list[str]:
    """Batches running on this checkout that no run of the database stands for: a week1.py, per_asset.py or rerun.py,
    or `python -m strategy_lab run`, started in a terminal from `root` (another checkout's batches write its own
    database). Returns 'pid: command' for each."""
    out = subprocess.run(["ps", "-ax", "-o", "pid=,command="], capture_output=True, text=True, check=False).stdout
    found = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        argv = cmd.split()
        if len(argv) < 2 or "python" not in Path(argv[0]).name or int(pid) == os.getpid():
            continue
        script = next((s for s in SCRIPTS if argv[1] == s or argv[1].endswith("/" + s)), None)
        cli = argv[1:3] == ["-m", "strategy_lab"] and argv[3:4] == ["run"]
        if not (script or cli):
            continue
        where = Path(argv[1]).parent.parent if script and os.path.isabs(argv[1]) else _cwd(int(pid))
        if where is not None and where.resolve() == root.resolve():
            found.append(f"{pid}: {cmd[:160]}")
    return found


def _cwd(pid: int) -> Path | None:
    got = subprocess.run(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], capture_output=True, text=True,
                         check=False).stdout
    line = next((x for x in got.splitlines() if x.startswith("n")), None)
    if line is None:
        LOG.info("process %d: its working directory could not be read (it ended, or is another user's)", pid)
        return None
    return Path(line[1:])


def uncommitted_code(root: Path = ROOT_DIR) -> list[str]:
    """Evaluation files (strategies and the modules that compute numbers) that differ from git's last commit, or are
    not in git: results a run saves would be stamped with code nobody can check out."""
    files = provenance.all_evaluation_files()
    got = subprocess.run(["git", "status", "--porcelain", "--", *files], cwd=root, capture_output=True, text=True,
                         check=False)
    if got.returncode != 0:
        LOG.warning("git status failed (%s): uncommitted evaluation code not checked", got.stderr.strip()[:200])
        return []
    return [line[3:] for line in got.stdout.splitlines() if line.strip()]


class Runner:
    def __init__(self, path: Path, command: Callable[[int], list[str]] = rerun_command, root: Path = ROOT_DIR):
        self.path = Path(path)
        self.command = command
        self.root = root
        self._lock = threading.Lock()
        self._procs: dict[int, subprocess.Popen] = {}
        self._stops: dict[int, float] = {}
        self._last_lost_check = 0.0

    def _conn(self):
        return db.connect(self.path)

    # -------------------------------------------------------------------------------------------- start, resume, stop
    def _refuse_if_busy(self, conn) -> None:
        on = db.active_run(conn)
        if on is not None:
            raise Refused(f"Run {on['id']} is on ({on['state']}): stop it, or wait for it to end, first.")
        others = batches_in_terminals(self.root)
        if others:
            raise Refused("A batch runs on this checkout outside the page: wait for it or stop it first ("
                          + "; ".join(others) + ").")

    def start(self, *, single_assets: bool, everything: bool, only: dict | None, confirm: bool) -> int:
        with self._lock:
            conn = self._conn()
            try:
                self._refuse_if_busy(conn)
                dirty = [] if confirm else uncommitted_code(self.root)
                if dirty:
                    shown = ", ".join(dirty[:4]) + (f" and {len(dirty) - 4} more" if len(dirty) > 4 else "")
                    raise Refused(f"The evaluation code has changes that are not committed ({shown}): the results "
                                  "would be stamped with code that is not in git. Start anyway?", confirm=True)
                rid = runs.create(conn, requested_by="ui", single_assets=single_assets, everything=everything,
                                  only=only)
                self._spawn(conn, rid)
                LOG.info("run %d started from the page (single assets %s, everything %s%s)", rid, single_assets,
                         everything, f", only {only}" if only else "")
                return rid
            finally:
                conn.close()

    def resume(self, run_id: int, *, confirm: bool) -> None:
        with self._lock:
            conn = self._conn()
            try:
                run = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
                if run is None:
                    raise NotFound(f"no run {run_id}")
                failed = conn.execute("SELECT count(*) FROM run_job WHERE run_id = ? AND state = 'failed'",
                                      (run_id,)).fetchone()[0]
                if run["state"] not in ("stopped", "interrupted", "done") or (run["state"] == "done" and not failed):
                    raise Refused(f"Run {run_id} is {run['state']}" + (", nothing failed" if run["state"] == "done"
                                                                       else "") + ": nothing to resume.")
                self._refuse_if_busy(conn)
                now = provenance.sha_of(provenance.all_evaluation_files()) or ""
                if not confirm and run["code_sha_at_start"] != now:
                    raise Refused(f"The evaluation code changed since run {run_id} started: what it finished was "
                                  "computed by the older code, the rest would be by the new. Resume anyway, or start "
                                  "a new run?", confirm=True)
                self._spawn(conn, run_id)
                LOG.info("run %d resumed from the page", run_id)
            finally:
                conn.close()

    def _spawn(self, conn, run_id: int) -> None:
        path = runs.log_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ) | {"STRATEGY_LAB_DB": str(self.path)}
        with open(path, "ab") as out:
            p = subprocess.Popen(self.command(run_id), cwd=self.root, stdin=subprocess.DEVNULL, stdout=out,
                                 stderr=subprocess.STDOUT, start_new_session=True, env=env)
        self._procs[run_id] = p
        with db.write(conn):
            conn.execute("UPDATE run SET log_path = ? WHERE id = ?", (shown(path), run_id))

    def stop(self, run_id: int) -> None:
        with self._lock:
            conn = self._conn()
            try:
                run = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
                if run is None:
                    raise NotFound(f"no run {run_id}")
                on = db.active_run(conn)
                if on is None or on["id"] != run_id:
                    raise Refused(f"Run {run_id} is not on ({run['state']}).")
                with db.write(conn):
                    conn.execute("UPDATE run SET state = 'stopping' WHERE id = ? AND state IN ('starting', 'running')",
                                 (run_id,))
                group = run["pgid"] or (self._procs[run_id].pid if run_id in self._procs else None)
                if group is None:
                    LOG.warning("run %d is starting in no process the server knows: marked stopped", run_id)
                    self._mark_stopped(conn, run_id, "stopped before its process started")
                    return
                self._signal(group, signal.SIGTERM)
                self._stops[run_id] = time.time() + STOP_GRACE
                LOG.info("run %d: stop asked (SIGTERM to process group %d)", run_id, group)
            finally:
                conn.close()

    @staticmethod
    def _signal(group: int, sig: int) -> None:
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            LOG.info("process group %d is gone already", group)
        except PermissionError:          # macOS answers so for a group whose processes all ended, not yet reaped
            LOG.info("process group %d: its processes have ended, nothing left to signal", group)

    @staticmethod
    def _mark_stopped(conn, run_id: int, error: str | None = None, state: str = "stopped") -> None:
        with db.write(conn):
            conn.execute("UPDATE run_job SET state = 'stopped' WHERE run_id = ? AND state = 'running'", (run_id,))
            conn.execute("UPDATE run SET state = ?, pid = NULL, pgid = NULL, finished_at = ?, error = coalesce(?, error) "
                         "WHERE id = ?", (state, db.now(), error, run_id))

    # -------------------------------------------------------------------------------------------- supervision
    def supervise(self) -> None:
        """Reap the runs' processes that ended, kill a stopped run past its grace, mark runs whose process vanished
        interrupted. Called by the server's watcher every second or so."""
        conn = None
        try:
            for rid, p in list(self._procs.items()):
                code = p.poll()
                if code is None:
                    continue
                del self._procs[rid]
                conn = conn or self._conn()
                run = conn.execute("SELECT state FROM run WHERE id = ?", (rid,)).fetchone()
                if run is not None and run["state"] in ("starting", "running", "stopping"):
                    # it ended without marking its end: it broke off before strategy_lab.runs took over, or was killed
                    LOG.error("run %d: its process ended (exit %s) while the run said %s: see %s", rid, code,
                              run["state"], runs.log_path(rid))
                    self._mark_stopped(conn, rid, f"its process ended with exit code {code} before the run could "
                                                  f"mark its end: see {shown(runs.log_path(rid))}",
                                       "stopped" if run["state"] == "stopping" else "interrupted")
                elif code not in (0, 1):
                    LOG.warning("run %d: its process ended with exit code %s", rid, code)
            for rid, deadline in list(self._stops.items()):
                conn = conn or self._conn()
                run = conn.execute("SELECT state, pgid FROM run WHERE id = ?", (rid,)).fetchone()
                if run is None or run["state"] != "stopping":
                    del self._stops[rid]
                    continue
                if time.time() >= deadline:
                    group = run["pgid"] or (self._procs[rid].pid if rid in self._procs else None)
                    LOG.warning("run %d did not end %d s after its stop: killed", rid, STOP_GRACE)
                    if group is not None:
                        self._signal(group, signal.SIGKILL)
                    self._mark_stopped(conn, rid, f"killed: it had not ended {STOP_GRACE} s after its stop")
                    del self._stops[rid]
            if time.time() - self._last_lost_check >= runs.HEARTBEAT:
                self._last_lost_check = time.time()
                conn = conn or self._conn()
                runs.interrupted(conn)
        finally:
            if conn is not None:
                conn.close()

    # -------------------------------------------------------------------------------------------- what the page shows
    def snapshot(self, conn, code_sha: str | None = None) -> dict | None:
        """The run that is on, else the last one: its stages, jobs and how long it has left (`code_sha`: the
        evaluation code's hash now, when the caller keeps it)."""
        on = db.active_run(conn)
        run = on or conn.execute("SELECT * FROM run ORDER BY id DESC LIMIT 1").fetchone()
        if run is None:
            return None
        rid = run["id"]
        counts: dict[str, dict[str, int]] = {}
        for r in conn.execute("SELECT stage, state, count(*) AS n FROM run_job WHERE run_id = ? GROUP BY stage, state",
                              (rid,)):
            counts.setdefault(r["stage"], {})[r["state"]] = r["n"]
        stages = [{"stage": s, "total": sum(c.values()),
                   **{k: c.get(k, 0) for k in ("done", "failed", "skipped", "stopped", "queued", "running")}}
                  for s, c in ((s, counts[s]) for s in ("lists", "single_assets") if s in counts)]
        running = [dict(r) for r in conn.execute(
            "SELECT stage, strategy, list_id, timeframe, started_at, expected_seconds FROM run_job WHERE run_id = ? "
            "AND state = 'running' ORDER BY started_at", (rid,))]
        failed = [dict(r) | {"error": None if r["error"] is None else r["error"][:ERROR_CHARS]} for r in conn.execute(
            "SELECT stage, strategy, list_id, timeframe, attempts, error FROM run_job WHERE run_id = ? AND "
            "state = 'failed' ORDER BY finished_at", (rid,))]
        only = json.loads(run["only"]) if run["only"] else None
        now_sha = code_sha if code_sha is not None else provenance.sha_of(provenance.all_evaluation_files()) or ""
        return {
            "id": rid, "state": run["state"], "stage": run["stage"], "active": on is not None,
            "requested_by": run["requested_by"], "requested_at": run["requested_at"], "started_at": run["started_at"],
            "finished_at": run["finished_at"], "single_assets": bool(run["single_assets"]),
            "everything": bool(run["everything"]), "narrowed_to": runs.narrowed_to(only), "resumed": run["resumed"],
            "code_changed": run["code_sha_at_start"] != now_sha, "stages": stages, "running": running,
            "failed": failed, "eta_seconds": self._eta(conn, rid) if on is not None else None,
            "cpu_seconds": run["cpu_seconds"], "error": run["error"], "log_path": run["log_path"]}

    @staticmethod
    def _eta(conn, rid: int) -> float | None:
        """Seconds left: the expected time of what is left over the pace of the jobs ended in the last RECENT
        seconds (expected seconds done per second), None before three have ended."""
        jobs = conn.execute("SELECT state, expected_seconds, started_at, finished_at FROM run_job WHERE run_id = ? AND "
                            "state IN ('queued', 'running', 'done', 'failed')", (rid,)).fetchall()
        weigh = any(j["expected_seconds"] > 0 for j in jobs)
        size = (lambda j: j["expected_seconds"]) if weigh else (lambda j: 1.0)
        now = time.time()
        recent = [j for j in jobs if j["state"] in ("done", "failed") and j["finished_at"]
                  and now - _epoch(j["finished_at"]) <= RECENT]
        if len(recent) < 3:
            return None
        began = min(_epoch(j["started_at"]) or now for j in recent)
        pace = sum(size(j) for j in recent) / max(now - max(began, now - RECENT), 1.0)
        left = sum(size(j) for j in jobs if j["state"] == "queued")
        for j in jobs:
            if j["state"] == "running":
                left += max(size(j) - (now - (_epoch(j["started_at"]) or now)) * (1.0 if weigh else 0.0), 0.0)
        return left / pace if pace > 0 else None
