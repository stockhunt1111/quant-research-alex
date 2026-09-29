"""The web UI as a service of this Mac (launchd): started at login and whenever it stops, whatever terminal, editor or
session is open or closed; its output appended to logs/server.log.

    python -m server.service install     # make service: written to ~/Library/LaunchAgents and started
    python -m server.service restart     # make service-restart: started again on the code on disk
    python -m server.service stop        # make service-stop: stopped, and no longer started at login
    python -m server.service status

It runs `python -m server` from this checkout's venv on 127.0.0.1:8600. A re-run it started goes on while it restarts
(a run is a process group of its own). After a change of server/ or of strategy_lab/ it needs a restart to use the new
code; a page built again (make ui) needs none.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from strategy_lab.config import ROOT_DIR

from server.__main__ import LOG, PORT

LABEL = "com.alexsilka.strategy-lab"            # as the Mac's other service of the user's, com.alexsilka.spot
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
URL = f"http://127.0.0.1:{PORT}"
# seconds the server takes to answer after a start: on a machine a re-run's workers load (load average ~100 on ten
# cores, 2026-09-28) it took over the 30 s this was, and the restart said it had failed while the server came up
STARTS_WITHIN = 120


def definition() -> dict:
    return {
        "Label": LABEL,
        "ProgramArguments": [str(ROOT_DIR / ".venv" / "bin" / "python"), "-m", "server"],
        "WorkingDirectory": str(ROOT_DIR),
        "RunAtLoad": True,                      # started at login
        "KeepAlive": True,                      # and again whenever it stops
        "ThrottleInterval": 10,                 # not more often than every 10 s when it cannot start
        "StandardOutPath": str(LOG),
        "StandardErrorPath": str(LOG),
    }


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    got = subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)
    if check and got.returncode != 0:
        raise SystemExit(f"launchctl {' '.join(args)} failed ({got.returncode}): {(got.stderr or got.stdout).strip()}")
    return got


def loaded() -> bool:
    return _launchctl("print", f"{_domain()}/{LABEL}", check=False).returncode == 0


def answers() -> bool:
    try:
        with urllib.request.urlopen(URL + "/api/runs/current", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _listener() -> str | None:
    """What holds the service's port now, as lsof names it (None: the port is free)."""
    got = subprocess.run(["lsof", "-nP", f"-iTCP:{PORT}", "-sTCP:LISTEN", "-Fpc"], capture_output=True, text=True,
                         check=False).stdout.split()
    pid = next((x[1:] for x in got if x.startswith("p")), None)
    cmd = next((x[1:] for x in got if x.startswith("c")), None)
    return None if pid is None else f"{cmd} (pid {pid})"


def _wait_for_answer() -> None:
    deadline = time.time() + STARTS_WITHIN
    while time.time() < deadline:
        if answers():
            print(f"Strategy Lab: {URL} (service {LABEL}; log {LOG.relative_to(ROOT_DIR)})")
            return
        time.sleep(0.5)
    tail = LOG.read_text(errors="replace").splitlines()[-15:] if LOG.exists() else []
    raise SystemExit(f"the service did not answer on {URL} within {STARTS_WITHIN} s; the log's end:\n" + "\n".join(tail))


def install() -> None:
    if loaded():
        _launchctl("bootout", f"{_domain()}/{LABEL}")
    holder = _listener()
    if holder is not None:
        raise SystemExit(f"port {PORT} is taken by {holder}: stop it (a server started in a terminal or a session), "
                         "then install the service again")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    with open(PLIST, "wb") as f:
        plistlib.dump(definition(), f)
    _launchctl("bootstrap", _domain(), str(PLIST))
    _wait_for_answer()


def restart() -> None:
    if not loaded():
        raise SystemExit("the service is not installed: make service")
    _launchctl("kickstart", "-k", f"{_domain()}/{LABEL}")
    _wait_for_answer()


def stop() -> None:
    if loaded():
        _launchctl("bootout", f"{_domain()}/{LABEL}")
    if PLIST.exists():
        PLIST.unlink()                          # not started at the next login either
    print(f"the service is stopped and no longer starts at login ({PLIST} removed)")


def status() -> None:
    if not loaded():
        print(f"not installed ({PLIST} {'exists' if PLIST.exists() else 'absent'})")
        return
    info = _launchctl("print", f"{_domain()}/{LABEL}").stdout
    keep = ("state =", "pid =", "last exit code =", "runs =")
    top = [line.strip() for line in info.splitlines()             # the job's own lines, not its sections'
           if line.startswith("\t") and not line.startswith("\t\t") and line.strip().startswith(keep)]
    print(LABEL, *top, sep="\n  ")
    print(f"  {URL}: {'answers' if answers() else 'does not answer'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="the web UI as a service of this Mac (launchd)")
    ap.add_argument("what", choices=["install", "restart", "stop", "status"])
    args = ap.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("a service of launchd, macOS's own: on a Linux server the same is a unit of systemd")
    {"install": install, "restart": restart, "stop": stop, "status": status}[args.what]()


if __name__ == "__main__":
    main()
