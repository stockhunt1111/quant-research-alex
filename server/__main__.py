"""The web UI's server.

    python -m server                          # http://127.0.0.1:8600 on db/app.sqlite (make serve)
    python -m server --port 8601 --db PATH    # another port, another database file (a worktree's, a snapshot)

As this Mac's service, started at login and restarted when it stops: server.service (make service).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import FrameType

import uvicorn

from strategy_lab import log
from strategy_lab.config import LOGS_DIR

from server.app import create_app
from server.live import Live

PORT = 8600
LOG = LOGS_DIR / "server.log"          # one file, appended to; the service (server.service) sends its output there too


def main() -> None:
    ap = argparse.ArgumentParser(description="Research and a result's popup, live, from the app's database")
    ap.add_argument("--host", default="127.0.0.1", help="the address to listen on (only this machine by default)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--db", type=Path, help="the database file (default db/app.sqlite)")
    args = ap.parse_args()
    log.setup(path=LOG)
    app = create_app(args.db)
    print(f"Strategy Lab: http://{args.host}:{args.port}", flush=True)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=3)
    Server(config, app.state.live).run()


class Server(uvicorn.Server):
    """uvicorn's server whose pages' event streams end as soon as it is asked to stop: an open stream never ends by
    itself, and would hold the shutdown until its grace ran out (a page reconnects to the next server)."""

    def __init__(self, config: uvicorn.Config, live: Live):
        super().__init__(config)
        self.live = live

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        self.live.close_soon()
        super().handle_exit(sig, frame)


if __name__ == "__main__":
    main()
