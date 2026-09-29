"""Logging setup: stderr always; a run log file under logs/ when a CLI asks for one."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from strategy_lab.config import LOGS_DIR

_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"strategy_lab.{name}")


def setup(run_name: str | None = None, level: int = logging.INFO, *, path: Path | None = None) -> None:
    """stderr, and a log file: `run_name`'s, stamped with the time, or `path`, appended to (a re-run's one file
    across its resumes) unless stderr already goes there (a run the server started writes its output to it)."""
    root = logging.getLogger("strategy_lab")
    if root.handlers:
        return
    root.setLevel(level)
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter(_FMT))
    root.addHandler(h)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not (path.exists() and os.path.samestat(os.fstat(sys.stderr.fileno()), path.stat())):
            fh = logging.FileHandler(path)
            fh.setFormatter(logging.Formatter(_FMT))
            root.addHandler(fh)
    elif run_name:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(LOGS_DIR / f"{run_name}_{stamp}.log")
        fh.setFormatter(logging.Formatter(_FMT))
        root.addHandler(fh)
