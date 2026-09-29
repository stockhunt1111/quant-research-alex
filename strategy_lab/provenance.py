"""Which code produced a saved result, so a result the current code would not reproduce is flagged as stale
instead of being mixed into a board of fresh ones.

A result is stamped with a hash of its evaluation's source files: every module of strategy_lab that computes
numbers, plus the strategy's own module. The hash is taken from the files as they were when this process first
read them (`freeze()` at the start of a batch pins that moment), and checked later against the files on disk.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from strategy_lab.config import ROOT_DIR

# modules that fetch data or present results: editing them does not change any evaluation
NOT_EVALUATION = {"__main__.py", "batch.py", "board.py", "catalog.py", "check.py", "db.py", "forward.py", "lists.py",
                  "log.py", "lookahead.py", "market_cap.py", "minute_refresh.py", "orderly.py", "portfolio.py",
                  "provenance.py", "refresh.py", "runs.py", "seed.py", "spread_refresh.py"}

_frozen: dict[str, bytes] = {}


def evaluation_files(strategy_file: str) -> list[str]:
    core = sorted(p for p in (ROOT_DIR / "strategy_lab").rglob("*.py") if p.name not in NOT_EVALUATION)
    return [str(p.relative_to(ROOT_DIR)) for p in core + [Path(strategy_file).resolve()]]


def _hash(files: list[str], read) -> str | None:
    h = hashlib.sha256()
    for f in files:
        data = read(f)
        if data is None:
            return None
        h.update(f.encode() + b"\0" + data)
    return h.hexdigest()[:16]


def _read_disk(f: str) -> bytes | None:
    p = ROOT_DIR / f
    return p.read_bytes() if p.exists() else None


def _read_frozen(f: str) -> bytes | None:
    if f not in _frozen:
        data = _read_disk(f)
        if data is None:
            return None
        _frozen[f] = data
    return _frozen[f]


def freeze(strategy_files: list[str]) -> None:
    """Pin the source of every evaluation file now, before a long batch runs on the code already imported."""
    for f in set(evaluation_files(strategy_files[0]) if strategy_files else []) | {
            str(Path(s).resolve().relative_to(ROOT_DIR)) for s in strategy_files}:
        _read_frozen(f)


def stamp(strategy_file: str, also: tuple[str, ...] = ()) -> dict:
    """The evaluation's files and their hash. `also`: repo files one kind of result reads beyond them, which the others
    do not (strategy_pick takes its candidates in the order of the lists `lists.py` gives): their edits make that kind
    stale, and every other result stays current."""
    files = evaluation_files(strategy_file) + [f for f in also if f not in evaluation_files(strategy_file)]
    return {"files": files, "sha": _hash(files, _read_frozen)}


def all_evaluation_files() -> list[str]:
    """Every module that computes numbers and every strategy: what a run's results can be stamped with."""
    core = sorted(p for p in (ROOT_DIR / "strategy_lab").rglob("*.py") if p.name not in NOT_EVALUATION)
    own = sorted((ROOT_DIR / "strategies").glob("*.py"))
    return [str(p.relative_to(ROOT_DIR)) for p in core + own]


def sha_of(files: list[str]) -> str | None:
    """The hash of these repo files as they are on disk now (None when one is missing)."""
    return _hash(files, _read_disk)


def is_current(code: dict | None) -> bool:
    return bool(code) and code.get("sha") is not None and _hash(code["files"], _read_disk) == code["sha"]
