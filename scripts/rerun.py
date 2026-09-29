"""A re-run of the research, from a terminal: the run the UI's Re-run starts (strategy_lab.runs).

    python scripts/rerun.py                       # every strategy on every list, skipping what the current code produced
    python scripts/rerun.py --single-assets       # and each strategy on each instrument alone, and the ML task's picks
    python scripts/rerun.py --everything          # re-evaluate what the current code produced too (after a data refresh)
    python scripts/rerun.py --run-id 12           # run or resume run 12 (the UI starts its runs this way)
    python scripts/rerun.py --workers 8           # processes evaluating at once (default 10)

    # some of it only: these lists, strategies or timeframes (any of the three), e.g. one market after its data changed
    python scripts/rerun.py --single-assets --everything -u stockhunt_commodities
    python scripts/rerun.py --everything --names ibs ibs_ml_filter -u etf_core fx_majors -t 1h 4h

A narrowed run evaluates the jobs of what it names on every stage (its lists as books, their instruments alone with
--single-assets) and ends as a run of everything does: the ML task's picks (with --single-assets), which read every
result, and the lists' figures the pages show, from every list's bars. It shows in the UI while it runs, and stops
there or with ^C; resuming it starts its jobs that are not done. Its log is logs/rerun_<id>.log, resumes appended.
"""
from __future__ import annotations

import argparse

from strategy_lab import db, log, runs
from strategy_lab.config import ROOT_DIR, TIMEFRAMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=int, help="run or resume this run")
    ap.add_argument("--single-assets", action="store_true", help="each strategy on each instrument alone too")
    ap.add_argument("--everything", action="store_true", help="re-evaluate what the current code already produced")
    ap.add_argument("--workers", type=int, default=10, help="processes evaluating at once")
    ap.add_argument("--names", nargs="+", help="these strategies only (strategies/<name>.py)")
    ap.add_argument("--universe", "-u", nargs="+", help="these lists only, as books and, with --single-assets, alone")
    ap.add_argument("--tf", "-t", nargs="+", choices=list(TIMEFRAMES), help="these timeframes only")
    args = ap.parse_args()
    only = runs.only_of(args.names, args.universe, args.tf)
    if args.run_id and (only or args.single_assets or args.everything):
        ap.error("--run-id resumes a run as it was asked for: it takes no other choice")
    conn = db.connect()
    run_id = args.run_id or runs.create(conn, requested_by="cli", single_assets=args.single_assets,
                                        everything=args.everything, only=only)
    path = runs.log_path(run_id)
    with db.write(conn):
        conn.execute("UPDATE run SET log_path = ? WHERE id = ?", (str(path.relative_to(ROOT_DIR)), run_id))
    conn.close()
    log.setup(path=path)
    raise SystemExit(runs.execute(run_id, workers=args.workers))


if __name__ == "__main__":
    main()
