"""Every strategy on the lists and timeframes it applies to (strategy_lab.runs.LIST_RUNS), as one run.

    python scripts/week1.py                                   # everything
    python scripts/week1.py --only trend                      # strategies whose name contains "trend"
    python scripts/week1.py --universes etf_core fx_majors    # only these lists
    python scripts/week1.py --names rsi2_connors keltner_breakout   # only these strategies
    python scripts/week1.py --skip-names ml_direction                # all but these strategies
    python scripts/week1.py --skip-fresh                             # only jobs the current code has not produced
    python scripts/week1.py --workers 8                              # processes evaluating at once (default 10)

A run like any other (strategy_lab.runs): it shows in the UI and stops and resumes there, and each evaluation is saved
to the app's database as it ends. `scripts/rerun.py` is the full re-run, the single assets included.
"""
from __future__ import annotations

import argparse

from strategy_lab import db, log, runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--universes", nargs="*")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--skip-names", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=10, help="processes evaluating at once")
    ap.add_argument("--skip-fresh", action="store_true", help="skip jobs whose result the current code produced")
    args = ap.parse_args()
    names = [n for n, _ in runs.LIST_RUNS if (not args.only or args.only in n) and (not args.names or n in args.names)
             and n not in args.skip_names]
    only = {"stages": ["lists"], "strategies": names, "lists": args.universes or []}
    conn = db.connect()
    run_id = runs.create(conn, requested_by="cli", single_assets=False, everything=not args.skip_fresh, only=only)
    conn.close()
    log.setup(f"week1_run{run_id}")
    raise SystemExit(runs.execute(run_id, workers=args.workers))


if __name__ == "__main__":
    main()
