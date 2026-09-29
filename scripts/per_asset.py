"""Each rule strategy on each instrument of a list alone (strategy_lab.per_asset), as one run: the rules on the widest
list of each market and on the ML task's lists, the firm's engine (ml_feature_search) on the ML task's
(strategy_lab.runs.alone_jobs).

    python scripts/per_asset.py                                     # every rule on every list it runs on
    python scripts/per_asset.py --names ml_feature_search --universes us_stocks_mcap10 crypto_mcap10 fx_majors
    python scripts/per_asset.py --skip-fresh                        # only runs the current code has not produced
    python scripts/per_asset.py --workers 8                         # processes evaluating at once (default 10)

A run like any other (strategy_lab.runs): it shows in the UI and stops and resumes there.
"""
from __future__ import annotations

import argparse

from strategy_lab import db, lists, log, runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", help="strategies by name (default: the rules and, on the ML task's lists, "
                                               "the firm's engine)")
    ap.add_argument("--universes", nargs="*")
    ap.add_argument("--workers", type=int, default=10, help="processes evaluating at once")
    ap.add_argument("--skip-fresh", action="store_true", help="skip runs whose results the current code produced")
    ap.add_argument("--outside-lists", action="store_true",
                    help="allow lists outside strategy_lab.lists: only when the user asks for them")
    args = ap.parse_args()
    if not args.outside_lists:
        lists.refuse_outside(args.universes or lists.PER_INSTRUMENT, lists.PER_INSTRUMENT, "each instrument alone")
    only = {"stages": ["single_assets"], "strategies": args.names or [], "lists": args.universes or [],
            "outside_lists": bool(args.outside_lists)}
    conn = db.connect()
    run_id = runs.create(conn, requested_by="cli", single_assets=True, everything=not args.skip_fresh, only=only)
    conn.close()
    log.setup(f"per_asset_run{run_id}")
    raise SystemExit(runs.execute(run_id, workers=args.workers))


if __name__ == "__main__":
    main()
