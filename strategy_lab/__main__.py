"""Command line.

    python -m strategy_lab run sma_cross --universe etf_core --tf 1d
    python -m strategy_lab run sma_cross ibs --universe crypto_top10 --tf 4h 1d     # strategies by name: strategies/<name>.py
    python -m strategy_lab board                  # the research board, from the app's database
    python -m strategy_lab db migrate             # create db/app.sqlite or bring it to the current schema
    python -m strategy_lab db judge               # work out the luck bars (the tries) over the list results and the
                                                  # single assets' now
    python -m strategy_lab db lists               # the lists' sizes and costs and the instruments' liquidity, from bars
    python -m strategy_lab db pick                # the ML task's choice of a strategy for each instrument
"""
from __future__ import annotations

import argparse

from strategy_lab import board, db, lists, log
from strategy_lab.config import TIMEFRAMES
from strategy_lab.evaluate import evaluate
from strategy_lab.strategy import load


def main() -> None:
    ap = argparse.ArgumentParser(prog="strategy_lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate strategies on a universe and timeframe")
    r.add_argument("names", nargs="+", help="strategies by name, each strategies/<name>.py, e.g. sma_cross ibs")
    r.add_argument("--universe", "-u", required=True, nargs="+")
    r.add_argument("--tf", "-t", required=True, nargs="+", choices=list(TIMEFRAMES))
    r.add_argument("--start")
    r.add_argument("--end")
    r.add_argument("--fill", default="next_open", choices=["next_open", "next_close"])
    r.add_argument("--no-mc", action="store_true")
    r.add_argument("--no-checks", action="store_true",
                   help="leave out the robustness checks but Monte Carlo and random timing (a quick look)")
    r.add_argument("--outside-lists", action="store_true",
                   help="allow lists outside strategy_lab.lists: only when the user asks for them")
    sub.add_parser("board", help="print the research board from the app's database")
    d = sub.add_parser("db", help="the app's database")
    d.add_argument("what", choices=["migrate", "judge", "lists", "pick"])
    args = ap.parse_args()
    if args.cmd == "run":
        if not args.outside_lists:
            for name in args.names:
                lists.refuse_outside(args.universe, lists.basket_lists(), f"{name} as a basket")
        log.setup("run")
        for s in map(load, args.names):
            for u in args.universe:
                for tf in args.tf:
                    evaluate(s, u, tf, start=args.start, end=args.end, fill=args.fill,
                             monte_carlo=not args.no_mc, robustness=not args.no_checks)
    elif args.cmd == "board":
        log.setup()
        n = board.tries()
        df = board.collect(n_tries=n)
        print(board._fmt(df).to_string(index=False))
        other = board.not_ranked()
        print(f"\npicked from {n}" + (f"; not ranked: {', '.join(f'{u} {k}' for u, k in other.items())}" if other else ""))
    else:
        log.setup(f"db_{args.what}")
        conn = db.connect()
        try:
            if args.what == "judge":
                print(board.refresh_tries(conn))
                alone = board.refresh_asset_tries(conn)
                print("no single-asset result" if alone is None else f"single assets: {alone.every}")
            elif args.what == "lists":
                from strategy_lab import catalog
                catalog.refresh(conn)
            elif args.what == "pick":
                from strategy_lab import strategy_pick
                strategy_pick.pick_all(conn)
            else:
                print(f"{db.DB_PATH}: schema {conn.execute('PRAGMA user_version').fetchone()[0]}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
