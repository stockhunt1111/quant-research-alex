"""Forward testing, dry-run: what a strategy would send to the firm's simulator at the last closed bar.

    python -m strategy_lab.forward ema_trend -u etf_core -t 1d [--params '{"fast": 20, "slow": 100}']

Without --params it trades the configuration the walk-forward picked for its latest window (the one behind the
out-of-sample record in reports/), so an evaluation must exist. It loads bars up to now, computes the
strategy's target at the LAST CLOSED bar, compares the side of each
instrument with the side recorded at the previous run (data/forward/<key>.json), and prints the BUY/SELL signals
that the change implies. Nothing is sent: the simulator decides open/add/close/reverse from each strategy's
Inverse/Pyramid settings, and that mapping must be confirmed before any real request. The mapping assumed here is
Inverse = off, Pyramid = off, one simulator strategy per instrument with a fixed size: flat->long BUY,
long->flat SELL, flat->short SELL, short->flat BUY, long->short SELL+SELL, short->long BUY+BUY.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from strategy_lab import db, evaluate, log
from strategy_lab.config import DATA_DIR
from strategy_lab.data.bars import Panel
from strategy_lab.engine import backtest as bt
from strategy_lab.strategy import load

LOG = log.get("forward")
STATE_DIR = DATA_DIR / "forward"
ACTIONS = {(0, 1): ["BUY"], (1, 0): ["SELL"], (0, -1): ["SELL"], (-1, 0): ["BUY"],
           (1, -1): ["SELL", "SELL"], (-1, 1): ["BUY", "BUY"]}


def walk_forward_params(strategy, universe: str, timeframe: str) -> dict:
    """The configuration the walk-forward chose for its latest window; a strategy without a grid has only one."""
    if len(strategy.configs()) == 1:
        return strategy.configs()[0]
    conn = db.connect()
    try:
        rid = db.result_id(conn, strategy.name, universe, timeframe)
        windows = [] if rid is None else db.windows(conn, rid)
    finally:
        conn.close()
    if not windows:
        raise LookupError(f"no walk-forward of {strategy.name} on {universe} {timeframe} saved: evaluate it first, "
                          "or pass --params")
    return windows[-1]["params"]


class StaleData(RuntimeError):
    pass


def max_age(timeframe: str) -> pd.Timedelta:
    """How old the last closed bar may be for a plan: two bars, and never less than a long weekend."""
    return max(pd.Timedelta(days=4), 2 * pd.Timedelta(timeframe.replace("w", "W").replace("d", "D")))


def latest_targets(strategy, universe: str, timeframe: str, params: dict,
                   now: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Series, Panel]:
    """Targets at the last closed bar, and the bars they come from. Refuses stale data (a plan on a weeks-old bar is
    not a forward test) and leaves out instruments without a bar at that time: their last price is older than the
    plan."""
    # the list's names as its evaluation holds them: a ranked list's seats come from its daily bars on every timeframe
    _, panel, member = evaluate._load(universe, timeframe, None, None, keep=False)
    tgt = strategy.target(panel, params, member)
    last = tgt.index.max()
    now = now or pd.Timestamp.now(tz="UTC")
    if now - last > max_age(timeframe):
        raise StaleData(f"{universe} {timeframe}: the last closed bar is {last} ({(now - last).days} days ago); "
                        "refresh the data before planning")
    printed = panel.close.loc[last].notna()
    if not printed.all():
        LOG.warning("%d instrument(s) have no bar at %s and are left out: %s", int((~printed).sum()), last,
                    ", ".join(printed.index[~printed]))
    return last, tgt.loc[last][printed], panel


def exit_levels(exits, panel: Panel, bar: pd.Timestamp) -> str:
    """The exits as the simulator takes them, fractions of the entry price: an exit given in ATRs is that many of each
    instrument's average true range at the last close, over its price there (a trailing one's ATR moves on after)."""
    parts = [f"{k} {v:.2%}" for k, v in (("stop", exits.stop), ("take", exits.take), ("trail", exits.trail))
             if v is not None]
    in_atr = [(k, v) for k, v in (("stop", exits.stop_atr), ("take", exits.take_atr), ("trail", exits.trail_atr))
              if v is not None]
    if in_atr:
        unit = pd.Series(bt.atr(panel)[panel.index.get_loc(bar)], index=panel.ids) / panel.close.loc[bar]
        for k, m in in_atr:
            parts.append(f"{k} {m:g} ATR: " + ", ".join(f"{i} {m * u:.2%}" for i, u in unit.dropna().items()))
    if exits.trail_every == "minute":
        parts.append("the trailing stop re-set at every minute's close from the best price of the minutes since entry")
    return "; ".join(parts)


def plan(key: str, bar: pd.Timestamp, target: pd.Series) -> list[dict]:
    state_path = STATE_DIR / f"{key.replace('/', '-')}.json"
    prev = json.loads(state_path.read_text()) if state_path.exists() else {"sides": {}, "bar": None}
    if prev["bar"] == str(bar):
        LOG.info("%s: bar %s already planned, nothing new", key, bar)
        return []
    orders = []
    sides = {}
    for inst, w in target.items():
        new = int(np.sign(w)) if abs(w) > 1e-12 else 0
        old = int(prev["sides"].get(inst, 0))
        sides[inst] = new
        for action in ACTIONS.get((old, new), []):
            orders.append({"instrument": inst, "action": action, "from": old, "to": new, "weight": float(w), "bar": str(bar)})
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"sides": sides, "bar": str(bar),
                                      "planned_at": datetime.now(timezone.utc).isoformat()}, indent=1))
    return orders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="a strategy by name: strategies/<name>.py")
    ap.add_argument("--universe", "-u", required=True)
    ap.add_argument("--tf", "-t", required=True)
    ap.add_argument("--params")
    args = ap.parse_args()
    log.setup("forward")
    strategy = load(args.name)
    params = json.loads(args.params) if args.params else walk_forward_params(strategy, args.universe, args.tf)
    LOG.info("%s parameters: %s", strategy.name, params)
    exits = strategy.exits(params)
    bar, target, panel = latest_targets(strategy, args.universe, args.tf, params)
    if exits.any():
        LOG.info("exits the simulator strategy must be set up with (fractions of the entry price): %s",
                 exit_levels(exits, panel, bar))
    key = f"{strategy.name}@{args.universe}@{args.tf}"
    orders = plan(key, bar, target)
    LOG.info("%s at bar %s: %d signal(s) (dry-run, nothing sent)", key, bar, len(orders))
    for o in orders:
        print(json.dumps(o))


if __name__ == "__main__":
    main()
