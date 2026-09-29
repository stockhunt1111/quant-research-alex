"""Trade ledger from the weights an engine run held.

A trade is a run of bars in which an instrument's weight keeps one sign. A resize with the same sign is part of
the same trade; a flip is an exit plus an entry. Prices follow the fill convention of the run, and so do the costs
of its entry and its exit (`engine.costs`: a spot quote of a commodity pays its broker's spread on the bar of each).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from strategy_lab.data.bars import Panel
from strategy_lab.engine import costs
from strategy_lab.engine.backtest import Result, ended

COLUMNS = ["instrument", "side", "entry_time", "exit_time", "entry_px", "exit_px", "bars", "gross_return",
           "net_return", "exit_reason"]


def ledger(panel: Panel, res: Result, live: pd.DataFrame | None = None) -> pd.DataFrame:
    """Trades of an engine run. With `live` (the universe's membership mask), a position that ends because its
    instrument left the universe is marked `left_list` rather than `signal`."""
    return trades_of(panel, res.weights, res.exits, res.fill, live, res.liquidated)


def trades_of(panel: Panel, weights: pd.DataFrame, exits: pd.DataFrame, fill: str,
              live: pd.DataFrame | None = None, liquidated: pd.DataFrame | None = None) -> pd.DataFrame:
    """Trades of the positions `weights` (the weight each was last filled to, on the panel's bars) with their intrabar
    exits `exits`, filled by the `fill` convention: an engine run's (`ledger`), or a record stitched from several. An
    exit the engine made by liquidating a short (`liquidated`) is marked `liquidated` rather than `exit_rule`."""
    idx = panel.index
    n = len(idx)
    stopped = ended(panel)
    rates = costs.rates(panel)
    next_open = fill == "next_open"
    parts = []
    for j, inst in enumerate(panel.ids):
        side_all = np.sign(weights[inst].to_numpy())
        ex_all = exits[inst].to_numpy()
        start, end = _spans(side_all, ~np.isnan(ex_all))
        if not len(start):
            continue
        o = panel.open[inst].to_numpy()
        c = panel.close[inst].ffill().to_numpy()
        stop = stopped[inst].to_numpy()
        at_open, inside, at_close = (rates.of(j, w) for w in costs.WHERE)
        side = side_all[start]
        entry_at = start if next_open else start - 1
        entry_px = o[start] if next_open else c[start - 1]
        ex = ex_all[end]
        by_rule = ~np.isnan(ex)
        liquidation = by_rule & (liquidated[inst].to_numpy()[end] if liquidated is not None else False)
        at_end = ~by_rule & (end + 1 >= n)
        k_exit = np.minimum(end + 1 if next_open else end, n - 1)
        decided = end if next_open else end - 1                         # the close at which the target went flat
        left = np.zeros(len(end), dtype=bool)
        if live is not None:
            left = ~live[inst].reindex(idx).fillna(False).astype(bool).to_numpy()[decided]
        delisted = ~by_rule & ~at_end & stop[k_exit]                      # closed at its last close
        after = o[k_exit] if next_open else c[k_exit]
        exit_px = np.select([by_rule, at_end, delisted], [ex, c[end], c[k_exit]], after)
        exit_at = np.select([by_rule | at_end], [end], k_exit)
        paid_in = at_open[start] if next_open else at_close[start - 1]
        paid_out = np.select([by_rule, at_end, delisted], [inside[end], at_close[end], at_close[k_exit]],
                             at_open[k_exit] if next_open else at_close[k_exit])
        reason = np.select([liquidation, by_rule, at_end, delisted, left],
                           ["liquidated", "exit_rule", "open_at_end", "delisted", "left_list"], "signal").astype(object)
        gross = side * (exit_px / entry_px - 1.0)
        parts.append(pd.DataFrame({"instrument": inst, "side": side.astype(np.int64), "entry_time": idx[entry_at],
                                   "exit_time": idx[exit_at], "entry_px": entry_px, "exit_px": exit_px,
                                   "bars": end - start + 1, "gross_return": gross,
                                   "net_return": gross - paid_in - paid_out, "exit_reason": reason}))
    if not parts:
        return pd.DataFrame([], columns=COLUMNS)
    return pd.concat(parts, ignore_index=True)[COLUMNS]


@njit(cache=True)
def _spans(side, exited):
    """First and last bar of every trade: a run of bars of one sign, which an intrabar exit also ends."""
    n = len(side)
    start = np.empty(n, dtype=np.int64)
    end = np.empty(n, dtype=np.int64)
    t = 0
    k = 0
    while k < n:
        if side[k] == 0:
            k += 1
            continue
        e = k
        while e + 1 < n and side[e + 1] == side[k] and not exited[e]:
            e += 1
        start[t] = k
        end[t] = e
        t += 1
        k = e + 1
    return start[:t], end[:t]
