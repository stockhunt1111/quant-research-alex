"""Trade ledger from the weights an engine run held.

A trade is a run of bars in which an instrument's weight keeps one sign. A resize with the same sign is part of
the same trade; a flip is an exit plus an entry. Its prices follow the fill convention of the run: `gross_return` is
its price move from its entry fill to its exit fill, on its side. `net_return` is what the account made from it, over
the money it was entered with: the price moves of the units it held between each of its fills (its entry, the resizes
a book's rebalance or a change of its share made, its exit), less the costs of those fills (`engine.costs`: a spot
quote of a commodity pays its broker's spread on the bar of each) and what holding it paid and was paid (a perp's
funding, a short's borrow, the dividends of the ex-dates it was held into), each as the engine's walk charged it
(`backtest.Parts`). A trade held at one size nets its price move less its two fills' costs and its carry; one a book
traded back to its weight each week does not: a winner is sold down at every rebalance and a loser bought up. Until
2026-10-01 every trade was measured as if its units never changed after its entry: the long of MYX a weekly book
held through its squeeze of 2025-09 (xs_momentum on crypto Top-50 1d) read +1423% net, +178% of the account, where it
made the account +102.5%: +820% of the money it entered with. `size`: the share of the equity it was entered with
(its last fill on its first bar), so that `made` (size x net_return) is what it made the account, a share of the equity
after its entry's fills.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from strategy_lab import log
from strategy_lab.data.bars import Panel
from strategy_lab.engine.backtest import Parts, Result, ended

LOG = log.get("trades")

COLUMNS = ["instrument", "side", "entry_time", "exit_time", "entry_px", "exit_px", "bars", "size", "gross_return",
           "net_return", "exit_reason"]


def ledger(panel: Panel, res: Result, live: pd.DataFrame | None = None) -> pd.DataFrame:
    """Trades of an engine run. With `live` (the universe's membership mask), a position that ends because its
    instrument left the universe is marked `left_list` rather than `signal`."""
    return trades_of(panel, res.weights, res.exits, res.fill, res.parts, live, res.liquidated)


def trades_of(panel: Panel, weights: pd.DataFrame, exits: pd.DataFrame, fill: str, parts: Parts | None,
              live: pd.DataFrame | None = None, liquidated: pd.DataFrame | None = None,
              stitched: bool = False) -> pd.DataFrame:
    """Trades of the positions `weights` (the weight each was last filled to, on the panel's bars) with their intrabar
    exits `exits`, filled by the `fill` convention, and what each made from `parts` (what every position made on every
    bar): an engine run's (`ledger`), or a record `stitched` from several runs, one a window: a window's first bar is
    its run's whole, the gap of what that run held into it included, and what the run sold there out of a position the
    record did not hold belongs to no trade of the record. An exit the engine made by liquidating a short
    (`liquidated`) is marked `liquidated` rather than `exit_rule`."""
    if parts is None:
        raise ValueError("a run made without what its positions made on each bar (backtest.run(parts=False)) has no "
                         "trades to measure")
    idx = panel.index
    n = len(idx)
    stopped = ended(panel)
    next_open = fill == "next_open"
    before = np.r_[1.0, np.cumprod(1.0 + parts.net)[:-1]]            # the equity at the close before each bar
    order = np.argsort(parts.inst, kind="stable")                      # each instrument's pieces, in the order of bars
    inst_of = parts.inst[order]
    lo_hi = np.searchsorted(inst_of, np.arange(len(panel.ids) + 1))
    bar_of, held_in, held_on = parts.bar[order], parts.held_in[order], parts.held_on[order]
    left_over = 0.0
    parts_out = []
    for j, inst in enumerate(panel.ids):
        side_all = np.sign(weights[inst].to_numpy())
        ex_all = exits[inst].to_numpy()
        start, end = _spans(side_all, ~np.isnan(ex_all))
        if not len(start):
            continue
        o = panel.open[inst].to_numpy()
        c = panel.close[inst].ffill().to_numpy()
        stop = stopped[inst].to_numpy()
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
        reason = np.select([liquidation, by_rule, at_end, delisted, left],
                           ["liquidated", "exit_rule", "open_at_end", "delisted", "left_list"], "signal").astype(object)
        size = np.abs(weights[inst].to_numpy()[start])
        a, b = lo_hi[j], lo_hi[j + 1]
        cash, unassigned = _made(start, end, bar_of[a:b], held_in[a:b], held_on[a:b], before)
        left_over += unassigned
        share = cash / (before[start] * parts.opened[start])              # over the equity after the entry's fills
        parts_out.append(pd.DataFrame({"instrument": inst, "side": side.astype(np.int64), "entry_time": idx[entry_at],
                                       "exit_time": idx[exit_at], "entry_px": entry_px, "exit_px": exit_px,
                                       "bars": end - start + 1, "size": size,
                                       "gross_return": side * (exit_px / entry_px - 1.0), "net_return": share / size,
                                       "exit_reason": reason}))
    if stitched:
        LOG.debug("%.3g of the equity a stitched record's runs made out of positions it did not hold, at its windows' "
                  "first bars, is in no trade", left_over)
    elif left_over > 1e-9 * float(before.max()):
        LOG.warning("%.3g of the equity the positions made on bars no trade of theirs held is left out of the trades",
                    left_over)
    if not parts_out:
        return pd.DataFrame([], columns=COLUMNS)
    return pd.concat(parts_out, ignore_index=True)[COLUMNS]


def made(trades: pd.DataFrame) -> pd.Series:
    """What each trade made for the account, as a share of the equity: its net return times the share it was entered
    with (`size`; a ledger kept without it counts each trade whole)."""
    size = trades["size"] if "size" in trades.columns else pd.Series(np.nan, index=trades.index)
    return trades["net_return"] * size.fillna(1.0)


@njit(cache=True)
def _made(start, end, bar, held_in, held_on, before):
    """What each trade (bars `start` to `end`, in order) made the account, in the units of the equity before the
    record's first bar, from one instrument's pieces of its bars (`backtest.Parts`, in the order of bars; `before`: the
    equity at the close before each bar). A bar's piece held into it is the trade's that held the position through the
    close before (a sale at the open, a liquidation in the gap), its piece held from the open the trade's that holds it
    from there; a piece whose own trade the weights do not show (a record stitched from several runs, whose window
    starts with what its run held into it) goes to the trade on the other side of the open. Beside it, what no trade
    took (none should be left)."""
    cash = np.zeros(len(start))
    left = 0.0
    for q in range(len(bar)):
        t = bar[q]
        on = _covering(start, end, t)
        into = _covering(start, end, t - 1)
        for piece, first, second in ((held_in[q], into, on), (held_on[q], on, into)):
            if piece == 0.0:
                continue
            k = first if first >= 0 else second
            if k >= 0:
                cash[k] += piece * before[t]
            else:
                left += abs(piece * before[t])
    return cash, left


@njit(cache=True)
def _covering(start, end, t):
    """The trade among `start`..`end` (ordered, apart) whose bars hold bar t; -1 when none does."""
    k = np.searchsorted(start, t, side="right") - 1
    return k if k >= 0 and end[k] >= t else -1


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
