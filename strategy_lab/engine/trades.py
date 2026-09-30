"""Trade ledger from the weights an engine run held.

A trade is a run of bars in which an instrument's weight keeps one sign. A resize with the same sign is part of
the same trade; a flip is an exit plus an entry. Prices follow the fill convention of the run, and so do the costs
of its entry and its exit (`engine.costs`: a spot quote of a commodity pays its broker's spread on the bar of each).
Its net return is also net of what holding it paid and was paid, per unit of the money it was entered with: a perp's
funding over the bars it was held, a short's borrow, and the dividends of the ex-dates it was held into (a long is
paid them, a short pays them), each on the position's value at the bar's open as the engine charges them (an exit
inside a bar holds half of that bar's funding and borrow, as the engine takes an exit whose minute is not known).
Taken on its prices alone, a long on a perp held through 2021's funding showed the funding as profit. `size`: the
share of the equity it was entered with (its last fill on its first bar), which weighs it in what the account made
from it: a trade a model sized to a tenth made a tenth of what a whole one made.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from strategy_lab.data.bars import Panel
from strategy_lab.engine import backtest as bt
from strategy_lab.engine import costs
from strategy_lab.engine.backtest import Result, ended

COLUMNS = ["instrument", "side", "entry_time", "exit_time", "entry_px", "exit_px", "bars", "size", "gross_return",
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
    terms = bt._terms(panel)
    funding = bt.funding_held(panel).to_numpy()
    dividends = terms.dividends.to_numpy()
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
        carry = _carry(o, c, funding[:, j], dividends[:, j], terms.borrow_rate[j] * terms.dt_years, start, end, side,
                       by_rule, ~by_rule & ~at_end & ~delisted & next_open, next_open) / entry_px
        parts.append(pd.DataFrame({"instrument": inst, "side": side.astype(np.int64), "entry_time": idx[entry_at],
                                   "exit_time": idx[exit_at], "entry_px": entry_px, "exit_px": exit_px,
                                   "bars": end - start + 1, "size": np.abs(weights[inst].to_numpy()[start]),
                                   "gross_return": gross, "net_return": gross - paid_in - paid_out - carry,
                                   "exit_reason": reason}))
    if not parts:
        return pd.DataFrame([], columns=COLUMNS)
    return pd.concat(parts, ignore_index=True)[COLUMNS]


def _carry(opn, close, funding, dividends, borrow, start, end, side, inside, held_into_next, next_open) -> np.ndarray:
    """What each trade (bars `start` to `end`, `side`) paid in carry, in the instrument's price units (over its entry
    price, per unit of the money it was entered with): its funding (`funding`: per unit of a position's value at a
    bar's open, the bar held whole, `backtest.funding_held`) and a short's `borrow` (a year's rate times each bar's
    years) on its value at each bar's open, less the dividends (`dividends`: over the close before) of the ex-dates it
    was held into: from the bar after its entry (a next_close fill holds its entry bar's close), to the bar after its
    last for one sold at that bar's open (`held_into_next`). An exit inside its last bar (`inside`) held half of it."""
    before = np.r_[np.nan, close[:-1]]
    at_open = np.where(np.isnan(opn), before, opn)          # a bar without an open starts at the close before
    total = lambda a: np.r_[0.0, np.cumsum(np.nan_to_num(a))]     # noqa: E731
    fund, owe, paid = total(funding * at_open), total(borrow * at_open), total(dividends * before)
    last = len(close) - 1
    half = np.where(inside, 0.5, 0.0)
    held_fund = fund[end + 1] - fund[start] - half * np.nan_to_num(funding * at_open)[end]
    held_owe = np.where(side < 0, owe[end + 1] - owe[start] - half * np.nan_to_num(borrow * at_open)[end], 0.0)
    lo = start + 1 if next_open else start
    hi = np.where(held_into_next, np.minimum(end + 1, last), end)
    got = np.where(hi >= lo, paid[hi + 1] - paid[np.minimum(lo, hi + 1)], 0.0)
    return side * held_fund + held_owe - side * got


def made(trades: pd.DataFrame) -> pd.Series:
    """What each trade made for the account, as a share of the equity: its net return times the share it was entered
    with (`size`; a ledger kept without it counts each trade whole)."""
    size = trades["size"] if "size" in trades.columns else pd.Series(np.nan, index=trades.index)
    return trades["net_return"] * size.fillna(1.0)


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
