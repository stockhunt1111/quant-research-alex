"""QuantifiedStrategies, "Internal Bar Strength": buy a close in the bottom fifth of the day's range, sell a close in
its top fifth (`strategies.ibs_reversion` holds one bar instead). The source's
fifths are the middle of the grid, the walk-forward choosing on each list's past; long only,
as the other mean reversions that hold to their exit: their short side did worse out of sample than the long side
alone on most lists and timeframes."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab.strategy import hold_between, rule


@rule(grid={"buy_below": [0.15, 0.2, 0.25], "sell_above": [0.75, 0.8, 0.85], "slots": [None, 5, 10, 20]})
def ibs(bars, buy_below, sell_above):
    """Long from a close near the bar's low (IBS < buy_below) until a close near its high (IBS > sell_above)."""
    v = ind.ibs(bars)
    return hold_between(v < buy_below, v > sell_above)
