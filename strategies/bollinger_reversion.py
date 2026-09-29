"""Bollinger band reversion, a well-known published rule: buy a close below the lower band, exit at the middle band.
Its mirror short (a close above the upper band, covered at the middle band) is left out: taken by the walk-forward on
a list's past, it did worse out of sample than the long side alone on most lists and timeframes. Written as it is
usually described, parameters fixed a priori or on a small grid: here to be measured, not trusted."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab.strategy import hold_between, rule


@rule(grid={"n": [20], "k": [1.5, 1.75, 2.0, 2.25, 2.5], "slots": [None, 5, 10, 20]})
def bollinger_reversion(bars, n, k):
    """Long from a close below the lower band until a close above the middle band."""
    bb = ind.bbands(bars.close, n, k)
    return hold_between(bars.close < bb["lower"], bars.close > bb["middle"])
