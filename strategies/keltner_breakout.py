"""Keltner channel breakout, a well-known published rule: long a close above the upper channel (EMA + k x ATR), exit
below the EMA. Its mirror short (a close below the lower channel, exit above the EMA) is left out: taken by the
walk-forward on a list's past, it did worse out of sample than the long side alone on most lists and timeframes.
Written as it is usually described, parameters fixed a priori or on a small grid: here to be measured, not trusted."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab.strategy import hold_between, rule


@rule(grid={"n": [10, 20, 50], "k": [1.5, 2.0, 2.5], "slots": [None, 5, 10, 20]})
def keltner_breakout(bars, n, k):
    """Long from a close above EMA + k x ATR until a close below the EMA."""
    mid = ind.ema(bars.close, n)
    return hold_between(bars.close > mid + k * ind.atr(bars.high, bars.low, bars.close, n), bars.close < mid)
