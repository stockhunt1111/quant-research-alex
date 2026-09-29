"""Trend following, ported from the previous project's trend family, where long-only did better than long-short on
crypto and equities, markets that drift up. Long only here as well: its short side, taken by the walk-forward on a
list's past, did worse out of sample than the long side alone on most lists and timeframes."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab.strategy import rule


@rule(grid={"fast": [10, 20, 50], "slow": [100, 200, 300]})
def ema_trend(bars, fast, slow):
    """Long while the fast EMA is above the slow EMA; flat otherwise."""
    return (ind.ema(bars.close, fast) > ind.ema(bars.close, slow)).astype(float)
