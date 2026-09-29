"""Reference rule: simple and well known, it exercises the whole pipeline."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab.strategy import rule, with_short


@rule(grid={"entry": [0.1, 0.15, 0.2, 0.25, 0.3], "slots": [None, 5, 10, 20], "long_only": [True, False]})
def ibs_reversion(bars, entry, long_only):
    """Long for one bar after a close near the bar's low (internal bar strength below `entry`); unless long only,
    short for one bar after a close near its high (above 1 - `entry`)."""
    v = ind.ibs(bars)
    return with_short((v < entry).astype(float), (v > 1.0 - entry).astype(float), long_only)
