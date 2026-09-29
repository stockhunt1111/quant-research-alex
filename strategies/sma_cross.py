"""Reference rule: simple and well known, it exercises the whole pipeline."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab.strategy import rule, with_short


@rule(grid={"fast": [10, 20, 50], "slow": [100, 200, 300], "long_only": [True, False]})
def sma_cross(bars, fast, slow, long_only):
    """Long while the fast simple moving average is above the slow one; short while it is below, unless long only."""
    fast_ma, slow_ma = ind.sma(bars.close, fast), ind.sma(bars.close, slow)
    return with_short((fast_ma > slow_ma).astype(float), (fast_ma < slow_ma).astype(float), long_only)
