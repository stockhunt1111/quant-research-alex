"""Regime-aware (week-1 idea list: regime detection): a trend traded only while volatility is calm. Long only: its
short side, taken by the walk-forward on a list's past, did worse out of sample than the long side alone on most lists
and timeframes."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab import regimes
from strategy_lab.strategy import rule


@rule(grid={"fast": [10, 20, 50], "slow": [100, 200, 300], "k": [3, 5, 10], "max_vol_state": [0, 1]})
def calm_trend(bars, fast, slow, k, max_vol_state):
    """Trend (long while the fast EMA is above the slow one) held only while volatility has been calm/normal for k bars
    in a row."""
    calm = regimes.confirmed(regimes.vol_state(bars.close), k) <= max_vol_state
    return (calm & (ind.ema(bars.close, fast) > ind.ema(bars.close, slow))).astype(float)
