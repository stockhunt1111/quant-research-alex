"""Regime-aware (week-1 idea list: regime detection, where observing a trend two or three bars late is acceptable):
a trend entered once it has shown itself, not predicted. Long only: its short side, taken by the walk-forward on a
list's past, did worse out of sample than the long side alone on most lists and timeframes."""
from __future__ import annotations

import pandas as pd

from strategy_lab import indicators as ind
from strategy_lab.strategy import rule


def _after_runs(side: pd.Series, k_in: int, k_out: int) -> pd.Series:
    """1 from k_in bars in a row where `side` holds until k_out bars in a row where it does not."""
    runs = side.groupby((side != side.shift()).cumsum()).cumcount() + 1
    state = pd.Series(float("nan"), index=side.index)
    state[(side == 1) & (runs >= k_in)] = 1.0
    state[(side == 0) & (runs >= k_out)] = 0.0
    return state.ffill().fillna(0.0)


@rule(grid={"ema_n": [20, 50, 100], "k_in": [2, 3, 5], "k_out": [1, 2, 3]})
def late_entry_trend(bars, ema_n, k_in, k_out):
    """Observe, don't predict: enter after k_in consecutive closes above the EMA, leave after k_out not above."""
    return _after_runs((bars.close > ind.ema(bars.close, ema_n)).astype(float), k_in, k_out)
