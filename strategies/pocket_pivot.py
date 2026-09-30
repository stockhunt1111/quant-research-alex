"""G. Morales & C. Kacher's pocket pivot ("Trade Like an O'Neil Disciple", 2010), with the trailing exit a momentum
trader puts on it (Chat With Traders, episode 110). Buy an up close above the 10- and 50-day moving averages that
trades more volume than any down day of the ten days before it: buyers stepping in before the breakout. Sell on the
second close below the 20-day exponential average since the entry. Spans are in days, as the source gives them, so
they cover the same time on every timeframe. Where the vendor records no volume (currency pairs, the spot metals), the
bar's range stands in for it: an up bar wider than every down bar of the ten days, range expanding where volume would
show demand. Morales and Kacher's own sell rules (the 7-week and 50-day lines) are not used.

With `long_only` off the rule also sells the mirror short: a down close below both averages on more volume than any
up day of the ten days before, covered on the second close above the exponential average. Written as its sources
describe it, with parameters on a small grid: here to be measured, not trusted."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from strategy_lab import indicators as ind
from strategy_lab.strategy import bars_in, rule, with_short

FAST_DAYS, SLOW_DAYS = 10, 50       # the averages a pocket pivot closes above
CLOSES_OUT = 2                      # the closes below the exponential average since the entry that end the trade


@njit(cache=True)
def held(entry, below):
    """1.0 from a bar where `entry` holds until the CLOSES_OUT-th bar since it where `below` holds, 0.0 elsewhere."""
    out = np.zeros(len(entry))
    on, count = False, 0
    for t in range(len(entry)):
        if on and below[t]:
            count += 1
            if count >= CLOSES_OUT:
                on = False
        if not on and entry[t]:
            on, count = True, 0
        out[t] = 1.0 if on else 0.0
    return out


def side(bars: pd.DataFrame, close: pd.Series, high: pd.Series, low: pd.Series, lookback_days: int,
         ema_days: int) -> np.ndarray:
    """The long side on these prices (the short side is it on the prices negated, the high and the low swapped)."""
    fast, slow = bars_in(bars, days=FAST_DAYS), bars_in(bars, days=SLOW_DAYS)
    volume = bars.volume.fillna(0.0)
    recorded = volume.rolling(slow, min_periods=1).sum() > 0           # any volume on the bars up to each one
    demand = volume.where(recorded, high - low)
    before = close.shift(1)
    n = bars_in(bars, days=lookback_days)
    hurdle = demand.where(close < before, 0.0).shift(1).rolling(n, min_periods=n).max()
    entry = (close > before) & (close > ind.sma(close, fast)) & (close > ind.sma(close, slow)) & (demand > hurdle)
    below = close < ind.ema(close, bars_in(bars, days=ema_days))
    return held(entry.to_numpy(), below.to_numpy())


@rule(grid={"lookback_days": [5, 10, 20], "ema_days": [10, 20, 50], "long_only": [True, False],
            "slots": [None, 5, 10, 20]})
def pocket_pivot(bars, lookback_days, ema_days, long_only):
    """Long an up close above the 10- and 50-day averages on more volume than any down day of the last lookback_days,
    until the second close below the ema_days-day exponential average (the mirror short too, unless long only)."""
    long = side(bars, bars.close, bars.high, bars.low, lookback_days, ema_days)
    short = side(bars, -bars.close, -bars.low, -bars.high, lookback_days, ema_days)
    return with_short(pd.Series(long, index=bars.index), pd.Series(short, index=bars.index), long_only)
