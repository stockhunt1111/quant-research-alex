"""A trend trade on each name, taken only while its market trends. Long a name while the market's index (SPY for stocks
and ETFs, bitcoin for coins) closes above its n-day simple moving average; out of the name on its first close below
its own e-day exponential average, and not back until it closes above its highest close of the h days before. A name
that the market's turn put out, not its own break, is bought again as soon as the market turns back up. Two textbook
pieces are put together here: the moving-average filter of W. Brock, J. Lakonishok & B. LeBaron (1992) and M. Faber
(2007) on the market's index, and a swing trader's trailing exit on a short exponential average. K. Kaminski & A. Lo
("When Do Stop-Loss Rules Stop Losses?", 2014) show why such an exit pays where returns switch between regimes and not
on a random walk. The market's side is read on the index's daily closes, whatever the timeframe run; the averages and
the highest close are on the name's own bars, their spans in days as the source gives them.

With `long_only` off the rule also sells a name short while the index is below its average. It covers on the name's
first close above its e-day average and is not short again until a close below its lowest close of the h days
before. Written as its sources describe it, with parameters on a small grid: here to be measured, not trusted."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from strategy_lab import indicators as ind
from strategy_lab import market
from strategy_lab.strategy import bars_in, rule, with_short


@njit(cache=True)
def trail(close, prior_best, average, market_on):
    """1.0 on the bars a name is held, 0.0 elsewhere. Out when the market turns (still armed) or on a close below its
    average (disarmed); armed again by a close above its prior best close, on any bar; in on a bar where the market is
    on and the name armed. Armed from the start: the market's first turn on buys every name."""
    out = np.zeros(len(close))
    held, armed = False, True
    for t in range(len(close)):
        if held:
            if not market_on[t]:
                held = False
            elif close[t] < average[t]:
                held, armed = False, False
        if not held and not armed and close[t] > prior_best[t]:
            armed = True
        if not held and armed and market_on[t]:
            held = True
        out[t] = 1.0 if held else 0.0
    return out


@rule(grid={"n_days": [50, 100, 200], "ema_days": [5, 10, 20], "high_days": [10, 20, 40], "long_only": [True, False],
            "slots": [None, 5, 10, 20]}, market=True)
def regime_ema_trail(bars, n_days, ema_days, high_days, long_only):
    """Long a name while the market's index is above its n_days-day average, out on a close below the name's
    ema_days-day average until a close above its high_days-day highest close (the mirror short below the index's
    average, unless long only)."""
    side = market.known_at(market.trend(market.index_closes(bars), n_days), bars.index).fillna(0.0).to_numpy()
    close = bars.close
    average = ind.ema(close, bars_in(bars, days=ema_days)).to_numpy()
    n = bars_in(bars, days=high_days)
    prior = close.shift(1).rolling(n, min_periods=n)
    c = close.to_numpy()
    long = trail(c, prior.max().to_numpy(), average, side > 0)
    short = trail(-c, -prior.min().to_numpy(), -average, side < 0)
    return with_short(pd.Series(long, index=bars.index), pd.Series(short, index=bars.index), long_only)
