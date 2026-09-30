"""The market's trend as a switch for a whole list: hold every name while the market's index (SPY for stocks and ETFs,
bitcoin for coins) closes above its n-day simple moving average, and none of them while it is below. This is the
moving-average filter of W. Brock, J. Lakonishok & B. LeBaron (1992) and M. Faber (2007), put on the market's index
instead of each name's own price. A list's names rise and fall with their market, and the index gives a cleaner read
of the market's trend than each name's own price does. On coins, bitcoin is also the collateral and the quote asset of
the others. The rule reads the index's daily closes whatever the timeframe run (a day is a session of SPY, a day of
bitcoin), and a day's decision holds from that day's close on.

Three sizings are in the grid, chosen on each list's past by the walk-forward:
* `grade` sizes the position by how far the index is above its average: in proportion to that distance, and whole
  from `grade` above it on (0: whole from the first close above). A close barely above its average is a weaker read of
  the trend than one well above it;
* `decay` shrinks the position as the trend ages: full size for its first 90 days, then less day by day, down to 30%
  at 270 days and held there. It tests the view that a trend held for many months has drawn in leveraged buyers whose
  exit makes its end sharper;
* `vol_parity` holds each name at min(1, the index's volatility / its own) over the last 30 days of daily returns, so
  a name that swings more than its market carries no more risk than the market does.

With `long_only` off the rule also sells the list short while the index is below its average, graded, decayed and
sized the same way. The position is a holding kept while the market's trend lasts (`exposure`): the list's leavers are
sold at the re-pick and its newcomers bought at once. Written as its sources describe it, with parameters on a small
grid: here to be measured, not trusted."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_lab import market
from strategy_lab.strategy import rule

FULL_DAYS, FLOOR_DAYS, FLOOR = 90, 270, 0.3     # decay: full size to 90 days into a trend, 30% from 270 days on
VOL_DAYS = 30                                   # vol_parity: the volatilities over the last 30 days


def age(side: pd.Series) -> pd.Series:
    """How many days in a row a daily series has held its side, this day included (0 while on neither side)."""
    run = (side != side.shift()).cumsum()
    return (side.groupby(run).cumcount() + 1).where(side != 0, 0)


@rule(grid={"n_days": [50, 100, 200], "grade": [0.0, 0.05, 0.10, 0.20], "decay": [False, True],
            "vol_parity": [False, True], "long_only": [True, False]}, exposure=True, market=True)
def market_regime(bars, n_days, grade, decay, vol_parity, long_only):
    """Long every name while the market's index is above its n_days-day average (short below it, unless long only),
    in proportion to its distance from the average up to `grade` (0: whole), shrunk as that trend ages (`decay`) and
    sized to the index's volatility (`vol_parity`)."""
    index = market.index_closes(bars)
    distance = market.above_average(index, n_days)
    side = np.sign(distance).fillna(0.0)
    size = side if grade == 0 else side * (distance.abs() / grade).clip(upper=1.0).fillna(0.0)
    if long_only:
        size = size.clip(lower=0.0)
    if decay:
        run = age(np.sign(size))
        size = size * (1 - (1 - FLOOR) * (run - FULL_DAYS) / (FLOOR_DAYS - FULL_DAYS)).clip(FLOOR, 1.0)
    position = market.known_at(size, bars.index)
    if vol_parity:
        own = market.own_day_closes(bars)
        vol_index = market.known_at(np.log(index).diff().rolling(VOL_DAYS).std(), bars.index)
        vol_own = market.known_at(np.log(own).diff().rolling(VOL_DAYS).std(), bars.index)
        position = position * (vol_index / vol_own).clip(upper=1.0)
    return position.fillna(0.0)
