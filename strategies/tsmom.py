"""T. Moskowitz, Y. H. Ooi and L. H. Pedersen, "Time Series Momentum" (2012): each month, long an instrument whose return
over the past months is positive, short one whose return is negative (long only: flat instead), held to the next
month; the source's lookback is twelve months. The source sizes each position to a 40% volatility with leverage;
here each instrument gets its equal share of the capital, as under every rule, and nothing is levered."""
from __future__ import annotations

import numpy as np

from strategy_lab.strategy import bars_in, rebalanced, rule


@rule(grid={"months": [3, 6, 12, 18, 24], "long_only": [True, False]})
def tsmom(bars, months, long_only):
    """The sign of the trailing `months`-month return, decided at each month's turn (long only: its positive side)."""
    side = np.sign(bars.close / bars.close.shift(bars_in(bars, months=months)) - 1.0)
    return rebalanced(side.clip(lower=0.0) if long_only else side, "M")
