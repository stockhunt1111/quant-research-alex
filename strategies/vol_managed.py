"""A. Moreira & T. Muir, "Volatility-Managed Portfolios" (2017), without leverage: once a month, hold c divided by the
realised variance of the returns just before (the source's past month; here also half a month and a quarter), capped
at the whole capital: less after the market has swung, all of it while it is calm. c is the square of a target
volatility, so the position is (target / realised volatility)^2. Written as the source describes it, parameters fixed
a priori or on a small grid: here to be measured, not trusted."""
from __future__ import annotations

import numpy as np

from strategy_lab.strategy import bars_in, rebalanced, rule


@rule(grid={"vol_months": [0.5, 1, 3], "target_vol": [0.10, 0.15, 0.20]}, exposure=True)
def vol_managed(bars, vol_months, target_vol):
    """Long target_vol^2 / the annualised realised variance of the last `vol_months` months, capped at 100% of capital
    (no leverage), decided at each month's turn."""
    r = np.log(bars.close).diff()
    variance = (r ** 2).rolling(bars_in(bars, months=vol_months)).mean() * bars_in(bars, months=12)
    return rebalanced((target_vol ** 2 / variance).clip(upper=1.0), "M")
