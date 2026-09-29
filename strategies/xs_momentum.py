"""Cross-sectional momentum, ported from the previous project: a dollar-neutral long/short book with gross exposure 1
(half long, half short), rebuilt at each week's turn and held in between. The ranking uses only the instruments in
the universe at that bar."""
from __future__ import annotations

import numpy as np

from strategy_lab.strategy import bars_in, long_short, panel, rebalanced


@panel(grid={"months": [1, 3, 6], "frac": [0.1, 0.2, 0.3], "rebalance": ["W"]})
def xs_momentum(p, months, frac, rebalance, live):
    """Long the strongest, short the weakest trailing `months`-month returns, each over its volatility."""
    n = bars_in(p, months=months)
    r = p.close.pct_change()
    score = (p.close / p.close.shift(n) - 1.0) / (r.rolling(n).std() * np.sqrt(n))
    return rebalanced(long_short(score, live, frac), rebalance)
