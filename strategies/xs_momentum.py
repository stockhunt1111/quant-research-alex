"""Cross-sectional momentum, ported from the previous project: a dollar-neutral long/short book with gross exposure 1
(half long, half short), rebuilt at each week's turn and held in between. The ranking uses only the instruments in
the universe at that bar."""
from __future__ import annotations

import numpy as np

from strategy_lab.strategy import as_of, calendar_months, long_short, panel, rebalanced


@panel(grid={"months": [1, 3, 6], "frac": [0.1, 0.2, 0.3], "rebalance": ["W"]})
def xs_momentum(p, months, frac, rebalance, live):
    """Long the strongest, short the weakest trailing `months`-month returns, each over its volatility over those months
    (the spread of its bars' returns times the square root of their count)."""
    span = calendar_months(months)
    r = p.close.pct_change()
    score = (p.close / as_of(p.close, months=months) - 1.0) / (r.rolling(span).std() * np.sqrt(r.rolling(span).count()))
    return rebalanced(long_short(score, live, frac), rebalance)
