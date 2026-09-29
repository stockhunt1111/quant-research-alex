"""Funding carry on perpetual futures, ported from the previous project: a dollar-neutral long/short book with gross
exposure 1 (half long, half short), rebuilt every `rebalance` days (at the turn of each such span) and held in
between. The ranking uses only the instruments in the universe at that bar."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_lab.data.bars import load_funding
from strategy_lab.strategy import long_short, panel, rebalanced


def trailing_funding(settlements: pd.Series, at: pd.DatetimeIndex, days: int) -> pd.Series:
    """Funding paid per day over the `days` before each time in `at`, from settlements at or before that time."""
    cs = settlements.sort_index().cumsum()
    def asof(times):
        return cs.reindex(cs.index.union(times)).ffill().reindex(times).fillna(0.0).to_numpy()
    total = asof(at) - asof(at - pd.Timedelta(days=days))
    started = at - pd.Timedelta(days=days) >= settlements.index.min()
    return pd.Series(np.where(started, total / days, np.nan), index=at)


@panel(grid={"lookback_days": [7, 14, 30], "frac": [0.1, 0.2, 0.3], "rebalance": ["1D", "3D", "7D"]})
def funding_carry(p, lookback_days, frac, rebalance, live):
    """Perps: short the richest trailing funding (collect it), long the cheapest. Funding known at each bar close."""
    carry = pd.DataFrame(np.nan, index=p.index, columns=p.ids)
    for i in p.ids:
        f = load_funding(i)
        if f is not None and len(f):
            carry[i] = trailing_funding(f, p.index, lookback_days).to_numpy()
    return rebalanced(long_short(-carry, live, frac), rebalance)
