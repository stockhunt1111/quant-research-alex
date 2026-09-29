"""G. Antonacci, "Dual Momentum Investing" (2014): once a month hold the assets with the best 12-month return (relative
momentum), each only while that return beats Treasury bills over the same twelve months (absolute momentum); the
money of an asset that does not is kept in cash. Antonacci moves that money into bonds, an asset a list of stocks or
coins does not hold. Written as the source describes it, parameters fixed a priori or on a small grid: here to be
measured, not trusted."""
from __future__ import annotations

from strategy_lab.data import rates
from strategy_lab.strategy import bars_in, panel, rebalanced


@panel(grid={"months": [12], "top_k": [1, 2, 3, 4, 5], "rebalance": ["M"]})
def dual_momentum(p, months, top_k, rebalance, live):
    """Hold the top_k assets by trailing `months`-month return, each only if that return is above T-bills' over the
    same months; decided at each month's turn."""
    ret = (p.close / p.close.shift(bars_in(p, months=months)) - 1.0).where(live)
    rank = ret.rank(axis=1, ascending=False, method="first")
    pick = ((rank <= top_k) & ret.gt(rates.trailing_return(p.index, months), axis=0)).astype(float) / top_k
    return rebalanced(pick, rebalance)
