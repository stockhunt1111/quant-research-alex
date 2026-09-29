"""Relative strength rotation: hold the top-k of a universe by trailing 6-month return, rebuilt once a month. Written as
the rule is usually described, parameters fixed a priori or on a small grid: here to be measured, not trusted."""
from __future__ import annotations

from strategy_lab.strategy import bars_in, panel, rebalanced


@panel(grid={"months": [6], "top_k": [1, 2, 3, 4, 5], "rebalance": ["M"]})
def sector_rotation(p, months, top_k, rebalance, live):
    """Hold the top_k assets by trailing `months`-month return, equal capital, decided at each month's turn (always
    invested)."""
    ret = (p.close / p.close.shift(bars_in(p, months=months)) - 1.0).where(live)
    rank = ret.rank(axis=1, ascending=False, method="first")
    pick = (rank <= top_k).astype(float) / top_k
    return rebalanced(pick, rebalance)
