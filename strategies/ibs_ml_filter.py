"""IBS (strategies.ibs) with a model deciding which of its trades to take: IBS says when, a model trained on IBS's own
past trades says whether (strategy_lab.trade_model, `take`). Machine learning that only grades a rule's
trades, not its signals: this is the filter form of that idea, strategies.ibs_ml_sized the sizing
one. On a list the model learns from the trades of every name in it; on one instrument alone, from that instrument's."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab import trade_model
from strategy_lab.strategy import hold_between, rule


@rule(grid={"buy_below": [0.15, 0.2, 0.25], "sell_above": [0.75, 0.8, 0.85], "slots": [None, 5, 10, 20],
            "threshold": trade_model.THRESHOLDS}, grade=trade_model.take)
def ibs_ml_filter(bars, buy_below, sell_above):
    """IBS's trades (long from IBS < buy_below until IBS > sell_above) that the model gives a probability above
    `threshold` of making money after costs."""
    v = ind.ibs(bars)
    return hold_between(v < buy_below, v > sell_above)
