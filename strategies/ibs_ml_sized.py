"""IBS (strategies.ibs) with each trade sized by a model: IBS says when, a model trained on IBS's own past trades says
how much, 2p - 1 of the position for a probability p of making money, nothing below one half
(strategy_lab.trade_model, `size`). The sizing form of machine learning that only grades a rule's
trades; strategies.ibs_ml_filter is the filter form. On a list the model learns from the trades of every name in
it; on one instrument alone, from that instrument's."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab import trade_model
from strategy_lab.strategy import hold_between, rule


@rule(grid={"buy_below": [0.15, 0.2, 0.25], "sell_above": [0.75, 0.8, 0.85], "slots": [None, 5, 10, 20]},
      grade=trade_model.size)
def ibs_ml_sized(bars, buy_below, sell_above):
    """IBS's trades (long from IBS < buy_below until IBS > sell_above), each scaled by the model's confidence."""
    v = ind.ibs(bars)
    return hold_between(v < buy_below, v > sell_above)
