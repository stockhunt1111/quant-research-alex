"""Bollinger band reversion (strategies.bollinger_reversion) with a model deciding which of its trades to take: the rule
says when, a model trained on the rule's own past trades says whether (strategy_lab.trade_model, `take`). Machine
learning that only grades a rule's trades, not its signals: this is the filter form
of that idea, strategies.bollinger_ml_sized the sizing one. On a list the model learns from the trades of every name in
it; on one instrument alone, from that instrument's."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab import trade_model
from strategy_lab.strategy import hold_between, rule


@rule(grid={"n": [20], "k": [1.75, 2.0, 2.25], "slots": [None, 5, 10, 20], "threshold": trade_model.THRESHOLDS},
      grade=trade_model.take)
def bollinger_ml_filter(bars, n, k):
    """The rule's trades (long from a close below the lower band until a close above the middle band) that the model
    gives a probability above `threshold` of making money after costs."""
    bb = ind.bbands(bars.close, n, k)
    return hold_between(bars.close < bb["lower"], bars.close > bb["middle"])
