"""Bollinger band reversion (strategies.bollinger_reversion) with each trade sized by a model: the rule says when, a
model trained on the rule's own past trades says how much, 2p - 1 of the position for a probability p of making money,
nothing below one half (strategy_lab.trade_model, `size`). The sizing form of machine learning that only grades a
rule's trades; strategies.bollinger_ml_filter is the filter form. On a list the model learns from
the trades of every name in it; on one instrument alone, from that instrument's."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab import trade_model
from strategy_lab.strategy import hold_between, rule


@rule(grid={"n": [20], "k": [1.5, 1.75, 2.0, 2.25, 2.5], "slots": [None, 5, 10, 20]}, grade=trade_model.size)
def bollinger_ml_sized(bars, n, k):
    """The rule's trades (long from a close below the lower band until a close above the middle band), each scaled by
    the model's confidence."""
    bb = ind.bbands(bars.close, n, k)
    return hold_between(bars.close < bb["lower"], bars.close > bb["middle"])
