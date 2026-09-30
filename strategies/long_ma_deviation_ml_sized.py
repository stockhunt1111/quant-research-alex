"""The long-average deviation (strategies.long_ma_deviation) with each trade sized by a model: the rule says when, a
model trained on the rule's own past trades says how much, 2p - 1 of the position for a probability p of making money,
nothing below one half (strategy_lab.trade_model, `size`). The sizing form of machine learning that only grades a
rule's trades; strategies.long_ma_deviation_ml_filter is the filter form. Long only, as the other rules graded so: the
model's features describe the market at a trade's first bar, not the trade's side, and a long and a short taken on the
same bars would teach it opposite outcomes. On a list the model learns from the trades of every name in it; on one
instrument alone, from that instrument's."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab import trade_model
from strategy_lab.strategy import hold_between, rule


@rule(grid={"n": [500, 1000, 2000], "gap": [0.015, 0.025, 0.04], "slots": [None, 5, 10, 20]}, grade=trade_model.size)
def long_ma_deviation_ml_sized(bars, n, gap):
    """The rule's trades (long a close more than `gap` below the n-bar average until a close back at it), each scaled
    by the model's confidence."""
    average = ind.sma(bars.close, n)
    return hold_between(bars.close < average * (1 - gap), bars.close >= average)
