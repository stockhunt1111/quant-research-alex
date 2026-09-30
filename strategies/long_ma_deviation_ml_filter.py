"""The long-average deviation (strategies.long_ma_deviation) with a model deciding which of its trades to take: the
rule says when, a model trained on the rule's own past trades says whether (strategy_lab.trade_model, `take`). Machine
learning that only grades a rule's trades, not its signals: this is the filter form of that idea,
strategies.long_ma_deviation_ml_sized the sizing one. Long only, as the other rules graded so: the model's features
describe the market at a trade's first bar, not the trade's side, and a long and a short taken on the same bars would
teach it opposite outcomes. On a list the model learns from the trades of every name in it; on one instrument alone,
from that instrument's."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab import trade_model
from strategy_lab.strategy import hold_between, rule


@rule(grid={"n": [500, 1000, 2000], "gap": [0.015, 0.025, 0.04], "slots": [None, 5, 10, 20],
            "threshold": trade_model.THRESHOLDS}, grade=trade_model.take)
def long_ma_deviation_ml_filter(bars, n, gap):
    """The rule's trades (long a close more than `gap` below the n-bar average until a close back at it) that the model
    gives a probability above `threshold` of making money after costs."""
    average = ind.sma(bars.close, n)
    return hold_between(bars.close < average * (1 - gap), bars.close >= average)
