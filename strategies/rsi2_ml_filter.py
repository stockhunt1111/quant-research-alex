"""Connors' RSI(2) (strategies.rsi2_connors) with a model deciding which of its trades to take: the rule says when, a
model trained on the rule's own past trades says whether (strategy_lab.trade_model, `take`). Machine
learning that only grades a rule's trades, not its signals: this is the filter form of
that idea, strategies.rsi2_ml_sized the sizing one. On a list the model learns from the trades of every name in it; on
one instrument alone, from that instrument's."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab import trade_model
from strategy_lab.strategy import bars_in, hold_between, rule


@rule(grid={"rsi_entry": [5, 10, 15], "slots": [None, 5, 10, 20], "threshold": trade_model.THRESHOLDS},
      grade=trade_model.take)
def rsi2_ml_filter(bars, rsi_entry):
    """RSI(2)'s trades (long on RSI(2) < rsi_entry above the 200-day SMA until a close above the 5-bar SMA) that the
    model gives a probability above `threshold` of making money after costs."""
    trend = ind.sma(bars.close, bars_in(bars, days=200))
    return hold_between((ind.rsi(bars.close, 2) < rsi_entry) & (bars.close > trend), bars.close > ind.sma(bars.close, 5))
