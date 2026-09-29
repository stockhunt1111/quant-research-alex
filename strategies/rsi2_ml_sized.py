"""Connors' RSI(2) (strategies.rsi2_connors) with each trade sized by a model: the rule says when, a model trained on
the rule's own past trades says how much, 2p - 1 of the position for a probability p of making money, nothing below
one half (strategy_lab.trade_model, `size`). The sizing form of machine learning that only grades a rule's
trades; strategies.rsi2_ml_filter is the filter form. On a list the model learns from the trades of
every name in it; on one instrument alone, from that instrument's."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab import trade_model
from strategy_lab.strategy import bars_in, hold_between, rule


@rule(grid={"rsi_entry": [2.5, 5, 10, 15, 20], "slots": [None, 5, 10, 20]}, grade=trade_model.size)
def rsi2_ml_sized(bars, rsi_entry):
    """RSI(2)'s trades (long on RSI(2) < rsi_entry above the 200-day SMA until a close above the 5-bar SMA), each
    scaled by the model's confidence."""
    trend = ind.sma(bars.close, bars_in(bars, days=200))
    return hold_between((ind.rsi(bars.close, 2) < rsi_entry) & (bars.close > trend), bars.close > ind.sma(bars.close, 5))
