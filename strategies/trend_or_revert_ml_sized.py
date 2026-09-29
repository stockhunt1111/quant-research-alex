"""The regime switch (strategies.trend_or_revert) with each trade sized by a model: the switch says when, a model
trained on its own past trades says how much, 2p - 1 of the position for a probability p of making money, nothing
below one half (strategy_lab.trade_model, `size`). The sizing form of machine learning that only grades a rule's
trades; strategies.trend_or_revert_ml_filter is the filter form. On a list the model learns from the
trades of every name in it; on one instrument alone, from that instrument's."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab import regimes
from strategy_lab import trade_model
from strategy_lab.strategy import hold_between, rule


@rule(grid={"adx_threshold": [20, 25, 30], "k": [2, 3, 5], "rsi_entry": [20, 30, 40]}, grade=trade_model.size)
def trend_or_revert_ml_sized(bars, adx_threshold, k, rsi_entry):
    """The switch's trades (long in a confirmed up-trend while the 20-bar EMA is above the 50-bar one, long from an
    RSI(2) dip below rsi_entry to the first close above the 5-bar SMA while the market ranges, flat in a confirmed
    down-trend), each scaled by the model's confidence."""
    state = regimes.confirmed(regimes.trend_state(bars, adx_threshold=adx_threshold), k)
    fast, slow = ind.ema(bars.close, 20), ind.ema(bars.close, 50)
    r, mean = ind.rsi(bars.close, 2), ind.sma(bars.close, 5)
    # a dip is bought only while the market ranges: one bought in a trend would become a position at the switch
    dips = hold_between((r < rsi_entry) & (state == 0.0), bars.close > mean)
    return regimes.switch(state, {1.0: (fast > slow).astype(float), 0.0: dips})
