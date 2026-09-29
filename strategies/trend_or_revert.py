"""Regime-switched (week-1 idea list: strategies switched per regime): trend, mean reversion or flat by regime. Long
only: its short side (a confirmed down-trend followed short, RSI(2) surges sold while the market ranges), taken by the
walk-forward on a list's past, did worse out of sample than the long side alone on most lists and timeframes."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab import regimes
from strategy_lab.strategy import hold_between, rule


@rule(grid={"adx_threshold": [20, 25, 30], "k": [2, 3, 5], "rsi_entry": [20, 30, 40]})
def trend_or_revert(bars, adx_threshold, k, rsi_entry):
    """Switch by regime: follow an up-trend when a confirmed trend exists, buy RSI(2) dips when the market ranges and
    hold them until the first close above the 5-bar SMA (Connors' exit: the bounce the dip bets on), stay flat in a
    confirmed down-trend."""
    state = regimes.confirmed(regimes.trend_state(bars, adx_threshold=adx_threshold), k)
    fast, slow = ind.ema(bars.close, 20), ind.ema(bars.close, 50)
    r, mean = ind.rsi(bars.close, 2), ind.sma(bars.close, 5)
    # a dip is bought only while the market ranges: one bought in a trend would become a position at the switch
    dips = hold_between((r < rsi_entry) & (state == 0.0), bars.close > mean)
    return regimes.switch(state, {1.0: (fast > slow).astype(float), 0.0: dips})
