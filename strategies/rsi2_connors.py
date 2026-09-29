"""L. Connors & C. Alvarez, "Short Term Trading Strategies That Work" (2008): buy a 2-period RSI dip while the close is
above its 200-day simple moving average and sell on the first close above the 5-period simple moving average. Connors
sells short an RSI(2) surge below the 200-day average as well; here long only: that short side, taken by the
walk-forward on a list's past, did worse out of sample than the long side alone on most lists and timeframes. Connors
found a dip below 5 better than below 10, and stops hurting (none here). The 200-day average is the trend over 200
trading days on every timeframe; the RSI and the exit's average count the timeframe's own bars. The long side written
as the source describes it, parameters fixed a priori or on a small grid: here to be measured, not trusted."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab.strategy import bars_in, hold_between, rule


@rule(grid={"rsi_entry": [2.5, 5, 10, 15, 20], "slots": [None, 5, 10, 20]})
def rsi2_connors(bars, rsi_entry):
    """Long on RSI(2) < rsi_entry above the 200-day SMA until a close above the 5-bar SMA."""
    trend = ind.sma(bars.close, bars_in(bars, days=200))
    return hold_between((ind.rsi(bars.close, 2) < rsi_entry) & (bars.close > trend), bars.close > ind.sma(bars.close, 5))
