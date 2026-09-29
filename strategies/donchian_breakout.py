"""Channel breakout, the Turtles' system (C. Faith, "Way of the Turtle", 2007): long a close above the highest high of
the last n_in bars (20 in their first system, 55 in their second), out on a close below the lowest low of the last
n_out (10, 20), with a stop 2 average true ranges from the entry (their 2N). The stop is in ATRs, as theirs, so that it
sits as far from the price on an hour of a currency pair as on a day of a coin. The Turtles traded the mirror short as
well; here long only: that short side, taken by the walk-forward on a list's past, did worse out of sample than the
long side alone on most lists and timeframes."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab.strategy import hold_between, rule


@rule(grid={"n_in": [20, 55, 100], "n_out": [10, 20, 40], "stop_atr": [None, 2.0, 3.0], "slots": [None, 5, 10, 20]})
def donchian_breakout(bars, n_in, n_out):
    """Long from a close above the n_in-bar high until a close below the n_out-bar low."""
    entry, exit_ = ind.donchian(bars, n_in), ind.donchian(bars, n_out)
    return hold_between(bars.close > entry["upper"], bars.close < exit_["lower"])
