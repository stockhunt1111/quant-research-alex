"""Channel breakout with a trailing stop, ported from the previous project's breakout family.

Entry: close above the prior n-bar high. The position is held until the engine's trailing stop, a chandelier exit (C.
LeBeau: a multiple of the average true range below the best price since entry, checked inside every bar), or a close
below the prior n_out-bar low. Unless long only, the mirror too: a close below the prior n-bar low goes short, held until
the trailing stop (above the lowest price since entry) or a close above the prior n_out-bar high. The trail is in
ATRs, so that it gives an hour of a currency pair and a day of a coin the same room relative to their swings. Its
level is re-set at each bar's close, as LeBeau's, or at each minute's (a bot's stop following the price through the
bar): the walk-forward chooses, as it chooses the room.
"""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab.strategy import hold_between, rule, with_short


@rule(grid={"n_in": [20, 55, 100], "n_out": [10, 20, 50], "trail_atr": [2.0, 3.0, 4.0], "trail_every": ["bar", "minute"],
            "slots": [None, 5, 10, 20], "long_only": [True, False]})
def breakout_trail(bars, n_in, n_out, long_only):
    """Long from a close above the n_in-bar high until a close below the n_out-bar low (plus the trailing stop); short
    in the mirror, unless long only."""
    entry, exit_ = ind.donchian(bars, n_in), ind.donchian(bars, n_out)
    return with_short(hold_between(bars.close > entry["upper"], bars.close < exit_["lower"]),
                      hold_between(bars.close < entry["lower"], bars.close > exit_["upper"]), long_only)
