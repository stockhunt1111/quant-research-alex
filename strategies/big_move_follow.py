"""Big-move capture across a wide universe (week-1 idea list): few trades, ride the large moves.

An event is a bar whose return is at least `z` times the instrument's trailing volatility, on volume at least
`vol_mult` times its trailing median; an instrument its vendor records no traded volume of (FX pairs and spot metals:
their volume is zero on every bar) is judged on the move alone, or it would never trade. The book follows an up move
for `hold` bars (a down move followed short, the walk-forward's choice on a list's past, did worse out of sample
wherever it was taken, once losing more than the account on a coin's surge), with the engine's trailing stop (3
average true ranges from the best price since entry: the same room relative to the swings on every timeframe; its
level re-set at each bar's close or at each minute's, as the walk-forward chooses); capital is split across at most
`max_positions` concurrent events: an event takes a free seat on its first bar, the strongest of those starting on
one bar first (`strategy.slotted`), and keeps it to its end or until its name leaves the list; one that finds every
seat taken is not traded. Each event is a position of its own (`panel(book=False)`): bought at its share when it
starts and held as units until it ends, the others left as they are, where a book rebalanced whole would sell the
running moves down to their shares at every new event. Until 2026-09-29 the events were ranked again on every bar: a
stronger new one sold a running one, an event left waiting was bought later into its move, and a name gone from the
list kept a seat it could not use.
"""
from __future__ import annotations

import numpy as np

from strategy_lab.strategy import panel, slotted


@panel(grid={"z": [2.5, 3.0, 3.5], "vol_mult": [1.5, 2.0, 3.0], "hold": [5, 10, 20], "max_positions": [5],
             "trail_atr": [3.0], "trail_every": ["bar", "minute"]}, book=False)
def big_move_follow(p, z, vol_mult, hold, max_positions, live):
    """Follow each big up move for `hold` bars: a bar whose return is at least `z` of its trailing standard deviations,
    on at least `vol_mult` times its trailing median volume where the instrument has one recorded; at most
    `max_positions` at once, each 1/max_positions of the capital, a seat taken on an event's first bar (the strongest
    of those starting together first) and kept to its end, with a trailing stop."""
    r = np.log(p.close).diff()
    sigma = r.rolling(60, min_periods=30).std().shift(1).replace(0.0, np.nan)    # a flat stretch is no yardstick
    zscore = r / sigma
    recorded = (p.volume > 0).cummax()          # from an instrument's first bar with a traded volume on
    vol_ok = (p.volume > vol_mult * p.volume.rolling(60, min_periods=30).median().shift(1)) | ~recorded
    score = zscore.where(vol_ok & live)
    score = score.where(score >= z)
    # hold each event `hold` bars (a new event of the name carries it on), while its name is in the list
    held = np.sign(score).ffill(limit=hold - 1).fillna(0.0).where(live, 0.0)
    strength = score.abs().ffill(limit=hold - 1).fillna(0.0)
    return slotted(held, max_positions, strength) / float(max_positions)
