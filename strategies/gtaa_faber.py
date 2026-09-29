"""M. Faber, "A Quantitative Approach to Tactical Asset Allocation" (2007): hold each asset while its price at the
month's end is above its 10-month simple moving average, the average of the last ten month-end prices, and hold cash
otherwise; the signal is looked at once a month and the price's moves in between are ignored. Equal capital. The
source's 10 months are the middle of the grid, the walk-forward choosing on each list's past."""
from __future__ import annotations

from strategy_lab.strategy import period_starts, rule


@rule(grid={"months": [6, 8, 10, 12, 14]})
def gtaa_faber(bars, months):
    """Hold from a month's turn at which the close is above the average of the last `months` month-turn closes until a
    turn at which it is not (a month's turn: its first close, `period_starts`)."""
    turns = bars.close[period_starts(bars.index, "M")]
    above = (turns > turns.rolling(months).mean()).astype(float)
    return above.reindex(bars.index).ffill().fillna(0.0)
