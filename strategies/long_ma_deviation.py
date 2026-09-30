"""A long average as a price's anchor, and a fall well below it as an overreaction to buy (a futures trader's rule, Will
Gogolak on Chat With Traders, episode 276: the average of the last thousand bars, bought 2-3% below it). Long a close
more than `gap` under the average of the last n bars; out once a close is back at the average. On daily bars a thousand
bars are four years, and the rule comes close to W. De Bondt and R. Thaler's long-term reversal (1985); on hourly bars
they are six weeks of a coin, and the rule buys a short-term overreaction. n counts bars, as the source does. There is
no stop: a mean reversion's stop sells the dip it bought.

With `long_only` off the rule also sells short a close more than `gap` over the average, covered once a close is back
at it. Written as its source describes it, with parameters on a small grid: here to be measured, not trusted."""
from __future__ import annotations

from strategy_lab import indicators as ind
from strategy_lab.strategy import hold_between, rule, with_short


@rule(grid={"n": [500, 1000, 2000], "gap": [0.015, 0.025, 0.04], "long_only": [True, False],
            "slots": [None, 5, 10, 20]})
def long_ma_deviation(bars, n, gap, long_only):
    """Long a close more than `gap` below the n-bar average until a close back at it (the mirror short above it too,
    unless long only)."""
    average = ind.sma(bars.close, n)
    long = hold_between(bars.close < average * (1 - gap), bars.close >= average)
    short = hold_between(bars.close > average * (1 + gap), bars.close <= average)
    return with_short(long, short, long_only)
