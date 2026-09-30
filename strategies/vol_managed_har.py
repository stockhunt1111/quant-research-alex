"""A. Moreira & T. Muir's volatility-managed position (strategies.vol_managed), sized on the variance of the months ahead
as a model forecasts it instead of the variance of the month just past. The model is F. Corsi's heterogeneous
autoregressive model of realized volatility (HAR, "A Simple Approximate Long-Memory Model of Realized Volatility",
2009): an ordinary least squares of the coming days' mean squared daily return on the day's squared return and the
week's and the month's means. It is refit on the walk-forward calendar (`strategy_lab.ml.refit_rows`) on the days whose
coming days had passed. Volatility clusters but also reverts to its mean, which the realized variance of one month does
not know: after a calm month it sizes up less, after a turbulent one it sizes down less. Once a month, hold
target_vol^2 / the forecast annualised variance, capped at the whole capital, on the instrument's daily returns
whatever the timeframe run.

Measured before it went in (walk-forward out of sample, against `vol_managed` on the same ten lists at 1d and 4h,
2026-09-30): a higher Sharpe on 8 of 10 and a shallower drawdown on 8, less return on 5 (it holds less). Offered as a
choice inside `vol_managed`'s grid instead, the walk-forward kept the realized variance and gained nothing, so it is a
rule of its own. Written as its sources describe it, with parameters on a small grid: here to be measured, not
trusted."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_lab import config, market, ml
from strategy_lab.strategy import rebalanced, rule

MIN_DAYS = 120                  # the fewest known days a fit learns from
FLOOR = 1e-10                   # a forecast variance never taken below this (a least squares can go under zero)


def har_variance(r: pd.Series, horizon: int) -> pd.Series:
    """The mean squared daily return over the next `horizon` days, forecast at each day's close by HAR, refit at each
    of `ml.refit_rows` on the days i whose next `horizon` days had passed by then (i + horizon <= the refit day);
    NaN before the first fit."""
    rv = r ** 2
    x = np.column_stack([np.ones(len(r)), rv, rv.rolling(5).mean(), rv.rolling(22).mean()])
    target = rv.rolling(horizon).mean().shift(-horizon).to_numpy()        # the mean of days i+1 .. i+horizon
    usable = ~np.isnan(x).any(axis=1)
    out = np.full(len(r), np.nan)
    starts = ml.refit_rows(r.index)
    for a, b in zip(starts, starts[1:] + [len(r)]):
        known = np.arange(max(0, a - horizon + 1))
        known = known[usable[known] & ~np.isnan(target[known])]
        if len(known) < MIN_DAYS:
            continue
        beta, *_ = np.linalg.lstsq(x[known], target[known], rcond=None)
        out[a:b] = np.maximum(x[a:b] @ beta, FLOOR)
    return pd.Series(out, index=r.index).where(usable)


@rule(grid={"horizon_months": [0.5, 1, 3], "target_vol": [0.10, 0.15, 0.20]}, exposure=True)
def vol_managed_har(bars, horizon_months, target_vol):
    """Long target_vol^2 / the annualised variance HAR forecasts for the next `horizon_months` months, capped at 100% of
    capital (no leverage), decided at each month's turn."""
    own = market.own_day_closes(bars)
    days = config.TRADING_DAYS[bars.attrs["asset_class"]]
    variance = har_variance(np.log(own).diff(), max(5, round(horizon_months * days / 12))) * days
    return rebalanced(market.known_at((target_vol ** 2 / variance).clip(upper=1.0), bars.index), "M").fillna(0.0)
