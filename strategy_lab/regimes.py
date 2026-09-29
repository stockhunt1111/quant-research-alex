"""Market regimes from price alone, causal at every bar.

Volatility is predictable enough to label; direction is not, so direction is OBSERVED, not predicted: a state
counts only after it has held for k consecutive bars (`confirmed`), trading a later but surer entry. A regime is
computed once for the signal configurations of a grid that ask for it on the same bars (`strategy_lab.shared`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_lab import indicators as ind
from strategy_lab.shared import shared


@shared
def vol_state(close: pd.Series, lookback: int = 20, ref: int = 250) -> pd.Series:
    """0 = calm, 1 = normal, 2 = stressed: where today's realised vol sits within its own trailing `ref` bars."""
    rv = ind.realized_vol(close, lookback)
    pct = rv.rolling(ref, min_periods=ref // 2).rank(pct=True)
    out = pd.Series(np.select([pct <= 1 / 3, pct <= 2 / 3], [0, 1], default=2), index=close.index, dtype=float)
    return out.where(pct.notna())


@shared
def trend_state(bars: pd.DataFrame, adx_n: int = 14, adx_threshold: float = 25.0, ema_n: int = 50) -> pd.Series:
    """+1 trending up, -1 trending down, 0 ranging: ADX says whether there is a trend, the EMA slope which way."""
    a = ind.adx(bars["high"], bars["low"], bars["close"], adx_n)
    slope = ind.ema(bars["close"], ema_n).diff()
    out = pd.Series(0.0, index=bars.index)
    trending = a > adx_threshold
    out[trending & (slope > 0)] = 1.0
    out[trending & (slope < 0)] = -1.0
    return out.where(a.notna())


@shared
def confirmed(state: pd.Series, k: int) -> pd.Series:
    """A state takes effect only once it has held for k consecutive bars; until then the previous one stands."""
    s = state.copy()
    run = s.groupby((s != s.shift()).cumsum()).cumcount() + 1
    accepted = s.where(run >= k)
    return accepted.ffill()


def switch(state: pd.Series, by_state: dict[float, pd.Series], default: float = 0.0) -> pd.Series:
    """Position from the strategy assigned to the current state (e.g. {1: trend, 0: mean_reversion, -1: flat})."""
    out = pd.Series(default, index=state.index, dtype=float)
    for value, pos in by_state.items():
        mask = state == value
        out[mask] = pos.reindex(state.index)[mask]
    return out
