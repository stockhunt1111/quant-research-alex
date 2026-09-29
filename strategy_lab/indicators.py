"""Indicators on pandas Series, computed by TA-Lib (C implementation, 150+ functions, 61 candle patterns).

Only thin wrappers live here: TA-Lib does the arithmetic, so values match any other TA-Lib user. A few
price-structure helpers TA-Lib lacks (IBS, Donchian channel) are plain rolling operations.
Every function is causal: the value at bar t uses bars up to and including t. Each is computed once for the signal
configurations of a grid that ask for it on the same bars (`strategy_lab.shared`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from strategy_lab.shared import shared

_f = lambda s: np.asarray(s, dtype="float64")   # noqa: E731


def _wrap(values, like: pd.Series) -> pd.Series:
    return pd.Series(values, index=like.index)


@shared
def sma(close: pd.Series, n: int) -> pd.Series:
    return _wrap(talib.SMA(_f(close), timeperiod=n), close)


@shared
def ema(close: pd.Series, n: int) -> pd.Series:
    return _wrap(talib.EMA(_f(close), timeperiod=n), close)


@shared
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    return _wrap(talib.RSI(_f(close), timeperiod=n), close)


@shared
def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return _wrap(talib.ATR(_f(high), _f(low), _f(close), timeperiod=n), close)


@shared
def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return _wrap(talib.ADX(_f(high), _f(low), _f(close), timeperiod=n), close)


@shared
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    m, s, h = talib.MACD(_f(close), fastperiod=fast, slowperiod=slow, signalperiod=signal)
    return pd.DataFrame({"macd": m, "signal": s, "hist": h}, index=close.index)


@shared
def bbands(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    up, mid, lo = talib.BBANDS(_f(close), timeperiod=n, nbdevup=k, nbdevdn=k)
    return pd.DataFrame({"upper": up, "middle": mid, "lower": lo}, index=close.index)


@shared
def roc(close: pd.Series, n: int) -> pd.Series:
    return _wrap(talib.ROC(_f(close), timeperiod=n), close) / 100.0


@shared
def ibs(bars: pd.DataFrame) -> pd.Series:
    """Internal bar strength: where the close sits in the bar's range (0 = at the low, 1 = at the high)."""
    rng = (bars["high"] - bars["low"]).replace(0.0, np.nan)
    return (bars["close"] - bars["low"]) / rng


@shared
def donchian(bars: pd.DataFrame, n: int) -> pd.DataFrame:
    """Highest high / lowest low of the n bars BEFORE this one (so a close above `upper` is a breakout)."""
    return pd.DataFrame({"upper": bars["high"].rolling(n).max().shift(1),
                         "lower": bars["low"].rolling(n).min().shift(1)}, index=bars.index)


@shared
def realized_vol(close: pd.Series, n: int) -> pd.Series:
    return np.log(close).diff().rolling(n).std()


@shared
def call(name: str, *series: pd.Series, **params) -> pd.Series | pd.DataFrame:
    """Any TA-Lib function by name, e.g. call("CCI", high, low, close, timeperiod=20)."""
    fn = getattr(talib, name)
    out = fn(*[_f(s) for s in series], **params)
    like = series[0]
    if isinstance(out, tuple):
        return pd.DataFrame({f"out{i}": o for i, o in enumerate(out)}, index=like.index)
    return _wrap(out, like)
