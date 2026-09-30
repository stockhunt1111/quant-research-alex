"""The market an instrument trades in, read through its index: SPY for US stocks and ETFs, bitcoin on spot for coins
(the index a result's popup shows beside a list's buy & hold). A rule that times its names by the market rather than
by their own prices (`@rule(market=True)`: the market's trend, how long it has lasted, its volatility) reads the
index's daily closes here, each as a bar knows it (`known_at`): the last daily close stamped at or before the bar's
close. A bar inside a session sees the day before's close until its own day has closed, and a daily bar sees its own
day's, decided at its close and filled at the next open as every position is.

Commodities and currency pairs have no such index. Gold is one metal, not the commodities' market: its daily returns
correlate 0.04 with crude's since 2016. A currency pair is one currency priced in another, and a dollar index moves
EUR/USD and USD/JPY in opposite directions. A rule that reads the market refuses them.

A rule's positions are kept on disk by its bars (`strategy.fill_signals`). A rule that reads the market is also kept
by its index's bars (`version`), so positions worked out on an index since restated are never read back."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from strategy_lab import log
from strategy_lab.data.bars import Panel, load_bars, stored_version

LOG = log.get("market")
INDEX = {"us_equity": "td:SPY", "crypto_perp": "spot:BTCUSDT", "crypto_spot": "spot:BTCUSDT"}
TIMEFRAME = "1d"

_read: dict[str, tuple[int, pd.Series, str]] = {}      # an index's closes and their hash, by the stored version


def index_of(asset_class: str) -> str:
    """The index of an asset class's market; a class without one is refused (see the module)."""
    try:
        return INDEX[asset_class]
    except KeyError:
        raise ValueError(f"{asset_class} has no market index (gold is one metal, a currency pair no market): a rule "
                         f"that reads the market runs on {', '.join(sorted(INDEX))} only") from None


def _index_closes(index_id: str) -> tuple[int, pd.Series, str]:
    """The index's daily closes as stored, by their close times, with their hash; read again once the store changes."""
    version = stored_version([index_id], TIMEFRAME)
    got = _read.get(index_id)
    if got is None or got[0] != version:
        closes = load_bars(index_id, TIMEFRAME, fields=("close",))["close"].dropna()
        h = hashlib.blake2b(closes.index.asi8.tobytes(), digest_size=8)
        h.update(closes.to_numpy(dtype=np.float64).tobytes())
        got = _read[index_id] = version, closes, h.hexdigest()
    return got


def index_closes(bars: pd.DataFrame) -> pd.Series:
    """The daily closes of the index of the market the bars trade in (their `attrs`' asset class), over the whole
    stored history: what a bar may read of it is `known_at` its close."""
    if "asset_class" not in bars.attrs:
        raise ValueError("bars without their asset class (Panel.one sets it): no market to read")
    return _index_closes(index_of(bars.attrs["asset_class"]))[1].copy()


def version(panel: Panel) -> str:
    """A hash of the daily closes of the index of the panel's market: what a rule reading it is kept under, beside
    its own bars."""
    classes = {ins.asset_class for ins in panel.instruments.values()}
    ids = {index_of(c) for c in classes}
    if len(ids) != 1:
        raise ValueError(f"markets of different indices in one panel ({sorted(ids)})")
    return _index_closes(ids.pop())[2]


def known_at(daily: pd.Series | pd.DataFrame, times: pd.DatetimeIndex) -> pd.Series | pd.DataFrame:
    """`daily` as known at each of `times`: its last value stamped at or before it (NaN before its first)."""
    return daily.reindex(times, method="ffill")


def own_day_closes(bars: pd.DataFrame) -> pd.Series:
    """An instrument's close at the end of each of its days, stamped at the close of the day's last bar: known from that
    bar on. A bar belongs to the day it closes in (a coin's bar closing at midnight UTC ends the day before)."""
    c = bars.close.dropna()
    day = (c.index - pd.Timedelta(microseconds=1)).normalize()
    return c[~day.duplicated(keep="last")]


def trend(closes: pd.Series, n_days: int) -> pd.Series:
    """The side of a daily series' close against its own `n_days`-day simple moving average, day by day: 1 above, -1
    below, 0 on it and before the average has its days (a day is a row of the series: a session of SPY, a day of
    bitcoin)."""
    avg = closes.rolling(n_days, min_periods=n_days).mean()
    return np.sign(closes - avg).fillna(0.0)
