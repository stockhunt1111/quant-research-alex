"""Build coarser bars from finer ones, per calendar. Only bars whose time is over are emitted: an FX block with all
its hours (a quote feed's missing hour is missing data), a US listing's or a future's with the hours that traded in
it, stamped at its end (an exchange prints no bar for an hour without a trade).

Inputs and outputs are indexed by bar close time (UTC) with the store's bar columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_lab.data import calendars as cal

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "dollar_volume": "sum"}


def _aggregate(df: pd.DataFrame, key) -> pd.DataFrame:
    out = df.groupby(key, sort=True).agg(AGG)
    ct = pd.Series(df.index, index=df.index).groupby(key, sort=True)
    out["close_time"] = ct.max().to_numpy()
    out["n"] = ct.size().to_numpy()
    return out


def _finish(out: pd.DataFrame, keep: np.ndarray) -> pd.DataFrame:
    return out[keep].set_index("close_time")[list(AGG)].sort_index()


def equity_4h_from_1h(h1: pd.DataFrame) -> pd.DataFrame:
    """Two bars per regular session: open -> 13:30 New York, and 13:30 -> close (only the first on early closes), each
    stamped at its end and made of the hours that traded in it (a thin ETF's quiet hour has no bar). A session whose
    hourly bars all close on the whole hour (clock-hour bars) splits at 13:00 instead."""
    if h1.empty:
        return h1
    ny = h1.index.tz_convert(cal.NY)
    session = ny.normalize().tz_localize(None)
    half_hours = pd.Series(np.asarray(ny.minute == 30), index=session).groupby(level=0).any()
    split_hm = pd.Series(np.where(half_hours, cal.EQUITY_4H_SPLIT, cal.EQUITY_4H_SPLIT_CLOCK_HOURS), index=half_hours.index)

    def split_at(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return pd.to_datetime(pd.Index(days.strftime("%Y-%m-%d")) + " " + pd.Index(split_hm.reindex(days).to_numpy())
                              ).tz_localize(cal.NY)

    seg = np.asarray(ny > split_at(session), dtype=int)
    out = _aggregate(h1, [session, seg])
    days = out.index.get_level_values(0)
    sched = cal.nyse_sessions(h1.index.min() - pd.Timedelta(days=1), h1.index.max() + pd.Timedelta(days=1))
    sess_close = pd.DatetimeIndex(sched["market_close"].reindex(days)).tz_convert("UTC")
    split_utc = split_at(pd.DatetimeIndex(days)).tz_convert("UTC")
    first_end = np.minimum(split_utc.to_numpy(), sess_close.to_numpy())
    expected = np.where(out.index.get_level_values(1).to_numpy() == 1, sess_close.to_numpy(), first_end)
    out["close_time"] = pd.to_datetime(expected, utc=True)
    return _finish(out, out["close_time"].notna().to_numpy())


def fx_4h_from_1h(h1: pd.DataFrame, daily_break: bool = False) -> pd.DataFrame:
    """Six bars per FX trading day, in 4h blocks from 17:00 New York; a block is complete with all four hours.

    With `daily_break` (spot metals and crude pause from 17:00 to 18:00 New York on most days) the day's first block
    is complete with the three hours after the pause.
    """
    if h1.empty:
        return h1
    starts = h1.index - pd.Timedelta(hours=1)
    day = cal.fx_day_label(starts)
    ny = starts.tz_convert(cal.NY)
    block = np.asarray(((ny.hour - cal.FX_DAY_CUT_HOUR_NY) % 24) // 4)
    out = _aggregate(h1, [day, block])
    first = out.index.get_level_values(1).to_numpy() == 0
    need = np.where(first & daily_break, 3, 4)
    return _finish(out, out["n"].to_numpy() >= need)


def fx_1d_from_1h(h1: pd.DataFrame, min_hours: int = 20) -> pd.DataFrame:
    """Daily FX bar for the day ending 17:00 New York.

    Kept when most hours exist and its last hour closes within two hours of the cut: the vendor's final Friday
    hour often ends at 15:00-16:00 New York, and dropping those days would leave a hole most weeks.
    """
    if h1.empty:
        return h1
    starts = h1.index - pd.Timedelta(hours=1)
    out = _aggregate(h1, cal.fx_day_label(starts))
    day_close = cal.fx_day_close(out.index)
    last = pd.DatetimeIndex(out["close_time"]).tz_convert("UTC")
    near_cut = np.asarray((last <= day_close) & (last >= day_close - pd.Timedelta(hours=2)))
    return _finish(out, near_cut & (out["n"].to_numpy() >= min_hours))


def futures_4h_from_1h(h1: pd.DataFrame) -> pd.DataFrame:
    """Six bars per trading day of a CME future, in the FX day's 4h blocks from 17:00 New York, each stamped at its
    block's end. The exchange prints no bar for an hour without a trade (a thin night, the pause from 17:00, a
    holiday's halt), so a block holds the hours that traded in it, and it is complete only when its time is over: a
    later trade could still join it before."""
    if h1.empty:
        return h1
    starts = h1.index - pd.Timedelta(hours=1)
    day = cal.fx_day_label(starts)
    ny = starts.tz_convert(cal.NY)
    out = _aggregate(h1, [day, np.asarray(((ny.hour - cal.FX_DAY_CUT_HOUR_NY) % 24) // 4)])
    opened = cal.fx_day_close(pd.DatetimeIndex(out.index.get_level_values(0)) - pd.Timedelta(days=1))
    out["close_time"] = opened + pd.to_timedelta(4 * (out.index.get_level_values(1).to_numpy() + 1), unit="h")
    return _finish(out, np.ones(len(out), dtype=bool))


def futures_1d_from_1h(h1: pd.DataFrame) -> pd.DataFrame:
    """Daily bar of a CME future for the day ending 17:00 New York, where CME's metals and energy end their trading
    day, stamped at that hour: every hour that traded in the day, a holiday's short session included."""
    if h1.empty:
        return h1
    out = _aggregate(h1, cal.fx_day_label(h1.index - pd.Timedelta(hours=1)))
    out["close_time"] = cal.fx_day_close(pd.DatetimeIndex(out.index))
    return _finish(out, np.ones(len(out), dtype=bool))

