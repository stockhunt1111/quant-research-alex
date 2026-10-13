"""Trading calendars and bar-close normalisation.

Every bar in the store is indexed by the moment it CLOSED (UTC). A bar that has not closed does not exist.
Vendors stamp bars differently (TwelveData and Binance stamp the start), so the conversion lives here, once.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

NY = "America/New_York"
# FX trades from Sunday 17:00 to Friday 17:00 New York time; the daily FX bar closes at 17:00 New York.
FX_DAY_CUT_HOUR_NY = 17
# The regular US equity session is split into two 4h bars at 13:30 New York; a session whose hourly bars cover clock
# hours (the vendor's first half of 2020) at 13:00, its nearest bar boundary.
EQUITY_4H_SPLIT = "13:30"
EQUITY_4H_SPLIT_CLOCK_HOURS = "13:00"


@lru_cache(maxsize=8)
def _xnys(start: str, end: str) -> pd.DataFrame:
    return mcal.get_calendar("XNYS").schedule(start_date=start, end_date=end)


def nyse_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Regular XNYS sessions (DST and early closes included): index = session date, columns market_open/close UTC."""
    sched = _xnys(str(start.date()), str(end.date()))
    return sched[["market_open", "market_close"]]


def equity_intraday_close(starts: pd.DatetimeIndex) -> pd.Series:
    """Close time of each US-equity intraday bar: the next bar's start inside the session, or the session close.

    Bars that start outside a regular session (pre/post market, prints after an early close) map to NaT.
    """
    starts = starts.tz_convert("UTC")
    sched = nyse_sessions(starts.min() - pd.Timedelta(days=1), starts.max() + pd.Timedelta(days=1))
    day = starts.tz_convert(NY).normalize().tz_localize(None)
    opens = sched["market_open"].reindex(day).to_numpy()
    closes = sched["market_close"].reindex(day).to_numpy()
    inside = (starts.to_numpy() >= opens) & (starts.to_numpy() < closes)
    out = pd.Series(pd.NaT, index=starts, dtype="datetime64[ns, UTC]")
    s = pd.Series(starts[inside], index=starts[inside])
    sess_close = pd.Series(pd.to_datetime(closes[inside], utc=True), index=starts[inside])
    nxt = s.shift(-1)
    same_session = nxt.notna() & (nxt < sess_close)
    out.loc[s.index] = nxt.where(same_session, sess_close)
    return out


def equity_daily_close(dates: pd.DatetimeIndex) -> pd.Series:
    """Close time of each US-equity daily bar (vendor stamps the session date). Non-session dates map to NaT."""
    d = pd.DatetimeIndex(dates).tz_localize(None).normalize() if dates.tz is not None else pd.DatetimeIndex(dates).normalize()
    sched = nyse_sessions(d.min() - pd.Timedelta(days=1), d.max() + pd.Timedelta(days=1))
    closes = sched["market_close"].reindex(d)
    return pd.Series(pd.to_datetime(closes.to_numpy(), utc=True), index=dates)


def fx_in_session(starts: pd.DatetimeIndex) -> pd.Series:
    """True for FX bars inside the trading week (Sun 17:00 to Fri 17:00 New York)."""
    ny = starts.tz_convert(NY)
    minute_of_week = ny.dayofweek * 1440 + ny.hour * 60 + ny.minute   # Monday 00:00 = 0
    fri_close = 4 * 1440 + FX_DAY_CUT_HOUR_NY * 60
    sun_open = 6 * 1440 + FX_DAY_CUT_HOUR_NY * 60
    return pd.Series((minute_of_week < fri_close) | (minute_of_week >= sun_open), index=starts)


def fx_day_label(starts: pd.DatetimeIndex) -> pd.Index:
    """Trading date of an FX bar: bars from 17:00 New York belong to the next day's bar."""
    ny = starts.tz_convert(NY)
    shifted = ny + pd.Timedelta(hours=24 - FX_DAY_CUT_HOUR_NY)
    return shifted.tz_localize(None).normalize()


def fx_day_close(labels: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC close time (17:00 New York) of the FX trading date."""
    return (pd.DatetimeIndex(labels) + pd.Timedelta(hours=FX_DAY_CUT_HOUR_NY)).tz_localize(NY).tz_convert("UTC")


# US holidays of 2012-2014 on which CME's metals and energy did not trade at all, where the exchange's calendar has a
# short session (Thanksgiving 2012 and 2013, Independence Day 2013, the Monday holidays of 2013 and early 2014): no bar
# of gold, silver, platinum, palladium, crude or copper, on days Databento's feed is whole (every gold contract asked
# on 2013-01-21 and 2014-02-17, none traded; 2026-09-27)
CME_CLOSED = pd.DatetimeIndex(["2012-11-22", "2013-01-21", "2013-02-18", "2013-05-27", "2013-07-04", "2013-09-02",
                               "2013-11-28", "2014-01-20", "2014-02-17"])


@lru_cache(maxsize=8)
def _cme(start: str, end: str) -> pd.DatetimeIndex:
    sched = mcal.get_calendar("CMEGlobex_EnergyAndMetals").schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(sched.index).difference(CME_CLOSED)


def cme_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """The days CME's metals and energy trade, as naive dates: each the day its session ends at 17:00 New York (the
    FX day's label), a holiday's short session included."""
    return _cme(str(start.date()), str(end.date()))
