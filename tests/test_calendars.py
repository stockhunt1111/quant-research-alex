import numpy as np
import pandas as pd

from strategy_lab.data import calendars as cal
from strategy_lab.data import resample


def _utc(s):
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")


def test_equity_hourly_close_handles_dst_early_close_and_out_of_session_prints():
    starts = pd.DatetimeIndex([_utc("2024-03-08 15:30"), _utc("2024-03-11 15:30"),      # around the DST switch
                               _utc("2024-11-29 12:30"), _utc("2024-11-29 15:30"),      # early close at 13:00
                               _utc("2024-07-05 08:00")])                               # pre-market
    close = cal.equity_intraday_close(starts)
    assert close.iloc[0] == _utc("2024-03-08 16:00") and close.iloc[1] == _utc("2024-03-11 16:00")
    assert close.iloc[2] == _utc("2024-11-29 13:00")
    assert pd.isna(close.iloc[3]) and pd.isna(close.iloc[4])


def test_fx_week_and_trading_day_cut_at_17_new_york():
    starts = pd.DatetimeIndex([_utc("2024-06-07 16:00"), _utc("2024-06-07 17:00"),     # Friday: last hour, closed
                               _utc("2024-06-09 16:00"), _utc("2024-06-09 17:00")])    # Sunday: closed, open
    assert list(cal.fx_in_session(starts)) == [True, False, False, True]
    labels = cal.fx_day_label(pd.DatetimeIndex([_utc("2024-06-09 17:00"), _utc("2024-06-10 16:00")]))
    assert list(labels.strftime("%Y-%m-%d")) == ["2024-06-10", "2024-06-10"]


def _hourly(closes):
    idx = pd.DatetimeIndex(closes)
    v = np.arange(1, len(idx) + 1, dtype=float)
    return pd.DataFrame({"open": v, "high": v + 1, "low": v - 1, "close": v, "volume": 1.0, "dollar_volume": v}, index=idx)


def test_equity_4h_is_two_bars_per_session_and_one_on_early_close():
    day = [_utc(f"2024-07-10 {t}") for t in ("10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00")]
    early = [_utc(f"2024-11-29 {t}") for t in ("10:30", "11:30", "12:30", "13:00")]
    b = resample.equity_4h_from_1h(_hourly(day + early))
    assert list(b.index) == [_utc("2024-07-10 13:30"), _utc("2024-07-10 16:00"), _utc("2024-11-29 13:00")]
    assert b["open"].iloc[1] == 5 and b["close"].iloc[1] == 7 and b["high"].iloc[1] == 8
    quiet = [_utc(f"2024-07-11 {t}") for t in ("10:30", "11:30", "12:30", "14:30", "15:30", "16:00")]   # no trade 12:30-13:30
    q = resample.equity_4h_from_1h(_hourly(quiet))
    assert list(q.index) == [_utc("2024-07-11 13:30"), _utc("2024-07-11 16:00")]                    # stamped at its end
    assert q["close"].iloc[0] == 3 and q["open"].iloc[1] == 4


def test_a_metal_keeps_its_evening_4h_bar_although_it_pauses_17_to_18_new_york():
    closes = pd.date_range(_utc("2024-06-09 19:00"), _utc("2024-06-10 17:00"), freq="h")   # first print 18:00-19:00
    fx = resample.fx_4h_from_1h(_hourly(closes))
    metal = resample.fx_4h_from_1h(_hourly(closes), daily_break=True)
    assert len(fx) == 5 and len(metal) == 6
    assert metal.index[0] == _utc("2024-06-09 21:00") and metal["open"].iloc[0] == 1 and metal["close"].iloc[0] == 3


def test_fx_daily_from_hourly_closes_at_17_new_york():
    closes = pd.date_range(_utc("2024-06-09 18:00"), _utc("2024-06-10 17:00"), freq="h")   # Sun 17:00 -> Mon 17:00
    d = resample.fx_1d_from_1h(_hourly(closes))
    assert list(d.index) == [_utc("2024-06-10 17:00")]
    assert d["open"].iloc[0] == 1 and d["close"].iloc[0] == 24



def test_a_futures_bar_holds_the_hours_that_traded_and_is_stamped_at_its_end():
    # no bar for an hour without a trade: a thin Sunday evening, a night with none, a holiday halted at 13:30
    closes = [_utc("2025-01-12 19:00"), _utc("2025-01-12 21:00"), _utc("2025-01-13 04:00"),
              _utc("2025-01-13 14:00"), _utc("2025-01-13 15:00"),
              _utc("2025-01-19 19:00"), _utc("2025-01-20 14:00")]                        # MLK day: halt at 13:30
    b4 = resample.futures_4h_from_1h(_hourly(closes))
    assert list(b4.index) == [_utc("2025-01-12 21:00"), _utc("2025-01-13 05:00"), _utc("2025-01-13 17:00"),
                              _utc("2025-01-19 21:00"), _utc("2025-01-20 17:00")]
    assert b4["open"].iloc[0] == 1 and b4["close"].iloc[0] == 2 and b4["volume"].iloc[2] == 2
    d = resample.futures_1d_from_1h(_hourly(closes))
    assert list(d.index) == [_utc("2025-01-13 17:00"), _utc("2025-01-20 17:00")]
    assert d["open"].iloc[1] == 6 and d["close"].iloc[1] == 7 and d["high"].iloc[1] == 8


def test_cme_days_keep_a_holiday_s_short_session_and_leave_out_the_days_it_did_not_open():
    days = cal.cme_days(pd.Timestamp("2013-01-17"), pd.Timestamp("2025-01-21"))
    assert pd.Timestamp("2025-01-20") in days                  # MLK day 2025: a session halted at 14:30 New York
    assert pd.Timestamp("2013-01-21") not in days              # MLK day 2013: no trade in any gold contract
    assert pd.Timestamp("2024-03-29") not in days              # Good Friday
    assert pd.Timestamp("2022-12-26") not in days              # Christmas on a Sunday, closed the Monday
    assert pd.Timestamp("2013-01-18") in days and pd.Timestamp("2013-01-19") not in days
