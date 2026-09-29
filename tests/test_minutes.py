import numpy as np
import pandas as pd
import pytest

from strategy_lab.data import minutes
from strategy_lab.data.minute_refresh import agreeing_months, conformed
from tests.conftest import make_minutes


def test_kept_minutes_are_checked_before_they_are_written():
    mins, _ = make_minutes(seed=6, days=1)
    with pytest.raises(ValueError, match="high is below"):
        minutes.write("td:AAA", mins.assign(high=mins["low"] * 0.99), {})
    with pytest.raises(ValueError, match="increasing"):
        minutes.write("td:AAA", mins.iloc[::-1], {})
    assert minutes.read("td:AAA") is None
    assert minutes.write("td:AAA", mins, {"source": "test"}) == len(mins)
    times, prices = minutes.read("td:AAA")
    assert prices.shape == (len(mins), 3) and not prices.flags.writeable
    assert times[0] == mins.index[0].value and minutes.read_meta("td:AAA")["minutes"] == len(mins)


def test_minutes_that_do_not_make_their_hour_are_walked_on_the_hours_own_prices():
    """An hour whose minutes do not make the stored hour, and an hour inside their span with no minute, are each one
    bar of the stored hour's prices; the other minutes are kept as they are."""
    mins, hours = make_minutes(seed=7, days=1)
    broken = mins.copy()
    broken.loc["2024-01-01 05:01", "open"] *= 0.99            # 05:01 closes the first minute of the 06:00 hour
    broken.loc["2024-01-01 05:01", "low"] = broken.loc["2024-01-01 05:01", "open"]
    holed = broken.drop(broken.index[(broken.index > "2024-01-01 09:00") & (broken.index <= "2024-01-01 10:00")])
    path, replaced, filled = conformed(holed, hours, 1e-6)
    assert replaced == ["2024-01-01 06:00:00+00:00"] and filled == 1
    for hour in ("2024-01-01 06:00", "2024-01-01 10:00"):
        inside = path[(path.index > pd.Timestamp(hour, tz="UTC") - pd.Timedelta(hours=1)) & (path.index <= hour)]
        assert len(inside) == 1 and inside.index[0] == pd.Timestamp(hour, tz="UTC")
        np.testing.assert_allclose(inside.iloc[0].to_numpy(), hours.loc[hour, ["open", "high", "low", "close"]].to_numpy())
    untouched = path[path.index <= "2024-01-01 05:00"]
    pd.testing.assert_frame_equal(untouched, mins[mins.index <= "2024-01-01 05:00"][["open", "high", "low", "close"]],
                                  check_freq=False)
    assert conformed(mins, hours, 1e-6)[1:] == ([], 0)


def test_a_second_sources_month_is_kept_only_on_the_stored_clock_and_at_a_steady_ratio_to_its_prices():
    """Minutes of another feed are kept month by month: on the stored clock and at one ratio to its closes (another side
    of the spread); a month stamped an hour late lines up best shifted, a month whose closes wander around the stored
    ones is another feed."""
    mins, hours = make_minutes(seed=8, days=60)          # January and February
    mins = mins[mins.index <= "2024-02-29 23:00"]             # the hour closing at midnight is March's
    feed = mins * (1 + 3e-4)                                  # quoted 3 bp above the stored side, steadily
    kept, refused = agreeing_months(feed, hours, 3.0)
    assert list(kept) == ["2024-01", "2024-02"] and refused == {}
    np.testing.assert_allclose(list(kept.values()), 1 + 3e-4, rtol=1e-6)
    late = feed.copy()
    jan = late.index < "2024-02-01"
    late.index = late.index.where(~jan, late.index + pd.Timedelta(hours=1))
    kept, refused = agreeing_months(late.sort_index(), hours, 3.0)
    assert list(kept) == ["2024-02"] and "lined up +1 h" in refused["2024-01"]
    rng = np.random.default_rng(3)
    wander = feed.copy()
    feb = wander.index >= "2024-02-01"
    wander.loc[feb] = wander.loc[feb].mul(1 + rng.normal(0, 1e-3, feb.sum()), axis=0)   # 10 bp of noise
    kept, refused = agreeing_months(wander, hours, 3.0)
    assert list(kept) == ["2024-01"] and "bp from their median ratio" in refused["2024-02"]


def test_a_vendors_minutes_are_put_on_the_stored_hours_basis_session_by_session():
    """The stored hours are the vendor's rescaled where its intraday series misses a corporate action: its minutes,
    scaled by each session's ratio of stored to made closes, make the stored hours' highs and lows exactly again, and
    minutes of a session the store has no hour of are not kept."""
    from strategy_lab.data.minute_refresh import on_stored_hours
    mins, hours = make_minutes(seed=10, days=6)
    stored = hours.copy()
    first_days = stored.index <= "2024-01-04 05:00"           # New York's sessions to January 3rd
    stored.loc[first_days, ["open", "high", "low", "close"]] *= 1.241        # a spin-off the minutes do not carry
    stored = stored[~((stored.index > "2024-01-05 12:00") & (stored.index <= "2024-01-06 12:00"))]
    assert len(conformed(mins, stored, 1e-6, ("high", "low"))[1]) > 0
    put, basis = on_stored_hours(mins, stored)
    assert basis["sessions_scaled"] >= 3 and basis["farthest"] == 1.241
    assert basis["minutes_without_a_stored_session"] == 0 or len(put) < len(mins)
    assert conformed(put, stored, 1e-6, ("high", "low"))[1] == []


def test_an_archive_minute_without_a_trade_is_not_kept():
    """Binance's archive writes a minute without a trade at the last close, volume zero: it makes no price, and kept it
    would open the hour where the exchange's hourly bar, opened by the hour's first trade, does not."""
    import io
    import zipfile
    from strategy_lab.data.minute_refresh import _klines
    rows = ["open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
            "taker_buy_quote_volume,ignore",
            "1567296000000,0.0026,0.0026,0.0026,0.0026,0,1567296059999,0,0,0,0,0",           # no trade
            "1567296060000,0.0027,0.0028,0.0027,0.0028,15000,1567296119999,41,3,9000,25,0"]  # a trade at 0.0027-0.0028
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("DOGEUSDT-1m-2019-09.csv", "\n".join(rows))
    got = _klines(buf.getvalue())
    assert list(got.index) == [pd.Timestamp("2019-09-01 00:02", tz="UTC")]           # stamped at the minute's close
    assert got.iloc[0].tolist() == [0.0027, 0.0028, 0.0027, 0.0028]
