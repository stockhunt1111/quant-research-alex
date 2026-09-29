import numpy as np
import pandas as pd
import pytest

from strategy_lab import metrics


def _daily_from_months(monthly, start="2024-01-01"):
    """Daily series whose calendar months compound exactly to `monthly`."""
    parts = []
    month = pd.Period(start[:7], "M")
    for r in monthly:
        days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D", tz="UTC")
        per_day = (1 + r) ** (1 / len(days)) - 1
        parts.append(pd.Series(per_day, index=days))
        month += 1
    return pd.concat(parts)


def test_monthly_counts_streaks_and_extremes():
    months = [0.02, -0.01, 0.03, 0.0, -0.02, -0.01, 0.01]
    card = metrics.core(_daily_from_months(months))
    assert card["months"] == 7
    assert card["pct_green"] == pytest.approx(3 / 7)
    assert card["pct_red"] == pytest.approx(3 / 7)
    assert card["longest_red_streak"] == 2
    assert card["worst_month"] == pytest.approx(-0.02)


def test_the_average_month_is_the_steady_rate_that_ends_where_the_account_ended():
    assert metrics.core(_daily_from_months([1.0, -0.5]))["avg_monthly"] == pytest.approx(0.0, abs=1e-12)  # 100 -> 200 -> 100
    months = [0.02, -0.01, 0.03, 0.0, -0.02, -0.01, 0.01]
    assert metrics.core(_daily_from_months(months))["avg_monthly"] < np.mean(months)   # swings cost: below the plain mean
    assert metrics.core(_daily_from_months([0.05, -1.0, 0.05]))["avg_monthly"] == -1.0        # wiped out stays wiped
    year = pd.Series(2 ** (1 / 365) - 1, index=pd.date_range("2023-01-01", periods=365, freq="D", tz="UTC"))
    assert metrics.core(year)["avg_monthly"] == pytest.approx(2 ** (1 / 12) - 1)    # doubled in a year: 5.95% a month


def test_a_partial_month_counts_by_its_days():
    """2.5 months of the same days give the same average month wherever the calendar cuts them."""
    days = np.random.default_rng(3).normal(0.002, 0.01, 76)
    aligned = pd.Series(days, index=pd.date_range("2024-01-01", periods=76, freq="D", tz="UTC"))
    shifted = pd.Series(days, index=pd.date_range("2024-01-16", periods=76, freq="D", tz="UTC"))
    assert metrics.core(aligned)["avg_monthly"] == pytest.approx(metrics.core(shifted)["avg_monthly"])
    assert np.isnan(metrics.core(aligned[:20])["avg_monthly"])             # not one whole month: no average month


def test_a_month_without_a_position_is_neither_green_nor_red_for_a_strategy_but_not_green_for_capital():
    months = [0.02, 0.0, 0.0, 0.0, -0.01, 0.03]                      # three months out of the market
    card = metrics.scorecard(_daily_from_months(months))
    assert card["pct_months_active"] == pytest.approx(3 / 6)
    assert card["pct_green_active"] == pytest.approx(2 / 3) and card["pct_red_active"] == pytest.approx(1 / 3)
    assert card["pct_green"] == pytest.approx(2 / 6)
    many = [0.02] * 8 + [0.0] * 8 + [-0.01] * 2                     # 80% green in the market, 44% of all months
    strategy = metrics.scorecard(_daily_from_months(many))
    capital = metrics.scorecard(_daily_from_months(many), portfolio=True)
    assert strategy["targets"]["green_months>=70%"] and not capital["targets"]["green_months>=70%"]


def test_walk_forward_windows_grow_from_the_first_bar():
    from strategy_lab import walkforward
    start, end = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2021-10-01", tz="UTC")
    fl = walkforward.folds(start, end, "365D", "91D")
    assert all(f.train_start == start for f in fl)
    assert fl[0].test_start == start + pd.Timedelta("365D") and fl[1].test_start == fl[0].test_end
    assert fl[-1].test_end == end


def test_drawdown_of_a_known_path():
    eq = np.array([1.0, 1.1, 0.88, 0.99, 1.2])
    d = pd.Series(eq[1:] / eq[:-1] - 1, index=pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"))
    mdd, days = metrics.max_drawdown(d)
    assert mdd == pytest.approx(0.88 / 1.1 - 1)
    assert days == 2


def test_a_bar_closing_at_midnight_belongs_to_the_previous_day_and_empty_days_are_zero():
    idx = pd.DatetimeIndex(["2024-01-31 12:00", "2024-02-01 00:00", "2024-02-03 12:00"], tz="UTC")
    d = metrics.daily_returns(pd.Series([0.01, 0.02, 0.03], index=idx))
    assert d.loc["2024-01-31"] == pytest.approx(1.01 * 1.02 - 1)
    assert d.loc["2024-02-02"] == 0.0
    assert len(d) == 4


def test_partial_first_and_last_months_are_excluded():
    d = pd.Series(0.001, index=pd.date_range("2024-01-15", "2024-04-10", freq="D", tz="UTC"))
    assert list(metrics.monthly_returns(d).index.astype(str)) == ["2024-02", "2024-03"]


def test_weekday_series_on_calendar_days_keeps_the_business_day_sharpe():
    rng = np.random.default_rng(0)
    bdays = pd.bdate_range("2020-01-01", periods=2000, tz="UTC")
    r = pd.Series(rng.normal(0.0005, 0.01, len(bdays)), index=bdays)
    cal = metrics.daily_returns(r.set_axis(r.index + pd.Timedelta(hours=21)))
    sharpe_bd = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert metrics.core(cal)["sharpe"] == pytest.approx(sharpe_bd, rel=0.02)


def test_targets_and_buy_and_hold_comparison():
    good = _daily_from_months([0.02] * 12)
    bh = _daily_from_months([0.03, -0.05] * 6)
    card = metrics.scorecard(good, benchmark=bh)
    assert card["beats_bh"] and card["targets_met"] >= 4


def _bench(n=600, seed=1):
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(np.random.default_rng(seed).normal(0.0005, 0.02, n), index=idx)


WARM = metrics.EQUAL_RISK_DAYS // 3 + 1          # days before the trailing volatility sizes the record


def test_holding_the_benchmark_itself_does_not_beat_it():
    bench = _bench()
    card = metrics.scorecard(bench, benchmark=bench)
    assert card["vs_bh"] == pytest.approx(0.0, abs=1e-12) and not card["beats_bh"]


def test_the_benchmark_held_at_half_size_is_the_benchmark_once_sized_to_its_risk():
    bench = _bench()
    sized = metrics.equal_risk(0.5 * bench, pd.Series(0.5, index=bench.index), bench)
    assert np.allclose(sized.iloc[WARM:], bench.iloc[WARM:])          # doubled: nothing idle, nothing borrowed


def test_idle_equity_earns_t_bills_and_money_borrowed_past_equity_pays_the_spread():
    from tests.conftest import CASH_RATE
    bench, rf = _bench(), CASH_RATE / metrics.DAYS
    flat = metrics.equal_risk(pd.Series(0.0, index=bench.index), pd.Series(0.0, index=bench.index), bench)
    assert np.allclose(flat, rf)
    quarter = 0.25 * bench                        # a quarter of the risk, fully invested: sized up to the cap, on credit
    sized = metrics.equal_risk(quarter, pd.Series(1.0, index=bench.index), bench)
    cap = metrics.EQUAL_RISK_CAP
    expected = cap * quarter + (1 - cap) * (rf + metrics.FINANCING_SPREAD / metrics.DAYS)
    assert np.allclose(sized.iloc[WARM:], expected.iloc[WARM:])


def test_against_cash_a_record_is_not_sized_and_its_idle_equity_earns_t_bills_too():
    from tests.conftest import CASH_RATE
    bench, rf = _bench(), CASH_RATE / metrics.DAYS
    zero = pd.Series(0.0, index=bench.index)
    assert metrics.vs_hold(zero, zero, zero, cash=True) == pytest.approx(0.0, abs=1e-12)     # holding nothing ties
    half = 0.5 * bench                            # half invested: not sized up, its other half at T-bills
    expected = metrics.core(half + 0.5 * rf)["cagr"] - metrics.core(zero + rf)["cagr"]
    assert metrics.vs_hold(half, pd.Series(0.5, index=bench.index), zero, cash=True) == pytest.approx(expected)


def test_a_fully_invested_record_beats_cash_when_it_compounds_faster_than_t_bills():
    from tests.conftest import CASH_RATE
    idx = _bench().index
    for times, beats in ((1.5, True), (0.5, False)):
        card = metrics.scorecard(pd.Series(times * CASH_RATE / metrics.DAYS, index=idx),
                                 benchmark=pd.Series(0.0, index=idx), cash=True)
        assert card["bh_cash"] and card["beats_bh"] == beats


def test_holding_futures_leaves_the_money_in_t_bills_on_both_sides():
    from tests.conftest import CASH_RATE
    bench, rf = _bench(), CASH_RATE / metrics.DAYS
    zero = pd.Series(0.0, index=bench.index)
    # a record that holds the futures as buy & hold does ties it: both keep their money in T-bills
    assert metrics.vs_hold(bench, pd.Series(1.0, index=bench.index), bench, margined=True) == pytest.approx(0.0, abs=1e-12)
    # out of the market, the record earns T-bills, as holding does on top of the futures
    expected = metrics.core(zero + rf)["cagr"] - metrics.core(bench + rf)["cagr"]
    assert metrics.vs_hold(zero, zero, bench, margined=True) == pytest.approx(expected)
    # a quarter of the risk sized up to the cap borrows nothing: its equity still earns T-bills
    sized = metrics.equal_risk(0.25 * bench, pd.Series(1.0, index=bench.index), bench, margined=True)
    assert np.allclose(sized.iloc[WARM:], (metrics.EQUAL_RISK_CAP * 0.25 * bench + rf).iloc[WARM:])
