"""The scorecard against the firm's target, computed the same way for every asset class.

Bar returns are compounded into UTC calendar days (a bar belongs to the day in which it closed; one closing at
00:00 belongs to the day before) and days without a bar count as 0. Crypto and equities therefore share one
annualisation (365) and their Sharpe ratios are comparable. Monthly figures use complete calendar months only, except
the average month (`average_month`): the steady compounded rate that takes the account from the record's start to its
end over the record's own length, a partial month at either end counted by its days.
A month in which the strategy held no position returns exactly 0: it is neither green nor red. A strategy's green
and red shares are taken over the months it was in the market (`pct_green_active`), next to the share of months
it was (`pct_months_active`); a portfolio of capital is judged on all months (`pct_green`), since a month in
which the money earned nothing is not a green month for its owner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from strategy_lab.config import FIRM_TARGETS
from strategy_lab.data import rates

DAYS = 365.0
DAY_NS = 86_400_000_000_000


def daily_returns(bar_returns: pd.Series) -> pd.Series:
    """Bar returns compounded into the UTC day each bar closed in, every day from the first to the last (0 on a day
    without a bar); a bar's NaN return counts as none."""
    if bar_returns.empty:
        return bar_returns
    idx = bar_returns.index
    if not (isinstance(idx, pd.DatetimeIndex) and idx.tz is not None and idx.is_monotonic_increasing):
        day = (idx - pd.Timedelta(microseconds=1)).tz_convert("UTC").normalize()
        d = (1.0 + bar_returns).groupby(day).prod() - 1.0
        full = pd.date_range(d.index.min(), d.index.max(), freq="D", tz="UTC")
        return d.reindex(full, fill_value=0.0)
    day = (idx.asi8 - 1000) // DAY_NS                   # a bar closing at 00:00 belongs to the day before
    out = _compounded(bar_returns.to_numpy(dtype=np.float64), day)
    full = pd.date_range(pd.Timestamp(int(day[0]) * DAY_NS, tz="UTC"), periods=len(out), freq="D")
    return pd.Series(out, index=full, name=bar_returns.name)


@njit(cache=True)
def _compounded(values, day):
    """`values` compounded by `day` (a day number each, in order), one entry per day from the first to the last."""
    out = np.ones(day[-1] - day[0] + 1)
    for i in range(len(values)):
        if not np.isnan(values[i]):
            out[day[i] - day[0]] *= 1.0 + values[i]
    return out - 1.0


def monthly_returns(daily: pd.Series) -> pd.Series:
    """Compounded calendar-month returns, complete months only."""
    if daily.empty:
        return daily
    m = (1.0 + daily).groupby(daily.index.tz_localize(None).to_period("M")).prod() - 1.0
    first, last = daily.index.min(), daily.index.max()
    if first.day != 1:
        m = m.iloc[1:]
    if last.day != last.days_in_month:
        m = m.iloc[:-1]
    return m


def average_month(growth: float, days: int) -> float:
    """The steady monthly rate that compounds to `growth` (end value over start value of the account) in `days`
    calendar days: a month is 1/12 of a year, so a partial month or year counts by its days. A plain mean of the
    months overstates a record that swings: +100% then -50% averages +25% while the account is back where it
    started; here it averages 0%. A record that lost everything averages -100%."""
    months = days / DAYS * 12.0
    if months <= 0:
        return np.nan
    return growth ** (1.0 / months) - 1.0 if growth > 0 else -1.0


def max_drawdown(daily: pd.Series) -> tuple[float, int]:
    """Deepest peak-to-trough loss of compounded equity, and the longest time under water in days."""
    eq = (1.0 + daily).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0, _longest_run((dd < 0).to_numpy())


def _longest_run(flags: np.ndarray) -> int:
    """The length of the longest run of True (the Monte Carlo asks it of a thousand resamples of a record)."""
    if not flags.any():
        return 0
    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def longest_streak(values: pd.Series, predicate) -> int:
    longest = run = 0
    for v in values:
        run = run + 1 if predicate(v) else 0
        longest = max(longest, run)
    return longest


def core(daily: pd.Series) -> dict:
    """Return/risk figures of a daily series (no trades, no targets)."""
    d = daily.dropna()
    months = monthly_returns(d)
    sd = d.std(ddof=1)
    downside = d[d < 0].std(ddof=1)
    eq_end = float((1.0 + d).prod())
    years = len(d) / DAYS
    mdd, mdd_days = max_drawdown(d)
    return {
        "start": str(d.index.min().date()) if len(d) else None,
        "end": str(d.index.max().date()) if len(d) else None,
        "months": int(len(months)),
        "avg_monthly": average_month(eq_end, len(d)) if len(months) else np.nan,
        "median_monthly": float(months.median()) if len(months) else np.nan,
        "pct_green": float((months > 0).mean()) if len(months) else np.nan,
        "pct_red": float((months < 0).mean()) if len(months) else np.nan,
        "months_active": int((months != 0).sum()),
        "pct_months_active": float((months != 0).mean()) if len(months) else np.nan,
        "pct_green_active": float((months[months != 0] > 0).mean()) if (months != 0).any() else np.nan,
        "pct_red_active": float((months[months != 0] < 0).mean()) if (months != 0).any() else np.nan,
        "worst_month": float(months.min()) if len(months) else np.nan,
        "best_month": float(months.max()) if len(months) else np.nan,
        "longest_red_streak": longest_streak(months, lambda v: v < 0),
        "cagr": float(eq_end ** (1.0 / years) - 1.0) if years > 0 and eq_end > 0 else np.nan,
        "ann_vol": float(sd * np.sqrt(DAYS)) if len(d) > 1 else np.nan,
        "sharpe": float(d.mean() / sd * np.sqrt(DAYS)) if len(d) > 1 and sd > 0 else 0.0,
        "sortino": float(d.mean() / downside * np.sqrt(DAYS)) if len(d) > 2 and downside > 0 else np.nan,
        "max_dd": mdd,
        "max_dd_days": mdd_days,
    }


def trade_stats(trades: pd.DataFrame | None, months: int) -> dict:
    if trades is None or trades.empty:
        return {"n_trades": 0, "trades_per_month": 0.0, "win_rate": np.nan, "avg_trade": np.nan,
                "median_trade_bars": np.nan}
    return {
        "n_trades": int(len(trades)),
        "trades_per_month": float(len(trades) / months) if months else np.nan,
        "win_rate": float((trades["net_return"] > 0).mean()),
        "avg_trade": float(trades["net_return"].mean()),
        "median_trade_bars": float(trades["bars"].median()),
    }


def exposure_stats(held: pd.Series | None) -> dict:
    """The share of days with a position and the average exposure, from the exposure held through each bar
    (`backtest.Result.exposure`: sum of |weight| of what the account holds, drifted since the fills)."""
    if held is None or held.empty:
        return {"time_in_market": np.nan, "avg_gross": np.nan}
    by_day = held.groupby((held.index - pd.Timedelta(microseconds=1)).normalize()).max()
    return {"time_in_market": float((by_day > 0).mean()), "avg_gross": float(held.mean())}


EQUAL_RISK_CAP = 2.0          # the research desk's ceiling on the sizing: a retail margin account's leverage
EQUAL_RISK_DAYS = 90          # the trailing window the sizing reads, calendar days (the desk does not publish its own)
FINANCING_SPREAD = 0.015      # a year above T-bills on money borrowed past 100% of equity (the desk's benchmark + 1.5%)


def daily_gross(held: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series:
    """Each day's average gross exposure over its bars, from the exposure held through each bar (`exposure_stats`); fully
    invested when it is not known."""
    if held is None or held.empty:
        return pd.Series(1.0, index=index)
    day = (held.index - pd.Timedelta(microseconds=1)).tz_convert("UTC").normalize()
    return held.groupby(day).mean().reindex(index, fill_value=0.0)


def t_bills(index: pd.DatetimeIndex) -> pd.Series:
    """Each day's return of cash at the 3-month T-bill rate."""
    return rates.daily_rate().reindex(index).ffill().bfill()


def _with_cash(daily: pd.Series, gross: pd.Series, rf: pd.Series, margined: bool = False) -> pd.Series:
    """`daily` with the equity its positions leave idle earning T-bills, and money borrowed past 100% paying T-bills +
    FINANCING_SPREAD. A futures position (`margined`) takes margin, not cash: all the equity earns T-bills, whatever
    the exposure, and nothing is borrowed (a joined futures series carries its roll, not the interest on its money)."""
    if margined:
        return daily + rf
    idle = 1.0 - gross
    return daily + idle * np.where(idle >= 0.0, rf, rf + FINANCING_SPREAD / DAYS)


def equal_risk(daily: pd.Series, gross: pd.Series, benchmark: pd.Series, margined: bool = False) -> pd.Series:
    """`daily` sized to `benchmark`'s risk before its money is counted, as the research desk sizes a book ("more money
    than buy-and-hold once sized to equal risk"): each day's multiple is the benchmark's trailing volatility over the
    record's, both known the day before, at most EQUAL_RISK_CAP (1 until a third of the window has passed); the equity
    the positions leave idle earns T-bills, and money borrowed past 100% pays T-bills + FINANCING_SPREAD (`margined`:
    all the equity earns T-bills, see `_with_cash`)."""
    vol = lambda s: s.rolling(EQUAL_RISK_DAYS, min_periods=EQUAL_RISK_DAYS // 3).std().shift(1)   # noqa: E731
    k = (vol(benchmark) / vol(daily)).replace(np.inf, EQUAL_RISK_CAP).clip(upper=EQUAL_RISK_CAP).fillna(1.0)
    return _with_cash(k * daily, k * gross.reindex(daily.index).fillna(0.0), t_bills(daily.index), margined)


def vs_hold(daily: pd.Series, gross: pd.Series, benchmark: pd.Series, cash: bool = False,
            margined: bool = False) -> float:
    """How much more a year than buy-and-hold the record makes, the research desk's money test: sized to buy-and-hold's
    risk (`equal_risk`), against buy-and-hold. With nothing to hold (`cash`: FX pairs, crude's spot quote),
    buy-and-hold is cash at T-bills and the record is taken as it is, its idle equity at T-bills too: cash has no risk
    to size to. Holding futures (`margined`) leaves the money in T-bills on both sides."""
    if cash:
        rf = t_bills(daily.index)
        return core(_with_cash(daily, gross.reindex(daily.index).fillna(0.0), rf))["cagr"] - core(rf)["cagr"]
    held = benchmark + t_bills(daily.index) if margined else benchmark
    return core(equal_risk(daily, gross, benchmark, margined))["cagr"] - core(held)["cagr"]


def scorecard(daily: pd.Series, *, trades: pd.DataFrame | None = None, exposure: pd.Series | None = None,
              benchmark: pd.Series | None = None, cash: bool = False, margined: bool = False,
              portfolio: bool = False) -> dict:
    """Everything the firm's target is judged on, plus the buy-and-hold comparison: `exposure` is the exposure the
    record held through each bar (what the money test leaves idle, or borrowed), `benchmark` is buy-and-hold's daily
    returns, `cash` says there was nothing to hold (`engine.hold.nothing_to_hold`), its returns are zero and the money
    test is against T-bills; `margined` says the positions are futures (`engine.hold.margined`), whose money stays in
    T-bills. `portfolio` judges the green target on all months (a portfolio of capital) instead of the
    months with a position (a strategy)."""
    card = core(daily)
    card.update(trade_stats(trades, card["months"]))
    card.update(exposure_stats(exposure))
    if benchmark is not None:
        bench = benchmark.reindex(daily.index).fillna(0.0)
        bh = core(bench)
        card.update({f"bh_{k}": bh[k] for k in ("avg_monthly", "pct_green", "sharpe", "max_dd", "cagr")})
        card["bh_cash"] = cash
        card["vs_bh"] = vs_hold(daily, daily_gross(exposure, daily.index), bench, cash, margined)
        card["beats_bh"] = bool(card["vs_bh"] > 0)
    card["targets"] = targets(card, portfolio)
    card["targets_met"] = int(sum(card["targets"].values()))
    return card


def targets(card: dict, portfolio: bool = False) -> dict[str, bool]:
    t = FIRM_TARGETS
    green = card.get("pct_green", np.nan) if portfolio else card.get("pct_green_active", np.nan)
    return {
        "avg_monthly>=1.5%": bool(card.get("avg_monthly", np.nan) >= t["avg_monthly"]),
        "green_months>=70%": bool(green >= t["pct_green"]),
        "max_dd>=-10%": bool(card.get("max_dd", np.nan) >= t["max_dd"]),
        "sharpe>=1": bool(card.get("sharpe", np.nan) >= t["sharpe"]),
        "beats_buy_and_hold": bool(card.get("beats_bh", False)),
    }
