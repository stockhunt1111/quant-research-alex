import numpy as np
import pandas as pd

from strategy_lab import portfolio


def _rets(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"a": rng.normal(0.001, 0.01, n), "b": rng.normal(0.0, 0.03, n), "c": rng.normal(-0.001, 0.01, n)},
                        index=idx)


def test_weights_sum_to_one_are_fixed_within_a_month_and_use_only_the_past():
    r = _rets()
    w = portfolio.monthly_weights(r, "equal_risk")
    live = w[w.sum(axis=1) > 0]
    assert np.allclose(live.sum(axis=1), 1.0)
    month = live.index.tz_localize(None).to_period("M")
    assert (live.groupby(month).nunique() <= 1).all().all()
    changed = r.copy()
    changed.iloc[400:] = changed.iloc[400:] * -5                    # rewrite the future
    w2 = portfolio.monthly_weights(changed, "equal_risk")
    first_changed_month_start = w.index[400:][w.index[400:].day == 1][0]
    assert np.allclose(w.loc[:first_changed_month_start - pd.Timedelta(days=1)],
                       w2.loc[:first_changed_month_start - pd.Timedelta(days=1)])


def test_equal_risk_gives_less_capital_to_the_volatile_strategy():
    w = portfolio.monthly_weights(_rets(), "equal_risk")
    last = w.iloc[-1]
    assert last["b"] < last["a"] and abs(last["a"] - last["c"]) < 0.15


def test_trailing_sharpe_selection_drops_a_losing_strategy():
    r = _rets(seed=1)
    r["c"] = -0.002 + r["c"] * 0.1                                  # a steady loser
    w = portfolio.monthly_weights(r, "equal", select_trailing_sharpe=120)
    assert (w["c"].iloc[-200:] == 0).all()


def test_a_months_split_is_held_through_the_month_each_strategys_money_growing_with_it():
    idx = pd.date_range("2024-01-01", periods=31, freq="D", tz="UTC")
    rets = pd.DataFrame(0.0, index=idx, columns=["a", "b"])
    rets.iloc[1] = [0.10, 0.0]
    rets.iloc[2] = [0.05, -0.05]
    port = portfolio.held_returns(rets, pd.DataFrame(0.5, index=idx, columns=["a", "b"]))
    # by hand: half in each; a +10% makes 0.55 and 0.5; then +5% and -5%: 0.5775 and 0.475, the book 1.0525
    assert np.isclose(port.iloc[1], 0.05) and np.isclose(port.iloc[2], 1.0525 / 1.05 - 1)   # daily rebalanced: 0
    assert np.isclose((1 + port).prod(), 1.0525)
    nxt = pd.date_range("2024-02-01", periods=3, freq="D", tz="UTC")
    both = pd.concat([rets, pd.DataFrame([[0.10, 0.0]] * 3, index=nxt, columns=["a", "b"])])
    w = pd.DataFrame(0.5, index=both.index, columns=["a", "b"])
    w.loc[nxt, "a"] = 0.25                                   # February's split: a quarter and a half, a quarter in cash
    feb = portfolio.held_returns(both, w).loc[nxt]
    assert np.isclose(feb.iloc[0], 0.25 * 0.10)             # the new split from its first day, from a month worth one
