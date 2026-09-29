import numpy as np
import pandas as pd

from strategy_lab import metrics, significance
from strategy_lab.engine import backtest as bt
from strategies.ibs_reversion import ibs_reversion
from tests.conftest import make_panel


def _days(panel):
    first = (panel.index[0] - pd.Timedelta(microseconds=1)).normalize()
    last = (panel.index[-1] - pd.Timedelta(microseconds=1)).normalize()
    return pd.date_range(first, last, freq="D", tz="UTC")


def test_the_noise_bar_is_what_the_best_of_as_many_worthless_tries_reaches():
    rng = np.random.default_rng(3)
    n_trials, n_days = 200, 1000
    tries = rng.normal(0, 0.01, (150, n_trials, n_days))
    sharpe = tries.mean(axis=2) / tries.std(axis=2, ddof=1) * np.sqrt(metrics.DAYS)
    one = pd.Series(tries[0, 0], index=pd.date_range("2020-01-01", periods=n_days, freq="D", tz="UTC"))
    assert np.isclose(metrics.core(one)["sharpe"], sharpe[0, 0])          # the Sharpe the scorecard reports
    bar = significance.deflated(one, n_trials)["noise_bar"]
    assert abs(sharpe.max(axis=1).mean() - bar) < 0.05 * bar


def test_the_more_tries_a_record_was_picked_from_the_less_likely_it_is_more_than_luck():
    rng = np.random.default_rng(4)
    record = pd.Series(rng.normal(0.0008, 0.01, 1500))            # a true daily Sharpe of 0.08, about 1.5 a year
    probs = [significance.deflated(record, n)["prob"] for n in (1, 10, 100, 1000, 10000)]
    assert probs[0] > 0.95 and all(a > b for a, b in zip(probs, probs[1:]))


def test_holding_all_along_has_no_timing_for_a_rotation_to_change():
    p = make_panel(ids=("td:AAA",), n=500, seed=1)
    w = pd.DataFrame(1.0, index=p.index, columns=p.ids)
    out = significance.random_timing(p, w, _days(p), n=200)
    assert out["p"] == 1.0 and out["null_median"] == out["sharpe"]


def test_positions_that_know_the_next_session_beat_every_rotation():
    p = make_panel(ids=("td:AAA", "td:BBB"), n=600, seed=2)
    session = p.close / p.open - 1.0
    w = (session > 0).astype(float) / 2                         # held on exactly the bars that rise: impossible foresight
    out = significance.random_timing(p, w, _days(p), n=200)
    assert out["p"] == 1 / 201 and out["sharpe"] > out["null_p95"]


def test_positions_that_know_nothing_are_rarely_called_timing():
    p = make_panel(ids=("td:AAA", "td:BBB"), n=500, seed=3)
    rng = np.random.default_rng(5)
    significant = 0
    for _ in range(40):
        runs = np.repeat(rng.integers(0, 2, len(p.index) // 5 + 1), 5)[:len(p.index)]
        w = pd.DataFrame(np.column_stack([runs, runs[::-1]]) / 2, index=p.index, columns=p.ids)
        significant += significance.random_timing(p, w, _days(p), n=200)["p"] < significance.PASS_P
    assert significant <= 6                                     # about 2 of 40 expected by chance


def test_the_simple_valuation_follows_the_engine_on_a_strategy_without_intrabar_exits():
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), n=700, seed=4)
    res = bt.run(p, ibs_reversion.target(p, {"entry": 0.3, "long_only": True}))
    engine = metrics.core(metrics.daily_returns(res.returns))["sharpe"]
    simple = significance.random_timing(p, res.weights, _days(p), n=20)["sharpe"]
    assert abs(simple - engine) < 0.05
