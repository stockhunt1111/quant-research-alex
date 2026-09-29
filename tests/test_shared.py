import numpy as np
import pandas as pd

from strategy_lab import regimes
from strategy_lab.shared import shared, sharing
from tests.conftest import make_panel


def _counted(calls: list):
    @shared
    def mean(close, n):
        calls.append(n)
        return close.rolling(n).mean()

    @shared
    def above(line, level):
        calls.append(level)
        return line > level
    return mean, above


def test_a_result_is_worked_out_once_for_the_calls_that_ask_for_it_and_each_gets_its_own_copy():
    bars = make_panel(n=300, seed=3).one("td:AAA")
    calls: list = []
    mean, _ = _counted(calls)
    with sharing(bars):
        first = mean(bars.close, 10)
        first.iloc[50:60] = -1.0                           # a configuration may change what it was handed
        again, third = mean(bars["close"], 10), mean(bars.close, 10)
    assert calls == [10]
    expected = bars.close.rolling(10).mean()
    pd.testing.assert_series_equal(again, expected)
    pd.testing.assert_series_equal(third, expected)
    assert again is not third


def test_a_result_changed_by_its_caller_is_worked_out_on_what_it_holds_now():
    bars = make_panel(n=300, seed=3).one("td:AAA")
    calls: list = []
    mean, above = _counted(calls)
    with sharing(bars):
        line = mean(bars.close, 10)
        untouched = above(line, 1e6)
        line.iloc[100:120] = 1e9
        changed = above(line, 1e6)
        again = above(mean(bars.close, 10), 1e6)           # a fresh copy of the mean: the first result, shared
    assert calls == [10, 1e6, 1e6]
    pd.testing.assert_series_equal(changed, line > 1e6)
    pd.testing.assert_series_equal(again, untouched)
    assert changed.iloc[100:120].all() and not untouched.any()


def test_an_input_it_cannot_recognise_and_a_call_outside_sharing_are_worked_out_every_time():
    bars = make_panel(n=300, seed=3).one("td:AAA")
    calls: list = []
    mean, above = _counted(calls)
    with sharing(bars):
        mean(bars.close * 1.0, 10)
        mean(bars.close * 1.0, 10)                         # the same values, but not a column of the bars
        above(bars.close, 100)
        above(bars.close, 100.0)                           # 100 and 100.0 are not taken for one another
    mean(bars.close, 10)
    mean(bars.close, 10)
    assert calls == [10, 10, 100, 100.0, 10, 10]


def test_a_regime_confirmed_on_a_shared_regime_is_the_one_worked_out_without_sharing():
    bars = make_panel(n=900, seed=4).one("td:AAA")
    alone = {k: regimes.confirmed(regimes.vol_state(bars.close), k) for k in (3, 5, 3)}
    with sharing(bars):
        together = {k: regimes.confirmed(regimes.vol_state(bars.close), k) for k in (3, 5, 3)}
    for k in alone:
        assert np.array_equal(alone[k].to_numpy(), together[k].to_numpy(), equal_nan=True)
        assert alone[k].index.equals(together[k].index)
