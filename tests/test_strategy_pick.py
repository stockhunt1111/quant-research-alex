"""The choice of a strategy for an instrument: the walk-forward that chooses parameters, one level up, on the past
only, paying the trade each switch causes."""
import numpy as np
import pandas as pd

from strategy_lab import strategy_pick, walkforward
from strategy_lab.config import WF_SCHEMES

DAYS = pd.date_range("2020-01-01", periods=4 * 365, freq="D", tz="UTC")


def _candidate(name, returns, position=1.0, start=0):
    idx = DAYS[start:]
    r = pd.Series(returns[start:], index=idx)
    return {"id": hash(name) % 1000, "strategy": name, "oos": r, "position": pd.Series(position, index=idx)}


def _two_regimes():
    rng = np.random.default_rng(4)
    half = len(DAYS) // 2
    a = np.r_[rng.normal(0.002, 0.01, half), rng.normal(-0.002, 0.01, len(DAYS) - half)]
    b = np.r_[rng.normal(-0.001, 0.01, half), rng.normal(0.003, 0.01, len(DAYS) - half)]
    return [_candidate("a", a, 1.0), _candidate("b", b, 0.5)]


def test_each_window_takes_the_candidate_best_on_all_the_days_before_it():
    cands = _two_regimes()
    record, position, windows, choices = strategy_pick.choose(cands, "1d", cost=0.0)
    scheme = WF_SCHEMES["1d"]
    folds = walkforward.folds(DAYS[0], DAYS[-1] + pd.Timedelta(days=1), scheme["first_train"], scheme["test"],
                              scheme["train"])
    assert len(windows) == len(folds)
    for f, c in zip(folds, choices):              # the same choice worked out plainly with pandas, window by window
        past = [x["oos"][(x["oos"].index >= f.train_start) & (x["oos"].index < f.train_end)] for x in cands]
        sharpe = [p.mean() / p.std(ddof=1) for p in past]
        assert c == int(np.argmax(sharpe))
    assert {w["params"]["strategy"] for w in windows} == {"a", "b"}       # the regimes change: so does the choice
    held = pd.concat([x["oos"] for x in cands], axis=1)
    for f, c in zip(folds, choices):
        span = (record.index >= f.test_start) & (record.index < f.test_end)
        assert np.allclose(record[span], held.loc[record.index[span], c])


def test_a_switch_pays_the_cost_of_the_difference_between_the_two_positions():
    cands = _two_regimes()
    free, _, _, choices = strategy_pick.choose(cands, "1d", cost=0.0)
    paid, _, windows, _ = strategy_pick.choose(cands, "1d", cost=0.001)
    switches = [pd.Timestamp(w["test_start"], tz="UTC") for w, prev, cur in zip(windows[1:], choices[:-1], choices[1:])
                if prev != cur]
    assert switches
    for day in switches:                          # a (position 1) to b (position 0.5) or back: half the capital traded
        assert np.isclose(1 + paid[day], (1 + free[day]) * (1 - 0.001 * 0.5))
    assert np.allclose(paid.drop(switches), free.drop(switches))


def test_no_choice_looks_ahead():
    cands = _two_regimes()
    _, _, whole, _ = strategy_pick.choose(cands, "1d", cost=0.001)
    cut = DAYS[3 * 365]
    short = [dict(c, oos=c["oos"][c["oos"].index < cut], position=c["position"][c["position"].index < cut])
             for c in cands]
    _, _, part, _ = strategy_pick.choose(short, "1d", cost=0.001)
    ended = [w for w in part if pd.Timestamp(w["test_end"], tz="UTC") < cut]
    assert ended and [w["params"] for w in ended] == [w["params"] for w in whole[:len(ended)]]


def test_a_candidate_without_positions_on_60_days_of_the_past_is_not_chosen_and_none_left_holds_nothing():
    rng = np.random.default_rng(5)
    idle = _candidate("idle", np.zeros(len(DAYS)), 0.0)
    late = _candidate("late", rng.normal(0.003, 0.01, len(DAYS)), 1.0, start=len(DAYS) - 100)
    record, position, windows, choices = strategy_pick.choose([idle, late], "1d", cost=0.0)
    assert choices[0] is None and windows[0]["params"] == {"strategy": None}
    assert (record[record.index < pd.Timestamp(windows[1]["test_start"], tz="UTC")] == 0).all()
