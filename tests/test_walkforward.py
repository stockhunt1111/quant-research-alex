import numpy as np
import pandas as pd

from strategy_lab import walkforward as wf

DAYS = pd.date_range("2020-01-01", periods=900, freq="D", tz="UTC")


def _series(rng, mean, n=len(DAYS)):
    return rng.normal(mean, 0.01, n)


def _folds():
    return wf.folds(DAYS.min(), DAYS.max() + pd.Timedelta(days=1), "365D", "91D")


def _peers(rng, n, first, second, hold):
    """n peers priced every day, on which configuration 0 earns `first`, configuration 1 `second` and holding `hold` a
    day; `hold` None: nothing to hold, cash."""
    held = (lambda: pd.Series(0.0, index=DAYS)) if hold is None else (lambda: pd.Series(_series(rng, hold), index=DAYS))
    return [(pd.DataFrame({0: _series(rng, first), 1: _series(rng, second)}, index=DAYS), held(),
             pd.Series(True, index=DAYS)) for _ in range(n)]


def test_a_rolling_window_chooses_on_the_last_year_where_an_expanding_one_chooses_on_all_the_history():
    end = DAYS.max() + pd.Timedelta(days=1)
    grown, rolled = _folds(), wf.folds(DAYS.min(), end, "365D", "91D", train="365D")
    assert [(f.test_start, f.test_end) for f in rolled] == [(f.test_start, f.test_end) for f in grown]
    assert rolled[0] == grown[0] and all(f.train_start == DAYS.min() for f in grown)
    assert all(f.train_end == f.test_start and f.train_end - f.train_start == pd.Timedelta("365D") for f in rolled)
    # a configuration that led in the first year only, against a steady one: all the history still prefers the former
    rng = np.random.default_rng(0)
    faded = np.where(np.arange(len(DAYS)) < 365, 0.004, -0.001) + rng.normal(0, 0.002, len(DAYS))
    steady = 0.0005 + rng.normal(0, 0.002, len(DAYS))
    daily = pd.DataFrame({0: faded, 1: steady}, index=DAYS)
    assert wf.choose(daily, grown)[-1][0] == 0 and wf.choose(daily, rolled)[-1][0] == 1


def test_the_choice_is_the_best_of_the_configurations_that_beat_holding_on_most_peers():
    rng = np.random.default_rng(1)
    own = pd.DataFrame({0: _series(rng, 0.003), 1: _series(rng, 0.001)}, index=DAYS)    # 0 is the better one here
    peers = _peers(rng, 5, first=-0.002, second=0.003, hold=0.0005)                        # ... and loses on every peer
    assert {c for c, _ in wf.choose(own, _folds())} == {0}
    picks = wf.choose_with_peers(own, _folds(), peers)
    assert {c for c, _, _ in picks} == {1} and {j for _, _, j in picks} == {5}


def test_a_window_where_no_configuration_passes_holds_nothing():
    rng = np.random.default_rng(2)
    own = pd.DataFrame({0: _series(rng, 0.003), 1: _series(rng, 0.001)}, index=DAYS)
    picks = wf.choose_with_peers(own, _folds(), _peers(rng, 4, first=-0.002, second=-0.002, hold=0.002))
    assert {c for c, _, _ in picks} == {None}
    assert (wf.stitch(own, _folds(), [c for c, _, _ in picks]) == 0).all()


def test_with_fewer_than_three_peers_with_history_the_check_is_not_made():
    rng = np.random.default_rng(3)
    own = pd.DataFrame({0: _series(rng, 0.003), 1: _series(rng, 0.001)}, index=DAYS)
    peers = _peers(rng, 2, first=-0.002, second=0.003, hold=0.0005)
    short = _peers(rng, 3, first=-0.002, second=0.003, hold=0.0005)
    for d, b, priced in short:                           # peers that only start after the last window: never judged
        d.iloc[:] = 0.0
        b.iloc[:] = 0.0
        priced.iloc[:] = False
    picks = wf.choose_with_peers(own, _folds(), peers + short)
    assert [c for c, _, _ in picks] == [c for c, _ in wf.choose(own, _folds())]
    assert {j for _, _, j in picks} == {0}


def test_a_peer_held_as_cash_is_judged_and_a_configuration_beats_it_by_making_money_there():
    rng = np.random.default_rng(5)
    own = pd.DataFrame({0: _series(rng, 0.003), 1: _series(rng, 0.001)}, index=DAYS)
    picks = wf.choose_with_peers(own, _folds(), _peers(rng, 4, first=-0.002, second=0.002, hold=None))
    assert {c for c, _, _ in picks} == {1} and {j for _, _, j in picks} == {4}
    picks = wf.choose_with_peers(own, _folds(), _peers(rng, 4, first=-0.002, second=-0.002, hold=None))
    assert {c for c, _, _ in picks} == {None}


def test_a_window_sharpe_from_running_sums_is_the_walk_forward_sharpe():
    rng = np.random.default_rng(4)
    x = rng.normal(0.001, 0.01, (500, 3))
    x[:40, 2] = 0.0
    s1, s2, n = wf._cumulative(x)
    for a, z in ((0, 500), (37, 211), (100, 101), (0, 40)):
        expected = [wf.sharpe(pd.Series(x[a:z, c])) for c in range(3)]
        np.testing.assert_allclose(wf._window_sharpe(s1, s2, n, a, z), expected, rtol=1e-9)


def test_a_window_before_which_no_configuration_traded_enough_has_no_choice_and_holds_nothing():
    rng = np.random.default_rng(3)
    trades = np.arange(len(DAYS)) >= 400                                  # both configurations trade from day 400
    daily = pd.DataFrame({0: np.where(trades, rng.normal(0.001, 0.01, len(DAYS)), 0.0),
                          1: np.where(trades, rng.normal(0.0005, 0.01, len(DAYS)), 0.0)}, index=DAYS)
    fl = _folds()
    picks = wf.choose(daily, fl)
    # the first window's year held nothing, the second's 56 days of trading: too few to judge; then a choice
    assert [c for c, _ in picks[:2]] == [None, None] and all(s == -np.inf for _, s in picks[:2])
    assert all(c is not None for c, _ in picks[2:])
    oos = wf.stitch(daily, fl, [c for c, _ in picks])
    for f, (c, _) in zip(fl, picks):
        if c is None:
            assert (oos[(oos.index >= f.test_start) & (oos.index < f.test_end)] == 0).all()
