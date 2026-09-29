import numpy as np
import pandas as pd

import strategies.ml_direction as ml
from strategies.big_move_follow import big_move_follow
from strategy_lab import lookahead, regimes
from strategy_lab.data.bars import Panel
from tests.conftest import make_panel


def test_confirmed_state_waits_k_bars():
    s = pd.Series([0, 1, 1, 0, 1, 1, 1, 1], dtype=float)
    out = regimes.confirmed(s, 3)
    assert list(out.fillna(-9)) == [-9, -9, -9, -9, -9, -9, 1, 1]


def test_switch_takes_the_position_of_the_active_state():
    state = pd.Series([1.0, 0.0, -1.0, 1.0])
    trend = pd.Series([1.0, 1.0, 1.0, 0.5])
    revert = pd.Series([0.2, 0.3, 0.4, 0.5])
    out = regimes.switch(state, {1.0: trend, 0.0: revert})
    assert list(out) == [1.0, 0.3, 0.0, 0.5]


def test_ml_model_is_refit_on_the_past_only():
    p = make_panel(ids=("td:AAA",), n=1300, seed=31)
    small = ml.ml_direction.__class__(ml.ml_direction.name, "rule", ml.ml_direction.fn,
                                       {"horizon": [5], "min_train": [300], "threshold": [0.5], "long_only": [True]})
    lookahead.check(small, p, cuts=(0.6, 0.85))
    assert small.target(p, small.configs()[0]).abs().sum().sum() > 0      # it does trade after min_train


def test_the_models_fitted_in_parallel_before_the_grid_give_the_same_positions():
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), n=900, seed=37)
    grid = {"horizon": [5], "min_train": [300], "threshold": [0.5], "long_only": [True]}
    small = ml.ml_direction.__class__(ml.ml_direction.name, "rule", ml.ml_direction.fn, grid, prepare=ml.prepare)
    ml._PREDICTIONS.clear()
    one_by_one = small.target(p, small.configs()[0])            # each instrument fitted in the rule, in turn
    ml._PREDICTIONS.clear()
    small.prepare(p, small.configs())                            # all of them at once, one process per core
    assert len(ml._PREDICTIONS) == 3
    pd.testing.assert_frame_equal(small.target(p, small.configs()[0]), one_by_one)
    ml._PREDICTIONS.clear()


def _leaving_across_a_refit(bars: pd.DataFrame, held: pd.Series) -> int:
    """A bar to leave the list at while a trade is held that is still open at the next refit and ends before the one
    after: its seat is kept into a block of bars the list no longer holds the name for."""
    from strategy_lab.ml import refit_rows
    starts = [t for t in refit_rows(bars.index) if t >= 200]
    for a, b in zip(starts[2:], starts[3:]):
        for k in range(a - 30, a):
            if held.iloc[k - 1:a + 1].all() and not held.iloc[a:b].all():       # open before it leaves
                return k
    raise AssertionError("no trade held across a refit on this walk")


def test_models_fitted_only_where_a_list_reads_them_give_the_same_positions_as_fitting_everything(monkeypatch):
    p = make_panel(ids=("td:AAA", "td:BBB", "td:CCC"), n=1200, seed=43)
    grid = {"horizon": [5], "min_train": [200], "threshold": [0.5, 0.55], "long_only": [True]}
    small = ml.ml_direction.__class__(ml.ml_direction.name, "rule", ml.ml_direction.fn, grid, prepare=ml.prepare)
    bars = p.one("td:BBB")
    proba = ml.predict_expanding(ml.features(bars), ml._label(bars, 5), 5, 200)
    leave = _leaving_across_a_refit(bars, proba > 0.55)
    member = pd.DataFrame(False, index=p.index, columns=p.ids)
    member.iloc[300:500, 0] = True                   # in the list for a while, out, and back near the end
    member.iloc[1000:, 0] = True
    member.iloc[250:leave, 1] = True                 # leaves in the middle of a trade, which keeps its seat
    member.iloc[:, 2] = True                         # always in
    monkeypatch.setattr(ml, "_PREDICTIONS", {})
    everything = [small.target(p, c, member) for c in small.configs()]      # every block fitted, in the rule
    assert all(full["td:BBB"].iloc[leave] > 0 for full in everything)          # the kept trade runs past the list
    whole = dict(ml._PREDICTIONS)
    monkeypatch.setattr(ml, "_PREDICTIONS", {})
    undo = small.prepare(p, small.configs(), member)
    for c, full in zip(small.configs(), everything):
        pd.testing.assert_frame_equal(small.target(p, c, member), full)
    part = dict(ml._PREDICTIONS)
    by_fit = {k[:-1]: v for k, v in whole.items()}   # the same fits, kept under the list's scope instead of every block
    assert {k[:-1] for k in part} == by_fit.keys() and all(k[-1] is not None for k in part)
    for k, proba in part.items():                    # a block fitted either way predicts the same
        fitted = proba.notna()
        pd.testing.assert_series_equal(proba[fitted], by_fit[k[:-1]][fitted])
    assert sum(v.notna().sum() for v in part.values()) < sum(v.notna().sum() for v in whole.values())
    undo()
    assert not ml._PREDICTIONS and ml._SCOPE is None     # nor kept for an evaluation with another membership


def test_a_short_kept_past_its_list_has_its_blocks_fitted_as_a_long_does(monkeypatch):
    p = make_panel(ids=("td:AAA", "td:BBB"), n=1200, seed=44)
    grid = {"horizon": [5], "min_train": [200], "threshold": [0.55], "long_only": [False]}
    small = ml.ml_direction.__class__(ml.ml_direction.name, "rule", ml.ml_direction.fn, grid, prepare=ml.prepare)
    bars = p.one("td:AAA")
    proba = ml.predict_expanding(ml.features(bars), ml._label(bars, 5), 5, 200)
    leave = _leaving_across_a_refit(bars, proba < 0.45)
    member = pd.DataFrame(True, index=p.index, columns=p.ids)
    member.iloc[leave:, 0] = False                   # leaves while short: the short keeps its seat until it ends
    monkeypatch.setattr(ml, "_PREDICTIONS", {})
    everything = small.target(p, small.configs()[0], member)
    assert everything["td:AAA"].iloc[leave] < 0
    monkeypatch.setattr(ml, "_PREDICTIONS", {})
    undo = small.prepare(p, small.configs(), member)
    pd.testing.assert_frame_equal(small.target(p, small.configs()[0], member), everything)
    undo()


def test_a_neighbouring_list_fitted_inside_a_list_leaves_the_list_its_own_fits(monkeypatch):
    p = make_panel(ids=("td:AAA", "td:BBB"), n=900, seed=47)
    grid = {"horizon": [5], "min_train": [300], "threshold": [0.5], "long_only": [True]}
    small = ml.ml_direction.__class__(ml.ml_direction.name, "rule", ml.ml_direction.fn, grid, prepare=ml.prepare)
    listed = pd.DataFrame(False, index=p.index, columns=p.ids)
    listed.iloc[400:, 0] = True
    other = listed.copy()
    other.iloc[400:, 1] = True
    monkeypatch.setattr(ml, "_PREDICTIONS", {})
    undo = small.prepare(p, small.configs(), listed)
    own = small.target(p, small.configs()[0], listed)
    inner = small.prepare(p, small.configs(), other)        # a neighbouring list, evaluated inside this one
    assert small.target(p, small.configs()[0], other)["td:BBB"].abs().sum() > 0     # its own blocks were fitted
    inner()
    pd.testing.assert_frame_equal(small.target(p, small.configs()[0], listed), own)
    undo()
    assert not ml._PREDICTIONS and ml._SCOPE is None


def test_another_seed_reaches_the_processes_that_fit_and_is_not_kept(monkeypatch):
    p = make_panel(ids=("td:AAA", "td:BBB"), n=900, seed=53)
    grid = {"horizon": [5], "min_train": [300], "threshold": [0.5], "long_only": [True]}
    small = ml.ml_direction.__class__(ml.ml_direction.name, "rule", ml.ml_direction.fn, grid, prepare=ml.prepare)
    monkeypatch.setattr(ml, "_PREDICTIONS", {})
    monkeypatch.setattr(ml, "SEED", 3)
    undo = small.prepare(p, small.configs())                  # two instruments: fitted in a pool of processes
    pooled = small.target(p, small.configs()[0])
    undo()
    assert not ml._PREDICTIONS                                # a fit with another seed than the configured one goes
    in_process = small.target(p, small.configs()[0])         # fitted here, with the module's seed 3
    pd.testing.assert_frame_equal(pooled, in_process)
    monkeypatch.setattr(ml, "_PREDICTIONS", {})
    monkeypatch.setattr(ml, "SEED", 7)
    bars = p.one("td:AAA")
    seven = ml.predict_expanding(ml.features(bars), ml._label(bars, 5), 5, 300)
    three = ml.predict_expanding(ml.features(bars), ml._label(bars, 5), 5, 300, seed=3)
    assert not seven.dropna().equals(three.dropna())


def test_big_move_book_respects_capital_and_is_causal():
    ids = tuple(f"td:S{k}" for k in range(12))
    p = make_panel(ids=ids, n=700, seed=41)
    cfg = {"z": 2.0, "vol_mult": 0.5, "hold": 5, "max_positions": 3}
    w = big_move_follow.target(p, cfg)
    assert (w.abs().sum(axis=1) <= 1 + 1e-9).all() and w.to_numpy().max() > 0
    lookahead.check(big_move_follow, p, cfg)
    assert np.isclose(w[w > 0].min().min(), 1 / 3)


def test_big_move_without_a_recorded_volume_is_judged_on_the_move_alone():
    """FX pairs and spot metals carry no traded volume: the volume condition cannot hold there, and the rule never
    traded them. Without a volume the events are the price moves the same rule takes where every bar's volume passes."""
    ids = tuple(f"td:S{k}" for k in range(6))
    p = make_panel(ids=ids, n=700, seed=41)                       # the same volume on every bar
    silent = Panel(p.timeframe, p.instruments, p.open, p.high, p.low, p.close, p.volume * 0.0, p.dollar_volume * 0.0)
    strict = {"z": 2.0, "vol_mult": 3.0, "hold": 5, "max_positions": 3}
    assert (big_move_follow.target(p, strict) == 0).all().all()    # a flat volume never triples: no event where recorded
    on_moves = big_move_follow.target(p, {**strict, "vol_mult": 0.5})          # every bar's volume passes
    assert on_moves.to_numpy().max() > 0
    pd.testing.assert_frame_equal(big_move_follow.target(silent, strict), on_moves)
    lookahead.check(big_move_follow, silent, strict)


def test_ml_predictions_are_cached_per_instrument_data_not_per_parameters(monkeypatch):
    monkeypatch.setattr(ml, "_PREDICTIONS", {})
    p = make_panel(ids=("td:AAA", "td:BBB"), n=700, seed=41)
    for inst in p.ids:
        bars = p.one(inst)
        X = ml.features(bars)
        y = (bars["close"].shift(-5) > bars["close"]).astype(float).where(bars["close"].shift(-5).notna())
        cached = ml._cached_predictions(X, y, 5, 300)
        pd.testing.assert_series_equal(cached, ml.predict_expanding(X, y, 5, 300))
    assert len(ml._PREDICTIONS) == 2
