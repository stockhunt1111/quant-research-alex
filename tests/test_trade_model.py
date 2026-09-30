import collections
import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from strategy_lab import config, lookahead, ml, trade_model
from strategy_lab.data.bars import FIELDS, Panel
from strategy_lab.engine import backtest as bt
from strategy_lab.engine.trades import ledger
from strategy_lab.strategy import load
from strategies.ibs import ibs
from strategies.ibs_ml_filter import ibs_ml_filter
from tests.conftest import make_panel

DAY = 86_400 * 10**9
FOUR = ("td:AAA", "td:BBB", "td:CCC", "td:DDD")      # enough IBS trades for the model to be refit within 800 bars


def _wanted(p):
    cfg = ibs.configs()[0]
    return pd.DataFrame({i: ibs.fn(p.one(i), **ibs.signal_params(cfg)) for i in p.ids}).reindex(p.index).ffill().fillna(0.0)


def test_a_trade_is_the_rule_trade_the_engine_fills_entered_and_left_at_the_next_opens():
    p = make_panel(ids=("td:AAA", "td:BBB"), n=500, seed=7)
    t, _ = trade_model.trades(p, _wanted(p), p.started)
    res = bt.run(p, ibs.target(p, ibs.configs()[0]), fill="next_open")
    book = ledger(p, res)
    book = book[book["exit_reason"] == "signal"]
    rate = (config.COSTS["us_equity"].commission_bps + config.COSTS["us_equity"].half_spread_bps) / 1e4
    closed = t.dropna(subset=["net"])
    for inst in p.ids:
        ours = np.sort(closed.loc[closed["instrument"] == inst, "net"].to_numpy() + 2 * rate)
        engine = np.sort(book.loc[book["instrument"] == inst, "gross_return"].to_numpy())
        assert len(ours) == len(engine) > 20
        np.testing.assert_allclose(ours, engine, rtol=1e-12, atol=1e-12)


def _market(n, seed, planted):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    signal = x[:, 0] + (rng.normal(scale=0.7, size=n) if planted else rng.normal(scale=1e6, size=n))
    decided = np.arange(n, dtype=np.int64) * DAY
    return x, (signal > 0).astype(float), decided, decided + 3 * DAY, decided[::400]


def _auc(p, y):
    ok = ~np.isnan(p)
    return roc_auc_score(y[ok], p[ok])


def test_the_model_finds_an_edge_in_the_trades_and_nothing_in_shuffled_outcomes():
    x, y, decided, known, refits = _market(6000, 1, planted=True)
    learn = np.ones(len(y), dtype=bool)
    assert _auc(trade_model.probabilities(x, y, decided, known, learn, refits, 7), y) > 0.75
    shuffled = trade_model.probabilities(x, y, decided, known, learn, refits, 7, shuffle=True)
    assert abs(_auc(shuffled, y) - 0.5) < 0.05


def test_a_model_that_saw_the_trades_it_grades_would_show_an_edge_in_pure_noise():
    x, y, decided, known, refits = _market(6000, 2, planted=False)
    learn = np.ones(len(y), dtype=bool)
    honest = trade_model.probabilities(x, y, decided, known, learn, refits, 7)
    assert abs(_auc(honest, y) - 0.5) < 0.05
    everything_known = np.full(len(y), np.iinfo(np.int64).min)       # each refit also learns from the trades it grades
    assert _auc(trade_model.probabilities(x, y, decided, everything_known, learn, refits, 7), y) > 0.6


def test_a_rejected_trade_is_not_taken_and_an_ungraded_one_neither():
    p = make_panel(ids=FOUR, n=800, seed=8)
    wanted = _wanted(p)
    t = trade_model.graded(p, wanted, p.started)
    assert t["p"].notna().any() and t["p"].isna().any()               # graded once a refit had enough closed trades
    assert (trade_model.take(p, wanted, p.started, threshold=1.0) == 0).all().all()
    kept = trade_model.take(p, wanted, p.started, threshold=0.0)
    col = {i: k for k, i in enumerate(p.ids)}
    for r in t.itertuples(index=False):
        held = kept.iloc[r.first:r.last + 1, col[r.instrument]]
        assert (held == (wanted.iloc[r.first:r.last + 1, col[r.instrument]] if pd.notna(r.p) else 0.0)).all()


def test_on_a_list_the_model_learns_only_from_trades_started_in_the_universe(monkeypatch):
    p = make_panel(ids=FOUR, n=800, seed=9)
    member = p.started.copy()
    member["td:BBB"] = False                                          # BBB never in the universe
    seen = []
    real = trade_model.probabilities

    def spy(x, outcome, decided, known, learn, *a, **k):
        seen.append(learn.copy())
        return real(x, outcome, decided, known, learn, *a, **k)

    monkeypatch.setattr(trade_model, "probabilities", spy)
    t = trade_model.graded(p, _wanted(p), member)
    assert len(seen) == 1 and (seen[0] == (t["instrument"] != "td:BBB").to_numpy()).all()


def test_the_graded_rule_passes_through_the_rule_framework_with_the_threshold_kept_from_the_rule():
    p = make_panel(ids=FOUR, n=800, seed=8)
    cfg = {"buy_below": 0.2, "sell_above": 0.8, "threshold": 0.0}
    assert ibs_ml_filter.signal_params(cfg) == {"buy_below": 0.2, "sell_above": 0.8}
    w = ibs_ml_filter.target(p, cfg)
    t = trade_model.graded(p, _wanted(p), p.started)
    ungraded = t[t["p"].isna()]
    for r in ungraded.itertuples(index=False):
        assert (w[r.instrument].iloc[r.first:r.last + 1] == 0).all()


@pytest.mark.parametrize("graded, plain", [
    ("ibs_ml_filter", "ibs"), ("ibs_ml_sized", "ibs"), ("rsi2_ml_filter", "rsi2_connors"),
    ("rsi2_ml_sized", "rsi2_connors"), ("bollinger_ml_filter", "bollinger_reversion"),
    ("bollinger_ml_sized", "bollinger_reversion"), ("trend_or_revert_ml_filter", "trend_or_revert"),
    ("trend_or_revert_ml_sized", "trend_or_revert"), ("long_ma_deviation_ml_filter", "long_ma_deviation"),
    ("long_ma_deviation_ml_sized", "long_ma_deviation")])
def test_a_graded_rule_grades_the_long_trades_of_the_rule_it_copies(graded, plain):
    """A graded rule is its rule written again, long only: each of its signal configurations gives the rule's long
    side bit for bit, so a change of the rule that misses its graded copies fails here."""
    g, r = load(graded), load(plain)
    long_only = {"long_only": True} if "long_only" in inspect.signature(r.fn).parameters else {}
    p = make_panel(n=600, seed=11)
    for sp in {repr(sorted(g.signal_params(c).items())): g.signal_params(c) for c in g.configs()}.values():
        for i in p.ids:
            pd.testing.assert_series_equal(g.fn(p.one(i), **sp), r.fn(p.one(i), **sp, **long_only))


def test_a_graded_rule_does_not_see_the_future():
    p = make_panel(ids=FOUR, n=800, seed=10)
    cfg = {"buy_below": 0.2, "sell_above": 0.8, "threshold": 0.5}
    assert (ibs_ml_filter.target(p, cfg).iloc[:int(len(p.index) * 0.8)] != 0).any().any()   # a model trades before the cut
    lookahead.check(ibs_ml_filter, p, cfg)


def test_a_rejected_trade_leaves_nothing_on_the_bars_its_instrument_missed_after_it():
    p = make_panel(ids=FOUR, n=800, seed=8)
    w = _wanted(p)
    k = int(np.flatnonzero((w["td:AAA"].to_numpy()[:-2] != 0) & (w["td:AAA"].to_numpy()[1:-1] == 0))[-1])
    for f in ("open", "high", "low", "close", "volume", "dollar_volume"):
        getattr(p, f).iloc[k + 1, 0] = np.nan                        # the bar that would have ended the trade is missed
    w = _wanted(p)
    assert w["td:AAA"].iloc[k + 1] != 0                               # the rule keeps its position through the gap
    assert (trade_model.take(p, w, p.started, threshold=1.0) == 0).all().all()


def _features_counted(monkeypatch) -> list:
    """The model's features, each time they are worked out recorded (the bars' last close); nothing kept from an
    earlier test."""
    monkeypatch.setattr(trade_model, "_FEATURES_AT", collections.OrderedDict())
    monkeypatch.setattr(trade_model, "_features_bytes", 0)
    worked_out: list = []
    real = ml.features
    monkeypatch.setattr(ml, "features", lambda bars: worked_out.append(float(bars["close"].iloc[-1])) or real(bars))
    return worked_out


def test_the_features_of_a_trade_are_kept_for_another_grading_of_it_on_the_same_bars(monkeypatch):
    """Another seed of the model and another list holding the name grade the same trades on the same bars: the
    features of their first bars are worked out once. Bars written anew are other bars."""
    real = ml.features
    worked_out = _features_counted(monkeypatch)
    p = make_panel(ids=FOUR, n=800, seed=8)
    wanted = _wanted(p)
    t, x = trade_model.trades(p, wanted, p.started)
    assert len(worked_out) == len(p.ids)

    def direct(panel, table, inst):
        own = np.flatnonzero(panel.close[inst].notna().to_numpy())
        at = np.searchsorted(own, table.loc[table["instrument"] == inst, "first"].to_numpy())
        return pd.concat(real(panel.one(inst)).values(), axis=1).to_numpy(dtype=np.float64)[at]

    for inst in p.ids:
        assert np.array_equal(x[(t["instrument"] == inst).to_numpy()], direct(p, t, inst), equal_nan=True)
    _, again = trade_model.trades(p, wanted, p.started)                  # another seed of the model
    two = ["td:AAA", "td:CCC"]
    other_list = Panel(p.timeframe, {i: p.instruments[i] for i in two}, **{f: getattr(p, f)[two] for f in FIELDS})
    _, on_other = trade_model.trades(other_list, wanted[two], other_list.started)
    assert len(worked_out) == len(p.ids)
    assert np.array_equal(again, x, equal_nan=True)
    assert np.array_equal(on_other, x[t["instrument"].isin(two).to_numpy()], equal_nan=True)

    rewritten = Panel(p.timeframe, dict(p.instruments), **{f: getattr(p, f).copy() for f in FIELDS})
    rewritten.close.iloc[400, 1] *= 1.01                                 # one close of BBB corrected by its vendor
    t3, x3 = trade_model.trades(rewritten, wanted, rewritten.started)
    assert len(worked_out) == len(p.ids) + 1                             # BBB's worked out again, on its new bars
    bbb = (t3["instrument"] == "td:BBB").to_numpy()
    assert np.array_equal(x3[bbb], direct(rewritten, t3, "td:BBB"), equal_nan=True)
    assert not np.array_equal(x3[bbb], x[(t["instrument"] == "td:BBB").to_numpy()], equal_nan=True)


def test_the_features_kept_stay_within_their_bytes_the_longest_unused_dropped_first(monkeypatch):
    worked_out = _features_counted(monkeypatch)
    p = make_panel(ids=FOUR, n=800, seed=8)
    wanted = _wanted(p)
    t, x = trade_model.trades(p, wanted, p.started)
    sizes = [v.nbytes for v in trade_model._FEATURES_AT.values()]
    room = sizes[2] + sizes[3]                                            # room for the last two names' features alone
    monkeypatch.setattr(trade_model, "_FEATURES_AT", collections.OrderedDict())
    monkeypatch.setattr(trade_model, "_features_bytes", 0)
    monkeypatch.setattr(trade_model, "FEATURES_KEPT_BYTES", room)
    trade_model.trades(p, wanted, p.started)
    assert trade_model._features_bytes == sum(v.nbytes for v in trade_model._FEATURES_AT.values()) <= room
    del worked_out[:]
    for names, worked in ((["td:CCC", "td:DDD"], 0), (["td:AAA", "td:BBB"], 2)):     # kept, dropped
        q = Panel(p.timeframe, {i: p.instruments[i] for i in names}, **{f: getattr(p, f)[names] for f in FIELDS})
        _, got = trade_model.trades(q, wanted[names], q.started)
        assert len(worked_out) == worked and np.array_equal(got, x[t["instrument"].isin(names).to_numpy()],
                                                            equal_nan=True)
