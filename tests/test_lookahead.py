import importlib
import pkgutil

import numpy as np
import pandas as pd
import pytest

import strategies
from strategy_lab import indicators as ind
from strategy_lab import lookahead
from strategy_lab.strategy import _aligned, fill_signals, rule
from tests.conftest import make_panel

MODULES = [importlib.import_module(f"strategies.{m.name}") for m in pkgutil.iter_modules(strategies.__path__)]
ALL = lookahead.discover(MODULES)


@pytest.mark.parametrize("strategy", ALL, ids=lambda s: s.name)
def test_every_strategy_passes_the_truncation_test(strategy):
    lookahead.check_all_configs(strategy, make_panel(n=600, seed=11))


@pytest.mark.parametrize("strategy", [s for s in ALL if s.kind == "rule"], ids=lambda s: s.name)
def test_no_rule_writes_into_its_bars(strategy):
    """A grid's signal configurations are given one instrument's bars in turn (`strategy.fill_signals`): a rule that
    wrote into them would change what the next configuration sees."""
    p = make_panel(n=600, seed=11)
    bars = p.one(p.ids[0])
    before = bars.copy()
    signal_configs = {repr(sorted(strategy.signal_params(c).items())): strategy.signal_params(c)
                      for c in strategy.configs()}
    for sp in signal_configs.values():
        strategy.fn(bars, **sp)
    pd.testing.assert_frame_equal(bars, before)


@pytest.mark.parametrize("strategy", [s for s in ALL if s.kind == "rule" and s.prepare is None], ids=lambda s: s.name)
def test_a_grid_sharing_its_indicators_gives_each_configuration_the_positions_it_computes_alone(strategy):
    """A grid's signal configurations share the indicators and regimes worked out on an instrument's bars
    (`strategy.fill_signals`, `strategy_lab.shared`): each still gets its own positions, bit for bit."""
    p = make_panel(n=600, seed=11)
    configs = strategy.configs()
    signals: dict = {}
    fill_signals(strategy, p, configs, signals)
    for c in configs:
        sp = strategy.signal_params(c)
        for i in p.ids:
            alone = _aligned(strategy.fn(p.one(i), **sp), p.index)
            assert signals[(repr(sorted(sp.items())), i)].astype(np.float64).tobytes() == alone.tobytes(), (sp, i)


def test_the_truncation_test_catches_a_centred_window():
    @rule()
    def peeks(bars):
        return (bars.close > bars.close.rolling(21, center=True).mean()).astype(float)
    with pytest.raises(lookahead.LookAhead):
        lookahead.check(peeks, make_panel(n=300))


def test_the_truncation_test_catches_a_full_sample_statistic():
    @rule()
    def zscore_whole_history(bars):
        r = bars.close.pct_change()
        return ((r - r.mean()) / r.std() > 0).astype(float)
    with pytest.raises(lookahead.LookAhead):
        lookahead.check(zscore_whole_history, make_panel(n=300))


def test_talib_indicators_are_causal():
    @rule()
    def rsi_rule(bars):
        return (ind.rsi(bars.close, 14) < 30).astype(float)
    lookahead.check(rsi_rule, make_panel(n=300))


def test_slots_hold_at_most_k_trades_each_at_its_share_and_never_exceed_capital():
    from strategy_lab.strategy import rule as _rule

    @_rule(grid={"slots": [3]})
    def dips(bars):
        return (ind.rsi(bars.close, 2) < 20).astype(float)
    ids = tuple(f"td:S{k}" for k in range(8))
    p = make_panel(ids=ids, n=400, seed=5)
    w = dips.target(p, {"slots": 3})
    every = dips.target(p, {})                           # one seat per name: every trade the rule wants
    assert ((every != 0).sum(axis=1) > 3).any()          # more at once than there are slots, on some bars
    assert ((w != 0).sum(axis=1) <= 3).all()
    assert ((w[w != 0].stack() - 1 / 3).abs() < 1e-12).all()
    assert (w.abs().sum(axis=1) <= 1 + 1e-12).all()
    lookahead.check(dips, p, {"slots": 3})


def test_a_missed_bar_keeps_the_rule_position_and_the_panel_sees_the_last_close():
    from strategy_lab.strategy import panel as _panel
    p = make_panel(n=300, seed=21)
    for f in ("open", "high", "low", "close", "volume", "dollar_volume"):
        getattr(p, f).loc[p.index[150], "td:AAA"] = float("nan")

    @rule()
    def always(bars):
        return (bars.close > 0).astype(float)

    @_panel()
    def by_price(q):
        return q.close.notna().astype(float) / 2

    for s in (always, by_price):
        w = s.target(p, {})
        assert w["td:AAA"].iloc[150] == w["td:AAA"].iloc[149] == 0.5 and w["td:BBB"].iloc[150] == 0.5
        lookahead.check(s, p, {}, cuts=(0.5, 0.9))


@pytest.mark.parametrize("name", ["gtaa_faber", "tsmom", "vol_managed", "rsi2_connors", "dual_momentum",
                                  "sector_rotation", "xs_momentum"])
def test_rules_timed_in_days_and_months_see_no_future_on_hourly_bars(name):
    from strategy_lab.data.bars import Panel
    from strategy_lab.strategy import load
    ids = tuple(f"perp:S{k}USDT" for k in range(7))
    daily = make_panel(ids=ids, n=24 * 330, seed=13, freq="h", start="2023-01-01 01:00")
    p = Panel("1h", daily.instruments, **{f: getattr(daily, f) for f in ("open", "high", "low", "close", "volume",
                                                                           "dollar_volume")})
    s = load(name)
    lookahead.check(s, p, s.configs(len(ids))[0], cuts=(0.75, 0.95))
