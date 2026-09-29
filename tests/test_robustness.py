"""The robustness checks measured inside an evaluation, each against another way to the same figure: the walk-forward
run again with the change the check makes (costs, a delay, a seed, a neighbouring list) on the choices the evaluation
made."""
import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest

from strategy_lab import db, metrics, montecarlo, per_asset, significance, trade_model, walkforward
from strategy_lab import evaluate as ev
from strategy_lab import robustness as rb
from strategy_lab.config import COSTS, SEED
from strategies.donchian_breakout import donchian_breakout
from strategies.dual_momentum import dual_momentum
from strategies.ibs import ibs
from strategies.ibs_ml_filter import ibs_ml_filter
from strategies.rsi2_connors import rsi2_connors
from strategies.sector_rotation import sector_rotation
from tests.conftest import make_panel, rising_list_market, store_of

LIST = "demo_top10"


def _evaluated(monkeypatch, strategy, universe=LIST, **kw):
    e = ev.evaluate(strategy, universe, "1d", save=False, **kw)
    _, panel, member = ev._load(universe, "1d", None, None)
    return e, panel, member


def _configs(strategy, universe=LIST):
    """The configurations an evaluation on the list runs: its grid for the list's number of names."""
    return strategy.configs(ev._load(universe, "1d", None, None)[0].capacity)


def _same_choices(monkeypatch, e):
    picks = [(None if pd.isna(c) else int(c), s) for c, s in zip(e.folds["config"], e.folds["train_sharpe_daily"])]
    monkeypatch.setattr(walkforward, "choose", lambda daily, fl: picks)


def _chosen_days(e) -> pd.DatetimeIndex:
    """The out-of-sample days of the evaluation's windows a configuration was chosen for."""
    days = e.oos_daily.index
    keep = np.zeros(len(days), dtype=bool)
    for r in e.folds.itertuples(index=False):
        if not pd.isna(r.config):
            keep |= (days >= pd.Timestamp(r.test_start, tz="UTC")) & (days < pd.Timestamp(r.test_end, tz="UTC"))
    return days[keep]


def _figures(oos):
    c = metrics.core(oos)
    return c["sharpe"], c["cagr"]


def test_a_bar_later_is_the_walk_forward_with_every_decision_a_bar_later_on_the_same_choices(monkeypatch):
    rising_list_market(monkeypatch)
    e, panel, member = _evaluated(monkeypatch, donchian_breakout)
    assert e.robustness is not None and len(e.folds)
    _same_choices(monkeypatch, e)
    *_, oos, _ = ev._walk_forward(donchian_breakout, panel, member, "next_open", _configs(donchian_breakout), "1d",
                                  delay=1)
    sharpe, cagr = _figures(oos)
    assert e.robustness["delay"]["sharpe"] == pytest.approx(sharpe, rel=1e-12)
    assert e.robustness["delay"]["cagr"] == pytest.approx(cagr, rel=1e-12)


@pytest.mark.parametrize("strategy", [donchian_breakout, dual_momentum])
def test_costs_three_times_the_model_are_the_walk_forward_charged_three_times_the_costs(monkeypatch, strategy):
    rising_list_market(monkeypatch)
    e, _, _ = _evaluated(monkeypatch, strategy)
    got = e.robustness["costs"]
    assert got["multiple"] == 3.0
    _same_choices(monkeypatch, e)
    tripled = {k: dataclasses.replace(c, commission_bps=3 * c.commission_bps, half_spread_bps=3 * c.half_spread_bps)
               for k, c in COSTS.items()}
    from strategy_lab.engine import costs
    monkeypatch.setattr(costs, "COSTS", tripled)
    ev._LOADED.clear()                               # a panel of its own: the engine keeps a panel's cost rates
    ev._SEATING.clear()
    _, panel, member = ev._load(LIST, "1d", None, None)
    *_, oos, _ = ev._walk_forward(strategy, panel, member, "next_open", _configs(strategy), "1d")
    sharpe, cagr = _figures(oos)
    assert got["sharpe"] == pytest.approx(sharpe, rel=1e-10) and got["cagr"] == pytest.approx(cagr, rel=1e-10)


def test_a_cme_future_is_charged_ten_basis_points_a_side_and_a_spot_quote_three_times_its_brokers_spread():
    modelled = COSTS["commodity"].commission_bps + COSTS["commodity"].half_spread_bps
    assert rb.cost_multiple(make_panel(ids=("cme:PL",))) == pytest.approx(10.0 / modelled)
    assert rb.cost_multiple(make_panel(ids=("td:XPT/USD",))) == rb.COST_MULTIPLE
    assert rb.cost_multiple(make_panel()) == rb.COST_MULTIPLE


@pytest.mark.parametrize("universe, near", [("demo_top10", ["demo_top5", "demo_top15"]),
                                            ("demo_top3", ["demo_top2", "demo_top4"])])    # never fewer than one name
def test_a_neighbouring_list_measured_inside_is_that_list_evaluated_on_its_own(monkeypatch, universe, near):
    for strategy in (rsi2_connors, ibs_ml_filter, sector_rotation):    # a panel always invested: it earns on a rise
        rising_list_market(monkeypatch)
        ev._LOADED.clear()
        ev._SEATING.clear()
        e, _, _ = _evaluated(monkeypatch, strategy, universe)
        lists = {x["universe"]: x for x in e.robustness["neighbour_lists"]["lists"]}
        assert list(lists) == near
        for u, got in lists.items():
            ev._LOADED.clear()
            ev._SEATING.clear()
            alone = ev.evaluate(strategy, u, "1d", save=False, robustness=False, monte_carlo=False)
            assert got["sharpe"] == pytest.approx(alone.oos["sharpe"], rel=1e-12), (strategy.name, u)


def test_another_seed_is_the_walk_forward_with_the_model_drawing_with_it_on_the_same_choices(monkeypatch):
    rising_list_market(monkeypatch)
    e, panel, member = _evaluated(monkeypatch, ibs_ml_filter)
    got = dict(zip(e.robustness["seeds"]["seeds"], e.robustness["seeds"]["sharpes"]))
    assert trade_model.SEED == SEED                                   # the evaluation gave the model its seed back
    _same_choices(monkeypatch, e)
    monkeypatch.setattr(trade_model, "SEED", 3)
    *_, oos, _ = ev._walk_forward(ibs_ml_filter, panel, member, "next_open", _configs(ibs_ml_filter), "1d")
    assert got[3] == pytest.approx(metrics.core(oos)["sharpe"], rel=1e-12)


def test_the_rule_without_its_model_is_the_rule_itself(monkeypatch):
    """Over the windows the model grades, the rule without it is the rule evaluated alone on the same windows' choices,
    each window's configuration less the model's threshold: a window before them that holds nothing holds nothing
    there too, and the first window chosen after it buys the rule's positions at their cost in both."""
    rising_list_market(monkeypatch)
    natural = walkforward.choose
    monkeypatch.setattr(walkforward, "choose", lambda daily, fl: [(None, -np.inf)] + natural(daily, fl)[1:])
    e, _, _ = _evaluated(monkeypatch, ibs_ml_filter)
    days = _chosen_days(e)
    assert 0 < len(days) < len(e.oos_daily)
    configs = _configs(ibs)
    picks = [(None, -np.inf) if pd.isna(c) else
             (configs.index({k: v for k, v in json.loads(p).items() if k != "threshold"}), s)
             for c, p, s in zip(e.folds["config"], e.folds["params"], e.folds["train_sharpe_daily"])]
    monkeypatch.setattr(walkforward, "choose", lambda daily, fl: picks)
    plain = ev.evaluate(ibs, LIST, "1d", save=False, robustness=False, monte_carlo=False)
    got = e.robustness["vs_rule"]
    assert got["sharpe_plain"] == pytest.approx(rb._sr(plain.oos_daily.reindex(days).to_numpy()), rel=1e-12)
    assert got["sharpe"] == pytest.approx(rb._sr(e.oos_daily.reindex(days).to_numpy()), rel=1e-12)


def test_a_model_check_that_fails_leaves_the_models_seed_as_it_found_it(monkeypatch):
    rising_list_market(monkeypatch)
    before = trade_model.SEED
    real = ev._run

    def fails_on_another_seed(*a, **k):
        if trade_model.SEED != before:
            raise RuntimeError("a run that fails")
        return real(*a, **k)
    monkeypatch.setattr(ev, "_run", fails_on_another_seed)
    with pytest.raises(RuntimeError):
        ev.evaluate(ibs_ml_filter, LIST, "1d", save=False)
    assert trade_model.SEED == before


def test_a_record_that_loses_money_is_not_checked(monkeypatch):
    p = rising_list_market(monkeypatch)
    for k in range(len(p.ids)):                       # every name falls: a long-only rule loses money
        p.close.iloc[:, k] = 100 * np.exp(-0.002 * np.arange(len(p.index)))
        for f in ("open", "high", "low"):
            getattr(p, f).iloc[:, k] = p.close.iloc[:, k]
    monkeypatch.setattr(montecarlo, "bootstrap", lambda *a, **k: pytest.fail("Monte Carlo ran"))
    monkeypatch.setattr(significance, "random_timing", lambda *a, **k: pytest.fail("random timing ran"))
    e = ev.evaluate(ibs, LIST, "1d", save=False)
    assert e.oos["sharpe"] <= 0 or e.oos["cagr"] <= 0
    assert e.robustness is None and e.monte_carlo == {} and e.random_timing == {}


def _alone(monkeypatch, strategy, p):
    """The strategy on the instruments of `p` alone: the first whose record makes money, its id and the configurations
    it chose from."""
    monkeypatch.setattr(per_asset, "load_panel", store_of(p))
    configs = per_asset.single_asset_configs(strategy)
    for i in p.ids:
        o, _ = per_asset.evaluate_instrument(strategy, i, "1d", configs, LIST)
        if o is not None and o.robustness is not None:
            return o, i, configs
    pytest.fail(f"{strategy.name} makes money on no instrument alone")


def test_an_instrument_alone_is_checked_as_a_list_is_but_for_a_lists_own_checks(monkeypatch):
    p = rising_list_market(monkeypatch)
    o, i, configs = _alone(monkeypatch, donchian_breakout, p)
    assert set(o.robustness) - {"seconds"} == set(db.CHECKS) - set(rb.LIST_ONLY)
    assert o.monte_carlo["reps"] > 0
    # a bar later: the instrument's own walk-forward with every decision a bar later, on the same choices
    picks = [(w["config"], w["train_sharpe_daily"]) for w in o.windows]
    assert len(picks) > 1
    monkeypatch.setattr(walkforward, "choose", lambda daily, fl: picks)
    alone = store_of(p)([i], "1d")
    *_, oos, _ = ev._walk_forward(donchian_breakout, alone, None, "next_open", configs, "1d", delay=1)
    sharpe, cagr = _figures(oos)
    assert o.robustness["delay"]["sharpe"] == pytest.approx(sharpe, rel=1e-12)
    assert o.robustness["delay"]["cagr"] == pytest.approx(cagr, rel=1e-12)
    # beating holding: the record against holding the instrument itself over the same days
    held = ev._buy_and_hold(alone, None, "next_open", o.oos.index[0]).reindex(o.oos.index).fillna(0.0)
    assert o.robustness["vs_hold"]["sharpe_other"] == pytest.approx(rb._sr(held.to_numpy()), rel=1e-12)


def test_an_instrument_alone_that_loses_money_is_not_checked(monkeypatch):
    p = rising_list_market(monkeypatch)
    p.close.iloc[:, 0] = 100 * np.exp(-0.002 * np.arange(len(p.index)))      # it falls: a long-only rule loses money
    for f in ("open", "high", "low"):
        getattr(p, f).iloc[:, 0] = p.close.iloc[:, 0]
    monkeypatch.setattr(per_asset, "load_panel", store_of(p))
    monkeypatch.setattr(montecarlo, "bootstrap", lambda *a, **k: pytest.fail("Monte Carlo ran"))
    o, _ = per_asset.evaluate_instrument(ibs, p.ids[0], "1d", per_asset.single_asset_configs(ibs), LIST)
    assert o.card["sharpe"] <= 0 or o.card["cagr"] <= 0
    assert o.robustness is None and o.monte_carlo == {}


def test_pbo_is_nil_when_one_configuration_is_best_in_every_part_of_the_record():
    days = pd.date_range("2020-01-01", periods=1000, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.01, (1000, 4))
    daily = pd.DataFrame(noise + np.array([0.004, 0.0, 0.0, 0.0]), index=days)
    fit = rb.Fit(ibs, LIST, "1d", None, None, "next_open", None, None, [{}] * 4, daily, [0], [], daily[0], {}, {},
                 None, pd.DataFrame(), daily[0])
    assert rb.pbo(fit)["pbo"] == 0.0
    shuffled = rb.pbo(dataclasses.replace(fit, daily=pd.DataFrame(noise, index=days)))["pbo"]
    assert 0.2 < shuffled < 0.8                         # nothing to find: the in-sample best is a coin flip


def test_eras_count_two_year_windows_and_a_last_one_of_a_year():
    days = lambda n: pd.Series(np.random.default_rng(1).normal(0.001, 0.01, n))      # noqa: E731
    assert "too_short" in rb.eras(days(2 * 730 + 364))                             # two windows and a short end
    got = rb.eras(days(2 * 730 + 365))
    s = np.array(got["sharpes"])
    assert len(s) == 3
    rho = 0.5 * (s > 0).mean() + 0.3 * min(max(1 - s.std(ddof=1) / 2, 0), 1) + 0.2 / (1 + math.exp(-s.min()))
    assert got["rho"] == pytest.approx(rho)


def test_the_plateau_is_the_configurations_one_step_from_the_most_chosen():
    days = pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC")
    configs = rsi2_connors.configs()     # rsi_entry [2.5 .. 20] x slots [None .. 20]
    daily = pd.DataFrame(np.random.default_rng(2).normal(0.001, 0.01, (400, len(configs))), index=days)
    top = configs.index({"rsi_entry": 10, "slots": 5})
    choices = [top, top, 4, top]                  # rsi_entry 10, slots 5 the most: a neighbour on either side
    fit = rb.Fit(rsi2_connors, LIST, "1d", None, None, "next_open", None, None, configs, daily, choices, [1, 2, 3, 4],
                 daily[top], {}, {}, None, pd.DataFrame(), daily[top])
    got = rb.plateau(fit)
    assert got["config"] == configs[top]
    assert sorted(map(str, (x["params"] for x in got["neighbours"]))) == sorted(
        map(str, [{"slots": None}, {"slots": 10}, {"rsi_entry": 5}, {"rsi_entry": 15}]))
