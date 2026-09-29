import numpy as np
import pandas as pd
from scipy import stats

from strategy_lab import board, db
from strategy_lab import evaluate as ev
from strategy_lab.universes import Universe
from strategies.ibs import ibs
from strategies.sma_cross import sma_cross
from tests.conftest import make_panel


def test_board_ranks_our_lists_and_counts_any_other_as_not_ranked(tmp_path, monkeypatch):
    panel = make_panel(n=900, seed=3)
    monkeypatch.setattr(ev, "resolve", lambda name, tf: Universe(name, list(panel.ids)))
    monkeypatch.setattr(ev, "load_panel", lambda ids, tf, start=None, end=None: panel)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.sqlite")
    for strategy, universe in ((sma_cross, "us_stocks_top10"), (ibs, "stockhunt_stocks"),
                               (sma_cross, "stockhunt_stocks"), (sma_cross, "coiniq")):
        ev.evaluate(strategy, universe, "1d", monte_carlo=False, robustness=False)
    ranked = board.collect()
    assert set(zip(ranked["strategy"], ranked["universe"])) == {("sma_cross", "us_stocks_top10")}
    assert dict(zip(ranked["universe"], ranked["market"])) == {"us_stocks_top10": "Stocks"}
    assert board.not_ranked() == {"coiniq": 1, "stockhunt_stocks": 2}


def test_luck_counts_copies_once_and_strangers_each():
    rng = np.random.default_rng(0)
    days = pd.date_range("2020-01-01", periods=1500, freq="D", tz="UTC")
    strangers = pd.DataFrame(rng.normal(0, 0.01, (1500, 40)), index=days)
    copies = pd.concat([strangers[[0]]] * 40, axis=1, ignore_index=True)
    two_bets = pd.concat([strangers[[0]]] * 20 + [strangers[[1]]] * 20, axis=1, ignore_index=True)
    independent, best_95 = board.luck_of(strangers)
    assert 36 <= independent <= 40
    assert abs(best_95 - stats.norm.ppf(0.95 ** (1 / 40))) < 0.08          # Sidak's point for 40 independent tests
    independent, best_95 = board.luck_of(copies)
    assert independent < 1.05 and abs(best_95 - stats.norm.ppf(0.95)) < 0.05    # one test, however many copies
    assert 1.8 <= board.luck_of(two_bets)[0] <= 2.3
    # a record and its own last tenth share a tenth of the evidence: their Sharpe ratios correlate at about 0.32
    tail = pd.DataFrame({"whole": strangers[0], "tail": strangers[0].where(days >= days[1350])})
    assert 1.5 < board.luck_of(tail)[0] < 2.0
    # records not known count as independent tries of their own
    assert abs(board.luck_of(copies, extra=1)[0] - board.luck_of(two_bets)[0]) < 0.15
    assert board.luck_of(strangers.iloc[:, :0], extra=40)[0] >= 36
    # blocks are independent of each other: two blocks of copies are two tries, two blocks of strangers all forty
    assert 1.8 <= board.luck_of([copies.iloc[:, :20], two_bets.iloc[:, 20:]])[0] <= 2.3
    assert 36 <= board.luck_of([strangers.iloc[:, :20], strangers.iloc[:, 20:]])[0] <= 40


def test_each_instruments_tries_are_its_own_count_and_all_of_them_at_once_the_whole_familys():
    rng = np.random.default_rng(0)
    days = pd.date_range("2020-01-01", periods=1500, freq="D", tz="UTC")
    strangers = pd.DataFrame(rng.normal(0, 0.01, (1500, 40)), index=days)
    copies = pd.concat([strangers[[0]]] * 20, axis=1, ignore_index=True)
    two_bets = pd.concat([strangers[[1]]] * 10 + [strangers[[2]]] * 10, axis=1, ignore_index=True)
    blocks = [strangers.iloc[:, 20:40], copies, two_bets, strangers.iloc[:, 3:4]]
    (independent, best_95), each = board.luck_of_each(blocks)
    for got, block in zip(each, blocks):
        alone = board.luck_of(block)
        assert abs(got[0] - alone[0]) <= 0.03 * alone[0] + 0.1 and abs(got[1] - alone[1]) < 0.05
    assert np.allclose(each[-1], (1.0, stats.norm.ppf(0.95)))            # a lone record: one try
    together = board.luck_of(blocks)
    assert abs(independent - together[0]) <= 0.03 * together[0] and abs(best_95 - together[1]) < 0.05


def test_an_instrument_alone_is_judged_on_how_many_of_the_others_of_its_run_make_money():
    rows = pd.DataFrame(
        [(1, "ibs", "etf_core", "1d", 0.9, 0.10), (2, "ibs", "etf_core", "1d", 0.5, 0.05),
         (3, "ibs", "etf_core", "1d", 0.4, -0.01), (4, "ibs", "etf_core", "1d", -0.2, -0.03),
         (5, "ibs", "etf_core", "1d", 0.7, 0.08),
         (6, "ibs", "etf_core", "4h", 0.9, 0.10), (7, "ibs", "etf_core", "4h", 0.3, 0.02),
         (8, "ibs", "etf_core", "4h", None, None)],
        columns=["id", "strategy", "list_id", "timeframe", "sharpe", "cagr"])
    got = board.peers(rows)
    # 1d: 1, 2 and 5 make money (3's Sharpe is positive but it loses money compounded, 4 loses)
    assert got[1] == {"others": 4, "positive": 2, "share": 0.5} and got[4]["positive"] == 3
    assert "too_short" in got[6] and set(got) == set(rows["id"])        # 4h: two others, three needed
    card = {"out_of_sample": {"sharpe": 0.9}, "robustness": {"peers": got[1], "luck": {"mc_sharpe_p5": 0.2}}}
    own, every = board.Tries(72, 31.0, 2.9), board.Tries(17000, 13000.0, 4.5)
    judged = board.robustness(card, None, own, {"prob": 0.97, "noise_bar": 0.8}, alone=True,
                              every=(every, {"prob": 0.12, "noise_bar": 1.5}))
    checks = {c.id: c for c in judged.checks}
    assert [c.id for c in judged.checks] == board.checks_of(alone=True)
    assert not set(checks) & set(board.LIST_ONLY) and "peers" not in board.checks_of(alone=False)
    assert (checks["peers"].state, checks["peers"].value) == ("passed", "2 of 4 others make money (50%)")
    assert checks["luck"].state == "passed"
    assert "97% beyond the best of this asset's 31 independent of 72 tries" in checks["luck"].value
    assert "12% beyond the best of every asset's 13000 independent of 17000 tries" in checks["luck"].value
    card["robustness"]["peers"] = got[4]
    assert {c.id: c.state for c in board.robustness(card, None, own, {"prob": 0.97, "noise_bar": 0.8},
                                                    alone=True).checks}["peers"] == "passed"
    card["robustness"]["peers"] = {"others": 4, "positive": 1, "share": 0.25}
    assert {c.id: c.state for c in board.robustness(card, None, own, {"prob": 0.97, "noise_bar": 0.8},
                                                    alone=True).checks}["peers"] == "failed"


def _record(years: dict[int, tuple[float, int]]) -> pd.Series:
    """A daily record whose calendar years each compound to a return over their first days (year -> (return, days)),
    the same rate every day of a year."""
    parts = [pd.Series((1.0 + r) ** (1.0 / n) - 1.0, index=pd.date_range(f"{y}-01-01", periods=n, freq="D", tz="UTC"))
             for y, (r, n) in years.items()]
    return pd.concat(parts)


def test_a_record_that_made_its_money_in_one_year_keeps_little_of_its_average_month_without_it():
    # the middle year doubles the account and the other two add 2% each: 730 days, 24 months of 1/12 of a year, are
    # left without it, of 1095 days, 36 months, in all
    daily = _record({2021: (0.02, 365), 2022: (1.0, 365), 2023: (0.02, 365)})
    got = board.best_year(daily)
    assert got["year"] == 2022 and abs(got["year_return"] - 1.0) < 1e-9
    assert abs(got["rest_avg_monthly"] - ((1.02 * 1.02) ** (1 / 24) - 1.0)) < 1e-12
    assert abs(got["avg_monthly"] - ((2.0 * 1.02 * 1.02) ** (1 / 36) - 1.0)) < 1e-12
    card = {"out_of_sample": {"sharpe": 1.0}, "robustness": {}}
    check = {c.id: c for c in board.robustness(card, daily, 100, {"prob": 0.99, "noise_bar": 0.5}).checks}["best_year"]
    assert (check.state, check.value, check.threshold) == (
        "failed", "without 2022 (+100%): +0.17% a month vs +2.06%", "≥ ½ × 2.06%")


def test_a_record_that_earns_year_after_year_keeps_its_average_month_and_a_short_one_is_not_judged():
    steady = board.best_year(_record({2021: (0.2, 365), 2022: (0.2, 365), 2023: (0.2, 365)}))
    assert abs(steady["rest_avg_monthly"] - (1.2 ** (1 / 12) - 1.0)) < 1e-12            # 20% a year is 1.53% a month
    assert abs(steady["rest_avg_monthly"] - steady["avg_monthly"]) < 1e-12
    card = {"out_of_sample": {"sharpe": 1.0}, "robustness": {"best_year": steady}}
    assert {c.id: c.state for c in board.robustness(card, None, 100, {"prob": 0.99, "noise_bar": 0.5}).checks}[
        "best_year"] == "passed"
    short = board.best_year(_record({2025: (0.3, 365), 2026: (0.1, 200)}))             # 200 days left without 2025
    assert short == {"too_short": "200 days out-of-sample besides its best year 2025, 365 needed"}
