"""What an evaluation measures for robustness reaches the board: a check whose measure does not reach it would read as
not measured for ever."""
import json

from strategy_lab import board, db
from strategy_lab import evaluate as ev
from strategies.rsi2_connors import rsi2_connors
from tests.conftest import rising_list_market


def test_an_evaluations_robustness_checks_reach_the_board(tmp_path, monkeypatch):
    rising_list_market(monkeypatch)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.sqlite")
    ev.evaluate(rsi2_connors, "us_stocks_top10", "1d")
    conn = db.connect()
    measured = db.measures(conn, db.result_id(conn, "rsi2_connors", "us_stocks_top10", "1d"))
    conn.close()
    # every check was measured or said why not, but what is worked out when the result is read
    assert set(board.checks_of(alone=False)) - set(measured) == set(board.WHEN_READ) & set(board.checks_of(alone=False))
    r = next(board.collect().itertuples(index=False))
    checks = json.loads(r.robustness_checks)
    assert [c[0] for c in checks] == board.checks_of(alone=False)
    assert r.robustness == f"{r.robustness_passed}/{r.robustness_applicable}"
    assert {c[4] for c in checks} <= {"passed", "failed", "too_short", "not_applicable"}
    assert dict((c[0], c[4]) for c in checks)["seeds"] == "not_applicable"       # a rule without a model


def test_a_threshold_decides_a_check_and_the_count_leaves_out_what_does_not_apply():
    card = {"out_of_sample": {"sharpe": 0.8}, "robustness": {
        "luck": {"mc_sharpe_p5": 0.4}, "timing": {"p": 0.2, "null_median": 0.1},
        "pbo": {"na": "one configuration: nothing is chosen"}, "eras": {"too_short": "2 two-year windows, 3 needed"},
        "names": {"instruments": 10, "positive": 6, "share": 0.6}}}
    got = board.robustness(card, None, 100, {"prob": 0.99, "noise_bar": 0.5})
    states = {c.id: c.state for c in got.checks}
    assert (states["luck"], states["timing"], states["pbo"], states["eras"], states["names"]) == (
        "passed", "failed", "not_applicable", "too_short", "passed")
    assert all(states[k] == "not_computed" for k in ("vs_hold", "plateau", "best_year", "delay", "costs",
                                                    "neighbour_lists", "seeds", "vs_rule"))
    assert "not read" in {c.id: c for c in got.checks}["best_year"].value           # no record given to work it out
    assert str(got) == "2/4" and got.not_computed == 8            # too short counts against, n/a and not measured out
    assert board.robustness({"out_of_sample": {"sharpe": -0.1}, "robustness": None}, None, 100) is None
    older = board.robustness({"out_of_sample": {"sharpe": 0.8}}, None, 100, {"prob": 0.99, "noise_bar": 0.5})
    assert str(older) == "0/0" and older.not_computed == len(board.checks_of(alone=False))   # saved before the checks


def test_a_neighbour_seed_or_list_that_loses_money_fails_its_check_whatever_its_sharpe():
    card = {"strategy": "x", "universe": "crypto_top100", "timeframe": "1h", "out_of_sample": {"sharpe": 0.8},
            "robustness": {
                "plateau": {"sharpe": 0.8, "neighbours": [{"params": {"n": 10}, "sharpe": 0.7, "cagr": 0.2},
                                                          {"params": {"n": 30}, "sharpe": 0.6, "cagr": -0.1}]},
                "neighbour_lists": {"lists": [{"universe": "crypto_top80", "sharpe": 0.53, "cagr": -0.77},
                                              {"universe": "crypto_top120", "sharpe": 0.7, "cagr": 0.3}]},
                "seeds": {"seeds": [1, 2], "sharpes": [0.7, 0.6], "cagrs": [0.2, 0.1]},
                "delay": {"sharpe": 0.5}}}                           # measured by code that kept no return
    got = {c.id: c for c in board.robustness(card, None, 100, {"prob": 0.99, "noise_bar": 0.5}).checks}
    assert (got["plateau"].state, got["neighbour_lists"].state, got["seeds"].state) == ("failed", "failed", "passed")
    assert "Top-80 0.53 (-77.0%/yr)" in got["neighbour_lists"].value              # why a Sharpe ≥ ½ still fails
    assert got["delay"].state == "not_computed" and "re-run" in got["delay"].value
