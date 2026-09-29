"""Nothing runs on a list outside strategy_lab.lists unless the user asks for it (--outside-lists)."""
import sys
from pathlib import Path

import pytest

from strategy_lab import __main__ as cli
from strategy_lab import db, lists, per_asset, runs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import per_asset as per_asset_script  # noqa: E402
import product_protocol  # noqa: E402


def _never(*a, **k):
    raise AssertionError("a run started on a list the guard should have refused")


def test_the_command_line_refuses_a_list_outside_ours_even_the_desks_for_its_own_rule(monkeypatch):
    monkeypatch.setattr(cli, "evaluate", _never)
    for argv in (["run", "sma_cross", "-u", "coiniq", "-t", "1d"],
                 ["run", "ibs", "-u", "stockhunt_stocks", "-t", "1d"]):
        monkeypatch.setattr(sys, "argv", ["strategy_lab", *argv])
        with pytest.raises(SystemExit, match="--outside-lists"):
            cli.main()


def test_the_command_line_takes_our_lists_and_any_list_when_asked(monkeypatch):
    ran = []
    monkeypatch.setattr(cli, "evaluate", lambda s, u, tf, **k: ran.append((s.name, u)))
    for argv in (["run", "ibs", "-u", "us_stocks_top100", "-t", "1d"],
                 ["run", "ibs", "-u", "stockhunt_stocks", "-t", "1d", "--outside-lists"]):
        monkeypatch.setattr(sys, "argv", ["strategy_lab", *argv])
        cli.main()
    assert ran == [("ibs", "us_stocks_top100"), ("ibs", "stockhunt_stocks")]


def test_the_per_instrument_batch_runs_only_the_widest_lists_and_the_ml_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.sqlite")
    monkeypatch.setattr(per_asset, "evaluate", _never)
    monkeypatch.setattr(sys, "argv", ["per_asset.py", "--names", "ibs", "--universes", "coiniq", "stockhunt_stocks"])
    with pytest.raises(SystemExit, match="coiniq, stockhunt_stocks"):
        per_asset_script.main()
    ran = []
    monkeypatch.setattr(per_asset, "evaluate", lambda s, u, tf: ran.append(u))
    monkeypatch.setattr(runs, "_covered", lambda inner, outer, tf: False)
    from strategy_lab import catalog, strategy_pick      # the stages after the jobs read every list's bars and spreads
    monkeypatch.setattr(catalog, "refresh", lambda conn=None: None)
    monkeypatch.setattr(strategy_pick, "pick_all", lambda conn: None)
    monkeypatch.setattr(sys, "argv", ["per_asset.py", "--names", "ibs", "--universes", "crypto_mcap10", "--workers", "1"])
    with pytest.raises(SystemExit) as done:
        per_asset_script.main()
    assert done.value.code == 0 and set(ran) == {"crypto_mcap10"}


def test_a_list_whose_instruments_the_widest_list_holds_is_not_run_again_for_the_rules(monkeypatch):
    """The ML task's ten largest stocks are inside the hundred most liquid: their rules come from that run, and only
    the firm's engine runs on them; a list its market's widest one does not cover runs the rules too."""
    only = {"lists": ["us_stocks_mcap10", "crypto_mcap10"]}
    monkeypatch.setattr(runs, "_covered", lambda inner, outer, tf: inner == "us_stocks_mcap10")
    jobs = runs.alone_jobs(only)
    assert {s for s, u, _ in jobs if u == "us_stocks_mcap10"} == {runs.ENGINE}
    assert {s for s, u, _ in jobs if u == "crypto_mcap10"} == set(runs.ALONE_RULES) | {runs.ENGINE}
    assert {tf for *_, tf in jobs} == {"1h", "4h", "1d"}


def test_the_protocol_replay_runs_only_the_ml_tasks_markets(monkeypatch):
    monkeypatch.setattr(product_protocol, "run", _never)
    monkeypatch.setattr(sys, "argv", ["product_protocol.py", "--universes", "coiniq"])
    with pytest.raises(SystemExit, match="--outside-lists"):
        product_protocol.main()


def test_every_run_on_a_list_as_one_book_is_on_an_agreed_list():
    for name, universes in runs.LIST_RUNS:
        lists.refuse_outside(universes, lists.basket_lists(), name)
