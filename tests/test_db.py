"""The app's database: what an evaluation computed goes in whole and comes back the same, results of the files before
it import to what the old board showed, and writers in many processes never step on each other."""
import json
import math
import multiprocessing as mp
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from strategy_lab import board, db, metrics
from strategy_lab import evaluate as ev
from strategy_lab.config import FIRM_TARGETS
from strategy_lab.universes import Universe
from strategies.sma_cross import sma_cross
from tests.conftest import make_panel

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.sqlite")
    c = db.connect()
    yield c
    c.close()


def _evaluated(monkeypatch, strategy=sma_cross, name="us_stocks_top10", **kw):
    panel = make_panel(n=900, seed=3)
    monkeypatch.setattr(ev, "resolve", lambda n, tf: Universe(n, list(panel.ids)))
    monkeypatch.setattr(ev, "load_panel", lambda ids, tf, start=None, end=None, **k: panel)
    return ev.evaluate(strategy, name, "1d", **kw)


def test_an_evaluation_saved_comes_back_as_it_was_computed(conn, monkeypatch):
    e = _evaluated(monkeypatch, monte_carlo=True, robustness=False)
    rid = db.result_id(conn, "sma_cross", "us_stocks_top10", "1d")
    row = conn.execute("SELECT * FROM result WHERE id = ?", (rid,)).fetchone()
    assert row["fill"] == "next_open" and row["seconds"] > 0 and json.loads(row["grid"]) == {
        k: list(v) for k, v in sma_cross.grid_for(len(ev.load_panel(None, "1d").ids)).items()}
    got = db.figures(conn, rid)
    for k in db.CORE + db.TRADE_STATS + db.EXPOSURE:
        want = e.oos[k]
        assert got[k] == (want if isinstance(want, str) or want is None or np.isfinite(want) else None), k
    pd.testing.assert_series_equal(db.series(conn, rid), e.oos_daily.rename(None), check_freq=False)
    pd.testing.assert_series_equal(db.series(conn, rid, "in_sample"), e.is_daily.rename(None), check_freq=False)
    # the K-ratio of each record kept, worked out from the series saved with it
    assert got["k_ratio"] == pytest.approx(db.k_ratio(e.oos_daily), rel=1e-12)
    assert db.figures(conn, rid, "in_sample")["k_ratio"] == pytest.approx(db.k_ratio(e.is_daily), rel=1e-12)
    # and its average month at the target's drawdown, with the multiple and the last five years', from the same series
    for scope, daily in (("out_of_sample", e.oos_daily), ("in_sample", e.is_daily)):
        f = db.figures(conn, rid, scope)
        want = db.at_target_dd(daily, f["avg_gross"], f["time_in_market"])
        assert want["at_target_dd"] is not None, scope
        assert [f[k] for k in db.AT_TARGET_DD] == [None if want[k] is None else pytest.approx(want[k], rel=1e-12)
                                                  for k in db.AT_TARGET_DD], scope
    assert [w["params"] for w in db.windows(conn, rid)] == [json.loads(p) for p in e.folds["params"]]
    assert db.monte_carlo(conn, rid) == e.monte_carlo
    t = db.trades(conn, rid)
    assert len(t) == len(e.trades) and np.allclose(t["net_return"], e.trades["net_return"])
    # each name's trades in the book, as the popup shows them: compounded from the trades themselves
    names = {r["instrument_id"]: r for r in conn.execute("SELECT * FROM result_name WHERE result_id = ?", (rid,))}
    for inst, g in e.trades.groupby("instrument"):
        assert names[inst]["trades"] == len(g)
        assert math.isclose(names[inst]["compounded"], float(np.prod(1 + g["net_return"]) - 1), rel_tol=1e-9)
    # the buy & hold kept beside the result gives back the figures the scorecard compared it with
    bench = db.benchmark_series(conn, row["benchmark_id"])
    core = metrics.core(bench)
    for k in db.BH:
        assert math.isclose(row[f"bh_{k}"], core[k], rel_tol=1e-12, abs_tol=1e-15), k
    held = conn.execute("SELECT k_ratio FROM benchmark WHERE id = ?", (row["benchmark_id"],)).fetchone()[0]
    assert held == pytest.approx(db.k_ratio(bench), rel=1e-12)
    # and its average month at the target's drawdown, holding all its money in the list as buy & hold does
    sized = conn.execute(f"SELECT {', '.join(db.AT_TARGET_DD)} FROM benchmark WHERE id = ?",
                         (row["benchmark_id"],)).fetchone()
    want = db.at_target_dd(bench, 1.0, 1.0)
    assert sized[0] is not None and 0 < sized[1] < 1
    assert list(sized) == [pytest.approx(want[k], rel=1e-12) if want[k] is not None else None for k in db.AT_TARGET_DD]


def test_a_result_saved_again_keeps_its_id_and_nothing_of_the_evaluation_before(conn, monkeypatch):
    _evaluated(monkeypatch, monte_carlo=True, robustness=False)
    rid = db.result_id(conn, "sma_cross", "us_stocks_top10", "1d")
    _evaluated(monkeypatch, monte_carlo=False, robustness=False)
    assert db.result_id(conn, "sma_cross", "us_stocks_top10", "1d") == rid
    assert db.monte_carlo(conn, rid) == {}                      # the second evaluation ran none
    assert conn.execute("SELECT count(*) FROM benchmark").fetchone()[0] == 1
    assert [tuple(r) for r in conn.execute("SELECT result_id, kind, op FROM change ORDER BY seq")] == [
        (rid, "list", "saved"), (rid, "list", "saved")]


def _outcome(instrument, days=400, seed=0, bench=True):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days, freq="D", tz="UTC")
    oos = pd.Series(rng.normal(0.0005, 0.01, days), index=idx)
    bh = pd.Series(rng.normal(0.0003, 0.012, days), index=idx) if bench else None
    card = metrics.scorecard(oos, benchmark=bh if bench else pd.Series(0.0, index=idx), cash=not bench)
    return db.Outcome(strategy="ibs", list_id="crypto_mcap10", timeframe="1d", instrument_id=instrument, card=card,
                      oos=oos, position=pd.Series(1.0, index=idx), benchmark=bh)


CODE = {"sha": "0" * 16, "files": ["strategies/ibs.py"]}


def test_a_run_of_instruments_alone_replaces_the_one_before_it_whole(conn):
    db.save_outcomes(conn, [_outcome("spot:AAA", seed=1), _outcome("spot:BBB", seed=2)], description="", code=CODE,
                     fill="next_open", replace_run=("ibs", "crypto_mcap10", "1d"), too_short=[("spot:CCC", 3)])
    assert conn.execute("SELECT count(*) FROM benchmark").fetchone()[0] == 2
    kept = db.result_id(conn, "ibs", "crypto_mcap10", "1d", "spot:AAA")
    db.save_outcomes(conn, [_outcome("spot:AAA", seed=1)], description="", code=CODE, fill="next_open",
                     replace_run=("ibs", "crypto_mcap10", "1d"), too_short=[])
    rows = conn.execute("SELECT id, instrument_id FROM result").fetchall()
    assert [(r["id"], r["instrument_id"]) for r in rows] == [(kept, "spot:AAA")]
    assert conn.execute("SELECT count(*) FROM benchmark").fetchone()[0] == 1         # BBB's holding went with it
    assert conn.execute("SELECT count(*) FROM asset_too_short").fetchone()[0] == 0
    assert conn.execute("SELECT op FROM change ORDER BY seq DESC LIMIT 1").fetchone()[0] == "deleted"


def test_a_transaction_that_fails_leaves_no_result_and_no_change(conn):
    bad = _outcome("spot:AAA")
    bad.card["targets_met"] = 9                                  # out of range: the schema refuses it
    with pytest.raises(sqlite3.IntegrityError):
        db.save_outcomes(conn, [_outcome("spot:ZZZ"), bad], description="", code=CODE, fill="next_open")
    assert conn.execute("SELECT count(*) FROM result").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM change").fetchone()[0] == 0


def test_a_series_is_kept_to_the_last_bit_and_a_day_missing_is_refused():
    s = pd.Series(np.random.default_rng(1).normal(0, 0.01, 1000),
                  index=pd.date_range("2020-01-01", periods=1000, freq="D", tz="UTC"))
    first, days, blob = db.encode_series(s)
    back = db.decode_series(first, days, blob)
    assert np.array_equal(back.to_numpy(), s.to_numpy()) and back.index.equals(s.index)
    with pytest.raises(ValueError, match="one value per UTC day"):
        db.encode_series(s.drop(s.index[10]))


def _write_from_a_process(path, k, ready):
    import numpy as np  # noqa: F401 - the spawned process imports what it needs
    from strategy_lab import db as d
    d.DB_PATH = Path(path)
    ready.wait()
    c = d.connect()                                  # every process opens the new file at once and brings its schema
    for j in range(5):
        d.save_outcomes(c, [_outcome(f"spot:P{k}N{j}", seed=k * 10 + j)], description="", code=CODE, fill="next_open")
    c.close()


def test_writers_in_many_processes_opening_a_new_file_at_once_all_get_their_results_in(tmp_path):
    path = tmp_path / "app.sqlite"
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    procs = [ctx.Process(target=_write_from_a_process, args=(str(path), k, ready)) for k in range(8)]
    for p in procs:
        p.start()
    ready.set()
    for p in procs:
        p.join(120)
    assert [p.exitcode for p in procs] == [0] * 8
    c = sqlite3.connect(path)
    assert c.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert c.execute("SELECT count(*) FROM result").fetchone()[0] == 40
    assert c.execute("SELECT count(DISTINCT seq) FROM change").fetchone()[0] == 40
    c.close()


def _open_new_files_at_once(paths, start, failures):
    from strategy_lab import db as d
    for p in paths:
        start.wait()                                     # every process opens the same new file at the same instant
        try:
            d.connect(Path(p)).close()
        except sqlite3.OperationalError as e:
            failures.put(f"{Path(p).name}: {e}")
    failures.put(None)                                   # this process is through


def test_processes_switching_a_new_file_to_wal_at_once_all_open_it(tmp_path):
    """Processes switching one new file to WAL at once could each wait for the other: SQLite answers one of them
    'locked' at once instead of calling its busy handler, and that one tries again."""
    ctx = mp.get_context("spawn")
    paths = [str(tmp_path / f"new_{r}.sqlite") for r in range(20)]
    start, failures = ctx.Barrier(8, timeout=60), ctx.Queue()
    procs = [ctx.Process(target=_open_new_files_at_once, args=(paths, start, failures)) for _ in range(8)]
    for p in procs:
        p.start()
    got, through = [], 0
    while through < len(procs):                          # read before joining: a process ends once its queue is read
        m = failures.get(timeout=120)
        through += m is None
        got += [] if m is None else [m]
    for p in procs:
        p.join(120)
    assert got == [] and [p.exitcode for p in procs] == [0] * 8
    for p in paths:
        c = sqlite3.connect(p)
        assert c.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        c.close()


def test_the_figures_kept_are_the_keys_the_metrics_return():
    d = pd.Series(np.random.default_rng(2).normal(0.001, 0.01, 400),
                  index=pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC"))
    assert tuple(metrics.core(d)) == db.CORE
    trades = pd.DataFrame({"net_return": [0.01, -0.02], "bars": [3, 4]})
    assert tuple(metrics.trade_stats(trades, 12)) == db.TRADE_STATS
    assert tuple(metrics.exposure_stats(pd.Series([0.5, 0.0], index=d.index[:2]))) == db.EXPOSURE
    assert tuple(db.at_target_dd(d, 1.0, 1.0)) == db.AT_TARGET_DD
    # every check an evaluation measures is kept; an instrument alone's peers (from their results) and a record without
    # its best year (from the record) are worked out when a result is read
    assert db.CHECKS == tuple(c for c in board.CHECKS if c not in board.WHEN_READ)
    assert set(metrics.targets({})) == set(db.TARGETS)
    c = db.connect(Path(":memory:"))
    assert {r[1] for r in c.execute("PRAGMA table_info(result_figures)")} == {"result_id", "scope", *db.FIGURES}
    assert set(db.BENCHMARK_FIGURES) <= {r[1] for r in c.execute("PRAGMA table_info(benchmark)")}


def _days(values) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="D", tz="UTC"))


def test_the_k_ratio_is_the_log_equitys_slope_over_its_standard_error_times_the_root_of_a_year_over_the_days():
    """Kestner's K-ratio as he corrected it in 2013, checked against scipy's least squares."""
    d = _days(np.random.default_rng(5).normal(0.0006, 0.012, 1500))
    fit = stats.linregress(np.arange(len(d)), np.log((1.0 + d).cumprod()))
    assert db.k_ratio(d) == pytest.approx(fit.slope / fit.stderr * np.sqrt(365) / len(d), rel=1e-9)


def test_a_climb_made_in_one_burst_scores_far_below_the_same_climb_made_steadily_at_the_same_sharpe():
    """What the K-ratio tells that the Sharpe does not: the same days' noise and the same gain, earned evenly or all in
    the last tenth of the record."""
    noise = np.random.default_rng(0).normal(0.0, 0.01, 2000)
    steady = _days(noise + 0.0008)
    burst = _days(noise + np.where(np.arange(2000) >= 1800, 0.008, 0.0))
    assert metrics.core(burst)["sharpe"] == pytest.approx(metrics.core(steady)["sharpe"], rel=0.05)
    assert float((1 + burst).prod()) == pytest.approx(float((1 + steady).prod()), rel=0.01)
    assert db.k_ratio(steady) > 1.0 and db.k_ratio(burst) < db.k_ratio(steady) / 4


def test_independent_days_score_about_1_1_times_their_sharpe_however_long_the_record():
    """A random walk strays from its fitted line by √(n/15) of a day's spread, so a record of independent days scores
    about √1.25 times its Sharpe; the corrections of 2013 keep that for a short record and a long one alike (the 1996
    K-ratio grows with the square root of the days, the raw t-statistic with the days)."""
    rng = np.random.default_rng(1)
    for n in (1000, 8000):
        k, sharpe = [], []
        for _ in range(400):
            d = np.expm1(rng.normal(0.01 / np.sqrt(365), 0.01, n))          # a Sharpe of about 1
            k.append(db.k_ratio(d))
            sharpe.append(d.mean() / d.std(ddof=1) * np.sqrt(365))
        assert 0.95 < np.median(k) / np.median(sharpe) < 1.25, n


def test_a_record_that_lost_everything_or_is_too_short_has_no_k_ratio_and_a_flat_one_scores_zero():
    assert np.isnan(db.k_ratio(_days([0.01, 0.02, -1.0, 0.01, 0.03])))        # no log equity after its third day
    assert np.isnan(db.k_ratio(_days([0.01, 0.02])))
    assert np.isnan(db.k_ratio(None))
    assert db.k_ratio(_days(np.zeros(90))) == 0.0                            # never held a position: as its Sharpe


def test_a_file_at_another_schema_is_refused_by_a_writer_and_a_reader(tmp_path):
    path = tmp_path / "app.sqlite"
    db.connect(path).close()                                        # a new file: made at the code's schema
    c = sqlite3.connect(path)
    assert c.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    c.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION - 1}")       # a file of the schema before
    c.close()
    for readonly in (False, True):
        with pytest.raises(RuntimeError, match=f"is at schema {db.SCHEMA_VERSION - 1}, "):
            db.connect(path, readonly=readonly)


def test_a_record_kept_with_its_series_has_its_k_ratio_and_one_kept_without_has_none(tmp_path, monkeypatch):
    """The writer works a record's K-ratio out from the series it saves with it: an instrument alone's in-sample record,
    whose series is not kept, has none; each buy & hold has its own."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.sqlite")
    alone = _outcome("spot:AAA", seed=4)
    alone.in_sample = db.Figures(metrics.core(alone.oos))           # an instrument alone: its in-sample series not kept
    listed = _outcome("spot:BBB", seed=5)
    listed.in_sample_daily = listed.oos * 1.5
    listed.in_sample = db.Figures(metrics.core(listed.in_sample_daily))
    c = db.connect()
    db.save_outcomes(c, [alone, listed], description="", code=CODE, fill="next_open")
    written = {(r[0], r[1]): r[2] for r in c.execute("SELECT result_id, scope, k_ratio FROM result_figures")}
    held = {r[0]: r[1] for r in c.execute("SELECT id, k_ratio FROM benchmark")}
    aaa, bbb = db.result_id(c, "ibs", "crypto_mcap10", "1d", "spot:AAA"), db.result_id(c, "ibs", "crypto_mcap10", "1d",
                                                                                         "spot:BBB")
    assert written[aaa, "in_sample"] is None and None not in (written[aaa, "out_of_sample"], written[bbb, "in_sample"])
    assert len(held) == 2 and None not in held.values()
    c.close()


# ---------------------------------------------------------------------------------------------------- at the target's drawdown
def _one_fall(fall, n=400, day=200, gain=0.001):
    """A record that gains every day but one, whose fall from the top is its only drawdown."""
    d = _days(np.full(n, gain))
    d.iloc[day] = -fall
    return d


def test_a_record_scaled_to_the_target_draws_down_as_deep_as_the_target_and_earns_what_that_size_earns():
    d = _days(np.random.default_rng(4).normal(0.0008, 0.012, 3 * 365))
    got = db.at_target_dd(d, 0.3, 0.9)          # 30% of the equity in positions: scaled up from idle money alone
    k = got["at_target_dd_multiple"]
    assert 0 < k * 0.3 < 1
    assert metrics.max_drawdown(k * d)[0] == pytest.approx(FIRM_TARGETS["max_dd"], abs=1e-9)
    assert got["at_target_dd"] == pytest.approx(metrics.core(k * d)["avg_monthly"], rel=1e-9)


def test_a_record_that_fell_as_far_as_the_target_is_taken_as_it_is():
    d = _one_fall(-FIRM_TARGETS["max_dd"])
    got = db.at_target_dd(d, 1.0, 1.0)
    assert got["at_target_dd_multiple"] == pytest.approx(1.0, rel=1e-9)
    assert got["at_target_dd"] == pytest.approx(metrics.core(d)["avg_monthly"], rel=1e-9)
    # one that fell twice as far is held at half its size, the other half of the money idle and earning nothing
    half = db.at_target_dd(_one_fall(-2 * FIRM_TARGETS["max_dd"]), 1.0, 1.0)
    assert half["at_target_dd_multiple"] == pytest.approx(0.5, rel=1e-9)
    assert half["at_target_dd"] == pytest.approx(metrics.core(0.5 * _one_fall(-2 * FIRM_TARGETS["max_dd"]))["avg_monthly"])


def test_money_borrowed_past_the_equity_pays_t_bills_and_the_spread_whatever_multiple_the_target_needs():
    from tests.conftest import CASH_RATE
    rate = (CASH_RATE + metrics.FINANCING_SPREAD) / metrics.DAYS
    d = _one_fall(0.04)                         # 2.5 times its size would fall 10% before any interest
    full = db.at_target_dd(d, 1.0, 1.0)         # fully invested: all past 1 of the multiple borrowed
    k = full["at_target_dd_multiple"]
    assert 2.4 < k < 2.5                        # the interest on the money borrowed deepens the fall a little
    scaled = k * d - (k - 1.0) * rate
    assert metrics.max_drawdown(scaled)[0] == pytest.approx(FIRM_TARGETS["max_dd"], abs=1e-9)
    assert full["at_target_dd"] == pytest.approx(metrics.core(scaled)["avg_monthly"], rel=1e-9)
    # no ceiling on the multiple: a record that fell 1% is held at nearly ten times its size
    calm = db.at_target_dd(_one_fall(0.01), 1.0, 1.0)
    assert 9.5 < calm["at_target_dd_multiple"] < 10.0
    # the same record holding a tenth of the equity takes the 2.5 times the target needs from its idle money
    tenth = db.at_target_dd(d, 0.1, 0.5)
    assert tenth["at_target_dd_multiple"] == pytest.approx(2.5, rel=1e-9)
    assert tenth["at_target_dd"] == pytest.approx(metrics.core(2.5 * d)["avg_monthly"], rel=1e-9)
    # an instrument alone saved before its exposure was kept is counted all in on its days in the market
    assert db.at_target_dd(d, None, 1.0) == full


def test_the_last_five_years_are_scaled_on_their_own_and_a_record_no_longer_has_none():
    d = _days(np.random.default_rng(5).normal(0.0006, 0.01, 7 * 365))
    recent = db.at_target_dd(d.iloc[-db.RECENT_DAYS:], 0.5, 0.5)
    assert db.at_target_dd(d, 0.5, 0.5)["at_target_dd_5y"] == recent["at_target_dd"]
    assert recent["at_target_dd_5y"] is None


def test_a_record_that_lost_everything_is_held_small_enough_to_lose_only_the_target():
    d = _one_fall(1.0)
    got = db.at_target_dd(d, 1.0, 1.0)
    assert got["at_target_dd_multiple"] == pytest.approx(-FIRM_TARGETS["max_dd"], rel=1e-9)
    assert got["at_target_dd"] == pytest.approx(metrics.core(got["at_target_dd_multiple"] * d)["avg_monthly"], rel=1e-9)


def test_a_record_that_never_lost_a_day_has_no_drawdown_to_size_by():
    got = db.at_target_dd(_days(np.full(400, 0.001)), 1.0, 1.0)
    assert got == dict.fromkeys(db.AT_TARGET_DD)


def test_a_record_without_a_position_has_nothing_to_scale():
    got = db.at_target_dd(_days(np.zeros(400)), 0.0, 0.0)
    assert got == {"at_target_dd": 0.0, "at_target_dd_multiple": None, "at_target_dd_5y": None}
    assert db.at_target_dd(None, 1.0, 1.0) == dict.fromkeys(db.AT_TARGET_DD)


def test_the_single_assets_count_keeps_each_instruments_own_tries_beside_the_lists_count(conn):
    first = db.save_tries(conn, "lists", db.results_fingerprint(conn, "lists"), 3, 2.5, 2.1, "abc")
    tid = db.save_tries(conn, "assets", db.results_fingerprint(conn, "assets"), 5, 4.2, 2.6, "abc",
                        instruments={"td:SPY": (3, 2.4, 2.2), "td:QQQ": (2, 1.9, 2.0)})
    assert {r["instrument_id"]: (r["saved"], r["independent"], r["best_95"]) for r in db.instrument_tries(conn, tid)} \
        == {"td:SPY": (3, 2.4, 2.2), "td:QQQ": (2, 1.9, 2.0)}
    assert tuple(db.latest_tries(conn, "lists"))[:2] == (first, "lists")


def _import(root, path, monkeypatch):
    import import_reports
    monkeypatch.setattr(sys, "argv", ["import_reports.py", "--reports", str(root), "--db", str(path)])
    import_reports.main()


def test_results_saved_as_files_import_to_what_the_old_board_showed(tmp_path, monkeypatch):
    """The board worked out from the imported results is the board the files gave (tests/fixtures/golden_board.json,
    the old board.collect over the same folders): every figure, every check's value, threshold and state, the order."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.sqlite")
    _import(FIXTURES / "reports", tmp_path / "app.sqlite", monkeypatch)
    golden = json.loads((FIXTURES / "golden_board.json").read_text())
    n = board.tries()
    assert (n.saved, round(n.independent, 9), round(n.best_95, 9)) == (
        golden["tries"]["saved"], round(golden["tries"]["independent"], 9), round(golden["tries"]["best_95"], 9))
    got = board.collect(n_tries=n).to_dict(orient="records")
    assert [(r["strategy"], r["universe"]) for r in got] == [(r["strategy"], r["universe"]) for r in golden["rows"]]
    for want, have in zip(golden["rows"], got):
        for k, v in want.items():
            h = have[k]
            if k == "stale":
                continue                            # whether the stamps match the files on disk today
            if isinstance(v, float) or isinstance(h, float):
                v = None if v is None or v != v else float(v)
                h = None if h is None or h is pd.NA or h != h else float(h)
                assert (v is None) == (h is None) and (v is None or math.isclose(v, h, rel_tol=1e-9, abs_tol=1e-12)), k
            else:
                assert (v in (None, "") and h in (None, "")) or v == h or (k == "n_trades" and int(v) == int(h)), k


def test_a_figure_that_is_not_a_number_is_kept_as_null_and_judged_as_before(tmp_path, monkeypatch):
    root = tmp_path / "reports"
    shutil.copytree(FIXTURES / "reports" / "donchian_breakout", root / "donchian_breakout")
    card_path = root / "donchian_breakout" / "crypto_top10__1d" / "card.json"
    card = json.loads(card_path.read_text())
    card["out_of_sample"]["win_rate"] = float("nan")
    card["in_sample"]["sortino"] = float("inf")
    card_path.write_text(json.dumps(card))                     # Python writes the NaN and Infinity literals
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.sqlite")
    _import(root, tmp_path / "app.sqlite", monkeypatch)
    c = db.connect()
    rid = db.result_id(c, "donchian_breakout", "crypto_top10", "1d")
    assert db.figures(c, rid)["win_rate"] is None and db.figures(c, rid, "in_sample")["sortino"] is None
    c.close()
    row = board.collect().iloc[0]
    golden = next(r for r in json.loads((FIXTURES / "golden_board.json").read_text())["rows"]
                  if r["strategy"] == "donchian_breakout")
    # the luck bar and the familywise bar count the tries in the database: one result here, five for the golden
    got, want = json.loads(row["robustness_checks"]), json.loads(golden["robustness_checks"])
    assert [c[4] for c in got] == [c[4] for c in want]
    assert [c for c in got if c[0] not in ("luck", "vs_hold")] == [c for c in want if c[0] not in ("luck", "vs_hold")]
