"""The web UI's server: what the pages read is what the mockup showed from the same results, every answer holds to the
shapes the page's types are generated from, changes reach an open page within a second, and a re-run starts, stops
and resumes from the page."""
import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
import uvicorn
from fastapi.testclient import TestClient

from strategy_lab import board, db, lists, per_asset, runs
from strategy_lab.config import ROOT_DIR
from strategies.donchian_breakout import donchian_breakout

from server import live as live_module
from server import market_index, schemas
from server.__main__ import Server
from server.app import create_app
from server.live import Live
from server.research import MIN_TRADES, Research, asset_label, name_apart
from server.runner import Runner, batches_in_terminals
from tests.conftest import make_panel, rising_list_market, store_of
from tests.test_db import FIXTURES, _import

_CONNECT = socket.socket.connect                      # before the network ban of every test replaces it


@pytest.fixture
def imported(tmp_path, monkeypatch):
    """The fixture results (tests/fixtures/reports) imported, with the luck bar worked out over them; the logs of the
    runs and luck bars a test starts go beside it, not into the repo's logs/."""
    path = tmp_path / "app.sqlite"
    monkeypatch.setattr(db, "DB_PATH", path)
    monkeypatch.setattr(runs, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(live_module, "LOGS_DIR", tmp_path / "logs")
    _import(FIXTURES / "reports", path, monkeypatch)
    conn = db.connect(path)
    board.refresh_tries(conn)
    conn.close()
    return path


@pytest.fixture
def client(imported):
    with TestClient(create_app(imported, watch=False)) as c:
        yield c


def _close(a, b) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1e-6 * max(1, abs(a)))


# ---------------------------------------------------------------------------------------------------- the mockup's views
def test_every_list_row_is_the_one_the_mockup_showed(client):
    """tests/fixtures/golden_mockup.json holds what the old mockup build worked out from the same folders."""
    golden = json.loads((FIXTURES / "golden_mockup.json").read_text())
    labels = {c["id"]: c["label"] for c in client.get("/api/research/meta").json()["checks"]}
    got = client.get("/api/research/lists").json()["rows"]
    assert [r["key"] for r in got] == [r["key"] for r in golden["rows"]]
    for want, have in zip(golden["rows"], got):
        for k in ("strategy", "start", "end", "trades", "beats_bh", "bh_cash", "targets_met", "months", "lose"):
            assert have["loses" if k == "lose" else k] == want[k], (want["key"], k)
        for k in ("avg_monthly", "cagr", "green", "max_dd", "sharpe", "vs_bh", "bh_sharpe", "bh_max_dd", "worst_month"):
            assert _close(have[k], want[k]), (want["key"], k)
        assert have["targets"] == list(want["targets"].values())
        if want["rob"] is None:
            assert have["robustness"] is None
            continue
        rob = have["robustness"]
        assert (rob["passed"], rob["applicable"], rob["not_computed"]) == (
            want["rob"]["passed"], want["rob"]["applicable"], want["rob"]["not_computed"])
        assert [[labels[c[0]], *c[1:]] for c in rob["checks"]] == want["rob"]["checks"]


def test_every_popup_shows_what_the_mockup_showed(client):
    golden = json.loads((FIXTURES / "golden_mockup.json").read_text())
    for i, row in enumerate(golden["rows"]):
        card, key = golden["cards"][str(i)], row["key"]
        got = client.get("/api/research/result", params={"key": key}).json()
        assert got["kind"] == "list" and got["row"]["key"] == key
        for k, v in card["o"].items():
            assert _close(got["figures"][k], v), (key, k)
        for k, v in card["t"].items():
            assert _close(got["figures"][k], v), (key, k)
        assert got["curve"]["growth"] == card["g"] and got["curve"]["dd"] == card["dd"]
        assert got["curve"]["months"] == {"first": card["m"]["from"], "v": card["m"]["v"]}
        assert got["curve"]["pts"] == golden["series"][str(i)]
        if card["bh"] is None:
            assert got["benchmark"] is None
        else:
            held = golden["bh_cards"][card["bh"]]
            assert got["benchmark"]["pts"] == held["pts"] and got["benchmark"]["growth"] == held["g"]
            assert got["benchmark"]["months"] == {"first": held["m"]["from"], "v": held["m"]["v"]}
            for k, v in held["core"].items():
                assert _close(got["benchmark"]["figures"][k], v), (key, k)
        assert got["windows"] == card["f"] and got["params"] == card["p"] and got["grid"] == card["grid"]
        assert got["notes"] == card["notes"]
        assert [[n["instrument_id"], n["trades"], round(n["won"] * 1000), None if n["compounded"] is None
                 else round(n["compounded"] * 1e4)] for n in got["names"]] == sorted(card["n"])
        assert {n["instrument_id"]: n["label"] for n in got["names"]} == {
            n[0]: golden["labels"][n[0]] for n in card["n"]}
        assert got["description"] == golden["strategies"][row["strategy"]]["description"]
        curves = client.get("/api/research/curves", params={"keys": [key]}).json()["curves"]
        assert curves[key]["pts"] == golden["series"][str(i)]


def test_every_single_asset_row_is_the_one_the_mockup_showed(client):
    golden = json.loads((FIXTURES / "golden_mockup.json").read_text())
    names = golden["asset_rows"]["strategies"]
    got = {(r["strategy"], r["instrument_id"], r["timeframe"]): r for r in client.get("/api/research/assets").json()["rows"]}
    assert len(got) == len(golden["asset_rows"]["rows"])
    for w in golden["asset_rows"]["rows"]:
        have = got[names[w[0]], w[1], w[2]]
        assert have["list_id"] == lists.PER_INSTRUMENT[w[21]]
        assert [have["avg_monthly"], have["green"], have["max_dd"], have["sharpe"]] == w[3:7]
        assert (have["trades"], have["beats_bh"], have["targets_met"]) == (w[7], bool(w[8]), w[9])
        assert have["targets"] == [bool((w[10] >> k) & 1) for k in range(5)]
        assert (have["start"], have["end"], have["months"], have["params_now"]) == (w[11], w[12], w[13], json.loads(w[14]))
        assert (have["cagr"], have["stale"], have["vs_bh"], have["bh_sharpe"], have["bh_max_dd"],
                have["bh_avg_monthly"], have["bh_cash"]) == (w[15], bool(w[16]), w[17], w[18], w[19], w[20], bool(w[22]))
        assert have["held"] is False            # results imported from files carry no buy & hold series


@pytest.fixture
def alone(tmp_path, monkeypatch):
    """A breakout rule on six instruments of a rising market alone, saved as a run of a list of ours."""
    p = rising_list_market(monkeypatch)
    path = tmp_path / "app.sqlite"
    monkeypatch.setattr(db, "DB_PATH", path)
    monkeypatch.setattr(per_asset, "load_panel", store_of(p))
    monkeypatch.setattr(per_asset, "instruments_now", lambda universe, tf: (list(p.ids[:6]), []))
    per_asset.evaluate(donchian_breakout, "etf_core", "1d")
    return path


def test_a_single_assets_row_is_checked_against_the_tries_on_its_instrument_and_the_others_of_its_run(alone):
    rows, _ = Research(alone).asset_rows()
    waiting = [c for r in rows if r["robustness"] for c in r["robustness"]["checks"] if c[0] in ("luck", "vs_hold")]
    assert waiting and all(c[3] == "not_computed" for c in waiting), "no luck bar yet: its checks wait for it"
    conn = db.connect(alone)
    n = board.refresh_asset_tries(conn)
    conn.close()
    assert set(n.each) == {r["instrument_id"] for r in rows} and all(t.saved == 1 for t in n.each.values())
    research = Research(alone)
    rows, _ = research.asset_rows()
    earning = {r["instrument_id"] for r in rows if r["sharpe"] > 0 and r["cagr"] > 0}
    assert earning and len(rows) == 6
    for r in rows:
        if r["instrument_id"] not in earning:
            assert r["robustness"] is None and r["loses"], r["key"]
            continue
        checks = {c[0]: c for c in r["robustness"]["checks"]}
        assert list(checks) == board.checks_of(alone=True) and r["loses"] is None
        others = len(earning - {r["instrument_id"]})
        assert checks["peers"][1] == f"{others} of 5 others make money ({others / 5:.0%})", r["key"]
        assert f"this asset's {n.each[r['instrument_id']]}" in checks["luck"][1]
        assert f"every asset's {n.every}" in checks["luck"][1]
        got = research.result(r["key"])
        assert got["kind"] == "asset" and got["row"]["robustness"] == r["robustness"]


@pytest.fixture
def btc_stored(tmp_path, monkeypatch):
    """The fixture results' index (they are all crypto but FX's) held on a synthetic BTC, not on the store's."""
    from strategy_lab.data import bars, store
    monkeypatch.setattr(store, "STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(bars, "STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(market_index, "_panels", {})
    p = make_panel(ids=("spot:BTCUSDT",), n=3650, seed=5, start="2017-01-01 00:00")
    store.write_bars("spot", "1d", "BTCUSDT", pd.DataFrame({f: getattr(p, f)["spot:BTCUSDT"] for f in bars.FIELDS}))


def test_a_popup_and_its_chart_line_show_the_markets_index_over_the_records_days(client, btc_stored):
    lists_rows = client.get("/api/research/lists").json()["rows"]
    asset_rows = client.get("/api/research/assets").json()["rows"]
    shown = []
    for row in lists_rows + asset_rows:
        got = client.get("/api/research/result", params={"key": row["key"]}).json()
        line = client.get("/api/research/curves", params={"keys": [row["key"]]}).json()["curves"][row["key"]]
        if row["list_id"] == "fx_majors" or market_index.HELD_AS.get(row.get("instrument_id"), row.get("instrument_id")) \
                == "spot:BTCUSDT":                  # a pair is compared with cash; BTC alone is its own buy & hold
            assert got["index"] is None and line["index"] is None and line["index_name"] is None, row["key"]
            shown.append(False)
            continue
        ix, f = got["index"], got["figures"]
        assert ix["name"] == line["index_name"] == "BTC" and line["index"] == ix["pts"], row["key"]
        assert (ix["figures"]["start"], ix["figures"]["end"]) == (f["start"], f["end"]), row["key"]
        assert ix["pts"][0][0] == got["curve"]["pts"][0][0] and ix["pts"][-1][0] == got["curve"]["pts"][-1][0]
        days = (pd.Timestamp(f["end"]) - pd.Timestamp(f["start"])).days + 1
        assert abs(ix["growth"] ** (365 / days) - 1 - ix["figures"]["cagr"]) < 1e-3, row["key"]    # the line drawn
        shown.append(True)
    assert any(shown) and not all(shown)


def test_a_rows_k_ratio_is_the_one_of_the_record_its_popup_draws_and_holdings_of_its_buy_and_hold(client, imported):
    conn = db.connect(imported, readonly=True)
    rows = client.get("/api/research/lists").json()["rows"]
    held = 0
    for r in rows:
        assert _close(r["k_ratio"], db.k_ratio(db.series(conn, r["id"]))), r["key"]
        got = client.get("/api/research/result", params={"key": r["key"]}).json()
        assert got["figures"]["k_ratio"] == r["k_ratio"]
        bid = conn.execute("SELECT benchmark_id FROM result WHERE id = ?", (r["id"],)).fetchone()[0]
        if bid is None:
            assert r["bh_k_ratio"] is None and got["benchmark"] is None, r["key"]
            continue
        want = db.k_ratio(db.benchmark_series(conn, bid))
        assert _close(r["bh_k_ratio"], want) and _close(got["benchmark"]["figures"]["k_ratio"], want), r["key"]
        held += 1
    assert 0 < held < len(rows)                 # lists held and lists compared with cash
    for r in client.get("/api/research/assets").json()["rows"]:
        assert _close(r["k_ratio"], db.k_ratio(db.series(conn, r["id"]))) and r["bh_k_ratio"] is None, r["key"]
    conn.close()


def test_a_rows_buy_and_hold_at_the_target_drawdown_is_the_one_of_its_kept_series_and_its_popups(client, imported):
    """Buy & hold's average month sized to the target's drawdown holds all its money in the list at a multiple of 1,
    worked out from the series kept with it: the row's hover and the popup's metrics show that one."""
    conn = db.connect(imported, readonly=True)
    rows = client.get("/api/research/lists").json()["rows"]
    held = 0
    for r in rows:
        bid = conn.execute("SELECT benchmark_id FROM result WHERE id = ?", (r["id"],)).fetchone()[0]
        if bid is None:
            assert r["bh_at_target_dd"] is None and r["bh_at_target_dd_multiple"] is None, r["key"]
            continue
        want = db.at_target_dd(db.benchmark_series(conn, bid), 1.0, 1.0)
        assert want["at_target_dd"] is not None and want["at_target_dd_multiple"] > 0, r["key"]
        assert _close(r["bh_at_target_dd"], want["at_target_dd"]), r["key"]
        assert _close(r["bh_at_target_dd_multiple"], want["at_target_dd_multiple"]), r["key"]
        got = client.get("/api/research/result", params={"key": r["key"]}).json()["benchmark"]["figures"]
        assert all(_close(got[k], want[k]) for k in db.AT_TARGET_DD), r["key"]
        held += 1
    assert 0 < held < len(rows)                 # lists held and lists compared with cash
    for r in client.get("/api/research/assets").json()["rows"]:
        assert r["bh_at_target_dd"] is None and r["bh_at_target_dd_multiple"] is None, r["key"]
    conn.close()


def test_every_answer_holds_to_the_shapes_the_page_is_typed_with(client):
    meta = schemas.Meta.model_validate(client.get("/api/research/meta").json())
    schemas.ListRows.model_validate_json(client.get("/api/research/lists").content)
    rows = schemas.AssetRows.model_validate_json(client.get("/api/research/assets").content).rows
    for key in [r.key for r in schemas.ListRows.model_validate_json(client.get("/api/research/lists").content).rows] \
            + [rows[0].key]:
        schemas.ResultView.model_validate(client.get("/api/research/result", params={"key": key}).json())
    assert {a.id for a in meta.assets} == {r.instrument_id for r in rows}
    assert schemas.RunCurrent.model_validate(client.get("/api/runs/current").json()).run is None
    committed = json.loads((ROOT_DIR / "ui" / "web" / "openapi.json").read_text())
    assert committed == client.app.openapi(), "the API changed: `make api` writes openapi.json and the page's types"


def test_a_server_on_a_new_database_answers_every_view_with_nothing_in_it(tmp_path):
    with TestClient(create_app(tmp_path / "app.sqlite", watch=False)) as c:
        meta = schemas.Meta.model_validate(c.get("/api/research/meta").json())
        assert meta.tries is None and meta.assets == [] and meta.updated_at is None
        assert c.get("/api/research/lists").json() == {"rows": []}
        assert c.get("/api/research/assets").json() == {"rows": []}
        assert c.get("/api/runs/current").json() == {"run": None}
        assert c.get("/api/research/curves", params={"keys": ["ibs@etf_core@1d"]}).json() == {"curves": {}}


def test_rows_are_sent_compressed_once_and_not_again_while_nothing_changed(client):
    first = client.get("/api/research/assets")
    assert first.headers["content-encoding"] == "gzip"
    again = client.get("/api/research/assets", headers={"if-none-match": first.headers["etag"]})
    assert again.status_code == 304
    assert client.get("/api/research/result", params={"key": "nothing@here@1d"}).status_code == 404


def test_a_row_of_fewer_trades_than_the_bar_shows_no_average_month_at_the_target_drawdown(client, imported):
    """The figure is kept whatever the trades (db.at_target_dd); a row below MIN_TRADES shows none of it."""
    conn = db.connect(imported, readonly=True)
    kept = {r[0]: r[1:] for r in conn.execute(f"SELECT result_id, {', '.join(db.AT_TARGET_DD)} FROM result_figures "
                                              "WHERE scope = 'out_of_sample'")}
    conn.close()
    rows = client.get("/api/research/lists").json()["rows"] + client.get("/api/research/assets").json()["rows"]
    few = [r for r in rows if r["trades"] is not None and r["trades"] < MIN_TRADES]
    enough = [r for r in rows if r["trades"] is not None and r["trades"] >= MIN_TRADES]
    assert few and enough
    for r in few:
        assert kept[r["id"]][0] is not None and [r[k] for k in db.AT_TARGET_DD] == [None] * 3, r["key"]
    for r in enough:
        assert all(_close(r[k], v) for k, v in zip(db.AT_TARGET_DD, kept[r["id"]])), r["key"]
    for r in few[:3] + enough[:3]:              # the popup's metrics show the row's
        got = client.get("/api/research/result", params={"key": r["key"]}).json()["figures"]
        assert [got[k] for k in db.AT_TARGET_DD] == [r[k] for k in db.AT_TARGET_DD], r["key"]


def test_assets_are_named_apart_wherever_two_would_read_the_same():
    assets = [{"id": i, "label": asset_label(i), "market": m} for i, m in [
        ("sh:T", "Stocks"), ("perp:TUSDT", "Crypto"), ("perp:BTCUSDT", "Crypto"), ("spot:BTCUSDT", "Crypto"),
        ("td:AAPL", "Stocks"), ("sh:AAPL", "Stocks"), ("cme:GC", "CME futures"), ("sh:MSFT", "Stocks")]]
    name_apart(assets)
    assert [a["label"] for a in assets] == ["T (Stocks)", "T (Crypto)", "BTC", "BTC spot", "AAPL · Twelve Data",
                                            "AAPL", "Gold", "MSFT"]


# ---------------------------------------------------------------------------------------------------- live
@pytest.fixture
def loopback(monkeypatch):
    """This machine's own server may be reached; the network stays banned."""
    def only_loopback(sock, address):
        if not (isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1")):
            raise RuntimeError("tests must not open network connections")
        return _CONNECT(sock, address)
    monkeypatch.setattr(socket.socket, "connect", only_loopback)


@pytest.fixture
def serving(imported, loopback):
    app = create_app(imported, judge=[sys.executable, "-c", "pass"])
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield app, server.servers[0].sockets[0].getsockname()[1]
    server.should_exit = True
    thread.join(10)


class Stream:
    """A page's event stream: (event, data) as they arrive."""

    def __init__(self, port: int):
        self.conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
        self.conn.request("GET", "/api/events")
        self.resp = self.conn.getresponse()
        assert self.resp.status == 200

    def next(self) -> tuple[str, dict]:
        name = None
        while True:
            line = self.resp.fp.readline().decode()
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:") and name is not None:
                return name, json.loads(line[5:])

    def until(self, name: str) -> dict:
        while True:
            got, data = self.next()
            if got == name:
                return data

    def close(self):
        self.conn.close()


TOUCH = """
import sys
from strategy_lab import db
c = db.connect(sys.argv[1])
with db.write(c):
    c.execute("UPDATE result SET evaluated_at = evaluated_at WHERE strategy = 'gtaa_faber'")
"""


def test_a_result_saved_by_another_process_reaches_every_open_page_within_a_second(imported, serving):
    app, port = serving
    pages = [Stream(port), Stream(port)]
    assert all(p.next()[0] == "hello" for p in pages)
    t0 = time.time()
    subprocess.run([sys.executable, "-c", TOUCH, str(imported)], cwd=ROOT_DIR, check=True)
    for p in pages:
        got = p.until("results")
        assert got["keys"] == ["gtaa_faber@crypto_top3@1d"] and got["kinds"] == ["list"] and not got["every"]
    assert time.time() - t0 < 1.5 + 1.0             # the process's own start is most of it
    live = app.state.live
    assert len(live.clients) == 2
    for p in pages:
        p.close()
    deadline = time.time() + 5
    while live.clients and time.time() < deadline:
        time.sleep(0.1)
    assert not live.clients, "a page that closed its stream keeps no queue on the server"


def test_a_server_asked_to_stop_ends_the_pages_streams_and_stops_at_once(imported, loopback, caplog):
    app = create_app(imported, judge=[sys.executable, "-c", "pass"])
    server = Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", timeout_graceful_shutdown=10),
                    app.state.live)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    page = Stream(server.servers[0].sockets[0].getsockname()[1])
    assert page.next()[0] == "hello"
    t0 = time.time()
    server.handle_exit(signal.SIGTERM, None)
    thread.join(15)
    assert not thread.is_alive() and time.time() - t0 < 3, "the open stream held the stop until its grace ran out"
    assert "graceful shutdown exceeded" not in caplog.text
    page.close()


def test_the_luck_bar_is_worked_out_once_after_a_run_and_never_while_one_is_on(imported):
    conn = db.connect(imported)
    with db.write(conn):                    # a result saved after the bar: it no longer covers the results
        conn.execute("UPDATE result SET evaluated_at = '2030-01-01T00:00:00.000000Z' WHERE strategy = 'gtaa_faber'")
        rid = conn.execute("INSERT INTO run (requested_at, requested_by, single_assets, everything, state, "
                           "code_sha_at_start, pid, heartbeat_at) VALUES (?, 'ui', 0, 0, 'running', '', ?, ?) "
                           "RETURNING id", (db.now(), os.getpid(), db.now())).fetchone()[0]
    assert board.current_tries(conn) is None
    live = Live(Research(imported), Runner(imported), judge=[sys.executable, "-m", "strategy_lab", "db", "judge"])
    live.poll()
    assert live._judging is None, "a run is on: its results are judged against the last bar"
    with db.write(conn):
        conn.execute("UPDATE run SET state = 'done', pid = NULL WHERE id = ?", (rid,))
    live.poll()
    spawned = live._judging
    assert spawned is not None and spawned.wait(120) == 0
    events = []
    for _ in range(3):
        events += live.poll()
    assert live._judging is None and board.current_tries(conn) is not None
    assert [e.name for e in events].count("judged") == 1
    conn.close()


def test_a_judge_that_ends_well_leaving_a_bar_uncovered_is_not_started_again_until_the_results_change(imported):
    live = Live(Research(imported), Runner(imported), judge=[sys.executable, "-c", "pass"])     # it keeps nothing
    live.poll()
    first = live._judging
    assert first is not None and first.wait(60) == 0, "the single assets' bar was never worked out: due"
    for _ in range(4):
        live.poll()
    assert live._judging is None, "it ended well on the same results and code: not started again at every poll"
    conn = db.connect(imported)
    with db.write(conn):
        conn.execute("UPDATE result SET evaluated_at = '2030-01-01T00:00:00.000000Z' WHERE strategy = 'gtaa_faber'")
    live.poll()
    assert live._judging is not None and live._judging.wait(60) == 0, "the results changed: worked out again"
    conn.close()


WEB = ROOT_DIR / "ui" / "web"


@pytest.mark.skipif(not (WEB / "node_modules").exists() or shutil.which("npm") is None,
                    reason="the page's dependencies are not installed (make ui)")
def test_every_view_of_the_page_renders_from_the_servers_answers(serving):
    """ui/web/check/render.tsx renders the page in node from this server's answers: the lists, the single assets and
    each kind of popup, every section present, nothing in another language than English."""
    _, port = serving
    got = subprocess.run(["npm", "run", "--silent", "check:render", "--", f"http://127.0.0.1:{port}"], cwd=WEB,
                         capture_output=True, text=True, timeout=300)
    assert got.returncode == 0 and "every view rendered" in got.stdout, got.stdout + got.stderr


# ---------------------------------------------------------------------------------------------------- runs
FAKE_RUN = textwrap.dedent("""
    import sys, time
    from strategy_lab import catalog, db, runs, strategy_pick
    def job(j):
        job_id, stage, name, universe, tf = j
        c = runs._conn()
        runs._mark(job_id, "state = 'running', attempts = attempts + 1, started_at = ?", db.now())
        first = c.execute("SELECT attempts FROM run_job WHERE id = ?", (job_id,)).fetchone()[0] == 1
        if universe == "etf_core" and first:
            time.sleep(120)                     # the job a stop cuts off: quick when resumed
        runs._mark(job_id, "state = 'done', finished_at = ?", db.now())
    runs._job = job
    strategy_pick.pick_all = lambda conn: 0          # the stages after the jobs read every result and every list's
    catalog.refresh = lambda conn=None: None         # bars: nothing a fake run's jobs made
    raise SystemExit(runs.execute(int(sys.argv[1]), workers=1))
""")
ONLY = {"strategies": ["sma_cross"], "lists": ["us_stocks_top10", "etf_core", "fx_majors"], "timeframes": ["1d"]}


def _wait(client, test, seconds=60) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        run = client.get("/api/runs/current").json()["run"]
        if run is not None and test(run):
            return run
        time.sleep(0.2)
    raise AssertionError(f"the run never got there: {run}")


def _group_gone(runner: Runner, pgid: int, seconds: float = 10) -> bool:
    """Every process of the run's group ended (and reaped: the server's watcher reaps the runs it started)."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        runner.supervise()
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:                     # macOS: only processes that ended and are not reaped yet
            pass
        time.sleep(0.1)
    return False


def test_a_run_started_from_the_page_stops_at_once_and_resumes_where_it_stopped(imported):
    app = create_app(imported, run_command=lambda rid: [sys.executable, "-c", FAKE_RUN, str(rid)], watch=False)
    with TestClient(app) as client:
        started = client.post("/api/runs", json={"everything": True, "only": ONLY, "confirm": True})
        assert started.status_code == 202
        rid = started.json()["id"]
        run = _wait(client, lambda r: r["state"] == "running" and any(j["list_id"] == "etf_core" for j in r["running"]))
        assert run["narrowed_to"] == "Stocks Top-10, ETFs All 34, FX 7 majors · sma_cross · 1d"
        assert run["stages"][0]["total"] == 3
        again = client.post("/api/runs", json={"everything": True, "only": ONLY, "confirm": True})
        assert again.status_code == 409 and f"Run {rid} is on" in again.json()["message"]
        conn = db.connect(imported)
        pgid = conn.execute("SELECT pgid FROM run WHERE id = ?", (rid,)).fetchone()[0]
        done_before = {r[0]: r[1] for r in conn.execute("SELECT list_id, finished_at FROM run_job WHERE run_id = ? "
                                                        "AND state = 'done'", (rid,))}
        t0 = time.time()
        assert client.post(f"/api/runs/{rid}/stop").status_code == 202
        run = _wait(client, lambda r: r["state"] == "stopped", 30)
        assert time.time() - t0 < 10 and _group_gone(app.state.runner, pgid)
        assert {s: n for s, n in run["stages"][0].items() if n and s not in ("stage", "total")} == {
            "done": len(done_before), "stopped": 1, "queued": 3 - len(done_before) - 1}
        assert client.post(f"/api/runs/{rid}/stop").status_code == 409
        assert client.post(f"/api/runs/{rid}/resume", json={"confirm": True}).status_code == 202
        run = _wait(client, lambda r: r["state"] == "done")
        assert run["id"] == rid and run["resumed"] == 1 and run["stages"][0]["done"] == 3
        after = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT list_id, finished_at, attempts FROM run_job WHERE "
                                                          "run_id = ?", (rid,))}
        assert all(after[u][0] == t and after[u][1] == 1 for u, t in done_before.items()), "a job done ran again"
        assert after["etf_core"][1] == 2
        assert client.post(f"/api/runs/{rid}/resume", json={}).status_code == 409      # done, nothing failed
        log_path = conn.execute("SELECT log_path FROM run WHERE id = ?", (rid,)).fetchone()[0]
        assert Path(log_path) == runs.log_path(rid) and runs.log_path(rid).exists()
        conn.close()


def test_a_run_whose_process_is_killed_is_interrupted_and_resumes_after_asking_about_the_changed_code(imported):
    app = create_app(imported, run_command=lambda rid: [sys.executable, "-c", FAKE_RUN, str(rid)], watch=False)
    with TestClient(app) as client:
        rid = client.post("/api/runs", json={"everything": True, "only": ONLY, "confirm": True}).json()["id"]
        _wait(client, lambda r: r["state"] == "running" and any(j["list_id"] == "etf_core" for j in r["running"]))
        conn = db.connect(imported)
        pgid = conn.execute("SELECT pgid FROM run WHERE id = ?", (rid,)).fetchone()[0]
        os.killpg(pgid, signal.SIGKILL)                  # a crash: nothing marks the run's end
        assert _group_gone(app.state.runner, pgid)
        run = client.get("/api/runs/current").json()["run"]
        assert run["state"] == "interrupted" and not run["active"] and "exit code -9" in run["error"]
        assert {j["list_id"] for j in run["running"]} == set()
        with db.write(conn):                             # the evaluation code changed since the run started
            conn.execute("UPDATE run SET code_sha_at_start = 'older' WHERE id = ?", (rid,))
        asked = client.post(f"/api/runs/{rid}/resume", json={})
        assert asked.status_code == 409 and asked.json()["confirm"] and "changed since" in asked.json()["message"]
        assert client.post(f"/api/runs/{rid}/resume", json={"confirm": True}).status_code == 202
        run = _wait(client, lambda r: r["state"] == "done")
        assert run["resumed"] == 1 and run["stages"][0]["done"] == 3
        conn.close()


def test_a_batch_started_in_a_terminal_on_this_checkout_is_found_and_another_checkouts_is_not(tmp_path):
    here, there = tmp_path / "here", tmp_path / "there"
    for root in (here, there):
        (root / "scripts").mkdir(parents=True)
        (root / "scripts" / "week1.py").write_text("import time\ntime.sleep(30)\n")
    p = subprocess.Popen([sys.executable, "scripts/week1.py"], cwd=here)
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not batches_in_terminals(here):
            time.sleep(0.2)
        assert [x.split(":")[0] for x in batches_in_terminals(here)] == [str(p.pid)]
        assert batches_in_terminals(there) == []
    finally:
        p.send_signal(signal.SIGTERM)
        p.wait(10)
