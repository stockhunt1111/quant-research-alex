"""What the Research page reads, worked out from the app's database: the rows of both views, a result's popup, the
lines of the P&L chart and the page's meta. The market's index beside a result's buy & hold is held on the bar store's
prices when it is asked for (`market_index`).

The figures are the evaluation's own; what is judged from them (a check's state, the deflated Sharpe, whether a result
is stale, why a result is not checked) comes from the functions the board uses (strategy_lab.board), so the page and
`python -m strategy_lab board` never disagree. A list result's row is judged once per (evaluation, luck bar, code on
disk) and kept; the single assets' rows are read again when a result changes, each one's robustness judged once per
(evaluation, luck bar, the results of its market's other instruments) and kept. Each view's answer is kept compressed
until the database or the code changes.

The curves and months follow the mockup the user approved (ui/mockup, before this server): growth of 1 at each week's
end, thinned to at most CAP points and ending on the record's last week, the drawdown's low between two points kept;
complete calendar months in basis points.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import sqlite3
import threading
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from strategy_lab import board, db, lists, log, metrics, provenance, significance
from strategy_lab.config import FIRM_TARGETS, RESEARCH_CAPITAL_USD

from server import market_index

LOG = log.get("server.research")
CAP = 520                             # the most points a curve carries: a year a week for ten years
MARKETS = ("Stocks", "ETFs", "Crypto", "FX", "Commodities", "CME futures")
VENUE = {"perp": "Binance perpetual futures", "spot": "Binance spot"}     # a coin trades on either: its list says which
VENDOR = {"td": "Twelve Data", "sh": "Sharadar", "perp": "Binance perpetual futures", "spot": "Binance spot",
          "cme": "CME"}
CME_LABEL = {"GC": "Gold", "SI": "Silver", "PL": "Platinum", "PA": "Palladium", "CL": "WTI crude", "HG": "Copper"}
TARGET_KEYS = list(db.TARGETS)
TARGET_COLUMNS = list(db.TARGETS.values())
PICK = "strategy_pick"
GZIP_LEVEL = 5                        # level 9 took ten times as long for a few percent less
NO_TRIES = board.Tries(0, 1.0, 1.6448536269514722)       # stands in until the first luck bar is worked out
NO_LUCK = {"noise_bar": math.nan, "prob": math.nan}       # a record's deflated Sharpe when it cannot be worked out
# a record of fewer trades has no At 10% DD on its row: its drawdown measures nothing, and sized to the target it can
# read hundreds of percent a month; the research desk greys a strategy's trade figures below 30 ("Under 30 trades none
# of those is a measurement")
MIN_TRADES = 30


# ---------------------------------------------------------------------------------------------------- the mockup's views
def fnum(v, nd: int = 6) -> float | None:
    if v is None or v is pd.NA:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def whole(v) -> int | None:
    v = fnum(v)
    return None if v is None else int(v)


def at_target_dd(trades: int | None, figures: dict) -> dict:
    """A row's average month at the target's drawdown, its multiple and its last five years (db.AT_TARGET_DD), none for
    a record of fewer than MIN_TRADES trades whatever the figures kept; a record whose trades are not counted (a
    strategy_pick's) keeps them."""
    if trades is not None and trades < MIN_TRADES:
        return dict.fromkeys(db.AT_TARGET_DD)
    return {k: fnum(figures[k]) for k in db.AT_TARGET_DD}


def bp(v) -> int | None:
    """A return in basis points (1 = 0.01%): the compact form of the popup's months and drawdowns."""
    v = fnum(v)
    return None if v is None else int(round(v * 1e4))


def ms(ts) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


def weekly(daily: pd.Series | None, cap: int = CAP) -> tuple[list, list]:
    """Growth of 1 (compounded) at the end of each week, thinned to at most `cap` points and always ending on the
    record's last week, the first point the start at 1; and at each point, in basis points, the drawdown's low since
    the point before it (a trough inside the weeks a thinned curve skips is kept)."""
    d = pd.Series(dtype=float) if daily is None else daily.dropna()
    if d.empty:
        return [], []
    eq = (1.0 + d).cumprod()
    wk = eq.resample("W-SUN").last().dropna()
    low = (eq / eq.cummax().clip(lower=1.0) - 1.0).resample("W-SUN").min().reindex(wk.index)     # from the start at 1
    step = max(1, int(np.ceil(len(wk) / cap)))
    keep = np.unique(np.r_[np.arange(0, len(wk), step), len(wk) - 1])
    lows = pd.Series(low.to_numpy()).groupby(np.searchsorted(keep, np.arange(len(wk)))).min()
    pts = [[ms(d.index[0]), 1.0]] + [[ms(t), round(float(v), 4)] for t, v in wk.iloc[keep].items()]
    return pts, [0] + [bp(v) for v in lows]


def months_bp(daily: pd.Series | None) -> dict | None:
    """The complete calendar months' compounded returns in basis points, month after month from the first."""
    if daily is None:
        return None
    m = metrics.monthly_returns(daily.dropna())
    if m.empty:
        return None
    m = m.reindex(pd.period_range(m.index[0], m.index[-1], freq="M"))
    return {"first": str(m.index[0]), "v": [bp(v) for v in m]}


def growth(daily: pd.Series | None) -> float | None:
    return None if daily is None else fnum(float((1.0 + daily.dropna()).prod()), 4)


def windows_view(wins: list[dict]) -> dict | None:
    """The walk-forward's test windows: each one's first day and the parameter set it traded (an index into `sets`),
    and the days of history the first one was chosen on; None for a result with one configuration."""
    if not wins:
        return None
    keys = [json.dumps(w["params"], sort_keys=True) for w in wins]
    order = list(dict.fromkeys(keys))
    return {"sets": [json.loads(k) for k in order], "end": wins[-1]["test_end"],
            "train": (pd.Timestamp(wins[0]["test_start"]) - pd.Timestamp(wins[0]["train_start"])).days,
            "w": [[w["test_start"], order.index(k)] for w, k in zip(wins, keys)]}


def asset_label(i: str) -> str:
    """A coin by its symbol alone, as a stock by its ticker: whether it is a perp or spot is said once, by its list."""
    src, sym = i.split(":", 1)
    if src == "cme":
        return CME_LABEL.get(sym, sym)
    return sym.removesuffix("USDT") if src in VENUE else sym


def name_apart(assets: list[dict]) -> None:
    """Every asset under a label of its own: a coin and a stock under one ticker (AT&T and Threshold are both T) take
    their market; a coin on both Binance markets (the ML task's list trades spot, ours perpetual futures) is named so on
    spot, the rarer of the two; a stock kept under two vendors' ids (a result on Twelve Data's bars from before the
    stock lists moved to Sharadar's, beside one on Sharadar's) is named by its vendor on the one that is not
    Sharadar's. Two still alike are named by their ids."""
    marks = (lambda a, same: f" ({a['market']})" if len({x["market"] for x in same}) > 1 else "",
             lambda a, same: " spot" if a["id"].startswith("spot:") else "",
             lambda a, same: "" if a["id"].startswith("sh:") else f" · {VENDOR.get(a['id'].split(':', 1)[0], '?')}")
    for mark in marks:
        groups = defaultdict(list)
        for a in assets:
            groups[a["label"]].append(a)
        for same in (g for g in groups.values() if len(g) > 1):
            for a in same:
                a["label"] += mark(a, same)
    left = {k for k, n in Counter(a["label"] for a in assets).items() if n > 1}
    for a in assets:
        if a["label"] in left:
            a["label"] = a["id"]


def list_key(strategy: str, list_id: str, tf: str) -> str:
    return f"{strategy}@{list_id}@{tf}"


def asset_key(strategy: str, list_id: str, tf: str, instrument: str) -> str:
    return f"{strategy}@{list_id}@{tf}@{instrument}"


def parse_key(key: str) -> tuple[str, str, str, str | None]:
    """A result's key, `strategy@list@timeframe` or `strategy@list@timeframe@instrument`."""
    parts = key.split("@")
    if len(parts) not in (3, 4) or not all(parts):
        raise KeyError(key)
    return parts[0], parts[1], parts[2], parts[3] if len(parts) == 4 else None


def dumps(v) -> bytes:
    return json.dumps(v, separators=(",", ":"), allow_nan=False).encode()


def _loads(v):
    return None if v is None else json.loads(v)


def _cell(x: dict) -> dict:
    """A result as a cell of Same strategy elsewhere."""
    return {k: x[k] for k in ("key", "list_id", "timeframe", "sharpe", "avg_monthly", "cagr", "vs_bh", "beats_bh",
                              "bh_cash", "targets_met", "stale")}


def _pct(v) -> str:
    return "" if pd.isna(v) else f"{v * 100:+.2f}%"


def _reason(r) -> str:
    """Why a result falls short, in words: it loses money, compounded or not, or it does not beat holding."""
    if r.sharpe <= 0:
        return f"loses money out-of-sample (Sharpe {r.sharpe:.2f}; {r.is_sharpe:.2f} with parameters fitted in-sample)"
    if r.cagr <= 0:
        return f"its Sharpe is positive but compounded it loses {abs(r.cagr) * 100:.2f}% a year: the swings eat the gains"
    if not r.beats_bh:
        if r.bh_cash:
            return f"does not beat cash: {_pct(r.vs_bh)} a year against T-bills (nothing in the list can be held)"
        return f"does not beat buy & hold: sized to its risk, {_pct(r.vs_bh)} a year against holding the same list"
    return r.why_not


def _loses(j: dict) -> str:
    """Why a result is not checked for robustness: how it loses money out-of-sample."""
    if j["sharpe"] is None:
        return "no out-of-sample figures"
    if j["cagr"] is None:                   # an account that lost everything has no compounded return
        return "loses everything out-of-sample"
    if j["sharpe"] <= 0:
        return f"loses money out-of-sample (Sharpe {j['sharpe']:.2f})"
    return _reason(SimpleNamespace(**j))


def _earns(r) -> bool:
    return r["sharpe"] is not None and r["cagr"] is not None and r["sharpe"] > 0 and r["cagr"] > 0


def _rob(items: list[list], luck_known: bool) -> dict:
    """A row's robustness as the page shows it, its checks' [id, value, threshold, state]; while the luck bar is being
    worked out, the checks it decides are not judged yet."""
    if not luck_known:
        for c in items:
            if c[0] in ("luck", "vs_hold") and c[3] in ("passed", "failed", "too_short"):
                c[1], c[2], c[3] = "the luck bar is being worked out", "", "not_computed"
    return {"passed": sum(c[3] == "passed" for c in items),
            "applicable": sum(c[3] in ("passed", "failed", "too_short") for c in items),
            "not_computed": sum(c[3] == "not_computed" for c in items), "checks": items}


# ---------------------------------------------------------------------------------------------------- the research view
_ASSET_SQL = (
    "SELECT r.id, r.strategy, r.list_id, r.instrument_id, r.timeframe, r.evaluated_at, r.code_sha, r.vs_bh, r.beats_bh, "
    "r.bh_cash, "
    "r.bh_sharpe, r.bh_max_dd, r.bh_avg_monthly, r.targets_met, r.params_now, r.benchmark_id, "
    f"{', '.join('r.' + c for c in TARGET_COLUMNS)}, "
    "o.start, o.end, o.months, o.avg_monthly, o.cagr, o.pct_green_active, o.max_dd, "
    f"{', '.join('o.' + c for c in db.AT_TARGET_DD)}, o.sharpe, o.k_ratio, o.n_trades, "
    "b.k_ratio AS bh_k_ratio, b.at_target_dd AS bh_at_target_dd, b.at_target_dd_multiple AS bh_at_target_dd_multiple "
    "FROM result r LEFT JOIN result_figures o ON o.result_id = r.id AND o.scope = 'out_of_sample' "
    "LEFT JOIN benchmark b ON b.id = r.benchmark_id "
    "WHERE r.instrument_id IS NOT NULL {where} ORDER BY r.id")


class Research:
    """The server's view of one database: a read-only connection a thread, the stamps judged, the rows kept."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._local = threading.local()
        self._lock = threading.RLock()
        self._judged: dict[int, tuple[tuple, dict]] = {}      # list result id -> (what it was judged on, its row)
        self._lists: tuple[tuple, list[dict], dict[int, dict]] | None = None
        self._assets: tuple[tuple, list[dict], dict[str, dict], list[dict]] | None = None
        self._asset_judged: dict[int, tuple[tuple, tuple]] = {}   # asset result id -> (what it was judged on, judged)
        self._payloads: dict[str, tuple[tuple, str, bytes]] = {}
        self._current: dict[str, bool] = {}
        self.code_version = 0

    # -------------------------------------------------------------------------------------------- plumbing
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._local.conn = db.connect(self.path, readonly=True)
        return c

    def seq(self) -> int:
        """The last change of a result (the change log keeps a day; its counter keeps counting)."""
        r = self.conn().execute("SELECT seq FROM sqlite_sequence WHERE name = 'change'").fetchone()
        return 0 if r is None else r[0]

    def tries(self) -> tuple[int | None, board.Tries | None]:
        """The last luck bar kept for the lists (a result saved since is judged by it until it is worked out again)."""
        row = db.latest_tries(self.conn(), "lists")
        return (None, None) if row is None else (row["id"], board.Tries(row["saved"], row["independent"],
                                                                        row["best_95"]))

    def asset_tries(self) -> tuple[int | None, board.AssetTries | None]:
        """The last luck bar kept for the single assets (a result saved since is judged by it until it is worked out
        again)."""
        got = board.last_asset_tries(self.conn())
        return (None, None) if got is None else got

    def _asset_tries_id(self) -> int | None:
        row = db.latest_tries(self.conn(), "assets")
        return None if row is None else row["id"]

    def picks_tries(self) -> board.Tries | None:
        row = db.latest_tries(self.conn(), "picks")
        return None if row is None else board.Tries(row["saved"], row["independent"], row["best_95"])

    def is_current(self, sha: str, files: list[str]) -> bool:
        with self._lock:
            if sha not in self._current:
                self._current[sha] = provenance.is_current({"files": files, "sha": sha})
            return self._current[sha]

    def code_changed(self) -> None:
        """The code on disk changed: every stamp is judged again."""
        with self._lock:
            self._current.clear()
            self.code_version += 1

    def _stamps(self) -> dict[str, bool]:
        return {r["sha"]: self.is_current(r["sha"], json.loads(r["files"]))
                for r in self.conn().execute("SELECT sha, files FROM code_stamp")}

    # -------------------------------------------------------------------------------------------- lists
    def _lists_version(self) -> tuple:
        tries_id, _ = self.tries()
        return self.seq(), tries_id, self.code_version

    def list_rows(self) -> list[dict]:
        """Every result of a strategy on one of our lists as one book, judged, in the board's order."""
        with self._lock:
            version = self._lists_version()
            if self._lists is not None and self._lists[0] == version:
                return self._lists[1]
            c = self.conn()
            tries_id, n_tries = self.tries()
            frame = db.results_frame(c, "list")
            frame = board.ranked_only(frame)
            rows = [self._judge(r, tries_id, n_tries) for r in frame.itertuples(index=False)]
            ids = set(frame["id"])
            self._judged = {k: v for k, v in self._judged.items() if k in ids}
            if rows:
                order = board.ordered(pd.DataFrame({
                    "n": range(len(rows)), "sharpe": [np.nan if x["sharpe"] is None else x["sharpe"] for x in rows],
                    "cagr": [np.nan if x["cagr"] is None else x["cagr"] for x in rows],
                    "targets_met": [x["targets_met"] for x in rows]}))
                rows = [rows[k] for k in order["n"]]
            self._lists = (version, rows, {x["id"]: x for x in rows})
            return rows

    def _judge(self, r, tries_id: int | None, n_tries: board.Tries | None) -> dict:
        current = self.is_current(r.code_sha, r.code_files)
        seen = (r.evaluated_at, tries_id, current)
        kept = self._judged.get(r.id)
        if kept is not None and kept[0] == seen:
            return kept[1]
        c = self.conn()
        judged = board.row_of(r, db.measures(c, r.id), db.series(c, r.id), n_tries or NO_TRIES, current)
        row = self._list_row(r, judged, n_tries is not None)
        self._judged[r.id] = (seen, row)
        return row

    @staticmethod
    def _list_row(r, j: dict, luck_known: bool) -> dict:
        checks = json.loads(j["robustness_checks"]) if j["robustness_checks"] else None
        rob = None if checks is None else _rob([[cid, value, threshold, state]
                                                for cid, _label, value, threshold, state in checks], luck_known)
        j = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in j.items()}
        return {
            "key": list_key(r.strategy, r.list_id, r.timeframe), "id": int(r.id), "strategy": r.strategy,
            "list_id": r.list_id, "timeframe": r.timeframe, "start": r.start, "end": r.end, "months": whole(r.months),
            "avg_monthly": fnum(r.avg_monthly), "cagr": fnum(r.cagr), "green": fnum(r.pct_green_active),
            "max_dd": fnum(r.max_dd), **at_target_dd(whole(r.n_trades), {k: getattr(r, k) for k in db.AT_TARGET_DD}),
            "sharpe": fnum(r.sharpe), "k_ratio": fnum(r.k_ratio), "trades": whole(r.n_trades),
            "beats_bh": bool(r.beats_bh), "vs_bh": fnum(r.vs_bh), "bh_cash": bool(r.bh_cash),
            "bh_sharpe": fnum(r.bh_sharpe), "bh_max_dd": fnum(r.bh_max_dd), "bh_k_ratio": fnum(r.bh_k_ratio),
            "bh_at_target_dd": fnum(r.bh_at_target_dd), "bh_at_target_dd_multiple": fnum(r.bh_at_target_dd_multiple),
            "targets": [bool(r.targets[k]) for k in TARGET_KEYS], "targets_met": int(r.targets_met),
            "robustness": rob, "loses": _loses(j) if rob is None else None, "stale": bool(j["stale"]),
            "worst_month": fnum(r.worst_month), "evaluated_at": r.evaluated_at}

    def list_row(self, rid: int) -> dict | None:
        """A list result's row: from the kept rows, or judged now for a result off our lists (an ad-hoc run)."""
        self.list_rows()
        with self._lock:
            got = self._lists[2].get(rid)
        if got is not None:
            return got
        frame = db.results_frame(self.conn(), "list", "r.id = ?", (rid,))
        if frame.empty:
            return None
        tries_id, n_tries = self.tries()
        with self._lock:
            return self._judge(next(frame.itertuples(index=False)), tries_id, n_tries)

    # -------------------------------------------------------------------------------------------- single assets
    def _assets_version(self) -> tuple:
        as_of = self.conn().execute("SELECT max(as_of) FROM instrument_stats").fetchone()[0]
        return self.seq(), self.code_version, as_of, self._asset_tries_id()

    def _series_of(self, ids: list[int]) -> dict[int, pd.Series]:
        """The out-of-sample records of results `ids`."""
        out = {}
        for k in range(0, len(ids), 900):                     # SQLite's limit of bound values a statement
            chunk = ids[k:k + 900]
            for r in self.conn().execute("SELECT result_id, first_day, days, series FROM result_series WHERE kind = "
                                         f"'out_of_sample' AND result_id IN ({', '.join('?' * len(chunk))})", chunk):
                out[r["result_id"]] = db.decode_series(r["first_day"], r["days"], r["series"])
        return out

    def _judge_assets(self, rows: list[sqlite3.Row], runs: list[sqlite3.Row]) -> dict[int, tuple[dict | None, str | None]]:
        """Each of `rows`' robustness as the page shows it and why a row has none; `runs`: the single-asset results of
        the runs `rows` come from, whose other instruments a row's `peers` check counts (`board.peers`)."""
        tries_id, n = self.asset_tries()
        alone = [r for r in runs if r["strategy"] != PICK]
        peers = board.peers(pd.DataFrame([(r["id"], r["strategy"], r["list_id"], r["timeframe"], r["sharpe"], r["cagr"])
                                          for r in alone], columns=["id", "strategy", "list_id", "timeframe", "sharpe",
                                                                    "cagr"])) if alone else {}
        out, todo = {}, []
        with self._lock:
            for r in rows:
                seen = (r["evaluated_at"], tries_id, json.dumps(peers.get(r["id"]), sort_keys=True))
                kept = self._asset_judged.get(r["id"])
                if kept is not None and kept[0] == seen:
                    out[r["id"]] = kept[1]
                else:
                    todo.append((r, seen))
        checked = [r["id"] for r, _ in todo if r["strategy"] != PICK and _earns(r)]
        measured = db.measures_of(self.conn(), checked)
        daily = self._series_of([i for i in checked if i in measured])
        older = 0
        for r, seen in todo:
            got = self._judge_asset(r, measured.get(r["id"]), daily.get(r["id"]), peers.get(r["id"]), n)
            older += got[0] is not None and got[0]["not_computed"] == len(got[0]["checks"])
            out[r["id"]] = got
            with self._lock:
                self._asset_judged[r["id"]] = (seen, got)
        if older:
            LOG.info("%d single-asset results that make money were saved by code older than their checks: re-run them",
                     older)
        return out

    @staticmethod
    def _judge_asset(r: sqlite3.Row, measured: dict | None, daily: pd.Series | None, peer: dict | None,
                     n: board.AssetTries | None) -> tuple[dict | None, str | None]:
        """A single-asset result's robustness (`board.robustness(alone=True)`, against the tries on its instrument and
        those on every instrument beside them) and why it has none."""
        if r["strategy"] == PICK:
            return None, "a choice among the strategies run on it alone, not a strategy: its luck is on its popup"
        if not _earns(r):
            return None, _loses({"sharpe": r["sharpe"], "cagr": r["cagr"]})
        if measured is None:                    # saved by code older than the checks
            return _rob([[cid, "not measured in this evaluation: older code", "", "not_computed"]
                         for cid in board.checks_of(alone=True)], True), None
        own = None if n is None else n.each.get(r["instrument_id"])
        luck, wide = NO_LUCK, NO_LUCK
        if daily is not None and n is not None:
            wide, *mine = significance.deflated_each(daily, [n.every.independent] + ([] if own is None
                                                                                    else [own.independent]))
            luck = mine[0] if mine else NO_LUCK
        card = {"strategy": r["strategy"], "universe": r["instrument_id"], "timeframe": r["timeframe"],
                "out_of_sample": {"sharpe": r["sharpe"]},
                "robustness": measured | ({} if peer is None else {"peers": peer})}
        rob = board.robustness(card, daily, own or NO_TRIES, luck, alone=True,
                               every=None if n is None else (n.every, wide))
        return _rob([[x.id, x.value, x.threshold, x.state] for x in rob.checks], own is not None), None

    @staticmethod
    def _asset_row(r: sqlite3.Row, current: bool, rob: dict | None, loses: str | None) -> dict:
        return {
            "key": asset_key(r["strategy"], r["list_id"], r["timeframe"], r["instrument_id"]), "id": r["id"],
            "strategy": r["strategy"], "instrument_id": r["instrument_id"], "list_id": r["list_id"],
            "timeframe": r["timeframe"], "start": r["start"], "end": r["end"], "months": whole(r["months"]),
            "avg_monthly": fnum(r["avg_monthly"]), "cagr": fnum(r["cagr"]), "green": fnum(r["pct_green_active"]),
            "max_dd": fnum(r["max_dd"]), **at_target_dd(whole(r["n_trades"]), {k: r[k] for k in db.AT_TARGET_DD}),
            "sharpe": fnum(r["sharpe"]), "k_ratio": fnum(r["k_ratio"]),
            "trades": whole(r["n_trades"]), "beats_bh": bool(r["beats_bh"]), "vs_bh": fnum(r["vs_bh"]),
            "bh_cash": bool(r["bh_cash"]), "bh_sharpe": fnum(r["bh_sharpe"]), "bh_max_dd": fnum(r["bh_max_dd"]),
            "bh_k_ratio": fnum(r["bh_k_ratio"]), "bh_at_target_dd": fnum(r["bh_at_target_dd"]),
            "bh_at_target_dd_multiple": fnum(r["bh_at_target_dd_multiple"]),
            "bh_avg_monthly": fnum(r["bh_avg_monthly"]), "targets": [bool(r[c]) for c in TARGET_COLUMNS],
            "targets_met": int(r["targets_met"]), "params_now": _loads(r["params_now"]),
            "held": r["benchmark_id"] is not None, "robustness": rob, "loses": loses, "stale": not current}

    def asset_rows(self) -> tuple[list[dict], list[dict]]:
        """Every strategy alone on an instrument, once a (strategy, instrument, timeframe) — from the list earlier in
        `lists.PER_INSTRUMENT` — and the instruments they name, labelled apart, the most liquid first."""
        with self._lock:
            version = self._assets_version()
            if self._assets is not None and self._assets[0] == version:
                return self._assets[1], self._assets[3]
            c = self.conn()
            stamps = self._stamps()
            rank = {u: k for k, u in enumerate(lists.PER_INSTRUMENT)}
            runs = [r for r in c.execute(_ASSET_SQL.format(where="")) if r["list_id"] in rank]
            picked: dict[tuple, sqlite3.Row] = {}
            for r in runs:
                k = (r["strategy"], r["instrument_id"], r["timeframe"])
                if k not in picked or rank[r["list_id"]] < rank[picked[k]["list_id"]]:
                    picked[k] = r
            judged = self._judge_assets(list(picked.values()), runs)
            with self._lock:
                self._asset_judged = {k: v for k, v in self._asset_judged.items() if k in judged}
            rows = [self._asset_row(r, stamps[r["code_sha"]], *judged[r["id"]]) for r in sorted(
                picked.values(), key=lambda r: (rank[r["list_id"]], r["id"]))]
            market: dict[str, str] = {}
            for x in rows:
                market.setdefault(x["instrument_id"], lists.per_instrument_market(x["list_id"]))
            liq = {r[0]: r[1] for r in c.execute("SELECT instrument_id, liquidity_usd FROM instrument_stats")}
            assets = [{"id": i, "label": asset_label(i), "market": m, "liquidity": fnum(liq.get(i), 0)}
                      for i, m in market.items()]
            name_apart(assets)
            assets.sort(key=lambda a: (-(a["liquidity"] or 0), a["label"]))
            self._assets = (version, rows, {x["key"]: x for x in rows}, assets)
            return rows, assets

    def asset_row(self, key: str) -> dict | None:
        """A single-asset result's row: from the kept rows, or read now (a result another list's run repeats)."""
        self.asset_rows()
        with self._lock:
            got = self._assets[2].get(key)
        if got is not None:
            return got
        strategy, list_id, tf, inst = parse_key(key)
        run = self.conn().execute(_ASSET_SQL.format(where="AND r.strategy = ? AND r.list_id = ? AND r.timeframe = ?"),
                                  (strategy, list_id, tf)).fetchall()
        r = next((x for x in run if x["instrument_id"] == inst), None)
        if r is None:
            return None
        return self._asset_row(r, self._stamps()[r["code_sha"]], *self._judge_assets([r], run)[r["id"]])

    # -------------------------------------------------------------------------------------------- answers kept compressed
    def rows_payload(self, view: str) -> tuple[str, bytes]:
        """A view's rows as gzipped JSON with its ETag, worked out again only after the database or the code changed."""
        with self._lock:
            version = self._lists_version() if view == "lists" else self._assets_version()
            kept = self._payloads.get(view)
            if kept is not None and kept[0] == version:
                return kept[1], kept[2]
            rows = self.list_rows() if view == "lists" else self.asset_rows()[0]
            body = gzip.compress(dumps({"rows": rows}), GZIP_LEVEL)
            etag = '"' + hashlib.sha1(repr((view, version)).encode()).hexdigest()[:16] + '"'
            self._payloads[view] = (version, etag, body)
            return etag, body

    # -------------------------------------------------------------------------------------------- the page's meta
    def meta(self) -> dict:
        c = self.conn()
        stats = {r["list_id"]: r for r in c.execute("SELECT * FROM list_stats")}
        out_lists = []
        for x in lists.OURS + lists.ML_TASK_LISTS:
            s = stats.get(x.universe)
            sources = json.loads(s["sources"]) if s else []
            out_lists.append({
                "id": x.universe, "market": x.market, "label": x.label, "title": lists.title(x.universe),
                "research": lists.market(x.universe) is not None,
                "per_instrument": x.universe in lists.PER_INSTRUMENT,
                "venue": ", ".join(VENUE[v] for v in sources if v in VENUE) or None,
                "costs": None if s is None else (f"{s['cost_bp_min']:g}bp" if s["cost_bp_min"] == s["cost_bp_max"]
                                                 else f"{s['cost_bp_min']:g}bp to {s['cost_bp_max']:g}bp"),
                "size": None if s is None else {"fixed": s["size_fixed"], "trading": s["size_trading"],
                                                 "held": s["size_held"], "ever": s["size_ever"]}})
        _, n = self.tries()
        rows = self.list_rows()
        assets_rows, assets = self.asset_rows()
        return {
            "targets": {k: FIRM_TARGETS[k] for k in ("avg_monthly", "avg_monthly_stretch", "pct_green",
                                                     "pct_green_stretch", "max_dd", "sharpe")},
            "capital": RESEARCH_CAPITAL_USD, "min_trades": MIN_TRADES, "markets": list(MARKETS), "lists": out_lists,
            "checks": [{"id": k, "label": v} for k, v in board.CHECKS.items()], "assets": assets,
            "strategies": {"lists": sorted({x["strategy"] for x in rows}),
                           "assets": sorted({x["strategy"] for x in assets_rows})},
            "tries": None if n is None else {"saved": n.saved, "independent": n.independent, "best_95": n.best_95},
            "updated_at": c.execute("SELECT max(evaluated_at) FROM result").fetchone()[0],
            "stale": {"lists": sum(x["stale"] for x in rows), "assets": sum(x["stale"] for x in assets_rows)}}

    # -------------------------------------------------------------------------------------------- curves
    def _result(self, key: str) -> sqlite3.Row | None:
        try:
            strategy, list_id, tf, inst = parse_key(key)
        except KeyError:
            return None
        return self.conn().execute("SELECT * FROM result WHERE strategy = ? AND list_id = ? AND instrument_key = ? "
                                   "AND timeframe = ?", (strategy, list_id, inst or "", tf)).fetchone()

    def _benchmark(self, r: sqlite3.Row) -> tuple[sqlite3.Row, pd.Series] | None:
        if r["benchmark_id"] is None or r["bh_cash"]:
            return None
        b = self.conn().execute("SELECT * FROM benchmark WHERE id = ?", (r["benchmark_id"],)).fetchone()
        return b, db.decode_series(b["first_day"], b["days"], b["series"])

    def _index(self, r: sqlite3.Row, oos: pd.Series | None) -> tuple[market_index.Index, pd.Series] | None:
        """The market's index over the record's days (`market_index`), when the result's market has one."""
        index = market_index.of(r["list_id"], r["instrument_id"])
        if index is None or oos is None or oos.empty:
            return None
        s = market_index.daily(index, oos.index, r["fill"])
        return None if s is None else (index, s)

    def curves(self, keys: list[str]) -> dict:
        """Each key's cumulative P&L (weekly growth of 1) and, when it has them, the buy & hold it was compared with and
        its market's index over the same days."""
        out = {}
        for key in dict.fromkeys(keys):
            r = self._result(key)
            if r is None:
                continue
            oos = db.series(self.conn(), r["id"])
            held, index = self._benchmark(r), self._index(r, oos)
            out[key] = {"pts": weekly(oos)[0], "benchmark": None if held is None else weekly(held[1])[0],
                        "index": None if index is None else weekly(index[1])[0],
                        "index_name": None if index is None else index[0].name}
        return {"curves": out}

    # -------------------------------------------------------------------------------------------- a result's popup
    def result(self, key: str) -> dict | None:
        r = self._result(key)
        if r is None:
            return None
        c = self.conn()
        rid, inst = r["id"], r["instrument_id"]
        kind = "list" if inst is None else "pick" if r["strategy"] == PICK else "asset"
        row = self.list_row(rid) if kind == "list" else self.asset_row(key)
        oos = db.series(c, rid)
        pts, lows = weekly(oos)
        held, index = self._benchmark(r), self._index(r, oos)
        figs = db.figures(c, rid) or {}
        wins = db.windows(c, rid)
        out = {
            "key": key, "kind": kind, "row": row, "description": _description(c, r["strategy"]),
            "notes": json.loads(r["notes"]), "fill": r["fill"], "evaluated_at": r["evaluated_at"],
            "figures": {k: (figs.get(k) if k in ("start", "end") else fnum(figs.get(k))) for k in db.FIGURES}
            | at_target_dd(whole(figs.get("n_trades")), {k: figs.get(k) for k in db.AT_TARGET_DD}),
            "curve": {"pts": pts, "dd": lows, "months": months_bp(oos), "growth": growth(oos)},
            "benchmark": None if held is None else {
                "pts": weekly(held[1])[0], "months": months_bp(held[1]), "growth": growth(held[1]),
                "figures": {k: (held[0][k] if k in ("start", "end") else fnum(held[0][k])) for k in db.BENCHMARK_FIGURES}},
            "index": None if index is None else {
                "name": index[0].name, "pts": weekly(index[1])[0], "months": months_bp(index[1]),
                "growth": growth(index[1]),
                "figures": {k: (v if k in ("start", "end") else fnum(v))
                            for k, v in market_index.figures(index[1]).items()}},
            "windows": windows_view(wins), "params": None if wins else _loads(r["params_in_sample"]),
            "grid": _loads(r["grid"]), "params_now": _loads(r["params_now"]),
            "names": self._names(rid, r) if kind == "list" else [],
            "elsewhere": self._elsewhere(r["strategy"], inst), "candidates": [], "beyond_luck": None}
        if kind == "pick":
            out["candidates"] = self._candidates(rid)
            n = self.picks_tries()
            if n is not None and oos is not None:
                out["beyond_luck"] = fnum(significance.deflated(oos, n.independent)["prob"])
        return out

    def _names(self, rid: int, r: sqlite3.Row) -> list[dict]:
        rows, _ = self.asset_rows()
        alone = {(x["instrument_id"], x["timeframe"]): x for x in rows if x["strategy"] == r["strategy"]}
        out = []
        for n in self.conn().execute("SELECT * FROM result_name WHERE result_id = ? ORDER BY instrument_id", (rid,)):
            a = alone.get((n["instrument_id"], r["timeframe"]))
            out.append({"instrument_id": n["instrument_id"], "label": asset_label(n["instrument_id"]),
                        "trades": n["trades"], "won": fnum(n["won"]), "compounded": fnum(n["compounded"]),
                        "alone": None if a is None else {"key": a["key"], "sharpe": a["sharpe"], "vs_bh": a["vs_bh"],
                                                         "beats_bh": a["beats_bh"], "stale": a["stale"]}})
        return out

    def _elsewhere(self, strategy: str, instrument: str | None) -> dict:
        """The same strategy on every list and timeframe, and on this instrument alone on each timeframe."""
        cells = [_cell(x) for x in self.list_rows() if x["strategy"] == strategy]
        alone = []
        if instrument is not None:
            rows, _ = self.asset_rows()
            alone = [_cell(x) for x in rows if x["strategy"] == strategy and x["instrument_id"] == instrument]
        return {"lists": cells, "alone": alone}

    def _candidates(self, rid: int) -> list[dict]:
        rows, _ = self.asset_rows()
        by_id = {x["id"]: x for x in rows}
        out = []
        for p in self.conn().execute("SELECT r.id, r.strategy, r.list_id, r.timeframe, r.instrument_id, p.windows, "
                                     "f.sharpe FROM pick_candidate p JOIN result r ON r.id = p.candidate_id "
                                     "LEFT JOIN result_figures f ON f.result_id = r.id AND f.scope = 'out_of_sample' "
                                     "WHERE p.result_id = ? ORDER BY p.windows DESC, f.sharpe DESC", (rid,)):
            shown = by_id.get(p["id"])
            out.append({"key": asset_key(p["strategy"], p["list_id"], p["timeframe"], p["instrument_id"]),
                        "strategy": p["strategy"], "sharpe": fnum(p["sharpe"]), "windows": p["windows"],
                        "stale": bool(shown and shown["stale"])})
        return out


def _description(c: sqlite3.Connection, name: str) -> str:
    r = c.execute("SELECT description FROM strategy WHERE name = ?", (name,)).fetchone()
    return "" if r is None else r[0]
