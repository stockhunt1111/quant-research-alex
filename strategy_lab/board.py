"""The research summary: one row per evaluated (strategy, list, timeframe), ranked against the target.

Reads every list result in the app's database (strategy_lab.db) on the research lists (strategy_lab.lists); any other
result (an ad-hoc run, a list no longer researched) is counted apart (`not_ranked`), not ranked.
Rows that make money out-of-sample come first — positive Sharpe AND positive compounded return, since a positive
Sharpe can still lose money once big swings compound (and a losing strategy meeting the drawdown target
by barely moving is not progress) — then by how many of the five targets the OOS record meets, then by OOS Sharpe.
A row's `why_not` names the targets it misses in plain words; `stale` marks a result the current code did not
produce (its source changed since, or it predates the stamp): it must be re-run before it is used. `noise_bar` is the
Sharpe the best of as many worthless tries as the saved evaluations add up to (`tries`: a rule on nested lists or on
neighbouring timeframes, or with its model, wins and loses with itself and is not counted as another chance) would
reach on a record this long, and `deflated_prob` the probability that the record is beyond it
(`significance.deflated`); `random_timing_p` is the share
of time-shifted copies of its positions that do as well (`significance.random_timing`). `robustness` is how many of
the robustness checks that apply to the row it passes (`robustness()`: what `strategy_lab.robustness` measured in the
evaluation, judged here against the thresholds below; empty for a row that loses money, which is not checked), and
`robustness_checks` each check's value, threshold and state.

The tries are worked out over every list result at once (tens of seconds) and kept in the database with the results
they cover (`refresh_tries`): a reader takes the kept count while it still covers the results, and a result saved
since is judged against the last count until it is worked out again. A strategy run on an instrument alone is judged
the same way (`robustness(alone=True)`): against the tries on its instrument (`refresh_asset_tries`, kept with the
tries on every instrument at once, which are shown beside), and on the other instruments of its market (`peers`)
instead of a list's names and neighbouring lists.
"""
from __future__ import annotations

import fcntl
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy_lab import db, lists, log, provenance, significance
from strategy_lab.config import SEED
from strategy_lab.walkforward import MIN_PEERS, PEER_SHARE

LOG = log.get("board")
COLUMNS = {
    "strategy": "strategy", "market": "market", "universe": "universe", "timeframe": "tf",
    "oos_start": "OOS from", "oos_end": "OOS to",
    "avg_monthly": "avg/month", "cagr": "compounded/yr", "pct_green_active": "green (of months in market)",
    "pct_months_active": "months in market", "max_dd": "max DD",
    "sharpe": "Sharpe",
    "n_trades": "trades", "beats_bh": "beats B&H", "vs_bh": "vs B&H/yr at equal risk", "bh_sharpe": "B&H Sharpe",
    "bh_max_dd": "B&H max DD",
    "targets_met": "targets met", "robustness": "robustness (passed/applicable)",
    "is_sharpe": "IS Sharpe", "mc_sharpe_p5": "MC Sharpe P5",
    "noise_bar": "Sharpe of the best noise try", "deflated_prob": "P(beyond luck)",
    "random_timing_p": "random timing as good (p)",
    "why_not": "misses", "stale": "stale",
}
EXTRA = ["random_timing_null", "robustness_passed", "robustness_applicable", "robustness_checks", "caveats", "bh_cash"]
PLAIN = {"avg_monthly>=1.5%": "return < 1.5%/month", "green_months>=70%": "green months < 70%",
         "max_dd>=-10%": "drawdown deeper than 10%", "sharpe>=1": "Sharpe < 1",
         "beats_buy_and_hold": "does not beat buy & hold"}


def ranked(strategy: str, universe: str) -> bool:
    """A research result: on one of our lists."""
    return lists.market(universe) is not None


def ranked_only(frame: pd.DataFrame) -> pd.DataFrame:
    """The research results of `db.results_frame` (a mask indexed as the frame: a plain empty list would pick no
    columns rather than no rows)."""
    return frame[pd.Series([ranked(s, u) for s, u in zip(frame["strategy"], frame["list_id"])], index=frame.index,
                           dtype=bool)]


@contextmanager
def _db(conn=None) -> Iterator:
    """The caller's connection, or one opened (and closed) here."""
    if conn is not None:
        yield conn
        return
    own = db.connect()
    try:
        yield own
    finally:
        own.close()


def not_ranked(conn=None) -> dict[str, int]:
    """Saved list results that are not research results: list -> how many."""
    with _db(conn) as c:
        rows = c.execute("SELECT list_id, count(*) FROM result WHERE instrument_id IS NULL GROUP BY list_id").fetchall()
    return {u: n for u, n in sorted((r[0], r[1]) for r in rows) if not ranked("", u)}


TRY_MIN_DAYS = 60           # two records that share fewer days are taken as independent
TRY_DRAWS = 20_000          # Monte Carlo draws of the best of the tries when none of them has skill
TRY_BATCH = 2_000


@dataclass(frozen=True)
class Tries:
    """The evaluations the ranked results were picked from (`luck_of`): how many are saved; how many independent tries
    they add up to, the count the deflated Sharpe takes; and the score the best of them reaches by luck one time in
    twenty, a familywise test at 5%."""
    saved: int
    independent: float
    best_95: float

    def __str__(self) -> str:
        return f"{self.independent:.0f} independent of {self.saved} tries"


TRY_CODE = ["strategy_lab/board.py", "strategy_lab/significance.py"]     # the code the count comes from


def _tries_row(row) -> Tries:
    return Tries(int(row["saved"]), float(row["independent"]), float(row["best_95"]))


def current_tries(conn) -> Tries | None:
    """The kept count, when it still covers the list results and was worked out by the code on disk; else None."""
    row = db.latest_tries(conn, "lists")
    if row is None or row["covers"] != db.results_fingerprint(conn, "lists") \
            or row["code_sha"] != provenance.sha_of(TRY_CODE):
        return None
    return _tries_row(row)


def last_tries(conn) -> Tries | None:
    """The last count kept, whether or not it still covers the list results (a result saved since is judged by it
    until the count is worked out again)."""
    row = db.latest_tries(conn, "lists")
    return None if row is None else _tries_row(row)


def tries(conn=None) -> Tries:
    """Every saved list evaluation is a try the ranked results were picked from, on a research list or not, whatever
    code produced it: the kept count while it covers them, else worked out now (`refresh_tries`)."""
    with _db(conn) as c:
        return current_tries(c) or refresh_tries(c)


@contextmanager
def _one_count_at_a_time() -> Iterator[None]:
    """A lock beside the database: one process works the count out, the others wait and take its result."""
    lock = db.DB_PATH.with_name(db.DB_PATH.name + ".tries.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def list_records(conn) -> tuple[pd.DataFrame, int, str]:
    """Every list result's out-of-sample record (a column each), read in one transaction with the count of list
    results and the fingerprint they make."""
    conn.execute("BEGIN")
    try:
        cover = db.results_fingerprint(conn, "lists")
        saved = conn.execute("SELECT count(*) FROM result WHERE instrument_id IS NULL").fetchone()[0]
        rows = conn.execute("SELECT r.id, s.first_day, s.days, s.series FROM result r JOIN result_series s ON "
                            "s.result_id = r.id AND s.kind = 'out_of_sample' WHERE r.instrument_id IS NULL").fetchall()
    finally:
        conn.execute("COMMIT")
    daily = {r["id"]: db.decode_series(r["first_day"], r["days"], r["series"]) for r in rows}
    return pd.DataFrame(daily), saved, cover


def refresh_tries(conn=None) -> Tries:
    """Work the count out over every list result now and keep it (one process at a time; a count another process kept
    meanwhile is taken instead)."""
    with _db(conn) as c, _one_count_at_a_time():
        kept = current_tries(c)
        if kept is not None:
            return kept
        daily, saved, cover = list_records(c)
        if saved > daily.shape[1]:
            LOG.warning("%d list results have no out-of-sample record to correlate: counted as independent tries",
                        saved - daily.shape[1])
        n = Tries(saved, *luck_of(daily, extra=saved - daily.shape[1]))
        db.save_tries(c, "lists", cover, n.saved, n.independent, n.best_95, provenance.sha_of(TRY_CODE))
        LOG.info("luck is counted over %s", n)
        return n


@dataclass(frozen=True)
class AssetTries:
    """The tries of the strategies run on instruments alone: every (strategy, instrument, timeframe) pair ever scored,
    once however many lists scored it, whatever code produced it. A result is judged against the tries on its own
    instrument (`each`: its timeframes, a rule and its variants on it, correlated as their records are), the choice
    it answers: which strategy for this instrument, as the firm's products and the ML task pick one for a symbol; the
    tries on every instrument at once (`every`: pairs of different instruments taken as independent, though they
    share their market) are how far luck takes the best row of the whole table, shown beside."""
    every: Tries
    each: dict[str, Tries]


def _asset_tries_of(conn, row) -> AssetTries:
    return AssetTries(_tries_row(row), {r["instrument_id"]: _tries_row(r) for r in db.instrument_tries(conn, row["id"])})


def current_asset_tries(conn) -> AssetTries | None:
    """The kept count, when it still covers the single-asset results and was worked out by the code on disk."""
    row = db.latest_tries(conn, "assets")
    if row is None or row["covers"] != db.results_fingerprint(conn, "assets") \
            or row["code_sha"] != provenance.sha_of(TRY_CODE):
        return None
    return _asset_tries_of(conn, row)


def last_asset_tries(conn) -> tuple[int, AssetTries] | None:
    """The last count kept and its id, whether or not it still covers the single-asset results (a result saved since
    is judged by it until the count is worked out again)."""
    row = db.latest_tries(conn, "assets")
    return None if row is None else (row["id"], _asset_tries_of(conn, row))


def asset_records(conn) -> tuple[dict[str, pd.DataFrame], str]:
    """Every pair's out-of-sample record, a block of columns an instrument (a pair without a record: a column with no
    day, an independent try), read in one transaction with the fingerprint they make."""
    conn.execute("BEGIN")
    try:
        cover = db.results_fingerprint(conn, "assets")
        rows = conn.execute("SELECT r.strategy, r.instrument_id, r.timeframe, s.first_day, s.days, s.series FROM result r "
                            "LEFT JOIN result_series s ON s.result_id = r.id AND s.kind = 'out_of_sample' WHERE "
                            f"{db.FAMILIES['assets']} ORDER BY r.id").fetchall()
    finally:
        conn.execute("COMMIT")
    pairs: dict[str, dict] = {}
    for r in rows:
        mine = pairs.setdefault(r["instrument_id"], {})
        key = (r["strategy"], r["timeframe"])
        if key not in mine:                     # scored from two lists, a pair is the same bet twice
            mine[key] = (pd.Series(dtype=float) if r["series"] is None
                         else db.decode_series(r["first_day"], r["days"], r["series"]))
    return {i: pd.DataFrame(p) for i, p in pairs.items()}, cover


def refresh_asset_tries(conn=None) -> AssetTries | None:
    """Work the single assets' count out now, on every instrument at once and on each alone, and keep it (one process
    at a time; a count another process kept meanwhile is taken instead); None without a single-asset result."""
    with _db(conn) as c, _one_count_at_a_time():
        kept = current_asset_tries(c)
        if kept is not None:
            return kept
        blocks, cover = asset_records(c)
        if not blocks:
            return None
        names = sorted(blocks)
        (independent, best_95), each = luck_of_each([blocks[i] for i in names])
        n = AssetTries(Tries(sum(b.shape[1] for b in blocks.values()), independent, best_95),
                       {i: Tries(blocks[i].shape[1], *e) for i, e in zip(names, each)})
        db.save_tries(c, "assets", cover, n.every.saved, n.every.independent, n.every.best_95,
                      provenance.sha_of(TRY_CODE),
                      instruments={i: (t.saved, t.independent, t.best_95) for i, t in n.each.items()})
        LOG.info("single assets' luck is counted over %s on %d instruments (a median of %.0f independent tries each)",
                 n.every, len(n.each), float(np.median([t.independent for t in n.each.values()])))
        return n


def asset_tries(conn=None) -> AssetTries | None:
    """The kept single assets' count while it covers them, else worked out now (`refresh_asset_tries`)."""
    with _db(conn) as c:
        return current_asset_tries(c) or refresh_asset_tries(c)


def luck_of(returns: pd.DataFrame | list[pd.DataFrame], extra: int = 0, seed: int = SEED) -> tuple[float, float]:
    """How far luck takes the best of the records in `returns` (one column of daily returns each, NaN outside its
    span; or blocks of such records, a record of one block taken as independent of those of another) and of `extra`
    records not known, taken as independent of everything. Were none of them skilled, their scores
    (Sharpe ratios, or t statistics against holding) would be standard normals correlated as the records are, and the
    best of them would reach the maximum of such draws (Monte Carlo). Returns the number of independent draws with the
    same expected maximum (`significance.expected_max`, at most the number of records), the count the deflated Sharpe
    takes, and the maximum's 95th percentile: a familywise test at 5%, maxT (Westfall and Young; Romano and Wolf).
    Two records' scores correlate as their returns over the days both cover, times the share of each record those
    days are: two records that overlap for a tenth of their length share a tenth of their evidence. The same rule on
    nested lists, on neighbouring timeframes or with its model wins and loses with itself: counted as independent
    tries, it would raise the bar a result is judged against as if luck had that many more chances. The correlation is
    the records' own, not their excess over a shared buy-and-hold, which moves tries on one list closer still: the
    bar errs on the strict side."""
    roots = []
    for block in [returns] if isinstance(returns, pd.DataFrame) else returns:
        x = block.to_numpy(dtype=np.float64)
        if x.shape[1] < 2:                      # a lone record has nothing to correlate with: an independent draw
            extra += x.shape[1]
        else:
            roots.append(_score_root(x))
    n = extra + sum(len(r) for r in roots)
    if n < 2:
        return float(n), 1.6448536269514722     # one try: the one-sided 5% point of a single score
    rng = np.random.default_rng(seed)
    best = []
    for _ in range(TRY_DRAWS // TRY_BATCH):
        top = rng.standard_normal((TRY_BATCH, extra)).max(axis=1) if extra else np.full(TRY_BATCH, -np.inf)
        for root in roots:
            top = np.maximum(top, (rng.standard_normal((TRY_BATCH, len(root))) @ root.T).max(axis=1))
        best.append(top)
    best = np.concatenate(best)
    return _as_independent(float(best.mean()), n), float(np.quantile(best, 0.95))


def luck_of_each(blocks: list[pd.DataFrame], seed: int = SEED) -> tuple[tuple[float, float], list[tuple[float, float]]]:
    """`luck_of` over all the blocks at once and over each block alone, from the same draws: in each draw the best of a
    block's scores is its own family's best, and the best of those the whole family's. Each block's correlation is
    worked out once for both (it is the count's whole cost). A block of one record is one try."""
    roots = [_score_root(b.to_numpy(dtype=np.float64)) for b in blocks]
    rng = np.random.default_rng(seed)
    best: list[list[np.ndarray]] = [[] for _ in roots]
    for _ in range(TRY_DRAWS // TRY_BATCH):
        for k, root in enumerate(roots):
            best[k].append((rng.standard_normal((TRY_BATCH, len(root))) @ root.T).max(axis=1))
    each = [np.concatenate(b) for b in best]
    overall = np.max(np.vstack(each), axis=0)
    one = (1.0, 1.6448536269514722)             # one try: the one-sided 5% point of a single score
    return ((_as_independent(float(overall.mean()), sum(len(r) for r in roots)), float(np.quantile(overall, 0.95))),
            [one if len(r) < 2 else (_as_independent(float(e.mean()), len(r)), float(np.quantile(e, 0.95)))
             for e, r in zip(each, roots)])


def _score_root(x: np.ndarray) -> np.ndarray:
    """A square root of the correlation of the records' scores (`luck_of`), each row of unit length: standard normals
    times its transpose are draws with that correlation."""
    have = ~np.isnan(x)
    z, h = np.where(have, x, 0.0), have.astype(np.float64)
    shared = h.T @ h                                        # days each pair covers together
    sums, squares, products = z.T @ h, (z * z).T @ h, z.T @ z     # a's sum and sum of squares over the days b has too
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = products - sums * sums.T / shared
        var = squares - sums * sums / shared
        rho = cov / np.sqrt(var * var.T)
        corr = rho * shared / np.sqrt(np.outer(np.diag(shared), np.diag(shared)))
    corr = np.where(np.isfinite(corr) & (shared >= TRY_MIN_DAYS), corr, 0.0)
    np.fill_diagonal(corr, 1.0)
    # pairs measured over different days need not make a valid correlation matrix: the nearest one with no negative
    # eigenvalue, each record's variance kept at one
    w, v = np.linalg.eigh(corr)
    root = v * np.sqrt(np.clip(w, 0.0, None))
    return root / np.sqrt((root ** 2).sum(axis=1, keepdims=True))


def _as_independent(expected_best: float, k: int) -> float:
    """The number of independent standard normal draws whose expected maximum is `expected_best` (1 to `k`); below two
    draws the count runs straight from one draw (expected maximum 0) to two."""
    two = significance.expected_max(2)
    if expected_best <= two:
        return 1.0 + max(expected_best, 0.0) / two
    lo, hi = 2.0, float(k)
    if significance.expected_max(hi) <= expected_best:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if significance.expected_max(mid) < expected_best else (lo, mid)
    return (lo + hi) / 2


# The pass rule of each robustness check (strategy_lab.robustness measures them in the evaluation; they are judged here,
# so a threshold can change without a re-run): the deflated Sharpe and random timing's bars are significance's
KEEP = 0.5              # a neighbouring configuration, list or model seed keeps at least half of the Sharpe
PBO_MAX = 0.5
ERAS_MIN = 0.60         # the Minerva tester's bar of consistency across eras
NAMES_MIN = 0.5
RULE_T_MIN = 2.0        # a model beats the rule it grades by two standard errors of the difference
CHECKS = {"luck": "Sharpe beyond luck", "vs_hold": "Beats buy & hold beyond luck", "timing": "Timing beats random",
          "pbo": "Not overfitted (PBO)", "plateau": "Parameter plateau", "eras": "Stable across eras",
          "delay": "A bar later", "costs": "Costs ×3", "names": "Most names make money",
          "peers": "Works on most of its market", "neighbour_lists": "Neighbouring lists",
          "seeds": "Model: other seeds", "vs_rule": "Model beats the plain rule"}
# an instrument alone has one name and no list around it (`robustness.LIST_ONLY`): it is judged instead on the other
# instruments of its market, as a rule's choice on peers is (`walkforward.MIN_PEERS`, `PEER_SHARE`)
LIST_ONLY = ("names", "neighbour_lists")
ALONE_ONLY = ("peers",)
STATES = ("passed", "failed", "too_short", "not_applicable", "not_computed")
TOP_N = re.compile(r"_top(\d+)$")


def checks_of(alone: bool) -> list[str]:
    """The checks of a list's result, or of a strategy's on an instrument alone."""
    return [c for c in CHECKS if c not in (LIST_ONLY if alone else ALONE_ONLY)]


def peers(results: pd.DataFrame) -> dict[int, dict]:
    """The `peers` check of each single-asset result (`results`: id, strategy, list_id, timeframe, sharpe, cagr of a
    run's results): of the other instruments of the same run (the same strategy and timeframe on the rest of its list,
    its market) scored, how many make money; fewer than MIN_PEERS scored: too few to judge."""
    out: dict[int, dict] = {}
    earning = (pd.to_numeric(results["sharpe"], errors="coerce") > 0) & (pd.to_numeric(results["cagr"], errors="coerce") > 0)
    for _, g in results.assign(earning=earning).groupby(["strategy", "list_id", "timeframe"], sort=False):
        n, k = len(g), int(g["earning"].sum())
        for rid, e in zip(g["id"], g["earning"]):
            others, positive = n - 1, k - int(e)
            out[int(rid)] = ({"too_short": f"{others} other instrument{'' if others == 1 else 's'} of its market scored, "
                                           f"{MIN_PEERS} needed"} if others < MIN_PEERS
                             else {"others": others, "positive": positive, "share": positive / others})
    return out


@dataclass(frozen=True)
class Check:
    id: str
    label: str
    value: str
    threshold: str
    state: str                  # one of STATES


@dataclass(frozen=True)
class Robustness:
    checks: list[Check]

    @property
    def passed(self) -> int:
        return sum(c.state == "passed" for c in self.checks)

    @property
    def applicable(self) -> int:
        """Checks that apply and were measured: passed, failed or on a record too short to judge (not passed)."""
        return sum(c.state in ("passed", "failed", "too_short") for c in self.checks)

    @property
    def not_computed(self) -> int:
        return sum(c.state == "not_computed" for c in self.checks)

    def __str__(self) -> str:
        return f"{self.passed}/{self.applicable}"


def _s(v) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.2f}"


def _yr(v) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v * 100:+.1f}%/yr"


def _earns(sharpe, cagr) -> bool:
    """Makes money: a positive Sharpe and a positive compounded return (a positive Sharpe can still lose money once
    large swings compound)."""
    return sharpe is not None and cagr is not None and sharpe > 0 and cagr > 0


def _sc(sharpe, cagr) -> str:
    """A Sharpe, with the compounded return beside it when that loses money."""
    return _s(sharpe) + (f" ({_yr(cagr)})" if cagr is not None and cagr <= 0 else "")


def _list_name(u: str) -> str:
    m = TOP_N.search(u)
    return f"Top-{m[1]}" if m else u


def _judged(cid: str, m: dict, sharpe: float, n_tries: Tries, deflated: dict,
            every: tuple[Tries, dict] | None = None) -> tuple[str, str, bool | None]:
    """(value, threshold, passed); passed None: the record is too short to judge. `every`: an instrument alone's tries
    on every instrument and its deflated Sharpe against them, shown beside its own."""
    half = f"≥ ½ × {_s(sharpe)}"
    if cid == "luck":
        prob = deflated["prob"]
        if not np.isfinite(prob):
            return f"MC P5 {_s(m['mc_sharpe_p5'])}", "", None
        own = "" if every is None else "this asset's "
        value = f"MC P5 {_s(m['mc_sharpe_p5'])} · {prob:.0%} beyond the best of {own}{n_tries}"
        if every is not None and np.isfinite(every[1]["prob"]):
            value += f"; {every[1]['prob']:.0%} beyond the best of every asset's {every[0]}"
        return (value, f"P5 > 0 and ≥ {significance.PASS_PROB:.0%}",
                m["mc_sharpe_p5"] is not None and m["mc_sharpe_p5"] > 0 and prob >= significance.PASS_PROB)
    if cid == "vs_hold":
        bar = n_tries.best_95
        t = m["t"]
        return (f"t {_s(t)} (Sharpe {_s(m['sharpe'])} vs {_s(m['sharpe_other'])} held)", f"t ≥ {bar:.2f}",
                t is not None and t >= bar)
    if cid == "timing":
        return f"p {_s(m['p']) if m['p'] is None else format(m['p'], '.3f')}", f"p < {significance.PASS_P}", \
            m["p"] is not None and m["p"] < significance.PASS_P
    if cid == "pbo":
        return f"{m['pbo']:.2f} ({m['configs']} combinations)", f"≤ {PBO_MAX:.2f}", m["pbo"] <= PBO_MAX
    if cid == "plateau":
        near = m["neighbours"]
        return (" / ".join(_sc(x["sharpe"], x["cagr"]) for x in near) + f" vs {_s(m['sharpe'])}",
                f"each makes money, median ≥ ½ × {_s(m['sharpe'])}",
                all(_earns(x["sharpe"], x["cagr"]) for x in near)
                and float(np.median([x["sharpe"] for x in near])) >= KEEP * m["sharpe"])
    if cid == "eras":
        return f"ρ {m['rho']:.2f} ({len(m['sharpes'])} windows)", f"ρ ≥ {ERAS_MIN:.2f}", m["rho"] >= ERAS_MIN
    if cid in ("delay", "costs"):
        at = f"at ×{m['multiple']:g}: " if cid == "costs" else ""
        return (f"{at}Sharpe {_s(m['sharpe'])}, {_yr(m['cagr'])}", "Sharpe and return > 0",
                m["sharpe"] is not None and m["cagr"] is not None and m["sharpe"] > 0 and m["cagr"] > 0)
    if cid == "names":
        return f"{m['share']:.0%} of {m['instruments']}", f"≥ {NAMES_MIN:.0%}", m["share"] >= NAMES_MIN
    if cid == "peers":
        return (f"{m['positive']} of {m['others']} others make money ({m['share']:.0%})", f"≥ {PEER_SHARE:.0%}",
                m["share"] >= PEER_SHARE)
    if cid == "neighbour_lists":
        got = m["lists"]
        return (" · ".join(f"{_list_name(x['universe'])} {_sc(x['sharpe'], x['cagr'])}" for x in got)
                + f" vs {_s(sharpe)}", f"each makes money, {half}",
                all(_earns(x["sharpe"], x["cagr"]) and x["sharpe"] >= KEEP * sharpe for x in got))
    if cid == "seeds":
        pairs = list(zip(m["sharpes"], m["cagrs"], strict=True))
        return (" / ".join(_sc(v, c) for v, c in pairs) + f" vs {_s(sharpe)}", f"each makes money, {half}",
                all(_earns(v, c) and v >= KEEP * sharpe for v, c in pairs))
    if cid == "vs_rule":
        t = m["t"]
        return (f"t {_s(t)} (Sharpe {_s(m['sharpe'])} vs {_s(m['sharpe_plain'])} without the model)",
                f"t ≥ {RULE_T_MIN:g}", t is not None and t >= RULE_T_MIN)
    raise ValueError(f"no pass rule for the robustness check {cid!r}")


def robustness(card: dict, daily: pd.Series, n_tries: Tries, deflated: dict | None = None, *, alone: bool = False,
               every: tuple[Tries, dict] | None = None) -> Robustness | None:
    """A result's robustness checks, judged: None for a result that loses money out-of-sample (nothing is checked); a
    result saved by code older than the checks has every check not computed. `alone`: a strategy on an instrument
    alone, its checks `checks_of(alone=True)` and `n_tries` the tries on its instrument; `every`: see `_judged`."""
    if "robustness" not in card:
        LOG.info("%s on %s %s was saved before its robustness was measured: re-run it", card.get("strategy"),
                 card.get("universe"), card.get("timeframe"))
        measured = {}
    else:
        measured = card["robustness"]
        if measured is None:
            return None
    deflated = deflated if deflated is not None else significance.deflated(daily, n_tries.independent)
    sharpe = card["out_of_sample"]["sharpe"]
    out = []
    for cid in checks_of(alone):
        label, m = CHECKS[cid], measured.get(cid)
        if m is None:
            out.append(Check(cid, label, "not measured in this evaluation", "", "not_computed"))
        elif "na" in m:
            out.append(Check(cid, label, m["na"], "", "not_applicable"))
        elif "too_short" in m:
            out.append(Check(cid, label, m["too_short"], "", "too_short"))
        else:
            try:
                value, threshold, ok = _judged(cid, m, sharpe, n_tries, deflated, every)
            except KeyError as e:            # its pass rule reads a figure the code that measured it did not keep
                LOG.warning("%s on %s %s: %s was measured without %s: re-run it", card.get("strategy"),
                            card.get("universe"), card.get("timeframe"), cid, e.args[0])
                out.append(Check(cid, label, f"measured without {e.args[0]}: re-run it", "", "not_computed"))
                continue
            out.append(Check(cid, label, value, threshold,
                             "too_short" if ok is None else "passed" if ok else "failed"))
    return Robustness(out)


def card_of(r, measures: dict | None) -> dict:
    """What `robustness` reads of a result: its key, its out-of-sample Sharpe and what the evaluation measured (None:
    it loses money, nothing measured)."""
    return {"strategy": r.strategy, "universe": r.list_id, "timeframe": r.timeframe,
            "out_of_sample": {"sharpe": r.sharpe}, "robustness": measures}


def row_of(r, measured: dict | None, daily: pd.Series, n_tries: Tries, current: bool) -> dict:
    """One list result's board row (`r`: a row of `db.results_frame`), judged against the tries `n_tries`."""
    luck = significance.deflated(daily, n_tries.independent)
    rob = robustness(card_of(r, measured), daily, n_tries, luck)
    return {
        "strategy": r.strategy, "market": lists.market(r.list_id), "universe": r.list_id,
        "timeframe": r.timeframe, "oos_start": r.start, "oos_end": r.end,
        "avg_monthly": r.avg_monthly, "cagr": r.cagr, "pct_green_active": r.pct_green_active,
        "pct_months_active": r.pct_months_active, "max_dd": r.max_dd, "sharpe": r.sharpe,
        "n_trades": r.n_trades, "beats_bh": bool(r.beats_bh), "vs_bh": _nan(r.vs_bh),
        "bh_sharpe": _nan(r.bh_sharpe), "bh_max_dd": _nan(r.bh_max_dd), "targets_met": r.targets_met,
        "robustness": "" if rob is None else str(rob),
        "is_sharpe": _nan(r.is_sharpe), "mc_sharpe_p5": _nan(r.mc_sharpe_p5),
        "noise_bar": luck["noise_bar"], "deflated_prob": luck["prob"],
        "random_timing_p": _nan(r.rt_p), "random_timing_null": _nan(r.rt_null_median),
        "why_not": "; ".join(PLAIN[k] for k, ok in r.targets.items() if not ok) or "—",
        "stale": not current,
        "robustness_passed": np.nan if rob is None else rob.passed,
        "robustness_applicable": np.nan if rob is None else rob.applicable,
        "robustness_checks": "" if rob is None else json.dumps([[x.id, x.label, x.value, x.threshold, x.state]
                                                                 for x in rob.checks]),
        "caveats": "; ".join(n for n in r.notes if not n.startswith("excluded")),
        # a result scored before cash comparisons held its list, currency pairs and crude included
        "bh_cash": bool(r.bh_cash),
    }


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that make money first, then by targets met, then by Sharpe."""
    df = df.assign(_earns=(df["sharpe"] > 0) & (df["cagr"] > 0))
    return df.sort_values(["_earns", "targets_met", "sharpe"], ascending=False).drop(columns="_earns").reset_index(
        drop=True)


def collect(conn=None, n_tries: Tries | None = None) -> pd.DataFrame:
    rows = []
    with _db(conn) as c:
        n_tries = n_tries or tries(c)
        frame = db.results_frame(c, "list")
        frame = ranked_only(frame)
        measured = db.measures_of(c, frame["id"])
        current: dict[str, bool] = {}
        for r in frame.itertuples(index=False):
            if r.code_sha not in current:
                current[r.code_sha] = provenance.is_current({"files": r.code_files, "sha": r.code_sha})
            rows.append(row_of(r, measured.get(r.id), db.series(c, r.id), n_tries, current[r.code_sha]))
    return ordered(pd.DataFrame(rows, columns=[*COLUMNS, *EXTRA]))


def _nan(v) -> float:
    return np.nan if v is None else v


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=EXTRA, errors="ignore").copy()
    pct = lambda v: "" if pd.isna(v) else f"{v * 100:+.2f}%"          # noqa: E731
    out["avg_monthly"] = out["avg_monthly"].map(pct)
    out["max_dd"] = out["max_dd"].map(pct)
    out["cagr"] = out["cagr"].map(pct)
    out["bh_max_dd"] = out["bh_max_dd"].map(pct)
    out["vs_bh"] = out["vs_bh"].map(pct)
    for c in ("pct_green_active", "pct_months_active"):
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v * 100:.0f}%")
    if "stale" in out:
        out["stale"] = out["stale"].map(lambda v: "STALE" if v else "")
    out["deflated_prob"] = out["deflated_prob"].map(lambda v: "" if pd.isna(v) else f"{v * 100:.0f}%")
    out["random_timing_p"] = out["random_timing_p"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    for c in ("sharpe", "bh_sharpe", "is_sharpe", "mc_sharpe_p5", "noise_bar"):
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    return out.rename(columns=COLUMNS)
