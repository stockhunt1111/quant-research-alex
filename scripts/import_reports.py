"""One-off: the results saved as files under reports/ before the app's database, into it (db/app.sqlite), through the
writer an evaluation uses (strategy_lab.db). Importing again replaces what it imported.

    python scripts/import_reports.py                       # reports/ of this checkout
    python scripts/import_reports.py --reports path/to/reports --only donchian_breakout --db /tmp/x.sqlite

A list's result: card.json, oos_daily.parquet, comparison_daily.parquet (in_sample, buy_and_hold), trades.csv,
folds.csv; its evaluation time from evaluation_seconds.csv when kept there. A run of a list's instruments alone:
run.json, instruments.csv, oos_daily.parquet (a column an instrument). Such a run never saved its instruments'
buy-and-hold, windows or positions, and today's bars would not give the buy-and-hold its figures came from: those
results keep the figures they had and no series behind them until they are run again.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow import csv as pa_csv

from strategy_lab import db, log, metrics
from strategy_lab.config import REPORTS_DIR
from strategy_lab.strategy import load

LOG = log.get("import_reports")
TOLERANCE = 1e-9             # a card's buy-and-hold figures against the same figures of the series saved beside it


def _mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


_DESCRIPTIONS: dict[str, str] = {}


def description(name: str) -> str:
    """The strategy's rule as its module describes it today (a strategy no longer in strategies/: none)."""
    if name not in _DESCRIPTIONS:
        try:
            _DESCRIPTIONS[name] = load(name).description
        except (ImportError, FileNotFoundError, ValueError, AttributeError) as e:
            LOG.warning("%s: no strategy module today (%s): saved without a description", name, e)
            _DESCRIPTIONS[name] = ""
    return _DESCRIPTIONS[name]


def _trades(path: Path) -> pd.DataFrame:
    t = pa_csv.read_csv(path).to_pandas()
    if t.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in db.TRADE_COLUMNS})
    for c in ("entry_time", "exit_time"):
        t[c] = pd.to_datetime(t[c], utc=True)
    return t


def _checked_benchmark(card: dict, bh: pd.Series, where: Path) -> None:
    """The card's bh_* are metrics.core of the buy-and-hold over the record's days: the series kept beside it must give
    them back, or the page's Buy & hold column would disagree with the table."""
    core = metrics.core(bh)
    off = {k: (card[f"bh_{k}"], core[k]) for k in db.BH
           if card.get(f"bh_{k}") is not None and np.isfinite(card[f"bh_{k}"]) and abs(card[f"bh_{k}"] - core[k]) > TOLERANCE}
    if off:
        LOG.warning("%s: the card's buy & hold figures differ from its saved series: %s", where, off)


def list_result(folder: Path, seconds: dict) -> tuple[db.Outcome, dict, str]:
    card = json.loads((folder / "card.json").read_text())
    o = card["out_of_sample"]
    oos = pd.read_parquet(folder / "oos_daily.parquet")["net"]
    comp = pd.read_parquet(folder / "comparison_daily.parquet") if (folder / "comparison_daily.parquet").exists() else None
    trades = _trades(folder / "trades.csv")
    windows = pd.read_csv(folder / "folds.csv").to_dict(orient="records") if (folder / "folds.csv").exists() else []
    bench = None
    if comp is not None and not o.get("bh_cash"):
        bench = comp["buy_and_hold"]
        _checked_benchmark(o, bench, folder)
    key = (card["strategy"], card["universe"], card["timeframe"])
    outcome = db.Outcome(
        strategy=card["strategy"], list_id=card["universe"], timeframe=card["timeframe"], instrument_id=None, card=o,
        oos=oos, in_sample=db.Figures(card["in_sample"]), oos_deals=db.deals(trades),
        in_sample_daily=None if comp is None else comp["in_sample"], benchmark=bench, windows=windows,
        names=db.by_name(trades), trades=trades, monte_carlo=card.get("monte_carlo") or {},
        random_timing=card.get("random_timing") or {},
        # a card without the key predates the checks: every check not computed; null: it loses money, nothing checked
        robustness=card["robustness"] if "robustness" in card else {}, notes=card.get("notes", []),
        grid=card.get("grid"), params_in_sample=card.get("in_sample_params"),
        params_now=json.loads(windows[-1]["params"]) if windows else card.get("in_sample_params"),
        seconds=seconds.get(key))
    return outcome, card["code"], _mtime(folder / "card.json")


def _grid_of(configs: list[dict]) -> dict:
    """The grid a run's configurations came from: each parameter's values, in their first order."""
    grid: dict[str, list] = {}
    for cfg in configs:
        for k, v in cfg.items():
            if v not in grid.setdefault(k, []):
                grid[k].append(v)
    return grid


def _value(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def asset_run(folder: Path) -> tuple[list[db.Outcome], list[tuple[str, int]], dict, dict, str]:
    meta = json.loads((folder / "run.json").read_text())
    table = pd.read_csv(folder / "instruments.csv")
    wide = pd.read_parquet(folder / "oos_daily.parquet") if (folder / "oos_daily.parquet").exists() else pd.DataFrame()
    grid, outcomes, short = _grid_of(meta.get("configs") or []), [], []
    for r in table.to_dict(orient="records"):
        if r["status"] != "scored":
            short.append((r["instrument"], int(r.get("oos_months") or 0)))
            continue
        card = {"start": r["oos_start"], "end": r["oos_end"], "months": r["oos_months"]}
        # an empty figure stays NaN here, as the scorecard had it (metrics.targets compares it); the writer keeps it NULL
        card |= {k: r.get(k, np.nan) for k in ("avg_monthly", "cagr", "pct_green", "pct_red", "pct_green_active",
                                                "pct_red_active", "pct_months_active", "max_dd", "sharpe", "n_trades",
                                                "time_in_market", "bh_sharpe", "bh_max_dd", "bh_avg_monthly", "vs_bh")}
        card["beats_bh"] = bool(r["beats_bh"]) if _value(r.get("beats_bh")) is not None else False
        card["bh_cash"] = None if _value(r.get("bh_cash")) is None else bool(r["bh_cash"])
        card["targets"] = metrics.targets(card)          # the file kept only their count; the figures give them back
        card["targets_met"] = int(r["targets_met"])
        outcomes.append(db.Outcome(
            strategy=meta["strategy"], list_id=meta["universe"], timeframe=meta["timeframe"],
            instrument_id=r["instrument"], card=card, oos=wide[r["instrument"]].dropna(),
            in_sample=db.Figures({"sharpe": _value(r.get("is_sharpe"))}),
            random_timing={"p": _value(r.get("random_timing_p")), "null_median": _value(r.get("random_timing_null_sharpe"))},
            notes=list(meta.get("notes") or []), grid=grid, params_now=json.loads(r["params_now"]),
            best5_share=_value(r.get("best5_share"))))
    return outcomes, short, meta, meta["code"], _mtime(folder / "run.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=REPORTS_DIR)
    ap.add_argument("--only", nargs="*", help="strategies by name")
    ap.add_argument("--db", type=Path, help=f"the database to import into (default {db.DB_PATH})")
    args = ap.parse_args()
    if args.db:
        db.DB_PATH = args.db
    log.setup("import_reports")
    root: Path = args.reports
    seconds = {}
    if (root / "evaluation_seconds.csv").exists():
        s = pd.read_csv(root / "evaluation_seconds.csv")
        seconds = {(r.strategy, r.universe, r.timeframe): float(r.seconds) for r in s.itertuples(index=False)}
    conn = db.connect()
    cards = [p.parent for p in sorted(root.glob("*/*/card.json")) if p.parent.parent.name != "portfolios"
             and (not args.only or p.parent.parent.name in args.only)]
    runs = [p.parent for p in sorted((root / "per_asset").glob("*/*/run.json"))
            if not args.only or p.parent.parent.name in args.only]
    LOG.info("importing %d list results and %d runs of instruments alone from %s into %s", len(cards), len(runs), root,
             db.DB_PATH)
    for k, folder in enumerate(cards, 1):
        outcome, code, when = list_result(folder, seconds)
        db.save_outcomes(conn, [outcome], description=description(outcome.strategy), code=code, fill=None,
                         evaluated_at=when)
        if k % 100 == 0:
            LOG.info("%d of %d list results", k, len(cards))
    for k, folder in enumerate(runs, 1):
        outcomes, short, meta, code, when = asset_run(folder)
        db.save_outcomes(conn, outcomes, description=description(meta["strategy"]), code=code, fill=None,
                         evaluated_at=when, replace_run=(meta["strategy"], meta["universe"], meta["timeframe"]),
                         too_short=short)
        if k % 50 == 0:
            LOG.info("%d of %d runs of instruments alone", k, len(runs))
    counts = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in (
        "result", "result_figures", "result_series", "result_window", "result_name", "result_trades",
        "result_monte_carlo", "result_check", "benchmark", "asset_too_short")}
    LOG.info("imported: %s", counts)
    conn.close()


if __name__ == "__main__":
    main()
