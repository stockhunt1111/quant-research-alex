"""A per-symbol engine's 70/30 protocol replayed
on the ML matrix of strategies/ml_feature_search.py, next to the walk-forward of the same matrix.

For each instrument and timeframe:
  * every configuration (indicator families x entry threshold x direction, long only or long/short)
    gets one model per family set, fit on the first 70% of the history;
    the configuration with the best Sharpe on the last 30% is the one the protocol picks;
  * "test" is that configuration on the last 30% only: the figure the engine compares candidates on;
  * "shown" is the same configuration over the whole history, the fitted 70% included;
  * "walk-forward" is the out-of-sample record of the same matrix from `scripts/per_asset.py --names
    ml_feature_search` (configuration re-picked every 3 months on the past only, models refit on the past only),
    over the same last 30% and over its whole span;
  * buy-and-hold of the instrument over the same last 30% (`evaluate._buy_and_hold`: cash for a currency pair or
    crude).
Fills, costs and metrics are this engine's: next-open fills, commission + half-spread by asset class.

    python scripts/product_protocol.py --universes us_stocks_mcap10 --timeframes 1d    -> reports/product_protocol/<u>__<tf>.csv
    python scripts/product_protocol.py --render                                        -> reports/product_protocol.md
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.ml_feature_search import FAMILY_SETS, HORIZON, ml_feature_search  # noqa: E402
from strategy_lab import db, lists, log, metrics, ml, provenance  # noqa: E402
from strategy_lab.config import REPORTS_DIR, SEED, TIMEFRAMES  # noqa: E402
from strategy_lab.data.bars import load_panel  # noqa: E402
from strategy_lab.engine import backtest as bt  # noqa: E402
from strategy_lab.engine.hold import margined, nothing_to_hold  # noqa: E402
from strategy_lab.evaluate import _buy_and_hold  # noqa: E402
from strategy_lab.strategy import with_short  # noqa: E402
from strategy_lab.universes import instruments_now  # noqa: E402

LOG = log.get("product_protocol")
OUT = REPORTS_DIR / "product_protocol"
TRAIN_SHARE = 0.7
THRESHOLDS = ml_feature_search.grid["threshold"]
LONG_ONLY = ml_feature_search.grid["long_only"]


def _run(panel, weights: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Daily returns and the exposure held on each bar of holding `weights` of the panel's one instrument."""
    r = bt.run(panel, weights.to_frame(panel.ids[0]), fill="next_open")
    return metrics.daily_returns(r.returns), r.exposure


def _card(daily: pd.Series) -> dict:
    c = metrics.core(daily)
    return {"sharpe": c["sharpe"], "avg_monthly": c["avg_monthly"], "max_dd": c["max_dd"],
            "pct_green": c["pct_green_active"], "months": c["months"]}


def _meets(daily: pd.Series, exposure: pd.Series | None, bh: pd.Series, cash: bool, futures: bool) -> int:
    """How many of the firm's targets a record meets, against buy-and-hold over the same days; without its `exposure`
    (the walk-forward's is not saved) the record counts as fully invested: its idle cash earns no T-bills in the
    money test."""
    card = metrics.scorecard(daily, exposure=exposure, benchmark=bh.reindex(daily.index).fillna(0.0), cash=cash,
                             margined=futures)
    return card["targets_met"]


def replay(instrument: str, timeframe: str, wf_daily: pd.Series | None) -> dict:
    panel = load_panel([instrument], timeframe)
    bars = panel.one(instrument).dropna(subset=["close"])
    split = bars.index[int(len(bars) * TRAIN_SHARE)]
    later = bars["close"].shift(-HORIZON)
    y = (later > bars["close"]).astype(float).where(later.notna())
    fam = ml.features(bars)
    live = panel.started[instrument]
    runs = []
    for families in FAMILY_SETS:
        X = pd.concat([fam[f] for f in families], axis=1)
        valid = X.notna().all(axis=1)
        # the label of a row is known once HORIZON more bars have closed: the fit sees none after the split
        train = valid & y.notna() & (pd.Series(np.arange(len(bars)), index=bars.index) <= bars.index.get_loc(split) - HORIZON)
        m = ml.model(SEED).fit(X[train], y[train].astype(int))
        proba = pd.Series(np.nan, index=bars.index)
        proba[valid] = m.predict_proba(X[valid])[:, 1]
        for thr in THRESHOLDS:
            for long_only in LONG_ONLY:
                pos = with_short((proba > thr).astype(float), (proba < 1.0 - thr).astype(float), long_only)
                w = pos.where(proba.notna()).reindex(panel.index).ffill().fillna(0.0).where(live, 0.0)
                runs.append((families, thr, long_only, *_run(panel, w)))
    day = split.normalize()
    test_sharpes = [metrics.core(d[d.index >= day])["sharpe"] for *_, d, _ in runs]
    best = int(np.argmax(test_sharpes))
    families, thr, long_only, daily, exposure = runs[best]
    first = daily[daily != 0].index.min()
    bh, cash, futures = _buy_and_hold(panel, None, "next_open"), nothing_to_hold(panel), margined(panel)
    bh_test = _card(bh[bh.index >= day])
    row = {"instrument": instrument, "timeframe": timeframe, "history_from": str(bars.index[0].date()),
           "split": str(split.date()), "picked": "+".join(families), "threshold": thr, "long_only": long_only,
           "configs": len(runs), "median_config_test_sharpe": float(np.median(test_sharpes))}
    row.update({f"test_{k}": v for k, v in _card(daily[daily.index >= day]).items()})
    shown = _card(daily[daily.index >= first])
    row.update({f"shown_{k}": v for k, v in shown.items()})
    row["shown_targets_met"] = _meets(daily[daily.index >= first], exposure, bh, cash, futures)
    row.update({f"bh_test_{k}": v for k, v in bh_test.items()})
    if wf_daily is not None and len(wf_daily.dropna()):
        wf = wf_daily.dropna()
        wf_test = _card(wf[wf.index >= day])
        wf_all = _card(wf)
        row.update({f"wf_test_{k}": v for k, v in wf_test.items()})
        row.update({f"wf_{k}": v for k, v in wf_all.items()})
        row["wf_targets_met"] = _meets(wf, None, bh, cash, futures)
        row["wf_from"] = str(wf.index.min().date())
    return row


def _wf_record(universe: str, timeframe: str) -> pd.DataFrame | None:
    """The walk-forward records of ml_feature_search on each instrument of the list alone (the app's database): a column
    an instrument, None when that run is not saved."""
    conn = db.connect()
    try:
        ids = [r["id"] for r in conn.execute("SELECT id FROM result WHERE strategy = ? AND list_id = ? AND timeframe = ? "
                                             "AND instrument_id IS NOT NULL", (ml_feature_search.name, universe, timeframe))]
        names = {r["id"]: r["instrument_id"] for r in conn.execute(
            f"SELECT id, instrument_id FROM result WHERE id IN ({', '.join('?' * len(ids))})", ids)} if ids else {}
        records = {names[i]: db.series(conn, i) for i in ids}
    finally:
        conn.close()
    return pd.DataFrame(records) if records else None


def _pct(v) -> str:
    return "" if pd.isna(v) else f"{v * 100:+.2f}%"


def _num(v) -> str:
    return "" if pd.isna(v) else f"{v:.2f}"


def render(table: pd.DataFrame) -> str:
    lines = ["# A 70/30 protocol next to the walk-forward", "",
             f"Every configuration of the ML matrix (`strategies/ml_feature_search.py`: LightGBM on {len(FAMILY_SETS)} "
             f"combinations of five indicator families x {len(THRESHOLDS)} entry thresholds x long only or long/short = "
             f"{len(FAMILY_SETS) * len(THRESHOLDS) * len(LONG_ONLY)}) is fit once on the first 70% of an instrument's "
             "history; the one with "
             "the best Sharpe on the last 30% is what the protocol picks. **Test** is that pick on the last 30%, "
             "**shown** is the same pick over the whole history with the fitted part included, "
             "**walk-forward** is the same matrix re-picked every 3 months on the past only, with models refit on the past "
             "only, over the same last 30% and over its whole out-of-sample span. Targets met: of avg month >= 1.5%, "
             ">= 70% green months, max drawdown within 10%, Sharpe >= 1, more money than buy-and-hold at equal risk "
             "(cash at T-bills for a currency pair).", ""]
    agg = table.groupby(["universe", "timeframe"]).agg(
        instruments=("instrument", "size"), shown_sharpe=("shown_sharpe", "median"), test_sharpe=("test_sharpe", "median"),
        median_config_test_sharpe=("median_config_test_sharpe", "median"), wf_test_sharpe=("wf_test_sharpe", "median"),
        wf_sharpe=("wf_sharpe", "median"), bh_test_sharpe=("bh_test_sharpe", "median"),
        shown_3plus=("shown_targets_met", lambda s: int((s >= 3).sum())), wf_3plus=("wf_targets_met", lambda s: int((s >= 3).sum())))
    lines += ["## Medians per market", "",
              "| universe | tf | instruments | Sharpe shown | Sharpe of the pick on the 30% | median config on the 30% | "
              "walk-forward on the 30% | walk-forward, whole span | buy & hold on the 30% | 3+ targets as shown | "
              "3+ targets walk-forward |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (u, tf), r in agg.iterrows():
        lines.append(f"| {u} | {tf} | {int(r.instruments)} | {_num(r.shown_sharpe)} | {_num(r.test_sharpe)} | "
                     f"{_num(r.median_config_test_sharpe)} | {_num(r.wf_test_sharpe)} | {_num(r.wf_sharpe)} | "
                     f"{_num(r.bh_test_sharpe)} | {int(r.shown_3plus)} | {int(r.wf_3plus)} |")
    lines += ["", "## Each instrument", "",
              "| instrument | tf | split | pick | Sharpe shown | avg month shown | max DD shown | Sharpe on the 30% | "
              "walk-forward Sharpe on the 30% | walk-forward Sharpe | walk-forward avg month | walk-forward max DD | "
              "buy & hold Sharpe on the 30% |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table.sort_values(["universe", "timeframe", "instrument"]).itertuples(index=False):
        side = "long" if getattr(r, "long_only", True) else "long/short"
        lines.append(f"| {r.instrument} | {r.timeframe} | {r.split} | {r.picked} > {r.threshold} {side} | "
                     f"{_num(r.shown_sharpe)} | "
                     f"{_pct(r.shown_avg_monthly)} | {_pct(r.shown_max_dd)} | {_num(r.test_sharpe)} | "
                     f"{_num(getattr(r, 'wf_test_sharpe', np.nan))} | {_num(getattr(r, 'wf_sharpe', np.nan))} | "
                     f"{_pct(getattr(r, 'wf_avg_monthly', np.nan))} | {_pct(getattr(r, 'wf_max_dd', np.nan))} | "
                     f"{_num(r.bh_test_sharpe)} |")
    return "\n".join(lines) + "\n"


def run(universe: str, timeframe: str) -> None:
    wf = _wf_record(universe, timeframe)
    if wf is None:
        LOG.warning("%s %s: no walk-forward record of %s yet (scripts/per_asset.py --names ml_feature_search)",
                    universe, timeframe, ml_feature_search.name)
    ids, _ = instruments_now(universe, timeframe)
    rows = []
    for i in ids:
        try:
            rows.append({"universe": universe, **replay(i, timeframe, wf[i] if wf is not None and i in wf else None)})
        except (ValueError, KeyError) as e:
            LOG.warning("%s %s: not replayed: %s", i, timeframe, e)
        LOG.info("%s %s done", i, timeframe)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"{universe}__{timeframe}.csv", index=False)
    (OUT / f"{universe}__{timeframe}.json").write_text(json.dumps(
        {"code": provenance.stamp(inspect.getsourcefile(ml_feature_search.fn)), "walk_forward_found": wf is not None},
        indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes", nargs="*", default=[])
    ap.add_argument("--timeframes", nargs="*", default=list(TIMEFRAMES))
    ap.add_argument("--render", action="store_true", help="write reports/product_protocol.md from the saved tables")
    ap.add_argument("--outside-lists", action="store_true",
                    help="allow lists outside the ML task's markets: only when the user asks for them")
    args = ap.parse_args()
    if not args.outside_lists:
        lists.refuse_outside(args.universes, lists.ML_TASK, "the replay of the 70/30 protocol")
    log.setup("product_protocol")
    for u in args.universes:
        for tf in args.timeframes:
            run(u, tf)
    if args.render:
        tables = [pd.read_csv(f) for f in sorted(OUT.glob("*__*.csv"))]
        (REPORTS_DIR / "product_protocol.md").write_text(render(pd.concat(tables, ignore_index=True)))
        LOG.info("wrote %s from %d tables", REPORTS_DIR / "product_protocol.md", len(tables))


if __name__ == "__main__":
    main()
