"""Does checking a pick on similar instruments help? Here the
idea is a selection rule inside the walk-forward, defined with the user on 2026-09-24 (`walkforward.choose_with_peers`):
over the window a configuration is chosen on, it must have beaten holding on at least half of the other instruments
of the same list (on a currency pair, whose buy-and-hold is cash, made money), and the best of those by the
instrument's own past is taken; with none, the next window holds no position. Every instrument's walk-forward is run with and without the check, on the same configurations, windows and
costs, and the two are compared out-of-sample: the ML matrix (strategies/ml_feature_search.py) on
the ML task's markets, and IBS and RSI(2) on the stock Top-100.

    python scripts/peer_check.py [--workers 10]     -> reports/peer_check.csv, reports/peer_check.md
"""
from __future__ import annotations

import argparse
import inspect
import json

import pandas as pd

from strategy_lab import batch, lists, log, metrics, per_asset, provenance, walkforward
from strategy_lab.config import REPORTS_DIR, TIMEFRAMES, WF_SCHEMES
from strategy_lab.data.bars import load_panel
from strategy_lab.engine.hold import margined, nothing_to_hold
from strategy_lab.evaluate import _buy_and_hold, _stitched, _walk_forward
from strategy_lab.strategy import load
from strategy_lab.universes import instruments_now

LOG = log.get("peer_check")
JOBS = ([("ml_feature_search", u, tf) for u in lists.ML_TASK for tf in TIMEFRAMES]
        + [("ibs", "us_stocks_top100", "1d"), ("rsi2_connors", "us_stocks_top100", "1d")])


def _exposure(runs, fl, choices) -> pd.Series:
    """The exposure the chosen configurations held over their test windows; a window without a choice holds none."""
    parts = []
    for f, c in zip(fl, choices):
        e = runs[c if c is not None else next(iter(runs))].exposure
        e = e[(e.index > f.test_start) & (e.index <= f.test_end)]
        parts.append(e if c is not None else e * 0.0)
    return pd.concat(parts)


def run(job: tuple[str, str, str]) -> list[dict]:
    name, universe, tf = job
    s = load(name)
    code = json.dumps(provenance.stamp(inspect.getsourcefile(s.fn)))
    configs = per_asset.single_asset_configs(s)
    ids, _ = instruments_now(universe, tf)
    data = {}
    for i in ids:
        panel = load_panel([i], tf)
        if panel.close[i].notna().sum() < 100:
            LOG.info("%s %s: fewer than 100 bars, left out", i, tf)
            continue
        runs, daily, _, _, _, _ = _walk_forward(s, panel, None, "next_open", configs, tf,
                                                signals={} if s.prepare is None else None)   # kept positions read back
        day = (panel.index - pd.Timedelta(microseconds=1)).tz_convert("UTC").normalize()
        priced = panel.close[i].notna().groupby(day).any().reindex(daily.index, fill_value=False)
        data[i] = (panel, runs, daily, _buy_and_hold(panel, None, "next_open").reindex(daily.index).fillna(0.0), priced,
                   nothing_to_hold(panel), margined(panel))
    rows = []
    scheme = WF_SCHEMES[tf]
    for i, (panel, runs, daily, bh, _, cash, futures) in data.items():
        fl = walkforward.folds(daily.index.min(), daily.index.max() + pd.Timedelta(days=1), scheme["first_train"],
                               scheme["test"], scheme["train"])
        peers = [(d, b, p) for j, (_, _, d, b, p, _, _) in data.items() if j != i]
        checked = walkforward.choose_with_peers(daily, fl, peers)
        variants = (("without", [c for c, _ in walkforward.choose(daily, fl)], [0] * len(fl)),
                    ("with", [c for c, _, _ in checked], [j for _, _, j in checked]))
        for variant, choices, judged in variants:
            oos = _stitched(panel, runs, daily, fl, choices)
            months = len(metrics.monthly_returns(oos))
            if months < per_asset.MIN_MONTHS:
                LOG.info("%s %s %s: %d out-of-sample months, too short to judge", name, i, tf, months)
                continue
            c = metrics.scorecard(oos, exposure=_exposure(runs, fl, choices),
                                  benchmark=bh.reindex(oos.index).fillna(0.0), cash=cash, margined=futures)
            rows.append({"strategy": name, "universe": universe, "timeframe": tf, "instrument": i, "variant": variant,
                         "oos_start": c["start"], "months": months, "sharpe": c["sharpe"],
                         "avg_monthly": c["avg_monthly"], "pct_green_active": c["pct_green_active"],
                         "max_dd": c["max_dd"], "cagr": c["cagr"], "bh_sharpe": c["bh_sharpe"],
                         "bh_max_dd": c["bh_max_dd"], "bh_cash": cash, "vs_bh": c["vs_bh"], "beats_bh": c["beats_bh"],
                         "windows": len(fl), "windows_flat": sum(x is None for x in choices),
                         "windows_checked": sum(j > 0 for j in judged), "code": code})
    LOG.info("%s %s %s: %d instruments compared", name, universe, tf, len(data))
    return rows


def _pct(v, signed=True) -> str:
    if pd.isna(v):
        return ""
    return f"{v * 100:+.2f}%" if signed else f"{v * 100:.0f}%"


def render(t: pd.DataFrame) -> str:
    out = ["# Checking a pick on similar instruments\n",
           "Here the idea of checking a pick on similar instruments is a rule inside "
           "the walk-forward, defined with the user on 2026-09-24: over the window a configuration is chosen on, it must "
           "have beaten holding (a higher Sharpe than the peer's buy-and-hold; above 0 on a currency pair, whose "
           "buy-and-hold is cash) on at least half of the other instruments of the same list that have enough history "
           "there, and the best of those by the instrument's own past is taken; if none passes, the next window (3 "
           "months) holds no position; with fewer than 3 such peers the check is not made. Each instrument's "
           "walk-forward runs with and without the check on the same configurations, windows, costs and bars; every "
           "figure is out-of-sample, the median over the list's instruments; beating buy & hold is the target's money "
           "test (metrics.vs_hold) (scripts/peer_check.py).\n"]
    head = ("| Strategy | List, tf | instruments | Sharpe without → with | avg/month without → with | green without → with "
            "| max DD without → with | beat B&H without → with | windows without a position | windows checked |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for (name, universe, tf), g in t.groupby(["strategy", "universe", "timeframe"], sort=False):
        wo, wi = g[g["variant"] == "without"], g[g["variant"] == "with"]
        # no month with a position (every window held nothing) leaves a figure undefined: no median of nothing
        med = lambda d, c: d[c].median() if d[c].notna().any() else float("nan")    # noqa: E731
        lines.append(
            f"| {name} | {lists.title(universe)} {tf} | {len(wi)} "
            f"| {med(wo, 'sharpe'):.2f} → {med(wi, 'sharpe'):.2f} "
            f"| {_pct(med(wo, 'avg_monthly'))} → {_pct(med(wi, 'avg_monthly'))} "
            f"| {_pct(med(wo, 'pct_green_active'), False)} → {_pct(med(wi, 'pct_green_active'), False)} "
            f"| {_pct(med(wo, 'max_dd'))} → {_pct(med(wi, 'max_dd'))} "
            f"| {int(wo['beats_bh'].sum())} → {int(wi['beats_bh'].sum())} "
            f"| {_pct(wi['windows_flat'].sum() / wi['windows'].sum(), False)} "
            f"| {_pct(wi['windows_checked'].sum() / wi['windows'].sum(), False)} |")
    out.append(head + "\n".join(lines) + "\n")
    better = t.pivot_table(index=["strategy", "universe", "timeframe", "instrument"], columns="variant",
                           values="sharpe").dropna()
    out.append(f"On {int((better['with'] > better['without']).sum())} of {len(better)} instrument-timeframe pairs the "
               "check raised the out-of-sample Sharpe, on "
               f"{int((better['with'] < better['without']).sum())} it lowered it.\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    log.setup("peer_check")
    times = batch.recorded()
    done = batch.run(run, JOBS, workers=args.workers, group=lambda j: (j[1], j[2]),
                     seconds=lambda j: batch.expected(times, *j))
    rows = [r for part, _ in done for r in part]
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS_DIR / "peer_check.csv", index=False)
    (REPORTS_DIR / "peer_check.md").write_text(render(t))
    LOG.info("wrote %s", REPORTS_DIR / "peer_check.md")


if __name__ == "__main__":
    main()
