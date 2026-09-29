"""Is a rolling walk-forward better? The walk-forward here chooses each window's parameters on all the history before
it (a growing window); a rolling one chooses on the last `train` of it only (`config.WF_SCHEMES`): it follows a
changed market sooner, and late in a long record it chooses on a small part of the data. The first window is chosen
on the first year either way, and a window of three years grows to three years before it rolls (user, 2026-09-27,
who asked for the comparison on more than ten results, and for three years beside one).

Every result the research runs that has a choice to make is evaluated with each window on the same data, code, windows
and costs, and compared out-of-sample with the growing one: the jobs of scripts/week1.py whose strategy has no model
(a model's own training data would be a second difference) and more than one configuration on its list.

    python scripts/rolling_walkforward.py [--workers 8] [--train 365D 1095D]
        -> reports/rolling_walkforward.csv, reports/rolling_walkforward.md
"""
from __future__ import annotations

import argparse
import functools
from unittest import mock

import numpy as np
import pandas as pd

from strategy_lab import batch, config, lists, log
from strategy_lab import evaluate as ev
from strategy_lab import robustness as rb
from strategy_lab.config import REPORTS_DIR
from strategy_lab.strategy import load
from strategy_lab.universes import resolve
from week1 import RUNS

LOG = log.get("rolling_walkforward")
FIGURES = ("sharpe", "avg_monthly", "cagr", "max_dd", "pct_green_active", "vs_bh")
MARKET = {x.universe: x.market for x in lists.OURS}


SAMPLE_LISTS = ("us_stocks_top10", "crypto_top10", "stockhunt_etfs", "fx_majors", "stockhunt_commodities",
                "cme_futures")                  # a market's list in the sample: its ten-name step or the market whole
TIMEFRAMES = ("1d", "4h", "1h")


def sample(todo: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Each strategy on one list of each market, the timeframes taken in turn across markets and strategies, so every
    strategy, market and timeframe is in it: about a sixth of the results, and not the heaviest (a hundred names by the
    hour). A strategy that runs on wider lists only takes the market's first of them, on its daily bars."""
    markets = list(dict.fromkeys(MARKET.values()))
    out = []
    for i, name in enumerate(dict.fromkeys(j[0] for j in todo)):
        for k, market in enumerate(markets):
            its = [j for j in todo if j[0] == name and MARKET[j[1]] == market]
            if not its:
                continue
            u = next((x for x in SAMPLE_LISTS if any(j[1] == x for j in its)), its[0][1])
            tf = TIMEFRAMES[(i + k) % 3] if u in SAMPLE_LISTS else "1d"
            on_list = [j for j in its if j[1] == u]
            out.append(next((j for j in on_list if j[2] == tf), on_list[0]))
    return out


def jobs() -> list[tuple[str, str, str]]:
    """week1's jobs whose walk-forward has a choice to make and no model of its own."""
    capacity: dict[tuple[str, str], int] = {}
    out = []
    for name, universes, timeframes in RUNS:
        s = load(name)
        if s.grade is not None or s.prepare is not None:
            continue
        for u in universes:
            for tf in timeframes:
                if (u, tf) not in capacity:
                    capacity[u, tf] = resolve(u, tf).capacity
                if len(s.configs(capacity[u, tf])) > 1:
                    out.append((name, u, tf))
    return out


def _evaluated(job, train: str | None) -> ev.Evaluation:
    strategy, universe, tf = job
    with mock.patch.dict(config.WF_SCHEMES[tf], {"train": train}):
        return ev.evaluate(load(strategy), universe, tf, start=lists.start(universe), monte_carlo=False,
                           robustness=False, save=False)


def _figures(name: str, e: ev.Evaluation) -> dict:
    out = {f"{name}_{k}": e.oos[k] for k in FIGURES}
    out[f"{name}_changes"] = int((e.folds["config"].diff().fillna(0) != 0).sum())
    return out


def run(job, trains: tuple[str, ...]) -> dict:
    """One result with every window: each one's out-of-sample figures and how often its choice changed; for a rolling
    one, the windows where it chose otherwise than the growing one, and how far its Sharpe is above the growing one's
    in standard errors of the difference (`robustness.paired_t`: the two records resampled together)."""
    row = {"strategy": job[0], "universe": job[1], "timeframe": job[2]}
    try:
        grown = _evaluated(job, None)
        row.update({"start": grown.oos["start"], "end": grown.oos["end"], "windows": len(grown.folds),
                    "configurations": int(np.prod([len(v) for v in grown.grid.values()]))})
        row.update(_figures("expanding", grown))
        for train in trains:
            rolled = _evaluated(job, train)
            row.update(_figures(train, rolled))
            row[f"{train}_picks_differ"] = int((grown.folds["config"].to_numpy()
                                                != rolled.folds["config"].to_numpy()).sum())
            paired = rb.paired_t(rolled.oos_daily, grown.oos_daily)
            if "t" not in paired:
                LOG.warning("%s on %s %s, %s: no paired t (%s)", *job, train, paired)
            row[f"{train}_t"] = paired.get("t")
    except Exception as e:                      # noqa: BLE001 - reported in the table, never swallowed
        LOG.error("FAILED %s on %s %s: %s", *job, e)
        row["error"] = repr(e)
    return row


def _med(s: pd.Series, pct: bool = False) -> str:
    v = s.median()
    return "—" if pd.isna(v) else f"{v * 100:+.2f}%" if pct else f"{v:+.2f}"


def _summary(t: pd.DataFrame, train: str) -> list[str]:
    """How the rolling window of `train` did against the growing one, over all results and by their groups."""
    d = t[f"{train}_sharpe"] - t["expanding_sharpe"]
    tt = t[f"{train}_t"]
    month = t[f"{train}_avg_monthly"] - t["expanding_avg_monthly"]
    dd = t[f"{train}_max_dd"] - t["expanding_max_dd"]                 # negative: deeper
    held = t[f"{train}_vs_bh"] - t["expanding_vs_bh"]
    out = [f"\n## Rolling {train} against growing\n",
           f"Sharpe higher on {int((d > 0).sum())} of {len(t)} results, lower on {int((d < 0).sum())}, the same on "
           f"{int((d == 0).sum())}; median difference {_med(d)}, mean {d.mean():+.3f}. Beyond noise (|t| ≥ 2): higher "
           f"on {int((tt >= 2).sum())}, lower on {int((tt <= -2).sum())}. Average month {_med(month, True)} at the "
           f"median; max drawdown deeper on {int((dd < 0).sum())} (median {_med(dd, True)}); vs buy & hold "
           f"{_med(held, True)} a year at the median. Changes of choice: median "
           f"{t['expanding_changes'].median():.0f} growing, {t[f'{train}_changes'].median():.0f} rolling.\n",
           "| Group | Results | Sharpe higher | lower | Median Sharpe difference | Beyond noise, higher / lower |",
           "|---|---:|---:|---:|---:|---|"]
    windows = pd.cut(t["windows"], [0, 30, 80, 10_000], labels=["up to 30 windows", "31-80", "over 80"])
    for label, key in (("Timeframe", t["timeframe"]), ("Market", t["universe"].map(MARKET)), ("History", windows)):
        for g, part in t.groupby(key, observed=True):
            dg, tg = d[part.index], tt[part.index]
            out.append(f"| {label}: {g} | {len(part)} | {int((dg > 0).sum())} | {int((dg < 0).sum())} | {_med(dg)} "
                       f"| {int((tg >= 2).sum())} / {int((tg <= -2).sum())} |")
    per = d.groupby(t["strategy"]).median().sort_values()
    out.append(f"\nBy strategy (a rule on nested lists and neighbouring timeframes is one bet): the median difference "
               f"is above zero for {int((per > 0).sum())} of {len(per)} strategies, below for "
               f"{int((per < 0).sum())}.\n")
    out.append("| Strategy | Results | Median Sharpe difference | Higher / lower |")
    out.append("|---|---:|---:|---|")
    for s, v in per.items():
        ds = d[t["strategy"] == s]
        out.append(f"| {s} | {len(ds)} | {v:+.2f} | {int((ds > 0).sum())} / {int((ds < 0).sum())} |")
    return out


def render(t: pd.DataFrame, trains: tuple[str, ...]) -> str:
    failed = t[t["error"].notna()] if "error" in t else t.iloc[:0]
    done = t.drop(failed.index)
    out = [f"# Rolling walk-forward against the growing one, {len(done)} results\n",
           "Out-of-sample, the same windows, data, code and costs; the first window chosen on the first year either "
           "way. Sharpe difference: rolling minus growing. Beyond noise: the rolling Sharpe above or below the growing "
           "one by two standard errors of the difference or more.\n"]
    for train in trains:
        out += _summary(done, train)
    if len(failed):
        out.append(f"\nFailed ({len(failed)}): " + "; ".join(f"{r.strategy} on {r.universe} {r.timeframe}: {r.error}"
                                                             for r in failed.itertuples()))
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train", nargs="+", default=["365D", "1095D"],
                    help="the history each rolling window chooses on (default: one year, three years)")
    ap.add_argument("--all", action="store_true", help="every result, not the sample of each strategy on each market")
    args = ap.parse_args()
    log.setup("rolling_walkforward")
    todo = jobs() if args.all else sample(jobs())
    lists.refuse_outside([u for _, u, _ in todo], [x.universe for x in lists.OURS], "a basket")
    LOG.info("%d results, each with the growing window and %s", len(todo), ", ".join(args.train))
    times = batch.recorded()
    done = batch.run(functools.partial(run, trains=tuple(args.train)), todo, workers=args.workers,
                     group=lambda j: (j[1], j[2]), seconds=lambda j: (1 + len(args.train)) * batch.expected(times, *j))
    t = pd.DataFrame([r for r, _ in done])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(REPORTS_DIR / "rolling_walkforward.csv", index=False)
    (REPORTS_DIR / "rolling_walkforward.md").write_text(render(t, tuple(args.train)))
    LOG.info("wrote %s", REPORTS_DIR / "rolling_walkforward.md")


if __name__ == "__main__":
    main()
