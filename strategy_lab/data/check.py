"""Is the data behind the research lists whole? Read-only, no network:

    python -m strategy_lab.data.check                                   # every agreed list, 1h 4h 1d
    python -m strategy_lab.data.check --lists crypto_top100 --timeframes 1h --out reports/data_check.md

For every agreed list (strategy_lab.lists) and research timeframe, the problems a run would otherwise meet in
silence, each counted on the days the list holds the name:

  * missing: no bars of that timeframe on a day the list holds the name. A ranked list is ranked from what is stored,
    so such a name leaves no hole in a record: the next name takes its seat unseen. Who a list holds is therefore read
    from its daily bars and ranking, where every name has bars, as the same names at every timeframe. A perp with no
    funding settlements stored is missing too: a backtest refuses it;
  * short: a series that starts later than the vendor's history. All of a Binance pair's klines start at its listing,
    so its intraday series must start with its daily one; a US listing's intraday series must start at its first
    session since 2019-01-07, from where Databento's minutes fill the sessions Twelve Data lacks (`refresh
    databento-hourly`); an FX pair's or a commodity's at Twelve Data's first bar (data/reference/
    twelvedata_first_bars.csv, `refresh twelvedata-first-bars`); a CME future's with its daily one, which is built
    from it. Hourly bars dropped on purpose because they belonged to another company (the series' meta) are named, not
    counted;
  * holes: bars missing inside a series against its market's calendar: an NYSE session of a US listing, an FX
    weekday of a currency pair or metal, a day CME trades a future on (`calendars.cme_days`), a bar of a coin; a long
    one, or short ones that add up to more than a long one;
  * stopped: a series that ends more than a week before the list's last day although the instrument still trades
    (Binance lists the contract as trading, the stock is in the S&P 500 today, a name of a fixed list): a refresh
    that no longer reaches it;
  * dead: three days or more at an unchanged price with no volume: a halted or settled market still printing, where
    a position would be marked at a price nobody traded;
  * joined: after more than three days without trading (five for a US listing or an FX pair, whose long weekends
    last up to four), the price resumes at less than half or more than twice the last traded one: two contracts
    under one name (a delisted coin and a later one) or a redenomination, where a record would see a move nobody
    could trade;
  * not crypto: a name in a crypto list that is not a crypto asset: a contract Binance does not class as a coin, a
    stablecoin, a token of gold or silver.

The exit status is 1 when a list has a problem.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from strategy_lab import lists, log, universes
from strategy_lab.config import REFERENCE_DIR, TIMEFRAMES
from strategy_lab.data import calendars as cal
from strategy_lab.data import store
from strategy_lab.data.bars import load_panel
from strategy_lab.data.instruments import parse
from strategy_lab.data.refresh import DATABENTO_FROM

LOG = log.get("check")
SLACK = pd.Timedelta(days=3)          # a long weekend: a series may start or stop this far off without a bar missing
DEAD = pd.Timedelta(days=3)
STEP = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}
RECENT = pd.Timedelta(days=365)
# a hole worth naming: longer than this, in the units of its calendar (bars of a coin, sessions, FX weekdays)
LONG_HOLE = {"24x7": {"1h": 24, "4h": 6, "1d": 1}, "xnys": 1, "fx": 1}
# the fixed lists as defined, before `universes.resolve` keeps only the names with bars
DEFINED = {"etf_core": [f"td:{s}" for s in universes.ETF_CORE],
           "stockhunt_etfs": [f"td:{s}" for s in universes.STOCKHUNT_ETFS],
           "fx_majors": [f"td:{s}" for s in universes.FX_MAJORS],
           "stockhunt_commodities": [f"td:{s}" for s in universes.STOCKHUNT_COMMODITIES],
           "cme_futures": universes.CME_FUTURES}


@dataclass
class Series:
    """One stored series, summarised once: its span, its holes and its dead runs (each as first and last date)."""
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    holes: list[tuple[pd.Timestamp, pd.Timestamp, int]] = field(default_factory=list)   # (from, to, units missing)
    dead: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    joins: list[tuple[pd.Timestamp, float, pd.Timestamp, float]] = field(default_factory=list)  # (day, close, day, open)


@dataclass
class Finding:
    universe: str
    tf: str
    kind: str          # missing | short | holes | stopped | dead | joined | not crypto
    instrument: str
    days: int          # days the list holds the name inside the problem (0: a series-level fact)
    detail: str
    recent_days: int = 0


def _dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC dates, without a time zone: the unit every span here is counted in."""
    return idx.tz_convert("UTC").tz_localize(None).normalize()


def _day(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_convert("UTC").tz_localize(None).normalize()


def _session_days(ins, idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The trading day each bar belongs to: the New York session date, the FX day, the UTC date of a coin's bar
    (a bar is stamped when it closes, so a second before the stamp is inside it)."""
    inside = idx - pd.Timedelta(seconds=1)
    if ins.calendar == "fx":
        return pd.DatetimeIndex(cal.fx_day_label(inside))
    if ins.calendar == "xnys":
        return inside.tz_convert(cal.NY).tz_localize(None).normalize()
    return inside.tz_convert("UTC").tz_localize(None).normalize()


def _expected_days(ins, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DatetimeIndex:
    """Trading days from lo to hi (naive dates): NYSE sessions, CME's days for its futures, or FX weekdays without
    Christmas and New Year."""
    if ins.source == "cme":
        return cal.cme_days(pd.Timestamp(lo), pd.Timestamp(hi))
    if ins.calendar == "xnys":
        return pd.DatetimeIndex(cal.nyse_sessions(pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC")).index)
    days = pd.bdate_range(lo, hi)
    return days[~((days.month == 12) & (days.day == 25) | (days.month == 1) & (days.day == 1))]


@lru_cache(maxsize=None)
def series(instrument: str, tf: str) -> Series:
    ins = parse(instrument)
    if not store.path(ins.source, tf, ins.symbol).exists():
        return Series()
    df = store.read_bars(ins.source, tf, ins.symbol)
    if df.empty:
        return Series()
    idx = df.index
    out = Series(idx.min(), idx.max())
    if ins.calendar == "24x7":
        gap = idx.to_series().diff()
        for end, g in gap[gap > STEP[tf]].items():
            out.holes.append((_day(end - g + STEP[tf]), _day(end), int(g / STEP[tf]) - 1))
    else:
        have = pd.DatetimeIndex(_session_days(ins, idx).unique()).sort_values()
        want = _expected_days(ins, have.min(), have.max())
        miss = want.difference(have)
        if len(miss):
            run = (pd.Series(want.isin(miss), index=want).astype(int).diff().fillna(1) != 0).cumsum()
            for _, days in pd.Series(want[want.isin(miss)], index=want[want.isin(miss)]).groupby(run[want.isin(miss)]):
                out.holes.append((days.iloc[0], days.iloc[-1], len(days)))
    flat = (df["volume"].fillna(0) == 0) & (df["close"].diff() == 0)
    if flat.any():
        run = (flat != flat.shift()).cumsum()
        for _, part in flat[flat].groupby(run[flat]):
            lo, hi = part.index[0] - STEP[tf], part.index[-1]          # the price stood still since the bar before
            if hi - lo >= DEAD:
                out.dead.append((_day(lo), _day(hi)))
    between = [(pd.Timestamp(lo), pd.Timestamp(hi)) for lo, hi in
               store.read_meta(ins.source, tf, ins.symbol).get("delisted_periods", [])]
    if between:                                        # delisted, then listed anew: no market, not a hole
        out.holes = [h for h in out.holes if not any(_day(lo) <= h[0] and h[1] <= _day(hi) for lo, hi in between)]
    traded = df[df["volume"].fillna(0) > 0]           # FX and metals print no volume: nothing to compare there
    pause = traded.index.to_series().diff()
    # a coin trades every day; a US listing or an FX pair pauses over a long weekend, a clock change adding an hour
    closed = SLACK if ins.calendar == "24x7" else pd.Timedelta(days=5)
    for k in np.flatnonzero((pause > closed).to_numpy()):
        before, after = traded["close"].iloc[k - 1], traded["open"].iloc[k]
        relisted = any(lo <= traded.index[k - 1] and traded.index[k] <= hi for lo, hi in between)
        if before > 0 and not relisted and not 0.5 <= after / before <= 2.0:
            out.joins.append((_day(traded.index[k - 1]), float(before), _day(traded.index[k]), float(after)))
    return out


def held(universe: str) -> pd.DataFrame:
    """Dates x instruments, True where the list holds the name, read from its daily bars: a ranked list's daily
    ranking, a fixed list's names from their first daily bar to their last (a name with no daily bars: every date)."""
    uni = universes.resolve(universe, "1d")
    start = lists.start(universe)
    panel = load_panel(uni.ids, "1d", start=start)
    if uni.member is not None:
        mask = uni.member(panel)
    else:
        has = panel.close.notna()
        mask = panel.started & (has[::-1].cumsum()[::-1] > 0)
        for i in DEFINED.get(universe, []):
            if i not in mask:
                mask[i] = True
    mask.index = _dates(mask.index)
    return mask[~mask.index.duplicated(keep="last")].astype(bool)


@lru_cache(maxsize=1)
def _vendor_first() -> dict[str, pd.Timestamp]:
    """The vendor's first hourly bar per Twelve Data symbol (`refresh twelvedata-first-bars`)."""
    p = REFERENCE_DIR / "twelvedata_first_bars.csv"
    if not p.exists():
        LOG.warning("%s is missing: Twelve Data intraday series are not checked against the vendor's history", p)
        return {}
    t = pd.read_csv(p).dropna(subset=["first"])
    return {r.symbol: pd.Timestamp(r.first) for r in t[t["interval"] == "1h"].itertuples(index=False)}


def _cut_on_purpose(ins, first: pd.Timestamp) -> str | None:
    """Why the hourly bars before `first` were dropped on purpose, if they were (another company's bars)."""
    if ins.source not in ("td", "sh") or not store.meta_path(ins.source, "1h", ins.symbol).exists():
        return None
    basis = store.read_meta(ins.source, "1h", ins.symbol).get("daily_basis") or {}
    through = basis.get("dropped_through")
    if through and pd.Timestamp(through, tz="UTC") >= first.normalize() - SLACK:
        return f"hourly bars through {through} dropped: they did not match the daily bars (another company)"
    return None


def _short(instrument: str, tf: str) -> str | None:
    ins, s = parse(instrument), series(instrument, tf)
    if tf == "1d" or s.first is None:
        return None
    if ins.source in ("perp", "spot", "cme"):
        d = series(instrument, "1d")
        why = "both start at the listing" if ins.source != "cme" else "its daily bars are built from its hourly ones"
        if d.first is not None and s.first - d.first > SLACK:
            return f"{tf} bars from {s.first.date()}, daily bars from {d.first.date()} ({why})"
        return None
    if "/" not in ins.symbol:                          # a US listing: its sessions since Databento's minutes begin
        d = series(instrument, "1d")
        want = max(d.first, pd.Timestamp(DATABENTO_FROM, tz="UTC")) if d.first is not None else None
        if want is not None and s.first - want > SLACK:
            why = _cut_on_purpose(ins, s.first)
            return f"{tf} bars from {s.first.date()}, its sessions from {want.date()}" + (f" ({why})" if why else "")
        return None
    v = _vendor_first().get(ins.symbol)
    if v is not None and s.first - v > SLACK:
        why = _cut_on_purpose(ins, s.first)
        return f"{tf} bars from {s.first.date()}, the vendor's from {v.date()}" + (f" ({why})" if why else "")
    return None


def _prices(instrument: str) -> pd.Series:
    ins = parse(instrument)
    c = store.read_bars(ins.source, "1d", ins.symbol)["close"]
    c.index = _dates(c.index)
    return c[~c.index.duplicated(keep="last")]


@lru_cache(maxsize=None)
def _warn_once(message: str) -> None:
    LOG.warning(message)


@lru_cache(maxsize=1)
def _perp_contracts() -> pd.DataFrame:
    p = REFERENCE_DIR / "binance_perp_contracts.csv"
    cols = ["symbol", "underlying_type", "status"]
    return pd.read_csv(p, usecols=cols).set_index("symbol") if p.exists() else pd.DataFrame(columns=cols[1:])


def _perp_types() -> dict[str, str]:
    return _perp_contracts()["underlying_type"].to_dict()


def _still_trades(universe: str, instrument: str) -> str | None:
    """Why the instrument should still print bars today, if it should."""
    ins = parse(instrument)
    if instrument in DEFINED.get(universe, []) and ins.source in ("td", "cme"):
        return "a name of the list"
    if ins.source == "perp":
        c = _perp_contracts()
        return "Binance lists the contract as trading" if c.get("status", {}).get(ins.symbol) == "TRADING" else None
    if ins.source in ("td", "sh") and "/" not in ins.symbol:
        m = universes.sp500_membership()
        return "an S&P 500 member today" if (m[m["ticker"] == ins.symbol]["end"].isna()).any() else None
    return None


def not_crypto(instrument: str) -> str | None:
    """Why a name in a crypto list is not a crypto asset, if it is not."""
    ins = parse(instrument)
    kind = _perp_types().get(ins.symbol) if ins.source == "perp" else None
    if kind is not None and kind != "COIN":
        return f"Binance classes the contract as {kind}, not a coin"
    if not store.path(ins.source, "1d", ins.symbol).exists():
        return None
    px = _prices(instrument)
    if len(px) >= 30 and ((px - 1.0).abs() < 0.03).mean() >= 0.9:
        return f"a stablecoin: within 3% of $1 on {((px - 1.0).abs() < 0.03).mean():.0%} of days"
    for metal, name in (("XAU/USD", "gold"), ("XAG/USD", "silver")):
        if not store.path("td", "1d", metal).exists():
            _warn_once(f"td:{metal} has no daily bars in the store: tokens of {name} are not looked for")
            continue
        ref = _prices(f"td:{metal}")
        both = pd.concat([px, ref], axis=1, join="inner").dropna()
        if len(both) >= 30:
            near = ((both.iloc[:, 0] / both.iloc[:, 1] - 1.0).abs() < 0.1).mean()
            if near >= 0.9:
                return f"a token of {name}: within 10% of its price on {near:.0%} of days"
    return None


def _has_funding(instrument: str) -> bool:
    """A perp's funding settlements are stored (anything but a perp needs none)."""
    ins = parse(instrument)
    return ins.source != "perp" or (store.STORE_DIR / "perp" / "funding" / f"{store.safe_name(ins.symbol)}.parquet").exists()


def _held_inside(hold: pd.Series, spans, last_day: pd.Timestamp) -> tuple[int, int]:
    """Held days inside (lo, hi) date spans, overall and within the list's last 12 months."""
    days = hold.index[hold.to_numpy()]
    if not len(days) or not spans:
        return 0, 0
    inside = np.zeros(len(days), dtype=bool)
    for lo, hi in spans:
        inside |= (days >= lo) & (days <= hi)
    return int(inside.sum()), int((inside & (days >= last_day - RECENT)).sum())


def _every_day(hold: pd.Series) -> pd.Series:
    """A name's held days on every calendar date between its first and last: a date the panel has no row for is held
    as the date before it."""
    if hold.empty:
        return hold
    every = pd.date_range(hold.index.min(), hold.index.max(), freq="D")
    return hold.astype(float).reindex(every).ffill().fillna(0.0).astype(bool)


def _holes(universe: str, tf: str, spans: dict[str, tuple[Series, pd.Series]], last_day) -> list[Finding]:
    """Long holes while the list holds the name, and short ones when they add up to more than a long one (AUD/USD
    lacked 97 Fridays of 2007-2009 one at a time, unnamed, until the New York clock of Dukascopy's candles there was
    found). A hole most of the list shares is the vendor's or the market's, not one series': it is named once for the
    list."""
    # a day no name of the list printed is not in its daily panel (a vendor's lost days: CME's 2014-09-23..25): held
    # as the day before it, a hole every name shares is still seen
    spans = {i: (s, _every_day(hold)) for i, (s, hold) in spans.items()}
    big = {}
    for i, (s, hold) in spans.items():
        cal_ = parse(i).calendar
        limit = LONG_HOLE[cal_][tf] if cal_ == "24x7" else LONG_HOLE[cal_]
        held = [h for h in s.holes if _held_inside(hold, [(h[0], h[1])], last_day)[0]]
        many_short = sum(h[2] for h in held if h[2] <= limit) > limit
        big[i] = [h for h in held if h[2] > limit or many_short]
    with_bars = sum(1 for s, hold in spans.values() if s.first is not None and hold.any())
    count = Counter((h[0], h[1]) for hs in big.values() for h in hs)
    shared = {span for span, k in count.items() if k >= max(3, with_bars / 2)}
    out = []
    for lo, hi in sorted(shared):
        units = next(h[2] for hs in big.values() for h in hs if (h[0], h[1]) == (lo, hi))
        unit = {"24x7": "bars", "xnys": "sessions", "fx": "weekdays"}[parse(next(iter(spans))).calendar]
        out.append(Finding(universe, tf, "holes", f"{count[(lo, hi)]} of {with_bars} names", 0,
                           f"the same hole in most of the list: {lo.date()}..{hi.date()}, {units} {unit}"))
    for i, hs in big.items():
        own = [h for h in hs if (h[0], h[1]) not in shared]
        if not own:
            continue
        d, r = _held_inside(spans[i][1], [(h[0], h[1]) for h in own], last_day)
        worst = max(own, key=lambda h: h[2])
        unit = {"24x7": "bars", "xnys": "sessions", "fx": "weekdays"}[parse(i).calendar]
        out.append(Finding(universe, tf, "holes", i, d, f"{len(own)} hole(s) while held; the longest "
                           f"{worst[0].date()}..{worst[1].date()}, {worst[2]} {unit}", r))
    return out


def check_list(universe: str, tfs: tuple[str, ...]) -> tuple[list[Finding], dict]:
    mask = held(universe)
    ever = [i for i in mask.columns if mask[i].any()]
    found: list[Finding] = []
    summary: dict = {"names": len(ever)}
    last_day = mask.index.max()
    crypto = all(parse(i).source in ("perp", "spot") for i in ever)
    for i in ever:
        if not _has_funding(i):
            d, r = _held_inside(mask[i], [(mask.index.min(), last_day)], last_day)
            found.append(Finding(universe, "all", "missing", i, d, "no funding settlements stored: a backtest refuses "
                                 "a perp without them", r))
        elif parse(i).source == "perp" and series(i, "1d").first is not None:
            first_rate = pd.read_parquet(store.STORE_DIR / "perp" / "funding" /
                                         f"{store.safe_name(parse(i).symbol)}.parquet").index.min()
            if first_rate - series(i, "1d").first > SLACK:
                found.append(Finding(universe, "all", "short", i, 0, f"funding settlements from {first_rate.date()}, "
                                     f"daily bars from {series(i, '1d').first.date()}: a position before then carries "
                                     "for free"))
    if crypto:
        for i in ever:
            why = not_crypto(i)
            if why:
                d, r = _held_inside(mask[i], [(mask.index.min(), last_day)], last_day)
                found.append(Finding(universe, "all", "not crypto", i, d, why, r))
    for tf in tfs:
        spans = {i: series(i, tf) for i in ever}
        firsts = [_day(s.first) for s in spans.values() if s.first is not None]
        history = min(firsts) if firsts else last_day      # the store's first bar of this timeframe for the list
        window = mask.index >= history
        recent_seats = int(mask[mask.index >= last_day - RECENT].to_numpy().sum())
        recent_lost = 0
        for i in ever:
            s, hold, ins = spans[i], mask[i] & window, parse(i)
            if not hold.any():
                continue
            vendor = _vendor_first().get(ins.symbol) if ins.source == "td" and tf != "1d" else None
            begin = max(history, _day(vendor)) if vendor is not None else history
            if s.first is None:
                d, r = _held_inside(hold, [(begin, last_day)], last_day)
                parts = [(d, r, f"no {tf} bars")] if d else []
            else:
                before = _held_inside(hold, [(begin, _day(s.first) - SLACK)], last_day)
                after = _held_inside(hold, [(_day(s.last) + SLACK, last_day)], last_day)
                parts = [(before[0], before[1], f"before its first {tf} bar ({s.first.date()})"),
                         (after[0], after[1], f"after its last {tf} bar ({s.last.date()})")]
            for d, r, text in parts:
                if d:
                    found.append(Finding(universe, tf, "missing", i, d, text, r))
                    recent_lost += r
            why = _short(i, tf)
            if why:
                found.append(Finding(universe, tf, "short", i, 0, why))
            if s.last is not None and last_day - _day(s.last) > pd.Timedelta(days=7):
                why = _still_trades(universe, i)
                if why:
                    found.append(Finding(universe, tf, "stopped", i, 0,
                                         f"{tf} bars end {s.last.date()}, though it still trades ({why})"))
            for b_day, b_px, a_day, a_px in s.joins:
                found.append(Finding(universe, tf, "joined", i, 0,
                                     f"{b_px:.6g} on {b_day.date()}, then {a_px:.6g} on {a_day.date()} after "
                                     f"{(a_day - b_day).days} days without trading"))
            d, r = _held_inside(hold, s.dead, last_day)
            if d:
                lo, hi = max(s.dead, key=lambda x: x[1] - x[0])
                found.append(Finding(universe, tf, "dead", i, d, f"unchanged price, no volume: {len(s.dead)} run(s), "
                                     f"the longest {lo.date()}..{hi.date()}", r))
        found += _holes(universe, tf, {i: (spans[i], mask[i] & window) for i in ever}, last_day)
        summary[tf] = {"from": history, "recent_share": recent_lost / recent_seats if recent_seats else 0.0}
    return found, summary


def report(results: dict[str, tuple[list[Finding], dict]], tfs: tuple[str, ...]) -> str:
    kinds = ("missing", "short", "holes", "stopped", "dead", "joined", "not crypto")
    lines = ["# Data check", "",
             "Each list on each timeframe, against the days it holds each name (a ranked list: its daily ranking). "
             "A count is of names; *missing* also gives the share of the list's seats of the last 12 months held "
             "by a name with no bars of that timeframe (another name takes the seat unseen).", "",
             "| list | tf | names | bars from | missing | short | holes | stopped | dead | joined | not crypto |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for u, (found, summ) in results.items():
        nc = len({f.instrument for f in found if f.kind == "not crypto"})
        for tf in tfs:
            f_tf = [f for f in found if f.tf == tf]
            n = {k: len({f.instrument for f in f_tf if f.kind == k}) for k in kinds[:-1]}
            s = summ[tf]
            miss = f"{n['missing']} ({s['recent_share']:.1%} of seats)" if n["missing"] else "0"
            lines.append(f"| {lists.title(u)} | {tf} | {summ['names']} | {s['from'].date()} | {miss} | {n['short']} | "
                         f"{n['holes']} | {n['stopped']} | {n['dead']} | {n['joined']} | {nc} |")
    for kind in kinds:
        rows = [f for found, _ in results.values() for f in found if f.kind == kind]
        if not rows:
            continue
        lines += ["", f"## {kind}", ""]
        for u in results:
            mine = sorted((f for f in rows if f.universe == u), key=lambda f: (f.tf, -f.days, f.instrument))
            for f in mine:
                held_txt = f", held {f.days} days ({f.recent_days} in the last 12 months)" if f.days else ""
                lines.append(f"* {lists.title(u)} {f.tf}: `{f.instrument}` — {f.detail}{held_txt}")
    return "\n".join(lines) + "\n"


def agreed_lists() -> list[str]:
    return list(dict.fromkeys(lists.names(lists.OURS) + list(lists.ML_TASK)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lists", nargs="*", help="default: every agreed list (strategy_lab.lists)")
    ap.add_argument("--timeframes", nargs="*", default=list(TIMEFRAMES))
    ap.add_argument("--out", help="also write the report to this Markdown file")
    args = ap.parse_args()
    log.setup("data_check")
    names = args.lists or agreed_lists()
    lists.refuse_outside(names, agreed_lists(), "the data check")
    tfs = tuple(args.timeframes)
    results = {}
    for u in names:
        LOG.info("checking %s", u)
        results[u] = check_list(u, tfs)
    text = report(results, tfs)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    sys.exit(1 if any(found for found, _ in results.values()) else 0)


if __name__ == "__main__":
    main()
