"""Bring the store up to date: fetch only the missing tail of each series, with a request budget.

Rules (vendors are metered or rate-limited):
  * plan first: `--dry-run` computes how many requests the refresh needs and makes none;
  * every request is appended to logs/requests/<vendor>.jsonl (never the key);
  * HTTP 429 / 418 stops the run at once, no retry loop; what was fetched before it is already saved;
  * an empty answer is recorded in the series' meta, so the same question is not asked again;
  * Binance: only symbols the exchange lists as TRADING are asked; the bar still forming is dropped;
  * TwelveData: the tail is fetched with a two-bar overlap; if the overlap disagrees (a split or a vendor
    restatement) the whole series is re-fetched rather than stitched onto a differently adjusted history;
  * TwelveData history goes back to the vendor's first bar (`twelvedata-extend` fetches what is missing before the
    first stored one); its hourly US-equity bars before 2020-06-29 are re-stamped from their labels to where they
    really are (`td_equity_hourly_starts`).

    python -m strategy_lab.data.refresh binance --dry-run
    python -m strategy_lab.data.refresh twelvedata --symbols SPY EUR/USD --dry-run
    python -m strategy_lab.data.refresh binance-gaps --markets perp --dry-run      # interior gaps, from the exchange
    python -m strategy_lab.data.refresh binance-seed --markets perp --symbols HYPEUSDT   # new series, 1h 4h 1d
    python -m strategy_lab.data.refresh binance-delistings --dry-run    # delisted perps end at their delivery
    python -m strategy_lab.data.refresh binance-extend --dry-run        # perp history before the archive's 2020-01
    python -m strategy_lab.data.refresh dukascopy-extend --dry-run      # FX majors' hourly history before 2020
    python -m strategy_lab.data.refresh dukascopy-extend --symbols XAU/USD XAG/USD XPT/USD XPD/USD WTI/USD
    python -m strategy_lab.data.refresh dukascopy-fill --dry-run        # the hours the FX majors and metals lack
    python -m strategy_lab.data.refresh forexite-fill --dry-run         # the hours the FX majors lack, from Forexite
    python -m strategy_lab.data.refresh cme-hourly --dry-run            # the CME futures list's contracts
    python -m strategy_lab.data.refresh sharadar-tables                 # Sharadar's sp500, tickers, actions
    python -m strategy_lab.data.refresh sharadar-stocks --dry-run       # daily bars of the S&P 500 members since 1996
    python -m strategy_lab.data.refresh etf-daily --dry-run             # the ETFs' daily bars, Sharadar's
    python -m strategy_lab.data.refresh databento-hourly --dry-run      # hourly sessions Twelve Data lacks, priced
    python -m strategy_lab.data.refresh binance-archive --markets spot --symbols FTTUSDT --timeframes 1h   # delisted
    python -m strategy_lab.data.refresh twelvedata-seed --timeframes 1h 1d --symbols XAU/USD BRK.B --dry-run
    python -m strategy_lab.data.refresh twelvedata-extend --timeframes 1h 1d --universes us_stocks_top100 --dry-run
    python -m strategy_lab.data.refresh twelvedata-first-bars --dry-run      # the vendor's first hourly bars
    python -m strategy_lab.data.refresh dividends --dry-run                  # cash dividends of the stock and ETF lists
"""
from __future__ import annotations

import argparse
import io
import json
import lzma
import math
import re
import struct
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from strategy_lab import log
from strategy_lab.config import DATA_DIR, LOGS_DIR, REFERENCE_DIR, env
from strategy_lab.data import calendars as cal
from strategy_lab.data import integrity, resample, sharadar, store
from strategy_lab.data.bars import dividends_path
from strategy_lab.data.instruments import COMMODITIES, parse

LOG = log.get("refresh")
FAPI = "https://fapi.binance.com"
BINANCE_OPENED = pd.Timestamp("2017-07-14", tz="UTC")          # the exchange's first trading day
SPOT_API = "https://api.binance.com"
PERP_ARCHIVE_START = pd.Timestamp("2020-01-01", tz="UTC")     # the first month of the exchange's perp archive
ARCHIVE = "https://data.binance.vision"                        # the exchange's public monthly archives
ARCHIVE_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE_KLINES = {"perp": "data/futures/um/monthly/klines", "spot": "data/spot/monthly/klines"}
SHARADAR = "https://api.sharadar.com/v1.0"
SHARADAR_DIR = sharadar.DIR
SHARADAR_TABLES = ("sp500", "tickers", "actions")
SHARADAR_FROM = "1997-12-01"                                  # its stock prices begin at 1997-12-31
SHARADAR_PRICE_STEP = 0.001                                   # it rounds split-adjusted prices to three decimals
SP500_SINCE = pd.Timestamp("1996-01-01", tz="UTC")           # the stock lists' members: in the index since then
DATABENTO_DIR = DATA_DIR / "raw" / "databento"
DATABENTO_VENUES = ("XNAS.ITCH", "XNYS.PILLAR", "ARCX.PILLAR", "BATS.PITCH", "XBOS.ITCH", "XPSX.ITCH", "XASE.PILLAR")
DATABENTO_FROM = "2019-01-07"                                 # the store's first hourly US-equity bar
# a daily price published to three decimals, and a split ratio rounded to five (Sharadar's DD 1-for-3 as 0.33333),
# move a split-adjusted price by up to 0.2 bp: a daily price this close to a range of Databento prices lies in it
PRICE_ROUNDING_BP = 1.0
DATABENTO_CHANGE_SLACK = 5                # sessions each side of a ticker change asked under both tickers
DATABENTO_WORKERS = 6                     # its requests in flight at once: each prices or fetches for seconds
MINUTE_COLUMNS = ["open", "high", "low", "close", "volume"]
CME_DATASET = "GLBX.MDP3"                                     # Databento's CME Globex, from 2010-06-06
CME_FROM = "2010-06-06"
CME_DIR = DATABENTO_DIR / "futures"
# the CME futures list's contracts (user, 2026-09-26: futures beside the vendor's spot quotes, thin for platinum,
# palladium and crude, and copper, which it has no quote of): gold, silver, platinum, palladium, WTI crude, copper
CME_PRODUCTS = ("GC", "SI", "PL", "PA", "CL", "HG")
# what a contract holds, in the units its price is quoted in (CME's contract specifications): a bar's dollar volume is
# its contracts times that times the price, as a list's liquidity compares it with the other products'
CME_CONTRACT_SIZE = {"GC": 100, "SI": 5_000, "PL": 50, "PA": 100, "CL": 1_000, "HG": 25_000}
# how far before a roll the last hour both contracts printed is looked for: a roll at Sunday's open meets them on
# Friday, one after a holiday weekend on Thursday
CME_ROLL_LOOKBACK = pd.Timedelta(days=5)
CME_READ_TIMEOUT = 1200                                        # seconds the client waits for a stream to start
DUKASCOPY = "https://datafeed.dukascopy.com/datafeed"         # its public hourly bid candles of FX pairs
DUKASCOPY_FIRST_YEAR = 2003                                   # the FX majors' candles there begin in 2003
# where an instrument's Dukascopy candles become the market's: its silver before 2011 is not (in 2005 a 2.3% spread
# and a mid 1.7% under the vendor's daily close; the median gap of its closes to those: 2.6% in 2005, 1.3% in 2006,
# 0.7% in 2007-2008, 0.4-0.5% in 2009-2010, 0.1% from 2011 as gold's)
DUKASCOPY_FIRST_MONTH = {"XAG/USD": "2011-01"}
# its names of the metals and crude; an FX pair is named by its pair without the slash
DUKASCOPY_NAMES = {"XAU/USD": "XAUUSD", "XAG/USD": "XAGUSD", "XPT/USD": "XPTCMDUSD", "XPD/USD": "XPDCMDUSD",
                   "WTI/USD": "LIGHTCMDUSD"}
# a mid more than this far from the vendor's close at the median is another instrument or price basis (Dukascopy's
# crude, a CFD on the future: 47 bp from the vendor's spot)
DUKASCOPY_MATCH_BP = 3.0
DUKASCOPY_DIR = DATA_DIR / "raw" / "dukascopy"                # its monthly files as they arrive, kept to resume
# seconds after each request: at 1.5 it refused one month four times in a row within 20 minutes (2026-09-26)
DUKASCOPY_PAUSE = 5.0
# spans of an instrument's Dukascopy candles (their stamps, as UTC) that carry New York's wall clock written as UTC,
# four hours early in summer and five in winter: AUD/USD's moves line up with EUR/USD's hour by hour only once put on
# that clock (a return correlation of 0.35-0.73 in each month of both spans, 0.0-0.2 as stamped; 99 of their 106 weeks
# line up best at exactly New York's offset), and Forexite's minutes of September 2007 match them to 2 bp so, 8 bp and
# more otherwise. As stamped, their Friday afternoons looked missing. Every other series of theirs lines up as stamped
# (each month of 2003-2020 against EUR/USD, EUR/USD's against GBP/USD; checked 2026-09-27).
DUKASCOPY_NEW_YORK_CLOCK = {"AUD/USD": (("2007-04-01", "2008-09-21"), ("2009-04-05", "2009-09-20"))}
# Dukascopy's first months stamp their hours an hour late (checked 2026-09-28 against Forexite's minutes, whose clock the
# US payrolls confirm: their 12:30 UTC release is Forexite's widest minute, at 12:31, and falls in Dukascopy's hour that
# closes at 14:00 in June and July 2003): day by day, the four pairs it has from 2003-05 line up with Forexite an hour
# earlier on every day to 2003-07-25, and gold on every day to 2003-08-01; they are put an hour earlier. In the pairs'
# week of 2003-07-28 the hours are late and on time in turn within a day (USD/JPY's late to 05:00 and from 20:00 on the
# 29th, USD/CHF's from 16:00 on the 30th): not taken, so Forexite's minutes fill them (`forexite-fill`); gold's
# Forexite quote is 8-35 bp from Dukascopy's and cannot, and its days of that week are late throughout
DUKASCOPY_HOUR_LATE = {**{s: ("2003-05-01", "2003-07-27") for s in ("EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF")},
                       "XAU/USD": ("2003-05-01", "2003-08-02")}
DUKASCOPY_CLOCK_UNKNOWN = {s: ("2003-07-27", "2003-08-02") for s in ("EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF")}
US_TICK = 0.01                                               # a US listing's price step above $1
FOREXITE = "https://www.forexite.com/free_forex_quotes"      # its free one-minute bars of FX pairs, a file a day
FOREXITE_DIR = DATA_DIR / "raw" / "forexite"                  # its daily files as they arrive (an empty one: no file)
FOREXITE_ZONE = "Europe/Berlin"   # its clock: Central European, summer time included; a minute is stamped at its end
# a second source's closes more than this far from the stored ones at the median are another feed or clock: Forexite's
# are 1.2-1.3 bp from Dukascopy's over two weeks of November 2008 (its prices have four decimals), 2 bp over
# September 2007 with AUD/USD put on UTC
FOREXITE_MATCH_BP = 3.0
FOREXITE_PAUSE = 2.0                                          # seconds after each request
TD_API = "https://api.twelvedata.com/time_series"
TD_EARLIEST_API = "https://api.twelvedata.com/earliest_timestamp"
TD_FIRST_BARS = "twelvedata_first_bars.csv"            # data/reference: the vendor's first hourly bar per symbol
TF_DELTA = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}
TD_INTERVAL = {"1h": "1h", "1d": "1day"}
TD_PAGE = 5000
# The vendor's first bars (its earliest_timestamp, 2026-09-24): daily from 1970-01-02 (IBM, KO), hourly US equities
# from 2019-01-07 (AAPL), hourly FX, metals and crude from 2020-01 (EUR/USD, XAU/USD).
TD_HISTORY_START = {"1h": "2019-01-01", "1d": "1970-01-01", "fx_1h": "2020-01-01"}

# The vendor's hourly US-equity bars before 2020-06-29 are not where their labels say (matched against its daily and
# 15-minute bars of AAPL, MSFT and SPY): until the end of 2019 they cover 09:30-10:30 ... 15:30-16:00 New York, as today,
# but are labelled on the whole UTC hour before their start, and one hour later while Europe is on summer time; from
# 2020 to 2020-06-26 they cover clock hours (the first one 09:30-10:00) under true labels, and 2020-06-26 also carries
# a 15:30 bar off that grid. From 2020-06-29 bars sit on the half hour under true labels.
TD_HOURLY_CLOCK_HOURS_FROM = pd.Timestamp("2020-01-01")         # New York session dates
TD_HOURLY_HALF_HOURS_FROM = pd.Timestamp("2020-06-29")


class RateLimited(RuntimeError):
    """The vendor refused to answer; stop the run (a partial universe must never pass for a complete one)."""


class BudgetExceeded(RuntimeError):
    pass


class SymbolRejected(RuntimeError):
    """The vendor refuses this symbol (unknown or retired ticker, or data outside the plan such as another
    exchange's add-on): remembered in the series' meta and skipped; the rest of the run goes on."""


@dataclass
class Budget:
    vendor: str
    per_minute: int | None
    max_requests: int | None
    used: int = 0
    window: list = field(default_factory=list)

    def acquire(self) -> None:
        if self.max_requests is not None and self.used >= self.max_requests:
            raise BudgetExceeded(f"{self.vendor}: run ceiling of {self.max_requests} requests reached")
        if self.per_minute:
            now = time.monotonic()
            self.window = [t for t in self.window if now - t < 60.0]
            if len(self.window) >= self.per_minute:
                wait = 60.0 - (now - self.window[0]) + 0.1
                LOG.info("%s: %d requests in the last minute, waiting %.1fs", self.vendor, len(self.window), wait)
                time.sleep(wait)
            self.window.append(time.monotonic())
        self.used += 1


def _ledger(vendor: str, **entry) -> None:
    p = LOGS_DIR / "requests" / f"{vendor}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    entry["at"] = datetime.now(timezone.utc).isoformat()
    with p.open("a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def _append(source: str, tf: str, symbol: str, new: pd.DataFrame) -> int:
    try:
        old = store.read_bars(source, tf, symbol)
    except FileNotFoundError:
        old = new.iloc[:0]
    new = new[new.index > old.index.max()] if len(old) else new
    if new.empty:
        return 0
    merged, bad = integrity.clean_bars(pd.concat([old, new]))
    if bad:
        LOG.warning("%s:%s %s: dropped invalid bars on append %s", source, symbol, tf, bad)
    store.write_bars(source, tf, symbol, merged)
    return len(new)


# ---------------------------------------------------------------- Binance
def _binance_get(session, url: str, params: dict, budget: Budget, what: dict) -> list:
    budget.acquire()
    r = session.get(url, params=params, timeout=30)
    used = r.headers.get("x-mbx-used-weight-1m")
    _ledger("binance", url=url.split("/")[-1], status=r.status_code, used_weight_1m=used, **what)
    if r.status_code in (418, 429):
        raise RateLimited(f"binance answered {r.status_code}: stop")
    if r.status_code == 400:
        try:
            body = r.json()
        except ValueError:
            body = {}
        if isinstance(body, dict) and body.get("code") == -1121:     # a contract the exchange no longer serves
            raise SymbolRejected(f"binance: {body.get('msg')}")
    r.raise_for_status()
    if used is not None and int(used) > 1800:            # futures limit is 2400/min: breathe before hitting it
        time.sleep(61 - datetime.now(timezone.utc).second)
    return r.json()


def save_perp_contracts(info: dict) -> None:
    """Keep what each USD-M contract is: Binance lists stock, commodity and FX perps next to crypto ones."""
    from strategy_lab.config import REFERENCE_DIR
    rows = [{"symbol": s["symbol"], "underlying_type": s.get("underlyingType"), "contract_type": s.get("contractType"),
             "status": s.get("status"), "onboard_date": s.get("onboardDate"), "delivery_date": s.get("deliveryDate")}
            for s in info["symbols"]]
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REFERENCE_DIR / "binance_perp_contracts.csv", index=False)


def binance_trading(session, market: str, budget: Budget) -> set[str]:
    url = f"{FAPI}/fapi/v1/exchangeInfo" if market == "perp" else f"{SPOT_API}/api/v3/exchangeInfo"
    info = _binance_get(session, url, {}, budget, {"market": market})
    if market == "perp":
        save_perp_contracts(info)
    return {s["symbol"] for s in info["symbols"] if s.get("status") == "TRADING"}


def _binance_plan(market: str, symbols: list[str], tfs: list[str], now: pd.Timestamp) -> list[tuple[str, str, int]]:
    plan = []
    limit = 1500 if market == "perp" else 1000
    for s in symbols:
        for tf in tfs:
            if not store.path(market, tf, s).exists():
                continue
            last = store.read_bars(market, tf, s).index.max()
            missing = int((now - last) / TF_DELTA[tf])
            if missing >= 1:
                plan.append((s, tf, math.ceil(missing / limit)))
    return plan


def _klines_frame(rows: list, tf: str) -> pd.DataFrame:
    """Binance kline rows as store bars, stamped at their close (open time + interval)."""
    df = pd.DataFrame(rows).iloc[:, :8]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_ms", "quote_volume"]
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True) + TF_DELTA[tf]
    df = df.astype({c: float for c in ("open", "high", "low", "close", "volume", "quote_volume")})
    df["dollar_volume"] = df["quote_volume"]
    return df[store.BAR_COLUMNS]


def missing_runs(index: pd.DatetimeIndex, tf: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Runs of consecutive bar closes missing from a 24/7 series between its first and last bar."""
    if len(index) < 2:
        return []
    step = TF_DELTA[tf]
    runs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for t in pd.date_range(index.min(), index.max(), freq=step).difference(index):
        if runs and t - runs[-1][1] == step:
            runs[-1] = (runs[-1][0], t)
        else:
            runs.append((t, t))
    return runs


def fill_binance_gaps(markets=("perp",), tfs=("1h", "4h", "1d"), dry_run: bool = True, max_requests: int | None = None,
                      symbols: list[str] | None = None) -> dict:
    """Fill interior gaps of stored Binance series from the exchange's own klines (the seed's monthly archives miss a
    few days). A run the exchange has no bars for (a halted or relisted market) is recorded in the series' meta as
    `exchange_has_no_bars` and never asked again."""
    now_ms = pd.Timestamp.now(tz="UTC").timestamp() * 1000
    session = requests.Session()
    budget = Budget("binance", per_minute=None, max_requests=max_requests)
    report = {}
    for market in markets:
        url = f"{FAPI}/fapi/v1/klines" if market == "perp" else f"{SPOT_API}/api/v3/klines"
        limit = 1500 if market == "perp" else 1000
        plan = []
        for tf in tfs:
            for sym in store.symbols(market, tf):
                if symbols and sym not in symbols or store.read_meta(market, tf, sym).get("vendor_rejected"):
                    continue
                known = {tuple(x) for x in store.read_meta(market, tf, sym).get("exchange_has_no_bars", [])}
                runs = [r for r in missing_runs(store.read_bars(market, tf, sym).index, tf)
                        if (str(r[0]), str(r[1])) not in known]
                if runs:
                    plan.append((sym, tf, runs))
        n_req = sum(math.ceil(((hi - lo) / TF_DELTA[tf] + 1) / limit) for _, tf, runs in plan for lo, hi in runs)
        report[market] = {"series_with_gaps": len(plan), "gaps": sum(len(r) for *_, r in plan),
                          "planned_requests": n_req}
        LOG.info("binance %s gaps: %s", market, report[market])
        if dry_run:
            continue
        filled = 0
        for sym, tf, runs in plan:
            rows, empty = [], []
            try:
                for lo, hi in runs:
                    start_ms = int((lo - TF_DELTA[tf]).timestamp() * 1000)      # open time of the first missing bar
                    end_ms = int((hi - TF_DELTA[tf]).timestamp() * 1000)        # open time of the last one
                    got = []
                    while start_ms <= end_ms:
                        part = _binance_get(session, url, {"symbol": sym, "interval": tf, "startTime": start_ms,
                                                           "endTime": end_ms, "limit": limit}, budget,
                                            {"symbol": sym, "tf": tf, "what": "gap"})
                        if not part:
                            break
                        got.extend(part)
                        if len(part) < limit:
                            break
                        start_ms = part[-1][0] + 1
                    got = [r for r in got if r[6] < now_ms]
                    if got:
                        rows.extend(got)
                    else:
                        empty.append([str(lo), str(hi)])
            except SymbolRejected as e:
                LOG.warning("%s:%s %s: the exchange refuses the symbol (%s): remembered, its gaps not asked again",
                            market, sym, tf, e)
                store.write_meta(market, tf, sym, {**store.read_meta(market, tf, sym),
                                                   "vendor_rejected": {"at": str(pd.Timestamp.now(tz="UTC")),
                                                                       "message": str(e)[:300]}})
                continue
            if rows:
                old = store.read_bars(market, tf, sym)
                add = _klines_frame(rows, tf)
                merged, bad = integrity.clean_bars(pd.concat([old, add[~add.index.isin(old.index)]]).sort_index())
                if bad:
                    LOG.warning("%s:%s %s: dropped invalid bars while filling gaps %s", market, sym, tf, bad)
                store.write_bars(market, tf, sym, merged)
                filled += int((~add.index.isin(old.index)).sum())
            if empty:
                meta = store.read_meta(market, tf, sym)
                meta["exchange_has_no_bars"] = meta.get("exchange_has_no_bars", []) + empty
                store.write_meta(market, tf, sym, meta)
        report[market].update({"bars_filled": filled, "requests_made": budget.used})
    return report


def _funding(session, s: str, budget: Budget) -> int:
    """Bring a perp's funding settlements up to date, paging forward from the last stored one: a backtest refuses a
    perp without them. With none stored, or stored from later than the perp's first daily bar, the settlements are
    asked from that bar (asked from time 0, the exchange answers with recent ones only). Returns the settlements
    added."""
    p = store.STORE_DIR / "perp" / "funding" / f"{store.safe_name(s)}.parquet"
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["rate"])
    listed = (int((store.read_bars("perp", "1d", s).index.min() - TF_DELTA["1d"]).timestamp() * 1000)
              if store.path("perp", "1d", s).exists() else int(BINANCE_OPENED.timestamp() * 1000))
    spans = [(int(old.index.max().timestamp() * 1000) + 1, None)] if len(old) else [(listed, None)]
    if len(old) and old.index.min().timestamp() * 1000 - listed > TF_DELTA["1d"].total_seconds() * 1000:
        spans.insert(0, (listed, int(old.index.min().timestamp() * 1000) - 1))      # the history before what is kept
    frames = []
    for since, until in spans:
        while True:
            ask = {"symbol": s, "startTime": since, "limit": 1000} | ({"endTime": until} if until else {})
            chunk = _binance_get(session, f"{FAPI}/fapi/v1/fundingRate", ask, budget, {"symbol": s, "what": "funding"})
            if not chunk:
                break
            f = pd.DataFrame(chunk)
            f.index = pd.to_datetime(f["fundingTime"], unit="ms", utc=True)
            frames.append(f[["fundingRate"]].astype(float).rename(columns={"fundingRate": "rate"}))
            if len(chunk) < 1000:
                break
            since = int(chunk[-1]["fundingTime"]) + 1
    if not frames:
        if old.empty:
            LOG.warning("perp:%s: the exchange has no funding settlements for it", s)
        return 0
    out = pd.concat([old, *frames] if len(old) else frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "settle_time"
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p)
    return len(out) - len(old)


DELISTED = {"SETTLING", "CLOSE"}         # Binance's status of a perp it no longer trades


def _contracts() -> pd.DataFrame:
    from strategy_lab.config import REFERENCE_DIR
    p = REFERENCE_DIR / "binance_perp_contracts.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `refresh binance-delistings` (one request to the exchange)")
    return pd.read_csv(p).set_index("symbol")


def perp_delistings(symbols) -> dict[str, pd.Timestamp | None]:
    """When each delisted perp among `symbols` stopped trading, from the exchange's contract list: a contract Binance
    lists as SETTLING or CLOSE stopped at its deliveryDate (a perpetual it trades carries a placeholder date in
    2100). A symbol no longer in the list was delisted before the list was saved and has no date there: None, its
    last trade then tells. Perps still trading are left out."""
    c = _contracts()
    if "delivery_date" not in c:
        raise RuntimeError("the saved contract list has no delivery dates: run `refresh binance-delistings`")
    out = {}
    for sym in symbols:
        if sym not in c.index:
            out[sym] = None
        elif c.at[sym, "status"] in DELISTED:
            out[sym] = pd.Timestamp(int(c.at[sym, "delivery_date"]), unit="ms", tz="UTC")
    return out


def cut_at_delisting(bars: pd.DataFrame, tf: str, delisted_at: pd.Timestamp | None) -> pd.DataFrame:
    """A delisted perp's bars up to its delisting: those that opened before the exchange's delivery time or, with no
    such time (a contract no longer listed), up to its last bar with trades. After it the exchange goes on printing
    its last price with no volume: not a market."""
    if delisted_at is not None:
        return bars[bars.index - TF_DELTA[tf] < delisted_at]
    traded = bars.index[bars["volume"].fillna(0) > 0]
    return bars[bars.index <= traded.max()] if len(traded) else bars.iloc[:0]


NO_TRADES = pd.Timedelta(days=3)


def delisted_between(bars: pd.DataFrame, tf: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Stretches inside a perp's series with no trades for more than three days that the exchange printed as such
    (its last price, no volume, on at least half of the stretch's bars) and after which it traded again: the contract
    was delisted and later listed anew under the same name (TLM, ICP, CVC). The exchange lists only the current
    contract, so these bars are the only record of the old one's end. A stretch with no bars at all is a hole in the
    data, not a delisting. Each is (the last bar with trades before, the first after)."""
    traded = bars.index[bars["volume"].fillna(0) > 0]
    out = []
    for k in np.flatnonzero((traded[1:] - traded[:-1]) > NO_TRADES):
        lo, hi = traded[k], traded[k + 1]
        printed = int(((bars.index > lo) & (bars.index < hi)).sum())
        if printed >= 0.5 * ((hi - lo) / TF_DELTA[tf] - 1):
            out.append((lo, hi))
    return out


def mark_delistings(tfs=("1h", "4h", "1d"), dry_run: bool = True) -> dict:
    """Save the exchange's contract list (one request) and end every stored series of a delisted perp at its delisting
    (`cut_at_delisting`), with the date in the series' meta, where a backtest closes the position and a list gives
    the seat to the next name (`engine.backtest.ended`)."""
    session = requests.Session()
    info = _binance_get(session, f"{FAPI}/fapi/v1/exchangeInfo", {}, Budget("binance", None, None),
                        {"market": "perp", "what": "contracts"})
    if not dry_run:
        save_perp_contracts(info)
    listed = {s["symbol"]: s for s in info["symbols"]}
    report = {}
    for tf in tfs:
        cut, dropped = 0, 0
        for sym in store.symbols("perp", tf):
            s = listed.get(sym)
            bars = store.read_bars("perp", tf, sym)
            meta = store.read_meta("perp", tf, sym)
            gaps = delisted_between(bars, tf)
            if gaps:                                          # delisted and listed again: the stretch is no market
                cut, dropped = cut + 1, dropped + sum(len(bars.loc[lo:hi]) - 2 for lo, hi in gaps)
                LOG.info("perp:%s %s: delisted and listed again %s", sym, tf,
                         ", ".join(f"{lo}..{hi}" for lo, hi in gaps))
                if not dry_run:
                    keep = pd.Series(True, index=bars.index)
                    for lo, hi in gaps:
                        keep[(bars.index > lo) & (bars.index < hi)] = False
                    bars = bars[keep.to_numpy()]
                    meta = {**meta, "delisted_periods": [[str(lo), str(hi)] for lo, hi in gaps]}
                    store.write_bars("perp", tf, sym, bars)
                    store.write_meta("perp", tf, sym, meta)
            if s is not None and s.get("status") not in DELISTED:
                continue
            at = pd.Timestamp(int(s["deliveryDate"]), unit="ms", tz="UTC") if s is not None else None
            kept = cut_at_delisting(bars, tf, at)
            if len(kept) == len(bars) and meta.get("delisted"):
                continue
            cut, dropped = cut + 1, dropped + len(bars) - len(kept)
            if len(bars) > len(kept):
                LOG.info("perp:%s %s: %d bars after its delisting (%s) dropped", sym, tf, len(bars) - len(kept),
                         at or "last trade")
            if not dry_run and len(kept):
                store.write_bars("perp", tf, sym, kept)
                store.write_meta("perp", tf, sym, {**meta, "last": str(kept.index.max()),
                                                   "delisted": str(at or kept.index.max()),
                                                   "delisted_from": "the exchange's delivery date" if at is not None
                                                   else "its last trade: no longer in the exchange's contract list"})
            elif not len(kept):
                LOG.warning("perp:%s %s: no bar before its delisting (%s): the series is left as it is", sym, tf, at)
        report[tf] = {"delisted_series": cut, "bars_dropped": dropped}
    LOG.info("binance delistings: %s", report)
    return report


def _funding_short(s: str) -> bool:
    """No funding stored for the perp, or stored only from later than its first daily bar."""
    p = store.STORE_DIR / "perp" / "funding" / f"{store.safe_name(s)}.parquet"
    if not p.exists():
        return True
    if not store.path("perp", "1d", s).exists():
        return False
    return pd.read_parquet(p).index.min() - store.read_bars("perp", "1d", s).index.min() > 2 * TF_DELTA["1d"]


def seed_binance_series(symbols: list[str], tfs=("1h", "4h", "1d"), market: str = "perp", dry_run: bool = True,
                        max_requests: int | None = None) -> dict:
    """Whole history of Binance series the store does not have yet (new coins), from the pair's first bar: the stored
    daily series' first bar, or the exchange's opening for a coin with no daily series (klines asked from a start
    time begin at the pair's listing). Klines page forward; the bar still forming is dropped. A perp also gets its
    funding settlements from its listing, whenever none are stored, and a delisted one ends at its delisting
    (`cut_at_delisting`, from the exchange's contract list, saved on the way)."""
    now_ms = pd.Timestamp.now(tz="UTC").timestamp() * 1000
    url = f"{FAPI}/fapi/v1/klines" if market == "perp" else f"{SPOT_API}/api/v3/klines"
    limit = 1500 if market == "perp" else 1000
    todo = []
    for sym in symbols:
        if store.path(market, "1d", sym).exists():
            first = store.read_bars(market, "1d", sym).index.min() - TF_DELTA["1d"]   # open of the first daily bar
        else:
            first = BINANCE_OPENED
            LOG.info("%s:%s has no daily series: asked from the exchange's opening, %s", market, sym, first.date())
        for tf in tfs:
            if store.read_meta(market, tf, sym).get("vendor_rejected"):
                continue
            if not store.path(market, tf, sym).exists():
                n = int((pd.Timestamp.now(tz="UTC") - first) / TF_DELTA[tf])
                todo.append((sym, tf, first, math.ceil(n / limit)))
    no_funding = [s for s in symbols if market == "perp" and _funding_short(s)]
    report = {"series": len(todo), "planned_requests": sum(t[3] for t in todo), "funding_to_seed": len(no_funding)}
    LOG.info("binance %s new series: %s", market, report)
    if dry_run:
        return report
    session = requests.Session()
    budget = Budget("binance", per_minute=None, max_requests=max_requests)
    written, rejected = 0, []
    delisted = {}
    if market == "perp" and todo:                       # which of them the exchange has delisted, and since when
        save_perp_contracts(_binance_get(session, f"{FAPI}/fapi/v1/exchangeInfo", {}, budget,
                                         {"market": "perp", "what": "contracts"}))
        delisted = perp_delistings(symbols)
    for sym, tf, first, _ in todo:
        start_ms, rows = int(first.timestamp() * 1000), []
        try:
            while True:
                part = _binance_get(session, url, {"symbol": sym, "interval": tf, "startTime": start_ms,
                                                   "limit": limit}, budget, {"symbol": sym, "tf": tf, "what": "seed"})
                if not part:
                    break
                rows.extend(part)
                if len(part) < limit:
                    break
                start_ms = part[-1][0] + 1
        except SymbolRejected as e:
            LOG.warning("%s:%s %s: the exchange refuses the symbol (%s): remembered, not asked again", market, sym, tf, e)
            store.write_meta(market, tf, sym, {"vendor_rejected": {"at": str(pd.Timestamp.now(tz="UTC")),
                                                                   "message": str(e)[:300]}})
            rejected.append(f"{sym}:{tf}")
            continue
        rows = [r for r in rows if r[6] < now_ms]
        if not rows:
            LOG.warning("%s:%s %s: the exchange returned no bars", market, sym, tf)
            continue
        bars, bad = integrity.clean_bars(_klines_frame(rows, tf))
        if market == "perp" and sym in delisted:
            bars = cut_at_delisting(bars, tf, delisted[sym])
        store.write_bars(market, tf, sym, bars)
        store.write_meta(market, tf, sym, {"seeded_from": "binance_api", "first": str(bars.index.min()),
                                           "last": str(bars.index.max())})
        written += 1
    for sym in no_funding:
        try:
            _funding(session, sym, budget)
        except SymbolRejected as e:
            LOG.warning("perp:%s: the exchange refuses the symbol (%s): no funding stored", sym, e)
    report.update({"series_written": written, "rejected": rejected, "requests_made": budget.used})
    return report


def extend_binance(tfs=("1h", "4h", "1d"), dry_run: bool = True, max_requests: int | None = None) -> dict:
    """History before a perp series' first stored bar, from the exchange's klines. The store was seeded from the
    exchange's monthly archives, whose perps begin in 2020-01, while the perps listed in 2019 (BTC, ETH, BCH) trade from
    then: only a series that starts in the archive's first month can have earlier bars, and klines asked from the
    exchange's opening begin at the perp's listing. Their funding settlements before the first kept one follow."""
    todo = []
    for sym in store.symbols("perp", "1d"):
        for tf in tfs:
            if not store.path("perp", tf, sym).exists():
                continue
            first = store.read_bars("perp", tf, sym, ["close"]).index.min() - TF_DELTA[tf]      # its first bar's open
            if first < PERP_ARCHIVE_START + TF_DELTA["1d"]:
                todo.append((sym, tf, first, math.ceil((first - BINANCE_OPENED) / TF_DELTA[tf] / 1500)))
    perps = sorted({t[0] for t in todo})
    report = {"series": len(todo), "perps": perps,
              "planned_requests_at_most": sum(t[3] for t in todo) + len(perps)}
    LOG.info("binance perp history before the archive: %s", report)
    if dry_run:
        return report
    session = requests.Session()
    budget = Budget("binance", per_minute=None, max_requests=max_requests)
    added = {}
    for sym, tf, first, _ in todo:
        start_ms, end_ms, rows = int(BINANCE_OPENED.timestamp() * 1000), int(first.timestamp() * 1000) - 1, []
        try:
            while True:
                part = _binance_get(session, f"{FAPI}/fapi/v1/klines", {"symbol": sym, "interval": tf,
                                                                        "startTime": start_ms, "endTime": end_ms,
                                                                        "limit": 1500},
                                    budget, {"symbol": sym, "tf": tf, "what": "extend"})
                rows.extend(part)
                if len(part) < 1500:
                    break
                start_ms = part[-1][0] + 1
        except SymbolRejected as e:
            LOG.warning("perp:%s %s: the exchange refuses the symbol (%s): its history is left as it is", sym, tf, e)
            continue
        old = store.read_bars("perp", tf, sym)
        earlier = _klines_frame(rows, tf) if rows else old.iloc[:0]
        earlier = earlier[earlier.index < old.index.min()]
        if earlier.empty:
            LOG.info("perp:%s %s: the exchange has no bars before %s", sym, tf, first)
            continue
        merged, bad = integrity.clean_bars(pd.concat([earlier, old]))
        if bad:
            LOG.warning("perp:%s %s: dropped invalid bars on extending %s", sym, tf, bad)
        store.write_bars("perp", tf, sym, merged)
        store.write_meta("perp", tf, sym, {**store.read_meta("perp", tf, sym), "first": str(merged.index.min()),
                                           "extended_from": "binance_api"})
        added[f"{sym}:{tf}"] = f"{len(earlier)} bars from {earlier.index.min()}"
    for sym in perps:
        _funding(session, sym, budget)
    report.update({"bars_added": added, "requests_made": budget.used})
    return report


def archive_klines(zipped: bytes, tf: str) -> pd.DataFrame:
    """One monthly archive file of the exchange as store bars stamped at their close. Its CSV opens with a header row
    in the newer files, and its spot files give times in microseconds from 2025 on, milliseconds before."""
    with zipfile.ZipFile(io.BytesIO(zipped)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]), header=None, dtype=str)
    if not raw.iat[0, 0].strip().isdigit():
        raw = raw.iloc[1:]
    t = raw[0].astype("int64").to_numpy()
    unit = "us" if t.max() > 10**14 else "ms"
    df = pd.DataFrame({c: raw[i].astype(float).to_numpy() for i, c in
                       ((1, "open"), (2, "high"), (3, "low"), (4, "close"), (5, "volume"), (7, "dollar_volume"))})
    df.index = pd.to_datetime(t, unit=unit, utc=True) + TF_DELTA[tf]
    return df[store.BAR_COLUMNS]


def seed_binance_from_archive(symbols: list[str], tfs=("1h", "4h", "1d"), market: str = "perp",
                              dry_run: bool = True, max_requests: int | None = None) -> dict:
    """Series of pairs the exchange no longer serves (its klines answer "Invalid symbol" for a delisted pair), from its
    public monthly archives, for the series the store does not have yet; a perp's end at its last bar with trades
    (`cut_at_delisting`). `binance-delistings` then marks a delisted perp's new series as it marks the others. One
    request lists a series' months, one fetches each month."""
    todo, months = [], 0
    for s in symbols:
        daily = store.read_bars(market, "1d", s, ["close"]).index if store.path(market, "1d", s).exists() else None
        for tf in tfs:
            if store.path(market, tf, s).exists():
                continue
            todo.append((s, tf))
            months += len(pd.period_range(daily.min().tz_localize(None), daily.max().tz_localize(None), freq="M")) \
                if daily is not None and len(daily) else 0
    report = {"series": len(todo), "planned_requests": len(todo) + months}
    LOG.info("binance %s series from the archive: %s", market, report)
    if dry_run:
        return report
    session = requests.Session()
    budget = Budget("binance", per_minute=None, max_requests=max_requests)
    written, empty = [], []
    for s, tf in todo:
        budget.acquire()
        listing = session.get(ARCHIVE_LIST, params={"prefix": f"{ARCHIVE_KLINES[market]}/{s}/{tf}/"}, timeout=60)
        keys = re.findall(r"<Key>([^<]+\.zip)</Key>", listing.text)
        _ledger("binance", url="archive-list", status=listing.status_code, symbol=s, tf=tf, files=len(keys))
        frames = []
        for k in keys:
            budget.acquire()
            r = session.get(f"{ARCHIVE}/{k}", timeout=120)
            _ledger("binance", url="archive", status=r.status_code, symbol=s, tf=tf, file=k.rsplit("/", 1)[-1])
            if r.status_code != 200:
                LOG.warning("%s:%s %s: the archive answered %s for %s: that month is missing", market, s, tf,
                            r.status_code, k)
                continue
            frames.append(archive_klines(r.content, tf))
        if not frames:
            LOG.warning("%s:%s %s: the archive has no files for it", market, s, tf)
            empty.append(f"{s}:{tf}")
            continue
        bars = pd.concat(frames)
        bars, bad = integrity.clean_bars(bars[~bars.index.duplicated(keep="last")].sort_index())
        if bad:
            LOG.warning("%s:%s %s: dropped invalid archive bars %s", market, s, tf, bad)
        if market == "perp":
            bars = cut_at_delisting(bars, tf, None)
        store.write_bars(market, tf, s, bars)
        store.write_meta(market, tf, s, {"seeded_from": "binance_archive", "first": str(bars.index.min()),
                                         "last": str(bars.index.max())})
        written.append(f"{s}:{tf} {bars.index.min().date()}..{bars.index.max().date()} ({len(bars)} bars)")
    report.update({"series_written": written, "no_files": empty, "requests_made": budget.used})
    return report


def _sharadar_get(session, path: str, params: dict, budget: Budget, what: dict) -> requests.Response:
    """One request to Sharadar, logged in the ledger with the requests it has left today; 429 stops the run."""
    key = env("SHARADAR_API_KEY")
    if not key:
        raise RuntimeError("SHARADAR_API_KEY is not set in the environment or .env")
    budget.acquire()
    r = session.get(f"{SHARADAR}{path}", params={**params, "api_key": key}, timeout=600)
    _ledger("sharadar", status=r.status_code, remaining=r.headers.get("X-RateLimit-Remaining"), **what)
    if r.status_code == 429:
        raise RateLimited("sharadar answered 429: stop")
    if r.status_code != 200:
        raise RuntimeError(f"sharadar {r.status_code} for {path} {what}: {r.text[:200]}")
    return r


def refresh_sharadar_tables(dry_run: bool = True) -> dict:
    """Sharadar's whole sp500, tickers and actions tables, one bulk download each (data/raw/sharadar/<table>.zip):
    the S&P 500's membership as Sharadar keys it, every security it covers with its permaticker, and the corporate
    actions (dividends and splits on the split-adjusted basis of its prices, delistings, ticker changes)."""
    report: dict = {"planned_requests": len(SHARADAR_TABLES)}
    if dry_run:
        return report
    session, budget = requests.Session(), Budget("sharadar", per_minute=None, max_requests=len(SHARADAR_TABLES))
    for table in SHARADAR_TABLES:
        r = _sharadar_get(session, f"/data/{table}", {"years": "full"}, budget, {"what": table})
        SHARADAR_DIR.mkdir(parents=True, exist_ok=True)
        (SHARADAR_DIR / f"{table}.zip").write_bytes(r.content)
        report[table] = len(r.content)
    return report


def fetch_sharadar_stocks(tickers: list[str], dry_run: bool = True, max_requests: int | None = None) -> dict:
    """Daily bars of Sharadar tickers since its first (1997-12-31), one request a ticker, each kept as it arrives
    (data/raw/sharadar/stocks/<ticker>.csv; an empty file for a ticker it has none of), so a stopped run resumes
    and a ticker is never asked twice."""
    folder = SHARADAR_DIR / "stocks"
    todo = [t for t in dict.fromkeys(tickers) if not (folder / f"{store.safe_name(t)}.csv").exists()]
    report: dict = {"tickers": len(todo), "planned_requests": len(todo)}
    LOG.info("sharadar stocks: %s", report)
    if dry_run:
        return report
    session, budget = requests.Session(), Budget("sharadar", per_minute=None, max_requests=max_requests)
    folder.mkdir(parents=True, exist_ok=True)
    empty = []
    for t in todo:
        r = _sharadar_get(session, "/data/stocks", {"ticker": t, "from": SHARADAR_FROM, "format": "csv",
                                                     "limit": 10000}, budget, {"what": "stocks", "ticker": t})
        rows = r.text.count("\n") - 1
        if rows >= 10000:
            raise RuntimeError(f"sharadar stocks {t}: {rows} rows, the page limit: its history would be cut")
        if rows <= 0:
            LOG.warning("sharadar stocks %s: no bars", t)
            empty.append(t)
        (folder / f"{store.safe_name(t)}.csv").write_text(r.text if rows > 0 else "")
    report.update({"no_bars": empty, "requests_made": budget.used})
    return report


def fetch_sharadar_funds(tickers: list[str], dry_run: bool = True, max_requests: int | None = None) -> dict:
    """Sharadar's daily bars of ETFs (its funds table) since its first, one request a ticker, each kept as it arrives
    (data/raw/sharadar/funds/<ticker>.csv; an empty file for a ticker it has none of), as `fetch_sharadar_stocks`
    keeps the stocks'."""
    folder = SHARADAR_DIR / "funds"
    todo = [t for t in dict.fromkeys(tickers) if not (folder / f"{store.safe_name(t)}.csv").exists()]
    report: dict = {"tickers": len(todo), "planned_requests": len(todo)}
    LOG.info("sharadar funds: %s", report)
    if dry_run:
        return report
    session, budget = requests.Session(), Budget("sharadar", per_minute=None, max_requests=max_requests)
    folder.mkdir(parents=True, exist_ok=True)
    empty = []
    for t in todo:
        r = _sharadar_get(session, "/data/funds", {"ticker": t, "from": SHARADAR_FROM, "format": "csv",
                                                    "limit": 10000}, budget, {"what": "funds", "ticker": t})
        rows = r.text.count("\n") - 1
        if rows >= 10000:
            raise RuntimeError(f"sharadar funds {t}: {rows} rows, the page limit: its history would be cut")
        if rows <= 0:
            LOG.warning("sharadar funds %s: no bars", t)
            empty.append(t)
        (folder / f"{store.safe_name(t)}.csv").write_text(r.text if rows > 0 else "")
    report.update({"no_bars": empty, "requests_made": budget.used})
    return report


def etf_daily_reconciled(tickers: list[str]) -> dict:
    """ETFs' daily bars from Sharadar's funds table (`fetch_sharadar_funds`), kept under their Twelve Data ids, except
    a day its bar contradicts the venues' own regular session, whose bar is then that session's: the hourly bars
    Databento's minutes of the lit venues make, closing auction included (`build_databento_hourly`), put on the
    split-adjusted basis, with their median close beyond Sharadar's day range by more than a price step (the test
    `conform_to_daily` drops a session by). Sharadar's SHY, IEF and LQD sit 5-40 bp under every venue from 2020-07-01
    to 07-24, where Twelve Data's hourly bars match the venues; Twelve Data's own daily bars are not the market's in
    2008 (DBC's and EEM's a single price a day with no volume all year; the sector ETFs' closes 50-140 bp off on 20-40
    days while open, high and low agree) and begin DBC and EEM in 2008, where Sharadar has them from their launch, so
    Sharadar's are the base. The stored bars before Sharadar's first day stay (SPY's of 1993-1997, Twelve Data's). A
    session Sharadar's day contradicts is dropped from the hourly bars, so `refresh databento-hourly` buys its minutes;
    the Twelve Data refresh leaves these daily series alone (`daily_from` in their meta)."""
    out = {}
    for t in tickers:
        path = SHARADAR_DIR / "funds" / f"{store.safe_name(t)}.csv"
        text = path.read_text() if path.exists() else ""
        if not text or not store.path("td", "1d", t).exists():
            LOG.warning("td:%s: no Sharadar fund bars on disk (`refresh etf-daily`) or no stored daily series: its "
                        "daily bars stay as they are", t)
            continue
        sh, old = sharadar_bars(text), store.read_bars("td", "1d", t)
        bars = pd.concat([old[old.index < sh.index.min()], sh]).astype({"volume": float})
        taken: list[str] = []
        layer = _databento_hourly_path("td", t)
        if layer.exists():
            got = pd.read_parquet(layer)
            days = _session_days(got.index)
            f = sharadar.split_factor(t, days)
            got = got.assign(**{c: got[c] * f for c in ("open", "high", "low", "close")})
            g = got.groupby(np.asarray(days))
            session = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                                    "close": g["close"].last(), "median": g["close"].median()})
            b_days = _session_days(bars.index)
            day = bars.set_axis(b_days)
            both = session.index.intersection(day.index)
            step = np.where(day.loc[both, "high"].to_numpy() >= 1.0, US_TICK, 1e-4)
            med = session.loc[both, "median"].to_numpy()
            off = (med < day.loc[both, "low"].to_numpy() - step) | (med > day.loc[both, "high"].to_numpy() + step)
            for d in both[off]:
                at = np.asarray(b_days == d)
                bars.loc[at, ["open", "high", "low", "close"]] = session.loc[d, ["open", "high", "low", "close"]].to_numpy()
                taken.append(str(d.date()))
        bars["dollar_volume"] = bars["close"] * bars["volume"]
        merged, bad = integrity.clean_bars(bars[store.BAR_COLUMNS])
        if bad:
            LOG.warning("td:%s 1d: dropped invalid bars on taking Sharadar's %s", t, bad)
        store.write_bars("td", "1d", t, merged)
        store.write_meta("td", "1d", t, {**store.read_meta("td", "1d", t), "daily_from": "sharadar",
                                         "sharadar_from": str(sh.index.min()), "days_from_the_venues": taken,
                                         "first": str(merged.index.min()), "last": str(merged.index.max())})
        if taken:
            LOG.warning("td:%s 1d: %d days Sharadar's bar contradicts the venues' session taken from the session (%s)",
                        t, len(taken), ", ".join(taken[:5]))
        _hourly_written(t)
        out[t] = {"sharadar_from": str(sh.index.min().date()), "days_from_the_venues": taken}
    return out


def sharadar_bars(text: str) -> pd.DataFrame:
    """Sharadar's daily rows of one ticker as store bars: its split-adjusted open, high, low, close and volume,
    stamped at the NYSE close of their session (a date that was no session is dropped)."""
    df = pd.read_csv(io.StringIO(text), parse_dates=["date"]).sort_values("date")
    close = cal.equity_daily_close(pd.DatetimeIndex(df["date"]))
    df.index = pd.DatetimeIndex(close.to_numpy())
    df = df[df.index.notna()]
    df["dollar_volume"] = df["close"] * df["volume"]
    return df[store.BAR_COLUMNS]


# Sharadar's split rows a day off, reviewed event by event against its own unadjusted closes and Twelve Data's bars:
# the row of the day before the split took effect, left on the unadjusted basis (a fake +25% then -20%)
SHARADAR_SPLIT_DAY_LATE = {("BF.B", "2018-02-28"): (1.25, "its 5:4 split took effect on 2018-03-01: the unadjusted "
                                                        "close fell 69.79 -> 55.37 then and moved 69.87 -> 69.79 on "
                                                        "02-28; Twelve Data's close of 02-28 is 55.83")}


def _split_rows(t: str, bars: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The reviewed split rows of SHARADAR_SPLIT_DAY_LATE put on the adjusted basis (prices over the split's ratio,
    volume times it); any other split of the ticker whose unadjusted close moves by its ratio on the next day instead
    of its own is named for the same review, not changed."""
    out, fixed = bars, []
    days = _session_days(bars.index)
    for (ticker, day), (ratio, _) in SHARADAR_SPLIT_DAY_LATE.items():
        at = np.asarray(days == pd.Timestamp(day))
        if ticker == t and at.any():
            out = out.astype({"volume": float})
            out.loc[at, ["open", "high", "low", "close"]] /= ratio
            out.loc[at, "volume"] *= ratio
            out.loc[at, "dollar_volume"] = out.loc[at, "close"] * out.loc[at, "volume"]
            fixed.append(day)
    a = sharadar.actions()
    splits = a[(a["action"] == "split") & (a["ticker"] == t)]
    u = raw.set_index(pd.to_datetime(raw["date"]))["closeunadj"].sort_index()
    for d, v in zip(pd.to_datetime(splits["date"]), splits["value"]):
        if (t, str(d.date())) in SHARADAR_SPLIT_DAY_LATE or d not in u.index or abs(float(v) - 1) < 0.05:
            continue
        k = u.index.get_loc(d)
        if 0 < k < len(u) - 1:
            on_day, next_day = u.iloc[k] / u.iloc[k - 1], u.iloc[k + 1] / u.iloc[k]
            if abs(np.log(on_day)) < 0.02 and abs(np.log(next_day * float(v))) < 0.02:
                LOG.warning("sh:%s 1d: its %s split of %s moved the unadjusted close on the next day (%.3f, then %.3f): "
                            "not reviewed yet (SHARADAR_SPLIT_DAY_LATE)", t, d.date(), v, on_day, next_day)
    return out, fixed


def _repeats_close_before(bars: pd.DataFrame) -> np.ndarray:
    """Rows whose open, high, low and close all repeat the close before them: a day a vendor carried the price
    forward (no trade, or a day it lost the ticker)."""
    c = bars["close"]
    return np.asarray((bars["open"] == c) & (bars["high"] == c) & (bars["low"] == c) & (c == c.shift()))


def repair_sharadar_rows(t: str, bars: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Sharadar's daily rows that only repeat the close before them on days Twelve Data's daily bars show trading
    (on reorganisation and merger days Sharadar loses the ticker: VTRS 2020-11-17..19 at 15.855 where it closed 16.34,
    16.19 and 18.11; HST, KIM, TEL, JEF, EVRG; the 2012-08-01 rows of DHI, DIS, EOG and EXC), replaced by Twelve Data's
    bar of the day on Sharadar's basis: its prices times the ratio of the two vendors' closes on the traded day before,
    which must equal the ratio on the traded day after to within a cent of rounding on each side (a split or spin-off
    in between would move it), and its volume divided by it. A day Twelve Data carries forward too is a halt (REGN
    2015-06-09 and VRTX 2015-05-12 during FDA panels) and stays, as does a row whose flatness Sharadar's rounding of
    split-adjusted prices to three decimals explains (Twelve Data's range under 0.001 on Sharadar's basis: MNST at
    $0.009 in 1998), and a run at the end of the series, which has no day after."""
    if not store.path("td", "1d", t).exists():
        return bars, []
    stale = _repeats_close_before(bars)
    if not stale.any():
        return bars, []
    td = store.read_bars("td", "1d", t)
    td_day = pd.Series(np.arange(len(td)), index=_session_days(td.index))
    td_day = td_day[~td_day.index.duplicated(keep="last")]
    td_stale = _repeats_close_before(td)
    days = _session_days(bars.index)
    out, fixed = bars.astype({"volume": float}), []
    k = 0
    while k < len(bars):
        if not stale[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(bars) and stale[j + 1]:
            j += 1
        before, after = k - 1, j + 1
        k = j + 1
        if before < 0 or after >= len(bars) or days[before] not in td_day or days[after] not in td_day:
            continue
        r_before = bars["close"].iloc[before] / td["close"].iloc[td_day[days[before]]]
        r_after = bars["close"].iloc[after] / td["close"].iloc[td_day[days[after]]]
        cent = 0.02 / min(bars["close"].iloc[before], bars["close"].iloc[after])
        if abs(r_before / r_after - 1) > cent:
            LOG.warning("sh:%s 1d: Twelve Data's closes stand %.5f and %.5f to Sharadar's around %s..%s: rows kept", t,
                        r_before, r_after, days[before + 1].date(), days[after - 1].date())
            continue
        for i in range(before + 1, after):
            pos = td_day.get(days[i])
            if pos is None or td_stale[pos]:
                continue                              # no trade there either: a halt
            row = td.iloc[pos]
            if (float(row["high"]) - float(row["low"])) * r_before < SHARADAR_PRICE_STEP:
                continue                              # a range Sharadar's rounding flattens: its row is no error
            for col in ("open", "high", "low", "close"):
                out.iloc[i, out.columns.get_loc(col)] = float(row[col]) * r_before
            out.iloc[i, out.columns.get_loc("volume")] = float(row["volume"]) / r_before
            fixed.append(str(days[i].date()))
    out["dollar_volume"] = out["close"] * out["volume"]
    return out, fixed


def store_sharadar_stocks(tickers: list[str]) -> dict:
    """The fetched daily bars of Sharadar tickers into the store as `sh:<ticker>` (its rows for days it lost the
    ticker taken from Twelve Data: `repair_sharadar_rows`), with what a holder is paid: its cash dividends and the
    value of what a spin-off hands out (`spinoffdividend`: its prices are not adjusted for spin-offs, HP's 16.62 a
    share in HPE on 2015-11-02), both on the split-adjusted basis of the bars (AAPL's 2019 dividend 0.1925, a quarter
    of the 0.77 paid before its 2020 4:1 split)."""
    actions = sharadar.table("actions")
    paid = actions[actions["action"].isin(["dividend", "spinoffdividend"])]
    paid = paid.assign(date=pd.to_datetime(paid["date"])).groupby(["ticker", "date"])["value"].sum()
    written, no_bars, repaired = [], [], []
    for t in tickers:
        raw = SHARADAR_DIR / "stocks" / f"{store.safe_name(t)}.csv"
        if not raw.exists() or not raw.stat().st_size:
            no_bars.append(t)
            continue
        bars, bad = integrity.clean_bars(sharadar_bars(raw.read_text()))
        if bad:
            LOG.warning("sh:%s 1d: dropped invalid bars %s", t, bad)
        bars, split_days = _split_rows(t, bars, pd.read_csv(raw, usecols=["date", "closeunadj"]))
        if split_days:
            LOG.info("sh:%s 1d: split rows a day off put on the adjusted basis: %s", t, ", ".join(split_days))
        bars, fixed = repair_sharadar_rows(t, bars)
        if fixed:
            LOG.info("sh:%s 1d: %d rows repeating the close before them taken from Twelve Data: %s", t, len(fixed),
                     ", ".join(fixed))
            repaired.append(t)
        store.write_bars("sh", "1d", t, bars)
        store.write_meta("sh", "1d", t, {"seeded_from": "sharadar", "first": str(bars.index.min()),
                                         "last": str(bars.index.max()), "rows_from_twelvedata": fixed})
        d = paid.loc[t] if t in paid.index.get_level_values(0) else pd.Series(dtype=float)
        path = dividends_path("sh", t)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"amount": d.to_numpy(dtype=float)},
                     index=pd.DatetimeIndex(d.index, name="ex_date")).to_parquet(path)
        written.append(t)
    LOG.info("sharadar: %d series stored, %d with no bars, %d with rows from Twelve Data", len(written), len(no_bars),
             len(repaired))
    return {"stored": len(written), "no_bars": no_bars, "rows_from_twelvedata": repaired}


def store_sharadar_hourly(tickers: list[str]) -> dict:
    """The stock lists' hourly bars: a member's Twelve Data hourly bars put on the basis of its Sharadar daily bars
    (`td_hourly_on_daily_basis`), the sessions they lack taken from Databento's minutes (`fill_from_databento`), stored
    as `sh:<ticker>` 1h with its 4h built from them: the grid and session of every hourly stock here. Twelve Data's
    hourly bars of a ticker that never match its Sharadar daily ones are another company's (a ticker taken up again
    since) and are not used, nor are its bars on days the company's daily series does not trade (Twelve Data serves
    AGN, ARG and AET as the companies that took those tickers up after Allergan, Airgas and Aetna)."""
    written, other, none = [], [], []
    for t in tickers:
        if not store.path("sh", "1d", t).exists():
            none.append(t)
            continue
        daily = store.read_bars("sh", "1d", t)
        base, rep = pd.DataFrame(columns=store.BAR_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"), dtype=float), {}
        if store.path("td", "1h", t).exists():
            base, rep = td_hourly_on_daily_basis(store.read_bars("td", "1h", t), daily)
            if base.empty:
                LOG.warning("sh:%s 1h: Twelve Data's hourly bars never match Sharadar's daily ones: another company "
                            "there, no hourly bars from it", t)
                other.append(t)
            elif rep.get("rescaled") or rep.get("dropped_through"):
                LOG.info("sh:%s 1h: put on Sharadar's daily basis: %s", t, rep)
        alien = ~_session_days(base.index).isin(_session_days(daily.index))
        if alien.any():                      # another company's bars under the ticker before or after this one's days
            LOG.info("sh:%s 1h: %d hourly bars on days its daily series does not trade dropped (%s..%s)", t,
                     int(alien.sum()), base.index[alien].min().date(), base.index[alien].max().date())
            base = base[~alien]
        # Twelve Data's sessions that are not the day's prices go first, so that Databento's fill those days
        base, conformed = conform_to_daily(base, daily)
        _log_conformed(f"sh:{t}", conformed)
        h1, filled = fill_from_databento("sh", t, base, daily)
        h1, _ = conform_to_daily(h1, daily)
        if h1.empty:
            if t not in other:
                none.append(t)
            for tf in ("1h", "4h"):              # a series written before, from another company's bars
                if store.path("sh", tf, t).exists():
                    where = store.quarantine("sh", tf, t, {"why": "no hourly bar of this company: Twelve Data's are "
                                                                  "another company's, Databento has none"})
                    LOG.warning("sh:%s %s: moved to %s: none of its hourly bars is this company's", t, tf, where)
            continue
        store.write_bars("sh", "1h", t, h1)
        store.write_meta("sh", "1h", t, {"seeded_from": "twelvedata hourly and databento minutes on sharadar daily",
                                         "daily_basis": rep, "databento": filled, "conformed": _brief(conformed),
                                         "first": str(h1.index.min()), "last": str(h1.index.max())})
        store.write_bars("sh", "4h", t, resample.equity_4h_from_1h(h1))
        written.append(t)
    LOG.info("sharadar hourly: %d stored, %d another company at Twelve Data, %d without hourly bars there or at "
             "Databento", len(written), len(other), len(none))
    return {"hourly_stored": len(written), "another_company": other, "no_hourly": len(none)}


def traded_periods(ticker: str, last_bar: pd.Timestamp) -> list[tuple[str, pd.Timestamp | None, pd.Timestamp]]:
    """The tickers a Sharadar security traded under, each with its span (start None: from before its first change),
    from its `tickerchangefrom` actions: INFO1 was MRKT until 2016-09-13 and INFO until 2022-02-25, when it was
    renamed on leaving the market, so INFO1 never traded; GAP was GPS until 2024-08-22 and GAP since."""
    a = sharadar.actions()
    changes = a[(a["action"] == "tickerchangefrom") & (a["ticker"] == ticker)]
    changes = sorted(zip(pd.to_datetime(changes["date"], utc=True), changes["contraticker"]))
    periods, start = [], None
    for d, old in changes:
        periods.append((old, start, d))
        start = d
    if start is None or last_bar > start + pd.Timedelta(days=1):
        periods.append((ticker, start, last_bar + pd.Timedelta(days=1)))
    return periods


def _session_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The New York date of the session each bar belongs to, from its close stamp."""
    return (index - pd.Timedelta(microseconds=1)).tz_convert(cal.NY).tz_localize(None).normalize()


def hourly_holes(source: str, symbol: str) -> pd.DatetimeIndex:
    """The sessions of a US listing's daily bars since DATABENTO_FROM that its hourly series lacks: all of them for a
    stock Twelve Data has no hourly bars of (a company that left the market), the years before a ticker change for
    one it serves under the new ticker only (GAP since 2024-08-22, GPS before; BNY since 2026-05-21, BK before), a
    stretch its hourly bars do not match (HON's before its 2026-06-29 split and spin-off), an ETF's year before Twelve
    Data's first hourly bar (2020-02-10), and the days its hourly series skips (2019-12-31 and 2020-01-02 for nearly
    every stock)."""
    days = _session_days(store.read_bars(source, "1d", symbol, ["close"]).index)
    days = days[days >= pd.Timestamp(DATABENTO_FROM)]
    if not store.path(source, "1h", symbol).exists():
        return days
    return days[~days.isin(_session_days(store.read_bars(source, "1h", symbol, ["close"]).index))]


def _periods(source: str, symbol: str) -> list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]]:
    """The tickers a US listing traded under, each with its span in New York dates (end excluded, None: open), from
    Sharadar's ticker changes (`traded_periods`): Twelve Data's META is FB before 2022-06-09 as Sharadar's is."""
    last = store.read_bars(source, "1d", symbol, ["close"]).index.max()
    day = lambda t: None if t is None else t.tz_convert(None).normalize()          # noqa: E731
    return [(raw, day(lo), day(hi)) for raw, lo, hi in traded_periods(symbol, last)]


def _near_change(sessions: pd.DatetimeIndex, lo: pd.Timestamp | None,
                 hi: pd.Timestamp | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """A traded span widened by DATABENTO_CHANGE_SLACK sessions on each side of a ticker change: Sharadar's date of a
    change can be a trading week off the exchange's (GL traded from 2019-08-09, Sharadar changes it on 2019-08-16;
    VIAC from 2019-12-05, a day before Sharadar's; RTX from 2020-04-03, a day after), so those days are asked under
    both tickers."""
    if not len(sessions):
        return lo, hi
    if lo is not None:
        lo = sessions[max(int(sessions.searchsorted(lo)) - DATABENTO_CHANGE_SLACK, 0)]
    if hi is not None:
        k = int(sessions.searchsorted(hi)) + DATABENTO_CHANGE_SLACK
        hi = sessions[k] if k < len(sessions) else sessions[-1] + pd.Timedelta(days=1)
    return lo, hi


def _within(days: pd.DatetimeIndex, lo: pd.Timestamp | None, hi: pd.Timestamp | None) -> np.ndarray:
    keep = np.ones(len(days), dtype=bool)
    if lo is not None:
        keep &= np.asarray(days >= lo)
    if hi is not None:
        keep &= np.asarray(days < hi)
    return keep


def _asked(venue: str, raw: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The spans (New York dates, end excluded) of a ticker's minutes already fetched from a venue, by their files."""
    name = store.safe_name(raw)
    spans = []
    for p in (DATABENTO_DIR / "minutes" / venue).glob(f"{name}_*.parquet"):
        stem, lo, hi = p.stem.rsplit("_", 2)
        if stem == name:
            spans.append((pd.Timestamp(lo), pd.Timestamp(hi)))
    return spans


def _runs(days: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The days as runs of consecutive sessions, each (its first day, the day after its last)."""
    pos = sessions.get_indexer(days)
    runs, start = [], 0
    for k in range(1, len(pos) + 1):
        if k == len(pos) or pos[k] != pos[k - 1] + 1:
            runs.append((days[start], days[k - 1] + pd.Timedelta(days=1)))
            start = k
    return runs


def databento_windows(source: str, symbol: str, holes: pd.DatetimeIndex) -> list[tuple[str, str, pd.Timestamp,
                                                                                         pd.Timestamp]]:
    """What to ask Databento for a listing's hourly holes: (venue, the ticker it traded under then, first day, the day
    after the last) for each run of consecutive sessions; a day a venue was already asked for is not asked again (its
    answer is on disk, empty when the ticker did not trade there)."""
    if holes.empty:
        return []
    sessions = _session_days(store.read_bars(source, "1d", symbol, ["close"]).index)
    out = []
    for raw, lo, hi in _periods(source, symbol):
        mine = holes[_within(holes, *_near_change(sessions, lo, hi))]
        for venue in DATABENTO_VENUES:
            done = np.zeros(len(mine), dtype=bool)
            for a, b in _asked(venue, raw):
                done |= _within(mine, a, b)
            out += [(venue, raw, a, b) for a, b in _runs(mine[~done], sessions)]
    return out


def fetch_databento_minutes(windows: list[tuple[str, str, pd.Timestamp, pd.Timestamp]], dry_run: bool = True,
                            max_usd: float | None = None) -> dict:
    """One-minute bars (`ohlcv-1m`) of the windows (venue, traded ticker, first day, the day after the last), the
    tickers of one venue and span asked in one request, each ticker's minutes kept as they arrive
    (data/raw/databento/minutes/<venue>/<ticker>_<first>_<end>.parquet, empty when it did not trade there), so a
    stopped run resumes and nothing is bought twice. Every request is priced first (`metadata.get_cost`, free); a
    real run buys nothing unless the total is within `max_usd`: the account's credit pays for it."""
    import databento as db
    client = db.Historical(env("DATABENTO_API_KEY"))
    groups: dict[tuple, set] = {}
    for venue, raw, lo, hi in windows:
        groups.setdefault((venue, lo, hi), set()).add(raw)
    asks = [{"dataset": v, "symbols": sorted(raws), "schema": "ohlcv-1m", "start": lo.strftime("%Y-%m-%d"),
             "end": hi.strftime("%Y-%m-%d"), "stype_in": "raw_symbol"} for (v, lo, hi), raws in groups.items()]

    def price(ask: dict) -> float | None:
        try:
            return _databento(client.metadata.get_cost, ask)
        except db.BentoClientError as e:
            if ((e.json_body or {}).get("detail") or {}).get("case") != "symbology_invalid_request":
                raise
            return None                       # none of its tickers traded on that venue in that span

    with ThreadPoolExecutor(DATABENTO_WORKERS) as pool:
        costs = list(pool.map(price, asks))
    total = sum(c for c in costs if c)
    report: dict = {"windows": len(windows), "requests": len(asks), "none_traded": sum(c is None for c in costs),
                    "cost_usd": round(total, 2)}
    LOG.info("databento minutes: %s", report)
    if dry_run or not asks:
        return report
    if max_usd is None or total > max_usd:
        raise RuntimeError(f"databento: the minutes cost ${total:.2f}, over the --max-usd allowed ({max_usd}): nothing "
                           "bought")

    def fetch(ask: dict, cost: float | None) -> None:
        frames = {}
        if cost is not None:
            got = _databento(client.timeseries.get_range, ask).to_df()
            frames = {s: g[MINUTE_COLUMNS] for s, g in got.groupby("symbol")} if len(got) else {}
        for raw in ask["symbols"]:
            path = _minutes_path(ask["dataset"], raw, pd.Timestamp(ask["start"]), pd.Timestamp(ask["end"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part")
            frames.get(raw, _no_minutes()).to_parquet(part)
            part.replace(path)                # a file on disk is a whole answer
        _ledger("databento", dataset=ask["dataset"], symbols=len(ask["symbols"]), first_symbol=ask["symbols"][0],
                start=ask["start"], end=ask["end"], cost_usd=cost)

    with ThreadPoolExecutor(DATABENTO_WORKERS) as pool:
        list(pool.map(fetch, asks, costs))
    report["bought_requests"] = len(asks)
    return report


def _no_minutes() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in MINUTE_COLUMNS},
                        index=pd.DatetimeIndex([], tz="UTC", name="ts_event"))


def _databento(call, ask: dict):
    """A Databento request, asked again after 1, 2 and 4 minutes when its gateway fails (a 5xx: 504 on pricing
    years of minutes, 2026-09-26) or its stream stops answering (a read timed out on sixteen years of a continuous
    future) before the run stops; files already fetched are kept, so a new run resumes."""
    import databento as db
    for wait in (60, 120, 240, None):
        try:
            return call(**ask)
        except (db.BentoServerError, db.BentoError) as e:
            retry = isinstance(e, db.BentoServerError) or "timed out" in str(e) or "Connection" in str(e)
            if wait is None or not retry:
                raise
            LOG.warning("databento %s %s %s..%s: %s, asking again in %ds", ask["dataset"], ask["symbols"][0],
                        ask["start"], ask["end"], e, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def _minutes_path(venue: str, raw: str, lo: pd.Timestamp, hi: pd.Timestamp):
    return DATABENTO_DIR / "minutes" / venue / f"{store.safe_name(raw)}_{lo.date()}_{hi.date()}.parquet"


def hourly_from_minutes(minutes: pd.DataFrame) -> pd.DataFrame:
    """One-minute bars of every venue (columns open high low close volume, a row per venue and minute, indexed by
    the minute's start) as hourly bars on the store's grid: 09:30-10:30 ... 15:30-16:00 New York, stamped at their
    close; a minute's open and close are those of the venue that traded most in it. The session ends with its closing
    cross, the print of the close minute (16:00, 13:00 on a half day) on the venue that traded most in it when that is
    more than any venue traded in the minute before (an auction gathers the day's closing orders): its open is taken
    for the auction price. The other venues' prints in that minute are trades after the close and are left out, as is
    the whole minute when none is an auction (a day the feed lacks the listing venue's cross)."""
    if minutes.empty:
        return pd.DataFrame(columns=store.BAR_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
    ts = minutes.index
    day = ts.tz_convert(cal.NY).tz_localize(None).normalize()
    sched = cal.nyse_sessions(pd.Timestamp(day.min(), tz="UTC"), pd.Timestamp(day.max(), tz="UTC"))
    m = minutes.reset_index(drop=True).assign(
        t=ts, o=pd.to_datetime(sched["market_open"].reindex(day).to_numpy(), utc=True),
        c=pd.to_datetime(sched["market_close"].reindex(day).to_numpy(), utc=True))
    regular = m[(m["t"] >= m["o"]) & (m["t"] < m["c"])]
    before = m[m["t"] == m["c"] - pd.Timedelta(minutes=1)].groupby("c")["volume"].max()
    at_close = m[m["t"] == m["c"]]
    top = at_close.loc[at_close.groupby("c")["volume"].idxmax()] if len(at_close) else at_close
    top = top[top["volume"].to_numpy(dtype=float) > before.reindex(top["c"]).fillna(0).to_numpy(dtype=float)]
    rows = pd.concat([regular.assign(at_close=False), top.assign(at_close=True)])
    offset = (rows["t"] - rows["o"]) // pd.Timedelta(minutes=1)
    last_k = ((rows["c"] - rows["o"]) // pd.Timedelta(minutes=1) - 1) // 60
    k = np.where(rows["at_close"], last_k, offset // 60)
    rows["stamp"] = np.minimum(rows["o"] + pd.to_timedelta((k + 1) * 60, unit="min"), rows["c"])
    lead = rows.sort_values(["t", "volume"]).groupby("t").tail(1).sort_values("t")
    lead["close"] = np.where(lead["at_close"], lead["open"], lead["close"])
    g = rows.groupby("stamp")
    bars = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "volume": g["volume"].sum()})
    firsts = lead.groupby("stamp")
    bars["open"] = firsts["open"].first()
    bars["close"] = firsts["close"].last()
    bars.index = pd.DatetimeIndex(bars.index).tz_convert("UTC")
    bars["dollar_volume"] = bars["close"] * bars["volume"]
    return bars[store.BAR_COLUMNS]


def _databento_hourly_path(source: str, symbol: str):
    return DATABENTO_DIR / "hourly" / source / f"{store.safe_name(symbol)}.parquet"


def build_databento_hourly(source: str, symbol: str) -> int:
    """A listing's hourly bars from the minutes of every venue fetched for it (`hourly_from_minutes`), as Databento
    prints them: prices not adjusted for splits, the volume of the lit venues only; a session that only the other
    ticker of a change printed is marked `borrowed` (the ticker may be another company's by then). Kept in
    data/raw/databento/hourly/<source>/<symbol>.parquet for `fill_from_databento`; returns how many sessions."""
    sessions = _session_days(store.read_bars(source, "1d", symbol, ["close"]).index)
    frames = []
    for raw, lo, hi in _periods(source, symbol):
        wide_lo, wide_hi = _near_change(sessions, lo, hi)
        for venue in DATABENTO_VENUES:
            for a, b in _asked(venue, raw):
                if (wide_lo is not None and b <= wide_lo) or (wide_hi is not None and a >= wide_hi):
                    continue                  # the ticker's span as another company
                m = pd.read_parquet(_minutes_path(venue, raw, a, b))
                day = m.index.tz_convert(cal.NY).tz_localize(None).normalize()
                keep = _within(day, wide_lo, wide_hi)
                if keep.any():
                    frames.append(m[keep].assign(own=_within(day[keep], lo, hi)))
    if not frames:
        return 0
    minutes = pd.concat(frames).sort_index()
    day = pd.Series(minutes.index.tz_convert(cal.NY).tz_localize(None).normalize(), index=minutes.index)
    has_own = minutes["own"].groupby(day.to_numpy()).transform("any").to_numpy()
    minutes = minutes[minutes["own"].to_numpy() | ~has_own]     # a day both tickers printed: the one of that span
    own_days = minutes.index[minutes["own"].to_numpy()].tz_convert(cal.NY).tz_localize(None).normalize().unique()
    h1 = hourly_from_minutes(minutes[MINUTE_COLUMNS])
    h1["borrowed"] = ~_session_days(h1.index).isin(own_days)    # a day only the other ticker of a change printed
    path = _databento_hourly_path(source, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    h1.to_parquet(path)
    return int(_session_days(h1.index).nunique())


def fill_from_databento(source: str, symbol: str, h1: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """A listing's hourly bars (on the basis of its daily bars) with the sessions they lack taken from Databento's
    (`build_databento_hourly`): prices put on the split-adjusted basis of the daily bars by Sharadar's split factor
    of that day (`sharadar.split_factor`), each session's volume on the daily bar's, which counts every venue (the lit
    ones Databento has carry 50-67% of it). A session's last close is the daily close, the closing auction's price
    (the last bar's range takes it in: the feeds miss the auction on some days, and a minute's first print is not
    always the auction's), and its first open the daily open when that lies inside its first hour. A daily bar that
    only repeats the close before it (open, high, low and close all at it: Sharadar's row for a day it lost the ticker,
    VTRS on 2020-11-17..19, COHR on 2022-09-08) holds no price of that day: the session keeps Databento's prices and
    volume, and is named in the report. A session borrowed from the other ticker of a change is kept only when the
    daily bar is the day's own and its close lies in the session's range: after Twenty-First Century Fox became TFCF
    for its last days, FOX was the new Fox Corporation."""
    path = _databento_hourly_path(source, symbol)
    if not path.exists():
        return h1, {}
    got = pd.read_parquet(path)
    days, d_days = _session_days(got.index), _session_days(daily.index)
    take = np.asarray(days.isin(d_days) & ~days.isin(_session_days(h1.index)))
    if not take.any():
        return h1, {"sessions": 0}
    add, days = got[take].copy(), days[take]
    f = sharadar.split_factor(symbol, days)
    for col in ("open", "high", "low", "close"):
        add[col] = add[col] * f
    flat = ((daily["open"] == daily["close"]) & (daily["high"] == daily["close"]) & (daily["low"] == daily["close"])
            & (daily["close"] == daily["close"].shift()))
    on_day = lambda x: pd.Series(np.asarray(x, dtype=float), index=d_days).reindex(days).to_numpy()  # noqa: E731
    tol = PRICE_ROUNDING_BP / 1e4
    stale, d_close = on_day(flat) == 1.0, on_day(daily["close"])
    low_s = add["low"].groupby(days).transform("min").to_numpy()
    high_s = add["high"].groupby(days).transform("max").to_numpy()
    borrowed = add["borrowed"].to_numpy(dtype=bool)
    alien = borrowed & (stale | (d_close < low_s * (1 - tol)) | (d_close > high_s * (1 + tol)))
    dropped = sorted({str(d.date()) for d in days[alien]})
    if dropped:
        LOG.warning("%s:%s 1h: sessions borrowed from the other ticker of a change that do not match its daily bars "
                    "left out: %s", source, symbol, ", ".join(dropped))
        add, days = add[~alien], days[~alien]
        if add.empty:
            return h1, {"sessions": 0, "borrowed_left_out": dropped}
    d_open, d_close, d_volume = on_day(daily["open"]), on_day(daily["close"]), on_day(daily["volume"])
    stale = on_day(flat) == 1.0
    vol = add["volume"].to_numpy(dtype=float)
    per_day = pd.Series(vol, index=days).groupby(level=0).transform("sum").to_numpy()
    scale = (per_day > 0) & ~stale
    add["volume"] = np.where(scale, vol * d_volume / np.where(per_day > 0, per_day, 1.0), vol)
    pos = pd.Series(np.arange(len(add)), index=days)
    first = np.zeros(len(add), dtype=bool)
    last = np.zeros(len(add), dtype=bool)
    first[pos.groupby(level=0).min().to_numpy()] = True
    last[pos.groupby(level=0).max().to_numpy()] = True
    o, h, lo, c = (add[x].to_numpy(dtype=float).copy() for x in ("open", "high", "low", "close"))
    use_open = first & ~stale & (d_open >= lo * (1 - tol)) & (d_open <= h * (1 + tol))
    o[use_open] = d_open[use_open]
    use_close = last & ~stale
    c[use_close] = d_close[use_close]
    h[use_close] = np.maximum(h[use_close], d_close[use_close])
    lo[use_close] = np.minimum(lo[use_close], d_close[use_close])
    add["open"], add["high"], add["low"], add["close"] = o, h, lo, c
    add["dollar_volume"] = add["close"] * add["volume"]
    report = {"sessions": int(last.sum()), "opens_from_the_daily_bar": int(use_open.sum()),
              "daily_bar_repeats_the_close_before": [str(d.date()) for d in days[last & stale]],
              "borrowed_left_out": dropped}
    if report["daily_bar_repeats_the_close_before"]:
        LOG.warning("%s:%s 1h: the daily bar only repeats the close before it on %s: those sessions keep Databento's "
                    "prices", source, symbol, ", ".join(report["daily_bar_repeats_the_close_before"]))
    out = pd.concat([h1, add[store.BAR_COLUMNS]]) if len(h1) else add[store.BAR_COLUMNS]
    return out.sort_index(), report


def _us_hourly_on_grid(index: pd.DatetimeIndex) -> np.ndarray:
    """Per bar, whether its session's hourly stamps all sit on one of a US session's grids: 10:30, 11:30 ... 16:00
    New York, or the clock hours 10:00 ... 16:00 the vendor served until 2020-06-26, the last bar at the close."""
    days = _session_days(index)
    sched = cal.nyse_sessions(pd.Timestamp(days.min(), tz="UTC"), pd.Timestamp(days.max(), tz="UTC"))
    o = pd.to_datetime(sched["market_open"].reindex(days).to_numpy(), utc=True)
    c = pd.to_datetime(sched["market_close"].reindex(days).to_numpy(), utc=True)
    m = np.asarray((index - o) / pd.Timedelta(minutes=1))
    at_close, inside = np.asarray(index == c), np.asarray((index > o) & (index <= c))
    half = pd.Series(inside & ((m % 60 == 0) | at_close)).groupby(np.asarray(days)).transform("all").to_numpy()
    clock = pd.Series(inside & (((m - 30) % 60 == 0) | at_close)).groupby(np.asarray(days)).transform("all").to_numpy()
    return half | clock


def conform_to_daily(h1: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """A US listing's hourly bars made the parts of its daily bars they are: within a regular session nothing trades
    above the day's high or below its low, the session opens at the day's open (the opening auction) and closes at its
    close (the closing auction), and its volume is the day's. Twelve Data's hourly bars break each of these: prints
    far outside the day (APA, HST and KDP on 2021-10-19 at a third below the day's low; BALL's lows at 3.2 for 52 in
    2023-03; a first bar holding the close before an earnings gap, FTNT on 2025-08-07; an IPO's offering price, CRWD),
    a first bar that opens elsewhere than the auction (0.5% off on 4.5% of the sessions), and 74-97% of the day's
    volume, a share that differs by name. So each bar's high and low are held within the day's, its open and close
    within them, the first bar opens at the day's open, the last closes at its close (their range taking them in), and
    the session's volume is put on the day's. A session whose bars are not the day's prices at all (their median close
    outside the day's range: a split day left on the other basis, FAST on 2019-05-22 at twice the price) or whose bars
    are off the session's grid (BALL on 2023-03-24) is dropped: a hole Databento's minutes fill (`hourly_holes`). A day
    whose daily bar only repeats the close before it (Sharadar's row for a day it lost the ticker) holds no price to
    conform to, and its session is kept as it is."""
    if h1.empty or daily.empty:
        return h1, {}
    days, d_days = _session_days(h1.index), _session_days(daily.index)
    dly = daily.set_axis(d_days)
    dly = dly[~dly.index.duplicated(keep="last")]
    c_d = dly["close"]
    repeats = ((dly["open"] == c_d) & (dly["high"] == c_d) & (dly["low"] == c_d) & (c_d == c_d.shift()))
    D = dly.reindex(days)
    held = D["close"].notna().to_numpy() & ~repeats.reindex(days, fill_value=False).to_numpy(dtype=bool)
    lo_d, hi_d = D["low"].to_numpy(dtype=float), D["high"].to_numpy(dtype=float)
    med = h1["close"].groupby(np.asarray(days)).transform("median").to_numpy(dtype=float)
    # a price step of slack: a bond ETF's whole day can span four cents, and two vendors' highs a cent apart
    step = np.where(hi_d >= 1.0, US_TICK, 1e-4)
    alien = held & ((med < lo_d - step) | (med > hi_d + step))
    off_grid = held & ~_us_hourly_on_grid(h1.index)
    drop = alien | off_grid
    report = {"sessions_dropped_not_the_days_prices": sorted({str(d.date()) for d in days[alien]}),
              "sessions_dropped_off_the_grid": sorted({str(d.date()) for d in days[off_grid & ~alien]})}
    out, days, held = h1[~drop].copy(), days[~drop], held[~drop]
    lo_d, hi_d = lo_d[~drop], hi_d[~drop]
    D = D[~drop]
    o, h, lo, c = (out[x].to_numpy(dtype=float).copy() for x in ("open", "high", "low", "close"))
    v = out["volume"].to_numpy(dtype=float).copy()
    clipped = held & ((h > hi_d) | (lo < lo_d))
    h = np.where(held, np.clip(h, lo_d, hi_d), h)
    lo = np.where(held, np.clip(lo, lo_d, hi_d), lo)
    o = np.where(held, np.clip(o, lo, h), o)
    c = np.where(held, np.clip(c, lo, h), c)
    pos = pd.Series(np.arange(len(out)), index=np.asarray(days))
    first, last = np.zeros(len(out), dtype=bool), np.zeros(len(out), dtype=bool)
    first[pos.groupby(level=0).min().to_numpy()] = True
    last[pos.groupby(level=0).max().to_numpy()] = True
    f, e = first & held, last & held
    o[f] = D["open"].to_numpy(dtype=float)[f]
    c[e] = D["close"].to_numpy(dtype=float)[e]
    h = np.where(f, np.maximum(h, o), h)
    lo = np.where(f, np.minimum(lo, o), lo)
    h = np.where(e, np.maximum(h, c), h)
    lo = np.where(e, np.minimum(lo, c), lo)
    per_day = pd.Series(v, index=np.asarray(days)).groupby(level=0).transform("sum").to_numpy()
    day_v = D["volume"].to_numpy(dtype=float)
    scale = held & (per_day > 0) & np.isfinite(day_v)
    v = np.where(scale, v * day_v / np.where(per_day > 0, per_day, 1.0), v)
    # a session whose hourly bars show trading with no volume takes the day's volume in the shape of the listing's
    # other sessions, each hour its median share of the day: Twelve Data's put none at all in ~1% of the sessions of
    # 2020-2022, and in 2019-2022 on some days all of the day's in one bar and none in the others (on 2022-05-27 for
    # hundreds of members), a split no hour of an S&P 500 member's session trades at
    slot = pd.Series(np.arange(len(out))).groupby(np.asarray(days)).cumcount().to_numpy()
    before = np.r_[np.nan, c[:-1]]
    still = (o == h) & (h == lo) & (lo == c) & (c == before)          # a bar that repeats the last price: no trade
    blind = pd.Series((v <= 0) & ~still).groupby(np.asarray(days)).transform("any").to_numpy()
    empty = held & blind & np.isfinite(day_v) & (day_v > 0)
    shaped = scale & (day_v > 0) & ~blind                     # a halted day's daily volume is 0: no shape in it
    if empty.any() and shaped.any():
        share = pd.Series(v[shaped] / day_v[shaped]).groupby(slot[shaped]).median()
        w = share.reindex(slot[empty]).fillna(share.median()).to_numpy()
        total = pd.Series(w).groupby(np.asarray(days)[empty]).transform("sum").to_numpy()
        v[empty] = day_v[empty] * w / np.where(total > 0, total, 1.0)
    report["sessions_given_the_days_volume"] = int(pd.Series(np.asarray(days)[empty]).nunique())
    out["open"], out["high"], out["low"], out["close"], out["volume"] = o, h, lo, c, v
    out["dollar_volume"] = out["close"] * out["volume"]
    report["bars_held_within_the_days_range"] = int(clipped.sum())
    report["sessions_conformed"] = int(e.sum())
    return out, report


def _brief(conformed: dict) -> dict:
    """A conform report for a series' meta: counts, and the first dropped sessions."""
    return {k: (v if not isinstance(v, list) else {"count": len(v), "first": v[:5]}) for k, v in conformed.items()}


def _log_conformed(iid: str, conformed: dict) -> None:
    dropped = conformed.get("sessions_dropped_not_the_days_prices", []) + conformed.get("sessions_dropped_off_the_grid", [])
    if dropped:
        LOG.warning("%s 1h: %d sessions dropped, their bars not the day's prices or off the session's grid (%s): holes "
                    "for Databento's minutes", iid, len(dropped), ", ".join(dropped[:5]))
    if conformed.get("sessions_given_the_days_volume"):
        LOG.info("%s 1h: %d sessions whose hourly bars showed trading with no volume given the day's, in the shape of "
                 "its other sessions", iid, conformed["sessions_given_the_days_volume"])


def _databento_targets(symbols: list[str] | None) -> list[tuple[str, str]]:
    """The US listings of the hourly lists: the S&P 500's members since DATABENTO_FROM with Sharadar daily bars
    (`sh:`), and the Twelve Data listings of the hourly lists (their ETFs, the largest stocks)."""
    from strategy_lab import lists, universes
    spans = sharadar.sp500_spans()
    since = pd.Timestamp(DATABENTO_FROM, tz="UTC")
    sh = {t for t, e in zip(spans["ticker"], spans["end"])
          if (pd.isna(e) or e >= since) and store.path("sh", "1d", t).exists()}
    td = set()
    for u in dict.fromkeys([x.universe for x in lists.OURS] + list(lists.ML_TASK)):
        for i in universes.resolve(u, "1h").ids:
            ins = parse(i)
            if ins.source == "td" and "/" not in ins.symbol and store.path("td", "1d", ins.symbol).exists():
                td.add(ins.symbol)
    out = [("sh", t) for t in sorted(sh)] + [("td", s) for s in sorted(td)]
    return [x for x in out if not symbols or x[1] in symbols]


def databento_hourly(symbols: list[str] | None = None, dry_run: bool = True, max_usd: float | None = None,
                     rebuild: bool = False) -> dict:
    """The hourly bars of the lists' US listings completed from Databento's minutes where they have holes
    (`hourly_holes`), then each such listing's 1h and 4h series rebuilt: a member's `sh:` series by
    `store_sharadar_hourly`, a Twelve Data listing's by `_hourly_written`; `rebuild` also rebuilds every listing
    with Databento hours already (after a change in how they are built). The dry run prices the minutes."""
    need, rebuilt = {}, []
    for source, symbol in _databento_targets(symbols):
        holes = hourly_holes(source, symbol)
        if len(holes):
            need[(source, symbol)] = holes
        elif rebuild and _databento_hourly_path(source, symbol).exists():
            rebuilt.append((source, symbol))
    windows = sorted({w for (source, symbol), holes in need.items() for w in databento_windows(source, symbol, holes)})
    report = fetch_databento_minutes(windows, dry_run, max_usd)
    report["listings_with_holes"] = {"sh": sum(s == "sh" for s, _ in need), "td": sum(s == "td" for s, _ in need)}
    report["sessions_missing"] = int(sum(len(h) for h in need.values()))
    report["listings_rebuilt_without_holes"] = len(rebuilt)
    if dry_run:
        return report
    touched = list(need) + rebuilt
    built = {f"{s}:{x}": build_databento_hourly(s, x) for s, x in touched}
    report["store"] = store_sharadar_hourly([x for s, x in touched if s == "sh"])
    for s, x in touched:
        if s == "td":
            _hourly_written(x)
    report["listings_without_minutes"] = sorted(k for k, n in built.items() if not n)
    left = {f"{s}:{x}": len(hourly_holes(s, x)) for s, x in touched}
    report["sessions_still_missing"] = {k: n for k, n in left.items() if n}
    return report


def _cme_path(product: str, rank: str):
    return CME_DIR / f"{product}.{rank}.ohlcv-1h.parquet"


def fetch_cme_hourly(products: list[str], dry_run: bool = True, max_usd: float | None = None) -> dict:
    """Databento's hourly bars of CME futures since its first day (2010-06-06): per product the most traded contract
    (`<P>.v.0`) and the next one (`<P>.v.1`, whose price at a roll puts the older contracts on the new one's basis),
    each kept in data/raw/databento/futures with the contract of every bar; one on disk is not asked again (its
    history is whole to the day it was bought). Every request is priced first (free); a real run buys nothing unless
    the total is within `max_usd`. Each continuous series takes Databento 2-12 minutes to serve; they are asked at
    once."""
    import databento as db
    from databento.common.http import BentoHttpAPI
    # the server takes minutes to start the stream of a continuous future's sixteen years, longer than the client's
    # 100 s read timeout (PL, SI and GC timed out on 2026-09-26; a year a request took as long as the whole)
    BentoHttpAPI.TIMEOUT = CME_READ_TIMEOUT
    client = db.Historical(env("DATABENTO_API_KEY"))
    end = pd.Timestamp(client.metadata.get_dataset_range(dataset=CME_DATASET)["schema"]["ohlcv-1h"]["end"]).floor("D")
    asks = [{"dataset": CME_DATASET, "symbols": [f"{p}.{rank}"], "stype_in": "continuous", "schema": "ohlcv-1h",
             "start": CME_FROM, "end": end.strftime("%Y-%m-%d")} for p in products for rank in ("v.0", "v.1")
            if not _cme_path(p, rank).exists()]                   # bought already: never twice
    costs = [_databento(client.metadata.get_cost, a) for a in asks]
    if not asks:
        return {"requests": 0, "cost_usd": 0.0}
    report: dict = {"requests": len(asks), "cost_usd": round(sum(costs), 2), "end": str(end.date())}
    LOG.info("cme hourly: %s", report)
    if dry_run:
        return report
    if max_usd is None or sum(costs) > max_usd:
        raise RuntimeError(f"databento: the CME bars cost ${sum(costs):.2f}, over the --max-usd allowed ({max_usd}): "
                           "nothing bought")
    CME_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(ask: dict, cost: float) -> None:
        got = _databento(client.timeseries.get_range, ask).to_df()
        product, rank = ask["symbols"][0].split(".", 1)
        path = _cme_path(product, rank)
        got[["instrument_id", "open", "high", "low", "close", "volume"]].to_parquet(path.with_suffix(".part"))
        path.with_suffix(".part").replace(path)
        _ledger("databento", dataset=CME_DATASET, symbols=1, first_symbol=ask["symbols"][0], start=ask["start"],
                end=ask["end"], cost_usd=cost)

    with ThreadPoolExecutor(DATABENTO_WORKERS) as pool:
        list(pool.map(fetch, asks, costs))
    report["bought"] = len(asks)
    return report


def _cme_in_session(bars: pd.DataFrame) -> pd.DataFrame:
    """A future's bars (stamped at their start) inside the week CME's metals and energy trade, without the hour from
    17:00 New York (`cme_back_adjusted`)."""
    ny_hour = bars.index.tz_convert(cal.NY).hour
    return bars[cal.fx_in_session(bars.index).to_numpy() & np.asarray(ny_hour != cal.FX_DAY_CUT_HOUR_NY)]


def cme_back_adjusted(product: str) -> tuple[pd.DataFrame, dict]:
    """The product's most traded contract hour by hour on the latest contract's basis: at each roll the contracts
    before it are scaled by the ratio of the new contract's close to the old one's at the last hour both printed
    (ratio back-adjustment: returns within a contract are its own, a roll adds no jump; a holder's returns, the roll
    yield in them). Stamped at the bar's close, inside the week CME's metals and energy trade (Sunday 18:00 to Friday
    17:00 New York, a pause from 17:00 each day, the FX week's). The hour from 17:00 is left out: until 2015-09-17 the
    pause began at 17:15, and that quarter hour, opening the next day, made a day of its own on the eve of a closed
    one (Good Friday, Christmas); after it, 1-20 contracts now and then printed in the pause. A roll with no hour both
    printed in the CME_ROLL_LOOKBACK before it is named and taken at the new contract's first price."""
    v0, v1 = (pd.read_parquet(_cme_path(product, rank)) for rank in ("v.0", "v.1"))
    v0, v1 = (_cme_in_session(x[~x.index.duplicated(keep="last")].sort_index()) for x in (v0, v1))
    ids = v0["instrument_id"].to_numpy()
    rolls = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    factor = np.ones(len(v0))
    unmatched = []
    for k in rolls[::-1]:
        new = ids[k]
        before = v1[(v1.index < v0.index[k]) & (v1.index >= v0.index[k] - CME_ROLL_LOOKBACK) & (v1["instrument_id"] == new)]
        common = before.index.intersection(v0.index[:k])
        if len(common):
            t = common.max()
            ratio = float(before.loc[t, "close"] / v0.loc[t, "close"])
        else:
            ratio = float(v0["open"].iloc[k] / v0["close"].iloc[k - 1])
            unmatched.append(str(v0.index[k]))
        factor[:k] *= ratio
    out = v0[["open", "high", "low", "close", "volume"]].astype(float).copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] * factor
    out.index = out.index + pd.Timedelta(hours=1)
    out["dollar_volume"] = out["close"] * out["volume"] * CME_CONTRACT_SIZE[product]
    if unmatched:
        LOG.warning("cme:%s: %d rolls with no hour both contracts printed the day before: taken at the new contract's "
                    "first price (%s)", product, len(unmatched), ", ".join(unmatched[:5]))
    return out[store.BAR_COLUMNS], {"rolls": len(rolls), "rolls_without_a_common_hour": len(unmatched)}


def store_cme_series(products: list[str]) -> dict:
    """The CME futures list's contracts stored as `cme:<product>`: its hourly bars (`cme_back_adjusted`), the 4h and
    daily bars built from the hours that traded, on the FX week (CME's metals and energy trade that week, a pause from
    17:00 New York each day). Twelve Data's spot quotes stay in the store for the firm's simulator, which fills at
    them; it has no copper at all (its "HG1 Copper Spot" is a stock on Frankfurt's exchange)."""
    out = {}
    for p in products:
        h1, rolls = cme_back_adjusted(p)
        store.write_bars("cme", "1h", p, h1)
        store.write_meta("cme", "1h", p, {"seeded_from": "databento GLBX continuous, ratio back-adjusted",
                                          "first": str(h1.index.min()), "last": str(h1.index.max()), "rolls": rolls})
        store.write_bars("cme", "4h", p, resample.futures_4h_from_1h(h1))
        store.write_bars("cme", "1d", p, resample.futures_1d_from_1h(h1))
        out[p] = f"{h1.index.min().date()}..{h1.index.max().date()}, {rolls['rolls']} rolls"
    return out


def dukascopy_on_utc(s: str, raw: pd.DataFrame) -> pd.DataFrame:
    """An instrument's Dukascopy bars (stamped at their start) on UTC: the spans it stamped on New York's wall clock
    put on UTC (DUKASCOPY_NEW_YORK_CLOCK), those it stamped an hour late put an hour earlier (DUKASCOPY_HOUR_LATE),
    and those whose clock changes hour to hour left out (DUKASCOPY_CLOCK_UNKNOWN)."""
    if raw.empty:
        return raw
    unknown = DUKASCOPY_CLOCK_UNKNOWN.get(s)
    if unknown:
        out = np.asarray((raw.index >= pd.Timestamp(unknown[0], tz="UTC")) & (raw.index < pd.Timestamp(unknown[1], tz="UTC")))
        if out.any():
            LOG.info("%s: %d Dukascopy hours of %s..%s, stamped late and on time in turn, left out", s,
                     int(out.sum()), *unknown)
            raw = raw[~out]
    late = DUKASCOPY_HOUR_LATE.get(s)
    if late:
        moved = np.asarray((raw.index >= pd.Timestamp(late[0], tz="UTC")) & (raw.index < pd.Timestamp(late[1], tz="UTC")))
        if moved.any():
            stamps = raw.index.to_series()
            stamps.iloc[np.flatnonzero(moved)] -= pd.Timedelta(hours=1)
            raw = raw.set_axis(pd.DatetimeIndex(stamps))
            LOG.info("%s: %d Dukascopy hours stamped an hour late put an hour earlier", s, int(moved.sum()))
            raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    spans = DUKASCOPY_NEW_YORK_CLOCK.get(s)
    if not spans:
        return raw
    idx = raw.index
    wall = np.zeros(len(idx), dtype=bool)
    for lo, hi in spans:
        wall |= np.asarray((idx >= pd.Timestamp(lo, tz="UTC")) & (idx < pd.Timestamp(hi, tz="UTC")))
    if not wall.any():
        return raw
    moved = idx[wall].tz_localize(None).tz_localize(cal.NY, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    stamps = idx.to_series().copy()
    stamps.iloc[np.flatnonzero(wall)] = moved
    out = raw.set_axis(pd.DatetimeIndex(stamps))
    out = out[out.index.notna()]
    LOG.info("%s: %d Dukascopy hours stamped on New York's clock put on UTC", s, int(wall.sum()))
    return out[~out.index.duplicated(keep="last")].sort_index()


def dukascopy_hours(blob: bytes, month_start: pd.Timestamp) -> pd.DataFrame:
    """One month of Dukascopy's hourly bid candles as raw bars stamped at their start, prices in its integer points:
    its bi5 file is LZMA-compressed 24-byte records of the seconds from the month's start, open, close, low, high and
    the tick volume. An hour without ticks is not a bar."""
    raw = lzma.decompress(blob) if blob else b""
    rows = [r for r in struct.iter_unpack(">5if", raw) if r[5] > 0]
    df = pd.DataFrame(rows, columns=["t", "open", "close", "low", "high", "ticks"], dtype=float)
    df.index = month_start + pd.to_timedelta(df["t"], unit="s")
    df["volume"] = 0.0                                       # an FX pair has no traded volume, as the vendor's
    df["dollar_volume"] = 0.0
    return df[store.BAR_COLUMNS]


def _dukascopy_month(session, s: str, p: pd.Period, budget: Budget, side: str = "BID") -> bytes:
    """One month of a pair's hourly candles on one side of the quote, kept on disk as it arrives (an empty file: a
    month it has none of), so a
    run that stops resumes where it stopped and a month is never asked twice. Dukascopy answers slowly and refuses
    with 503 or a dropped connection when asked fast: a pause after each request, and a refusal waits 1, 2 and 4
    minutes before the run stops."""
    name = DUKASCOPY_NAMES.get(s, s.replace("/", ""))
    cached = DUKASCOPY_DIR / name / (f"{p}.bi5" if side == "BID" else f"{p}.{side}.bi5")
    if cached.exists():
        return cached.read_bytes()
    url = f"{DUKASCOPY}/{name}/{p.year}/{p.month - 1:02d}/{side}_candles_hour_1.bi5"
    for wait in (60, 120, 240, None):
        budget.acquire()
        try:
            r = session.get(url, timeout=120, headers={"User-Agent": "strategy-lab"})
            status, body = r.status_code, r.content
        except requests.ConnectionError as e:
            status, body = f"connection: {type(e).__name__}", b""
        _ledger("dukascopy", status=status, symbol=s, month=str(p), bytes=len(body))
        time.sleep(DUKASCOPY_PAUSE)
        if status in (200, 404):
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(body if status == 200 else b"")
            return body if status == 200 else b""
        if wait is None:
            raise RateLimited(f"dukascopy refused {s} {p} four times ({status}): stop; a new run resumes here")
        LOG.warning("dukascopy %s %s: %s, waiting %ds before asking again", s, p, status, wait)
        time.sleep(wait)
    raise AssertionError("unreachable")


def extend_from_dukascopy(symbols: list[str], dry_run: bool = True, max_requests: int | None = None,
                          rebuild: bool = False) -> dict:
    """Hourly history of FX pairs, metals and crude before the vendor's, from Dukascopy's hourly bid candles.
    Twelve Data's hourly FX and commodities begin in 2020, and its daily FX bars before then are unusable: open == close on up to half the days of 2012, and
    false prints (EUR/USD at 1.505 on 2008-10-08, where Dukascopy's day ranges 1.354-1.375). Dukascopy's month that
    holds the vendor's first bar is fetched too, on both sides of the quote: its price point is the power of ten that
    best matches the vendor's closes there, and an instrument is extended only when the mid of its bid and ask agrees
    with them to DUKASCOPY_MATCH_BP at the median (the bid alone sits half a spread below: EUR/USD 0.4 bp, silver
    6 bp). The history kept is the bid's, with the spans Dukascopy stamped on New York's clock put on UTC
    (`dukascopy_on_utc`); a series extended already is left as it is, or with `rebuild` its part before the vendor's
    first hourly bar is built anew from the months on disk. The pair's 4h and daily bars are then rebuilt from the
    hourly ones (`_hourly_written`), a commodity keeping the vendor's daily bars (gold: 0.9 bp from the vendor over
    February 2020; Dukascopy's crude is a CFD on the future, which the match decides on). One request a month."""
    todo = []
    firsts = pd.read_csv(REFERENCE_DIR / TD_FIRST_BARS).dropna(subset=["first"])
    firsts = {r.symbol: pd.Timestamp(r.first) for r in firsts[firsts["interval"] == "1h"].itertuples(index=False)}
    for s in symbols:
        if "/" not in s or not store.path("td", "1h", s).exists():
            LOG.warning("td:%s: not an FX pair or commodity with stored hourly bars: not extended", s)
            continue
        if rebuild and s not in firsts:
            LOG.warning("td:%s: no vendor's first hourly bar on record (%s): not rebuilt", s, TD_FIRST_BARS)
            continue
        first = firsts[s] if rebuild else store.read_bars("td", "1h", s, ["close"]).index.min()
        months = pd.period_range(DUKASCOPY_FIRST_MONTH.get(s, f"{DUKASCOPY_FIRST_YEAR}-01"),
                                 first.tz_localize(None).to_period("M"), freq="M")
        todo.append((s, first, months))
    report = {"pairs": len(todo), "planned_requests": sum(len(m) for _, _, m in todo)}
    LOG.info("dukascopy FX history: %s", report)
    if dry_run:
        return report
    session = requests.Session()
    budget = Budget("dukascopy", per_minute=None, max_requests=max_requests)
    extended, skipped, unchanged = {}, {}, []
    for s, first, months in todo:
        frames = []
        for p in months:
            blob = _dukascopy_month(session, s, p, budget)
            if blob:
                frames.append(dukascopy_hours(blob, pd.Timestamp(p.start_time, tz="UTC")))
        if not frames:
            LOG.warning("td:%s: Dukascopy has no hourly candles for it", s)
            skipped[s] = "no candles"
            continue
        raw = pd.concat(frames)
        raw = dukascopy_on_utc(s, raw[~raw.index.duplicated(keep="last")].sort_index())
        now = pd.Timestamp.now(tz="UTC")
        bars = _td_normalise(s, "1h", raw, now)
        if not (bars.index < first).any():
            LOG.info("td:%s: nothing from Dukascopy before its first stored bar (%s): left as it is", s, first)
            unchanged.append(s)
            continue
        stored = store.read_bars("td", "1h", s)
        stored = stored[stored.index >= first]                   # with `rebuild`, the part built before goes
        seam = first.tz_localize(None).to_period("M")
        ask = _dukascopy_month(session, s, seam, budget, "ASK")
        mid = bars["close"]
        if ask:
            ask_close = _td_normalise(s, "1h", dukascopy_hours(ask, pd.Timestamp(seam.start_time, tz="UTC")), now)["close"]
            mid = ((bars["close"] + ask_close) / 2).dropna()
        else:
            LOG.warning("td:%s: no ask candles for %s: the bid alone is matched with the vendor", s, seam)
        both = mid.rename("close").to_frame().join(stored["close"], rsuffix="_vendor", how="inner")
        if both.empty:
            LOG.warning("td:%s: no hour where Dukascopy and the vendor both have a bar: not extended", s)
            skipped[s] = "no overlap"
            continue
        point = 10.0 ** -round(float(np.log10((both["close"] / both["close_vendor"]).median())))
        gap_bp = float(((both["close"] * point / both["close_vendor"] - 1).abs() * 1e4).median())
        if gap_bp > DUKASCOPY_MATCH_BP:
            LOG.warning("td:%s: Dukascopy's closes differ from the vendor's by %.1f bp at the median over %d hours: "
                        "not extended", s, gap_bp, len(both))
            skipped[s] = f"{gap_bp:.1f} bp apart"
            continue
        earlier = bars[bars.index < first].copy()
        earlier[["open", "high", "low", "close"]] *= point
        merged, bad = integrity.clean_bars(pd.concat([earlier, stored]))
        if bad:
            LOG.warning("td:%s 1h: dropped invalid bars on extending %s", s, bad)
        store.write_bars("td", "1h", s, merged)
        store.write_meta("td", "1h", s, {**store.read_meta("td", "1h", s), "first": str(merged.index.min()),
                                         "extended_from": "dukascopy", "dukascopy_overlap_bp": round(gap_bp, 2)})
        _hourly_written(s)
        extended[s] = f"{len(earlier)} hours from {earlier.index.min()}, {gap_bp:.2f} bp from the vendor"
    report.update({"extended": extended, "skipped": skipped, "unchanged": unchanged, "requests_made": budget.used})
    return report


def fx_hourly_holes(s: str) -> pd.DatetimeIndex:
    """The hours of an FX pair's or a metal's trading week (Sunday 17:00 to Friday 17:00 New York; a metal pauses from
    17:00 to 18:00 each day) between its first and last hourly bar that the series lacks, as close stamps: the
    vendor's holes (the last seven hours of every Friday from 2021-07-09 to 2021-10-01, 2023-02-22/23) and the hours
    of Christmas and New Year, which run thin or not at all."""
    have = store.read_bars("td", "1h", s, ["close"]).index
    starts = pd.date_range(have.min() - pd.Timedelta(hours=1), have.max() - pd.Timedelta(hours=1), freq="h")
    keep = cal.fx_in_session(starts).to_numpy()
    if s in COMMODITIES:
        keep &= np.asarray(starts.tz_convert(cal.NY).hour != cal.FX_DAY_CUT_HOUR_NY)
    closes = starts[keep] + pd.Timedelta(hours=1)
    return closes[~closes.isin(have)]


def fill_from_dukascopy(symbols: list[str], dry_run: bool = True, max_requests: int | None = None) -> dict:
    """The hours an FX pair's or a metal's hourly series lacks since Twelve Data's first hourly bar
    (`fx_hourly_holes`), from the mid of Dukascopy's bid and ask candles of that hour where it has one (an hour
    without ticks is no market). Only an instrument whose history was extended from Dukascopy (its prices match the
    vendor's there), and again only if the mid agrees with the vendor's closes to DUKASCOPY_MATCH_BP at the median
    over the months asked. Two requests for a month with a hole (bid and ask), each kept on disk; the 4h and daily
    bars are rebuilt from the hourly ones (`_hourly_written`)."""
    firsts = pd.read_csv(REFERENCE_DIR / TD_FIRST_BARS).dropna(subset=["first"])
    firsts = {r.symbol: pd.Timestamp(r.first) for r in firsts[firsts["interval"] == "1h"].itertuples(index=False)}
    todo = []
    for s in symbols:
        if store.read_meta("td", "1h", s).get("extended_from") != "dukascopy" or s not in firsts:
            LOG.warning("td:%s: not an instrument Dukascopy's prices were matched for, or no vendor's first bar: its "
                        "holes are not filled", s)
            continue
        holes = fx_hourly_holes(s)
        holes = holes[holes > firsts[s]]
        months = sorted({p for p in (holes - pd.Timedelta(hours=1)).tz_localize(None).to_period("M")})
        todo.append((s, holes, months))
    report = {"instruments": len(todo), "hours_missing": sum(len(h) for _, h, _ in todo),
              "planned_requests_at_most": 2 * sum(len(m) for _, _, m in todo)}
    LOG.info("dukascopy holes: %s", report)
    if dry_run:
        return report
    session, budget, now = requests.Session(), Budget("dukascopy", per_minute=None, max_requests=max_requests), \
        pd.Timestamp.now(tz="UTC")
    filled, skipped = {}, {}
    for s, holes, months in todo:
        mids = []
        for p in months:
            start = pd.Timestamp(p.start_time, tz="UTC")
            bid, ask = _dukascopy_month(session, s, p, budget), _dukascopy_month(session, s, p, budget, "ASK")
            if bid and ask:
                b, a = dukascopy_hours(bid, start), dukascopy_hours(ask, start)
                both = b.index.intersection(a.index)
                mids.append((b.loc[both] + a.loc[both]) / 2)
        if not mids:
            skipped[s] = "no candles"
            continue
        bars = _td_normalise(s, "1h", dukascopy_on_utc(s, pd.concat(mids).sort_index()), now)
        stored = store.read_bars("td", "1h", s)
        both = bars[["close"]].join(stored["close"], rsuffix="_vendor", how="inner")
        if both.empty:
            skipped[s] = "no hour in common with the vendor"
            continue
        point = 10.0 ** -round(float(np.log10((both["close"] / both["close_vendor"]).median())))
        gap_bp = float(((both["close"] * point / both["close_vendor"] - 1).abs() * 1e4).median())
        if gap_bp > DUKASCOPY_MATCH_BP:
            LOG.warning("td:%s: Dukascopy's mid differs from the vendor's closes by %.1f bp at the median over %d hours: "
                        "holes not filled", s, gap_bp, len(both))
            skipped[s] = f"{gap_bp:.1f} bp apart"
            continue
        add = bars[bars.index.isin(holes)].copy()
        add[["open", "high", "low", "close"]] *= point
        merged, bad = integrity.clean_bars(pd.concat([stored, add]).sort_index())
        if bad:
            LOG.warning("td:%s 1h: dropped invalid bars on filling holes %s", s, bad)
        store.write_bars("td", "1h", s, merged)
        store.write_meta("td", "1h", s, {**store.read_meta("td", "1h", s), "dukascopy_filled_hours": len(add),
                                         "dukascopy_fill_bp": round(gap_bp, 2)})
        _hourly_written(s)
        filled[s] = f"{len(add)} of {len(holes)} missing hours, {gap_bp:.2f} bp from the vendor"
    report.update({"filled": filled, "skipped": skipped, "requests_made": budget.used})
    return report


def _forexite_day(session, day: pd.Timestamp, budget: Budget) -> bytes:
    """Forexite's file of one day of its clock (a zip of every pair's minutes), kept on disk as it arrives; a day it
    has no file of (a weekend, a holiday) answers with a page, not a zip, and is kept as an empty file, so it is not
    asked again."""
    cached = FOREXITE_DIR / f"{day:%Y-%m-%d}.zip"
    if cached.exists():
        return cached.read_bytes()
    budget.acquire()
    url = f"{FOREXITE}/{day:%Y}/{day:%m}/{day:%d%m%y}.zip"
    r = session.get(url, timeout=60, headers={"User-Agent": "strategy-lab"})
    body = r.content if r.status_code == 200 and r.content[:2] == b"PK" else b""
    _ledger("forexite", status=r.status_code, day=str(day.date()), bytes=len(r.content), zip=bool(body))
    if r.status_code not in (200, 404):
        raise RateLimited(f"forexite refused {day.date()} ({r.status_code}): stop; a new run resumes here")
    if not body:
        LOG.info("forexite: no file for %s", day.date())
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(body)
    time.sleep(FOREXITE_PAUSE)
    return body


def forexite_hours(blob: bytes, pair: str) -> pd.DataFrame:
    """One Forexite day file's minutes of a pair (its name without the slash) as hourly bars stamped at their start,
    in UTC: its clock is Central European with summer time, a minute stamped at its end, so the minute stamped at a
    whole hour closes the hour before (a day's file ends with the next day's 00:00)."""
    if not blob:
        return pd.DataFrame(columns=store.BAR_COLUMNS)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        m = pd.read_csv(z.open(z.namelist()[0]))
    m.columns = [c.strip("<>").lower() for c in m.columns]
    m = m[m["ticker"] == pair]
    if m.empty:
        return pd.DataFrame(columns=store.BAR_COLUMNS)
    wall = pd.to_datetime(m["dtyyyymmdd"].astype(str) + m["time"].astype(str).str.zfill(6), format="%Y%m%d%H%M%S")
    end = pd.DatetimeIndex(wall).tz_localize(FOREXITE_ZONE, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    m = m.set_axis(end)[end.notna()]
    out = m.groupby((m.index - pd.Timedelta(minutes=1)).floor("h")).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    out["volume"] = 0.0                                       # an FX pair has no traded volume, as the vendor's
    out["dollar_volume"] = 0.0
    return out[store.BAR_COLUMNS]


def fill_from_forexite(symbols: list[str], dry_run: bool = True, max_requests: int | None = None) -> dict:
    """The hours an FX pair's hourly series lacks in its trading week (`fx_hourly_holes`), from Forexite's one-minute
    bars where it has that hour: Dukascopy, the history before the vendor's, has none of AUD/USD's first hours of the
    week while it stamped on New York's clock (2007-04..2008-09, 2009-04..09). Only when Forexite's closes agree with
    the stored ones to FOREXITE_MATCH_BP at the median over the days asked; a request a day of its clock with a hole,
    each kept on disk; the 4h and daily bars are rebuilt from the hourly ones (`_hourly_written`)."""
    todo = []
    for s in symbols:
        if "/" not in s or s in COMMODITIES or not store.path("td", "1h", s).exists():
            LOG.warning("td:%s: not an FX pair with stored hourly bars: its holes are not filled from Forexite", s)
            continue
        holes = fx_hourly_holes(s)
        # a day's file holds its hours by its clock: minutes stamped from 00:01 to the next day's 00:00
        days = sorted(set((holes - pd.Timedelta(hours=1)).tz_convert(FOREXITE_ZONE).tz_localize(None).normalize()))
        todo.append((s, holes, days))
    report = {"pairs": len(todo), "hours_missing": sum(len(h) for _, h, _ in todo),
              "planned_requests_at_most": len({d for _, _, ds in todo for d in ds})}
    LOG.info("forexite holes: %s", report)
    if dry_run:
        return report
    session = requests.Session()
    budget = Budget("forexite", per_minute=None, max_requests=max_requests)
    filled, skipped = {}, {}
    for s, holes, days in todo:
        pair = s.replace("/", "")
        frames = [forexite_hours(_forexite_day(session, d, budget), pair) for d in days]
        frames = [f for f in frames if len(f)]
        if not frames:
            skipped[s] = "no minutes"
            continue
        raw = pd.concat(frames)
        raw = raw.groupby(level=0).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                        "volume": "sum", "dollar_volume": "sum"})
        bars = _td_normalise(s, "1h", raw, pd.Timestamp.now(tz="UTC"))
        stored = store.read_bars("td", "1h", s)
        both = bars[["close"]].join(stored["close"], rsuffix="_stored", how="inner")
        if both.empty:
            skipped[s] = "no hour in common with the store"
            continue
        gap_bp = float(((both["close"] / both["close_stored"] - 1).abs() * 1e4).median())
        if gap_bp > FOREXITE_MATCH_BP:
            LOG.warning("td:%s: Forexite's closes differ from the stored ones by %.1f bp at the median over %d hours: "
                        "holes not filled", s, gap_bp, len(both))
            skipped[s] = f"{gap_bp:.1f} bp apart"
            continue
        add = bars[bars.index.isin(holes)]
        merged, bad = integrity.clean_bars(pd.concat([stored, add]).sort_index())
        if bad:
            LOG.warning("td:%s 1h: dropped invalid bars on filling holes from Forexite %s", s, bad)
        store.write_bars("td", "1h", s, merged)
        store.write_meta("td", "1h", s, {**store.read_meta("td", "1h", s), "forexite_filled_hours": len(add),
                                         "forexite_fill_bp": round(gap_bp, 2)})
        _hourly_written(s)
        filled[s] = f"{len(add)} of {len(holes)} missing hours, {gap_bp:.2f} bp from the stored closes over {len(both)}"
    report.update({"filled": filled, "skipped": skipped, "requests_made": budget.used})
    return report


def extend_crypto_from_hourly(market: str, symbols: list[str] | None = None) -> dict:
    """A coin's 4h and daily bars carried to the end of its hourly ones where they stop earlier: the exchange's 4h and
    daily candles are its hourly ones in UTC blocks, and a pair delisted in a month the archive has no file of yet kept
    only its hourly bars up to its last trade (ICX and STORJ on spot to 2026-09-03, their 4h and daily to 08-01). The
    last block may be a part of one, as the exchange's last candle of a delisted pair is."""
    done = {}
    for s in symbols or store.symbols(market, "1h"):
        h1 = store.read_bars(market, "1h", s)
        for tf, step in (("4h", "4h"), ("1d", "1D")):
            if not store.path(market, tf, s).exists():
                continue
            old = store.read_bars(market, tf, s)
            tail = h1[h1.index > old.index.max()]
            if tail.empty or tail.index.max() - old.index.max() <= pd.Timedelta(days=1):
                continue
            block = (tail.index - pd.Timedelta(microseconds=1)).floor(step) + pd.Timedelta(step)
            g = tail.groupby(block)
            new = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                                "close": g["close"].last(), "volume": g["volume"].sum(),
                                "dollar_volume": g["dollar_volume"].sum()})
            new.index = pd.DatetimeIndex(new.index, tz="UTC")
            store.write_bars(market, tf, s, pd.concat([old, new[~new.index.isin(old.index)]]).sort_index())
            done[f"{s} {tf}"] = f"{len(new)} bars to {new.index.max()}"
            LOG.info("%s:%s %s: %d bars built from its hourly ones to %s", market, s, tf, len(new), new.index.max())
    return done


def refresh_binance(markets=("perp", "spot"), tfs=("1h", "4h", "1d"), dry_run: bool = True,
                    max_requests: int | None = None, symbols: list[str] | None = None) -> dict:
    now = pd.Timestamp.now(tz="UTC")
    session = requests.Session()
    budget = Budget("binance", per_minute=None, max_requests=max_requests)
    report = {}
    for market in markets:
        in_store = sorted(set().union(*(store.symbols(market, tf) for tf in tfs)))
        if symbols:
            in_store = [x for x in in_store if x in symbols]
        trading = binance_trading(session, market, budget) if not dry_run else set(in_store)
        todo = [s for s in in_store if s in trading]
        plan = _binance_plan(market, todo, list(tfs), now)
        n_req = sum(p[2] for p in plan) + (len(todo) if market == "perp" else 0)
        report[market] = {"symbols_in_store": len(in_store), "trading": len(todo), "series_behind": len(plan),
                          "planned_requests": n_req}
        LOG.info("binance %s: %s", market, report[market])
        if dry_run:
            continue
        url = f"{FAPI}/fapi/v1/klines" if market == "perp" else f"{SPOT_API}/api/v3/klines"
        limit = 1500 if market == "perp" else 1000
        added = 0
        for s, tf, _ in plan:
            last = store.read_bars(market, tf, s).index.max()
            start_ms = int(last.timestamp() * 1000)          # the next bar opens when the last stored one closed
            rows = []
            while True:
                chunk = _binance_get(session, url, {"symbol": s, "interval": tf, "startTime": start_ms, "limit": limit},
                                     budget, {"symbol": s, "tf": tf})
                if not chunk:
                    break
                rows.extend(chunk)
                if len(chunk) < limit:
                    break
                start_ms = chunk[-1][0] + 1
            closed = [r for r in rows if r[6] < now.timestamp() * 1000]
            if not closed:
                meta = store.read_meta(market, tf, s)
                meta["checked_through"] = now
                store.write_meta(market, tf, s, meta)
                continue
            added += _append(market, tf, s, _klines_frame(closed, tf))
        if market == "perp":
            for s in todo:
                _funding(session, s, budget)
        report[market]["bars_added"] = added
        report[market]["requests_made"] = budget.used
    return report


# ---------------------------------------------------------------- TwelveData
def _td_get(session, params: dict, budget: Budget, what: dict) -> list:
    return _td_call(session, TD_API, params, budget, what).get("values") or []


def _td_call(session, url: str, params: dict, budget: Budget, what: dict) -> dict:
    """One request to the vendor: logged in the ledger, 429 stops the run, a symbol it refuses is SymbolRejected,
    "no data" is an empty answer."""
    key = env("TWELVEDATA_API_KEY")
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY is not set in the environment or .env")
    budget.acquire()
    r = session.get(url, params={**params, "apikey": key}, timeout=60)
    try:
        data = r.json()
    except ValueError:
        _ledger("twelvedata", status=r.status_code, error="non-json body", **what)
        raise RuntimeError(f"twelvedata {r.status_code}: {r.text[:200]}")
    code = data.get("code") if isinstance(data, dict) else None
    _ledger("twelvedata", status=r.status_code, code=code, rows=len(data.get("values") or []), **what)
    if r.status_code == 429 or code == 429:
        raise RateLimited("twelvedata answered 429: stop")
    if data.get("status") == "error":
        msg = data.get("message", "")
        if "No data is available" in msg or "Data not found" in msg:
            return {}
        if code in (400, 403, 404):          # about this symbol (unknown, retired, outside the plan): skip the series
            raise SymbolRejected(msg)
        raise RuntimeError(f"twelvedata: {msg}")
    return data


def _td_frame(values: list) -> pd.DataFrame:
    df = pd.DataFrame(values)
    df.index = pd.to_datetime(df["datetime"], utc=True)
    if "volume" not in df:
        df["volume"] = 0.0
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["dollar_volume"] = df["close"] * df["volume"]
    return df[store.BAR_COLUMNS].sort_index()


def td_equity_hourly_starts(labels: pd.DatetimeIndex) -> pd.Series:
    """Where each of the vendor's hourly US-equity bars really starts (UTC), indexed by its label; NaT for a bar off
    the grid of its period (see TD_HOURLY_CLOCK_HOURS_FROM)."""
    labels = labels.tz_convert("UTC")
    ny = labels.tz_convert(cal.NY)
    day = ny.tz_localize(None).normalize()
    before_2020 = np.asarray(day < TD_HOURLY_CLOCK_HOURS_FROM)
    clock_hours = np.asarray((day >= TD_HOURLY_CLOCK_HOURS_FROM) & (day < TD_HOURLY_HALF_HOURS_FROM))
    whole_hour = np.asarray(labels.minute == 0)
    europe_summer = np.asarray(labels.tz_convert("Europe/London").tz_localize(None) > labels.tz_localize(None))
    starts = pd.Series(labels, index=labels)
    shift = pd.to_timedelta(np.where(europe_summer, -30, 30), unit="min")
    starts[before_2020] = labels[before_2020] + shift[before_2020]
    first_half_hour = clock_hours & np.asarray(ny.hour == 9)
    starts[first_half_hour] = labels[first_half_hour] + pd.Timedelta(minutes=30)
    starts[(before_2020 | clock_hours) & ~whole_hour] = pd.NaT
    return starts


def _td_normalise(symbol: str, tf: str, raw: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Vendor rows (stamped at their start) re-stamped at their close; a bar still open at `now` is dropped: the
    vendor also returns the bar in progress (a session's first hour of volume passed off as its daily bar).

    A commodity's daily row is labelled with its trading date and runs from 17:00 New York the evening before, as
    an FX day does; rows labelled Saturday or Sunday hold the few ticks printed after Friday's close and are dropped.
    An hourly US-equity bar starts where `td_equity_hourly_starts` puts it, not at its label.
    """
    if symbol in COMMODITIES and tf == "1d":
        day = raw.index.tz_localize(None).normalize()
        weekday = day.dayofweek < 5
        out = raw[weekday].copy()
        out.index = cal.fx_day_close(day[weekday])
        # the vendor's daily open of a metal is often not the day's: beyond its own high or low (platinum's on 171 days
        # of 2019-2022 by 2-30 bp, palladium's on 63, silver's by a cent on some 220 days a year of 1992-2005), and
        # neither its hourly bars' first open nor the close before; its high, low and close agree with its hourly bars:
        # the open is held within the day's range
        ok = out["low"] <= out["high"]
        off = ok & ((out["open"] > out["high"]) | (out["open"] < out["low"]))
        if off.any():
            out.loc[off, "open"] = out.loc[off, "open"].clip(out.loc[off, "low"], out.loc[off, "high"])
            LOG.info("td:%s 1d: %d daily opens beyond the day's high or low held within it", symbol, int(off.sum()))
    elif "/" in symbol:
        keep = cal.fx_in_session(raw.index).to_numpy()
        out = raw[keep].copy()
        out.index = out.index + pd.Timedelta(hours=1)
    else:
        if tf == "1h":
            starts = td_equity_hourly_starts(raw.index)
            on_grid = starts.notna().to_numpy()
            if not on_grid.all():
                LOG.info("td:%s 1h: %d bar(s) off the vendor's hourly grid of their period dropped: %s", symbol,
                         int((~on_grid).sum()), ", ".join(str(t) for t in raw.index[~on_grid][:5]))
            raw = raw[on_grid].copy()
            raw.index = pd.DatetimeIndex(starts[on_grid])
            close = cal.equity_intraday_close(raw.index)
        else:
            close = cal.equity_daily_close(raw.index)
        ok = close.notna().to_numpy()
        out = raw[ok].copy()
        out.index = pd.DatetimeIndex(close[ok])
    return out[out.index <= now]


STALE_DAYS = 30     # a series that stopped this long before the store's typical end has been delisted: skip it


def td_series_to_refresh(symbols: list[str]) -> list[tuple[str, str]]:
    """Stored TwelveData series still trading: FX needs 1h only (its 4h/1d come from 1h); stocks need 1h and 1d, and so
    do commodities (the vendor's daily history starts years before its hourly one).

    Delisted = the vendor, asked after the series' last bar (the meta's "checked_through"), had nothing newer for
    STALE_DAYS. A series never checked since it was seeded is judged against the others never checked: its seed-time
    end (meta "last") that far before theirs, at the 90th percentile. Seed-time ends of different seeds are not
    compared: a series seeded today put the typical end at today and made every series of an older seed look
    delisted (2026-09-25: 1935 of 2433 skipped, AAPL among them); nor is the current end of series never checked, or a
    partial refresh would move the typical end forward and every series not yet refreshed would look delisted."""
    cands = []
    for s in symbols:
        for tf in ("1h", "1d"):
            if "/" in s and s not in COMMODITIES and tf == "1d":
                continue
            if store.path("td", tf, s).exists():
                meta = store.read_meta("td", tf, s)
                checked = meta.get("checked_through")
                end = store.read_bars("td", tf, s, ["close"]).index.max()
                cands.append((s, tf, end, pd.Timestamp(checked) if checked else None,
                              pd.Timestamp(meta["last"]) if meta.get("last") else end))
    if not cands:
        return []
    never = [seeded for _, _, _, checked, seeded in cands if checked is None]
    reference = pd.Series(never).quantile(0.9) if never else None
    stale = pd.Timedelta(days=STALE_DAYS)
    live = [(s, tf) for s, tf, end, checked, seeded in cands
            if (checked - end <= stale if checked is not None else reference - seeded <= stale)]
    LOG.info("twelvedata: %d stored series, %d still trading, %d skipped as delisted", len(cands), len(live),
             len(cands) - len(live))
    return live


def refresh_twelvedata(symbols: list[str], dry_run: bool = True, max_requests: int | None = None) -> dict:
    now = pd.Timestamp.now(tz="UTC")
    per_min = env("TWELVEDATA_REQUESTS_PER_MINUTE")
    ceiling = max_requests or (int(env("TWELVEDATA_MAX_REQUESTS_PER_RUN")) if env("TWELVEDATA_MAX_REQUESTS_PER_RUN") else None)
    budget = Budget("twelvedata", per_minute=int(per_min) if per_min else None, max_requests=ceiling)
    series = td_series_to_refresh(symbols)
    plan, unclosed, rejected_before, rejected_now, reconciled = [], [], [], [], []
    for s, tf in series:
        stored = store.read_bars("td", tf, s)
        early = stored.index > now
        if early.any():                          # written by an earlier run before they closed
            unclosed.append((s, tf))
            if not dry_run:
                LOG.warning("td:%s %s: dropping %d stored bar(s) written before they closed", s, tf, int(early.sum()))
                store.write_bars("td", tf, s, stored[~early])
                if tf == "1h":
                    _hourly_written(s)
        last = stored.index[~early].max()
        meta = store.read_meta("td", tf, s)
        if meta.get("vendor_rejected"):
            rejected_before.append((s, tf))
            continue
        if meta.get("daily_from"):                       # an ETF's daily bars: `etf_daily_reconciled` makes them
            reconciled.append((s, tf))
            continue
        checked = pd.Timestamp(meta["checked_through"]) if meta.get("checked_through") else last
        if now - max(last, checked) >= TF_DELTA[tf]:
            plan.append((s, tf))
    report = {"series": len(series), "behind": len(plan), "planned_requests_at_least": len(plan),
              "series_with_unclosed_bars": len(unclosed), "rejected_earlier_skipped": len(rejected_before),
              "daily_reconciled_skipped": len(reconciled)}
    # a quote whose hours are Exness's (`EXNESS_QUOTES`): its months of ticks not on disk and the one under way come
    # first, so that the vendor's hours stand only after Exness's last tick, and its spreads with them
    brokers = sorted({s for s, tf in plan if s in EXNESS_QUOTES and tf == "1h"})
    from strategy_lab.data import spread_refresh
    if brokers:
        report["exness_planned_requests"] = spread_refresh.fetch_exness(brokers, dry_run=True)["planned_requests"]
    LOG.info("twelvedata: %s", report)
    if dry_run:
        return report
    if not budget.per_minute:
        raise RuntimeError("set TWELVEDATA_REQUESTS_PER_MINUTE (the plan's limit) before a real refresh")
    if brokers:
        try:
            report["exness"] = spread_refresh.fetch_exness(brokers, dry_run=False)
            spread_refresh.build(brokers)
        except (RateLimited, requests.RequestException) as e:
            LOG.warning("exness: %s; the hours of %s after its last tick on disk stay the vendor's", e, brokers)
    session = requests.Session()
    added = refetched = 0
    for s, tf in plan:
        old = store.read_bars("td", tf, s)
        start = (old.index[-min(3, len(old))] - TF_DELTA[tf] * 2).strftime("%Y-%m-%d %H:%M:%S")
        meta = store.read_meta("td", tf, s)
        try:
            vals = _td_get(session, {"symbol": s, "interval": TD_INTERVAL[tf], "start_date": start, "outputsize": TD_PAGE,
                                     "order": "ASC", "timezone": "UTC"}, budget, {"symbol": s, "tf": tf, "what": "tail"})
        except SymbolRejected as e:
            LOG.warning("td:%s %s: the vendor rejects the symbol, skipped from now on: %s", s, tf, str(e)[:120])
            meta["vendor_rejected"] = {"at": str(now), "message": str(e)[:300]}
            store.write_meta("td", tf, s, meta)
            rejected_now.append((s, tf))
            continue
        meta["checked_through"] = now
        if not vals:
            store.write_meta("td", tf, s, meta)
            continue
        new = _td_normalise(s, tf, _td_frame(vals), now)
        overlap = old.index.intersection(new.index)
        broker = s in EXNESS_QUOTES             # its stored bars are Exness's, not the vendor's: nothing to restate
        if broker:
            added += _append("td", tf, s, new[new.index > old.index.max()])
        elif len(overlap) and not (abs(old.loc[overlap, "close"] / new.loc[overlap, "close"] - 1) < 1e-6).all():
            LOG.warning("td:%s %s: overlap disagrees (split or restatement): re-fetching the whole series", s, tf)
            if _replace_whole_series(s, tf, _td_full(session, s, tf, budget, now), old):
                refetched += 1
        else:
            added += _append("td", tf, s, new)
        store.write_meta("td", tf, s, meta)
        if tf == "1h" or broker:                # a broker's quote: its days are made of its hours again too
            _hourly_written(s)
    report.update({"bars_added": added, "series_refetched": refetched, "requests_made": budget.used,
                   "rejected_now": [f"{x}:{t}" for x, t in rejected_now]})
    return report


BASIS_TOL = 0.02            # a session whose hourly/daily close ratio moves this far from its stretch's level leaves it
BASIS_STEADY = 0.005        # a stretch whose ratio stays this close to its level is the same instrument ...
BASIS_MIN_SESSIONS = 20     # ... when it lasts this long: a few steady sessions of another company prove nothing


def td_hourly_on_daily_basis(h1: pd.DataFrame, d1: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """A stock's hourly bars put on the basis of its daily bars. The vendor's intraday series misses some corporate
    actions its daily series is adjusted for (APH's 2024 split, GE's 2024 and HON's 2025 spin-offs: a fake -49% and
    -22% overnight), and for a recycled ticker it serves another company's bars (BNY before 2024, GEN before 2022).

    Per session, the ratio of the last hourly close to the daily close is compared, going back in time, with the level
    of the stretch it belongs to (a five-session median decides where a stretch ends, so one bad print does not). The
    latest stretch is the reference. An earlier stretch that holds a steady level is the same instrument on another
    adjustment and is rescaled to the reference (prices by the ratio of levels, volumes by its inverse); a stretch that
    does not, or is too short to tell, is another instrument, and it and everything before it are dropped, as is the
    whole series when the latest stretch itself is not steady (however short, it is the current one)."""
    day_h = (h1.index - pd.Timedelta(microseconds=1)).tz_convert(cal.NY).tz_localize(None).normalize()
    day_d = (d1.index - pd.Timedelta(microseconds=1)).tz_convert(cal.NY).tz_localize(None).normalize()
    ratio = (h1["close"].groupby(day_h).last() / pd.Series(d1["close"].to_numpy(), index=day_d)).dropna()
    if len(ratio) < 20:
        return h1, {"sessions_compared": len(ratio)}
    med = ratio.rolling(5, center=True, min_periods=1).median()
    stretch = np.zeros(len(ratio), dtype=int)
    level, k = float(med.iloc[-1]), 0
    for j in range(len(ratio) - 1, -1, -1):
        if abs(med.iloc[j] / level - 1) > BASIS_TOL:
            k, level = k + 1, float(med.iloc[j])
        stretch[j] = k
    levels = ratio.groupby(stretch).median()
    steady = (ratio / levels.reindex(stretch).to_numpy() - 1).abs().groupby(stretch).median() <= BASIS_STEADY
    long_enough = pd.Series(stretch).value_counts().reindex(steady.index) >= BASIS_MIN_SESSIONS
    long_enough.loc[0] = True                   # the latest stretch is the current instrument, however short
    steady &= long_enough
    report: dict = {"sessions_compared": len(ratio)}
    if not steady.loc[0]:
        report["dropped_all"] = True
        return h1.iloc[:0], report
    factor = pd.Series(1.0, index=ratio.index)
    cut = None
    for k in range(1, int(stretch.max()) + 1):
        if not steady.loc[k]:
            cut = ratio.index[stretch == k].max()
            break
        factor[stretch == k] = levels.loc[0] / levels.loc[k]
    out = h1.copy()
    f = factor.reindex(day_h).fillna(1.0).to_numpy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] * f
    out["volume"] = out["volume"] / f
    rescaled = sorted({int(k) for k in stretch[factor.to_numpy() != 1.0]})
    if rescaled:
        report["rescaled"] = {f"{ratio.index[stretch == k].min().date()}..{ratio.index[stretch == k].max().date()}":
                              round(float(levels.loc[0] / levels.loc[k]), 4) for k in rescaled}
    if cut is not None:
        out = out[np.asarray(day_h > cut)]
        report["dropped_through"] = str(cut.date())
    return out, report


# Spot quotes whose hours are a broker's own, Exness's mid (`spread_refresh.exness_mid_minutes`), wherever it quotes:
# checked hour by hour against CME's future of the same commodity and against each other (2026-09-28). The vendor's
# crude mixes two prices a few percent apart within the hour from 2024-09 on and off, and in every month of 2025-05..10
# and 2026-04..07 (in 2026-05 its hours sit either about 1% or 4-5% above Exness's mid, 203 of 479 ranging over 3%):
# its hourly returns there correlate 0.41-0.80 with CME's crude where Exness's correlate 0.96-1.00, and swing back the
# next hour (autocorrelation -0.18 to -0.39, Exness's -0.09 to +0.07); a model fading the swing made +451,665% out of
# sample on 1h. Exness's crude moves with CME's in every month it quotes (0.97-1.00, on the same hour), no hour of it
# reaching 3% past CME's, and it rolls with no jump, so crude's hours are Exness's from its first tick (2019-02-18,
# before the vendor's 2020-10-04) to its last, the vendor's only after: one source, with no seam where the two quote
# different contract months (1-5% apart for months of 2021-2022, moving together).
EXNESS_QUOTES = ("WTI/USD",)
# The hours such a quote's broker lacks while the vendor quoted them and the market traded: Exness's crude has no tick
# from 2021-03-02 14:00 to 2021-03-04 00:00 UTC (the vendor's hours of that week move with CME's to 1.00). The
# vendor's returns stand in, spliced on Exness's close of the hour before. The other 35 hours of the vendor's it lacks
# are single hours in which it quoted nothing (its first hour after the daily or the weekly pause, a holiday's short
# session's last): left without a bar, as hours its clients could not trade in.
EXNESS_HOLES = {"WTI/USD": (("2021-03-02 15:00", "2021-03-03 23:00"),)}
# The vendor's hours of a spot quote that are not the market's, found hour by hour against CME's future of the same
# metal (2026-09-29, 2020-2026: the vendor's price over CME's more than 2% from that ratio's median over the day around
# it) and kept here where Exness's mid held to CME's while the vendor's did not: most a close the vendor left behind in
# a fast hour or before the daily pause and caught up the hour after (silver 2026-02-05 22:00: the vendor +1.60% into
# the hour, CME -3.80%, Exness -3.74%, then the vendor -5.08%; platinum 2026-01-29 16:00: -4.37% against -7.85% and
# -8.01%), some a price no market made (palladium at 972-992 and 894-897, flat, in 2025-06-27 and 06-30..07-01, where
# Exness quoted 1137-1142 and 1110-1114; at 1061 from 2025-09-01 19:00, a US holiday's close, carried flat until the
# market opened at 23:00 near 1145). Exness's returns stand in, spliced on the vendor's close of the hour before so the
# series does not jump to Exness's level and back; an hour Exness did not quote has no bar. Left as the vendor's where
# Exness departed with it (the spot's own moves against the futures: gold's and platinum's hours of March-May 2020,
# palladium's 2024-06-21 14:00 and 2025-01-31 17:00, its January 2020 squeeze with no Exness to judge by), and its 29
# hours of 2025-07-11 08:00..07-14 13:00, 5-6% above Exness's but not shown to be no market's price (a London premium
# over the futures would look the same; its platinum stood 5% above Exness's for two hours of that day).
VENDOR_HOURS_WRONG = {
    "XAG/USD": tuple((t, t) for t in ("2020-03-12 17:00", "2025-12-26 22:00", "2026-02-05 22:00", "2026-03-19 14:00",
                                      "2026-03-24 21:00")),
    "XPT/USD": (("2025-06-30 18:00", "2025-06-30 18:00"), ("2025-12-15 08:00", "2025-12-15 09:00"),
                ("2025-12-31 03:00", "2025-12-31 03:00"), ("2026-01-02 22:00", "2026-01-02 22:00"),
                ("2026-01-26 20:00", "2026-01-26 20:00"), ("2026-01-26 22:00", "2026-01-26 22:00"),
                ("2026-01-27 15:00", "2026-01-27 15:00"), ("2026-01-29 16:00", "2026-01-29 16:00"),
                ("2026-02-05 22:00", "2026-02-06 00:00")),
    "XPD/USD": (("2020-03-13 06:00", "2020-03-13 06:00"), ("2020-03-16 10:00", "2020-03-16 10:00"),
                ("2020-03-16 16:00", "2020-03-16 17:00"), ("2020-03-18 06:00", "2020-03-18 06:00"),
                ("2020-03-19 06:00", "2020-03-19 06:00"), ("2020-03-20 10:00", "2020-03-20 10:00"),
                ("2020-03-23 14:00", "2020-03-23 14:00"), ("2020-03-26 01:00", "2020-03-26 01:00"),
                ("2021-12-17 14:00", "2021-12-17 14:00"), ("2025-06-27 18:00", "2025-06-27 20:00"),
                ("2025-06-30 23:00", "2025-07-01 01:00"), ("2025-08-01 13:00", "2025-08-01 13:00"),
                ("2025-09-01 19:00", "2025-09-01 22:00"), ("2025-11-28 14:00", "2025-11-28 15:00"),
                ("2026-01-02 22:00", "2026-01-02 22:00"), ("2026-01-26 22:00", "2026-01-26 22:00"),
                ("2026-01-28 22:00", "2026-01-28 22:00")),
}


def exness_hourly(s: str) -> pd.DataFrame:
    """A quote's hours of Exness's mid, made of its minutes (`spread_refresh.exness_mid_minutes`), stamped at their
    close; no volume, as the vendor's quotes record none."""
    from strategy_lab.data.spread_refresh import exness_mid_minutes
    m = exness_mid_minutes(s)
    h = m.resample("1h", label="right", closed="right").agg({"open": "first", "high": "max", "low": "min",
                                                             "close": "last"}).dropna()
    return h.assign(volume=0.0, dollar_volume=0.0)[store.BAR_COLUMNS]


def _spliced(s: str, src: pd.DataFrame, onto: pd.DataFrame, a: str, b: str) -> pd.DataFrame | None:
    """`src`'s hours from `a` to `b` on `onto`'s level: scaled by the ratio of the two closes of the last hour before
    `a` that both have, so that the hours carry `src`'s returns and the series they go into does not jump. None when
    `src` has no hour there or no hour before it in common with `onto`."""
    a, b = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
    both = src.index.intersection(onto.index)
    before = both[both < a]
    hours = src.loc[a:b]
    if not len(before) or hours.empty:
        LOG.info("td:%s: nothing of %s..%s to splice (no hour before it both sources have, or none in it)", s, a, b)
        return None
    k = onto.at[before[-1], "close"] / src.at[before[-1], "close"]
    return hours.assign(**{c: hours[c] * k for c in ("open", "high", "low", "close")})


def broker_hours(s: str, h1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """A spot quote's hourly bars with Exness's where the vendor's are not the market's: every hour Exness quotes for a
    quote of `EXNESS_QUOTES` (the vendor's before its first and after its last, and in `EXNESS_HOLES` spliced on
    Exness's level), the hours `VENDOR_HOURS_WRONG` names spliced on the vendor's level, or left out where Exness did
    not quote them. The bars, and the hours not the vendor's own, which are not held to its days."""
    if s in EXNESS_QUOTES:
        ex = exness_hourly(s)
        holes = [x for a, b in EXNESS_HOLES.get(s, ()) if (x := _spliced(s, h1, ex, a, b)) is not None]
        holes = [x[~x.index.isin(ex.index)] for x in holes]
        out = pd.concat([h1[h1.index < ex.index[0]], ex, *holes, h1[h1.index > ex.index[-1]]]).sort_index()
        return out, ex.index.append([x.index for x in holes]).sort_values()
    spans = VENDOR_HOURS_WRONG.get(s, ())
    ex = exness_hourly(s) if spans else None
    out, taken, gone = h1.copy(), pd.DatetimeIndex([], tz="UTC"), pd.DatetimeIndex([], tz="UTC")
    prices = ["open", "high", "low", "close"]
    for a, b in spans:
        wrong = h1.loc[pd.Timestamp(a, tz="UTC"):pd.Timestamp(b, tz="UTC")].index
        hours = _spliced(s, ex, h1, a, b)
        spliced = wrong[:0] if hours is None else wrong.intersection(hours.index)
        if len(spliced):
            out.loc[spliced, prices] = hours.loc[spliced, prices]
        unquoted = wrong.difference(spliced)
        if len(unquoted):
            LOG.info("td:%s: %d hours of %s..%s not the market's, which Exness did not quote: no bar", s,
                     len(unquoted), a, b)
        taken, gone = taken.append(spliced), gone.append(unquoted)
    return out.drop(gone), taken.sort_values()


def _short_days(h1: pd.DataFrame) -> pd.DataFrame:
    """Each day of these hours as one bar however few they are (a holiday's short session, which
    `resample.fx_1d_from_1h` does not make a day of), stamped at its last hour's close; the last day, which may still
    be under way, left out."""
    label = pd.DatetimeIndex(cal.fx_day_label(h1.index - pd.Timedelta(hours=1)))
    out = h1.groupby(label).agg(resample.AGG)
    out.index = pd.DatetimeIndex(pd.Series(h1.index, index=label).groupby(level=0).max().to_numpy(), name=h1.index.name)
    return out.iloc[:-1]


def conform_commodity_hourly(h1: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """A spot commodity's hourly bars held to the vendor's daily bars (then `_commodity_daily`): within the day, 17:00
    to 17:00 New York, nothing trades above its high or below its low, and a first open or a last close (of an hour
    that ends the day) beyond the day's range is the day's open or close. The vendor's hourly quotes carry prints its
    daily bars do not: silver's first hour of 2022-05-16 opening at 25.24 on a day that traded 20.84-21.72, its last
    of 2020-08-03 closing at 25.48 where the day closed at 24.25 and the next opened at 24.25. An open or close inside
    the range stays: gold's and silver's hours before 2020 are Dukascopy's bid, 1-4 bp under the vendor's daily
    prices, which set on every day would jump the series at each day's turn; a day whose last hours are missing
    (crude's 2023-03-17, ten hours) keeps its last close."""
    if h1.empty or daily.empty:
        return h1, {}
    days = pd.DatetimeIndex(cal.fx_day_label(h1.index - pd.Timedelta(hours=1)))
    dly = daily.set_axis(pd.DatetimeIndex(cal.fx_day_label(daily.index - pd.Timedelta(seconds=1))))
    D = dly[~dly.index.duplicated(keep="last")].reindex(days)
    held = D["close"].notna().to_numpy()
    lo_d, hi_d = D["low"].to_numpy(dtype=float), D["high"].to_numpy(dtype=float)
    out = h1.copy()
    o, h, lo, c = (out[x].to_numpy(dtype=float).copy() for x in ("open", "high", "low", "close"))
    off_o, off_c = held & ((o > hi_d) | (o < lo_d)), held & ((c > hi_d) | (c < lo_d))
    clipped = held & ((h > hi_d) | (lo < lo_d))
    h = np.where(held, np.clip(h, lo_d, hi_d), h)
    lo = np.where(held, np.clip(lo, lo_d, hi_d), lo)
    o = np.where(held, np.clip(o, lo, h), o)
    c = np.where(held, np.clip(c, lo, h), c)
    pos = pd.Series(np.arange(len(out)), index=np.asarray(days))
    first, last = np.zeros(len(out), dtype=bool), np.zeros(len(out), dtype=bool)
    first[pos.groupby(level=0).min().to_numpy()] = True
    last[pos.groupby(level=0).max().to_numpy()] = True
    f = first & off_o
    e = last & off_c & np.asarray(out.index == cal.fx_day_close(days))
    o[f], c[e] = D["open"].to_numpy(dtype=float)[f], D["close"].to_numpy(dtype=float)[e]
    h, lo = np.maximum(h, np.where(f, o, h)), np.minimum(lo, np.where(f, o, lo))
    h, lo = np.maximum(h, np.where(e, c, h)), np.minimum(lo, np.where(e, c, lo))
    out["open"], out["high"], out["low"], out["close"] = o, h, lo, c
    return out, {"bars_held_within_the_days_range": int(clipped.sum()),
                 "days_opened_or_closed_at_the_days": int(f.sum() + e.sum())}


def _commodity_daily(s: str, h1: pd.DataFrame, vendor: pd.DataFrame) -> pd.DataFrame:
    """A spot commodity's daily bars, as the FX majors': a day built from its hourly bars (held to the vendor's day
    first, `conform_commodity_hourly`) wherever the hours make it whole (`resample.fx_1d_from_1h`), its close the 17:00
    New York quote the day's bar stands for. The vendor's own daily close is not: platinum's, palladium's and crude's
    sit 25 bp and more from their own last hour on a quarter to a third of the days since 2021 (silver's on a tenth,
    gold's on 2%), a fixing's or a settlement's round figures (platinum's 887.0, 886.0, 878.5 in 2019), and its open
    often lies beyond its high or low. The vendor's daily bar stays on the other days, before the hourly bars begin
    (gold's before 2003, silver's before 2011, platinum's and palladium's before 2020, crude's before 2019-02), but
    where the hours are a broker's (`EXNESS_QUOTES`: a day of theirs too short to be whole is made of its own hours,
    `_short_days`, and given here as another day), and the series runs from its last hole longer than MAX_HOLE on."""
    built = resample.fx_1d_from_1h(h1)
    b_days = pd.DatetimeIndex(cal.fx_day_label(built.index - pd.Timedelta(seconds=1)))
    v_days = pd.DatetimeIndex(cal.fx_day_label(vendor.index - pd.Timedelta(seconds=1)))
    kept, cut = _continuous_to(pd.concat([built, vendor[~v_days.isin(b_days)]]).sort_index(), vendor.index.max())
    LOG.info("td:%s 1d: %d days built from the hourly bars, %d the other days given; %d bars before its last hole "
             "longer than %d days left out", s, len(built), int((~v_days.isin(b_days)).sum()), cut, MAX_HOLE.days)
    return kept


def _hourly_written(s: str) -> None:
    """After a TwelveData hourly series changes: a stock's or ETF's hourly bars are put on its daily basis and the
    sessions they lack are taken from Databento's (`fill_from_databento`; the sessions of its Databento bars are
    Databento's, so a rebuild of those replaces them), then its 4h bars (and an FX pair's daily bars) are built from the
    hourly ones; a commodity's hourly bars are held to the vendor's daily bars (`conform_commodity_hourly`), then its
    days are built from them where they make a day whole, the vendor's elsewhere (`_commodity_daily`)."""
    h1 = store.read_bars("td", "1h", s)
    if "/" not in s:
        layer = _databento_hourly_path("td", s)
        if layer.exists():
            h1 = h1[~_session_days(h1.index).isin(_session_days(pd.read_parquet(layer, columns=["close"]).index))]
        if store.path("td", "1d", s).exists():
            fixed, rep = td_hourly_on_daily_basis(h1, store.read_bars("td", "1d", s))
            if rep.get("rescaled") or rep.get("dropped_through") or rep.get("dropped_all"):
                LOG.warning("td:%s 1h: put on the daily bars' basis: %s", s, rep)
                store.write_meta("td", "1h", s, {**store.read_meta("td", "1h", s), "daily_basis": rep})
                if fixed.empty:
                    for tf in ("1h", "4h"):
                        if store.path("td", tf, s).exists():
                            where = store.quarantine("td", tf, s, {"why": "hourly bars of another instrument", **rep})
                            LOG.warning("td:%s %s: moved to %s: its hourly bars never match its daily bars", s, tf, where)
                    return
                store.write_bars("td", "1h", s, fixed)
                h1 = fixed
            daily = store.read_bars("td", "1d", s)
            h1, conformed = conform_to_daily(h1, daily)     # its sessions that are not the day's go before the fill
            _log_conformed(f"td:{s}", conformed)
            h1, filled = fill_from_databento("td", s, h1, daily)
            if filled.get("sessions"):
                LOG.info("td:%s 1h: %d sessions it lacks taken from Databento's minutes", s, filled["sessions"])
            h1, _ = conform_to_daily(h1, daily)
            store.write_meta("td", "1h", s, {**store.read_meta("td", "1h", s), "databento": filled,
                                             "conformed": _brief(conformed)})
            store.write_bars("td", "1h", s, h1)
        else:
            LOG.warning("td:%s 1h: no daily series to check the hourly bars against", s)
    if s in COMMODITIES and store.path("td", "1d", s).exists():
        vendor = store.read_bars("td", "1d", s)
        h1, broker = broker_hours(s, h1)
        own = ~h1.index.isin(broker)                # a broker's hours are not held to the vendor's days
        held, conformed = conform_commodity_hourly(h1[own], vendor)
        h1 = pd.concat([held, h1[~own]]).sort_index()
        if conformed.get("bars_held_within_the_days_range"):
            LOG.info("td:%s 1h: %d bars held within the day's range", s, conformed["bars_held_within_the_days_range"])
        if len(broker):
            LOG.info("td:%s 1h: %d hours from Exness's ticks, or spliced with them, %s..%s", s, len(broker),
                     broker.min(), broker.max())
        if s in EXNESS_QUOTES and len(broker):      # the days Exness quotes are its: a short session's from its hours
            first, last = cal.fx_day_label(broker[[0, -1]] - pd.Timedelta(hours=1))
            v_days = pd.DatetimeIndex(cal.fx_day_label(vendor.index - pd.Timedelta(seconds=1)))
            vendor = pd.concat([vendor[(v_days < first) | (v_days > last)], _short_days(h1.loc[broker])]).sort_index()
        store.write_bars("td", "1h", s, h1)
        store.write_bars("td", "1d", s, _commodity_daily(s, h1, vendor))
    store.write_bars("td", "4h", s, resample.fx_4h_from_1h(h1, daily_break=s in COMMODITIES) if "/" in s
                     else resample.equity_4h_from_1h(h1))
    if "/" not in s and store.path("sh", "1d", s).exists():           # the stock lists' hourly bars follow
        store_sharadar_hourly([s])
    if "/" in s and s not in COMMODITIES:
        store.write_bars("td", "1d", s, resample.fx_1d_from_1h(h1))


def _td_start(s: str, tf: str) -> str:
    return TD_HISTORY_START["fx_1h" if "/" in s and tf == "1h" else tf]


def _td_full(session, s: str, tf: str, budget: Budget, now: pd.Timestamp,
             until: pd.Timestamp | None = None) -> pd.DataFrame:
    """The whole series since TD_HISTORY_START (only the bars that start by `until`, when given), paged backwards:
    given an output size, the vendor returns the LATEST bars of the requested window (a start date alone returns the
    newest page, not the oldest one)."""
    start = _td_start(s, tf) + " 00:00:00"
    frames, end = [], until
    while True:
        params = {"symbol": s, "interval": TD_INTERVAL[tf], "start_date": start, "outputsize": TD_PAGE, "order": "ASC",
                  "timezone": "UTC"}
        if end is not None:
            params["end_date"] = end.strftime("%Y-%m-%d %H:%M:%S")
        vals = _td_get(session, params, budget, {"symbol": s, "tf": tf, "what": "full" if until is None else "history"})
        if not vals:
            break
        f = _td_frame(vals)
        frames.append(f)
        if len(vals) < TD_PAGE:
            break
        end = f.index.min() - pd.Timedelta(seconds=1)
    if not frames:
        LOG.warning("td:%s %s: the vendor has no bars from %s to %s", s, tf, start, until or "now")
        return pd.DataFrame(columns=store.BAR_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))
    raw = pd.concat(frames)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    return _td_normalise(s, tf, raw, now)


def _replace_whole_series(s: str, tf: str, new: pd.DataFrame, old: pd.DataFrame) -> bool:
    """Write a re-fetched series unless it starts later than the stored one: a shorter history is never accepted in
    place of a longer one (the stored series stays, and the gap is reported)."""
    new = integrity.clean_bars(new)[0]
    if new.empty or (len(old) and new.index.min() > old.index.min() + pd.Timedelta(days=7)):
        LOG.error("td:%s %s: re-fetched history starts %s, the stored one %s: kept the stored series", s, tf,
                  None if new.empty else new.index.min().date(), old.index.min().date() if len(old) else None)
        return False
    store.write_bars("td", tf, s, new)
    if tf == "1h":
        _hourly_written(s)
    return True


def refetch_twelvedata(pairs: list[tuple[str, str]]) -> dict:
    """Re-fetch whole series: after a vendor restatement, or to restore a history an earlier run cut short."""
    now = pd.Timestamp.now(tz="UTC")
    per_min = env("TWELVEDATA_REQUESTS_PER_MINUTE")
    budget = Budget("twelvedata", per_minute=int(per_min) if per_min else None, max_requests=None)
    session = requests.Session()
    done = []
    for s, tf in pairs:
        old = store.read_bars("td", tf, s)
        new = _td_full(session, s, tf, budget, now)
        if _replace_whole_series(s, tf, new, old):
            done.append(f"{s}:{tf} {new.index.min().date()}..{new.index.max().date()} ({len(new)} bars)")
    return {"refetched": done, "requests_made": budget.used}


def seed_twelvedata_series(symbols: list[str], tfs=("1h", "1d"), dry_run: bool = True,
                           max_requests: int | None = None) -> dict:
    """Whole history of TwelveData series the store does not have yet (new instruments), since TD_HISTORY_START.

    The dry run counts pages as if the market never closed, so it is a ceiling: the vendor's history may also start
    later than TD_HISTORY_START. A symbol the vendor refuses is reported and skipped; the rest of the run goes on."""
    now = pd.Timestamp.now(tz="UTC")
    todo = []
    for s in symbols:
        for tf in tfs:
            if tf not in TD_INTERVAL:
                LOG.warning("td:%s %s: not fetched from the vendor (4h is built from 1h): skipped", s, tf)
                continue
            if store.path("td", tf, s).exists():
                continue
            start = pd.Timestamp(_td_start(s, tf), tz="UTC")
            todo.append((s, tf, math.ceil((now - start) / TF_DELTA[tf] / TD_PAGE)))
    report = {"series": len(todo), "planned_requests_at_most": sum(t[2] for t in todo)}
    LOG.info("twelvedata new series: %s", report)
    if dry_run:
        return report
    per_min = env("TWELVEDATA_REQUESTS_PER_MINUTE")
    if not per_min:
        raise RuntimeError("set TWELVEDATA_REQUESTS_PER_MINUTE (the plan's limit) before a real seed")
    budget = Budget("twelvedata", per_minute=int(per_min), max_requests=max_requests)
    session = requests.Session()
    written, rejected, empty = [], [], []
    for s, tf, _ in todo:
        try:
            bars = integrity.clean_bars(_td_full(session, s, tf, budget, now))[0]
        except SymbolRejected as e:
            LOG.warning("td:%s %s: the vendor rejects the symbol: %s", s, tf, str(e)[:120])
            rejected.append(f"{s}:{tf}")
            continue
        if bars.empty:
            LOG.warning("td:%s %s: no closed bars to store: the series is not created", s, tf)
            empty.append(f"{s}:{tf}")
            continue
        store.write_bars("td", tf, s, bars)
        store.write_meta("td", tf, s, {"seeded_from": "twelvedata_api", "first": str(bars.index.min()),
                                       "last": str(bars.index.max()), "checked_through": str(now)})
        if tf == "1h":
            _hourly_written(s)
        written.append(f"{s}:{tf} {bars.index.min().date()}..{bars.index.max().date()} ({len(bars)} bars)")
    report.update({"series_written": written, "rejected": rejected, "no_bars": empty, "requests_made": budget.used})
    return report


def twelvedata_first_bars(symbols: list[str], dry_run: bool = True, max_requests: int | None = None) -> dict:
    """The vendor's first hourly bar of each symbol (its earliest_timestamp), kept in
    data/reference/twelvedata_first_bars.csv for the data check (`strategy_lab.data.check`): a stored hourly series
    that starts later is short. One request a symbol, and a second one for a symbol with no hourly series stored: the
    vendor may name a first bar it has no bars for (EA, taken private in 2026: a 2021 date, and "no data" for any
    date), so the date is kept only if a bar is there. A symbol already in the file is not asked again; one the vendor
    refuses or has no hourly bars for is kept with an empty date."""
    path = REFERENCE_DIR / TD_FIRST_BARS
    cols = ["symbol", "interval", "first", "asked"]
    have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)
    todo = [s for s in dict.fromkeys(symbols) if s not in set(have["symbol"])]
    confirm = sum(1 for s in todo if not store.path("td", "1h", s).exists())
    report = {"symbols": len(todo), "planned_requests_at_most": len(todo) + confirm}
    LOG.info("twelvedata first hourly bars: %s", report)
    if dry_run:
        return report
    per_min = env("TWELVEDATA_REQUESTS_PER_MINUTE")
    if not per_min:
        raise RuntimeError("set TWELVEDATA_REQUESTS_PER_MINUTE (the plan's limit) before asking the vendor")
    budget = Budget("twelvedata", per_minute=int(per_min), max_requests=max_requests)
    session, rows, asked = requests.Session(), [], str(pd.Timestamp.now(tz="UTC").date())
    try:
        for s in todo:
            what = {"symbol": s, "tf": "1h", "what": "earliest"}
            try:
                data = _td_call(session, TD_EARLIEST_API, {"symbol": s, "interval": "1h"}, budget, what)
                first = pd.Timestamp(data["unix_time"], unit="s", tz="UTC") if "unix_time" in data else None
                if first is not None and not store.path("td", "1h", s).exists():
                    bar = _td_get(session, {"symbol": s, "interval": "1h", "start_date": str(first), "outputsize": 1,
                                            "timezone": "UTC"}, budget, {**what, "what": "earliest confirmed"})
                    if not bar:
                        LOG.warning("td:%s: the vendor names %s as its first hourly bar but has no bar there", s, first)
                        first = None
            except SymbolRejected as e:
                LOG.warning("td:%s: the vendor rejects the symbol: %s", s, str(e)[:120])
                first = None
            if first is None:
                LOG.warning("td:%s: no first hourly bar from the vendor: kept with an empty date", s)
            rows.append({"symbol": s, "interval": "1h", "first": first, "asked": asked})
    finally:
        if rows:                                 # what was asked before a stop is kept, and not asked again
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat([have, pd.DataFrame(rows, columns=cols)]).sort_values("symbol").to_csv(path, index=False)
    report.update({"written": len(rows), "requests_made": budget.used})
    return report


YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
TD_SPLITS_API = "https://api.twelvedata.com/splits"
TD_SPLITS_CREDITS = 20                  # what a splits request costs (its Api-Credits-Request header, 2026-09-25)
# Distributions the stored prices already carry: Twelve Data records these spin-offs as splits and adjusts its prices
# for them on the ex-date (the adjustment it applied that day, from its unadjusted and adjusted closes, is the size of
# the distribution), while Yahoo pays the same event again as a dividend. Reviewed 2026-09-25, event by event, among the
# 21 dividends that fall on a Twelve Data split date; the other 18 are real payments on top of what the prices carry:
# a "1-for-1 split" that adjusts nothing (GD 1993, PGR 2014, VNO 2009), a real split beside a regular dividend (COP
# 1985, JCI 1997), the cash of a merger or of a dividend paid partly in stock (RIG 2007, TMUS 2013, SPG 2009, DXC
# 2015, WY 2010), and Tyco's 2007 spin-offs of Covidien and TE, which its prices (JCI's history) do not carry.
CARRIED_BY_PRICES = {("ABT", "2004-05-03"): "Hospira spin-off, a 10000-for-9356 split at Twelve Data",
                     ("DHR", "2016-07-05"): "Fortive spin-off, a 1319-for-1000 split at Twelve Data",
                     ("XRX", "2017-01-03"): "Conduent spin-off, a 1518-for-1000 split at Twelve Data"}
PAID_ON_SPLIT_DATE = {                   # the same review: real payments beside what the prices carry that day
    ("ADBE", "1997-07-29"): "a 1-for-1 split adjusts nothing", ("GD", "1993-03-30"): "a 1-for-1 split adjusts nothing",
    ("GD", "1993-06-15"): "a 1-for-1 split adjusts nothing", ("PGR", "2014-01-27"): "a 1-for-1 split adjusts nothing",
    ("VNO", "2009-05-07"): "a 1-for-1 split adjusts nothing", ("VNO", "2009-08-07"): "a 1-for-1 split adjusts nothing",
    ("VNO", "2009-11-06"): "a 1-for-1 split adjusts nothing", ("COP", "1985-07-05"): "a regular dividend beside a 3:1 split",
    ("BLK", "2007-06-05"): "a 1-for-1 split adjusts nothing (0.67 as every quarter of 2007, no break in the prices)",
    ("GD", "1994-04-11"): "a regular dividend beside a 2:1 split", ("JCI", "1997-04-01"): "a regular dividend beside a 2:1 split",
    ("RIG", "2007-11-27"): "the cash of the GlobalSantaFe merger; the share exchange is the split",
    ("TMUS", "2013-05-01"): "the cash of the MetroPCS merger; the 1-for-2 exchange is the split",
    ("SPG", "2009-08-13"): "the cash part of a dividend paid mostly in stock; the stock part is the split",
    ("SPG", "2009-11-12"): "the cash part of a dividend paid mostly in stock; the stock part is the split",
    ("DXC", "2015-11-30"): "a special cash dividend beside the CSRA spin-off the split carries",
    ("WY", "2010-07-20"): "the REIT conversion's special dividend; the split carries its stock part",
    ("JCI", "2007-07-02"): "Tyco's spin-offs of Covidien and TE beside its 1-for-4 split (JCI's history is Tyco's)"}


YAHOO_RENAMED = {"BK": "BNY"}            # BNY Mellon has traded as BNY since 2024: Yahoo keeps its history there


def _yahoo_symbol(symbol: str) -> str:
    return YAHOO_RENAMED.get(symbol, symbol.replace(".", "-"))      # BRK.B at Twelve Data is BRK-B at Yahoo


def dividends(symbols: list[str], dry_run: bool = True, max_requests: int | None = None) -> dict:
    """Each symbol's cash dividends (ex-date, amount on the split-adjusted basis of the stored bars), which a holder is
    paid on the ex-date: the bars are Twelve Data's, adjusted for its splits and not for dividends.

    The amounts are Yahoo Finance's (its chart events, the whole history in one request): Twelve Data's own /dividends
    keeps amounts before a recent split unadjusted (KLAC before its 2026 10:1), inflates old ones by splits that never
    happened (EWZ, 27x before 2008) and lists spin-offs as dividends, and its dividend-adjusted series inherits those
    errors. A distribution is paid unless the bars already carry it (CARRIED_BY_PRICES, reviewed event by event); any
    other dividend on the date of a Twelve Data split is paid and flagged for the same review. A symbol that never paid is stored empty; every symbol is asked again on a later run (two requests
    bring its whole history: Yahoo's events and Twelve Data's splits, TD_SPLITS_CREDITS credits)."""
    todo = list(dict.fromkeys(symbols))
    report = {"symbols": len(todo), "planned_requests": {"yahoo": len(todo), "twelvedata": len(todo)}}
    LOG.info("dividends: %s", report)
    if dry_run:
        return report
    per_min = env("TWELVEDATA_REQUESTS_PER_MINUTE")
    if not per_min:
        raise RuntimeError("set TWELVEDATA_REQUESTS_PER_MINUTE (the plan's limit) before asking the vendor")
    yahoo = Budget("yahoo", per_minute=60, max_requests=max_requests)
    td = Budget("twelvedata", per_minute=max(1, int(per_min) // TD_SPLITS_CREDITS), max_requests=max_requests)
    session, written, paid, rejected, carried, unreviewed = requests.Session(), 0, 0, [], [], []
    now = int(pd.Timestamp.now(tz="UTC").timestamp())
    for s in todo:
        yahoo.acquire()
        r = session.get(YAHOO_CHART.format(symbol=_yahoo_symbol(s)), headers={"User-Agent": "Mozilla/5.0"},
                        params={"period1": 0, "period2": now, "interval": "1d", "events": "div,split"}, timeout=60)
        _ledger("yahoo", status=r.status_code, symbol=s, what="dividends")
        if r.status_code == 429:
            raise RateLimited("yahoo answered 429: stop")
        result = (r.json().get("chart") or {}).get("result") if r.status_code == 200 else None
        if not result:
            LOG.warning("td:%s: yahoo has no chart for %s (HTTP %s): its dividends are not stored", s, _yahoo_symbol(s),
                        r.status_code)
            rejected.append(s)
            continue
        events = result[0].get("events") or {}
        day = lambda v: pd.Timestamp(v["date"], unit="s").normalize()             # noqa: E731
        got = pd.DataFrame({"amount": [float(v["amount"]) for v in (events.get("dividends") or {}).values()]},
                           index=pd.DatetimeIndex([day(v) for v in (events.get("dividends") or {}).values()], name="ex_date"))
        got = got[~got.index.duplicated(keep="last")].sort_index()
        try:
            td_splits = {pd.Timestamp(x["date"]): x.get("description") for x in _td_call(
                session, TD_SPLITS_API, {"symbol": s, "start_date": "1970-01-01"}, td,
                {"symbol": s, "what": "splits"}).get("splits") or []}
        except SymbolRejected as e:
            LOG.warning("td:%s: twelvedata rejects the symbol for splits (%s): its dividends are not stored", s, str(e)[:80])
            rejected.append(s)
            continue
        in_prices = [t for t in got.index if (s, str(t.date())) in CARRIED_BY_PRICES]
        for t in in_prices:
            LOG.info("td:%s: the %s dividend of %.4f is left out: the prices carry it (%s)", s, t.date(),
                     got.loc[t, "amount"], CARRIED_BY_PRICES[(s, str(t.date()))])
            carried.append(f"{s} {t.date()}")
        for t in got.index:
            if t in td_splits and t not in in_prices and (s, str(t.date())) not in PAID_ON_SPLIT_DATE:
                LOG.warning("td:%s: the %s dividend of %.4f falls on a Twelve Data split (%s) and is paid: not reviewed "
                            "yet whether the prices already carry it (CARRIED_BY_PRICES)", s, t.date(),
                            got.loc[t, "amount"], td_splits[t])
                unreviewed.append(f"{s} {t.date()}")
        got = got.drop(index=in_prices)
        path = dividends_path("td", s)
        path.parent.mkdir(parents=True, exist_ok=True)
        got.to_parquet(path)
        written += 1
        paid += int(len(got) > 0)
    report.update({"written": written, "paying": paid, "rejected": rejected, "left_out_as_carried_by_prices": carried,
                   "paid_on_a_split_date_unreviewed": unreviewed,
                   "requests_made": {"yahoo": yahoo.used, "twelvedata": td.used}})
    LOG.info("dividends: %s", report)
    return report


def _bars_between(s: str, tf: str, lo: pd.Timestamp, hi: pd.Timestamp) -> int:
    """Bars a series can have in [lo, hi): NYSE sessions (seven hourly bars each) for a US listing, weekdays (24 hours
    each) for an FX pair, metal or crude."""
    if hi <= lo:
        return 0
    if "/" in s:
        days = int(((pd.bdate_range(lo.tz_convert(None), hi.tz_convert(None))).size))
        return days * (24 if tf == "1h" else 1)
    return len(cal.nyse_sessions(lo, hi)) * (7 if tf == "1h" else 1)


# No US market or FX week has paused longer than 7 days since 1970 (the NYSE after 2001-09-11): a longer hole in a
# series is missing data.
MAX_HOLE = pd.Timedelta(days=10)


def _continuous_to(bars: pd.DataFrame, first: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    """The bars from the last hole longer than MAX_HOLE before `first` on, and how many were left out: early vendor
    history is sparse in places (a few 1975 bars, then nothing until 1984), and a position held across such a hole
    would book years of price change as one bar's return."""
    before = bars.index[bars.index <= first]
    gaps = before.to_series().diff()
    holes = gaps[gaps > MAX_HOLE]
    if holes.empty:
        return bars, 0
    start = holes.index.max()
    return bars[bars.index >= start], int((bars.index < start).sum())


def extend_twelvedata(pairs: list[tuple[str, str]], dry_run: bool = True, max_requests: int | None = None) -> dict:
    """History before the first stored bar, back to the vendor's first one (TD_HISTORY_START), for (symbol, tf) pairs.

    One window per series, paged backwards; it ends one bar after the stored first bar, which is fetched again as the
    seam (the bar after it gives the seam its close time: an hourly bar closes where the next one starts). A seam that
    disagrees (a split or a restatement since the series was stored) re-fetches the whole series instead of stitching
    two adjustments. The meta records the start asked from, so a series whose earlier history the vendor does not have
    is not asked again. The dry run counts pages from TD_HISTORY_START on the trading calendar: a ceiling."""
    now = pd.Timestamp.now(tz="UTC")
    todo, done_before, missing = [], [], []
    for s, tf in pairs:
        if tf not in TD_INTERVAL or not store.path("td", tf, s).exists():
            missing.append(f"{s}:{tf}")
            continue
        meta = store.read_meta("td", tf, s)
        start = pd.Timestamp(_td_start(s, tf), tz="UTC")
        if meta.get("vendor_rejected") or meta.get("daily_from") or (
                meta.get("history_asked_from") and pd.Timestamp(meta["history_asked_from"], tz="UTC") <= start):
            done_before.append(f"{s}:{tf}")          # an ETF's daily bars: `etf_daily_reconciled` decides them
            continue
        first = store.read_bars("td", tf, s).index.min()
        todo.append((s, tf, first, max(1, math.ceil(_bars_between(s, tf, start, first) / TD_PAGE))))
    if missing:
        LOG.warning("twelvedata history: %d series not stored or not fetched from the vendor, skipped: %s", len(missing),
                    ", ".join(missing[:10]))
    report = {"series": len(todo), "planned_requests_at_most": sum(t[3] for t in todo),
              "asked_before_skipped": len(done_before), "not_stored_skipped": len(missing)}
    LOG.info("twelvedata history: %s", report)
    if dry_run:
        return report
    per_min = env("TWELVEDATA_REQUESTS_PER_MINUTE")
    if not per_min:
        raise RuntimeError("set TWELVEDATA_REQUESTS_PER_MINUTE (the plan's limit) before a real run")
    ceiling = max_requests or (int(env("TWELVEDATA_MAX_REQUESTS_PER_RUN")) if env("TWELVEDATA_MAX_REQUESTS_PER_RUN") else None)
    budget = Budget("twelvedata", per_minute=int(per_min), max_requests=ceiling)
    session = requests.Session()
    extended, nothing_earlier, refetched, rejected, added = [], [], [], [], 0
    for s, tf, first, _ in todo:
        meta = store.read_meta("td", tf, s)
        try:
            new = _td_full(session, s, tf, budget, now, until=first + TF_DELTA[tf] - pd.Timedelta(seconds=1))
        except SymbolRejected as e:
            LOG.warning("td:%s %s: the vendor rejects the symbol, skipped from now on: %s", s, tf, str(e)[:120])
            store.write_meta("td", tf, s, {**meta, "vendor_rejected": {"at": str(now), "message": str(e)[:300]}})
            rejected.append(f"{s}:{tf}")
            continue
        old = store.read_bars("td", tf, s)
        seam = old.index[:1].intersection(new.index)
        if len(seam) and not (abs(old.loc[seam, "close"] / new.loc[seam, "close"] - 1) < 1e-6).all():
            LOG.warning("td:%s %s: the stored first bar disagrees with the vendor's (split or restatement since): "
                        "re-fetching the whole series", s, tf)
            if _replace_whole_series(s, tf, _td_full(session, s, tf, budget, now), old):
                refetched.append(f"{s}:{tf}")
        else:
            if not len(seam):
                LOG.info("td:%s %s: the vendor's window does not contain the stored first bar %s: the history is joined "
                         "without a seam check", s, tf, first)
            merged, bad = integrity.clean_bars(pd.concat([new[new.index < first], old]))
            if bad:
                LOG.warning("td:%s %s: dropped invalid bars when joining the earlier history %s", s, tf, bad)
            merged, holed = _continuous_to(merged, first)
            if holed:
                LOG.warning("td:%s %s: %d earlier bars before a hole of more than %d days left out, history from %s", s,
                            tf, holed, MAX_HOLE.days, merged.index.min().date())
            n_new = int((merged.index < first).sum())
            if n_new:
                store.write_bars("td", tf, s, merged)
                if tf == "1h":
                    _hourly_written(s)
                added += n_new
                extended.append(f"{s}:{tf} from {merged.index.min().date()} (+{n_new} bars)")
            else:
                nothing_earlier.append(f"{s}:{tf}")
        meta = store.read_meta("td", tf, s)
        store.write_meta("td", tf, s, {**meta, "history_asked_from": str(pd.Timestamp(_td_start(s, tf), tz="UTC")),
                                       "first": str(store.read_bars("td", tf, s).index.min())})
    report.update({"series_extended": len(extended), "bars_added": added, "vendor_has_nothing_earlier": len(nothing_earlier),
                   "series_refetched": refetched, "rejected": rejected, "requests_made": budget.used,
                   "extended": extended})
    return report


def held_by_lists(universes: list[str], tfs=("1d", "1h")) -> list[str]:
    """The TwelveData symbols these lists hold at some point (a ranked list: only the names that ever take a seat)."""
    from strategy_lab.evaluate import _load
    out = []
    for u in universes:
        for tf in tfs:
            _, panel, member = _load(u, tf, None, None)
            ids = panel.ids if member is None else [i for i in member.columns if member[i].any()]
            out += [i.split(":", 1)[1] for i in ids if i.startswith("td:")]
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vendor", choices=["binance", "binance-gaps", "binance-seed", "binance-delistings", "binance-extend",
                                       "binance-archive", "dukascopy-extend", "dukascopy-fill", "forexite-fill", "cme-hourly", "etf-daily", "sharadar-tables", "sharadar-stocks", "databento-hourly", "twelvedata",
                                       "twelvedata-seed", "twelvedata-extend", "twelvedata-first-bars",
                                       "dividends"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--markets", nargs="*", default=["perp", "spot"])
    ap.add_argument("--timeframes", nargs="*", default=["1h", "4h", "1d"])
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--max-usd", type=float, help="databento-hourly: the most a run may spend (a real run needs it)")
    ap.add_argument("--rebuild", action="store_true", help="databento-hourly: rebuild every listing's Databento hours "
                    "from the minutes on disk too; dukascopy-extend: build the part before the vendor's first hourly "
                    "bar anew from the months on disk")
    ap.add_argument("--refetch", nargs="*", help="SYMBOL:TF pairs to re-fetch whole (twelvedata)")
    ap.add_argument("--universes", nargs="*", help="twelvedata-extend, twelvedata-first-bars: the TwelveData series "
                    "these universes use (twelvedata-first-bars: the hourly series of every agreed list by default)")
    args = ap.parse_args()
    log.setup(f"refresh_{args.vendor}")
    if args.vendor == "binance":
        rep = refresh_binance(args.markets, args.timeframes, args.dry_run, args.max_requests, args.symbols)
    elif args.vendor == "binance-seed":
        rep = seed_binance_series(args.symbols or [], args.timeframes, args.markets[0], args.dry_run, args.max_requests)
    elif args.vendor == "binance-delistings":
        rep = mark_delistings(args.timeframes, args.dry_run)
    elif args.vendor == "binance-extend":
        rep = extend_binance(args.timeframes, args.dry_run, args.max_requests)
    elif args.vendor == "databento-hourly":
        rep = databento_hourly(args.symbols, args.dry_run, args.max_usd, args.rebuild)
    elif args.vendor == "sharadar-tables":
        rep = refresh_sharadar_tables(args.dry_run)
    elif args.vendor == "sharadar-stocks":
        spans = sharadar.sp500_spans()
        members = sorted(set(spans.loc[spans["end"].isna() | (spans["end"] >= SP500_SINCE), "ticker"]))
        rep = fetch_sharadar_stocks(args.symbols or members, args.dry_run, args.max_requests)
        if not args.dry_run:
            rep["store"] = store_sharadar_stocks(args.symbols or members)
            rep["hourly"] = store_sharadar_hourly(args.symbols or members)
    elif args.vendor == "dukascopy-extend":
        from strategy_lab.universes import FX_MAJORS
        rep = extend_from_dukascopy(args.symbols or FX_MAJORS, args.dry_run, args.max_requests, args.rebuild)
    elif args.vendor == "cme-hourly":
        rep = fetch_cme_hourly(args.symbols or list(CME_PRODUCTS), args.dry_run, args.max_usd)
        if not args.dry_run:
            rep["stored"] = store_cme_series(args.symbols or list(CME_PRODUCTS))
    elif args.vendor == "dukascopy-fill":
        from strategy_lab.universes import FX_MAJORS, STOCKHUNT_COMMODITIES
        rep = fill_from_dukascopy(args.symbols or FX_MAJORS + STOCKHUNT_COMMODITIES, args.dry_run, args.max_requests)
    elif args.vendor == "etf-daily":
        from strategy_lab.universes import resolve
        etfs = args.symbols or [i.split(":", 1)[1] for i in resolve("etf_core", "1d").ids]
        rep = fetch_sharadar_funds(etfs, args.dry_run, args.max_requests)
        if not args.dry_run:
            rep["stored"] = etf_daily_reconciled(etfs)
    elif args.vendor == "forexite-fill":
        from strategy_lab.universes import FX_MAJORS
        rep = fill_from_forexite(args.symbols or FX_MAJORS, args.dry_run, args.max_requests)
    elif args.vendor == "binance-archive":
        rep = seed_binance_from_archive(args.symbols or [], args.timeframes, args.markets[0], args.dry_run,
                                        args.max_requests)
    elif args.vendor == "binance-gaps":
        rep = fill_binance_gaps(args.markets, args.timeframes, args.dry_run, args.max_requests, args.symbols)
    elif args.vendor == "twelvedata-seed":
        rep = seed_twelvedata_series(args.symbols or [], args.timeframes, args.dry_run, args.max_requests)
    elif args.vendor == "twelvedata-extend":
        tfs = [tf for tf in args.timeframes if tf in TD_INTERVAL]
        pairs = [(s, tf) for s in (args.symbols or []) for tf in tfs]
        if args.universes:
            from strategy_lab.universes import resolve
            pairs += [(i.split(":", 1)[1], tf) for u in args.universes for tf in tfs for i in resolve(u, tf).ids
                      if i.startswith("td:")]
        rep = extend_twelvedata(sorted(set(pairs)), args.dry_run, args.max_requests)
    elif args.vendor == "twelvedata-first-bars":
        from strategy_lab import lists
        from strategy_lab.universes import resolve
        names = args.universes or lists.names(lists.OURS) + list(lists.ML_TASK)
        syms = list(args.symbols or []) + [i.split(":", 1)[1] for u in dict.fromkeys(names)
                                           for i in resolve(u, "1h").ids if i.startswith("td:")]
        rep = twelvedata_first_bars(syms, args.dry_run, args.max_requests)
    elif args.vendor == "dividends":
        from strategy_lab import lists
        names = args.universes or [x.universe for x in lists.OURS if x.market in ("Stocks", "ETFs")] + [
            u for u in lists.ML_TASK if u.startswith("us_stocks")]
        rep = dividends(list(args.symbols or []) + held_by_lists(names), args.dry_run, args.max_requests)
    elif args.refetch:
        rep = refetch_twelvedata([tuple(x.split(":")) for x in args.refetch])
    else:
        syms = args.symbols or [s.replace("-", "/") if len(s) == 7 and s[3] == "-" else s for s in store.symbols("td", "1d")]
        rep = refresh_twelvedata(syms, args.dry_run, args.max_requests)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
