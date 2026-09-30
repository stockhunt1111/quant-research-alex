"""Named universes. Stocks and crypto are chosen point-in-time by liquidity, never by today's leaders.

`resolve(name, timeframe)` returns the candidate instrument ids and a membership rule. Membership at bar t uses
the trailing median dollar volume up to the previous bar and is re-ranked once a month, so a name enters the
universe only after it became liquid and leaves when it stops trading.

`stockhunt_*` are the markets the firm's research desk (Stockhunt) scores strategies on, with its own instrument
lists: its top-100 US stocks, ten ETFs, twenty coins and five commodities (its fifth market, nineteen CME futures,
needs a futures data vendor this store does not have).

`us_stocks_mcapN` and `crypto_mcapN` are today's N largest by market capitalisation (data/reference snapshots,
strategy_lab.data.market_cap), each over its whole history: the names a user of the firm's products picks today.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd

from strategy_lab import log
from strategy_lab.config import REFERENCE_DIR
from strategy_lab.data import calendars as cal
from strategy_lab.data import market_cap, sharadar, store
from strategy_lab.data.bars import Panel, liquidity, load_panel
from strategy_lab.data.instruments import COMMODITIES, parse
from strategy_lab.engine.backtest import ended

LOG = log.get("universes")

ETF_CORE = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE",
            "SMH", "KRE", "GLD", "SLV", "GDX", "USO", "UNG", "DBC", "DBA", "CPER", "TLT", "IEF", "SHY", "HYG",
            "LQD", "TIP", "EEM", "EFA", "EWZ", "FXI"]
FX_MAJORS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]
STABLECOINS = {"USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "USDPUSDT", "DAIUSDT", "EURUSDT", "USD1USDT"}
# Tokens of gold: a claim on bullion, priced as gold, not a crypto asset. Binance classes them as coins (its RWA
# subtype also covers crypto tokens such as MANTRA and CFG), so they are named here.
COMMODITY_TOKENS = {"PAXGUSDT", "XAUTUSDT"}
# Binance's index perps delisted before its contract list was taken, so their underlying type is not on record there:
# its BLUEBIRD and FOOTBALL indices (the listed ones, DEFI and BTCDOM, are excluded by their type)
INDEX_PERPS = {"BLUEBIRDUSDT", "FOOTBALLUSDT"}

# Stockhunt's lists. Stocks: every name its point-in-time top 100 has held since 2003.
STOCKHUNT_STOCKS = [
    "A", "AA", "AAL", "AAPL", "ABBV", "ABNB", "ABT", "ACN", "ADBE", "ADI", "ADP", "AES", "AIG", "AKAM", "ALL",
    "AMAT", "AMD", "AMGN", "AMZN", "ANET", "ANF", "APA", "APC", "APP", "ATI", "AVGO", "AXP", "AZO", "BA", "BAC",
    "BAX", "BBY", "BIIB", "BKNG", "BMY", "BNY", "BRK.B", "BSX", "C", "CAH", "CAT", "CCL", "CF", "CHTR", "CI",
    "CIEN", "CL", "CLF", "CLX", "CMCSA", "CME", "CMG", "CNX", "COF", "COIN", "COP", "COST", "CPRI", "CRM", "CRWD",
    "CSCO", "CSX", "CVNA", "CVS", "CVX", "DAL", "DD", "DE", "DELL", "DG", "DHR", "DIS", "DVN", "EA", "EBAY", "ELV",
    "EMR", "ENPH", "EOG", "F", "FCX", "FDX", "FFIV", "FIS", "FISV", "FITB", "FLR", "FSLR", "GD", "GE", "GEN", "GEV",
    "GILD", "GLW", "GOOG", "GOOGL", "GPS", "GS", "GT", "HAL", "HD", "HIG", "HOOD", "HPQ", "IBM", "ICE", "INTC",
    "INTU", "IP", "ISRG", "JBL", "JCI", "JNJ", "JPM", "KBH", "KMB", "KMI", "KO", "KR", "KSS", "LEN", "LIN", "LLY",
    "LMT", "LOW", "LRCX", "LUV", "M", "MA", "MAT", "MCD", "MCHP", "MCK", "MDLZ", "MDT", "MET", "META", "MMM", "MO",
    "MOS", "MPC", "MRK", "MRNA", "MRSH", "MS", "MSFT", "MSI", "MU", "NBR", "NCLH", "NEE", "NEM", "NFLX", "NKE",
    "NOC", "NOV", "NOW", "NTAP", "NUE", "NVDA", "OMC", "ORCL", "OXY", "PANW", "PAYX", "PEP", "PFE", "PG", "PHM",
    "PLTR", "PM", "PNC", "PSKY", "PYPL", "QCOM", "RCL", "REGN", "RIG", "RTX", "SBUX", "SCHW", "SLB", "SMCI", "SPGI",
    "STT", "T", "TER", "TGT", "THC", "TJX", "TMO", "TMUS", "TRV", "TSLA", "TXN", "UAL", "UBER", "UNH", "UNP", "UPS",
    "USB", "V", "VLO", "VRSN", "VRTX", "VTRS", "VZ", "WAT", "WDC", "WFC", "WM", "WMT", "WYNN", "XOM", "YUM", "ZBH",
]
STOCKHUNT_ETFS = ["SPY", "QQQ", "IWM", "TLT", "XLF", "GLD", "EFA", "DIA", "XLV", "XLE"]
STOCKHUNT_COINS = ["BTC", "ETH", "BNB", "XRP", "SOL", "DOGE", "ADA", "TRX", "AVAX", "LINK", "XMR", "BCH", "XLM", "VET",
                   "DOT", "ATOM", "HBAR", "UNI", "XTZ", "LTC"]
STOCKHUNT_COMMODITIES = ["XAU/USD", "XAG/USD", "XPT/USD", "XPD/USD", "WTI/USD"]
# the CME futures of the commodities, their contracts joined across rolls: a market of their own beside the spot
# quotes, as the firm's research desk keeps its CME futures (user, 2026-09-26: copper, which has no spot quote at
# Twelve Data, and futures for the thin spot quotes of platinum, palladium and crude)
CME_FUTURES = ["cme:GC", "cme:SI", "cme:PL", "cme:PA", "cme:CL", "cme:HG"]


def _td(timeframe: str) -> set[str]:
    return set(store.symbols("td", timeframe))


def fx_all(timeframe: str) -> list[str]:
    """Every stored currency pair; metals and crude are spelled like pairs (XAU/USD) and are not currencies."""
    pairs = (s.replace("-", "/") for s in _td(timeframe) if len(s) == 7 and s[3] == "-")
    return sorted(p for p in pairs if p not in COMMODITIES)


def _reference(name: str) -> pd.DataFrame:
    p = REFERENCE_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: README (data commands) names the command that writes it")
    return pd.read_csv(p)


SP500_SINCE = pd.Timestamp("1996-01-01", tz="UTC")   # the stock lists' candidates: in the index at any time since


def sp500_membership() -> pd.DataFrame:
    """Historical S&P 500 membership spans as Sharadar keys them (`data.sharadar`): ticker, start, end (NaT = still a
    member). Its tickers are those its prices are stored under (`sh:<ticker>`), a company that no longer trades
    carrying a suffix when its ticker was taken up again (WB1 Wachovia, WB Weibo)."""
    return sharadar.sp500_spans()


def stock_candidates(timeframe: str) -> list[str]:
    """The S&P 500's members since SP500_SINCE with Sharadar bars of the timeframe in the store (`sh:<ticker>`), the
    companies that no longer trade among them (Enron, Lehman, Wachovia): Twelve Data refuses those, and serves some of
    their tickers as the companies that took them up since."""
    m = sp500_membership()
    members = set(m.loc[m["end"].isna() | (m["end"] >= SP500_SINCE), "ticker"])
    return sorted(s for s in store.symbols("sh", timeframe) if s in members)


def sp500_member_mask(panel: Panel) -> pd.DataFrame:
    """True where the instrument was an S&P 500 member on that bar's date. Worked out once per panel: the lists next to
    a stock list (Top-80 and Top-120 around Top-100) read their seats from the same candidates."""
    got = panel.memo.get("sp500_member_mask")
    if got is None:
        m = sp500_membership()
        got = pd.DataFrame(False, index=panel.index, columns=panel.ids)
        for inst in panel.ids:
            spans = m[m["ticker"] == inst.split(":", 1)[1]]
            for _, r in spans.iterrows():
                end = r["end"] if pd.notna(r["end"]) else panel.index.max()
                got.loc[(panel.index >= r["start"]) & (panel.index <= end), inst] = True
        panel.memo["sp500_member_mask"] = got
    return got.copy()


def coverage_note(candidates: list[str]) -> str:
    """How many of the index's members since SP500_SINCE the candidates cover."""
    m = sp500_membership()
    members = set(m.loc[m["end"].isna() | (m["end"] >= SP500_SINCE), "ticker"])
    missing = sorted(members - {c.split(":", 1)[1] for c in candidates})
    return (f"{len(members) - len(missing)} of the {len(members)} S&P 500 members since {SP500_SINCE.year}, those that "
            f"no longer trade included (Sharadar's bars, from 1997-12-31); without bars: {', '.join(missing) or 'none'}")


MIN_MEMBER_SESSION_COVERAGE = 0.98


@lru_cache(maxsize=2)
def gappy_members(source: str) -> dict[str, float]:
    """S&P 500 members whose daily bars of a source miss more than 2% of NYSE sessions while they were in the index,
    with the share covered. A US listing prints every session; such a series is a recycled ticker or a broken one."""
    m = sp500_membership()
    sched = cal.nyse_sessions(m["start"].min(), pd.Timestamp.now(tz="UTC"))
    sessions = pd.DatetimeIndex(sched.index)
    out = {}
    for sym in sorted(set(m["ticker"]) & set(store.symbols(source, "1d"))):
        d = store.read_bars(source, "1d", sym)
        days = (d.index - pd.Timedelta(seconds=1)).tz_convert(cal.NY).tz_localize(None).normalize()
        want = got = 0
        for r in m[m["ticker"] == sym].itertuples(index=False):
            lo = max(r.start.tz_convert(None).normalize(), days.min())
            hi = days.max() if pd.isna(r.end) else min(r.end.tz_convert(None).normalize(), days.max())
            span = sessions[(sessions >= lo) & (sessions <= hi)]
            want += len(span)
            got += int(span.isin(days).sum())
        if want and got / want < MIN_MEMBER_SESSION_COVERAGE:
            out[sym] = got / want
    return out


def crypto_candidates(timeframe: str) -> list[str]:
    """USD-M perps on crypto only. Binance also lists stock, commodity, index and FX perps ("TradFi"); those are
    excluded by the exchange's own underlying type, stablecoins and tokens of gold by name. Delisted symbols absent
    from today's contract list were all crypto (TradFi perps are recent and still trading), so they stay."""
    contracts = _reference("binance_perp_contracts.csv")
    not_crypto = set(contracts.loc[contracts["underlying_type"] != "COIN", "symbol"])
    return sorted(s for s in store.symbols("perp", timeframe)
                  if s not in STABLECOINS and s not in COMMODITY_TOKENS and s not in INDEX_PERPS and s not in not_crypto)


def coiniq_coins(timeframe: str) -> list[str]:
    """Binance perps standing in for CoinIQ's markets: Orderly's plain markets whose Binance perp prints the same coin
    (data/reference/orderly_vs_binance.csv, from `python -m strategy_lab.data.orderly compare`), crypto only."""
    match = _reference("orderly_vs_binance.csv")
    same = set(match.loc[match["same_coin"] == True, "binance"].dropna())            # noqa: E712
    return [s for s in crypto_candidates(timeframe) if s in same]


LIST_BUFFER = 1.5      # a member keeps its seat until it ranks below 1.5 x n, a newcomer inside n / 1.5 takes one at once


def top_liquid(panel: Panel, n: int, lookback_days: int = 60, eligible: pd.DataFrame | None = None,
               band: float = LIST_BUFFER) -> pd.DataFrame:
    """Membership mask: the n instruments with the highest trailing median dollar volume per day, re-ranked monthly.

    A member keeps its seat while it ranks within `band` x n and still trades; a seat it gives up goes to the
    best-ranked instrument outside that trades, so the list holds n names (fewer only while fewer trade: a name
    whose median is zero, a market that stopped, never fills an empty seat). A buffer zone both ways, as index
    providers use: a name at the edge no longer flips in and out every month, a newcomer ranked inside the top n
    waits for a seat, and one ranked inside n / `band` takes one at once, from the member ranked lowest. With the
    buffer on the way out alone, a name ranked sixth waited for months outside a Top-100 while its members ranked 101st
    to 150th kept their seats (YHOO at the re-pick of 1999-12-31; 1000SHIB 14th and ICP 15th-20th of the crypto
    Top-100 for two and three re-picks), and a narrower list held names its wider one did not.

    The median runs over the last `lookback_days` calendar days, so the window means the same on every timeframe;
    a day on which other instruments traded and this one did not counts as zero, so a name that stops trading
    falls out at a following re-rank. An instrument competes once its first bar is at least half that window old.
    With `eligible`, only instruments eligible on the ranking day compete for the n slots, and a member that stops
    being eligible (a stock leaving the S&P 500) gives up its seat on its first day out to the best-ranked eligible
    name outside, as a delisted market does (its seat had stood empty to the month's end: stocks Top-100 held 97-99
    names on 2.3% of its days). A member keeps its slot through bars it missed until the next re-rank; a market that is
    delisted (`engine.backtest.ended`, where a backtest closes the position: for good, or until it is listed anew) gives
    up its seat the day after its last trade, and the best-ranked name outside that trades takes it then, the other
    seats staying as they are until the re-rank; a market listed anew competes again from the re-rank after it.
    """
    dv = panel.dollar_volume
    day = (dv.index - pd.Timedelta(microseconds=1)).normalize()
    med = liquidity(panel, lookback_days)
    if eligible is not None:
        ok = eligible.groupby(day).max().reindex(med.index).fillna(False).astype(bool)
        med = med.where(ok)
    month_end = set(med.groupby(med.index.tz_localize(None).to_period("M")).tail(1).index)
    if len(med) and med.index[-1].day != med.index[-1].days_in_month:
        month_end.discard(med.index[-1])        # the data's last day ends no month: its month goes on after it
    stopped = ended(panel)
    off = stopped.groupby(day).max().reindex(med.index).fillna(False).astype(bool)   # a day with a delisted stretch
    # a market's seat changes hands at the end of the last day it traded before each delisting
    stops = {}
    for c in stopped.columns:
        s = stopped[c].to_numpy()
        for k in np.flatnonzero(s & ~np.r_[False, s[:-1]]):
            before = panel.close[c].iloc[:k].last_valid_index()
            if before is not None:
                stops.setdefault(day[panel.index.get_loc(before)], set()).add(c)
    if eligible is not None:                                # ... and a member's at the end of its last eligible day
        okv = ok.to_numpy()
        for k, j in zip(*np.nonzero(okv[:-1] & ~okv[1:])):
            stops.setdefault(med.index[k], set()).add(med.columns[j])
    held: list = []
    seats, dates = [], sorted(month_end | set(stops))
    for d in dates:
        nxt = off.index[off.index > d]
        gone = stops.get(d, set()) | (set(off.columns[off.loc[nxt[0]].to_numpy()]) if len(nxt) else set())
        ranked = med.loc[d].drop(list(gone & set(med.columns))).dropna().sort_values(ascending=False, kind="stable")
        order = list(zip(ranked.index, ranked.to_numpy()))
        if d in month_end:
            was = set(held)
            # ranked inside n / band, a name holds a seat whatever held it before; a member keeps its own while it
            # ranks inside band x n, the lowest-ranked of them giving theirs up to the first
            core = [c for rank, (c, v) in enumerate(order, 1) if rank <= n / band and v > 0]
            inner = set(core)
            keep = (core + [c for rank, (c, v) in enumerate(order, 1)
                            if c in was and c not in inner and rank <= band * n and v > 0])[:n]
        else:                                               # a market stopped or left: only its seat changes hands
            keep = [c for c in held if c not in gone]
        kept = set(keep)
        held = keep + [c for c, v in order if c not in kept and v > 0][:n - len(keep)]
        seats.append(med.columns.isin(held))
    chosen = pd.DataFrame(seats, index=pd.DatetimeIndex(dates), columns=med.columns).astype(float)
    # decided at the end of a day, in force from the next day's first instant: on every bar that closes on that day or
    # later, the bar closing at that instant included (a coin's bar of a month's last hour or day, whose close is a
    # monthly strategy's decision at the month's turn; mapped by the day a bar's trading belongs to, that decision saw
    # the month before's list)
    chosen.index = chosen.index + pd.Timedelta(days=1)
    close_day = dv.index.normalize()
    mask = chosen.reindex(chosen.index.union(close_day.unique())).ffill().reindex(close_day).fillna(0.0)
    mask.index = dv.index
    return mask.astype(bool) & panel.started & ~stopped


def largest_stocks(n: int, timeframe: str) -> tuple[list[str], tuple[str, ...]]:
    """Today's n largest S&P 500 members by market capitalisation that the store has, one share class per company
    (GOOGL, not also GOOG), on Sharadar's bars as the stock lists are (user, 2026-09-27): one company is one instrument
    whichever list it is taken from."""
    caps = market_cap.load("stocks")
    m = sp500_membership()
    current = set(m.loc[m["end"].isna(), "ticker"])
    have = set(store.symbols("sh", timeframe))
    company = caps["name"].str.split(" Class ").str[0].str.replace(" Common Stock", "", regex=False)
    members = caps["symbol"].isin(current)
    ranked = caps[members][~company[members].duplicated()]
    picked = [s for s in ranked["symbol"] if store.safe_name(s) in have][:n]
    smallest = float(ranked.loc[ranked["symbol"] == picked[-1], "market_cap"].iloc[0]) if picked else 0.0
    larger_outside = caps[(caps["market_cap"] > smallest) & ~company.isin(set(company[members]))]["symbol"].tolist()
    return [f"sh:{s}" for s in picked], (
        f"the {n} largest S&P 500 members by market capitalisation on {caps['as_of'].iloc[0]} (Nasdaq's screener), one "
        "share class per company, each over its whole history: today's largest companies grew to be so, so long-only "
        "results are biased upward",
        f"larger US listings outside the S&P 500 (foreign ADRs, recent listings) are not US stocks here: "
        f"{', '.join(larger_outside) or 'none'}")


def largest_coins(n: int, timeframe: str) -> tuple[list[str], tuple[str, ...]]:
    """Today's n largest coins by market capitalisation (CoinGecko) that trade as a Binance USD-M perp, each over its
    whole perp history: every crypto list here trades perps, as the firm's CoinIQ does, so a short is the perp's and
    its carry the funding (on spot a short is a margin loan, which the firm does not trade); buy-and-hold holds the
    coins on spot where they have a market, as the other lists' does. Stablecoins and tokens of gold are not coins to
    trade; a coin with no current perp is skipped, and the next one by market cap takes its place."""
    caps = market_cap.load("coins")
    perps = set(crypto_candidates("1d"))
    have = set(store.symbols("perp", timeframe))
    end = store.read_bars("perp", "1d", "BTCUSDT").index.max()
    picked, skipped = [], []
    for r in caps.itertuples(index=False):
        pair = f"{r.symbol}USDT"
        if pair in STABLECOINS or ("USD" in r.symbol and abs(r.price - 1.0) < 0.03):
            skipped.append(f"{r.symbol} (stablecoin)")
            continue
        if pair in COMMODITY_TOKENS:
            skipped.append(f"{r.symbol} (a token of gold)")
            continue
        if pair not in perps:
            skipped.append(f"{r.symbol} (no Binance perp)")
            continue
        d = store.read_bars("perp", "1d", pair)
        if end - d.index.max() > pd.Timedelta(days=7):
            skipped.append(f"{r.symbol} (its Binance perp stopped on {d.index.max().date()})")
            continue
        if pair in have:
            picked.append(f"perp:{pair}")
        if len(picked) == n:
            break
    return picked, (
        f"the {n} largest coins by market capitalisation on {caps['as_of'].iloc[0]} (CoinGecko) with a Binance USD-M "
        "perp, each over its whole perp history: today's largest coins survived to be so, so long-only results are "
        "biased upward",
        f"skipped on the way down the ranking: {', '.join(skipped) or 'none'}")


@dataclass(frozen=True)
class Universe:
    name: str
    ids: list[str]
    member: Callable[[Panel], pd.DataFrame] | None = None
    notes: tuple[str, ...] = ()
    size: int | None = None           # instruments held when full: a top-N list's N; None = all its ids

    @property
    def capacity(self) -> int:
        """The instruments it holds when full."""
        return self.size or len(self.ids)


def resolve(name: str, timeframe: str) -> Universe:
    if name == "etf_core":
        return Universe(name, [f"td:{s}" for s in ETF_CORE if s in _td(timeframe)])
    if name.startswith("etf_top"):                                  # the n most liquid of etf_core, as stocks rank
        n = int(name.removeprefix("etf_top"))
        return Universe(name, [f"td:{s}" for s in ETF_CORE if s in _td(timeframe)], lambda p: top_liquid(p, n), size=n)
    if name == "fx_majors":
        return Universe(name, [f"td:{s}" for s in FX_MAJORS if s in fx_all(timeframe)])
    if name == "fx_all":
        return Universe(name, [f"td:{s}" for s in fx_all(timeframe)])
    if name.startswith("us_stocks_mcap"):
        ids, notes = largest_stocks(int(name.removeprefix("us_stocks_mcap")), timeframe)
        return Universe(name, ids, None, notes)
    if name.startswith("crypto_mcap"):
        ids, notes = largest_coins(int(name.removeprefix("crypto_mcap")), timeframe)
        return Universe(name, ids, None, notes)
    if name.startswith("us_stocks_top"):
        n = int(name.removeprefix("us_stocks_top"))
        listed = stock_candidates(timeframe)
        gappy = {s: v for s, v in gappy_members("sh").items() if s in listed}
        ids = [f"sh:{s}" for s in listed if s not in gappy]
        def member(p):
            sp = sp500_member_mask(p)
            return top_liquid(p, n, eligible=sp) & sp
        dropped = ", ".join(f"{s} {v:.0%}" for s, v in sorted(gappy.items(), key=lambda kv: kv[1]))
        return Universe(name, ids, member, (coverage_note(ids),
                                            f"excluded, data covers too few NYSE sessions while in the index: "
                                            f"{dropped or 'none'}"), size=n)
    if name == "stockhunt_stocks":
        have, gappy = _td(timeframe), gappy_members("td")
        ids = [f"td:{s}" for s in STOCKHUNT_STOCKS if store.safe_name(s) in have and s not in gappy]
        dropped = ", ".join(f"{s} {v:.0%}" for s, v in sorted(gappy.items(), key=lambda kv: kv[1])
                            if s in STOCKHUNT_STOCKS)
        return Universe(name, ids, lambda p: top_liquid(p, 100), (
            "Stockhunt's names; which 100 are held on a date is decided here by trailing dollar volume, re-ranked "
            "monthly",
            "survivors mostly: the list has no Lehman, Wachovia, Merrill Lynch or Bear Stearns, so "
            "long-only results are biased upward",
            f"excluded, data covers too few NYSE sessions while in the index: {dropped}"), size=100)
    if name == "stockhunt_etfs":
        return Universe(name, [f"td:{s}" for s in STOCKHUNT_ETFS if s in _td(timeframe)])
    if name == "stockhunt_crypto":
        have = set(store.symbols("spot", timeframe))
        return Universe(name, [f"spot:{c}USDT" for c in STOCKHUNT_COINS if f"{c}USDT" in have], None,
                        ("Stockhunt's twenty coins priced with their Binance spot USDT pairs (the desk prices USD pairs "
                         "from Twelve Data); Binance delisted XMR's spot market on 2024-02-20, so from then on the "
                         "universe holds nineteen coins",))
    if name == "stockhunt_commodities":
        return Universe(name, [f"td:{s}" for s in STOCKHUNT_COMMODITIES if store.safe_name(s) in _td(timeframe)])
    if name == "cme_futures":
        return Universe(name, [i for i in CME_FUTURES if store.path(parse(i).source, timeframe, parse(i).symbol).exists()])
    if name == "coiniq":
        return Universe(name, [f"perp:{s}" for s in coiniq_coins(timeframe)], None,
                        ("CoinIQ's coins (Orderly markets) priced with the Binance perp of the same coin: daily closes "
                         "agree to a median of about 5 bps, with years more history",))
    if name.startswith("crypto_top"):
        n = int(name.removeprefix("crypto_top"))
        return Universe(name, [f"perp:{s}" for s in crypto_candidates(timeframe)], lambda p: top_liquid(p, n, 30), size=n)
    if ":" in name:                                                  # a single instrument, e.g. "td:SPY"
        return Universe(name, [name])
    raise ValueError(f"unknown universe {name!r}: etf_core, etf_topN, fx_majors, fx_all, us_stocks_topN, us_stocks_mcapN, "
                     "crypto_topN, crypto_mcapN, coiniq, stockhunt_stocks, stockhunt_etfs, stockhunt_crypto, "
                     "stockhunt_commodities, cme_futures, or an id")


def instruments_now(name: str, timeframe: str) -> tuple[list[str], tuple[str, ...]]:
    """The instruments a per-instrument test covers, and the notes that go with them: a fixed list as it is; a
    monthly-ranked universe (us_stocks_topN, crypto_topN) as its members at the latest daily bar — the names a user
    picks today, each then tested over its whole history. Ranked on the daily bars whatever the timeframe, as a list
    seats its names (`evaluate._seats`), so every timeframe tests the same names: ranked on each timeframe's own bars,
    whose intraday bars miss the auctions, the stocks' Top-100 held ADI and BSX on 1d only and BLK and GLW on 1h and 4h
    only (2026-09-29). A member without bars of the timeframe is left out, and named in the log."""
    uni = resolve(name, timeframe)
    if uni.member is None:
        return uni.ids, uni.notes
    daily = resolve(name, "1d")
    panel = load_panel(daily.ids, "1d", fields=("close", "dollar_volume"))
    now = daily.member(panel).iloc[-1]
    members = [i for i in panel.ids if bool(now[i])]
    stored = set(uni.ids)
    missing = [i for i in members if i not in stored]
    if missing:
        LOG.warning("%s %s: %d of today's %d members have no bars of the timeframe and are not tested: %s", name,
                    timeframe, len(missing), len(members), ", ".join(missing))
    note = (f"today's {len(members)} members of {name}, each over its whole history: names that are liquid today "
            "survived to be so, so long-only results are biased upward")
    return [i for i in members if i in stored], (*uni.notes, note)
