"""The market's index beside a result's buy & hold: what a client could have bought instead of the strategy, held over
the record's days.

A strategy is judged against buy & hold of what it trades, the same list or asset over the same days at equal risk
(`metrics.vs_hold`, the target "beats buy & hold"): that separates a rule's timing from what its list holds. The index
answers a client's question instead, whether simply buying the market would have done better, so each market shows
one, held as buy & hold holds one instrument (`engine.hold`: bought on the record's first day, one purchase's cost,
dividends paid on the ex-date, a coin on spot). A list's:

    Stocks, ETFs   SPY, with its dividends
    Crypto         BTC on Binance spot
    Commodities    gold's spot quote (XAU/USD), the kind of price the list's own quotes are
    CME futures    gold's future (GC), joined across its rolls: a future's return without the interest on its money,
                   as the list's own records and its buy & hold are
    FX             none: a currency pair is compared with cash already

An instrument alone has its own market's, where an index measures that market: SPY beside a US stock or an ETF of US
stocks, BTC beside a coin. Gold is one metal, not the commodities' market (its daily returns correlate 0.04 with
crude's since 2016; user, 2026-09-28: why is oil compared with holding gold?): a metal or a future alone, and an ETF
of bonds, commodities, gold miners or foreign stocks (TLT's daily returns correlate -0.15 with SPY's), has its own buy &
hold beside it and no index. WTI's spot quote cannot be held (a holder rolls futures every month, whose cost the spot
price does not show): beside it stands crude's future held across its rolls (CL, 0.89 with the quote), what a holder
of crude earned. An instrument that is its market's index has none: its buy & hold is the index. The index is a
reference, not a target, and it is not sized to the record's risk: a list result keeps its average exposure, not its
daily one, and the money test worked out from the average is off the exact one by a median 0.23% a year (0.6-0.7% on
the stock lists; the sign differs on 21 of the 1,032 list results that keep their buy & hold, measured 2026-09-28).
Its return, drawdown and Sharpe beside the record's make the comparison without that error.
"""
from __future__ import annotations

import functools
import threading
from dataclasses import dataclass

import pandas as pd

from strategy_lab import db, lists, log, metrics
from strategy_lab.data.bars import Panel, dividends_version, load_panel, stored_version
from strategy_lab.engine.hold import buy_and_hold

LOG = log.get("market_index")
# the index is held on its daily bars whatever a record's timeframe: its days come out the same (SPY's daily returns
# from its hourly bars are its daily bars' to the day), and its daily bars go back further (SPY's hourly ones to 2019)
TIMEFRAME = "1d"
LATE_DAYS = 7          # an index whose bars end this much before the record does is named in the log: its tail is flat


@dataclass(frozen=True)
class Index:
    id: str            # the instrument held
    name: str          # what the page calls it


BY_MARKET = {
    "Stocks": Index("td:SPY", "SPY"),
    "ETFs": Index("td:SPY", "SPY"),
    "Crypto": Index("spot:BTCUSDT", "BTC"),
    "Commodities": Index("td:XAU/USD", "Gold"),
    "CME futures": Index("cme:GC", "Gold future"),
}
# the ETFs of US stocks, whose market SPY measures (the Dow, the Nasdaq-100, small caps, the sectors); the others of
# `universes.ETF_CORE` hold bonds, commodities, gold miners or foreign stocks
US_STOCK_ETFS = {f"td:{s}" for s in ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB",
                                     "XLU", "XLRE", "SMH", "KRE")}
HELD_THROUGH = {"td:WTI/USD": Index("cme:CL", "WTI future")}    # a quote that cannot be held: what a holder of it holds
HELD_AS = {"perp:BTCUSDT": "spot:BTCUSDT"}     # an instrument alone whose buy & hold holds the index: a perp's coin on spot

_lock = threading.Lock()                       # the server answers from several threads; a panel's memo is not shared
_panels: dict[str, tuple[tuple, Panel]] = {}


def of(list_id: str, instrument_id: str | None) -> Index | None:
    """The index beside a result: its list's market's (`lists.market`); for an instrument alone, its own market's where
    an index measures that market (a stock, a coin, an ETF of US stocks: the market of the list it was run from,
    `lists.per_instrument_market`), or what a holder of a quote that cannot be held holds (`HELD_THROUGH`). None for
    FX, for a list outside ours, for an instrument that is the index and for one whose own buy & hold is its market."""
    market = lists.market(list_id) if instrument_id is None else lists.per_instrument_market(list_id)
    index = BY_MARKET.get(market)
    if index is None:
        if market not in ("FX", *BY_MARKET):
            LOG.debug("%s is not a list of ours: no index beside its results", list_id)
        return None
    if instrument_id is None:
        return index
    if market in ("Stocks", "Crypto") or (market == "ETFs" and instrument_id in US_STOCK_ETFS):
        return None if HELD_AS.get(instrument_id, instrument_id) == index.id else index
    return HELD_THROUGH.get(instrument_id)


def _panel(index_id: str) -> Panel:
    """The index's daily bars, read again once its stored bars or its dividends change."""
    version = (stored_version([index_id], TIMEFRAME), dividends_version(index_id))
    got = _panels.get(index_id)
    if got is None or got[0] != version:
        got = _panels[index_id] = version, load_panel([index_id], TIMEFRAME)
    return got[1]


def daily(index: Index, days: pd.DatetimeIndex, fill: str | None) -> pd.Series | None:
    """The index's daily returns over a record's days: bought on the first of them and held (`engine.hold`), as the
    record's buy & hold is bought where the record starts, with the record's fill (`next_open` for a record saved
    without one); 0 on a day without a bar. None when the store has no bars of it, or none after the record's first
    day."""
    with _lock:
        try:
            panel = _panel(index.id)
        except FileNotFoundError as e:
            _warn_not_stored(index.id, str(e))
            return None
        live = panel.started
        live.loc[live.index < days[0]] = False
        if not live[index.id].any():
            LOG.warning("%s has no bar from %s on: no index beside a record that starts then", index.id,
                        days[0].date())
            return None
        bars = buy_and_hold(panel, live, fill or "next_open")
        last = panel.close[index.id].last_valid_index()
    if (days[-1] - last).days > LATE_DAYS:
        LOG.warning("%s's bars end on %s, %d days before a record that ends on %s: its last days are flat", index.id,
                    last.date(), (days[-1] - last).days, days[-1].date())
    return metrics.daily_returns(bars).reindex(days).fillna(0.0)


@functools.lru_cache(maxsize=None)
def _warn_not_stored(index_id: str, why: str) -> None:
    LOG.warning("%s is not in the store (%s): no index beside the results of its market", index_id, why)


def figures(daily_returns: pd.Series) -> dict:
    """The index's figures over the record's days: those a buy & hold has (db.BENCHMARK_FIGURES), from its functions."""
    core = metrics.core(daily_returns)
    return ({k: core[k] for k in db.CORE} | {"k_ratio": db.k_ratio(daily_returns)}
            | db.at_target_dd(daily_returns, db.HELD_GROSS, db.HELD_GROSS))
