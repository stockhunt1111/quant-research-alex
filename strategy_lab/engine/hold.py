"""Buy & hold of a list, the benchmark a strategy is judged against: held on spot terms.

Equal money in the list's instruments when it starts; then the units are held and the weights drift with prices.
Money moves only when the list's composition changes, traded on the bar after the decision (at its open for
next_open fills, at its close for next_close): a leaver is sold and its money is split evenly among the newcomers.
A ranked list refills a seat later than its leaver leaves when the leaver goes between re-ranks (a delisted name's
seat is taken at the end of its last day, one that left the S&P 500 at the month's re-rank): the money waits in
cash for the newcomer, which is bought with it and not out of every holding. A list that only grows buys each
newcomer an equal share out of every holding, pro rata; a fixed list, which only shrinks when a name stops trading,
spreads the leaver's money over the rest, pro rata. Only those trades pay costs, at spot rates on the bar of the
trade (`engine.costs`: a spot quote of a commodity pays its broker's spread there). A holder of a stock or
ETF is paid its dividends on the ex-date. A coin is held on spot, not on its perpetual, so nothing pays funding: its
prices are the spot pair's where Binance has it stored, the perp's own, on the pair's scale, before the pair lists
or for a coin Binance has no spot pair of (a perp and its pair differ by about 1% of the move since 2021).

Two kinds of instrument are not held (user, 2026-09-25): a currency pair, whose holder earns the difference of the two
currencies' interest rates (not modelled here) while its price goes nowhere over the years, and crude's spot quote
(Twelve Data's WTI/USD), which a holder cannot keep: crude is held through futures rolled every month, whose cost a
spot price does not show. Strategies still trade both. The CME futures list (user, 2026-09-26) is held whole, crude's
future among them: its joined contracts carry what a holder pays to roll, and the money of both the record and buy &
hold stays in T-bills (`margined`). A list with nothing else to hold (the FX majors, crude's spot quote alone) is
compared with cash: its buy & hold earns nothing, as the idle cash of every record here, and the money test credits
T-bills to both sides (`metrics.vs_hold`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from strategy_lab.data import store
from strategy_lab.data.bars import Panel, dividends_version
from strategy_lab.data.instruments import Instrument
from strategy_lab.engine import backtest as bt
from strategy_lab.engine import costs

SPOT = {"crypto_perp": "crypto_spot"}          # a coin is held on spot, not on its perpetual
MULTIPLIERS = ("1000000", "1000", "1M")         # a perp quoted on a multiple of its coin: 1000PEPEUSDT is PEPEUSDT x 1000
NOT_HELD = ("WTI/USD",)                         # besides every currency pair: crude's spot quote, which no one holds


def holdable(instrument: Instrument) -> bool:
    """Whether buy & hold holds an instrument: not a currency pair, not crude's spot quote."""
    return instrument.asset_class != "fx" and instrument.symbol not in NOT_HELD


def margined(panel: Panel) -> bool:
    """A universe of futures: a position takes margin, not cash, so the money of both the record and buy & hold stays
    in T-bills (`metrics.vs_hold`)."""
    return all(ins.source == "cme" for ins in panel.instruments.values())


def nothing_to_hold(panel: Panel) -> bool:
    """A universe buy & hold holds none of: it is compared with cash."""
    return not any(holdable(ins) for ins in panel.instruments.values())


def spot_pair(perp_symbol: str, stored: set[str]) -> str | None:
    """The stored Binance spot pair of a perp's coin: the same symbol, or the coin without the perp's multiplier."""
    if perp_symbol in stored:
        return perp_symbol
    base = perp_symbol.removesuffix("USDT")
    for m in MULTIPLIERS:
        if base.startswith(m) and f"{base[len(m):]}USDT" in stored:
            return f"{base[len(m):]}USDT"
    return None


def _held_prices(panel: Panel) -> tuple[np.ndarray, np.ndarray]:
    """The open and close each instrument is held at: a perp's coin at its spot pair's prices, the perp's own on the
    pair's scale (the median ratio of their closes) where the pair has no bar; any other instrument at its own.
    Worked out once per panel and version of the spot files it read."""
    stored = set(store.symbols("spot", panel.timeframe))
    pairs = {i: spot_pair(ins.symbol, stored) for i, ins in panel.instruments.items() if ins.source == "perp"}
    pairs = {i: p for i, p in pairs.items() if p is not None}
    version = tuple(sorted((p, store.path("spot", panel.timeframe, p).stat().st_mtime_ns) for p in pairs.values()))
    got = panel.memo.get("held_prices")
    if got is not None and got[0] == version:
        return got[1]
    o, c = panel.open.copy(), panel.close.copy()
    for i, p in pairs.items():
        spot = store.read_bars("spot", panel.timeframe, p, ["open", "close"]).reindex(panel.index)
        both = spot["close"].notna() & c[i].notna()
        if not both.any():
            continue
        scale = float((spot["close"][both] / c[i][both]).median())
        o[i] = spot["open"].combine_first(o[i] * scale)
        c[i] = spot["close"].combine_first(c[i] * scale)
    got = panel.memo["held_prices"] = version, (o.to_numpy(float), c.to_numpy(float))
    return got[1]


def _paid(panel: Panel) -> np.ndarray:
    """The dividends a holder is paid on each bar (`backtest._dividends_per_bar`), once per panel and dividends files;
    buy-and-hold pays no funding, so it does not take the engine's terms, which need every perp's."""
    version = tuple(dividends_version(i) for i in panel.ids)
    got = panel.memo.get("held_dividends")
    if got is None or got[0] != version:
        got = panel.memo["held_dividends"] = version, bt._dividends_per_bar(panel).to_numpy(float)
    return got[1]


@njit(cache=True)
def _rate(flat, column, quoted, t, i):
    """What instrument i's trade on bar t costs a side (`engine.costs.Rates`)."""
    return quoted[t, column[i]] if column[i] >= 0 else flat[i]


@njit(cache=True)
def _hold_walk(o, c, live, flat, column, quoted, paid, at_open, refills):
    t_n, n = c.shape
    out = np.zeros(t_n)
    w = np.zeros(n)                             # holdings as fractions of equity; 1 - w.sum() is cash
    held = np.zeros(n, dtype=np.bool_)
    last = np.full(n, np.nan)                   # last printed close
    waiting = False                             # a vacated seat's money is in cash, waiting for its newcomer
    for t in range(t_n):
        want = live[t - 1] if t > 0 else held
        change = False
        for i in range(n):
            if want[i] != held[i]:
                change = True
                break
        eq = 1.0
        # the bar's gap (last close -> open) when a trade is due at its open, else the whole bar; a dividend goes to
        # what was held at the last close
        g = np.zeros(n)
        for i in range(n):
            g[i] = paid[t, i]
            if not np.isnan(c[t, i]) and not np.isnan(last[i]):
                p = o[t, i] if (change and at_open and not np.isnan(o[t, i])) else c[t, i]
                g[i] += p / last[i] - 1.0
        r = 0.0
        for i in range(n):
            r += w[i] * g[i]
        eq *= 1.0 + r
        if r != -1.0:
            for i in range(n):
                w[i] = w[i] * (1.0 + g[i]) / (1.0 + r)
        if change:
            # in money, as fractions of the equity before the trade: a sale's cost comes off its proceeds, a
            # purchase's off the amount bought, and a holding that stays is not touched
            h = w.copy()
            cash = 1.0 - w.sum()
            joiners = 0
            leavers = 0
            members = 0
            for i in range(n):
                if held[i] and not want[i]:
                    cash += h[i] * (1.0 - _rate(flat, column, quoted, t, i))
                    h[i] = 0.0
                    leavers += 1
                if want[i] and not held[i]:
                    joiners += 1
                if want[i]:
                    members += 1
            stay = 0.0
            for i in range(n):
                if want[i] and held[i]:
                    stay += h[i]
            if joiners > 0 and leavers == 0 and not waiting and stay > 0.0:
                share = joiners / members          # a list that only grows: the newcomers' equal shares, pro rata
                for i in range(n):
                    if want[i] and held[i]:
                        sold = h[i] * share
                        cash += sold * (1.0 - _rate(flat, column, quoted, t, i))
                        h[i] -= sold
            if joiners > 0:                        # the leavers' money, of this bar or waiting since, buys the newcomers
                for i in range(n):
                    if want[i] and not held[i]:
                        h[i] = cash / joiners * (1.0 - _rate(flat, column, quoted, t, i))
                cash = 0.0
                waiting = False
            elif stay > 0.0 and cash > 0.0:
                if refills:                        # a ranked list's seat stands empty: its money waits for the newcomer
                    waiting = True
                else:                              # a fixed list only shrinks: the rest take the money, pro rata
                    for i in range(n):
                        if want[i] and held[i]:
                            h[i] += cash * h[i] / stay * (1.0 - _rate(flat, column, quoted, t, i))
                    cash = 0.0
            total = h.sum() + cash
            eq *= total
            for i in range(n):
                w[i] = h[i] / total if total > 0.0 else 0.0
                held[i] = want[i]
            if at_open:                          # the rest of the bar, open -> close, on the new holdings
                s = np.zeros(n)
                for i in range(n):
                    if not np.isnan(c[t, i]) and not np.isnan(o[t, i]) and not np.isnan(last[i]):
                        s[i] = c[t, i] / o[t, i] - 1.0
                r2 = 0.0
                for i in range(n):
                    r2 += w[i] * s[i]
                eq *= 1.0 + r2
                if r2 != -1.0:
                    for i in range(n):
                        w[i] = w[i] * (1.0 + s[i]) / (1.0 + r2)
        for i in range(n):
            if not np.isnan(c[t, i]):
                last[i] = c[t, i]
        out[t] = eq - 1.0
    return out


def bought_on(live: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """`live` held from the UTC day `day` on: nothing before the last bar of the day before, whose close decides the
    purchase, so it is bought at the day's first open, as a record's positions held that day were decided at that close
    (from the start of the day, a market with sessions was bought at the day's second open, its first bar's close
    deciding: a US stock's buy & hold lost the record's first day)."""
    bar_day = (live.index - pd.Timedelta(microseconds=1)).tz_convert("UTC").normalize()
    decided = max(int(bar_day.searchsorted(day, side="left")) - 1, 0)
    out = live.copy()
    out.iloc[:decided] = False
    return out


def buy_and_hold(panel: Panel, live: pd.DataFrame, fill: str = "next_open", refills: bool = False) -> pd.Series:
    """Net return per bar of holding `live`'s instruments as the module says; `live` is the list's composition decided
    at each bar's close (an instrument leaves it when it is delisted, on top of what `live` says); an instrument that
    is not held (`holdable`) is left out, and with nothing left the return is zero: cash. `refills`: `live` is a
    ranked list's, whose empty seats are taken by newcomers; else a fixed list's."""
    live = live.reindex(index=panel.index, columns=panel.ids, fill_value=False) & ~bt.ended(panel)
    live.loc[:, [i for i in panel.ids if not holdable(panel.instruments[i])]] = False
    rates = costs.rates(panel, SPOT)
    at = "at_open" if fill == "next_open" else "at_close"
    o, c = _held_prices(panel)
    paid = _paid(panel)
    out = _hold_walk(o, c, live.to_numpy(bool), rates.flat, rates.column, getattr(rates, at), paid,
                     fill == "next_open", refills)
    return pd.Series(out, index=panel.index)
