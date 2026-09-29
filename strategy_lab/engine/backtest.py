"""Portfolio backtest on target weights, with bar-level fills, costs, carry and optional intrabar exits.

Conventions (all returns are fractions of equity):
  * `target` row t is decided at the CLOSE of bar t, using nothing after it; weights are fractions of equity,
    sum of |w| <= config.MAX_GROSS ("without leverage"), checked, never silently clipped.
  * fill="next_open": the change is executed at the open of bar t+1. Bar k is split into its gap
    (close[k-1] -> open[k]), held with the weight carried from before, and its session (open[k] -> close[k]),
    held with the new weight. This is how the firm's simulator trades a signal emitted at a bar close.
    fill="next_close" executes at the close of bar t+1 instead (one bar more conservative).
  * A position is the units its fill bought, held until its target moves: then it is bought or sold to its target, a
    fraction of the equity at that moment. A rule's positions are independent, each traded only when its own target
    moves (its entry, its exit, a resize of its share); a panel strategy's book is rebalanced whole, every position
    traded back to its weight when any of them moves (`run(book=True)`). Between, a weight drifts with its price
    against the book's, as an account holds it (the firm's simulator and products; buy-and-hold here), and turnover
    counts the fills only. Until 2026-09-27 every weight was traded back to its target on every bar: a winner of a
    trend was sold down each bar (crypto Top-10 1d, SMA 20/200: 6.1% a year where holding the units makes 10.2%),
    and a book "rebalanced monthly" was in fact rebalanced every bar.
  * No fill takes the positions held past MAX_GROSS of the equity. Positions held as units grow past their shares,
    so a trade entered at its share of the equity can need more money than the account has free (two seats of half:
    one bought and doubled holds two thirds, and half of the equity bought for the other would borrow a sixth). What
    a bar's fills add to the exposure is then cut, pro rata, to the capital the positions it does not trade leave
    free, as a cash account without margin buys; a fill that reduces or closes a position always goes through, and a
    position bought short of its target keeps its units until its own target moves again. The drift itself is not
    cut: a short that goes against it grows past its share (`Result.exposure` shows what is held). Were every fill
    to take its full share, a trend rule's positions would hold 110-340% of the equity at their peak (crypto and
    stock Top-3 to Top-100, 1d), on money borrowed for nothing.
  * A short is liquidated where its price reaches LIQUIDATION times its average entry (fill="next_open" only, as the
    exits): closed at that price, inside a bar whose high reaches it or in the gap that passes it, so that it loses
    the money it was sold for and no more, as a position on its own margin at 1x does (the exchange's liquidation, its
    insurance fund taking what a gap carries past). It then stays out until its target leaves the short side (a book:
    until its next rebalance). Held as units without it, a short on a coin that multiplied lost more than the account
    (big_move_follow on crypto Top-100 1d, 2026-09-28: a drawdown of 105%).
  * An instrument that did not print a bar cannot be traded on it, nor on a bar without an open (the vendor gave
    only its close): a fill due there waits for its next bar, and the position rides through the bar, which it holds
    from the close before to its close. When it is delisted, the position is closed at its last close: a
    perp the exchange delisted (the date `refresh binance-delistings` keeps in its series' meta), including one
    delisted and later listed anew (nothing is held between the two; it trades again after), or any series whose
    data ends more than DATA_LAG before the panel's; a series ending within DATA_LAG of the others with no delisting
    on record is a vendor lag, and its position stays open to the end.
  * Costs per side, on traded notional (`engine.costs`): commission + half-spread of the instrument's asset class; a
    spot quote of a commodity pays half the spread its broker quoted at the fill instead, bar by bar (at the bar's
    open for a fill there, inside the bar for an intrabar exit, at the close before for a next_close fill). An
    intrabar exit or a liquidation trades the position's value at its exit price.
  * Dividends: a US stock's or ETF's cash dividend is paid on its ex-date to the weight held at the close before
    (a short pays it), as a return over that close: the bars are not adjusted for dividends.
  * Carry: perp funding (positive rate: longs pay) at each settlement, by the position held at its moment and on the
    position's value then, as the exchange charges it: at a bar's close (every settlement of a 1h bar, a 4h bar's, a
    daily bar's midnight) on its value at the close; inside a bar (a daily bar's 08:00 and 16:00, a 4h bar's hours
    on a perp settling hourly) on its value at the instrument's stored hourly close of that moment. A position an
    intrabar exit closed pays the settlements before the minute it was closed in and none at the bar's close; where
    that minute is not known (an exit found on the bar's own prices, a liquidation found on its high) it is taken at
    the middle of the bar. Charged instead on the weight at the bar's open, every settlement of the bar whether or not
    the position still held, it had moved records on crypto Top-10 by -0.13 to +0.08 points of compounded return a
    year and their Sharpe by 0.003 at most (breakout_trail and donchian_breakout, a configuration each, 1h to 1d,
    2019-2026). Borrow on shorts of spot/equities accrues over calendar time on the short's value at the open.
    Funding, borrow and dividends are cash: they move the equity and none of the positions' units, so a long that
    pays funding is a larger share of the equity after it, as a perp's margin pays it. Until 2026-09-29 each position
    kept its share of the equity through them, as if every carry were paid by selling a slice of every holding: a long
    held through 2021's funding was sold down (tsmom long only over 12 months on crypto Top-10 1d, its whole record:
    +0.42% a month that way, +0.22% holding its units, drawdowns of 73.6% and 77.6%; a configuration each of
    vol_managed, sma_cross and gtaa_faber on stocks Top-10 and the 34 ETFs moved by 0.004 pp a month at most).
  * Exits (fill="next_open" only): stop-loss / take-profit / trailing stop, as fractions of the entry price or as
    multiples of the instrument's average true range, checked at every minute inside a bar where the store has the
    instrument's one-minute bars (`data.minutes`), and against the bar's own high and low elsewhere: a stop hit
    inside a bar fills at its level, or at the open of the minute that gaps through it. A trailing stop's level is
    re-set at each bar's close from the best high (low) of the bars since entry (LeBeau's chandelier), or with
    `trail_every="minute"` at each minute's close from the best of the minutes: a level holds still through a minute,
    so the minutes measure it exactly whatever the order of a minute's own high and low (a trail following every
    trade is not measured: its result depends on that order, which only ticks record). Both levels touched in one
    minute (or bar) -> the stop is assumed first; a gap through a level fills at the open. After an exit the position
    stays flat until its target leaves that side (turns flat or to the other side): a resize of the same side is
    not a new signal. A signal exit at a bar's open happens before anything later in that bar, so a stop the bar
    reaches afterwards does not apply; the trailing peak includes the entry bar's high, reached while in the
    position (checked against ManifoldBT in tests/test_exits_vs_manifoldbt.py, which differs on exactly these two
    points). A rule's trades are cut at their exits before the capital is shared out (`exited`), and the engine
    closes its positions there, at those prices (`run(exit_fills=)`): an exit belongs to the rule's trade, measured
    from the trade's own entry, so a name seated in the middle of its rule's trade leaves with it and gives its seat
    back then. Walked again from each position's own fill, the exits of a name seated late would stop it out on its
    own (6.5-9.3% of the exits of donchian_breakout on 1d lists) while its seat stayed taken, empty, until the rule
    turned. A panel strategy's exits are the engine's own, walked on its weights.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, types
from numba.typed import List

from strategy_lab import indicators as ind
from strategy_lab import log
from strategy_lab.config import COSTS, MAX_GROSS
from strategy_lab.data import minutes, spreads, store
from strategy_lab.data.bars import Panel, dividends_path, funding_path, load_dividends, load_funding
from strategy_lab.engine import costs

LOG = log.get("backtest")
YEAR = pd.Timedelta(days=365)
MINUTE_NS = 60_000_000_000
DATA_LAG = pd.Timedelta(days=7)
ATR_BARS = 14                       # Wilder's period: the average true range the exits given in ATRs are measured in
LIQUIDATION = 2.0                   # a short at 1x has lost the money it was sold for when its price has doubled


@dataclass(frozen=True)
class Exits:
    """Intrabar exits, as fractions of the entry price or as multiples of the instrument's average true range
    (ATR_BARS bars of its own timeframe): a fixed fraction is a different exit on a currency pair's hour and a coin's
    day, a multiple of the ATR sits where the market's own swings put it on both. The stop and the target take the ATR
    at the close the entry was decided at; the trailing stop the ATR at the close before each bar. Where a fraction and
    a multiple of one kind are both set, the one nearer the price applies."""
    stop: float | None = None        # e.g. 0.02 = exit when price moves 2% against the entry
    take: float | None = None        # e.g. 0.04 = exit at +4%
    trail: float | None = None       # e.g. 0.03 = exit 3% below the best price since entry (longs)
    stop_atr: float | None = None    # e.g. 2.0 = exit 2 ATRs against the entry (the Turtles' 2N)
    take_atr: float | None = None    # e.g. 3.0 = exit 3 ATRs in the position's favour
    trail_atr: float | None = None   # e.g. 3.0 = exit 3 ATRs below the best price since entry (a chandelier exit)
    # when the trailing stop's level is re-set: at each bar's close from the best high (low) of the bars since entry,
    # as LeBeau's chandelier; or at each minute's close from the best of the minutes (where the store has them)
    trail_every: str = "bar"

    def __post_init__(self) -> None:
        if self.trail_every not in ("bar", "minute"):
            raise ValueError(f"trail_every is 'bar' or 'minute', got {self.trail_every!r}")
        if self.trail_every == "minute" and self.trail is None and self.trail_atr is None:
            raise ValueError("a level re-set every minute needs a trailing stop (trail or trail_atr)")

    def any(self) -> bool:
        return any(v is not None for v in (self.stop, self.take, self.trail, self.stop_atr, self.take_atr,
                                            self.trail_atr))


@dataclass(frozen=True)
class ExitFills:
    """Where a rule's trades were closed inside a bar by their exits (`exited`), for the engine to close its positions
    there (`run(exit_fills=)`): the bars' and instruments' positions, the prices, and the moments the minutes they
    were found in began (int64 ns, UTC; -1 where found on a bar's own prices), which tell the funding settlements
    inside the bar that the position still held. Sparse: a list's prices by bar and instrument are almost all empty,
    and a grid keeps one set per configuration."""
    rows: np.ndarray
    cols: np.ndarray
    prices: np.ndarray
    moments: np.ndarray

    def dense(self, n_bars: int, n_instruments: int) -> np.ndarray:
        out = np.full((n_instruments, n_bars), np.nan).T      # an instrument's bars contiguous, as the walks read them
        out[self.rows, self.cols] = self.prices
        return out


@dataclass
class Result:
    returns: pd.Series          # net return per bar
    gross: pd.Series            # before costs and carry
    cost: pd.Series             # trading cost per bar (fraction of equity)
    carry: pd.Series            # funding + borrow - dividends per bar (positive = paid)
    turnover: pd.Series         # traded notional per bar (fraction of equity)
    weights: pd.DataFrame       # the weight each position was last filled to at a bar's open (held as units since)
    exits: pd.DataFrame         # per bar and instrument: exit price where an intrabar exit happened, else NaN
    fill: str
    exposure: pd.Series         # sum of |weight| of the positions held through each bar, drifted since their fills
    unfunded: pd.Series         # exposure a bar's fills asked for beyond the capital free (not bought), per bar
    liquidated: pd.DataFrame    # per bar and instrument: a short liquidated (its `exits` price is where; a gap's: the
                                # bar before, the last it was held through)


def ended(panel: Panel) -> pd.DataFrame:
    """True where an instrument is delisted: on the bars after its last one when it stopped trading (a perp the
    exchange delisted, its date on record in the series' meta, or a series whose last bar is more than DATA_LAG
    before the panel's last bar; a shorter shortfall with no delisting on record is a vendor lag, not an end), and
    between the two listings of a perp delisted and later listed anew under the same name (the stretch on record).
    Worked out once per panel and delisting records; each caller gets its own copy."""
    records = _records(panel)
    got = panel.memo.get("ended")
    if got is None or got[0] != records:
        got = panel.memo["ended"] = records, _ended(panel, dict(zip(panel.instruments, records)))
    return got[1].copy()


def _records(panel: Panel) -> tuple:
    """Each instrument's delisting on record (`_delisting`), as the store holds it now."""
    return tuple(_delisting(ins, panel.timeframe) for ins in panel.instruments.values())


def _ended(panel: Panel, records: dict[str, tuple[bool, tuple]]) -> pd.DataFrame:
    printed = panel.close.notna().to_numpy()
    n = len(panel.index)
    ever = printed.any(axis=0)
    last = np.where(ever, n - 1 - np.argmax(printed[::-1], axis=0), -1)     # each instrument's last bar
    after = np.arange(n)[:, None] > last[None, :]                          # no bar from here on (all: never one)
    idx = panel.index
    stopped = ever & ((idx[-1] - idx[np.maximum(last, 0)]) > DATA_LAG) if n else ever
    stopped |= np.array([records[i][0] for i in panel.ids], dtype=bool)
    out = after & stopped[None, :]
    for k, i in enumerate(panel.ids):
        for lo, hi in records[i][1]:
            out[(idx > lo) & (idx < hi), k] = True
    return pd.DataFrame(out, index=idx, columns=panel.ids)


def _delisting(ins, timeframe: str) -> tuple[bool, tuple]:
    """What `refresh binance-delistings` kept in a perp's series' meta: delisted for good, and the stretches between
    a delisting and a new listing under the same name."""
    if ins.source != "perp":
        return False, ()
    p = store.meta_path("perp", timeframe, ins.symbol)
    return _meta_delisting(str(p), p.stat().st_mtime_ns) if p.exists() else (False, ())


@lru_cache(maxsize=None)
def _meta_delisting(path: str, version: int) -> tuple[bool, tuple]:
    meta = json.loads(Path(path).read_text())
    between = tuple((pd.Timestamp(lo), pd.Timestamp(hi)) for lo, hi in meta.get("delisted_periods", []))
    return bool(meta.get("delisted")), between


@dataclass(frozen=True)
class _Funding:
    """A panel's perp funding settlements on its bars (`_funding_terms`), each rate a return on the value of the
    position held when it is due: those due at a bar's close summed by bar (`at_close`); those due inside a bar (a
    daily bar's 08:00 and 16:00, a 4h bar's hours on a perp settling hourly) one by one, an instrument's in a run of
    their own (`start[j]` to `end[j]`, by bar and moment), with their bar, their moment and their rate times the price
    at that moment over the bar's close, which the bar's own return turns into the value of a position held from its
    open (`_funding_paid`)."""
    at_close: np.ndarray            # (bars, instruments), an instrument's bars contiguous
    start: np.ndarray               # (instruments,) int64
    end: np.ndarray                 # (instruments,) int64
    bar: np.ndarray                 # int64
    moment: np.ndarray              # int64 ns, UTC
    ratio: np.ndarray               # float64


@dataclass(frozen=True)
class _Terms:
    """What a backtest takes from the panel alone, the same for every target run on it."""
    open_to_trade: pd.DataFrame     # from an instrument's first bar on, until it is delisted
    stopped: np.ndarray             # delisted (`ended`)
    tradable: np.ndarray            # printed a bar showing a trade, or delisted (its weight is then set to zero)
    gap: pd.DataFrame               # the segment returns of `_segment_returns`
    intra: pd.DataFrame
    cc: pd.DataFrame
    rates: costs.Rates              # cost per side, by instrument and bar
    borrow_rate: np.ndarray
    funding: _Funding               # `_funding_terms`
    dividends: pd.DataFrame         # `_dividends_per_bar`
    dt_years: np.ndarray            # the length of each bar, in years


def _traded(panel: Panel) -> pd.DataFrame:
    """The bars an instrument can be dealt on: those it printed with an open, but for one that shows no trade (no
    volume, its open, high, low and close all at its last close: a thin ETF's quiet day, a halted stock's carried
    price, a coin's empty hour) in a series that reports its volume (every market here but FX pairs and spot metals,
    whose volume is nil): nobody dealt at that price, so a fill due there waits for the next bar that trades. A bar
    with a close and no open has no price to fill a next-open order at."""
    if any(getattr(panel, f) is None for f in ("open", "high", "low", "volume")):
        return panel.close.notna() if panel.open is None else panel.close.notna() & panel.open.notna()
    arrays = (panel.open, panel.high, panel.low, panel.close, panel.volume)
    return pd.DataFrame(_traded_bars(*(f.to_numpy(dtype=np.float64) for f in arrays)), index=panel.index,
                        columns=panel.ids)


@njit(cache=True)
def _traded_bars(opn, high, low, close, volume):
    """`_traded`, instrument by instrument: a printed bar with an open, unless it shows no trade (no volume, its open,
    high, low and close all at the close before it) in a series with some volume."""
    n, m = close.shape
    out = np.empty((m, n), np.bool_).T
    for j in range(m):
        has_volume = False
        for t in range(n):
            if volume[t, j] > 0:
                has_volume = True
                break
        last = np.nan                             # the close before, carried over the bars the instrument missed
        for t in range(n):
            c = close[t, j]
            v = 0.0 if np.isnan(volume[t, j]) else volume[t, j]
            still = v <= 0 and opn[t, j] == last and high[t, j] == last and low[t, j] == last and c == last
            out[t, j] = not np.isnan(c) and not np.isnan(opn[t, j]) and not (still and has_volume)
            if not np.isnan(c):
                last = c
    return out


def _terms(panel: Panel) -> _Terms:
    """The panel's `_Terms`, worked out once per panel (and again only if a delisting, funding, dividends or spreads
    file it read has changed since): a grid's configurations, its buy-and-hold and the random-timing check all run on
    one panel."""
    version = _files_written(panel)
    got = panel.memo.get("backtest")
    if got is None or got[0] != version:
        stopped = ended(panel)
        gap, intra, cc = _segment_returns(panel)
        idx = panel.index
        got = panel.memo["backtest"] = version, _Terms(
            open_to_trade=panel.started & ~stopped, stopped=stopped.to_numpy(),
            tradable=_traded(panel).to_numpy() | stopped.to_numpy(), gap=gap, intra=intra, cc=cc,
            rates=costs.rates(panel), borrow_rate=_borrow_rates(panel), funding=_funding_terms(panel),
            dividends=_dividends_per_bar(panel),
            dt_years=pd.Series(idx, index=idx).diff().fillna(pd.Timedelta(0)).to_numpy() / YEAR)
    return got[1]


def _files_written(panel: Panel) -> tuple:
    """When each file a panel's terms read was last written (-1: not there): a perp's delisting record, funding and,
    on bars longer than an hour, its hourly bars (the price a settlement inside a bar is valued at), a stock's or
    ETF's dividends, a spot quote's spreads. A backtest asks on every run, and a grid runs hundreds: the paths are
    worked out once per panel, and a file costs one stat."""
    files = panel.memo.get("term_files")
    if files is None:
        files = panel.memo["term_files"] = tuple(str(f) for i in panel.ids for f in _term_files(i, panel.instruments[i],
                                                                                                 panel.timeframe))
    return tuple(_written(f) for f in files)


def _term_files(instrument_id: str, ins, timeframe: str) -> list:
    if ins.source == "perp":
        hourly = [] if timeframe == "1h" else [store.path("perp", "1h", ins.symbol)]
        return [store.meta_path("perp", timeframe, ins.symbol), funding_path(instrument_id), *hourly]
    if ins.asset_class == "us_equity":
        return [dividends_path(ins.source, ins.symbol)]
    quoted = spreads.spread_path(instrument_id)
    return [] if quoted is None else [quoted]


def _written(path: str) -> int:
    try:
        return os.stat(path).st_mtime_ns
    except FileNotFoundError:
        return -1


def _validate_target(target: pd.DataFrame, panel: Panel, terms: _Terms) -> np.ndarray:
    """The target on the panel's bars and instruments: an instrument's weight held through a bar without one, 0 before
    its first; nothing before an instrument's first bar, nothing after a delisted instrument's last bar."""
    if not (target.index.equals(panel.index) and list(target.columns) == panel.ids):
        target = target.reindex(index=panel.index, columns=panel.ids)
    t, seen, gross = _validated(target.to_numpy(dtype=np.float64), terms.open_to_trade.to_numpy())
    if not seen:
        raise ValueError("target has no values on the panel's index/columns")
    if (gross > MAX_GROSS + 1e-9).any():
        worst = int(np.argmax(gross))
        raise ValueError(f"target gross {gross[worst]:.4f} > {MAX_GROSS} at {panel.index[worst]}: size down inside the "
                         "strategy")
    return t


@njit(cache=True)
def _validated(target, open_to_trade):
    """`_validate_target`'s arithmetic, instrument by instrument: the target held forward over NaN (0 before its first
    value) and 0 where the instrument cannot be traded; whether it had any value, and each bar's gross."""
    n, m = target.shape
    out = np.empty((m, n)).T                      # an instrument's bars contiguous, as pandas keeps a frame's columns
    gross = np.zeros(n)
    seen = False
    for j in range(m):
        last = np.nan
        for t in range(n):
            v = target[t, j]
            if not np.isnan(v):
                last = v
                seen = True
            x = last if open_to_trade[t, j] and not np.isnan(last) else 0.0
            out[t, j] = x
            gross[t] += abs(x)
    return out, seen, gross


def _segment_returns(panel: Panel) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Each bar's gap (the close before it to its open), session (its open to its close) and close-to-close returns."""
    gap, intra, cc = _segments(panel.open.to_numpy(dtype=np.float64), panel.close.to_numpy(dtype=np.float64))
    return tuple(pd.DataFrame(a, index=panel.index, columns=panel.ids) for a in (gap, intra, cc))


@njit(cache=True, error_model="numpy")
def _segments(opn, close):
    """`_segment_returns`, instrument by instrument: 0 on a bar the instrument did not print and where a return is
    undefined (no close before it yet); the close before a missed bar is carried over it. A bar with a close and no
    open is not traded (`_traded`): its whole move from the close before is its gap, held on the weight from before."""
    n, m = close.shape
    gap, intra, cc = np.empty((m, n)).T, np.empty((m, n)).T, np.empty((m, n)).T
    for j in range(m):
        last = np.nan
        for t in range(n):
            before, c = last, close[t, j]
            if np.isnan(c):
                gap[t, j] = 0.0
                intra[t, j] = 0.0
            else:
                o = c if np.isnan(opn[t, j]) else opn[t, j]
                g, s = o / before - 1.0, c / o - 1.0
                gap[t, j] = 0.0 if np.isnan(g) else g
                intra[t, j] = 0.0 if np.isnan(s) else s
                last = c
            r = last / before - 1.0
            cc[t, j] = 0.0 if np.isnan(r) else r
    return gap, intra, cc


def _borrow_rates(panel: Panel) -> np.ndarray:
    return np.array([COSTS[panel.instruments[i].asset_class].borrow_bps_annual / 1e4 for i in panel.ids])


def _dividends_per_bar(panel: Panel) -> pd.DataFrame:
    """Each cash dividend as a return on the first bar of its ex-date's New York session, over the close before that
    bar: what the holder at that close is paid."""
    out = np.zeros((len(panel.ids), len(panel.index))).T
    session = panel.index.tz_convert("America/New_York").normalize().tz_localize(None)
    close = None                                                   # the close before a bar, carried over missed bars
    for k, i in enumerate(panel.ids):
        d = load_dividends(i)
        if d is None or d.empty:
            continue
        if close is None:
            close = panel.close.ffill().to_numpy()
        pos = session.searchsorted(d.index, side="left")          # the first bar on or after the ex-date
        ok = (pos > 0) & (pos < len(session))
        paid = d.to_numpy()[ok] / close[pos[ok] - 1, k]
        keep = np.isfinite(paid)
        _summed_into(out[:, k], pos[ok][keep], paid[keep])
    return pd.DataFrame(out, index=panel.index, columns=panel.ids)


def _funding_terms(panel: Panel) -> _Funding:
    """Each perp's funding settlements on the panel's bars (`_Funding`): a settlement on the bar (close[k-1], close[k]]
    of the minute it was due, at the bar's close when that minute is the close. Binance settles on the hour and stamps
    45% of its settlements 1-10 ms after it (a stock perp's second one of a day a second after it), which would put
    them on the bar after, paid by the position held from that hour on rather than by the one held into it. A
    settlement inside a bar is valued at the instrument's price at its minute (`_price_at`); where that price is not
    known, at the bar's open. A settlement before the panel's first close is left out: nothing is held there."""
    n, m = len(panel.index), len(panel.ids)
    closes = panel.index
    at_close = np.zeros((m, n)).T
    bars, moments, ratios, counts = [], [], [], np.zeros(m, dtype=np.int64)
    for j, i in enumerate(panel.ids):
        f = load_funding(i)
        if f is None or f.empty:
            continue
        due = f.index.floor("min")
        pos = closes.searchsorted(due, side="left")                   # the first close at or after its minute
        keep = (pos < n) & ((pos > 0) | (due >= closes[0]))
        due, pos, rate = due[keep], pos[keep], f.to_numpy(dtype=np.float64)[keep]
        at_end = closes[pos] == due
        _summed_into(at_close[:, j], pos[at_end], rate[at_end])
        if at_end.all():
            continue
        due, pos, rate = due[~at_end], pos[~at_end], rate[~at_end]
        close = panel.close[i].to_numpy(dtype=np.float64)[pos]
        start = panel.open[i].to_numpy(dtype=np.float64)[pos] if panel.open is not None else close
        price, first_hour = _price_at(panel.instruments[i], due, closes[pos - 1])   # inside a bar: never the first
        unknown = ~np.isfinite(price)
        if unknown.any():
            # a gap in its hourly bars inside a bar the perp printed is the store's to fill; its first day (its daily bar
            # opens before its first hourly one) and bars before its own (its funding began first) have none to fill
            gap = (np.zeros(len(due), dtype=bool) if first_hour is None
                   else unknown & np.isfinite(close) & np.asarray(due > first_hour))
            if first_hour is None:
                LOG.warning("%s %s has no hourly bars: its %d funding settlements inside a bar are charged on the "
                            "position's value at the bar's open", i, panel.timeframe, int(unknown.sum()))
            elif gap.any():
                LOG.warning("%s %s: no hourly close inside the bar at %d funding settlements after its hourly bars "
                            "begin (the first at %s): charged on the position's value at the bar's open", i,
                            panel.timeframe, int(gap.sum()), due[gap][0])
            if (unknown & ~gap).any() and first_hour is not None:
                LOG.debug("%s %s: %d funding settlements before its first hourly bar valued at their bar's open", i,
                          panel.timeframe, int((unknown & ~gap).sum()))
            price = np.where(unknown, start, price)
        # over the bar's close: the bar's own return turns it into the value of a position held from its open; a bar
        # the instrument did not print has neither, and the position is valued at its start
        ratio = np.where(np.isfinite(price) & np.isfinite(close) & (close > 0), rate * price / close, rate)
        order = np.lexsort((due.asi8, pos))
        bars.append(pos[order].astype(np.int64))
        moments.append(due.asi8[order])
        ratios.append(ratio[order])
        counts[j] = len(order)
    end = np.cumsum(counts)
    joined = lambda parts, dtype: np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)   # noqa: E731
    return _Funding(at_close=at_close, start=end - counts, end=end, bar=joined(bars, np.int64),
                    moment=joined(moments, np.int64), ratio=joined(ratios, np.float64))


def _price_at(ins, moments: pd.DatetimeIndex, bar_opened: pd.DatetimeIndex) -> tuple[np.ndarray, pd.Timestamp | None]:
    """A perp's price at each moment (on the hour) inside the bar that opened at `bar_opened`: the close of its stored
    hourly bar ending then, or of the last one before it inside that bar where that hour has no bar; NaN where the bar
    has none that early, or the store no hourly bars of it. Beside it, its first hourly bar's close time (None: no
    hourly bars)."""
    try:
        hourly = store.read_bars(ins.source, "1h", ins.symbol, ["close"])["close"]
    except FileNotFoundError:
        LOG.debug("%s:%s has no hourly bars to value its funding inside a bar by", ins.source, ins.symbol)
        return np.full(len(moments), np.nan), None
    if hourly.empty:
        return np.full(len(moments), np.nan), None
    k = hourly.index.searchsorted(moments, side="right") - 1
    found = k >= 0
    found[found] = hourly.index[k[found]] > bar_opened[found]
    return np.where(found, hourly.to_numpy(dtype=np.float64)[np.maximum(k, 0)], np.nan), hourly.index[0]


def funding_held(panel: Panel) -> pd.DataFrame:
    """The funding a position held through a whole bar from its open pays, per unit of its value at the open: every
    settlement of the bar on the value the position has then (the random-timing check's positions, which have no
    intrabar exits). Worked out once per panel and version of its terms."""
    terms = _terms(panel)
    got = panel.memo.get("funding_held")
    if got is not None and got[0] is terms:
        return got[1]
    f = terms.funding
    inside = np.zeros_like(f.at_close)
    for j in range(len(panel.ids)):
        lo, hi = f.start[j], f.end[j]
        if hi > lo:
            np.add.at(inside[:, j], f.bar[lo:hi], f.ratio[lo:hi])
    grow = 1.0 + terms.intra.to_numpy()
    grow[~np.isfinite(grow)] = 1.0
    out = pd.DataFrame((f.at_close + inside) * grow, index=panel.index, columns=panel.ids)
    panel.memo["funding_held"] = terms, out
    return out


def _summed_into(column: np.ndarray, bars: np.ndarray, values: np.ndarray) -> None:
    """`values` summed by bar into `column` as pandas' groupby sum adds up a group (Kahan's compensated sum, in the
    values' order), so a bar with several settlements comes out as it did."""
    order = np.argsort(bars, kind="stable")
    _kahan_by_bar(column, bars[order].astype(np.int64), np.ascontiguousarray(values[order], dtype=np.float64))


@njit(cache=True)
def _kahan_by_bar(column, bars, values):
    k = 0
    while k < len(bars):
        bar, total, compensation = bars[k], 0.0, 0.0
        while k < len(bars) and bars[k] == bar:
            v = values[k]
            if not np.isnan(v):
                y = v - compensation
                t = total + y
                compensation = t - total - y
                if compensation != compensation:        # an infinite value: its sum stays infinite, not NaN
                    compensation = 0.0
                total = t
            k += 1
        column[bar] = total


def atr(panel: Panel) -> np.ndarray:
    """Each instrument's average true range over its own last ATR_BARS bars, as known at each bar's close (a bar it
    missed keeps the last one); NaN until it has that many. Worked out once per panel."""
    got = panel.memo.get("atr")
    if got is None:
        out = pd.DataFrame(np.nan, index=panel.index, columns=panel.ids)
        for i in panel.ids:
            b = panel.one(i)
            if len(b) > ATR_BARS:
                out.loc[b.index, i] = ind.atr(b["high"], b["low"], b["close"], ATR_BARS)
        got = panel.memo["atr"] = out.ffill().to_numpy()
    return got


def _apply_exits(target: np.ndarray, panel: Panel, exits: Exits, stopped: np.ndarray,
                 tradable: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sequential intrabar exits for next-open fills.

    Returns (w_session, exit_px, exit_at, after): the weight held from each bar's open; the exit price where a position
    was closed inside the bar (NaN elsewhere): a bar with an exit contributes its session return only up to the exit
    price, and the weight at its close is zero; the moment the minute it was found in began (int64 ns; -1 where it was
    found on the bar's own prices, and where there is none); and the target decided at each close as the walk acts on
    it, zero while a position an exit closed waits for its target to leave that side.
    """
    level = lambda v: np.nan if v is None else float(v)          # noqa: E731
    by_atr = any(v is not None for v in (exits.stop_atr, exits.take_atr, exits.trail_atr))
    unit = atr(panel) if by_atr else np.full((1, 1), np.nan)
    prices, stamps, first, last = _minutes_inside(panel)
    return _exits_walk(np.asarray(target, dtype=np.float64), panel.open.to_numpy(), panel.high.to_numpy(),
                       panel.low.to_numpy(), unit, level(exits.stop), level(exits.take), level(exits.trail),
                       level(exits.stop_atr), level(exits.take_atr), level(exits.trail_atr),
                       np.asarray(stopped, dtype=np.bool_), np.asarray(tradable, dtype=np.bool_), prices, stamps, first,
                       last, exits.trail_every == "minute")


_MINUTES = types.Array(types.float32, 2, "C", readonly=True)
_STAMPS = types.Array(types.int64, 1, "C", readonly=True)
_NONE = np.zeros((0, 3), dtype=np.float32)
_NONE.setflags(write=False)
_NO_STAMPS = np.zeros(0, dtype=np.int64)
_NO_STAMPS.setflags(write=False)


def _minutes_inside(panel: Panel) -> tuple:
    """Each instrument's one-minute bars (`data.minutes`, memory-mapped: open, high and low a row) with their close
    times and, for each of the panel's bars, the first and one past the last of its minutes: those that close after
    the bar's open (the close before it) and at or before its close. An instrument without minutes, and a bar without
    any, is walked on the bar's own prices. Worked out once per panel."""
    got = panel.memo.get("minutes")
    if got is None:
        n, m = len(panel.index), len(panel.ids)
        opened = np.r_[np.iinfo(np.int64).min, panel.index.asi8[:-1]]
        closed = panel.index.asi8
        first = np.zeros((m, n), dtype=np.int64).T              # an instrument's bars contiguous, as the walk goes
        last = np.zeros((m, n), dtype=np.int64).T
        prices = List.empty_list(_MINUTES)
        stamps = List.empty_list(_STAMPS)
        for j, i in enumerate(panel.ids):
            kept = minutes.read(i)
            if kept is None:
                prices.append(_NONE)
                stamps.append(_NO_STAMPS)
                continue
            times, ohl = kept
            first[:, j] = np.searchsorted(times, opened, side="right")
            last[:, j] = np.searchsorted(times, closed, side="right")
            prices.append(ohl)
            stamps.append(times)
        got = panel.memo["minutes"] = prices, stamps, first, last
    return got


def exited(panel: Panel, wanted: pd.DataFrame, exits: Exits) -> tuple[pd.DataFrame, ExitFills]:
    """A rule's positions with each trade cut at its intrabar exit: flat from the close of the bar its stop, target or
    trailing stop is hit until the rule's position turns (flat, or to the other side); and where those exits filled,
    for the engine to close the positions there (`run(exit_fills=)`). A trade's exit depends on the instrument's
    prices and the trade's side only, so it is known before the capital is shared out: a trade stopped out gives its
    seat or slot back at once, and a change of its share (the list's names changing) does not open it again."""
    terms = _terms(panel)
    wanted = wanted.reindex(index=panel.index, columns=panel.ids).fillna(0.0)
    _, px, at, after = _apply_exits(np.sign(wanted.to_numpy()), panel, exits, terms.stopped, terms.tradable)
    rows, cols = np.nonzero(~np.isnan(px))
    return wanted.where(after != 0.0, 0.0), ExitFills(rows, cols, px[rows, cols], at[rows, cols])


@njit(cache=True)
def _nearer(a, b, side, below):
    """Of two levels of one kind (NaN: not set), the one nearer the price. `below`: the kind sits below a long's price
    (a stop; a target sits above it), and on the other side of a short's."""
    if np.isnan(a):
        return b
    if np.isnan(b):
        return a
    if (side > 0) == below:
        return max(a, b)
    return min(a, b)


@njit(cache=True)
def _levels(side, entry, best, unit, atr_now, stop, take, trail, stop_atr, take_atr, trail_atr):
    """The stop (the nearer of the fixed stop and the trailing one) and the target of a position now."""
    level_stop = np.nan
    if not np.isnan(stop):
        level_stop = entry * (1 - side * stop)
    if not np.isnan(stop_atr):
        level_stop = _nearer(level_stop, entry - side * stop_atr * unit, side, True)
    if not np.isnan(trail):
        level_stop = _nearer(level_stop, best * (1 - side * trail), side, True)
    if not np.isnan(trail_atr):
        level_stop = _nearer(level_stop, best - side * trail_atr * atr_now, side, True)
    level_take = entry * (1 + side * take) if not np.isnan(take) else np.nan
    if not np.isnan(take_atr):
        level_take = _nearer(level_take, entry + side * take_atr * unit, side, False)
    return level_stop, level_take


@njit(cache=True)
def _hit(side, o, h, l, level_stop, level_take):
    """The exit price in a bar (or minute) of these prices, NaN for none: the stop assumed first when both are
    touched, a gap through a level filled at the open."""
    if not np.isnan(level_stop):
        if o <= level_stop if side > 0 else o >= level_stop:
            return o
        if l <= level_stop if side > 0 else h >= level_stop:
            return level_stop
    if not np.isnan(level_take):
        if o >= level_take if side > 0 else o <= level_take:
            return o
        if h >= level_take if side > 0 else l <= level_take:
            return level_take
    return np.nan


@njit(cache=True)
def _exits_walk(target, o, h, l, atr, stop, take, trail, stop_atr, take_atr, trail_atr, stopped, tradable, minute,
                stamps, first, last, per_minute):
    """The bar-by-bar walk of `_apply_exits`, compiled; a level that is not set is NaN, and so is `atr` when no level
    is given in ATRs (a 1x1 array then). Instrument j's bar k is walked minute by minute through `minute[j]`'s rows
    `first[k, j]` to `last[k, j]` - 1 where it has any, the levels checked against each minute's prices (an exit found
    there is dated by the minute's start: its close time in `stamps[j]` less a minute), and otherwise on the bar's own
    prices. The best price since entry moves after each minute with `per_minute` (a trailing stop re-set at every
    minute's close), and otherwise after each bar, from the bar's own high or low."""
    n, m = target.shape
    w_sess = np.zeros((m, n)).T                  # an instrument's bars contiguous: the walk goes instrument by instrument
    exit_px = np.full((m, n), np.nan).T
    exit_at = np.full((m, n), -1, dtype=np.int64).T
    after = np.zeros((m, n)).T
    after[:, :] = target
    with_atr = not (np.isnan(stop_atr) and np.isnan(take_atr) and np.isnan(trail_atr))
    for j in range(m):
        held = 0.0
        entry = best = unit = np.nan       # unit: the ATR at the close the entry was decided at
        out_side = 0.0                     # the side an exit closed: flat until the target leaves it
        for k in range(1, n):
            if stopped[k, j]:              # delisted: closed at its last close, nothing to trade after it
                held = 0.0
                entry = best = unit = np.nan
                out_side = 0.0
                continue
            want = target[k - 1, j]        # decided at close k-1, filled at open k
            if out_side != 0.0 and np.sign(want) == out_side:
                want = 0.0
            else:
                out_side = 0.0
            after[k - 1, j] = want
            if not tradable[k, j] or np.isnan(o[k, j]):    # nothing traded on this bar: a fill waits for the next
                w_sess[k, j] = held
                continue
            if want != held:
                if want != 0.0 and (held == 0.0 or np.sign(want) != np.sign(held)):
                    entry = best = o[k, j]
                    unit = atr[k - 1, j] if with_atr else np.nan
                held = want
            w_sess[k, j] = held
            if held == 0.0 or np.isnan(entry):
                continue
            side = np.sign(held)
            atr_now = atr[k - 1, j] if with_atr else np.nan
            px = np.nan
            at = -1
            if first[k, j] < last[k, j]:
                inside = minute[j]
                for q in range(first[k, j], last[k, j]):
                    level_stop, level_take = _levels(side, entry, best, unit, atr_now, stop, take, trail, stop_atr,
                                                     take_atr, trail_atr)
                    px = _hit(side, np.float64(inside[q, 0]), np.float64(inside[q, 1]), np.float64(inside[q, 2]),
                              level_stop, level_take)
                    if not np.isnan(px):
                        at = stamps[j][q] - MINUTE_NS
                        break
                    if per_minute:
                        best = max(best, np.float64(inside[q, 1])) if side > 0 else min(best, np.float64(inside[q, 2]))
                if np.isnan(px) and not per_minute:
                    best = max(best, h[k, j]) if side > 0 else min(best, l[k, j])
            else:
                level_stop, level_take = _levels(side, entry, best, unit, atr_now, stop, take, trail, stop_atr,
                                                 take_atr, trail_atr)
                px = _hit(side, o[k, j], h[k, j], l[k, j], level_stop, level_take)
                if np.isnan(px):
                    best = max(best, h[k, j]) if side > 0 else min(best, l[k, j])
            if not np.isnan(px):
                exit_px[k, j] = px
                exit_at[k, j] = at
                out_side = side
                held = 0.0
                entry = best = unit = np.nan
        if out_side != 0.0 and np.sign(target[n - 1, j]) == out_side:    # the last close: no bar left to walk
            after[n - 1, j] = 0.0
    return w_sess, exit_px, exit_at, after


def run(panel: Panel, target: pd.DataFrame, *, fill: str = "next_open", exits: Exits | None = None,
        book: bool = False, exit_fills: ExitFills | None = None) -> Result:
    """The record of holding `target` (see the module's conventions). `book`: the target is a book rebalanced whole
    (a panel strategy's weights: when any of them moves, every position is traded back to its weight), not
    independent positions each traded only when its own weight moves (a rule's). `exit_fills`: where the target's
    trades were already closed by their exits (a rule's, `exited`: the target is cut there): the positions are closed
    at those prices, and `exits` is not walked again."""
    if fill not in ("next_open", "next_close"):
        raise ValueError(f"fill must be 'next_open' or 'next_close', got {fill!r}")
    exits = exits or Exits()
    if (exits.any() or exit_fills is not None) and fill != "next_open":
        raise ValueError("intrabar exits are modelled for next_open fills only")
    terms = _terms(panel)
    T = _validate_target(target, panel, terms)
    stopped, tradable = terms.stopped, terms.tradable
    n, m = T.shape

    if fill == "next_close":
        F = _filled(T, tradable, stopped)                        # target of close t, filled at the close of t+1
        W = np.zeros((m, n)).T                                   # ... and held over bar t+2
        W[1:] = F[:-1]
        filled_on = np.zeros((m, n), dtype=np.bool_).T           # the bar whose close a fill held over bar t took place
        filled_on[1:] = tradable[:-1]
        cc = terms.cc.to_numpy()
        # no intrabar exit and no liquidation (next_open's only): no exit price, open or high to walk
        none = np.full((m, n), np.nan).T
        rates = terms.rates
        at_fill = np.concatenate([rates.at_close[:1], rates.at_close[:-1]])   # the close before: where W's fills are
        return _result(panel, fill, none, _held_returns(
            W, none.copy(), none, none, np.zeros((m, n)).T, cc, *_funding_args(terms.funding),
            *_exit_moments(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), m),
            panel.index.asi8, terms.dividends.to_numpy(), rates.flat, rates.column, at_fill, at_fill,
            terms.borrow_rate, terms.dt_years, filled_on, book, True, MAX_GROSS))

    if exit_fills is not None:
        w_session = _filled(T, tradable, stopped)
        exit_px = exit_fills.dense(n, m)
        exit_px[w_session == 0.0] = np.nan                     # a trade the target does not hold there is not closed
        kept = w_session[exit_fills.rows, exit_fills.cols] != 0.0
        rows, cols, moments = exit_fills.rows[kept], exit_fills.cols[kept], exit_fills.moments[kept]
    elif exits.any():
        w_session, exit_px, exit_at, _ = _apply_exits(T, panel, exits, stopped, tradable)
        rows, cols = np.nonzero(~np.isnan(exit_px))
        moments = exit_at[rows, cols]
    else:
        w_session = _filled(T, tradable, stopped)
        exit_px = np.full((m, n), np.nan).T
        rows = cols = moments = np.zeros(0, np.int64)
    exit_px = exit_px.copy()                                   # the walk writes its liquidations into it
    rates = terms.rates
    return _result(panel, fill, exit_px, _held_returns(
        w_session, exit_px, panel.open.to_numpy(), panel.high.to_numpy(), terms.gap.to_numpy(), terms.intra.to_numpy(),
        *_funding_args(terms.funding), *_exit_moments(rows, cols, moments, m), panel.index.asi8,
        terms.dividends.to_numpy(), rates.flat, rates.column, rates.at_open, rates.during, terms.borrow_rate,
        terms.dt_years, tradable, book, False, MAX_GROSS))


def _funding_args(f: _Funding) -> tuple:
    return f.at_close, f.start, f.end, f.bar, f.moment, f.ratio


def _exit_moments(rows: np.ndarray, cols: np.ndarray, moments: np.ndarray, m: int) -> tuple:
    """The intrabar exits' moments (`_apply_exits`' exit_at, `ExitFills.moments`) instrument by instrument, as the walk
    reads them: each instrument's run (`start[j]` to `end[j]`) of its exits' bars and moments, in the order of bars."""
    order = np.lexsort((rows, cols))
    counts = np.bincount(cols, minlength=m).astype(np.int64) if len(cols) else np.zeros(m, dtype=np.int64)
    end = np.cumsum(counts)
    return end - counts, end, rows[order].astype(np.int64), moments[order].astype(np.int64)


def _result(panel: Panel, fill: str, exit_px: np.ndarray, walked: tuple) -> Result:
    """The `Result` of `_held_returns`' walk: an exit price only where a position was held (a trade bought nothing
    for want of free capital closes nothing)."""
    net, gross, cost, carry, turnover, held, exposure, unfunded, liquidated = walked
    idx = panel.index
    mk = lambda a: pd.Series(a, index=idx)                        # noqa: E731
    exit_px[held == 0.0] = np.nan
    return Result(returns=mk(net), gross=mk(gross), cost=mk(cost), carry=mk(carry), turnover=mk(turnover),
                  weights=pd.DataFrame(held, index=idx, columns=panel.ids),
                  exits=pd.DataFrame(exit_px, index=idx, columns=panel.ids), fill=fill, exposure=mk(exposure),
                  unfunded=mk(unfunded), liquidated=pd.DataFrame(liquidated, index=idx, columns=panel.ids))


@njit(cache=True)
def _filled(target, tradable, stopped):
    """The weight after each bar's fill of the target decided at the close before it: a bar the instrument did not
    trade keeps the previous fill (0 before its first), and a delisted one holds nothing."""
    n, m = target.shape
    w = np.empty((m, n)).T
    for j in range(m):
        cur = np.nan
        for t in range(n):
            if tradable[t, j]:
                cur = target[t - 1, j] if t > 0 else 0.0
            w[t, j] = 0.0 if stopped[t, j] or np.isnan(cur) else cur
    return w


_MAX = np.finfo(np.float64).max


@njit(cache=True, error_model="numpy")
def _session_return(exit_px, opn, intra):
    """A bar's return from its open on the weight held through it: to the exit price where an intrabar exit closed
    the position, to the close otherwise; an infinite or undefined return as numpy's nan_to_num makes it."""
    r = exit_px / opn - 1.0 if not np.isnan(exit_px) else intra
    if np.isnan(r):
        return 0.0
    if r == np.inf:
        return _MAX
    if r == -np.inf:
        return -_MAX
    return r


@njit(cache=True, error_model="numpy")
def _close_weight(w, exit_px, opn, intra, grew):
    """The weight at a bar's close of `w` held from its open: 0 where an intrabar exit closed it, else its units at the
    close over the equity then (`grew`: the equity at the close over the equity at the open, after the bar's fills)."""
    if not np.isnan(exit_px):
        return 0.0
    return w * (1.0 + _session_return(exit_px, opn, intra)) / (1.0 if grew == 0.0 else grew)


@njit(cache=True)
def _involved(w):
    """The bars on which each instrument takes part in a book whose fills are `w`: from a bar it is to hold something
    through the bar it is sold on. Returned as the runs of those bars (start, end, instrument) ordered by start, and
    the bars on which any fill moves (a book's rebalance). One pass instrument by instrument over F-ordered `w`."""
    n, m = w.shape
    count = 0
    for j in range(m):
        inside = False
        for t in range(n):
            now = w[t, j] != 0.0 or (t > 0 and w[t - 1, j] != 0.0)
            if now and not inside:
                count += 1
            inside = now
    start = np.empty(count, dtype=np.int64)
    end = np.empty(count, dtype=np.int64)
    inst = np.empty(count, dtype=np.int64)
    moved = np.zeros(n, dtype=np.bool_)
    k = 0
    for j in range(m):
        inside = False
        for t in range(n):
            before = w[t - 1, j] if t > 0 else 0.0
            if w[t, j] != before:
                moved[t] = True
            now = w[t, j] != 0.0 or before != 0.0
            if now and not inside:
                start[k], inst[k] = t, j
            if inside and not now:
                end[k] = t - 1
                k += 1
            inside = now
        if inside:
            end[k] = n - 1
            k += 1
    order = np.argsort(start, kind="mergesort")
    return start[order], end[order], inst[order], moved


@njit(cache=True)
def _same_side(want, pre):
    """A fill of `want` keeps a position of `pre` on its side (a resize), rather than opening one or flipping it."""
    return want != 0.0 and pre != 0.0 and np.sign(want) == np.sign(pre)


@njit(cache=True)
def _funded(want, pre, scale):
    """The weight a fill of `want` from `pre` buys when the capital free takes only `scale` of the exposure the bar's
    fills add: what the position keeps goes through, what it adds is cut (a new position, a flip's new side, an
    increase); a reduction or a close is never cut."""
    if scale >= 1.0 or want == 0.0:
        return want
    if _same_side(want, pre):
        if abs(want) <= abs(pre):
            return want
        return np.sign(want) * (abs(pre) + scale * (abs(want) - abs(pre)))
    return want * scale


@njit(cache=True)
def _funding_paid(t, j, closed, liquidated_now, grow, at_close, f_next, f_end, f_bar, f_moment, f_ratio, e_next, e_end,
                  e_bar, e_moment, closes_ns):
    """The funding instrument j's position held from bar t's open pays over the bar, per unit of its value at the open:
    each settlement it still held at, on its value then (`_Funding`; `grow`: the bar's close over the price its session
    starts from). A position closed inside the bar (`closed`) pays none at the close, and inside the bar those before
    the minute it was closed in (`e_*`: the exits' moments); where that minute is not known (found on the bar's own
    prices, or a liquidation, `liquidated_now`) it is taken at the middle of the bar. `f_next` and `e_next` are each
    instrument's place in its run of settlements and exits, moved on as the bars go."""
    moment = closes_ns[t]
    if closed:
        moment = -1
        if not liquidated_now:
            while e_next[j] < e_end[j] and e_bar[e_next[j]] < t:
                e_next[j] += 1
            if e_next[j] < e_end[j] and e_bar[e_next[j]] == t:
                moment = e_moment[e_next[j]]
        if moment < 0:
            moment = (closes_ns[t - 1] + closes_ns[t]) // 2 if t > 0 else closes_ns[t]
    total = 0.0 if closed else at_close[t, j]
    while f_next[j] < f_end[j] and f_bar[f_next[j]] < t:
        f_next[j] += 1
    q = f_next[j]
    while q < f_end[j] and f_bar[q] == t:
        if f_moment[q] <= moment:                      # still held when it was due: before the minute it was closed in
            total += f_ratio[q]
        q += 1
    return total * grow


@njit(cache=True, error_model="numpy")
def _held_returns(w, exit_px, opn, high, gap, intra, fund_at_close, fund_start, fund_end, fund_bar, fund_moment,
                  fund_ratio, exit_start, exit_end, exit_bar, exit_moment, closes_ns, dividends, flat_rate, rate_column,
                  fill_rate, exit_rate, borrow_rate, dt_years, tradable, book, paid_after_fill, max_gross):
    """Each bar's net return, gross return, cost, carry and turnover of an account that holds units, the weight each
    position was last filled to, the exposure held through each bar and the exposure its fills could not buy: a
    position is bought or resized to its fill `w` (a fraction of the equity at that moment) and then held, its weight
    drifting with its price against the book's, until its fill moves again: a rule's own fill (`book` off: an entry,
    an exit, a resize of its share), or any fill of a book rebalanced whole (`book` on: a panel strategy's rebalance,
    each instrument traded to its weight where it trades that bar). No fill takes the positions held past `max_gross`
    of the equity: the exposure a bar's fills add beyond what the positions they trade already hold is cut, pro rata,
    to the capital the others leave free (`_funded`), as a cash account buys. `exit_px`: where an intrabar exit closed
    a position; the walk writes a short's liquidation into it (at LIQUIDATION times the short's average entry, where
    the bar's `high` reaches it, or on the bar before when its open is past it) and marks it in `liquidated`. A fill
    pays `fill_rate` of what it trades and an intrabar exit or liquidation `exit_rate` (a gap's liquidation the bar
    before's, where its exit is), on its bar, for an instrument with a column of its own there (`rate_column`:
    `engine.costs.Rates`), and `flat_rate` otherwise. A bar is its gap from the close before, on the weights held at
    that close, its fills at the open, then its session. A dividend goes to the weight held at the close before the
    ex-date's first bar; `paid_after_fill` gives it to the weight after the fill instead (fills at the close before,
    next_close's). Funding (`fund_*`: `_Funding`) is paid by each position at the settlements it held at, on its value
    then (`_funding_paid`; `exit_*`: the intrabar exits' moments, `_exit_moments`; `closes_ns`: the bars' close
    times), and it and a short's borrow come out of the equity after the bar's fills, where the positions are measured.
    Only the instruments taking part in the book on a bar are walked (`_involved`), bar by bar: a position's weight
    depends on the whole book's return."""
    n, m = w.shape
    start, end, inst, moved = _involved(w)
    active = np.empty(m, dtype=np.int64)          # the instruments walked this bar, in no order
    where = np.full(m, -1, dtype=np.int64)
    n_active = 0
    ends_at = np.full(m, -1, dtype=np.int64)
    w_close = np.zeros(m)                         # each position's weight at the last close
    w_open = np.zeros(m)                          # ... at this bar's open, before its fill
    fills = np.zeros(m, dtype=np.bool_)           # ... whether it is filled there
    x_now = np.zeros(m)                           # ... and from this bar's open, after the fill
    last_fill = np.zeros(m)                       # the weight each position was last filled to
    held = np.zeros((m, n)).T                     # ... on every bar (an instrument's bars contiguous)
    entry = np.zeros(m)                           # a short's average entry price (0: not short)
    out = np.zeros(m, dtype=np.bool_)             # a short liquidated: out until its target leaves the short side
    gap_move = np.zeros(m)                        # the return each position held through this bar's gap
    sold_in_gap = np.zeros(m)                     # the weight a liquidation in this bar's gap closed
    liquidated = np.zeros((m, n), dtype=np.bool_).T
    net = np.empty(n)
    gross = np.empty(n)
    cost = np.empty(n)
    carry = np.empty(n)
    turnover = np.empty(n)
    exposure = np.empty(n)
    unfunded = np.zeros(n)
    f_next = fund_start.copy()                    # each instrument's next funding settlement inside a bar
    e_next = exit_start.copy()                    # ... and its next intrabar exit's moment
    k = 0
    for t in range(n):
        while k < len(start) and start[k] == t:
            j = inst[k]
            if where[j] < 0:
                where[j] = n_active
                active[n_active] = j
                n_active += 1
            ends_at[j] = end[k]
            k += 1
        gap_sum = 0.0
        paid = 0.0
        for a in range(n_active):
            j = active[a]
            gap_move[j] = gap[t, j]
            sold_in_gap[j] = 0.0
            if w_close[j] != 0.0:
                if w_close[j] < 0.0 and entry[j] > 0.0 and opn[t, j] >= LIQUIDATION * entry[j]:
                    # the gap passed the short's liquidation price: it was closed there, not at the open
                    gap_move[j] = LIQUIDATION * entry[j] * (1.0 + gap[t, j]) / opn[t, j] - 1.0
                    exit_px[t - 1, j] = LIQUIDATION * entry[j]
                    liquidated[t - 1, j] = True
                    out[j] = True
                    entry[j] = 0.0
                    last_fill[j] = 0.0
                    sold_in_gap[j] = -1.0                # the weight it closed, once the gap's returns are known
                gap_sum += w_close[j] * gap_move[j]
                if not paid_after_fill:
                    paid += w_close[j] * dividends[t, j]
        den = 1.0 + gap_sum
        if den == 0.0:
            den = 1.0
        rebalance = book and moved[t]
        # the fills of the bar, and what they add to the exposure the positions keep: more than the capital left free
        # is not bought
        kept = 0.0
        added = 0.0
        for a in range(n_active):
            j = active[a]
            pre = w_close[j] * (1.0 + gap_move[j]) / den if w_close[j] != 0.0 else 0.0
            if sold_in_gap[j] != 0.0:
                sold_in_gap[j] = abs(pre)
                pre = 0.0
            before = w[t - 1, j] if t > 0 else 0.0
            if out[j] and (rebalance if book else w[t, j] >= 0.0):
                out[j] = False                           # a new decision: a book's rebalance, a rule's target turning
            filled = ((rebalance and tradable[t, j]) if book else w[t, j] != before) and not out[j]
            w_open[j] = pre
            fills[j] = filled
            if not filled:
                kept += abs(pre)
            elif _same_side(w[t, j], pre):
                kept += min(abs(w[t, j]), abs(pre))
                added += max(abs(w[t, j]) - abs(pre), 0.0)
            else:
                added += abs(w[t, j])
        scale = 1.0
        if added > 0.0 and kept + added > max_gross + 1e-12:
            scale = max(max_gross - kept, 0.0) / added
            unfunded[t] = (1.0 - scale) * added
        c = 0.0
        turn = 0.0
        session = 0.0
        fund = 0.0
        borrow = 0.0
        paid_on_fills = 0.0                           # next_close's dividends, as fractions of the equity after the fills
        held_now = 0.0
        for a in range(n_active):
            j = active[a]
            pre = w_open[j]
            if fills[j]:
                x = _funded(w[t, j], pre, scale)
                last_fill[j] = x
                if x >= 0.0:
                    entry[j] = 0.0
                elif pre >= 0.0:
                    entry[j] = opn[t, j]                 # a short opened, or a long turned short, at the open
                elif -x > -pre:
                    entry[j] = (-pre * entry[j] + (pre - x) * opn[t, j]) / -x      # added to at the open
            else:
                x = pre
            held[t, j] = last_fill[j]
            if x < 0.0 and entry[j] > 0.0 and high[t, j] >= LIQUIDATION * entry[j]:
                level = LIQUIDATION * entry[j]
                e = exit_px[t, j]
                # the price rising from the open meets the liquidation before an exit above it; an exit below the
                # open (a target) is assumed after it, as a stop is assumed before a target
                if np.isnan(e) or e > level or e < opn[t, j]:
                    exit_px[t, j] = level
                    liquidated[t, j] = True
                    out[j] = True
                    entry[j] = 0.0
                    last_fill[j] = 0.0
            filled = abs(x - pre)
            # an intrabar exit or a liquidation sells the position at its exit price: what it trades is its value there
            closed = abs(x) * (1.0 + _session_return(exit_px[t, j], opn[t, j], intra[t, j])) \
                if not np.isnan(exit_px[t, j]) else 0.0
            k_rate = rate_column[j]
            if k_rate >= 0:
                c += filled * fill_rate[t, k_rate] + closed * exit_rate[t, k_rate]
                if sold_in_gap[j] != 0.0:               # liquidated in the gap: its exit is on the bar before
                    c += sold_in_gap[j] * exit_rate[t - 1, k_rate]
            else:
                c += (filled + closed + sold_in_gap[j]) * flat_rate[j]
            turn += filled + closed + sold_in_gap[j]
            x_now[j] = x
            if x != 0.0:
                held_now += abs(x)
                session += x * _session_return(exit_px[t, j], opn[t, j], intra[t, j])
                grow = 1.0 + intra[t, j] if np.isfinite(intra[t, j]) else 1.0
                fund += x * _funding_paid(t, j, not np.isnan(exit_px[t, j]), liquidated[t, j], grow, fund_at_close,
                                          f_next, fund_end, fund_bar, fund_moment, fund_ratio, e_next, exit_end,
                                          exit_bar, exit_moment, closes_ns)
                borrow += max(-x, 0.0) * borrow_rate[j]
                if paid_after_fill:
                    paid_on_fills += x * dividends[t, j]
        exposure[t] = held_now
        opened = (1.0 + gap_sum) * (1.0 - c)       # the equity after the bar's fills, over the last close's
        paid += paid_on_fills * opened
        # the equity at the close over the equity after the fills: the positions' session, less the funding and borrow
        # paid, plus the dividends received; they are cash, and change no position's units
        grew = 1.0 + session - fund - borrow * dt_years[t] + (paid / opened if opened != 0.0 else 0.0)
        a = 0
        while a < n_active:
            j = active[a]
            x = x_now[j]
            w_close[j] = _close_weight(x, exit_px[t, j], opn[t, j], intra[t, j], grew) if x != 0.0 else 0.0
            if t >= ends_at[j] and w_close[j] == 0.0:        # out of the book: walked again from its next run
                n_active -= 1
                last = active[n_active]
                active[a] = last
                where[last] = a
                where[j] = -1
                continue
            a += 1
        gross[t] = (1.0 + gap_sum) * (1.0 + session) - 1.0
        # funding and borrow are paid on positions (x) measured as fractions of the equity after the bar's fills
        carry[t] = (1.0 + gap_sum) * (1.0 - c) * (fund + borrow * dt_years[t]) - paid
        net[t] = (1.0 + gap_sum) * (1.0 - c) * (1.0 + session) - 1.0 - carry[t]
        cost[t] = c
        turnover[t] = turn
    return net, gross, cost, carry, turnover, held, exposure, unfunded, liquidated
