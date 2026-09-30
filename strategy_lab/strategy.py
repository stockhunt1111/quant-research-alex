"""How a strategy is written: a small function plus a parameter grid.

    @rule(grid={"fast": [10, 20], "slow": [50, 100], "stop": [None, 0.03]})
    def sma_cross(bars, fast, slow):
        return (ind.sma(bars.close, fast) > ind.sma(bars.close, slow)).astype(float)

Each strategy is one file in `strategies/`, named after it, so that it can be read, or handed to another engine's
developer, on its own. The building blocks several of them share are here: `hold_between` (long from an entry
condition until an exit condition), `with_short` (a rule's long side and its mirror, the short side, as a grid
switch), `rebalanced` (a book or a position decided once a month, week or n days), `ends` and `long_short` (a
ranking's two ends, and a ranking into a long/short book), `as_of` and `calendar_months` (a panel's lookback in
calendar time) and `bars_in` (a span its source gives in days or months, as bars of the timeframe run: a rule's
indicator periods are bars, as on any chart, but a 12-month return or a 200-day average is the same time on 1h, 4h
and 1d).

A `rule` sees one instrument's bars and returns a position in [-1, 1] per bar (decided at that bar's close).
Across a universe every instrument gets an equal share of capital among the instruments in the universe at that
bar, so the portfolio never exceeds its capital. With grid key `slots` = k the book has k slots of 1/k of capital
instead (sparse signals then use the capital the idle instruments would leave in cash): a trade takes a free slot on
its first bar and keeps it, at that share, until it ends; a trade that finds every slot taken is not traded, and of
the trades starting on one bar the more liquid instrument's takes a slot first (`slotted`). A list runs only the
slot counts below its number of names, and holds its top `top_k` names only up to that number (`Strategy.grid_for`):
beyond, a configuration would not put the capital on fewer names, only leave part of it idle. A `panel` strategy sees
the whole universe and returns weights itself (sum of |w| <= 1); if its function takes a `live` argument it receives
the mask of instruments that are in the universe at each bar. Its weights are a book rebalanced whole, every position
traded back to its weight when any weight moves (an allocation decided once a month or week), unless it says they
are positions of their own (`panel(book=False)`: events, each traded only when its own weight moves, as a rule's):
rebalanced whole, a book of events would sell its running winners down at every new event. Grid keys of the
engine's exits (`stop`, `take`, `trail` as fractions of the entry price, `stop_atr`, `take_atr`, `trail_atr` in
average true ranges, and `trail_every`: the trailing stop's level re-set at each bar's close or at each minute's) and
`slots` (sizing) are not passed to the function; a rule's trades end at those exits before the capital is shared
out, so a trade stopped out gives its seat or slot back, and the engine closes its positions at those exits' prices
(`Strategy.target_and_exits`). A rule may name a `grade` (`strategy_lab.trade_model`): it sees
the rule's positions on every instrument of the run at once and keeps, drops or sizes the rule's trades before the
universe's seats are given out; the grid keys the grade takes (a threshold) go to it, not to the function.

A universe that changes (a liquidity list re-picked monthly) does not cut a rule's trade: a name that leaves it keeps
its seat until the trade it holds ends by the rule, and a name that joins waits for a free seat (`seats`: the
longest-waiting first, the more liquid first of those that joined together). An instrument that stopped trading
(delisted, or its data ended: `engine.backtest.ended`) is not in the universe any more, whatever its list says, so
its seat and its share of the capital go to the others. A panel
strategy's exits are its own rebalances: a name that leaves the universe is dropped there. So is a rule's that holds an
exposure rather than trades (`rule(exposure=True)`: a position always held, only its size changing), which would
otherwise keep the names it started with for ever: its names are the universe's members, a leaver sold at the re-pick
and a newcomer bought at once, as the list's buy-and-hold holds them.

A bar an instrument missed (a vendor gap, a halt) is not a decision point for it: a rule keeps its last position
there, and a panel strategy sees the instrument's last close (`Panel.as_known`). The engine cannot trade an
instrument on a bar it did not print and closes the position when its data ends.

A rule may read its market's index besides its bars (`rule(market=True)`, `strategy_lab.market`): SPY's or bitcoin's
daily closes, each as a bar knows it.
"""
from __future__ import annotations

import functools
import hashlib
import heapq
import importlib
import importlib.metadata
import inspect
import itertools
import os
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import talib
from numba import njit

from strategy_lab import config, log, market, provenance
from strategy_lab.data.bars import Panel, liquidity
from strategy_lab.data.instruments import FX_BY_TURNOVER
from strategy_lab.engine.backtest import ExitFills, Exits, ended, exited
from strategy_lab.shared import sharing

LOG = log.get("strategy")
EXIT_KEYS = ("stop", "take", "trail", "stop_atr", "take_atr", "trail_atr", "trail_every")
SIZING_KEYS = ("slots",)


@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str                       # "rule" | "panel"
    fn: Callable
    grid: dict = field(default_factory=dict)
    description: str = ""
    grade: Callable | None = None   # a rule's: (panel, wanted, live, **grade params) -> wanted
    prepare: Callable | None = None     # (panel, configs, member): work the grid shares, done before it (see `rule`)
    exposure: bool = False          # a rule's position is a holding kept while listed, not a trade (see `rule`)
    model: bool = False             # a rule's function fits a model drawing with its module's SEED (see `rule`)
    market: bool = False            # a rule's function reads its market's index besides its bars (see `rule`)
    book: bool = True               # a panel's weights are a book rebalanced whole, not positions of their own (`panel`)

    @property
    def rebalanced_whole(self) -> bool:
        """The engine trades every position back to its weight when any weight moves (`backtest.run(book=True)`)."""
        return self.kind == "panel" and self.book

    def decisions(self, panel: Panel, params: dict) -> np.ndarray | None:
        """The bars at whose close a book is decided (its `rebalance` grid key: a month, a week, n days;
        `period_starts`), where the engine brings every position back to its weight even when the weights repeat
        (`backtest.run(decided=)`); None for a strategy without them (a rule, a panel of positions of their own)."""
        if not self.rebalanced_whole or "rebalance" not in params:
            return None
        return period_starts(panel.index, params["rebalance"])

    def grid_for(self, names: int | None = None) -> dict:
        """The grid run on a list of `names` instruments: of the `slots` values only those below that number (user,
        2026-09-27: a Top-3 list runs one slot per name only), of the numbers of names to hold (`top_k`) only those up
        to it; beyond either, a configuration only leaves capital idle. The whole grid when `names` is None."""
        if names is None:
            return dict(self.grid)
        grid = dict(self.grid)
        if "slots" in grid:
            grid["slots"] = [v for v in grid["slots"] if v is None or v < names]
        if "top_k" in grid:
            grid["top_k"] = [v for v in grid["top_k"] if v <= names]
        empty = [k for k, v in grid.items() if not v]
        if empty:
            raise ValueError(f"{self.name}: no value of {', '.join(empty)} fits a list of {names} names")
        return grid

    def configs(self, names: int | None = None) -> list[dict]:
        """Every configuration of the grid run on a list of `names` instruments (`grid_for`)."""
        grid = self.grid_for(names)
        if not grid:
            return [{}]
        keys = list(grid)
        return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]

    @staticmethod
    def exits(params: dict) -> Exits:
        return Exits(**{k: params[k] for k in EXIT_KEYS if params.get(k) is not None})

    def grade_keys(self) -> tuple[str, ...]:
        """The grid keys its grade takes."""
        if self.grade is None:
            return ()
        return tuple(k for k in inspect.signature(self.grade).parameters if k not in ("panel", "wanted", "live"))

    def signal_params(self, params: dict) -> dict:
        return {k: v for k, v in params.items() if k not in EXIT_KEYS + SIZING_KEYS + self.grade_keys()}

    def target(self, panel: Panel, params: dict, member: pd.DataFrame | None = None,
               memo: dict | None = None, signals: dict | None = None) -> pd.DataFrame:
        """The weights the strategy holds on each bar, decided at its close (`target_and_exits`)."""
        return self.target_and_exits(panel, params, member, memo, signals)[0]

    def target_and_exits(self, panel: Panel, params: dict, member: pd.DataFrame | None = None,
                         memo: dict | None = None,
                         signals: dict | None = None) -> tuple[pd.DataFrame, ExitFills | None]:
        """The weights the strategy holds on each bar, and where a rule's trades were closed inside a bar by the
        engine's exits (None: no such exits, or a panel strategy, whose exits the engine walks on its weights): the
        engine closes the positions there, at those prices (`backtest.run(exit_fills=)`).

        `memo`: positions already computed on this panel and membership, by configuration less its sizing (and a
        panel strategy's exits, which the engine applies to its weights): a grid repeats them once per slot count.

        `signals` (a rule's): each instrument's own positions, before a grade and the seats, by signal configuration.
        They depend on the instrument's bars and the signal parameters only, so another membership of the same bars (a
        neighbouring list), another seed of the grade's model and the rule without its grade reuse them. Not for a
        rule whose function reads what a `prepare` step keeps: its positions depend on more than its bars."""
        sp = self.signal_params(params)
        gp = {k: params[k] for k in self.grade_keys() if k in params}
        exits = self.exits(params)
        # a rule's trades end at their exits before the seats are given out, so its positions depend on them
        ep = {k: params[k] for k in EXIT_KEYS if params.get(k) is not None} if self.kind == "rule" else {}
        key = repr(sorted({**sp, **gp, **ep}.items()))
        live = _live(panel, member, memo)
        if self.kind == "rule":
            if memo is not None and key in memo:
                packed, seated, fills = memo[key]
                raw = packed.astype(np.float64)
            else:
                # a rule's position is held through its instrument's missing bars, and so after its last one: its trade
                # ends where the instrument stopped trading (the engine closes the position at its last close), and
                # the seat it holds goes to a newcomer then
                wanted = _positions(self, panel, sp, signals).where(~ended(panel), 0.0)
                if self.grade is not None:
                    wanted = self.grade(panel, wanted, live, **gp)
                fills = None
                if exits.any():
                    wanted, fills = exited(panel, wanted, exits)
                seated = live if self.exposure else seats(live, wanted, _slot_liquidity(panel))
                raw = np.where(seated.to_numpy(), wanted.to_numpy(), 0.0)
                if memo is not None:
                    memo[key] = _packed(raw), seated, fills
            slots = params.get("slots")
            if slots:
                return slotted(pd.DataFrame(raw, index=panel.index, columns=panel.ids), slots,
                               _slot_liquidity(panel)) / slots, fills
            n_live = seated.to_numpy().sum(axis=1)
            n_live[n_live == 0] = 1
            return pd.DataFrame(np.divide(raw, n_live[:, None], order="F"), index=panel.index, columns=panel.ids), fills
        if memo is not None and key in memo:
            return memo[key], None
        if "live" in inspect.signature(self.fn).parameters:      # cross-sectional rules rank only live members
            sp = {**sp, "live": live.copy()}
        w = self.fn(panel.as_known(), **sp).reindex(index=panel.index, columns=panel.ids).fillna(0.0).where(live, 0.0)
        if memo is not None:
            memo[key] = w
        return w, None


_LIVE = ("live",)            # the key of `_live` in a memo, beside the configurations' (strings)


def _live(panel: Panel, member: pd.DataFrame | None, memo: dict | None) -> pd.DataFrame:
    """The instruments in the universe on each bar, from their first bar on (all of them without a membership) until
    they stop trading (`ended`: a list's membership already leaves them out; a fixed list's does not). A memo holds
    one panel and membership's work (see `Strategy.target`), so it is worked out once for a grid."""
    if memo is not None and _LIVE in memo:
        return memo[_LIVE]
    trading = panel.started & ~ended(panel)
    live = trading if member is None else (trading & member.reindex_like(panel.close).fillna(False))
    if memo is not None:
        memo[_LIVE] = live
    return live


def fill_signals(strategy: Strategy, panel: Panel, configs: list[dict], signals: dict | None) -> None:
    """Every signal configuration of `configs` on every instrument of the panel into `signals`, instrument by
    instrument: an instrument's bars are cut out of the panel once for all of them, where configuration by configuration
    a grid of nine signal configurations cut them nine times. The grid then finds its positions there (see
    `Strategy.target`); nothing is done for a panel strategy, or without `signals`. The configurations share the bars
    they are given (a rule never writes into its bars: tests/test_lookahead.py) and the indicators and regimes worked
    out on them, each computed once for all of them (`strategy_lab.shared`).

    An instrument's positions are also kept on disk (`_kept_path`), read back by any later evaluation that gives the
    rule the same bars with the same code: the same rule on the same instrument is asked for on every list that holds
    it (Top-3 to Top-100, and the lists next to each for their checks) and on the instrument alone, each a job of its
    own and often in another process of a batch."""
    if signals is None or strategy.kind != "rule":
        return
    _same_bars(strategy, panel, signals)
    todo: dict[str, dict] = {}
    for cfg in configs:
        sp = strategy.signal_params(cfg)
        todo.setdefault(repr(sorted(sp.items())), sp)
    code = _code_version(strategy)
    if code is not None and strategy.market:                # its positions depend on its market's index too
        code = f"{code}-{market.version(panel)}"
    for i in panel.ids:
        missing = [(sig, sp) for sig, sp in todo.items() if (sig, i) not in signals]
        if not missing:
            continue
        bars = panel.one(i)
        path = None if code is None else _kept_path(strategy, code, panel, i, bars)
        kept = {} if path is None else _read_kept(path)
        usable = {sig: own for sig, own in kept.items() if len(own) == len(bars)}
        if len(usable) < len(kept):
            LOG.warning("%s: positions kept there hold other bars than %s's %d: worked out again", path, i, len(bars))
        fresh = {}
        with sharing(bars):
            for sig, sp in missing:
                own = usable.get(sig)
                if own is not None:
                    signals[(sig, i)] = _packed(_aligned(pd.Series(own.astype(np.float64), index=bars.index),
                                                         panel.index))
                    continue
                out = pd.Series(strategy.fn(bars, **sp), dtype="float64")
                signals[(sig, i)] = _packed(_aligned(out, panel.index))
                if out.index.equals(bars.index):              # kept only what reads back as the same positions
                    fresh[sig] = _packed(_aligned(out, bars.index))
                else:
                    _not_kept(strategy.name, "its output is not on its bars' own index")
        if fresh and path is not None:
            _write_kept(path, {**usable, **fresh})


KEPT_DIR = config.CACHE_DIR / "positions"


def _code_version(strategy: Strategy) -> str | None:
    """The version of the code a rule's positions come from: every module of an evaluation and the rule's own file (the
    hash `provenance` stamps a result with), and the libraries they compute with (a rule's model, LightGBM, among
    them). None for a rule defined outside the repository (its positions are not kept)."""
    try:
        return _code_of(inspect.getsourcefile(strategy.fn))
    except (TypeError, ValueError, OSError) as e:
        _not_kept(strategy.name, repr(e))
        return None


@functools.lru_cache(maxsize=None)
def _code_of(strategy_file: str) -> str:
    """`_code_version` of a rule's file, worked out once a process: its code does not change once imported."""
    sha = provenance.stamp(strategy_file)["sha"]
    if sha is None:
        raise OSError("a source file of the evaluation is missing")
    libs = (f"{np.__version__} {pd.__version__} {talib.__version__} {_version_of('lightgbm')} "
            f"{sys.version_info.major}.{sys.version_info.minor}")
    return hashlib.blake2b(f"{sha} {libs}".encode(), digest_size=8).hexdigest()


def _version_of(package: str) -> str:
    """An installed library's version, without importing it (LightGBM is the [ml] extra: absent, nothing uses it)."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        LOG.debug("%s is not installed: the positions kept on disk are keyed without it", package)
        return "none"


@functools.lru_cache(maxsize=None)
def _not_kept(name: str, why: str) -> None:
    LOG.info("%s: its positions are not kept on disk (%s): computed in every evaluation", name, why)


def _kept_path(strategy: Strategy, code: str, panel: Panel, instrument: str, bars: pd.DataFrame) -> Path:
    """Where a rule's positions on an instrument's bars are kept: by the code's version, the rule and the bars it is
    given, every field of every bar hashed (`Panel.digest`: a bar written anew, another start of history, other
    positions)."""
    name = instrument.replace(":", "_").replace("/", "-")
    return KEPT_DIR / code / panel.timeframe / strategy.name / f"{name}-{panel.digest(instrument, bars)}.npz"


def _read_kept(path: Path) -> dict[str, np.ndarray]:
    """The positions kept at `path` by signal configuration, on the instrument's own bars; none when nothing is kept."""
    try:
        with np.load(path, allow_pickle=False) as f:
            return {str(sig): f[f"p{k}"] for k, sig in enumerate(f["sigs"])}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as e:
        LOG.warning("%s: positions kept there cannot be read (%r): worked out again", path, e)
        return {}


def _write_kept(path: Path, kept: dict[str, np.ndarray]) -> None:
    """Keep `kept` at `path`, whole or not at all: written aside and moved into place (several processes of a batch
    may write the same file; the last one's stays)."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        sigs = list(kept)
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, sigs=np.array(sigs), **{f"p{k}": kept[s] for k, s in enumerate(sigs)})
        os.replace(tmp, path)
    except OSError as e:
        LOG.warning("%s: positions not kept (%r): the next evaluation works them out again", path, e)
        tmp.unlink(missing_ok=True)


def _same_bars(strategy: Strategy, panel: Panel, signals: dict) -> None:
    bars = signals.setdefault("index", panel.index)
    if bars is not panel.index and not bars.equals(panel.index):
        raise ValueError(f"{strategy.name}: signals kept for other bars ({len(bars)}) than this panel's "
                         f"({len(panel.index)}): they are shared only between memberships of the same bars")


def _aligned(position, index: pd.DatetimeIndex) -> np.ndarray:
    """A rule's output on its own bars as its position on the panel's bars: held through a bar it missed (and through
    its own bars without a decision), 0 before its first, within [-1, 1]."""
    s = pd.Series(position, dtype="float64")
    if not s.index.is_unique:                       # the reindex refuses it, as it always did
        return s.reindex(index).ffill().clip(-1.0, 1.0).fillna(0.0).to_numpy()
    v = np.full(len(index), np.nan)
    at = index.get_indexer(s.index)
    found = at >= 0
    v[at[found]] = s.to_numpy()[found]
    last = np.where(np.isnan(v), 0, np.arange(len(v)))
    np.maximum.accumulate(last, out=last)           # the last bar with a decision: a forward fill
    v = np.clip(v[last], -1.0, 1.0)
    v[np.isnan(v)] = 0.0
    return v


def _positions(strategy: Strategy, panel: Panel, sp: dict, signals: dict | None) -> pd.DataFrame:
    """Each instrument's position in [-1, 1] from its own bars (`_aligned`), taken from `signals` where they are there
    already (see `Strategy.target`)."""
    if signals is not None:
        _same_bars(strategy, panel, signals)
    sig = repr(sorted(sp.items()))
    out = np.empty((len(panel.index), len(panel.ids)), order="F")
    for k, i in enumerate(panel.ids):
        kept = None if signals is None else signals.get((sig, i))
        if kept is None:
            kept = _aligned(strategy.fn(panel.one(i), **sp), panel.index)
            if signals is not None:
                signals[(sig, i)] = _packed(kept)
        out[:, k] = kept
    return pd.DataFrame(out, index=panel.index, columns=panel.ids)


def _packed(raw: np.ndarray) -> np.ndarray:
    """Positions kept for a grid's next configuration in the fewest bytes that hold them exactly: a rule's are mostly
    -1, 0 and 1, and a list's float64 positions for each signal configuration of a grid took gigabytes."""
    for dtype in (np.int8, np.float32):
        small = raw.astype(dtype)
        if np.array_equal(small.astype(np.float64), raw):
            return small
    return raw


def seats(listed: pd.DataFrame, wanted: pd.DataFrame, liquidity_at: pd.DataFrame | None = None) -> pd.DataFrame:
    """The instruments a rule trades on each bar. The universe's members, except that a name leaving it while its rule
    holds a position keeps its seat until that trade ends (the position closes or turns to the other side, which is
    then not opened), and a name joining it waits for a free seat, the longest-waiting first and, of those that joined
    on one bar, the more liquid there first (`liquidity_at`, as `slotted` gives its slots; none: the columns' order).
    There are as many seats as the universe holds names on that bar, so a seat's share of capital does not change.

    The seats change only where the universe does or a kept trade ends, so the walk goes from one such bar to the next.
    """
    listed_a = listed.to_numpy(dtype=bool)
    side = np.sign(wanted.to_numpy())
    rank = None
    if liquidity_at is not None:
        if not (liquidity_at.index.equals(listed.index) and liquidity_at.columns.equals(listed.columns)):
            liquidity_at = liquidity_at.reindex(index=listed.index, columns=listed.columns)
        rank = np.nan_to_num(liquidity_at.to_numpy(dtype=np.float64), nan=-np.inf)
    n_bars, n = listed_a.shape
    changed = np.vstack([np.zeros((1, n), dtype=bool), side[1:] != side[:-1]])
    bar_of_change = np.where(changed, np.arange(n_bars)[:, None], n_bars)
    ends = np.minimum.accumulate(bar_of_change[::-1], axis=0)[::-1]       # the first bar >= t where the side changes
    ends_after = np.vstack([ends[1:], np.full((1, n), n_bars)])          # ... strictly after t: where a trade at t ends
    list_bars = [0] + (np.flatnonzero((listed_a[1:] != listed_a[:-1]).any(axis=1)) + 1).tolist()
    out = np.zeros((n_bars, n), dtype=bool)
    seated: set[int] = set()
    queue: list[int] = []
    kept: list[tuple[int, int]] = []          # (bar its seat is given up, instrument): trades kept past leaving
    k, t = 0, 0
    while True:
        nxt_list = list_bars[k] if k < len(list_bars) else n_bars
        nxt_kept = kept[0][0] if kept else n_bars
        u = min(nxt_list, nxt_kept)
        if seated:
            out[t:u, sorted(seated)] = True
        if u >= n_bars:
            break
        while kept and kept[0][0] == u:
            seated.discard(heapq.heappop(kept)[1])
        if u == nxt_list:
            k += 1
            members = set(np.flatnonzero(listed_a[u]).tolist())
            kept = [(e, i) for e, i in kept if i not in members]          # back in the universe: a member again
            heapq.heapify(kept)
            keeping = {i for _, i in kept}
            for i in sorted(seated - members - keeping):
                if u > 0 and side[u - 1, i] != 0 and side[u, i] == side[u - 1, i]:
                    heapq.heappush(kept, (int(ends_after[u - 1, i]), i))
                else:
                    seated.discard(i)
            joined = [i for i in sorted(members - seated) if i not in queue]
            if rank is not None:
                joined.sort(key=lambda i: -rank[u, i])              # stable: the columns' order among equals
            queue = [i for i in queue if i in members] + joined
        capacity = int(listed_a[u].sum())
        while queue and len(seated) < capacity:
            seated.add(queue.pop(0))
        t = u
    return pd.DataFrame(out, index=listed.index, columns=listed.columns)


def slotted(wanted: pd.DataFrame, k: int, liquidity_at: pd.DataFrame) -> pd.DataFrame:
    """`wanted` with at most k trades held at once. A trade (an instrument on one side, from the bar its position turns
    on or to the other side to the bar before it turns off or flips) takes a free slot on its first bar and keeps it
    to its last; a trade that finds every slot taken is not traded, not even once a slot frees. Of the trades starting
    on one bar, the instrument more liquid there (`liquidity_at`; none: last) takes a slot first, the columns' order
    deciding between equals."""
    w = wanted.to_numpy(dtype=np.float64)
    col, start, end = _trades_of(w)
    if not (liquidity_at.index.equals(wanted.index) and liquidity_at.columns.equals(wanted.columns)):
        liquidity_at = liquidity_at.reindex(index=wanted.index, columns=wanted.columns)
    rank = np.nan_to_num(liquidity_at.to_numpy(dtype=np.float64)[start, col], nan=-np.inf)
    taken = _slots_taken(start, end, np.lexsort((col, -rank, start)), k)
    return pd.DataFrame(_kept(w, col, start, end, taken), index=wanted.index, columns=wanted.columns)


@njit(cache=True)
def _trades_of(w):
    """Each trade of the positions `w` (bars x instruments), instrument by instrument then bar by bar: its instrument,
    its first bar (its position turns on or to the other side) and its last (the bar before it turns off or flips)."""
    n, m = w.shape
    count = 0
    for j in range(m):
        prev = 0.0
        for t in range(n):
            side = np.sign(w[t, j])
            if side != 0.0 and side != prev:
                count += 1
            prev = side
    col, start, end = np.empty(count, np.int64), np.empty(count, np.int64), np.empty(count, np.int64)
    k = 0
    for j in range(m):
        prev = 0.0
        for t in range(n):
            side = np.sign(w[t, j])
            if side != 0.0 and side != prev:
                col[k], start[k] = j, t
            if side != 0.0 and side != (np.sign(w[t + 1, j]) if t + 1 < n else 0.0):
                end[k] = t
                k += 1
            prev = side
    return col, start, end


@njit(cache=True)
def _slots_taken(start, end, order, k):
    """Which trades get a slot, taken in `order`: a trade takes one when fewer than k of those taken before it still
    hold theirs on its first bar."""
    taken = np.zeros(len(start), np.bool_)
    ends = np.empty(k, np.int64)                  # the last bars of the trades holding a slot
    held = 0
    for j in order:
        kept = 0
        for i in range(held):
            if ends[i] >= start[j]:
                ends[kept] = ends[i]
                kept += 1
        held = kept
        if held < k:
            ends[held] = end[j]
            held += 1
            taken[j] = True
    return taken


@njit(cache=True)
def _kept(w, col, start, end, taken):
    """The positions of the trades taken, 0 elsewhere."""
    n, m = w.shape
    out = np.zeros((m, n)).T
    for i in range(len(col)):
        if taken[i]:
            for t in range(start[i], end[i] + 1):
                out[t, col[i]] = w[t, col[i]]
    return out


def _slot_liquidity(panel: Panel) -> pd.DataFrame:
    """Each instrument's liquidity at each bar, for `slotted`: its median dollar volume a day as the lists are ranked
    by (`data.bars.liquidity`, over the panel's own bars: intraday bars miss the auctions, which matters only between
    names of close liquidity), and a currency pair's, whose quotes carry no volume, its market's turnover rank
    (`FX_BY_TURNOVER`). Kept on the panel for the grid's other configurations."""
    got = panel.memo.get("slot_liquidity")
    if got is None:
        day = (panel.index - pd.Timedelta(microseconds=1)).normalize()
        got = liquidity(panel).reindex(day)
        got.index = panel.index
        for i in got.columns:
            pair = panel.instruments[i].symbol
            if pair in FX_BY_TURNOVER:
                got[i] = float(len(FX_BY_TURNOVER) - FX_BY_TURNOVER.index(pair))
        panel.memo["slot_liquidity"] = got
    return got


def rule(grid: dict | None = None, name: str | None = None, description: str = "", grade: Callable | None = None,
         prepare: Callable | None = None, exposure: bool = False, model: bool = False, market: bool = False):
    """A rule: one instrument's bars in, its position out. `prepare(panel, configs, member)`, when given, runs once
    before an evaluation's grid, with the list's membership (None: every instrument held from its first bar): a rule
    whose per-instrument work is heavy and shared by the grid does it there for every instrument at once (in
    parallel), only on the bars the evaluation reads, and keeps it where the rule then finds it; the positions do not
    change. It returns what forgets that work once the evaluation is over (None: nothing to forget): work done for
    one list's membership must not serve another evaluation.

    `exposure`: the rule's position is a holding to keep while the name is listed, never closed by the rule and only
    resized (vol_managed), not a trade. On a universe that changes, a trade keeps a leaver's seat until the rule ends
    it (`seats`); a holding would keep it for ever, and the newcomers would never be bought. An exposure rule's names
    are the universe's members instead: a leaver sold at the re-pick, a newcomer bought at once.

    `model`: the function fits a model itself, drawing with its module's `SEED` (ml_feature_search; a rule with a
    `prepare` step fits one there, ml_direction), so the robustness check of other seeds varies it.

    `market`: the function reads its market's index besides its bars (`strategy_lab.market`: the index's daily closes
    as each bar knows them), so its positions are kept on disk under the index's bars too."""
    def wrap(fn):
        return Strategy(name or fn.__name__, "rule", fn, grid or {}, description or (fn.__doc__ or "").strip(), grade,
                        prepare, exposure=exposure, model=model, market=market)
    return wrap


def panel(grid: dict | None = None, name: str | None = None, description: str = "", book: bool = True):
    """A panel strategy: the universe's bars in, its weights out. `book`: the weights are a book rebalanced whole
    (an allocation: when any weight moves, every position is traded back to its weight); off, each weight is a
    position of its own, traded only when it moves (events taken and left one by one, as a rule's trades)."""
    def wrap(fn):
        return Strategy(name or fn.__name__, "panel", fn, grid or {}, description or (fn.__doc__ or "").strip(),
                        book=book)
    return wrap


def load(name: str) -> Strategy:
    """The strategy of strategies/<name>.py."""
    found = [v for v in vars(importlib.import_module(f"strategies.{name}")).values() if isinstance(v, Strategy)]
    if [s.name for s in found] != [name]:
        raise ValueError(f"strategies/{name}.py must hold one strategy, named {name}; it holds {[s.name for s in found]}")
    return found[0]


def with_short(long: pd.Series, short: pd.Series, long_only: bool) -> pd.Series:
    """A rule's position from its long side and its mirror, the short side (1.0 where each holds): the long side alone
    with `long_only`, else long minus short, flat on a bar where both hold. The walk-forward chooses between the two
    on each list's past: a market without an upward drift (a currency pair, a future) has no side to favour."""
    return long if long_only else long - short


def hold_between(enter: pd.Series, leave: pd.Series) -> pd.Series:
    """Long (1.0) from a bar where `enter` holds until a bar where `leave` holds, flat before the first entry; on a bar
    where both hold, `enter` wins."""
    state = pd.Series(np.nan, index=enter.index)
    state[leave] = 0.0
    state[enter] = 1.0
    return state.ffill().fillna(0.0)


def rebalanced(w: pd.DataFrame | pd.Series, every: str) -> pd.DataFrame | pd.Series:
    """Weights (a panel's) or a position (a rule's) decided at the first bar of each calendar period (`period_starts`:
    "M" a month, "W" a week, "<n>D" n days) and held until the next; a decision uses the data up to its bar."""
    keep = period_starts(w.index, every)
    held = w.where(keep if isinstance(w, pd.Series) else np.broadcast_to(keep[:, None], w.shape))
    return held.ffill().fillna(0.0)


def period_starts(index: pd.DatetimeIndex, every: str) -> np.ndarray:
    """The bars that open a calendar period, by their close time (UTC): "M" a month, "W" a week from Monday, "<n>D" n
    days counted from 1970-01-01. A decision made once a period falls on the first close at or after the period's
    turn: on a market open around the clock the turn itself (a coin's bar of the month's last hour or day closes at
    the next month's first instant), on one with sessions the new period's first bar."""
    t = index.tz_convert("UTC") if index.tz is not None else index
    if every == "M":
        key = np.asarray(t.year * 12 + t.month)
    elif every == "W":
        key = np.asarray((t.normalize() - pd.to_timedelta(t.dayofweek, unit="D")).asi8)
    elif every.endswith("D") and every[:-1].isdigit():
        key = t.asi8 // (int(every[:-1]) * 86_400 * 10**9)
    else:
        raise ValueError(f"a period is 'M', 'W' or '<n>D', got {every!r}")
    return np.r_[True, key[1:] != key[:-1]] if len(key) else np.zeros(0, dtype=bool)


def bars_in(x: pd.DataFrame | Panel, *, days: float | None = None, months: float | None = None) -> int:
    """How many bars of `x` (a rule's bars, whose `attrs` name their timeframe and asset class, or a panel) a span of
    trading days or calendar months takes: a strategy whose source gives its time in days or months (a 200-day
    average, a 12-month return) means the same time on 1h, 4h and 1d bars (config.BARS_PER_DAY, a month a twelfth of
    config.TRADING_DAYS: 21 sessions of a US listing, 30.4 days of a coin)."""
    if (days is None) == (months is None):
        raise ValueError("a span is given in days or in months")
    if isinstance(x, Panel):
        timeframe, classes = x.timeframe, {ins.asset_class for ins in x.instruments.values()}
    else:
        if "timeframe" not in x.attrs or "asset_class" not in x.attrs:
            raise ValueError("bars without their timeframe and asset class (Panel.one sets them): no span in bars")
        timeframe, classes = x.attrs["timeframe"], {x.attrs["asset_class"]}
    per_day = {config.BARS_PER_DAY[c][timeframe] for c in classes}
    per_year = {config.TRADING_DAYS[c] for c in classes}
    if len(per_day) != 1 or len(per_year) != 1:
        raise ValueError(f"markets of different hours in one panel ({sorted(classes)}): no common span in bars")
    n = days if days is not None else months * per_year.pop() / 12
    return max(1, int(round(n * per_day.pop())))


def calendar_months(months: float) -> pd.Timedelta:
    """A span of calendar months as a fixed length of time (a rolling window's), a month a twelfth of 365.25 days."""
    return pd.Timedelta(days=months * 365.25 / 12)


def as_of(x: pd.DataFrame | pd.Series, *, months: float) -> pd.DataFrame | pd.Series:
    """`x` as it stood `months` calendar months before each of its times (the same date that many months back, as
    T-bills' trailing return is taken): its last value at or before that instant, NaN before it has one. A panel
    strategy's lookback: counted in bars, it counts the bars of every name of the list at once, and where their
    sessions sit on different clocks (the stock lists' hours of 2020-01..06, on the whole and on the half hour, about
    ten bars a session together where each name has seven) a year of bars was eight months."""
    back = x.index - (pd.DateOffset(months=int(months)) if float(months).is_integer() else calendar_months(months))
    pos = x.index.searchsorted(back, side="right") - 1
    out = x.iloc[np.maximum(pos, 0)].copy()
    out.iloc[np.flatnonzero(pos < 0)] = np.nan
    out.index = x.index
    return out


def ends(score: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The names at each end of every bar's ranking by `score` (NaN: not ranked), as many at the bottom as at the top:
    `frac` of the names ranked, at least one, at most half (1.0 where held). Taken as the ranks at or below `frac`
    and at or above 1 - `frac`, the top had held one name more than the bottom (2 against 1 of ten names at 0.1), and
    a bottom left empty had left the whole book flat (seven names at 0.1)."""
    n = score.notna().sum(axis=1).to_numpy()
    k = np.minimum(np.maximum(np.floor(n * frac + 1e-9), 1.0), n // 2)[:, None]
    order = score.rank(axis=1, method="first").to_numpy()             # 1: the lowest score; NaN: not ranked
    with np.errstate(invalid="ignore"):
        low, high = order <= k, order > n[:, None] - k
    return (pd.DataFrame(low.astype(float), index=score.index, columns=score.columns),
            pd.DataFrame(high.astype(float), index=score.index, columns=score.columns))


def long_short(score: pd.DataFrame, live: pd.DataFrame, frac: float, min_names: int = 6) -> pd.DataFrame:
    """Long the top `frac` of the universe's names by score, short the bottom `frac` (`ends`: as many names on each
    side), each leg equal-weighted to half the capital; flat while fewer than `min_names` names have a score."""
    s = score.where(live)
    n = s.notna().sum(axis=1)
    shorts, longs = ends(s, frac)
    wl = longs.div(longs.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    ws = shorts.div(shorts.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    return (wl - ws).where(n >= min_names, 0.0).fillna(0.0)
