"""What a fill costs a side, as a fraction of the notional it trades.

An instrument pays its asset class's commission and half-spread (`config.COSTS`), the same on every bar. A spot quote
of a commodity pays its class's commission and half the spread its broker quoted at the fill (`data.spreads`), bar by
bar: at a bar's open for a fill there (next_open), inside the bar for a stop or a target, at its close for a fill at
the close (next_close), over the bar's price there. A panel keeps its quotes' rates apart, a column a quote
(`Rates.column`), so a list of hundreds of stocks does not carry a rate for every bar of every name.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy_lab.config import COSTS
from strategy_lab.data import spreads
from strategy_lab.data.bars import Panel, load_panel
from strategy_lab.data.instruments import parse

WHERE = ("at_open", "during", "at_close")


@dataclass(frozen=True)
class Rates:
    """A panel's costs a side, by instrument and bar (the module's rules)."""
    flat: np.ndarray        # (instruments,) its class's commission and half-spread, where no spread is on record
    column: np.ndarray      # (instruments,) int64: its column in the quoted rates, -1 for one charged `flat`
    at_open: np.ndarray     # (bars, quotes) at each bar's open
    during: np.ndarray      # (bars, quotes) inside each bar
    at_close: np.ndarray    # (bars, quotes) at each bar's close

    def of(self, j: int, where: str) -> np.ndarray:
        """Instrument j's rate on every bar, `where` in the bar (WHERE)."""
        k = self.column[j]
        if k < 0:
            return np.full(self.at_open.shape[0], self.flat[j])
        return getattr(self, where)[:, k]

    def row(self, t: int, where: str) -> np.ndarray:
        """Every instrument's rate on bar t, `where` in the bar."""
        out = self.flat.copy()
        quoted = self.column >= 0
        out[quoted] = getattr(self, where)[t, self.column[quoted]]
        return out


def flat_rate(asset_class: str) -> float:
    c = COSTS[asset_class]
    return (c.commission_bps + c.half_spread_bps) / 1e4


def rates(panel: Panel, classes: dict[str, str] | None = None) -> Rates:
    """The panel's `Rates`, worked out once per panel and version of the spreads it read. `classes`: the cost class
    an asset class is charged at instead of its own (buy & hold holds a coin on spot, not on its perpetual)."""
    classes = classes or {}
    version = tuple(spreads.version(i) for i in panel.ids)
    key = ("rates", tuple(sorted(classes.items())))
    got = panel.memo.get(key)
    if got is None or got[0] != version:
        got = panel.memo[key] = version, _rates(panel, classes)
    return got[1]


def _rates(panel: Panel, classes: dict[str, str]) -> Rates:
    ids = panel.ids
    cls = [classes.get(panel.instruments[i].asset_class, panel.instruments[i].asset_class) for i in ids]
    flat = np.array([flat_rate(c) for c in cls], dtype=np.float64)
    quoted = [j for j, i in enumerate(ids) if spreads.spread_path(i) is not None]
    column = np.full(len(ids), -1, dtype=np.int64)
    n = len(panel.index)
    out = {w: np.zeros((n, len(quoted))) for w in WHERE}
    for k, j in enumerate(quoted):
        i = ids[j]
        column[j] = k
        commission = COSTS[cls[j]].commission_bps / 1e4
        o, d, c = spreads.per_bar(i, panel.index, panel.timeframe)
        opn, close = panel.open[i].to_numpy(dtype=np.float64), panel.close[i].to_numpy(dtype=np.float64)
        out["at_open"][:, k] = _nearest(commission + 0.5 * o / opn)
        out["during"][:, k] = _nearest(commission + 0.5 * d / close)
        out["at_close"][:, k] = _nearest(commission + 0.5 * c / close)
    return Rates(flat, column, out["at_open"], out["during"], out["at_close"])


def _nearest(rate: np.ndarray) -> np.ndarray:
    """A rate on every bar: one the instrument has no price on (no fill happens there) takes the one before it, or
    after it before its first bar."""
    s = pd.Series(rate).ffill().bfill()
    return s.fillna(0.0).to_numpy()


def typical_bps(instrument_id: str) -> float:
    """A side's cost of an instrument, in basis points, as a list's figures show it: its class's flat one, or the median
    of what a fill at the open of its hourly bars cost over their last year."""
    if spreads.spread_path(instrument_id) is None:
        return flat_rate(parse(instrument_id).asset_class) * 1e4
    panel = load_panel([instrument_id], "1h", fields=("open", "close"))
    last = panel.index >= panel.index[-1] - pd.Timedelta(days=365)
    return float(np.median(rates(panel).of(0, "at_open")[last]) * 1e4)
