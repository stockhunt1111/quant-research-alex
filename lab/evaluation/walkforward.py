"""Walk-forward selection: parameters are chosen on the past and used unchanged on the next window.

Expanding windows: the first test window starts `first_train` after the first bar; each test window [t, t + test)
uses the configuration with the best Sharpe on ALL the history before it [start, t), stepping by `test`; with
`train`, on the last `train` of it only [t - train, t) (a rolling window). The out-of-sample record is the chosen
configurations' returns stitched over the test windows.

`choose_with_peers` adds a check on similar instruments (how it is
judged was set with the user on 2026-09-24): over the same past window a
configuration must have beaten holding (a higher Sharpe than the peer's buy-and-hold; than 0 on a currency pair or
crude, whose buy-and-hold is cash) on at least half of the peers with enough history there, and the best of those by
the instrument's own Sharpe is taken; with none, the next window holds no position; with fewer than MIN_PEERS peers
with enough history the check is not made.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_TRAIN_DAYS_WITH_RETURNS = 60
MIN_PEERS = 3
PEER_SHARE = 0.5


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def folds(start: pd.Timestamp, end: pd.Timestamp, first_train: str, test: str, train: str | None = None) -> list[Fold]:
    first, te = pd.Timedelta(first_train), pd.Timedelta(test)
    back = None if train is None else pd.Timedelta(train)
    out = []
    t = start + first
    while t < end:
        out.append(Fold(start if back is None else max(start, t - back), t, t, min(t + te, end)))
        t += te
    return out


def sharpe(d: pd.Series) -> float:
    d = d.dropna()
    sd = d.std(ddof=1)
    if len(d) < 2 or not np.isfinite(sd) or sd == 0:
        return -np.inf
    return float(d.mean() / sd)


def choose(daily: pd.DataFrame, fold_list: list[Fold]) -> list[tuple[int | None, float]]:
    """For each fold, the column (config index) with the best train-window Sharpe, and that Sharpe; configurations
    that traded on fewer than MIN_TRAIN_DAYS_WITH_RETURNS days of the window are skipped, and with none left the
    window has no choice (None, a Sharpe of -inf): nothing it could be chosen on, it holds no position (`stitch`).
    Each window's Sharpe is `sharpe` of the column over it, with the same arithmetic on the same values; a day a column
    has no value (NaN: before its record starts) is neither traded nor counted."""
    x = np.asfortranarray(daily.to_numpy(dtype=np.float64))          # a column's window is a contiguous slice
    traded = np.vstack([np.zeros((1, x.shape[1]), dtype=np.int64), np.cumsum((x != 0) & ~np.isnan(x), axis=0)])
    out = []
    for f in fold_list:
        lo, hi = daily.index.searchsorted(f.train_start), daily.index.searchsorted(f.train_end)
        best, best_s = None, -np.inf
        for c in range(x.shape[1]):
            if traded[hi, c] - traded[lo, c] < MIN_TRAIN_DAYS_WITH_RETURNS:
                continue
            val = _sharpe_of(x[lo:hi, c])
            if best is None or val > best_s:
                best, best_s = c, val
        out.append((None if best is None else int(daily.columns[best]), float(best_s)))
    return out


def _sharpe_of(v: np.ndarray) -> float:
    """`sharpe` of a float array: pandas' mean and std(ddof=1) are this sum over the count and this two-pass variance."""
    if np.isnan(v).any():
        v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return -np.inf
    mean = v.sum(dtype=np.float64) / np.float64(n)
    sd = np.sqrt(((mean - v) ** 2).sum(dtype=np.float64) / np.float64(n - 1))
    if not np.isfinite(sd) or sd == 0:
        return -np.inf
    return float(mean / sd)


def choose_with_peers(daily: pd.DataFrame, fold_list: list[Fold],
                      peers: list[tuple[pd.DataFrame, pd.Series, pd.Series]]) -> list[tuple[int | None, float, int]]:
    """`choose`, among the configurations that beat holding on at least PEER_SHARE of the peers over the window.
    `peers`: each peer's daily returns by configuration (the columns of `daily`), its buy-and-hold's (zero: cash) and
    the days it has a price on (its history). Per fold: the column (None: no configuration passed, no position), its
    train Sharpe, and how many peers judged it (0: the check was not made)."""
    x = np.asfortranarray(daily.to_numpy(dtype=np.float64))
    traded = np.vstack([np.zeros((1, x.shape[1]), dtype=np.int64), np.cumsum((x != 0) & ~np.isnan(x), axis=0)])
    sums = [(_cumulative(p.to_numpy(dtype=np.float64)), _cumulative(b.to_numpy(dtype=np.float64)[:, None]),
             np.r_[0, np.cumsum(priced.reindex(p.index, fill_value=False).to_numpy(dtype=np.int64))], p.index)
            for p, b, priced in peers]
    out = []
    for f in fold_list:
        lo, hi = daily.index.searchsorted(f.train_start), daily.index.searchsorted(f.train_end)
        wins, judged = np.zeros(x.shape[1]), 0
        for (s1, s2, n), (b1, b2, bn), priced_days, idx in sums:
            a, z = idx.searchsorted(f.train_start), idx.searchsorted(f.train_end)
            if priced_days[z] - priced_days[a] < MIN_TRAIN_DAYS_WITH_RETURNS:
                continue
            judged += 1
            wins += _window_sharpe(s1, s2, n, a, z) > _window_sharpe(b1, b2, bn, a, z, flat=0.0)[0]
        eligible = np.ones(x.shape[1], dtype=bool) if judged < MIN_PEERS else wins >= PEER_SHARE * judged
        best, best_s = None, -np.inf
        for c in np.flatnonzero(eligible):
            if traded[hi, c] - traded[lo, c] < MIN_TRAIN_DAYS_WITH_RETURNS:
                continue
            val = _sharpe_of(x[lo:hi, c])
            if val > best_s:
                best, best_s = c, val
        if best is None and eligible.any():
            best = int(np.flatnonzero(eligible)[0])
        out.append((None if best is None else int(daily.columns[best]), float(best_s),
                    judged if judged >= MIN_PEERS else 0))
    return out


def _cumulative(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Running sums of the values and their squares, and running counts, with a leading zero row (NaN left out)."""
    ok = ~np.isnan(v)
    z = np.where(ok, v, 0.0)
    pad = lambda a: np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])      # noqa: E731
    return pad(z), pad(z * z), pad(ok.astype(np.float64))


def _window_sharpe(s1: np.ndarray, s2: np.ndarray, n: np.ndarray, a: int, z: int, flat: float = -np.inf) -> np.ndarray:
    """Sharpe of each column over rows [a, z) from running sums (as `sharpe`), `flat` when it cannot be computed: -inf
    for a configuration (never chosen), 0 for a buy-and-hold that does not move (cash)."""
    cnt = n[z] - n[a]
    mean = (s1[z] - s1[a]) / np.where(cnt > 0, cnt, 1)
    var = ((s2[z] - s2[a]) - cnt * mean ** 2) / np.where(cnt > 1, cnt - 1, 1)
    sd = np.sqrt(np.clip(var, 0.0, None))
    return np.where((cnt > 1) & (sd > 1e-15), mean / np.where(sd > 0, sd, 1), flat)


def stitch(daily: pd.DataFrame, fold_list: list[Fold], choices: list[int | None]) -> pd.Series:
    """The chosen configurations' returns over their test windows; a window without a choice holds no position."""
    parts = []
    for f, c in zip(fold_list, choices):
        s = daily[daily.columns[0]] * 0.0 if c is None else daily[c]
        parts.append(s[(s.index >= f.test_start) & (s.index < f.test_end)])
    return pd.concat(parts) if parts else pd.Series(dtype="float64")
