"""Machine-learning direction model (week-1 idea list: ML strategies), refit on the past only.

Features are causal indicators of the bar; the label is whether the close `horizon` bars later is higher. The
model is refit on the walk-forward calendar (`strategy_lab.ml.refit_rows`: every window start, a quarter apart on
every timeframe, and twice in the first year), from the first refit with `min_train` rows on, on an expanding window
of rows whose label was already known at the refit time (row j is usable once j + horizon <= t), then predicts the
bars up to the next refit. Long while the predicted probability of a rise exceeds `threshold`; unless long only,
short while it is below 1 - `threshold`. Needs the [ml] extra (lightgbm).
"""
from __future__ import annotations

import hashlib
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from strategy_lab import config
from strategy_lab import indicators as ind
from strategy_lab.ml import refit_rows
from strategy_lab.strategy import rule, with_short

SEED = config.SEED          # the model's random draws: the robustness check of other seeds varies it
# the list whose blocks the fits `prepare` made cover (None: every block); the rule reads the fits of this scope
_SCOPE: str | None = None


def features(bars: pd.DataFrame) -> pd.DataFrame:
    c = bars["close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=bars.index)
    for n in (1, 5, 20, 60):
        f[f"ret_{n}"] = c / c.shift(n) - 1.0
    f["rsi_14"] = ind.rsi(c, 14)
    f["rsi_2"] = ind.rsi(c, 2)
    f["atr_pct"] = ind.atr(bars["high"], bars["low"], c, 14) / c
    for n in (20, 50, 200):
        f[f"dist_ema_{n}"] = c / ind.ema(c, n) - 1.0
    f["vol_20"] = r.rolling(20).std()
    f["vol_ratio"] = f["vol_20"] / r.rolling(120).std()
    vsd = bars["volume"].rolling(60).std()
    # FX has no volume: a constant (zero) series has no spread, and 0/0 would drop every row of the pair
    f["volume_z"] = ((bars["volume"] - bars["volume"].rolling(60).mean()) / vsd).where(vsd > 0, 0.0)
    f["ibs"] = ind.ibs(bars)
    f["adx_14"] = ind.adx(bars["high"], bars["low"], c, 14)
    return f.replace([np.inf, -np.inf], np.nan)


def _model(seed: int):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, min_child_samples=40, random_state=seed, n_jobs=1, verbose=-1)


def predict_expanding(X: pd.DataFrame, y: pd.Series, horizon: int, min_train: int,
                      spans: list[tuple[int, int]] | None = None, exit_at: float = 0.0, shorts: bool = False,
                      seed: int | None = None) -> pd.Series:
    """Out-of-sample P(rise): the bars from one refit to the next are predicted by a model fit on labels known before
    them; the refits are the walk-forward calendar's from the row `min_train` on.

    With `spans` (the rows the instrument is in its list) only the blocks a list evaluation reads are fitted, the
    others left NaN: a block's model is fit on the past alone, so the blocks fitted come out the same either way.
    `seed`: the model's random draws, the module's SEED when None (a process that fits for another passes it)."""
    seed = SEED if seed is None else seed
    p = np.full(len(X), np.nan)
    valid = X.notna().all(axis=1).to_numpy()
    n = len(X)
    starts = [t for t in refit_rows(X.index) if t >= min_train]
    for t, nxt in zip(starts, starts[1:] + [n]):
        if spans is not None and not _read(t, nxt, spans, p, exit_at, shorts):
            continue
        usable = np.arange(n) <= t - horizon                 # the label of row j is known once j + horizon <= t
        train = usable & valid & y.notna().to_numpy()
        if train.sum() < min_train // 2 or y[train].nunique() < 2:
            continue
        m = _model(seed).fit(X[train], y[train].astype(int))
        block = np.arange(t, nxt)
        ok = block[valid[block]]
        if len(ok):
            p[ok] = m.predict_proba(X.iloc[ok])[:, 1]
    return pd.Series(p, index=X.index)


def _read(lo: int, hi: int, spans: list[tuple[int, int]], p: np.ndarray, exit_at: float, shorts: bool) -> bool:
    """Whether a list evaluation reads the predictions of rows lo..hi-1: they fall in a stretch in the list, or after
    one while the trade held on its last row is still open (a name that leaves keeps its seat until its trade ends:
    a long until the probability falls to the grid's lowest threshold `exit_at`, a short, where the grid has one,
    until it rises to 1 - `exit_at`)."""
    if any(lo <= b and hi > a for a, b in spans):
        return True
    ended = [b for _, b in spans if b < lo]
    if not ended:
        return False
    since = p[max(ended):lo]
    if np.isnan(since).any():
        return False
    return bool((since > exit_at).all() or (shorts and (since < 1.0 - exit_at).all()))


def _stretches(member: pd.Series | None, index: pd.DatetimeIndex) -> list[tuple[int, int]] | None:
    """An instrument's stretches in its list as (first, last) rows of its own bars, each from the row before it (on a
    bar the instrument did not print its position is its last bar's); None without a list (held from its first bar)."""
    if member is None:
        return None
    m = np.concatenate(([False], member.reindex(index).fillna(False).to_numpy(dtype=bool), [False]))
    edges = np.diff(m.astype(np.int8))
    return [(max(a - 1, 0), b - 1) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


# The model depends on the bars, (horizon, min_train) and its seed only: the grid's thresholds, sides and slots reuse
# one fit instead of refitting it for each. A fit `prepare` made for a list covers only the blocks that list reads, so
# it is kept under that list's scope; a fit of every block (scope None) serves any list.
_PREDICTIONS: dict[tuple, pd.Series] = {}


def _digest(X: pd.DataFrame, y: pd.Series) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes())
    h.update(pd.util.hash_pandas_object(y, index=True).to_numpy().tobytes())
    return h.hexdigest()


def _key(digest: str, horizon: int, min_train: int, scope: str | None) -> tuple:
    return digest, horizon, min_train, SEED, scope


def _scope(panel, member: pd.DataFrame, exit_at: float, shorts: bool) -> str:
    """Which blocks a list evaluation reads: its membership on the panel's bars and the exit thresholds of its grid."""
    h = hashlib.sha256(np.ascontiguousarray(member.reindex(index=panel.index, columns=panel.ids).fillna(False)
                                            .to_numpy(dtype=bool)).tobytes())
    h.update(repr((list(panel.ids), panel.index[0], panel.index[-1], len(panel.index), exit_at, shorts)).encode())
    return h.hexdigest()[:16]


def _label(bars: pd.DataFrame, horizon: int) -> pd.Series:
    return (bars["close"].shift(-horizon) > bars["close"]).astype(float).where(bars["close"].shift(-horizon).notna())


def _cached_predictions(X: pd.DataFrame, y: pd.Series, horizon: int, min_train: int) -> pd.Series:
    d = _digest(X, y)
    full = _key(d, horizon, min_train, None)
    for key in ([_key(d, horizon, min_train, _SCOPE)] if _SCOPE is not None else []) + [full]:
        if key in _PREDICTIONS:
            return _PREDICTIONS[key]
    if len(_PREDICTIONS) >= 4096:
        _PREDICTIONS.clear()
    _PREDICTIONS[full] = predict_expanding(X, y, horizon, min_train)
    return _PREDICTIONS[full]


def _predict(args: tuple) -> pd.Series:
    bars, horizon, min_train, spans, exit_at, shorts, seed = args
    return predict_expanding(features(bars), _label(bars, horizon), horizon, min_train, spans, exit_at, shorts, seed)


def prepare(panel, configs, member=None):
    """Fit every instrument's models before the grid, only on the blocks the evaluation reads (in a list a name was in
    the Top-10 for a few months, and two thirds of the fitting of Crypto Top-10 1h came after the names left it), one
    process per core in the main process and the longest histories first: one instrument's fits do not depend on
    another's, and they were the evaluation's whole cost. The rule then finds them in its cache under this list's
    scope. The returned undo gives the scope back to the evaluation that was running before (a neighbouring list or
    another seed is evaluated inside a list's own) and forgets the fits made for this list's membership, and any fit
    made with another seed than the configured one."""
    global _SCOPE
    exit_at = min(c["threshold"] for c in configs)
    shorts = not all(c.get("long_only", True) for c in configs)
    scope = None if member is None else _scope(panel, member, exit_at, shorts)
    todo = {}
    for horizon, min_train in {(c["horizon"], c["min_train"]) for c in configs}:
        for i in panel.ids:
            bars = panel.one(i)
            d = _digest(features(bars), _label(bars, horizon))
            key = _key(d, horizon, min_train, scope)
            if key not in _PREDICTIONS and _key(d, horizon, min_train, None) not in _PREDICTIONS:
                todo[key] = (bars, horizon, min_train,
                             _stretches(None if member is None else member[i], bars.index), exit_at, shorts, SEED)
    order = sorted(todo, key=lambda k: -len(todo[k][0]))    # the long ones first: the last to finish are short
    if len(order) > 1 and multiprocessing.parent_process() is None:
        with ProcessPoolExecutor(min(len(order), os.cpu_count() or 1)) as pool:
            fitted = dict(zip(order, pool.map(_predict, [todo[k] for k in order])))
    else:                                   # a worker of an already parallel batch fits them one after another
        fitted = {k: _predict(todo[k]) for k in order}
    _PREDICTIONS.update(fitted)
    before, _SCOPE = _SCOPE, scope
    forget = [k for k in fitted if k[-1] is not None or k[-2] != config.SEED]

    def undo() -> None:
        global _SCOPE
        _SCOPE = before
        for k in forget:
            _PREDICTIONS.pop(k, None)
    return undo


@rule(grid={"horizon": [5], "min_train": [500], "threshold": [0.5, 0.55, 0.6, 0.65, 0.7], "slots": [None, 5],
            "long_only": [True, False]}, prepare=prepare)
def ml_direction(bars, horizon, min_train, threshold, long_only):
    """Long while the model's out-of-sample probability of a higher close in `horizon` bars exceeds `threshold`; unless
    long only, short while it is below 1 - `threshold`."""
    X = features(bars)
    y = _label(bars, horizon)
    proba = _cached_predictions(X, y, horizon, min_train)
    return with_short((proba > threshold).astype(float), (proba < 1.0 - threshold).astype(float), long_only)
