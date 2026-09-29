"""A per-symbol ML engine, measured out-of-sample: a model
trained on indicator features, with the indicator combination picked for each symbol.

One model, LightGBM, with different combinations of features. Its inputs are one or more of five indicator
families; the grid holds every combination of families, three entry thresholds and both directions (long only, and
long/short, which trades the model's "not up" as a short), and the walk-forward picks the configuration with
the best Sharpe on the instrument's own past, re-picked every 3 months (per_asset). The model predicts whether the
close `HORIZON` bars later is higher. It is refit on every walk-forward window start, and twice in the first year so
that the first choice has half a year of predictions to judge, each time on the rows whose label was already known; a
block of bars is always predicted by a model fit before it. Long while the predicted probability of a rise exceeds
`threshold`; unless long only, short while it is below 1 - `threshold`.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd

from strategy_lab import config, ml
from strategy_lab.strategy import rule, with_short

HORIZON = 5
SEED = config.SEED          # the model's random draws: the robustness check of other seeds varies it
FAMILY_SETS = [c for k in range(1, len(ml.FAMILIES) + 1) for c in itertools.combinations(ml.FAMILIES, k)]


def predict_blocks(X: pd.DataFrame, y: pd.Series, rows: list[int], horizon: int) -> pd.Series:
    """P(rise) for each bar from the refit before it; a refit uses the rows whose label was known (j + horizon <= t)."""
    proba = pd.Series(np.nan, index=X.index)
    valid = X.notna().all(axis=1).to_numpy()
    known = y.notna().to_numpy()
    n = len(X)
    for a, b in zip(rows, rows[1:] + [n]):
        train = (np.arange(n) <= a - horizon) & valid & known
        if train.sum() < 100 or y[train].nunique() < 2:
            continue
        m = ml.model(SEED).fit(X[train], y[train].astype(int))
        block = np.arange(a, b)
        ok = block[valid[block]]
        if len(ok):
            proba.iloc[ok] = m.predict_proba(X.iloc[ok])[:, 1]
    return proba


# A fit depends on the bars, the families, the refit rows and the model's seed only: the thresholds, and the
# per-instrument runs of the same bars, reuse it.
_PREDICTIONS: dict[str, pd.Series] = {}


def _cached(X: pd.DataFrame, y: pd.Series, rows: list[int], horizon: int) -> pd.Series:
    h = hashlib.sha256(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes())
    h.update(pd.util.hash_pandas_object(y, index=True).to_numpy().tobytes())
    h.update(repr((rows, horizon, SEED)).encode())
    key = h.hexdigest()
    if key not in _PREDICTIONS:
        if len(_PREDICTIONS) >= 4096:
            _PREDICTIONS.clear()
        _PREDICTIONS[key] = predict_blocks(X, y, rows, horizon)
    return _PREDICTIONS[key]


@rule(grid={"families": FAMILY_SETS, "threshold": [0.5, 0.55, 0.6], "long_only": [True, False]}, model=True)
def ml_feature_search(bars, families, threshold, long_only):
    """Long while LightGBM on the chosen indicator families gives a rise in HORIZON bars a probability above
    `threshold`; unless long only, short while it gives it less than 1 - `threshold`. A bar without a prediction (a
    feature undefined on it) keeps the position before it."""
    fam = ml.features(bars)
    X = pd.concat([fam[f] for f in families], axis=1)
    later = bars["close"].shift(-HORIZON)
    y = (later > bars["close"]).astype(float).where(later.notna())
    proba = _cached(X, y, ml.refit_rows(bars.index), HORIZON)
    side = with_short((proba > threshold).astype(float), (proba < 1.0 - threshold).astype(float), long_only)
    return side.where(proba.notna())
