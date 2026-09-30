"""A model of a rule's own trades: from the market at the bar a trade is decided, the probability that the trade makes
money after its costs. It grades a rule's trades (`@rule(grade=...)`, on the rule's positions on every instrument of
a run, before the universe's seats are given out):

* `take` — only the trades whose probability is above `threshold` are kept: the rule decides when, the model whether
  (meta-labelling: M. Lopez de Prado, "Advances in Financial Machine Learning", 2018, section 3.6);
* `size` — every trade is scaled by the model's confidence, 2p - 1, and nothing below one half.

A trade is the rule's own: it starts on the bar its position turns on or to the other side and ends on the bar
before it turns off or flips; it is entered at the open after its first bar and left at the open after the bar that
ends it, as the engine fills a rule, and its outcome is that move less what its two fills cost (`engine.costs`: its
class's commission and half-spread, or a spot quote's broker spread at each open; the carry of a perp or a short is
left out). Its features are the indicator families of
`strategy_lab.ml` at its first bar. One model learns from every instrument of the run — a list's names pooled, an
instrument alone from its own trades — and from a name only the trades it started while in the universe. The model is
refit on the walk-forward calendar (`ml.refit_rows`) on the trades that had been left by then, so a trade is graded
by a model that saw neither it nor anything after it; before a refit has MIN_TRADES trades to learn from, no trade is
taken. Only a name's trades while in the universe: a list's panel holds the names it ever holds, and their trades
from before they joined are there because they joined later (learning from them too lifted IBS's Top-50 1d record from
a Sharpe of 0.87 to 1.03, a gain bought by knowing who joins).

The model (`model`) is a random forest, LightGBM's own random-forest mode: its trees are grown on bagged samples and
averaged, not each fit to the last one's errors. Whether a trade makes money is a noisy label, which boosting fits
more than averaging does. Measured walk-forward on 2026-09-30 against the boosted trees it had been (the settings
`ml.model` keeps), on the four graded rules' filters, stocks Top-10 and the 34 ETFs 1d, IBS also on stocks Top-50 and
CME futures 4h: a higher Sharpe on 8 of 10 (median +0.06; IBS +0.06 to +0.26 on all four) and a higher return on 7; a
forest of scikit-learn's did the same on IBS at 15 times the time.
"""
from __future__ import annotations

import collections
import hashlib

import numpy as np
import pandas as pd

from strategy_lab import config, log, ml
from strategy_lab.data.bars import FIELDS, Panel
from strategy_lab.engine import costs

LOG = log.get("trade_model")
THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7]     # the grid of `take`: the walk-forward picks one on the past
MIN_TRADES = 100                        # the fewest closed trades a refit learns from (as strategies/ml_feature_search)
SEED = config.SEED                      # the model's random draws: the robustness check of other seeds varies it

_GRADED: dict[str, pd.DataFrame] = {}   # a run's trades with their probability, by (data, positions, seed)


def model(seed: int):
    """A random forest of 200 trees of depth 5 on bagged samples of 63% of the trades and 80% of the features (LightGBM's
    random-forest mode), each leaf of 40 trades at least."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(boosting_type="rf", n_estimators=200, max_depth=5, num_leaves=31, min_child_samples=40,
                              subsample=0.632, subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=1,
                              verbose=-1)


def take(panel: Panel, wanted: pd.DataFrame, live: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """The rule's positions, keeping only the trades the model gives a probability above `threshold`."""
    t = graded(panel, wanted, live)
    return _scaled(panel, wanted, t, (t["p"] > threshold).to_numpy(dtype=np.float64))


def size(panel: Panel, wanted: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """The rule's positions, each trade scaled by 2p - 1 (zero below one half and while no model exists)."""
    t = graded(panel, wanted, live)
    return _scaled(panel, wanted, t, (2 * t["p"] - 1).clip(0.0, 1.0).fillna(0.0).to_numpy())


def _scaled(panel: Panel, wanted: pd.DataFrame, trades: pd.DataFrame, factor: np.ndarray) -> pd.DataFrame:
    out = wanted.to_numpy(copy=True)
    col = {i: k for k, i in enumerate(panel.ids)}
    for inst, first, last, f in zip(trades["instrument"], trades["first"], trades["last"], factor):
        if f != 1.0:
            out[first:last + 1, col[inst]] *= f
    return pd.DataFrame(out, index=wanted.index, columns=wanted.columns)


def _bars_digest(panel: Panel) -> bytes:
    """A hash of the panel's bars, worked out once per panel: a grid grades its rule's trades once per configuration."""
    if "bars_digest" not in panel.memo:
        h = hashlib.sha256(panel.index.asi8.tobytes())
        for f in FIELDS:
            h.update(np.ascontiguousarray(getattr(panel, f).to_numpy(dtype=np.float64)).tobytes())
        panel.memo["bars_digest"] = h.digest()
    return panel.memo["bars_digest"]


def graded(panel: Panel, wanted: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Every trade of the rule on the run with its probability `p` of making money (NaN: no model graded it)."""
    h = hashlib.sha256(repr((panel.ids, SEED)).encode())
    h.update(_bars_digest(panel))
    for frame in (wanted, live):
        h.update(np.ascontiguousarray(frame.to_numpy(dtype=np.float64)).tobytes())
    key = h.hexdigest()
    if key not in _GRADED:
        if len(_GRADED) >= 256:
            _GRADED.clear()
        _GRADED[key] = _grade(panel, wanted, live)
    return _GRADED[key]


def trades(panel: Panel, wanted: pd.DataFrame, live: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """The rule's trades on every instrument (panel rows `first`..`last` of its position, the times it was decided and
    its outcome became known, its net return, whether it started in the universe) and their features."""
    parts, feats = [], []
    idx = panel.index
    rates = costs.rates(panel)
    for j, inst in enumerate(panel.ids):
        own = np.flatnonzero(panel.close[inst].notna().to_numpy())      # the bars the instrument printed
        if len(own) < 3:
            continue
        side = np.sign(wanted[inst].to_numpy()[own])
        start = np.flatnonzero((side != 0) & (side != np.r_[0.0, side[:-1]]))
        end = np.flatnonzero((side != 0) & (side != np.r_[side[1:], 0.0]))
        if not len(start):
            continue
        opens = panel.open[inst].to_numpy()[own]
        # entered at the open after its first bar, out at the open after the bar ending it; a trade decided on the last
        # bar is graded all the same (its decision is known at that close), it only has no outcome yet
        left = end + 2 < len(own)
        exit_at = np.where(left, end + 2, len(own) - 1)
        entry = np.where(start + 1 < len(own), opens[np.minimum(start + 1, len(own) - 1)], np.nan)
        paid = rates.of(j, "at_open")[own]                          # a side, at the open each fill is made at
        net = side[start] * (opens[exit_at] / entry - 1.0) - paid[np.minimum(start + 1, len(own) - 1)] - paid[exit_at]
        known = np.where(left, idx.asi8[own[exit_at]], np.iinfo(np.int64).max)
        # its position also rides through the bars the instrument missed after its last bar (the rule keeps it there)
        last = np.where(end + 1 < len(own), own[np.minimum(end + 1, len(own) - 1)] - 1, len(idx) - 1)
        parts.append(pd.DataFrame({
            "instrument": inst, "first": own[start], "last": last, "decided": idx.asi8[own[start]],
            "known": known, "net": np.where(left, net, np.nan),
            "in_universe": live[inst].to_numpy()[own[start]]}))
        feats.append(_features_at(panel, inst, start))
    if not parts:
        return pd.DataFrame(columns=["instrument", "first", "last", "decided", "known", "net", "in_universe"]), \
            np.empty((0, 0))
    return pd.concat(parts, ignore_index=True), np.vstack(feats)


_FEATURES_AT: collections.OrderedDict[bytes, np.ndarray] = collections.OrderedDict()
FEATURES_KEPT_BYTES = 256 * 2**20       # the most `_features_at` keeps, the longest unused dropped first
_features_bytes = 0


def _features_at(panel: Panel, inst: str, start: np.ndarray) -> np.ndarray:
    """The model's features of an instrument's trades on their first bars (`start`, rows of the bars it printed). They
    depend on those bars and rows alone, so they are kept for the next grading of the same trades: with another seed of
    the model, on another list that holds the name. The features of every bar are not kept: those of a list's names on
    1h take gigabytes."""
    global _features_bytes
    key = hashlib.blake2b(panel.digest(inst).encode() + start.tobytes(), digest_size=16).digest()
    got = _FEATURES_AT.get(key)
    if got is not None:
        _FEATURES_AT.move_to_end(key)
        return got
    got = pd.concat(ml.features(panel.one(inst)).values(), axis=1).to_numpy(dtype=np.float64)[start]
    got.flags.writeable = False
    _FEATURES_AT[key] = got
    _features_bytes += got.nbytes
    while _features_bytes > FEATURES_KEPT_BYTES:
        _features_bytes -= _FEATURES_AT.popitem(last=False)[1].nbytes
    return got


def probabilities(x: np.ndarray, outcome: np.ndarray, decided: np.ndarray, known: np.ndarray, learn: np.ndarray,
                  refits: np.ndarray, seed: int, shuffle: bool = False) -> np.ndarray:
    """P(outcome) of each trade from the model refit last before it was decided; a refit learns from the trades whose
    outcome was known by then (`known` <= refit time) and `learn` allows; NaN where no model was fit yet or a feature
    is missing. `shuffle` permutes the outcomes a refit learns from (a control that must find nothing)."""
    p = np.full(len(decided), np.nan)
    usable = ~np.isnan(x).any(axis=1) if x.size else np.zeros(len(decided), dtype=bool)
    rng = np.random.default_rng(seed)
    starved = []
    for k, r in enumerate(refits):
        nxt = refits[k + 1] if k + 1 < len(refits) else np.iinfo(np.int64).max
        block = (decided >= r) & (decided < nxt) & usable
        if not block.any():
            continue
        train = (known <= r) & learn & usable & ~np.isnan(outcome)
        y = outcome[train].astype(int)
        if len(y) < MIN_TRADES or len(np.unique(y)) < 2:
            starved.append((r, int(block.sum())))
            continue
        if shuffle:
            y = rng.permutation(y)
        p[block] = model(seed).fit(x[train], y).predict_proba(x[block])[:, 1]
    if starved:
        LOG.info("%d of %d refits had fewer than %d closed trades of both outcomes to learn from (the last on %s): "
                 "their %d trades are not graded", len(starved), len(refits), MIN_TRADES,
                 pd.Timestamp(starved[-1][0], tz="UTC").date(), sum(n for _, n in starved))
    return p


def _grade(panel: Panel, wanted: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    t, x = trades(panel, wanted, live)
    if not len(t):
        return t.assign(p=np.array([], dtype=np.float64))
    refits = panel.index.asi8[ml.refit_rows(panel.index)]
    outcome = np.where(np.isnan(t["net"]), np.nan, (t["net"] > 0).astype(float))
    p = probabilities(x, outcome, t["decided"].to_numpy(), t["known"].to_numpy(),
                      t["in_universe"].to_numpy(dtype=bool), refits, SEED)
    return t.assign(p=p)
