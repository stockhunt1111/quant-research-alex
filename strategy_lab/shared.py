"""Work the signal configurations of a rule's grid share on one instrument's bars.

A grid gives one instrument's bars to each of its signal configurations in turn (`strategy.fill_signals`), and many of
them compute the same indicators there: calm_trend's 108 configurations take one volatility regime, three
confirmations of it and six moving averages, each of them computed 108 times over. A function marked `shared` (the
indicators and regimes of strategy_lab) is computed once for all of them while `sharing` holds the bars: called again
with the same inputs, it hands back a copy of its first result, which the configuration may change at will.

Inputs are recognised by what they are, never guessed from their values: the bars shared or one of their columns (a rule
never writes into its bars: tests/test_lookahead.py), a result a shared function handed out (as long as it is still
what was handed out: compared bit for bit with the first result), and plain values (numbers, strings, None). A call
given anything else, or made outside `sharing`, is computed as it always was, so a configuration's positions come out
as when it computed everything itself (tests/test_strategy.py).
"""
from __future__ import annotations

import contextlib
import functools
import weakref
from typing import Callable

import numpy as np
import pandas as pd

_PLAIN = (int, float, str, type(None), np.integer, np.floating, np.bool_)      # a bool is an int


class _Bars:
    """The bars being shared and what the shared functions worked out on them."""

    def __init__(self, bars: pd.DataFrame):
        self.bars = bars
        # a column is recognised as the object itself (pandas hands out one Series for bars["close"] and bars.close),
        # held here so that no other object can take its id meanwhile
        self.columns = {id(col): (name, col) for name, col in bars.items()}
        self.first: dict[tuple, pd.Series | pd.DataFrame] = {}      # by call: its first result, never handed out
        self.handed: dict[int, tuple[weakref.ref, tuple]] = {}       # by id: a result handed out and its call

    def token(self, x) -> tuple | None:
        """What an argument is, as a key: None for one that cannot be recognised."""
        if isinstance(x, _PLAIN):
            return type(x), x
        if x is self.bars:
            return ("bars",)
        col = self.columns.get(id(x))
        if col is not None and col[1] is x:
            return "column", col[0]
        got = self.handed.get(id(x))
        if got is not None and got[0]() is x and _unchanged(x, self.first[got[1]]):
            return "result", got[1]
        return None

    def hand(self, out, call: tuple):
        self.handed[id(out)] = (weakref.ref(out), call)
        return out


_active: _Bars | None = None


@contextlib.contextmanager
def sharing(bars: pd.DataFrame):
    """While it lasts, each `shared` function works out a result on `bars` once for the calls that ask for it."""
    global _active
    before, _active = _active, _Bars(bars)
    try:
        yield
    finally:
        _active = before


def shared(fn: Callable) -> Callable:
    """`fn` computed once for the same inputs while `sharing` holds bars. `fn` must be a function of its arguments
    alone, and give a Series or a frame for bars (anything else is not kept)."""
    @functools.wraps(fn)
    def call(*args, **kwargs):
        on = _active
        if on is None:
            return fn(*args, **kwargs)
        given = [on.token(a) for a in args]
        named = [(k, on.token(kwargs[k])) for k in sorted(kwargs)]
        if None in given or any(t is None for _, t in named):
            return fn(*args, **kwargs)
        key = (fn, tuple(given), tuple(named))
        first = on.first.get(key)
        if first is not None:
            return on.hand(first.copy(), key)
        out = fn(*args, **kwargs)
        if isinstance(out, (pd.Series, pd.DataFrame)):
            on.first[key] = out.copy()
            on.hand(out, key)
        return out
    return call


def _unchanged(x, first) -> bool:
    """`x` is still the result it was handed out as: of the same kind, labels, name, attrs and values, bit for bit."""
    if type(x) is not type(first) or x.shape != first.shape or x.attrs != first.attrs:
        return False
    if isinstance(x, pd.Series):
        labels = x.name == first.name and x.dtype == first.dtype
    else:
        labels = x.columns.equals(first.columns) and x.dtypes.equals(first.dtypes)
    if not labels or x.index.names != first.index.names or not x.index.equals(first.index):
        return False
    a, b = x.to_numpy(), first.to_numpy()
    return a.dtype.kind in "biuf" and a.dtype == b.dtype and a.tobytes() == b.tobytes()
