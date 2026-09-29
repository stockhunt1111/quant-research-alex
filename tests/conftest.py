"""Test helpers: synthetic panels and a network ban (tests never touch vendors)."""
from __future__ import annotations

import socket

import numpy as np
import pandas as pd
import pytest

from strategy_lab.data.bars import FIELDS, Panel
from strategy_lab.data.instruments import parse


@pytest.fixture(autouse=True)
def _no_list_kept_from_another_test():
    from strategy_lab import evaluate
    evaluate._LOADED.clear()
    evaluate._SEATING.clear()
    evaluate._DAILY_SEATING.clear()


@pytest.fixture(autouse=True)
def _positions_kept_apart(tmp_path, monkeypatch):
    """A test's rules keep their positions in the test's own directory, never in data/cache."""
    from strategy_lab import strategy
    monkeypatch.setattr(strategy, "KEPT_DIR", tmp_path / "positions")


@pytest.fixture(autouse=True)
def _minutes_kept_apart(tmp_path, monkeypatch):
    """A test's one-minute bars live in the test's own directory, never in data/store."""
    from strategy_lab.data import minutes
    monkeypatch.setattr(minutes, "ROOT", tmp_path / "store")


@pytest.fixture(autouse=True)
def _spreads_kept_apart(tmp_path, monkeypatch):
    """A test's quoted spreads live in the test's own directory, never in data/store."""
    from strategy_lab.data import spreads
    monkeypatch.setattr(spreads, "STORE_DIR", tmp_path / "store")


def quote_spreads(instrument_ids, hours: pd.DatetimeIndex, opening=0.2, closing=None):
    """Spot quotes' hours of spreads where the engine reads them (`data.spreads`): at each hour's first quote `opening`
    and at its last `closing` (the same when not given), a number or one per hour, stamped at the hours' closes."""
    from strategy_lab.data import spreads
    frame = pd.DataFrame({"open": opening, "close": opening if closing is None else closing},
                         index=pd.DatetimeIndex(hours, name="close_time"))
    for i in instrument_ids:
        path = spreads.spread_path(i)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)


CASH_RATE = 0.04              # the T-bill rate in tests, a year: steady, so a test never reads data/reference


@pytest.fixture(autouse=True)
def _steady_cash_rate(monkeypatch):
    from strategy_lab.data import rates
    days = pd.date_range("1990-01-01", "2040-12-31", freq="D", tz="UTC")
    monkeypatch.setattr(rates, "daily_rate", lambda: pd.Series(CASH_RATE / 365.0, index=days))


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*a, **k):
        raise RuntimeError("tests must not open network connections")
    monkeypatch.setattr(socket.socket, "connect", refuse)


def store_of(panel: Panel):
    """`load_panel` over a synthetic panel: any of its instruments, fields and bars, as the store serves them."""
    def load(ids, timeframe, start=None, end=None, fields=FIELDS, index=None):
        wide = {f: getattr(panel, f)[list(ids)] if f in fields else None for f in FIELDS}
        if index is not None:
            wide = {f: None if v is None else v.reindex(index) for f, v in wide.items()}
        return Panel(panel.timeframe, {i: panel.instruments[i] for i in ids}, **wide)
    return load


def top_lists(panel: Panel):
    """`resolve` for lists named <anything>_top<N> on a synthetic panel: its N most liquid instruments, re-picked
    monthly, as a stock or crypto Top-N list is."""
    from strategy_lab.universes import Universe, top_liquid

    def resolve(name, timeframe):
        n = int(name.rsplit("_top", 1)[1])
        return Universe(name, list(panel.ids), lambda p: top_liquid(p, n, 30), size=n)
    return resolve


def rising_list_market(monkeypatch, n=700, seed=11, names=18) -> Panel:
    """A rising market of `names` instruments behind every Top-N list (strategies checked for robustness make money on
    it: a losing one is not checked), their liquidity ranks shifting slowly: Top-5, Top-10 and Top-15 differ."""
    from strategy_lab import evaluate
    p = make_panel(ids=tuple(f"td:S{k:02d}" for k in range(names)), n=n, seed=seed)
    rise = np.exp(0.0008 * np.arange(n))[:, None]
    for f in ("open", "high", "low", "close", "dollar_volume"):
        setattr(p, f, getattr(p, f) * rise)
    for k in range(names):
        p.dollar_volume.iloc[:, k] *= 1 + (names - k) / 4
    monkeypatch.setattr(evaluate, "resolve", top_lists(p))
    monkeypatch.setattr(evaluate, "load_panel", store_of(p))
    monkeypatch.setattr(evaluate, "stored_version", lambda ids, tf: 1)
    return p


def make_panel(ids=("td:AAA", "td:BBB"), n=400, seed=0, freq="D", start="2021-01-04 21:00") -> Panel:
    """Random-walk OHLC bars with real gaps between close and next open."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    frames = {f: {} for f in FIELDS}
    for j, i in enumerate(ids):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        opn = np.r_[close[0], close[:-1] * np.exp(rng.normal(0, 0.004, n - 1))]
        hi = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
        lo = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
        for f, v in (("open", opn), ("high", hi), ("low", lo), ("close", close),
                     ("volume", np.full(n, 1e6)), ("dollar_volume", close * 1e6)):
            frames[f][i] = v
    wide = {f: pd.DataFrame(frames[f], index=idx) for f in FIELDS}
    return Panel("1d", {i: parse(i) for i in ids}, **wide)


def make_minutes(seed: int, days: int = 4):
    """A random walk of one-minute bars around the clock (stamped at their close, prices as the store keeps them, in
    single precision) and the hourly bars they make."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    opn = np.r_[100.0, close[:-1]]
    high = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.0005, n)))
    low = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.0005, n)))
    idx = pd.date_range("2024-01-01 00:01", periods=n, freq="min", tz="UTC")
    m = pd.DataFrame({"open": opn, "high": high, "low": low, "close": close}, index=idx).astype(np.float32).astype(float)
    hours = m.resample("1h", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return m, hours
