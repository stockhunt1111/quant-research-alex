"""Combine strategies into one portfolio of capital, scored against the same target.

Inputs are the out-of-sample daily returns each evaluation saved (each already a no-leverage book on its capital).
Capital is split across strategies with weights summing to 1 (so the portfolio has no leverage either) at each month
start, from data before it, and each strategy's money then grows with its own returns until the next split
(`held_returns`), as an account split once a month holds it: brought back to its weights every day, the portfolio
would sell its best strategy of the month down each day:
  * "equal"       — the same capital for every strategy trading that month;
  * "equal_risk"  — capital inversely proportional to each strategy's trailing volatility;
  * `select_trailing_sharpe` keeps, each month, only strategies whose trailing out-of-sample Sharpe is positive —
    choosing members on the past instead of picking this report's winners with hindsight; with `top_k`, only
    the k best of those by that trailing Sharpe.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from strategy_lab import db, metrics


def key(strategy: str, universe: str, timeframe: str) -> str:
    return f"{strategy}@{universe}@{timeframe}"


def load_oos(k: str, conn=None) -> pd.Series:
    """A member's out-of-sample record from the app's database."""
    strategy, universe, tf = k.split("@")
    own = conn is None
    conn = db.connect() if own else conn
    try:
        rid = db.result_id(conn, strategy, universe, tf)
        s = None if rid is None else db.series(conn, rid)
    finally:
        if own:
            conn.close()
    if s is None:
        raise LookupError(f"no saved evaluation for {k}")
    return s.rename(k)


def monthly_weights(rets: pd.DataFrame, method: str = "equal_risk", lookback_days: int = 90,
                    select_trailing_sharpe: int | None = None, top_k: int | None = None) -> pd.DataFrame:
    """Weights per day, decided at each month start from returns strictly before it."""
    active = rets.notna()
    month_start = rets.index.to_series().groupby(rets.index.tz_localize(None).to_period("M")).min()
    w = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
    for start in month_start:
        past = rets.loc[rets.index < start]
        live = active.loc[start]
        if past.empty or not live.any():
            continue
        cols = [c for c in rets.columns if live[c] and past[c].notna().sum() >= lookback_days]
        if select_trailing_sharpe:
            window = past.tail(select_trailing_sharpe)
            sr = window[cols].mean() / window[cols].std()
            cols = [c for c in cols if sr[c] > 0]
            if top_k:
                cols = sorted(cols, key=lambda c: sr[c], reverse=True)[:top_k]
        if not cols:
            w.loc[start] = 0.0
            continue
        if method == "equal":
            raw = pd.Series(1.0, index=cols)
        else:
            vol = past[cols].tail(lookback_days).std()
            raw = (1.0 / vol.replace(0, np.nan)).fillna(0.0)
        row = pd.Series(0.0, index=rets.columns)
        row[cols] = raw / raw.sum()
        w.loc[start] = row
    return w.ffill().fillna(0.0)


def held_returns(rets: pd.DataFrame, w: pd.DataFrame) -> pd.Series:
    """Daily returns of a portfolio that splits its capital by `w` at each month's start (`monthly_weights`: fixed
    within a month, the part not given out held in cash, earning nothing) and then holds: each strategy's money grows
    with its own returns through the month (a day without a return, none)."""
    r = rets.fillna(0.0)
    month = r.index.tz_localize(None).to_period("M")
    grown = (1.0 + r).groupby(month).cumprod()                   # each strategy's money since the month's split
    value = (w * grown).sum(axis=1) + (1.0 - w.sum(axis=1))      # the portfolio's, a month's start worth one
    before = value.groupby(month).shift(1).fillna(1.0)
    return value / before - 1.0


def combine(keys: list[str], method: str = "equal_risk", select_trailing_sharpe: int | None = None,
            top_k: int | None = None) -> dict:
    conn = db.connect()
    try:
        rets = pd.concat([load_oos(k, conn) for k in keys], axis=1).sort_index()
    finally:
        conn.close()
    w = monthly_weights(rets, method, select_trailing_sharpe=select_trailing_sharpe, top_k=top_k)
    port = held_returns(rets, w)
    started = w.sum(axis=1) > 0
    port = port[started.idxmax():] if started.any() else port.iloc[:0]
    card = metrics.scorecard(port, portfolio=True)
    held = w.loc[port.index]
    by_universe = held.T.groupby(lambda k: k.split("@")[1]).sum().T.mean().sort_values(ascending=False)
    return {"card": card, "weights": held, "daily": port, "correlation": rets.corr().round(2), "members": keys,
            "method": method, "select_trailing_sharpe": select_trailing_sharpe, "top_k": top_k,
            "avg_capital_by_universe": by_universe.round(3).to_dict(),
            "avg_members_held": float((held > 0).sum(axis=1).mean())}


