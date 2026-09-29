"""Orderly Network, the venue behind CoinIQ: its market list and its own daily candles (public API, no key).

CoinIQ trades Orderly's perpetuals, so a strategy meant for CoinIQ is tested on the coins Orderly lists. Orderly's
own history starts when a market was listed (BTC and ETH on 2023-10-26, most others in 2024-2025). The Binance
perpetual of the same coin carries years more history; `compare` checks, coin by coin, that both venues print the
same daily closes over their common period before the Binance history stands in for Orderly's.

    python -m strategy_lab.data.orderly markets     -> data/reference/orderly_markets.csv
    python -m strategy_lab.data.orderly compare     -> data/reference/orderly_vs_binance.csv (daily closes, common period)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import requests

from strategy_lab import log
from strategy_lab.config import REFERENCE_DIR
from strategy_lab.data import store
from strategy_lab.data.refresh import Budget, RateLimited, _ledger

LOG = log.get("orderly")
API = "https://api-evm.orderly.org"
MARKETS_FILE = "orderly_markets.csv"
COMPARE_FILE = "orderly_vs_binance.csv"
# the Binance perp stands in for an Orderly market only when both print the same coin
MAX_MEDIAN_GAP_BPS = 25.0
MIN_RETURN_CORR = 0.995


def _get(session, path: str, params: dict, budget: Budget, what: dict) -> dict:
    budget.acquire()
    r = session.get(API + path, params=params, timeout=30)
    _ledger("orderly", status=r.status_code, path=path, **what)
    if r.status_code == 429:
        raise RateLimited(f"orderly 429 on {path}")
    r.raise_for_status()
    return r.json()


def markets(session=None, budget: Budget | None = None) -> pd.DataFrame:
    """Every Orderly perpetual: symbol, base, listing date, and whether it is a plain coin market.

    Markets run by a third-party broker on Orderly carry a suffix (PERP_AAPL_USDC_mythos: tokenised shares); the plain
    PERP_<BASE>_USDC markets are the venue's own and include a few non-crypto ones (NAS100, XAU, EURUSD)."""
    session = session or requests.Session()
    budget = budget or Budget("orderly", per_minute=60, max_requests=None)
    rows = _get(session, "/v1/public/info", {}, budget, {"what": "markets"})["data"]["rows"]
    out = pd.DataFrame({
        "symbol": [r["symbol"] for r in rows],
        "base": [r["symbol"].split("_")[1] for r in rows],
        "plain": [r["symbol"].count("_") == 2 for r in rows],
        "listed": [pd.Timestamp(r["created_time"], unit="ms", tz="UTC").date() for r in rows],
    }).sort_values("symbol")
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(REFERENCE_DIR / MARKETS_FILE, index=False)
    LOG.info("orderly: %d markets (%d plain) -> %s", len(out), int(out["plain"].sum()), REFERENCE_DIR / MARKETS_FILE)
    return out


def daily_closes(symbol: str, session, budget: Budget) -> pd.Series:
    """Orderly's daily closes, indexed by the UTC close of each day (the store's convention)."""
    now = int(pd.Timestamp.now(tz="UTC").timestamp())
    d = _get(session, "/tv/history", {"symbol": symbol, "resolution": "1D", "from": now - 86400 * 4000, "to": now},
             budget, {"symbol": symbol, "what": "1D"})
    if d.get("s") != "ok" or not d.get("t"):
        return pd.Series(dtype=float)
    close_time = pd.to_datetime(d["t"], unit="s", utc=True) + pd.Timedelta(days=1)
    s = pd.Series(d["c"], index=close_time, dtype=float)
    return s[s.index <= pd.Timestamp.now(tz="UTC")]           # the day in progress has not closed


def compare() -> pd.DataFrame:
    """Per plain coin market with a Binance perp in the store: how closely the two venues' daily closes agree."""
    ref = pd.read_csv(REFERENCE_DIR / MARKETS_FILE)
    coins = ref[ref["plain"]]
    have = set(store.symbols("perp", "1d"))
    session, budget = requests.Session(), Budget("orderly", per_minute=60, max_requests=len(coins) + 5)
    rows = []
    for r in coins.itertuples(index=False):
        binance = f"{r.base}USDT"
        if binance not in have:
            rows.append({"base": r.base, "orderly": r.symbol, "binance": None, "days": 0})
            continue
        o = daily_closes(r.symbol, session, budget)
        b = store.read_bars("perp", "1d", binance)["close"]
        both = pd.concat({"o": o, "b": b}, axis=1).dropna()
        if len(both) < 30:
            rows.append({"base": r.base, "orderly": r.symbol, "binance": binance, "days": len(both)})
            continue
        gap = np.log(both["o"] / both["b"])
        ro, rb = both["o"].pct_change().dropna(), both["b"].pct_change().dropna()
        rows.append({"base": r.base, "orderly": r.symbol, "binance": binance, "days": len(both),
                     "median_gap_bps": float(gap.abs().median() * 1e4), "p99_gap_bps": float(gap.abs().quantile(0.99) * 1e4),
                     "return_corr": float(ro.corr(rb)), "first_common_day": both.index.min().date()})
    out = pd.DataFrame(rows)
    out["same_coin"] = (out["median_gap_bps"] < MAX_MEDIAN_GAP_BPS) & (out["return_corr"] > MIN_RETURN_CORR)
    out.to_csv(REFERENCE_DIR / COMPARE_FILE, index=False)
    LOG.info("orderly vs binance: %d coins compared, %d without a Binance perp in the store", int(out["days"].ge(30).sum()),
             int(out["binance"].isna().sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["markets", "compare"])
    args = ap.parse_args()
    log.setup(f"orderly_{args.what}")
    if args.what == "markets":
        markets()
    else:
        print(compare().to_string(index=False))


if __name__ == "__main__":
    main()
