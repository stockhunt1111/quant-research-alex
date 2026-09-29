"""Market capitalisation snapshots that name the largest stocks and coins today (universes `us_stocks_mcapN`,
`crypto_mcapN`). One request each, no key:

    python -m strategy_lab.data.market_cap stocks     # Nasdaq's stock screener: every US listing with its market cap
    python -m strategy_lab.data.market_cap coins      # CoinGecko's ranking

Twelve Data sells market capitalisation only on its Ultra and Enterprise plans (`/market_cap` answers 403 on this key
except for its demo symbol AAPL) and charges 50 credits per symbol for `/statistics`, so the snapshots come from free
public endpoints. Each is written to data/reference/market_cap_{stocks,coins}.csv with the day it was taken.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import requests

from strategy_lab import log
from strategy_lab.config import REFERENCE_DIR

LOG = log.get("market_cap")
NASDAQ_SCREENER = "https://api.nasdaq.com/api/screener/stocks"
COINGECKO_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"


def path(kind: str):
    return REFERENCE_DIR / f"market_cap_{kind}.csv"


def fetch_stocks(session=None) -> pd.DataFrame:
    """Every US listing with its market capitalisation; tickers spelled as the store spells them (BRK/B -> BRK.B)."""
    session = session or requests.Session()
    r = session.get(NASDAQ_SCREENER, params={"tableonly": "true", "download": "true"},
                    headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    rows = pd.DataFrame(r.json()["data"]["rows"])
    out = pd.DataFrame({"symbol": rows["symbol"].str.strip().str.replace("/", ".", regex=False), "name": rows["name"],
                        "market_cap": pd.to_numeric(rows["marketCap"], errors="coerce")})
    missing = int(out["market_cap"].isna().sum())
    if missing:
        LOG.info("nasdaq screener: %d listings without a market cap (funds, rights, units) left out", missing)
    return out.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False)


def fetch_coins(session=None, n: int = 100) -> pd.DataFrame:
    session = session or requests.Session()
    r = session.get(COINGECKO_MARKETS, params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": n, "page": 1},
                    timeout=60)
    r.raise_for_status()
    rows = pd.DataFrame(r.json())
    return pd.DataFrame({"id": rows["id"], "symbol": rows["symbol"].str.upper(), "name": rows["name"],
                         "market_cap": pd.to_numeric(rows["market_cap"]), "price": pd.to_numeric(rows["current_price"]),
                         "rank": rows["market_cap_rank"]})


def save(kind: str, table: pd.DataFrame) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    table.assign(as_of=pd.Timestamp.now(tz="UTC").date()).to_csv(path(kind), index=False)
    LOG.info("market cap of %d %s saved to %s", len(table), kind, path(kind))


def load(kind: str) -> pd.DataFrame:
    p = path(kind)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m strategy_lab.data.market_cap {kind}`")
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["stocks", "coins"])
    args = ap.parse_args()
    log.setup(f"market_cap_{args.kind}")
    table = fetch_stocks() if args.kind == "stocks" else fetch_coins()
    save(args.kind, table)
    print(json.dumps(table.head(15).to_dict(orient="records"), indent=1, default=str))


if __name__ == "__main__":
    main()
