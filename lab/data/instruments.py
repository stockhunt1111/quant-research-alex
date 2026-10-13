"""Instrument identifiers: `<source>:<symbol>`, e.g. `td:AAPL`, `td:EUR/USD`, `td:XAU/USD`, `sh:WB1`, `perp:BTCUSDT`,
`spot:BTCUSDT`.

`cme` is a CME future, its contracts joined into one series (`cme:GC` gold, `cme:HG` copper): the CME futures list's,
beside the spot quotes of `td:XAU/USD` and the like.

`sh` is a US stock whose daily history comes from Sharadar: an S&P 500 member Twelve Data does not serve (it no
longer trades, or its ticker now belongs to another company), under Sharadar's ticker, which gives a company that
no longer trades a suffix when its ticker was taken up again (WB1 is Wachovia, WB is Weibo).

The source says where the bars come from and, with the symbol, fixes the asset class, which fixes the costs and
the trading calendar. Nothing guesses an asset class from a bare ticker elsewhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SOURCES = ("td", "sh", "perp", "spot", "cme")
_FX = re.compile(r"^[A-Z]{3}/[A-Z]{3}$")
# Spot precious metals and WTI crude: quoted like FX pairs and traded on the FX week, but a commodity's costs.
COMMODITIES = ("XAU/USD", "XAG/USD", "XPT/USD", "XPD/USD", "WTI/USD")
# The FX majors by their market's daily turnover, the most traded first: a pair's quotes carry no volume, so this is
# its liquidity (BIS Triennial Survey 2025, table 5, April 2025, $ billion a day: 2,033, 1,372, 731, 505, 467, 467,
# 118; the survey ranks AUD/USD above USD/CHF where both round to 467).
FX_BY_TURNOVER = ("EUR/USD", "USD/JPY", "GBP/USD", "USD/CAD", "AUD/USD", "USD/CHF", "NZD/USD")


@dataclass(frozen=True)
class Instrument:
    id: str
    source: str
    symbol: str
    asset_class: str      # key into config.COSTS
    calendar: str         # "xnys" | "fx" | "24x7"


def parse(instrument_id: str) -> Instrument:
    source, sep, symbol = instrument_id.partition(":")
    if not sep or source not in SOURCES or not symbol:
        raise ValueError(f"instrument id must look like 'td:AAPL' / 'perp:BTCUSDT', got {instrument_id!r}")
    if source == "td":
        if symbol in COMMODITIES:
            return Instrument(instrument_id, source, symbol, "commodity", "fx")
        if _FX.match(symbol):
            return Instrument(instrument_id, source, symbol, "fx", "fx")
        return Instrument(instrument_id, source, symbol, "us_equity", "xnys")
    if source == "sh":
        return Instrument(instrument_id, source, symbol, "us_equity", "xnys")
    if source == "cme":                   # a CME future, its contracts joined: traded on the FX week
        return Instrument(instrument_id, source, symbol, "commodity", "fx")
    if source == "perp":
        return Instrument(instrument_id, source, symbol, "crypto_perp", "24x7")
    return Instrument(instrument_id, source, symbol, "crypto_spot", "24x7")
