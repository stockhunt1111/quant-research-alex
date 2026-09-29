"""The instrument lists the research runs on, by market.

One rule wherever a market is big enough: the N most liquid names, re-picked every month from the trading before it
with a buffer (`universes.top_liquid`: a member keeps its seat until it ranks below 1.5 x N), ranked on the daily bars
whatever the timeframe (`evaluate._seats`), in three steps. Ten is about what a $10,000 account can hold in whole shares at equal
weight ($1,000 a name); fifty and a hundred give a basket its breadth and show whether a result rests on a handful of
names. Liquidity rather than market capitalisation, because a list that changes through history needs its ranking at
every past date and only trading volume is known for every date here: capitalisation is a snapshot of today, and
today's largest names carried back in time are the winners picked with hindsight. A small market is taken whole.
Each market with a ranking has a fourth, narrower step, Top-3: crypto's first (user, 2026-09-25: BTC and ETH, which
have not left it since 2021, and the third most traded coin), then stocks' and ETFs' (user, 2026-09-26), the ETFs'
ranked from the 34 of `etf_core` by the same rule.

A list is scored from the first day it holds at least half of its names (`evaluate`): a basket of one or two is not
the list, whatever the history of its oldest member.

The firm's research desk (Stockhunt) scores its own lists; its ETFs and its five commodities (spot quotes, the prices
the firm's simulator fills at) are ours, and so is its kind of market of CME futures: the commodities' futures,
their contracts joined, copper among them (user, 2026-09-26), which Twelve Data quotes no spot price of. Its 216 US
stocks and
its 20 coins on spot are not research lists (user, 2026-09-25): `universes` still resolves them for a run asked for
with `--outside-lists`.

Each strategy on each instrument alone (`strategy_lab.per_asset`) runs on the widest list of a market only: the
narrower steps are today's leaders of the same ranking, so their instruments are inside it. The ML task has its own
markets: today's 10 largest US stocks and coins by
market capitalisation and the FX majors, and (user, 2026-09-28) the five commodities the firm's simulator fills at and
the firm's ten ETFs.

Nothing runs on any other list unless the user asks: the command line and the batch scripts refuse a list outside
these (`refuse_outside`), and take one only with `--outside-lists`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentList:
    universe: str           # a `strategy_lab.universes.resolve` name
    market: str
    label: str              # short, within its market
    start: str | None = None    # the first date of the bars a run loads; None: the whole history


OURS = (
    InstrumentList("us_stocks_top3", "Stocks", "Top-3"),
    InstrumentList("us_stocks_top10", "Stocks", "Top-10"),
    InstrumentList("us_stocks_top50", "Stocks", "Top-50"),
    InstrumentList("us_stocks_top100", "Stocks", "Top-100"),
    InstrumentList("crypto_top3", "Crypto", "Top-3"),
    InstrumentList("crypto_top10", "Crypto", "Top-10"),
    InstrumentList("crypto_top50", "Crypto", "Top-50"),
    InstrumentList("crypto_top100", "Crypto", "Top-100"),
    InstrumentList("etf_top3", "ETFs", "Top-3"),
    InstrumentList("stockhunt_etfs", "ETFs", "10 majors"),
    InstrumentList("etf_core", "ETFs", "All 34"),
    InstrumentList("fx_majors", "FX", "7 majors"),
    InstrumentList("stockhunt_commodities", "Commodities", "All 5"),
    InstrumentList("cme_futures", "CME futures", "All 6"),
)
PER_ASSET = ("us_stocks_top100", "crypto_top100", "etf_core", "fx_majors", "stockhunt_commodities", "cme_futures")
# the ML task's markets (user, 2026-09-24; the commodities and the ETFs 2026-09-28)
ML_TASK = ("us_stocks_mcap10", "crypto_mcap10", "fx_majors", "stockhunt_commodities", "stockhunt_etfs")
PER_INSTRUMENT = PER_ASSET + tuple(u for u in ML_TASK if u not in PER_ASSET)
# the ML task's lists that are not among ours: run on each instrument alone only, never ranked as a book (`market`
# answers None for them); their market and label say where their single-asset results stand
ML_TASK_LISTS = (
    InstrumentList("us_stocks_mcap10", "Stocks", "Top-10 by market cap"),
    InstrumentList("crypto_mcap10", "Crypto", "Top-10 by market cap"),
)

_BY_NAME = {x.universe: x for x in OURS}
_ANY = {x.universe: x for x in OURS + ML_TASK_LISTS}


def names(lists: tuple[InstrumentList, ...]) -> list[str]:
    return [x.universe for x in lists]


def market(universe: str) -> str | None:
    """The market of a research list; None for a universe outside the lists (an ad-hoc run)."""
    x = _BY_NAME.get(universe)
    return x.market if x else None


def title(universe: str) -> str:
    """The list's name outside its market's context: "Stocks Top-10" (a list of ours or of the ML task)."""
    x = _ANY.get(universe)
    return f"{x.market} {x.label}" if x else universe


def per_instrument_market(universe: str) -> str | None:
    """The market of a list its instruments are run on alone (ours or the ML task's); None for any other."""
    x = _ANY.get(universe)
    return x.market if x else None


def start(universe: str) -> str | None:
    x = _BY_NAME.get(universe)
    return x.start if x else None


def basket_lists() -> tuple[str, ...]:
    """The lists a strategy is evaluated on as a basket: ours."""
    return tuple(names(OURS))


def refuse_outside(universes, allowed, what: str) -> None:
    """Stop a run that names a list outside `allowed`: such lists run only when the user asks (`--outside-lists`)."""
    outside = [u for u in dict.fromkeys(universes) if u not in allowed]
    if outside:
        raise SystemExit(f"{', '.join(outside)}: not among the agreed lists for {what} ({', '.join(allowed)}); "
                         "these run only when the user asks, with --outside-lists")
