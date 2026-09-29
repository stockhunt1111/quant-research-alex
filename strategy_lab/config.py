"""Single source of paths, the firm's target and cost assumptions.

Importing this module has no side effects: directories are created by the code that writes into them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
STORE_DIR = DATA_DIR / "store"          # normalised bars: one parquet per (source, timeframe, symbol)
CACHE_DIR = DATA_DIR / "cache"          # what is worked out again only at a cost: rules' positions (`strategy`)
REFERENCE_DIR = DATA_DIR / "reference"  # index membership and exchange listings (copied at seeding)
REPORTS_DIR = ROOT_DIR / "reports"
LOGS_DIR = ROOT_DIR / "logs"

# The previous research project whose on-disk bars seed the store (read-only, never written).
LEGACY_PROJECT_DIR = Path(os.environ.get("LEGACY_PROJECT_DIR",
                                         "/Users/alexsilka/Dev/job/test_senior_quant_developer"))

SEED = 7
RESEARCH_CAPITAL_USD = 10_000.0         # the firm's example retail deposit; drives minimum-ticket feasibility

TIMEFRAMES = ("1h", "4h", "1d")         # the timeframes the research runs on

# The firm's target. Monthly figures are returns on the traded capital, no leverage.
FIRM_TARGETS = {
    "avg_monthly": 0.015,          # 1.5-2% a month, compounded (metrics.average_month): 19.6-26.8% a year
    "avg_monthly_stretch": 0.02,
    "pct_green": 0.70,             # 70% acceptable, 80% the plan's figure
    "pct_green_stretch": 0.80,
    "max_dd": -0.10,               # ~10%; 15% only when leverage is added, and this engine does not add it
    "sharpe": 1.0,
    "beat_bh": True,               # first bar: beat buy-and-hold of the same universe
}
MAX_GROSS = 1.0                    # "without leverage": sum of |weights| never exceeds the capital


@dataclass(frozen=True)
class CostSpec:
    """Per-side trading cost and the annual cost of holding a short, in basis points."""
    commission_bps: float
    half_spread_bps: float
    borrow_bps_annual: float = 0.0     # charged on the short leg only; perps pay funding instead


# Asset class -> cost of trading it. Crypto: Binance VIP-0 taker (spot 10, USD-M perp 5) plus a 1bp half-spread;
# US equities/ETFs: 1bp commission + 2bp half-spread; FX: spread only. Commodities: the CME futures 2bp a side, spread
# only, the cost the firm's research desk (Stockhunt) books on the five spot quotes (its cost drag over its turnover;
# the same arithmetic on its crypto sheet gives exactly the 11bp spot figure above; its CME futures it books at
# 0.75bp); the five spot quotes, the prices the firm's simulator fills at, pay the class's commission (none) and half
# the spread their broker quoted at the fill, bar by bar (`engine.costs`, `data.spreads`: Exness's Pro account; user,
# 2026-09-28): 2bp a side was 11 times gold's half-spread there in 2025-04, an eighth of palladium's in 2026-08 and a
# fiftieth in 2020-06. Spot-margin coin borrow and equity general-collateral borrow as in the previous project.
COSTS: dict[str, CostSpec] = {
    "crypto_perp": CostSpec(commission_bps=5.0, half_spread_bps=1.0),
    "crypto_spot": CostSpec(commission_bps=10.0, half_spread_bps=1.0, borrow_bps_annual=293.0),
    "us_equity": CostSpec(commission_bps=1.0, half_spread_bps=2.0, borrow_bps_annual=50.0),
    "fx": CostSpec(commission_bps=0.0, half_spread_bps=1.0),
    "commodity": CostSpec(commission_bps=0.0, half_spread_bps=2.0),
}

# A span of time a strategy's source gives in days or months means the same time on every timeframe
# (`strategy.bars_in`): a trading day holds this many bars of each timeframe, and a year this many trading days.
# Measured on the store over 2023-2025: a US listing's session 6.97 hourly and 1.99 four-hour bars (250 sessions a
# year), a coin's day 24.0 and 6.0, a currency pair's 24.0 and 6.0 (260 days), a commodity's 22.8-23.2 and 5.8-6.0
# (the daily pause at 17:00 New York; 258 days).
BARS_PER_DAY = {
    "us_equity": {"1h": 7, "4h": 2, "1d": 1},
    "crypto_perp": {"1h": 24, "4h": 6, "1d": 1},
    "crypto_spot": {"1h": 24, "4h": 6, "1d": 1},
    "fx": {"1h": 24, "4h": 6, "1d": 1},
    "commodity": {"1h": 23, "4h": 6, "1d": 1},
}
TRADING_DAYS = {"us_equity": 252, "crypto_perp": 365, "crypto_spot": 365, "fx": 260, "commodity": 258}

# Smallest order a venue accepts, in USD of notional (None = whole units: one share, see min_capital()).
MIN_TICKET_USD = {"crypto_perp": 5.0, "crypto_spot": 5.0, "us_equity": None, "fx": 1_000.0}

# Walk-forward: parameters first chosen on `first_train` of history, then re-chosen every `test` on all the history
# before it (`train` None), or on the last `train` of it only (a rolling window, e.g. "365D"). One year covers every
# season of the calendar.
WF_SCHEMES = {
    "1h": {"first_train": "365D", "test": "91D", "train": None},
    "4h": {"first_train": "365D", "test": "91D", "train": None},
    "1d": {"first_train": "365D", "test": "91D", "train": None},
}

MC_REPS = 1000                     # Monte Carlo resamples of the daily return series


def env(name: str) -> str | None:
    """Read a setting from the environment, falling back to the repo's .env (never logged)."""
    if name in os.environ:
        return os.environ[name] or None
    path = ROOT_DIR / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                return value.strip() or None
    return None
