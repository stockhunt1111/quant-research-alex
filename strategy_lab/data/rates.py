"""The cash rate: the 3-month US Treasury bill (FRED series DTB3, daily, percent a year on a discount basis). The
research desk credits a book's idle cash at T-bills when it compares the book with buy-and-hold at equal risk, and
so does `metrics.equal_risk`. One request, no key:

    python -m strategy_lab.data.rates

Written to data/reference/tbill_3m.csv (date, rate in percent) with the day it was taken.
"""
from __future__ import annotations

import io
from functools import lru_cache

import pandas as pd
import requests

from strategy_lab import log
from strategy_lab.config import REFERENCE_DIR

LOG = log.get("rates")
FRED_DTB3 = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
PATH = REFERENCE_DIR / "tbill_3m.csv"


def fetch(session=None) -> pd.DataFrame:
    session = session or requests.Session()
    r = session.get(FRED_DTB3, timeout=60)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text))
    out = pd.DataFrame({"date": pd.to_datetime(raw.iloc[:, 0]), "rate": pd.to_numeric(raw.iloc[:, 1], errors="coerce")})
    holidays = int(out["rate"].isna().sum())
    if holidays:
        LOG.info("fred DTB3: %d days without a quote (US holidays) left out", holidays)
    return out.dropna(subset=["rate"])


def save(table: pd.DataFrame) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    table.assign(as_of=pd.Timestamp.now(tz="UTC").date()).to_csv(PATH, index=False)
    LOG.info("T-bill rate, %d days from %s to %s, saved to %s", len(table), table["date"].min().date(),
             table["date"].max().date(), PATH)


@lru_cache(maxsize=1)
def daily_rate() -> pd.Series:
    """The T-bill rate earned on one calendar day (UTC days, the metrics' days): the annual rate / 365, a day without a
    quote taking the last one."""
    if not PATH.exists():
        raise FileNotFoundError(f"{PATH} missing: run `python -m strategy_lab.data.rates`")
    t = pd.read_csv(PATH, parse_dates=["date"])
    s = pd.Series(t["rate"].to_numpy() / 100.0 / 365.0, index=pd.DatetimeIndex(t["date"]).tz_localize("UTC"))
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D", tz="UTC")).ffill()


def trailing_return(at: pd.DatetimeIndex, months: int) -> pd.Series:
    """What T-bills returned over the `months` calendar months before each time in `at`, compounded day by day at the
    rate quoted the day before (a day's quote is published after it), to the end of the day a time's bar belongs to
    (a bar closing at midnight ends the day before: at that instant the new day's interest is not earned yet): the
    hurdle of absolute momentum. NaN before the rates begin."""
    growth = (1.0 + daily_rate().shift(1).fillna(0.0)).cumprod()
    day = (at - pd.Timedelta(microseconds=1)).floor("D")
    now = growth.reindex(day, method="ffill").to_numpy()
    then = growth.reindex((day - pd.DateOffset(months=months)), method="ffill").to_numpy()
    return pd.Series(now / then - 1.0, index=at)


def main() -> None:
    log.setup("rates")
    save(fetch())


if __name__ == "__main__":
    main()
