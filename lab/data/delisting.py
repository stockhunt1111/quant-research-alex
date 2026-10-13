"""When an instrument stopped trading: delisted, by the exchange's record in its series' meta, or its data ended long
before the panel's (`ended`), where a backtest closes its position and a list gives its seat to the next name."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from lab.data import store
from lab.data.bars import Panel

DATA_LAG = pd.Timedelta(days=7)


def ended(panel: Panel) -> pd.DataFrame:
    """True where an instrument is delisted: on the bars after its last one when it stopped trading (a perp the
    exchange delisted, its date on record in the series' meta, or a series whose last bar is more than DATA_LAG
    before the panel's last bar; a shorter shortfall with no delisting on record is a vendor lag, not an end), and
    between the two listings of a perp delisted and later listed anew under the same name (the stretch on record).
    Worked out once per panel and delisting records; each caller gets its own copy."""
    records = _records(panel)
    got = panel.memo.get("ended")
    if got is None or got[0] != records:
        got = panel.memo["ended"] = records, _ended(panel, dict(zip(panel.instruments, records)))
    return got[1].copy()


def _records(panel: Panel) -> tuple:
    """Each instrument's delisting on record (`_delisting`), as the store holds it now."""
    return tuple(_delisting(ins, panel.timeframe) for ins in panel.instruments.values())


def _ended(panel: Panel, records: dict[str, tuple[bool, tuple]]) -> pd.DataFrame:
    printed = panel.close.notna().to_numpy()
    n = len(panel.index)
    ever = printed.any(axis=0)
    last = np.where(ever, n - 1 - np.argmax(printed[::-1], axis=0), -1)     # each instrument's last bar
    after = np.arange(n)[:, None] > last[None, :]                          # no bar from here on (all: never one)
    idx = panel.index
    stopped = ever & ((idx[-1] - idx[np.maximum(last, 0)]) > DATA_LAG) if n else ever
    stopped |= np.array([records[i][0] for i in panel.ids], dtype=bool)
    out = after & stopped[None, :]
    for k, i in enumerate(panel.ids):
        for lo, hi in records[i][1]:
            out[(idx > lo) & (idx < hi), k] = True
    return pd.DataFrame(out, index=idx, columns=panel.ids)


def _delisting(ins, timeframe: str) -> tuple[bool, tuple]:
    """What `refresh binance-delistings` kept in a perp's series' meta: delisted for good, and the stretches between
    a delisting and a new listing under the same name."""
    if ins.source != "perp":
        return False, ()
    p = store.meta_path("perp", timeframe, ins.symbol)
    return _meta_delisting(str(p), p.stat().st_mtime_ns) if p.exists() else (False, ())


@lru_cache(maxsize=None)
def _meta_delisting(path: str, version: int) -> tuple[bool, tuple]:
    meta = json.loads(Path(path).read_text())
    between = tuple((pd.Timestamp(lo), pd.Timestamp(hi)) for lo, hi in meta.get("delisted_periods", []))
    return bool(meta.get("delisted")), between
