"""What the pages show about the lists and instruments, worked out from the daily bars and kept in the app's database:
a list's size (a fixed list's instruments with bars and those trading in its last ten days; a ranked list's most names
held at once and all it ever held), where its coins trade, what a trade costs it a side (a spot quote of a commodity:
the median over its hourly bars' last year, `engine.costs.typical_bps`); an instrument's median dollar
volume over its last 60 daily bars, the order of the Asset row's quick picks.

    python -m strategy_lab db lists          (also the last step of a run)

Loading every list's daily bars takes tens of seconds: the pages read what is kept.
"""
from __future__ import annotations

from collections import defaultdict

from strategy_lab import db, lists, log
from strategy_lab.data.bars import load_panel
from strategy_lab.engine import costs
from strategy_lab.data.instruments import parse
from strategy_lab.universes import resolve

LOG = log.get("catalog")
LIQUIDITY_DAYS = 60
TRADING_DAYS = 10


def list_row(universe: str) -> tuple:
    uni = resolve(universe, "1d")
    panel = load_panel(uni.ids, "1d", fields=("close", "dollar_volume"))
    printed = panel.close.notna()
    fixed = trading = held = ever = None
    if uni.member is None:
        fixed, trading = int(printed.any().sum()), int(printed.iloc[-TRADING_DAYS:].any().sum())
    else:
        m = uni.member(panel)
        held, ever = int(m.sum(axis=1).max()), int(m.any().sum())
    sides = [round(costs.typical_bps(i), 1) for i in panel.ids]
    sources = sorted({i.split(":", 1)[0] for i in uni.ids})
    return (universe, fixed, trading, held, ever, db.to_json(sources), float(min(sides)), float(max(sides)), db.now())


def liquidity(ids: list[str]) -> dict[str, float | None]:
    """Each instrument's median dollar volume over its last LIQUIDITY_DAYS daily bars (a panel holds one calendar)."""
    by_calendar: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        by_calendar[parse(i).calendar].append(i)
    out: dict[str, float | None] = {}
    for group in by_calendar.values():
        dv = load_panel(group, "1d", fields=("close", "dollar_volume")).dollar_volume
        med = dv.iloc[-LIQUIDITY_DAYS:].median()
        out |= {i: db.number(med.get(i)) for i in group}
    return out


def refresh(conn=None) -> None:
    """Work the lists' and the single-asset instruments' figures out again and keep them."""
    own = conn is None
    conn = db.connect() if own else conn
    try:
        universes = list(dict.fromkeys([*lists.names(lists.OURS), *lists.PER_INSTRUMENT]))
        rows = [list_row(u) for u in universes]
        ids = [r[0] for r in conn.execute("SELECT DISTINCT instrument_id FROM result WHERE instrument_id IS NOT NULL")]
        liq = liquidity(ids) if ids else {}
        with db.write(conn):
            conn.executemany("INSERT INTO list_stats (list_id, size_fixed, size_trading, size_held, size_ever, sources, "
                             "cost_bp_min, cost_bp_max, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT "
                             "(list_id) DO UPDATE SET size_fixed = excluded.size_fixed, size_trading = "
                             "excluded.size_trading, size_held = excluded.size_held, size_ever = excluded.size_ever, "
                             "sources = excluded.sources, cost_bp_min = excluded.cost_bp_min, cost_bp_max = "
                             "excluded.cost_bp_max, updated_at = excluded.updated_at", rows)
            now = db.now()
            conn.executemany("INSERT INTO instrument_stats (instrument_id, liquidity_usd, as_of) VALUES (?, ?, ?) "
                             "ON CONFLICT (instrument_id) DO UPDATE SET liquidity_usd = excluded.liquidity_usd, "
                             "as_of = excluded.as_of", [(i, v, now) for i, v in liq.items()])
        LOG.info("lists' figures for %d lists, liquidity for %d instruments", len(rows), len(liq))
    finally:
        if own:
            conn.close()
