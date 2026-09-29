"""Truncation test: a strategy that cannot see the future gives the same positions with the future cut off.

For each cut point the strategy is recomputed on history ending at the cut; its targets up to the cut must equal
the targets computed on the full history. Any difference means some position used data after its own bar
(a centred window, a full-sample statistic, a negative shift...).
"""
from __future__ import annotations

import numpy as np

from strategy_lab.data.bars import Panel
from strategy_lab.strategy import Strategy


class LookAhead(AssertionError):
    pass


def check(strategy: Strategy, panel: Panel, params: dict | None = None, cuts=(0.5, 0.8), tol: float = 1e-10) -> int:
    """Raises LookAhead where a position differs; returns how many positions held before the cuts it compared (none:
    the strategy held nothing that early, and the check proved nothing)."""
    params = params if params is not None else strategy.configs()[0]
    full = strategy.target(panel, params)
    compared = 0
    for frac in cuts:
        cut = panel.index[int(len(panel.index) * frac)]
        part = strategy.target(panel.truncate(cut), params)
        a = full.loc[:cut].to_numpy()
        b = part.reindex_like(full.loc[:cut]).to_numpy()
        bad = ~np.isclose(a, b, atol=tol, equal_nan=True)
        if bad.any():
            rows, cols = np.nonzero(bad)
            first = full.loc[:cut].index[rows.min()]
            raise LookAhead(f"{strategy.name} {params}: positions change when data after {cut} is removed "
                            f"(first at {first}, {int(bad.sum())} cells, e.g. {full.columns[cols[0]]})")
        compared += int(np.count_nonzero(np.nan_to_num(a)))
    return compared


def check_all_configs(strategy: Strategy, panel: Panel, configs: list[dict] | None = None, **kw) -> int:
    """`check` of every configuration (of `configs` when given); returns the positions compared over all of them."""
    return sum(check(strategy, panel, cfg, **kw) for cfg in (strategy.configs() if configs is None else configs))


def discover(modules) -> list[Strategy]:
    """Every Strategy object defined in the given modules."""
    out = []
    for m in modules:
        out.extend(v for v in vars(m).values() if isinstance(v, Strategy))
    return out

