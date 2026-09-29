"""Betting against beta, ported from the previous project (A. Frazzini & L. H. Pedersen, "Betting Against Beta", 2014,
simplified: equal weights in each leg, no leverage). A name's beta is measured over its own past months, before it
joined the list too, against the equal-weighted return of the list's names; the list's names are ranked by it, as
many in each leg (`strategy.ends`), and the book is rebuilt at each month's turn and held in between, and stays flat
while fewer than six of them have a beta. Measured over its months in the list alone, a name had no beta until it
had been a member that long: on crypto Top-100 over 36 months the book stood flat for half of its record."""
from __future__ import annotations

import numpy as np

from strategy_lab.strategy import as_of, calendar_months, ends, panel, rebalanced


@panel(grid={"months": [3, 6, 12, 24, 36], "frac": [0.2], "rebalance": ["M"]})
def betting_against_beta(p, months, frac, rebalance, live):
    """Long low-beta, short high-beta names, the short leg scaled so the two legs' betas cancel; gross 1."""
    r = p.close.pct_change()
    mkt = r.where(live).mean(axis=1)
    span = calendar_months(months)
    beta = r.rolling(span).cov(mkt).div(mkt.rolling(span).var().replace(0, np.nan), axis=0)
    beta = beta.where(live & as_of(p.close, months=months).notna())          # its months of history behind it
    lo, hi = ends(beta, frac)
    wl = lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0)
    ws = hi.div(hi.sum(axis=1).replace(0, np.nan), axis=0)
    scale = ((beta * wl).sum(axis=1) / (beta * ws).sum(axis=1).replace(0, np.nan)).clip(0.0, 5.0)
    w = wl - ws.mul(scale, axis=0)
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    w = w.div(gross, axis=0).where(beta.notna().sum(axis=1) >= 6, 0.0).fillna(0.0)
    return rebalanced(w, rebalance)
