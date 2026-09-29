"""Betting against beta, ported from the previous project (A. Frazzini & L. H. Pedersen, "Betting Against Beta", 2014,
simplified: equal weights in each leg, no leverage). Beta is measured against the equal-weighted return of the
universe's names over the past months; the book is rebuilt at each month's turn and held in between, and stays flat
while fewer than six names have a beta."""
from __future__ import annotations

import numpy as np

from strategy_lab.strategy import bars_in, panel, rebalanced


@panel(grid={"months": [3, 6, 12, 24, 36], "frac": [0.2], "rebalance": ["M"]})
def betting_against_beta(p, months, frac, rebalance, live):
    """Long low-beta, short high-beta names, the short leg scaled so the two legs' betas cancel; gross 1."""
    r = p.close.pct_change().where(live)
    mkt = r.mean(axis=1)
    n = bars_in(p, months=months)
    beta = r.rolling(n).cov(mkt).div(mkt.rolling(n).var().replace(0, np.nan), axis=0)
    ranks = beta.rank(axis=1, pct=True)
    lo = (ranks <= frac).astype(float)
    hi = (ranks >= 1 - frac).astype(float)
    wl = lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0)
    ws = hi.div(hi.sum(axis=1).replace(0, np.nan), axis=0)
    scale = ((beta * wl).sum(axis=1) / (beta * ws).sum(axis=1).replace(0, np.nan)).clip(0.0, 5.0)
    w = wl - ws.mul(scale, axis=0)
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    w = w.div(gross, axis=0).where(beta.notna().sum(axis=1) >= 6, 0.0).fillna(0.0)
    return rebalanced(w, rebalance)
