import numpy as np
import pandas as pd

import strategies.funding_carry as carry
from strategies.betting_against_beta import betting_against_beta
from strategies.xs_momentum import xs_momentum
from strategy_lab import lookahead
from tests.conftest import make_panel


def _fake_funding(monkeypatch, panel):
    rng = np.random.default_rng(3)
    settle = pd.date_range(panel.index.min() - pd.Timedelta(days=40), panel.index.max(), freq="8h")
    table = {i: pd.Series(rng.normal(1e-4, 2e-4, len(settle)), index=settle) for i in panel.ids}
    monkeypatch.setattr(carry, "load_funding", lambda i: table[i])
    return table


def test_trailing_funding_uses_only_settlements_up_to_each_time():
    s = pd.Series([1.0, 2.0, 4.0], index=pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 16:00"], tz="UTC"))
    at = pd.DatetimeIndex(["2024-01-02 03:00", "2024-01-02 12:00"], tz="UTC")
    s2 = pd.concat([pd.Series([10.0], index=pd.DatetimeIndex(["2023-12-31 00:00"], tz="UTC")), s,
                    pd.Series([100.0], index=pd.DatetimeIndex(["2024-01-02 08:00"], tz="UTC"))])
    out = carry.trailing_funding(s2, at, days=1)
    assert out.iloc[0] == 1.0 + 2.0 + 4.0 - 1.0 - 2.0 - 4.0 + 2.0 + 4.0      # window (01-01 03:00, 01-02 03:00]
    assert out.iloc[1] == 4.0 + 100.0                                          # (01-01 12:00, 01-02 12:00]


def test_carry_and_cross_sectional_books_pass_truncation_on_hourly_bars(monkeypatch):
    ids = tuple(f"td:S{k}" for k in range(8))
    p = make_panel(ids=ids, n=900, seed=21, freq="h", start="2024-01-01 01:00")
    _fake_funding(monkeypatch, p)
    for strat in (carry.funding_carry, xs_momentum, betting_against_beta):
        lookahead.check_all_configs(strat, p)
        w = strat.target(p, strat.configs()[0])
        assert (w.abs().sum(axis=1) <= 1 + 1e-9).all()
