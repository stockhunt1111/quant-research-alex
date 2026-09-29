"""Intrabar exits against an independent engine (ManifoldBT 0.26.0), on BTCUSDT daily bars 2021-2023.

The fixture holds ManifoldBT's round trips for SMA(10) > SMA(30), long-only, fills at the next open, with a 5%
stop, a 10% take-profit, an 8% trailing stop, and stop + take together. ManifoldBT re-enters right after an
exit while the signal still says long; this engine waits for the signal to change (how the firm's simulator
trades). So trades are matched on their entry: for every entry both engines made, the exit must be the same bar
and the same price. ManifoldBT stamps bars by their open time, this engine by their close time (one day later).
"""
from pathlib import Path

import pandas as pd
import pytest

from strategy_lab import indicators as ind
from strategy_lab.data.bars import FIELDS, Panel
from strategy_lab.data.instruments import parse
from strategy_lab.engine import backtest as bt
from strategy_lab.engine.trades import ledger

FIX = Path(__file__).parent / "fixtures"
VARIANTS = {"stop5": bt.Exits(stop=0.05), "take10": bt.Exits(take=0.10), "trail8": bt.Exits(trail=0.08),
            "stop5_take10": bt.Exits(stop=0.05, take=0.10)}


@pytest.fixture(scope="module")
def panel():
    b = pd.read_parquet(FIX / "btc_1d_2021_2023.parquet")
    return Panel("1d", {"perp:BTCUSDT": parse("perp:BTCUSDT")}, **{f: b[[f]].rename(columns={f: "perp:BTCUSDT"}) for f in FIELDS})


@pytest.fixture(scope="module")
def reference():
    r = pd.read_csv(FIX / "manifoldbt_exits_btc_1d.csv", parse_dates=["entry_timestamp", "exit_timestamp"])
    r["entry_close_time"] = pd.to_datetime(r["entry_timestamp"], utc=True) + pd.Timedelta(days=1)
    r["exit_close_time"] = pd.to_datetime(r["exit_timestamp"], utc=True) + pd.Timedelta(days=1)
    return r


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_exits_match_manifoldbt_on_every_shared_entry(panel, reference, variant, monkeypatch):
    monkeypatch.setattr(bt, "load_funding", lambda instrument_id: None)          # ManifoldBT's runs pay no funding
    c = panel.close["perp:BTCUSDT"]
    target = pd.DataFrame({"perp:BTCUSDT": (ind.sma(c, 10) > ind.sma(c, 30)).astype(float)})
    ours = ledger(panel, bt.run(panel, target, exits=VARIANTS[variant]))
    ours = ours[ours["exit_reason"] != "open_at_end"]
    ref = reference[reference["variant"] == variant]
    m = ours.merge(ref, left_on="entry_time", right_on="entry_close_time", how="inner")
    assert len(m) >= 0.8 * len(ours), f"only {len(m)} of {len(ours)} entries shared"
    assert (m["entry_px"].round(6) == m["entry_price"].round(6)).all()
    same_bar = m["exit_time"] == m["exit_close_time"]
    same_px = (m["exit_px"] / m["exit_price"] - 1).abs() < 1e-9
    diff = m[~(same_bar & same_px)]
    ex = VARIANTS[variant]
    high = panel.high["perp:BTCUSDT"]
    for _, x in diff.iterrows():
        assert x["exit_time"] == x["exit_close_time"], f"different exit bar for entry {x['entry_time']}"
        if x["exit_reason"] == "signal" and ex.stop is not None and abs(x["exit_price"] / (x["entry_px"] * (1 - ex.stop)) - 1) < 1e-9:
            # 1) the signal exits at the open; ManifoldBT books the stop that the bar reached later, after the exit
            assert x["exit_px"] > x["exit_price"]
            continue
        if ex.trail is not None and x["exit_reason"] == "exit_rule":
            # 2) this engine counts the entry bar's high (reached while in the position) toward the trailing peak;
            #    ManifoldBT starts the peak at the entry price and ignores the entry bar's high
            after_entry = high[(high.index > x["entry_time"]) & (high.index < x["exit_time"])]
            mbt_peak = max(x["entry_price"], after_entry.max() if len(after_entry) else x["entry_price"])
            assert abs(x["exit_price"] / (mbt_peak * (1 - ex.trail)) - 1) < 1e-9
            assert x["exit_px"] >= x["exit_price"]
            continue
        raise AssertionError(f"unexplained difference for entry {x['entry_time']}: ours {x['exit_px']} vs {x['exit_price']}")
    assert len(diff) <= 0.15 * len(m)
