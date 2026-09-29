"""What the strategies with a model share: indicator families as features, the refit calendar and the model.

* `features(bars)` — five families of scale-free indicators (trend, momentum, reversion, volatility, volume);
  every column is a ratio, a z-score
  or an oscillator, so one model form fits any price level, and every value at a bar uses bars up to it.
* `refit_rows(index)` — the bars a model is refit at: every walk-forward window start (`config.WF_SCHEMES`, calendar
  spans, the same on every timeframe), and twice in the first year so that the first choice has half a year of
  predictions to judge.
* `model(seed)` — LightGBM with small trees and default-like settings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_lab import indicators as ind
from strategy_lab.config import WF_SCHEMES

FAMILIES = ("trend", "momentum", "reversion", "volatility", "volume")


def features(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Indicator families, every column scale-free so that one model form fits any price level."""
    o, h, l, c, v = (bars[k] for k in ("open", "high", "low", "close", "volume"))
    r = np.log(c).diff()
    macd = ind.macd(c)
    bb = ind.bbands(c, 20, 2.0)
    stoch = ind.call("STOCH", h, l, c)
    vsd = v.rolling(60).std()
    obv = ind.call("OBV", c, v)
    fam = {
        "trend": pd.DataFrame({"dist_ema_20": c / ind.ema(c, 20) - 1, "dist_ema_50": c / ind.ema(c, 50) - 1,
                               "dist_ema_200": c / ind.ema(c, 200) - 1, "macd_hist": macd["hist"] / c,
                               "adx_14": ind.adx(h, l, c, 14)}),
        "momentum": pd.DataFrame({"ret_5": c / c.shift(5) - 1, "ret_20": c / c.shift(20) - 1, "ret_60": c / c.shift(60) - 1,
                                  "rsi_14": ind.rsi(c, 14), "stoch_k": stoch.iloc[:, 0]}),
        "reversion": pd.DataFrame({"rsi_2": ind.rsi(c, 2), "ibs": ind.ibs(bars),
                                   "bb_pct_b": (c - bb["lower"]) / (bb["upper"] - bb["lower"]),
                                   "willr_14": ind.call("WILLR", h, l, c, timeperiod=14),
                                   "dist_high_20": c / h.rolling(20).max() - 1, "dist_low_20": c / l.rolling(20).min() - 1}),
        "volatility": pd.DataFrame({"atr_pct": ind.atr(h, l, c, 14) / c, "vol_20": r.rolling(20).std(),
                                    "vol_ratio": r.rolling(20).std() / r.rolling(120).std(),
                                    "bb_width": (bb["upper"] - bb["lower"]) / bb["middle"],
                                    "gap": o / c.shift(1) - 1}),
        # FX has no volume: a constant (zero) series has no spread, and 0/0 would drop every row of the pair
        "volume": pd.DataFrame({"volume_z": ((v - v.rolling(60).mean()) / vsd).where(vsd > 0, 0.0),
                                "mfi_14": ind.call("MFI", h, l, c, v, timeperiod=14).where(vsd > 0, 50.0),
                                "obv_slope": ((obv - obv.shift(20)) / (v.rolling(20).mean() * 20)).where(vsd > 0, 0.0)}),
    }
    return {k: f.replace([np.inf, -np.inf], np.nan) for k, f in fam.items()}


def refit_rows(index: pd.DatetimeIndex) -> list[int]:
    """Row positions of the refits: every walk-forward window start, and two in the first year before the first (the
    walk-forward's windows are calendar spans, the same on every timeframe)."""
    scheme = WF_SCHEMES["1d"]
    first, step = pd.Timedelta(scheme["first_train"]), pd.Timedelta(scheme["test"])
    at = [index[0] + first - 2 * step, index[0] + first - step]
    t = index[0] + first
    while t <= index[-1]:
        at.append(t)
        t += step
    rows = sorted({int(index.searchsorted(a)) for a in at})
    return [k for k in rows if 0 < k < len(index)]


def model(seed: int):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, min_child_samples=40, random_state=seed, n_jobs=1, verbose=-1)
