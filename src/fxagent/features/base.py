"""Causal feature engineering.

Contract (anti-lookahead):
    A feature row at bar ``t`` is a function of information available at or
    before the close of bar ``t`` — i.e. prices up to and including close[t],
    and calendar information of bar t itself. Nothing may read close[t+1] or
    any rolling window that includes future rows.

The trading contract pairs with this: decisions computed from row ``t`` are
executed at the OPEN of bar ``t+1`` (see simulation/backtest.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxagent.data.candles import OHLC


def _log_returns(close: pd.Series, lag: int) -> pd.Series:
    return np.log(close / close.shift(lag))


def _rolling_vol(close: pd.Series, window: int) -> pd.Series:
    return np.log(close / close.shift(1)).rolling(window).std()


def _rolling_mean_distance(close: pd.Series, window: int) -> pd.Series:
    ma = close.rolling(window).mean()
    return close / ma - 1.0


def _range_feature(df: pd.DataFrame, window: int) -> pd.Series:
    rng = (df["high"] - df["low"]) / df["close"]
    return rng.rolling(window).mean()


def _momentum(close: pd.Series, window: int) -> pd.Series:
    """RSI-style bounded momentum in [0, 100]; causal (uses only past closes)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _time_features(ts: pd.Series) -> pd.DataFrame:
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    out = pd.DataFrame(index=ts.index)
    out["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    out["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    out["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)
    return out


def build_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Build the causal feature matrix aligned to ``df`` timestamps.

    Parameters
    ----------
    df : candles with columns timestamp/open/high/low/close (UTC, ascending)
    config : optional ``features`` section of configs/default.yaml

    Returns
    -------
    DataFrame indexed like ``df`` with feature columns only (no OHLC).
    """
    cfg = config or {}
    ret_lags = cfg.get("return_lags", [1, 2, 3, 5, 8, 13])
    vol_windows = cfg.get("vol_windows", [8, 24, 72, 168])
    ma_windows = cfg.get("ma_windows", [8, 24, 72, 168])
    range_windows = cfg.get("range_windows", [8, 24, 72])
    mom_windows = cfg.get("momentum_windows", [14, 24])
    use_tow = cfg.get("use_time_of_week", True)

    if list(df.columns) != ["timestamp", *OHLC]:
        df = df[["timestamp", *OHLC]]

    close = df["close"]
    feats = pd.DataFrame(index=df.index)

    for lag in ret_lags:
        feats[f"ret_{lag}"] = _log_returns(close, lag)

    for w in vol_windows:
        feats[f"vol_{w}"] = _rolling_vol(close, w)

    for w in ma_windows:
        feats[f"ma_dist_{w}"] = _rolling_mean_distance(close, w)

    for w in range_windows:
        feats[f"range_{w}"] = _range_feature(df, w)

    for w in mom_windows:
        feats[f"mom_{w}"] = _momentum(close, w)

    # Intra-bar position of close within the bar's range (causal: uses bar t only).
    span = (df["high"] - df["low"]).replace(0.0, np.nan)
    feats["close_pos"] = ((close - df["low"]) / span).clip(0.0, 1.0).fillna(0.5)

    # Distance from rolling extremes (support/resistance proxy), causal.
    for w in (24, 72, 168):
        hi = df["high"].rolling(w).max()
        lo = df["low"].rolling(w).min()
        rng = (hi - lo).replace(0.0, np.nan)
        feats[f"dist_hi_{w}"] = ((hi - close) / rng).clip(-1.0, 1.0).fillna(0.0)
        feats[f"dist_lo_{w}"] = ((close - lo) / rng).clip(-1.0, 1.0).fillna(0.0)

    if use_tow:
        feats = pd.concat([feats, _time_features(df["timestamp"])], axis=1)

    return feats


def feature_warmup_bars(config: dict | None = None) -> int:
    """Number of leading bars that produce NaN features (dropped before training)."""
    cfg = config or {}
    windows = (
        list(cfg.get("return_lags", [1, 2, 3, 5, 8, 13]))
        + list(cfg.get("vol_windows", [8, 24, 72, 168]))
        + list(cfg.get("ma_windows", [8, 24, 72, 168]))
        + list(cfg.get("range_windows", [8, 24, 72]))
        + list(cfg.get("momentum_windows", [14, 24]))
    )
    return int(max(windows)) + 2
