"""Target labelling for supervised learning.

Timing contract (must match the backtester):

    * Signal computed from data at bar ``t`` (close of t).
    * Position would be entered at ``open[t+1]`` (plus half-spread).
    * Label horizon = ``open[t+1] -> close[t+h]`` economic move,
      because that is what the trader actually earns on an h-bar hold.

The label for row t therefore reads forward prices — that is intentional
(label construction is the ONE place future data is allowed). Features for
row t must never use this function's output or any forward window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Economic forward return: entry at open[t+1], exit at close[t+h]."""
    entry = df["open"].shift(-1)
    exit_ = df["close"].shift(-horizon)
    return np.log(exit_ / entry)


def make_labels(
    df: pd.DataFrame,
    horizon: int = 4,
    deadzone_pips: float = 0.5,
    pip_size: float = 0.0001,
    target: str = "direction",
) -> pd.DataFrame:
    """Return DataFrame with columns: fwd_ret, label (and keep NaN tails)."""
    fwd = forward_return(df, horizon)
    deadzone = deadzone_pips * pip_size
    if target == "direction":
        label = pd.Series(np.nan, index=df.index, dtype="float64")
        label[fwd > deadzone] = 1.0
        label[fwd < -deadzone] = 0.0
        # Inside deadzone stays NaN and rows are dropped before training.
    elif target == "return":
        label = fwd.copy()
    else:
        raise ValueError(f"unknown target: {target}")
    return pd.DataFrame({"fwd_ret": fwd, "label": label})
