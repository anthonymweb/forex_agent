"""No-lookahead tests: features and labels never use future information."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tests.conftest import make_ohlc

from fxagent.features.base import build_features, feature_warmup_bars
from fxagent.features.labels import forward_return, make_labels


def test_features_invariant_to_future_data():
    """Changing bars AFTER time t must not change features AT time t."""
    h1 = make_ohlc(300, seed=11)
    full = build_features(h1)
    cut = 200
    truncated = build_features(h1.iloc[: cut + 1])
    assert len(truncated) == cut + 1
    pd.testing.assert_frame_equal(
        truncated.reset_index(drop=True),
        full.iloc[: cut + 1].reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
        atol=1e-12,
    )


def test_features_contain_no_future_columns():
    h1 = make_ohlc(100, seed=3)
    feats = build_features(h1)
    for col in feats.columns:
        assert not col.startswith(("fwd", "label", "future"))
    # index aligned with source bars
    assert list(feats.index) == list(h1.index)


def test_forward_return_matches_timing_contract():
    h1 = make_ohlc(100, seed=5)
    fwd = forward_return(h1, horizon=4)
    for t in range(0, 80, 13):
        expected = np.log(h1["close"].iloc[t + 4] / h1["open"].iloc[t + 1])
        assert abs(fwd.iloc[t] - expected) < 1e-12


def test_labels_from_forward_only():
    h1 = make_ohlc(100, seed=5)
    df = make_labels(h1, horizon=4, deadzone_pips=0.0, target="direction")
    # label 1 = up move, 0 = down move; NaN inside deadzone/tail
    mask = df["label"].notna()
    assert (df.loc[mask, "label"].isin([0.0, 1.0])).all()
    up = df["fwd_ret"] > 0
    check = mask & up
    assert (df.loc[check, "label"] == 1.0).all()
    check_dn = mask & ~up
    assert (df.loc[check_dn, "label"] == 0.0).all()


def test_deadzone_produces_nans():
    h1 = make_ohlc(200, seed=6, vol=0.00001)  # tiny moves -> many inside deadzone
    df = make_labels(h1, horizon=4, deadzone_pips=50.0)
    assert df["label"].isna().any()


def test_label_tail_is_nan():
    h1 = make_ohlc(50, seed=7)
    df = make_labels(h1, horizon=4, deadzone_pips=0.0)
    assert df["label"].iloc[-1] != df["label"].iloc[-1]  # NaN
    assert df["fwd_ret"].iloc[-1] != df["fwd_ret"].iloc[-1]


def test_warmup_bars_positive():
    assert feature_warmup_bars() > 0
    feats = build_features(make_ohlc(300, seed=8))
    warm = feature_warmup_bars()
    assert feats.iloc[:warm].isna().any().any()
