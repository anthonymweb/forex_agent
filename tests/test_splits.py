"""Chronological split integrity tests."""

from __future__ import annotations

import pytest
from tests.conftest import make_ohlc

from fxagent.features.splits import assert_sorted_unique, chronological_split

SPLIT_CFG = {
    "train_start": "2024-01-01", "train_end": "2024-03-31",
    "val_start": "2024-04-01", "val_end": "2024-05-31",
    "test_start": "2024-06-01", "test_end": "2024-12-31",
}


def test_splits_are_chronological_and_disjoint():
    h1 = make_ohlc(3000, seed=4)  # ~4 months from Jan 2024? actually ~124 days
    parts = chronological_split(h1, SPLIT_CFG).frames
    train, val, test = parts["train"], parts["val"], parts["test"]
    assert len(train) and len(val)
    assert train["timestamp"].max() < val["timestamp"].min()
    if len(test):
        assert val["timestamp"].max() < test["timestamp"].min()
    all_ts = __import__("pandas").concat([train["timestamp"], val["timestamp"], test["timestamp"]])
    assert not all_ts.duplicated().any()


def test_overlap_raises():
    # make_ohlc(500) spans ~21 days from 2024-01-01; keep ranges inside it
    h1 = make_ohlc(500, seed=8)
    bad = {
        "train_start": "2024-01-01", "train_end": "2024-01-15",
        "val_start": "2024-01-10", "val_end": "2024-01-18",
        "test_start": "2024-01-19", "test_end": "2024-12-31",
    }
    with pytest.raises(ValueError):
        chronological_split(h1, bad)


def test_sorted_unique_checker():
    h1 = make_ohlc(50, seed=2)
    assert_sorted_unique(h1)  # ok
    with pytest.raises(ValueError):
        assert_sorted_unique(h1.iloc[::-1])
    with pytest.raises(ValueError):
        import pandas as pd
        assert_sorted_unique(pd.concat([h1, h1.iloc[[3]]], ignore_index=True))
