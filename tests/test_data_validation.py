"""Data validation tests: schema, duplicates, gaps, invalid prices."""

from __future__ import annotations

import pandas as pd
from tests.conftest import make_ohlc

from fxagent.data.candles import clean_ohlc, validate_ohlc


def test_valid_frame_passes(ohlc):
    rep = validate_ohlc(ohlc)
    assert rep.ok, rep.problems
    assert rep.n_rows == len(ohlc)
    assert rep.checksum


def test_duplicate_timestamps_rejected(ohlc):
    bad = pd.concat([ohlc, ohlc.iloc[[10]]], ignore_index=True).sort_values("timestamp")
    rep = validate_ohlc(bad)
    assert not rep.ok
    assert rep.n_duplicates == 1


def test_unsorted_timestamps_rejected(ohlc):
    bad = ohlc.iloc[::-1].reset_index(drop=True)
    rep = validate_ohlc(bad)
    assert not rep.ok


def test_invalid_prices_rejected(ohlc):
    bad = ohlc.copy()
    bad.loc[5, "close"] = -1.0
    rep = validate_ohlc(bad)
    assert not rep.ok
    assert rep.n_invalid_prices >= 1


def test_ohlc_inconsistency_rejected(ohlc):
    bad = ohlc.copy()
    bad.loc[3, "high"] = bad.loc[3, "low"] - 0.001  # high < low
    rep = validate_ohlc(bad)
    assert not rep.ok
    assert rep.n_ohlc_inconsistent >= 1


def test_missing_column_rejected(ohlc):
    rep = validate_ohlc(ohlc.drop(columns=["high"]))
    assert not rep.ok


def test_naive_timestamps_rejected(ohlc):
    bad = ohlc.copy()
    bad["timestamp"] = bad["timestamp"].dt.tz_localize(None)
    rep = validate_ohlc(bad)
    assert not rep.ok


def test_gaps_flagged_but_not_fatal():
    df = make_ohlc(48)
    # punch a 30h hole
    df = pd.concat([df.iloc[:20], df.iloc[40:]], ignore_index=True)
    rep = validate_ohlc(df, max_gap="6h")
    assert rep.n_gap_flags >= 1
    assert rep.ok  # gaps are informational


def test_clean_ohlc_sorts_and_dedupes(ohlc):
    bad = pd.concat([ohlc.iloc[::-1], ohlc.iloc[[5]]], ignore_index=True)
    clean = clean_ohlc(bad)
    assert clean["timestamp"].is_monotonic_increasing
    assert not clean["timestamp"].duplicated().any()


def test_empty_frame_fails():
    rep = validate_ohlc(pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"]))
    assert not rep.ok
