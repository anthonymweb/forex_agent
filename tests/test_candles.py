"""Candle aggregation tests (M1->H1 and ticks->H1)."""

from __future__ import annotations

import pandas as pd
from tests.conftest import make_ohlc

from fxagent.data.candles import resample_ohlc, ticks_to_ohlc, validate_ohlc


def _m1_from_h1(h1: pd.DataFrame) -> pd.DataFrame:
    """Expand each H1 bar into 60 identical M1 bars (price constant)."""
    rows = []
    for _, r in h1.iterrows():
        for m in range(60):
            ts = r["timestamp"] + pd.Timedelta(minutes=m)
            rows.append(
                {"timestamp": ts, "open": r["open"], "high": r["high"],
                 "low": r["low"], "close": r["close"]}
            )
    return pd.DataFrame(rows)


def test_resample_m1_to_h1_ohlc_correct():
    h1 = make_ohlc(5, seed=3)
    m1 = _m1_from_h1(h1)
    out = resample_ohlc(m1, rule="1h")
    assert len(out) == 5
    assert out["open"].iloc[0] == h1["open"].iloc[0]
    assert out["close"].iloc[-1] == h1["close"].iloc[-1]
    assert (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-12).all()


def test_resample_preserves_utc():
    m1 = _m1_from_h1(make_ohlc(3))
    out = resample_ohlc(m1)
    assert str(out["timestamp"].dt.tz) == "UTC"


def test_resample_no_weekend_fill():
    # Only two hours of M1 data exist; nothing else is invented by resampling
    ts = pd.date_range("2024-06-03 10:00:00", periods=120, freq="min", tz="UTC")
    m1 = pd.DataFrame(
        {"timestamp": ts, "open": 1.10, "high": 1.11, "low": 1.09, "close": 1.105}
    )
    out = resample_ohlc(m1, rule="1h")
    assert len(out) == 2
    assert out["timestamp"].iloc[0] == pd.Timestamp("2024-06-03 10:00", tz="UTC")


def test_ticks_to_ohlc_basic():
    ts = pd.date_range("2024-06-03 10:00:00", periods=300, freq="17s", tz="UTC")
    bid = 1.10 + 0.0002 * (ts.minute % 3)
    ticks = pd.DataFrame({"timestamp": ts, "bid": bid, "ask": bid + 0.0001})
    out = ticks_to_ohlc(ticks, rule="1h", price_col="bid")
    assert len(out) >= 1
    row = out.iloc[0]
    hour_ticks = ticks[ticks["timestamp"] < row["timestamp"] + pd.Timedelta("1h")]
    assert row["open"] == hour_ticks["bid"].iloc[0]
    assert row["close"] == hour_ticks["bid"].iloc[-1]
    assert row["high"] == hour_ticks["bid"].max()
    assert row["low"] == hour_ticks["bid"].min()


def test_aggregated_output_validates():
    out = ticks_to_ohlc(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-06-03", periods=5000, freq="30s", tz="UTC"),
                "bid": [1.1 + 0.0001 * (i % 5) for i in range(5000)],
                "ask": [1.1001 + 0.0001 * (i % 5) for i in range(5000)],
            }
        )
    )
    rep = validate_ohlc(out, max_gap="6h")
    assert rep.ok, rep.problems
