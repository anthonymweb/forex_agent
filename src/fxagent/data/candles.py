"""OHLC candle utilities: validation, resampling and integrity checks."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

OHLC = ["open", "high", "low", "close"]
REQUIRED_COLS = ["timestamp", *OHLC]


@dataclass
class ValidationReport:
    ok: bool = True
    n_rows: int = 0
    n_duplicates: int = 0
    n_missing_values: int = 0
    n_invalid_prices: int = 0
    n_ohlc_inconsistent: int = 0
    n_gap_flags: int = 0
    start_ts: str | None = None
    end_ts: str | None = None
    checksum: str = ""
    problems: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.problems.append(msg)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_rows": self.n_rows,
            "n_duplicates": self.n_duplicates,
            "n_missing_values": self.n_missing_values,
            "n_invalid_prices": self.n_invalid_prices,
            "n_ohlc_inconsistent": self.n_ohlc_inconsistent,
            "n_gap_flags": self.n_gap_flags,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "checksum": self.checksum,
            "problems": self.problems,
        }


def frame_checksum(df: pd.DataFrame) -> str:
    """Deterministic content hash (timestamped OHLC only)."""
    cols = REQUIRED_COLS
    payload = pd.util.hash_pandas_object(df[cols], index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def validate_ohlc(
    df: pd.DataFrame,
    max_gap: str = "6h",
    expect_utc: bool = True,
) -> ValidationReport:
    """Validate schema, ordering, duplicates, prices and gaps.

    Returns a report; ``report.ok`` is False only for *fatal* problems
    (schema, invalid prices, OHLC inconsistencies, duplicate timestamps).
    Gaps are flagged but not fatal — forex weekends are legitimate.
    """
    rep = ValidationReport()
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        rep.fail(f"missing columns: {missing}")
        return rep

    work = df[REQUIRED_COLS].copy()
    rep.n_rows = len(work)
    if work.empty:
        rep.fail("empty dataset")
        return rep

    if not pd.api.types.is_datetime64_any_dtype(work["timestamp"]):
        rep.fail("timestamp column is not datetime")
        return rep
    if expect_utc and getattr(work["timestamp"].dt, "tz", None) is None:
        rep.fail("timestamps are not timezone-aware UTC")

    ts = work["timestamp"]
    rep.start_ts = str(ts.min())
    rep.end_ts = str(ts.max())

    dupes = int(ts.duplicated().sum())
    rep.n_duplicates = dupes
    if dupes:
        rep.fail(f"{dupes} duplicate timestamps")

    if not ts.is_monotonic_increasing:
        rep.fail("timestamps not sorted ascending")

    n_missing = int(work[OHLC].isna().sum().sum())
    rep.n_missing_values = n_missing
    if n_missing:
        rep.fail(f"{n_missing} missing OHLC values")

    numeric = work[OHLC].apply(pd.to_numeric, errors="coerce")
    invalid = int((numeric <= 0).sum().sum())
    rep.n_invalid_prices = invalid
    if invalid:
        rep.fail(f"{invalid} non-positive prices")

    bad = (
        (numeric["high"] < numeric["low"])
        | (numeric["high"] < numeric["open"])
        | (numeric["high"] < numeric["close"])
        | (numeric["low"] > numeric["open"])
        | (numeric["low"] > numeric["close"])
    )
    n_bad = int(bad.sum())
    rep.n_ohlc_inconsistent = n_bad
    if n_bad:
        rep.fail(f"{n_bad} OHLC-inconsistent bars")

    gaps = ts.diff()
    max_gap_td = pd.Timedelta(max_gap)
    n_gaps = int((gaps > max_gap_td).sum())
    rep.n_gap_flags = n_gaps

    rep.checksum = frame_checksum(work)
    return rep


def clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, drop duplicate timestamps (keep first) and reset index."""
    work = df[REQUIRED_COLS].copy()
    work = work.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")
    work = work.reset_index(drop=True)
    return work


def resample_ohlc(m1: pd.DataFrame, rule: str = "1h") -> pd.DataFrame:
    """Aggregate M1 candles into higher timeframe OHLC bars (UTC-indexed)."""
    if m1.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)
    work = m1.set_index("timestamp")[OHLC].sort_index()
    # Forex week: keep only bars that existed in the source (no weekend fill).
    agg = work.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    agg = agg.dropna(subset=["open", "close"])
    # A bar with open/close but no range data would be malformed; recompute high/low safety.
    agg["high"] = agg[["open", "high", "close"]].max(axis=1)
    agg["low"] = agg[["open", "low", "close"]].min(axis=1)
    out = agg.reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out[REQUIRED_COLS].astype(
        {"open": "float64", "high": "float64", "low": "float64", "close": "float64"}
    )


def ticks_to_ohlc(ticks: pd.DataFrame, rule: str = "1h", price_col: str = "bid") -> pd.DataFrame:
    """Aggregate bid (or ask) ticks into OHLC bars of ``rule`` size."""
    if ticks.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)
    work = ticks.set_index("timestamp")[[price_col]].sort_index()
    agg = work.resample(rule, label="left", closed="left")[price_col].agg(
        open="first", high="max", low="min", close="last"
    )
    agg = agg.dropna(subset=["open", "close"])
    out = agg.reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out[REQUIRED_COLS].astype(
        {"open": "float64", "high": "float64", "low": "float64", "close": "float64"}
    )


def tick_spread_stats(ticks: pd.DataFrame) -> dict:
    """Observed bid/ask spread statistics (pips if pip_size given)."""
    if ticks.empty or "bid" not in ticks.columns:
        return {}
    spread = ticks["ask"] - ticks["bid"]
    return {
        "mean_spread": float(spread.mean()),
        "median_spread": float(spread.median()),
        "p95_spread": float(spread.quantile(0.95)),
        "n_ticks": int(len(ticks)),
    }


def expected_h1_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Full hourly grid between two UTC timestamps (for gap accounting)."""
    return pd.date_range(start=start, end=end, freq="1h", tz="UTC")


def largest_gap(df: pd.DataFrame) -> pd.Timedelta:
    if len(df) < 2:
        return pd.Timedelta(0)
    return df["timestamp"].diff().max()


def assert_no_lookahead(features: pd.DataFrame, source_close: pd.Series) -> None:
    """Cheap sanity guard: every feature row must be computable from source rows
    with index <= that row's own source timestamp.

    Features are indexed by the same timestamps as ``source_close``. For each
    feature column, we verify that shifting the source by +1 (i.e. making the
    *next* bar's close available) does not change equality of the feature when
    recomputed — this is done in tests with concrete feature functions; here we
    only check that features do not correlate perfectly with the *future* close.
    """
    future_close = source_close.shift(-1)
    for col in features.columns:
        f = features[col].astype(float)
        if np.allclose(f.fillna(0), future_close.fillna(0)):
            raise AssertionError(f"feature {col} equals next-bar close (leakage)")
