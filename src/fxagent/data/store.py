"""Processed dataset store: CSV persistence + provenance rows in SQLite."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fxagent.data.candles import REQUIRED_COLS, clean_ohlc, frame_checksum
from fxagent.storage.db import Repository, session

log = logging.getLogger(__name__)

DATASET_NAME = "EURUSD_H1.csv"


def processed_path(processed_dir: str | Path) -> Path:
    return Path(processed_dir) / DATASET_NAME


def save_dataset(df: pd.DataFrame, processed_dir: str | Path) -> Path:
    path = processed_path(processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = clean_ohlc(df)
    out.to_csv(path, index=False, float_format="%.6f")
    log.info("wrote %s (%d rows)", path, len(out))
    return path


def load_dataset(processed_dir: str | Path) -> pd.DataFrame:
    path = processed_path(processed_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"processed dataset not found: {path} — run scripts/build_dataset.py first"
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[REQUIRED_COLS]


def record_provenance(
    db_path: str | Path,
    *,
    symbol: str,
    timeframe: str,
    source: str,
    source_url: str,
    license_note: str,
    n_rows: int,
    start_ts: str,
    end_ts: str,
    checksum: str,
    notes: str = "",
) -> None:
    with session(db_path) as conn:
        repo = Repository(conn)
        repo.conn.execute(
            """INSERT INTO datasets(symbol, timeframe, source, source_url, license,
               retrieved_at, start_ts, end_ts, n_rows, checksum, notes, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                symbol,
                timeframe,
                source,
                source_url,
                license_note,
                datetime.now(UTC).isoformat(timespec="seconds"),
                start_ts,
                end_ts,
                n_rows,
                checksum,
                notes,
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )


def latest_provenance(db_path: str | Path) -> dict | None:
    with session(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM datasets ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def dataset_checksum(df: pd.DataFrame) -> str:
    return frame_checksum(df)
