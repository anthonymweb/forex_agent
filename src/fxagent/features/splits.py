"""Chronological dataset splitting — no shuffling across time."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def frames(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "test": self.test}


def _slice(df: pd.DataFrame, start: str, end: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    ts = df[timestamp_col]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    mask = (ts >= start_ts) & (ts <= end_ts)
    return df.loc[mask].copy()


def chronological_split(df: pd.DataFrame, split_cfg: dict) -> SplitFrames:
    """Split on configured boundaries. Ranges must not overlap."""
    train = _slice(df, split_cfg["train_start"], split_cfg["train_end"])
    val = _slice(df, split_cfg["val_start"], split_cfg["val_end"])
    test = _slice(df, split_cfg["test_start"], split_cfg["test_end"])
    _assert_disjoint(train, val, test)
    return SplitFrames(train=train, val=val, test=test)


def _assert_disjoint(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    for a, b, name_a, name_b in (
        (train, val, "train", "val"),
        (val, test, "val", "test"),
        (train, test, "train", "test"),
    ):
        if a.empty or b.empty:
            continue
        if a["timestamp"].max() >= b["timestamp"].min():
            raise ValueError(f"{name_a} and {name_b} overlap in time")
        if b["timestamp"].min() <= a["timestamp"].max():
            raise ValueError(f"{name_b} starts before {name_a} ends")


def assert_sorted_unique(df: pd.DataFrame, timestamp_col: str = "timestamp") -> None:
    ts = df[timestamp_col]
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps not sorted ascending")
    if ts.duplicated().any():
        raise ValueError("duplicate timestamps present")
