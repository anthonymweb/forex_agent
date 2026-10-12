"""Shared dataset preparation: load -> features -> labels -> chronological split.

Used by training, backtesting and the simulation runner so that every entry
point sees exactly the same preprocessing (no per-script drift).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fxagent.config import Config
from fxagent.data.store import load_dataset
from fxagent.features.base import build_features, feature_warmup_bars
from fxagent.features.labels import make_labels
from fxagent.features.splits import SplitFrames, assert_sorted_unique, chronological_split


@dataclass
class PreparedData:
    bars: pd.DataFrame          # full H1 candles
    features: pd.DataFrame      # full feature matrix (NaN warm-up rows kept)
    labels: pd.DataFrame        # fwd_ret + label (NaN where undefined)
    splits: SplitFrames         # bar frames per split
    feature_splits: SplitFrames  # feature frames per split
    label_splits: SplitFrames   # label frames per split

    def clean_split(self, name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Bars, features, labels for one split with warm-up/NaN rows dropped."""
        bars = self.splits.frames[name]
        feats = self.feature_splits.frames[name]
        labs = self.label_splits.frames[name]
        mask = ~feats.isna().any(axis=1) & ~labs["label"].isna()
        return bars.loc[mask].reset_index(drop=True), feats.loc[mask].reset_index(drop=True), labs.loc[mask].reset_index(drop=True)


def prepare_dataset(cfg: Config) -> PreparedData:
    bars = load_dataset(cfg.path("project.paths.processed_dir"))
    assert_sorted_unique(bars)
    feat_cfg = cfg.get("features", {}) or {}
    features = build_features(bars, feat_cfg)
    lab_cfg = cfg.get("labels", {}) or {}
    labels = make_labels(
        bars,
        horizon=int(lab_cfg.get("horizon_bars", 4)),
        deadzone_pips=float(lab_cfg.get("deadzone_pips", 0.5)),
        pip_size=float(cfg.get("instrument.pip_size", 0.0001)),
        target=str(lab_cfg.get("target", "direction")),
    )
    splits = chronological_split(bars, cfg.get("split"))
    # Feature/label splits use the same boolean masks via timestamp slices.
    feature_splits = _aligned_frame_split(features, bars, splits)
    label_splits = _aligned_frame_split(labels, bars, splits)
    # Enforce the warm-up: leading rows of the *whole* frame may be NaN; they
    # are dropped per-split by clean_split. Nothing else is altered.
    _ = feature_warmup_bars(feat_cfg)
    return PreparedData(
        bars=bars,
        features=features,
        labels=labels,
        splits=splits,
        feature_splits=feature_splits,
        label_splits=label_splits,
    )


def _aligned_frame_split(frame: pd.DataFrame, bars: pd.DataFrame, splits: SplitFrames) -> SplitFrames:
    parts = {}
    for name, bar_part in splits.frames.items():
        if bar_part.empty:
            parts[name] = frame.iloc[0:0].copy()
        else:
            parts[name] = frame.loc[bar_part.index].copy()
    return SplitFrames(train=parts["train"], val=parts["val"], test=parts["test"])


def model_version_string(cfg: Config, tag: str) -> str:
    horizon = cfg.get("labels.horizon_bars", 4)
    return f"{tag}_h{horizon}"
