"""Supervised baselines and sklearn model wrappers.

Everything here is CPU-only, seeded and persisted with joblib + JSON metadata.
A model never sees future rows: features are causal and rows are chronological.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(__name__)


class NaiveBaselines:
    """Zero-fit baselines used as honest reference points."""

    @staticmethod
    def always_flat(n: int) -> np.ndarray:
        return np.full(n, -1, dtype=int)  # -1 = no trade

    @staticmethod
    def always_long(n: int) -> np.ndarray:
        return np.ones(n, dtype=int)

    @staticmethod
    def majority_class(y_train: pd.Series) -> NaiveBaselines:
        obj = NaiveBaselines()
        obj._majority = int(y_train.mode().iloc[0]) if len(y_train) else 0
        return obj

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), getattr(self, "_majority", 0), dtype=int)


@dataclass
class SupervisedModelMeta:
    model_name: str
    model_type: str
    version: str
    trained_at: str
    n_train_rows: int
    n_features: int
    feature_names: list[str]
    train_metrics: dict[str, Any]


def _build_estimator(model_type: str, params: dict, seed: int):
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            min_samples_leaf=params.get("min_samples_leaf", 50),
            random_state=seed,
            n_jobs=-1,
        )
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 250),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            random_state=seed,
        )
    if model_type == "logistic":
        return LogisticRegression(
            C=params.get("C", 0.5),
            max_iter=params.get("max_iter", 2000),
            random_state=seed,
        )
    raise ValueError(f"unknown model_type: {model_type}")


class SupervisedDirectionModel:
    """Binary P(up) classifier with persistence."""

    def __init__(self, model_type: str = "gradient_boosting", params: dict | None = None, seed: int = 42):
        self.model_type = model_type
        self.params = params or {}
        self.seed = seed
        self.estimator = _build_estimator(model_type, self.params, seed)
        self.meta: SupervisedModelMeta | None = None

    # ---- training -----------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, version: str = "v1") -> SupervisedModelMeta:
        if len(X) != len(y):
            raise ValueError("X/y length mismatch")
        if X.isna().any().any():
            raise ValueError("features contain NaN — drop warm-up rows before fit")
        self.estimator.fit(X.values, y.values.astype(int))
        proba = self.predict_proba(X)
        metrics = binary_direction_metrics(y.values.astype(int), proba)
        self.meta = SupervisedModelMeta(
            model_name="supervised_direction",
            model_type=self.model_type,
            version=version,
            trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
            n_train_rows=int(len(X)),
            n_features=int(X.shape[1]),
            feature_names=list(X.columns),
            train_metrics={k: float(v) for k, v in metrics.items()},
        )
        return self.meta

    # ---- inference ----------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(class=1) i.e. P(up)."""
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X.values)[:, 1]
        # fallback for estimators without probabilities
        raw = self.estimator.predict(X.values).astype(float)
        return raw

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    # ---- persistence --------------------------------------------------
    def save(self, models_dir: str | Path, name: str = "supervised_direction") -> Path:
        if self.meta is None:
            raise RuntimeError("model not trained")
        directory = Path(models_dir) / name / self.meta.version
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, directory / "model.joblib")
        (directory / "meta.json").write_text(
            json.dumps(asdict(self.meta), indent=2), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, models_dir: str | Path, name: str = "supervised_direction", version: str = "v1"):
        directory = Path(models_dir) / name / version
        meta_raw = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        model = cls(model_type=meta_raw["model_type"], params={}, seed=42)
        model.estimator = joblib.load(directory / "model.joblib")
        model.meta = SupervisedModelMeta(**meta_raw)
        return model


def binary_direction_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Accuracy, balanced accuracy, log-loss and AUC without sklearn.metrics import cycles."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    pred = (proba >= 0.5).astype(int)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, proba)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    else:
        out["roc_auc"] = float("nan")
    base_rate = float(np.mean(y_true))
    out["base_rate_up"] = base_rate
    out["majority_accuracy"] = float(max(base_rate, 1 - base_rate))
    return out
