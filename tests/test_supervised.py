"""Supervised model: train, metrics, persistence, overfit-sanity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tests.conftest import make_ohlc

from fxagent.features.base import build_features
from fxagent.features.labels import make_labels
from fxagent.models.supervised.core import (
    NaiveBaselines,
    SupervisedDirectionModel,
    binary_direction_metrics,
)


def _xy(n=800, seed=42):
    h1 = make_ohlc(n, seed=seed)
    feats = build_features(h1)
    labs = make_labels(h1, horizon=4, deadzone_pips=0.0)
    mask = ~feats.isna().any(axis=1) & labs["label"].notna()
    X = feats.loc[mask].reset_index(drop=True)
    y = labs.loc[mask, "label"].reset_index(drop=True)
    cut = int(len(X) * 0.8)
    return X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]


def test_fit_and_metrics():
    Xtr, ytr, Xte, yte = _xy()
    model = SupervisedDirectionModel(model_type="logistic", seed=42)
    meta = model.fit(Xtr, ytr)
    assert meta.n_train_rows == len(Xtr)
    assert meta.n_features == Xtr.shape[1]
    proba = model.predict_proba(Xte)
    assert len(proba) == len(Xte)
    assert np.all((proba >= 0) & (proba <= 1))
    m = binary_direction_metrics(yte.astype(int).to_numpy(), proba)
    for k in ("accuracy", "log_loss", "roc_auc", "brier"):
        assert k in m
    assert np.isfinite(m["accuracy"])


def test_persistence_roundtrip(tmp_path):
    Xtr, ytr, Xte, _ = _xy(n=400)
    model = SupervisedDirectionModel(model_type="logistic", seed=1)
    model.fit(Xtr, ytr, version="vtest")
    model.save(tmp_path, name="m")
    loaded = SupervisedDirectionModel.load(tmp_path, name="m", version="vtest")
    assert loaded.meta.feature_names == model.meta.feature_names
    np.testing.assert_allclose(model.predict_proba(Xte), loaded.predict_proba(Xte))


def test_gradient_boosting_trains():
    Xtr, ytr, Xte, _ = _xy(n=400, seed=7)
    model = SupervisedDirectionModel(model_type="gradient_boosting", seed=3)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    assert set(np.unique(pred)) <= {0, 1}


def test_rejects_nan_features():
    Xtr, ytr, _, _ = _xy(n=200)
    Xtr = Xtr.copy()
    Xtr.iloc[0, 0] = np.nan
    try:
        SupervisedDirectionModel().fit(Xtr, ytr)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_baselines():
    y = pd.Series([0, 0, 1, 0, 1, 1, 1])
    base = NaiveBaselines.majority_class(y)
    preds = base.predict(pd.DataFrame(np.zeros((4, 2))))
    assert set(preds) == {1}
