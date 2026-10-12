"""End-to-end smoke: synthetic bars -> features/labels -> train -> backtest -> report."""

from __future__ import annotations

import json

import numpy as np
from tests.conftest import make_ohlc

from fxagent.data.store import save_dataset
from fxagent.features.base import build_features
from fxagent.features.labels import make_labels
from fxagent.models.supervised.core import SupervisedDirectionModel
from fxagent.reporting.report import write_report
from fxagent.simulation.accounting import CostModel
from fxagent.simulation.backtest import run_backtest
from fxagent.strategy.strategies import SupervisedPolicy, features_with_close


def test_end_to_end_supervised_pipeline(tmp_path):
    h1 = make_ohlc(1500, seed=77)
    # 1. dataset persistence
    path = save_dataset(h1, tmp_path)
    assert path.exists()

    # 2. features + labels (no NaN leakage into training)
    feats = build_features(h1)
    labs = make_labels(h1, horizon=4, deadzone_pips=0.0)
    mask = ~feats.isna().any(axis=1) & labs["label"].notna()
    X = feats.loc[mask].reset_index(drop=True)
    y = labs.loc[mask, "label"].reset_index(drop=True)
    cut = int(len(X) * 0.7)
    Xtr, ytr = X.iloc[:cut], y.iloc[:cut]

    # 3. train
    model = SupervisedDirectionModel(model_type="logistic", seed=42)
    meta = model.fit(Xtr, ytr, version="e2e")
    assert meta.n_train_rows == len(Xtr)

    # 4. backtest the policy out-of-sample
    oos = h1.iloc[mask.to_numpy()].reset_index(drop=True)
    policy = SupervisedPolicy(model, prob_low=0.55, prob_high=0.60)
    feats_oos = features_with_close(X.iloc[cut:].reset_index(drop=True), oos.iloc[cut:])
    cost = CostModel(spread_pips=0.8, slippage_pips=0.2, commission_per_mm=3.0)
    res = run_backtest(
        oos.iloc[cut:].reset_index(drop=True), feats_oos, policy, cost, initial_balance=10_000.0
    )
    assert res.metrics["n_bars"] > 10
    assert np.isfinite(res.metrics["total_return_pct"])

    # 5. report writes artifacts
    paths = write_report({"supervised": res}, tmp_path / "reports")
    assert paths["json"].exists()
    payload = json.loads(paths["json"].read_text())
    assert "supervised" in payload["results"]


def test_end_to_end_with_realistic_split(tmp_path):
    h1 = make_ohlc(2000, seed=88)
    feats = build_features(h1)
    labs = make_labels(h1, horizon=4, deadzone_pips=0.2)
    bars = h1.copy()
    bars["_i"] = np.arange(len(bars))
    # chronological cut by position
    cut1, cut2 = int(len(bars) * 0.7), int(len(bars) * 0.85)
    tr_mask = bars["_i"] < cut1
    va_mask = (bars["_i"] >= cut1) & (bars["_i"] < cut2)
    te_mask = bars["_i"] >= cut2
    clean = ~feats.isna().any(axis=1) & labs["label"].notna()
    Xtr = feats[tr_mask & clean]
    ytr = labs.loc[Xtr.index, "label"]
    model = SupervisedDirectionModel(model_type="gradient_boosting", seed=1)
    model.fit(Xtr, ytr, version="split")
    # val features used only to size the OOS check
    Xva = feats[va_mask & clean]
    proba = model.predict_proba(Xva)
    assert len(proba) == len(Xva)
    # out-of-sample backtest on test segment
    te_bars = h1.loc[te_mask].reset_index(drop=True)
    te_feats = features_with_close(
        feats.loc[te_mask].reset_index(drop=True), te_bars
    )
    res = run_backtest(
        te_bars, te_feats,
        SupervisedPolicy(model),
        CostModel(),
    )
    assert res.metrics["n_bars"] == len(te_bars)
