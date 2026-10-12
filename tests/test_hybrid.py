"""Hybrid policy decision rules: RL veto, no-trade zone, direction agreement."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from fxagent.simulation.backtest import FLAT, LONG, SHORT, DecisionContext, PositionState
from fxagent.strategy.strategies import HybridPolicy, RLPolicy


class FakeSupervised:
    """Stub with SupervisedDirectionModel-compatible surface."""

    def __init__(self, prob_up: float):
        self._p = float(prob_up)
        self.meta = SimpleNamespace(feature_names=["f1", "f2"])

    def predict_proba(self, X) -> np.ndarray:
        return np.full(len(X), self._p)


class FakeRL:
    def __init__(self, action: str):
        self._a = action

    def decide(self, ctx) -> str:
        return self._a


def _ctx(p1=0.0, p2=0.0) -> DecisionContext:
    return DecisionContext(
        index=0,
        timestamp=pd.Timestamp("2024-06-03T10:00:00Z", tz="UTC"),
        features=pd.Series({"f1": p1, "f2": p2}),
        position=PositionState(),
        bars_seen=0,
    )


def _hybrid(
    prob: float, rl_action: str, low=0.55, high=0.60, agree=True, min_confidence=0.0
) -> HybridPolicy:
    return HybridPolicy(
        supervised_model=FakeSupervised(prob),
        rl_policy=FakeRL(rl_action),
        prob_low=low,
        prob_high=high,
        require_direction_agreement=agree,
        min_confidence=min_confidence,
    )


def test_rl_flat_vetoes_everything():
    assert _hybrid(0.99, FLAT).decide(_ctx()) == FLAT
    assert _hybrid(0.01, FLAT).decide(_ctx()) == FLAT


def test_no_trade_zone_returns_flat():
    assert _hybrid(0.57, LONG).decide(_ctx()) == FLAT
    assert _hybrid(0.58, SHORT).decide(_ctx()) == FLAT


def test_agreement_required():
    # sup says long, rl says short -> flat
    assert _hybrid(0.90, SHORT, agree=True).decide(_ctx()) == FLAT
    # sup says short, rl says long -> flat
    assert _hybrid(0.10, LONG, agree=True).decide(_ctx()) == FLAT
    # agreeing -> direction taken
    assert _hybrid(0.90, LONG, agree=True).decide(_ctx()) == LONG
    assert _hybrid(0.10, SHORT, agree=True).decide(_ctx()) == SHORT


def test_agreement_not_required_uses_supervised():
    assert _hybrid(0.90, SHORT, agree=False).decide(_ctx()) == LONG
    assert _hybrid(0.10, LONG, agree=False).decide(_ctx()) == SHORT


def test_nan_features_block():
    policy = _hybrid(0.90, LONG)
    ctx = DecisionContext(
        index=0,
        timestamp=pd.Timestamp("2024-06-03T10:00:00Z", tz="UTC"),
        features=pd.Series({"f1": np.nan, "f2": 0.0}),
        position=PositionState(),
        bars_seen=0,
    )
    assert policy.decide(ctx) == FLAT


def test_min_confidence_gate():
    policy = _hybrid(0.56, LONG, low=0.55, high=0.56)
    policy.min_confidence = 0.2  # |0.56-0.5|=0.06 < 0.2
    assert policy.decide(_ctx()) == FLAT
    policy2 = _hybrid(0.90, LONG, min_confidence=0.2)
    assert policy2.decide(_ctx()) == LONG


def test_rl_policy_maps_actions():
    class FakeModel:
        def predict(self, window, deterministic=True):
            return np.array(1), None  # 1 = LONG

    policy = RLPolicy(FakeModel(), window=3, feature_names=["a"])
    # feed 3 warmup bars then a real decision
    for _ in range(3):
        act = policy.decide(
            DecisionContext(0, pd.Timestamp("2024-06-03T10:00:00Z", tz="UTC"),
                            pd.Series({"a": 1.0}), PositionState(), 0)
        )
    assert act == LONG
