"""Strategy implementations.

Every strategy answers the same question at the close of bar ``t``:
"which position do you want held through bar ``t+1``?" — one of
``flat`` / ``long`` / ``short``. Strategies are pure functions of the
context they are given; they cannot bypass the risk manager or the broker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxagent.simulation.backtest import FLAT, LONG, SHORT, DecisionContext


class AlwaysFlat:
    name = "always_flat"

    def decide(self, ctx: DecisionContext) -> str:
        return FLAT


class AlwaysLong:
    name = "always_long"

    def decide(self, ctx: DecisionContext) -> str:
        return LONG


class SmaCross:
    """Classic fast/slow simple-moving-average crossover on close."""

    def __init__(self, fast: int = 8, slow: int = 48):
        if fast >= slow:
            raise ValueError("fast window must be < slow window")
        self.fast = fast
        self.slow = slow
        self.name = f"sma_cross_{fast}_{slow}"
        self._closes: list[float] = []

    def decide(self, ctx: DecisionContext) -> str:
        close = float(ctx.features["_close"])
        self._closes.append(close)
        if len(self._closes) < self.slow + 1:
            return FLAT
        arr = np.asarray(self._closes[-(self.slow + 1) :], dtype=float)
        fast_now = arr[-self.fast :].mean()
        slow_now = arr[-self.slow :].mean()
        fast_prev = arr[-self.fast - 1 : -1].mean()
        slow_prev = arr[-self.slow - 1 : -1].mean()
        if fast_now > slow_now and fast_prev <= slow_prev:
            return LONG
        if fast_now < slow_now and fast_prev >= slow_prev:
            return SHORT
        # hold current direction while the averages stay crossed
        if fast_now > slow_now:
            return LONG if ctx.want_long or ctx.is_flat else LONG
        if fast_now < slow_now:
            return SHORT
        return FLAT


class MeanReversion:
    """Fade large z-scored deviations from a rolling mean (mean-reversion baseline)."""

    def __init__(self, window: int = 24, entry_z: float = 1.5, exit_z: float = 0.25):
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.name = f"mean_reversion_{window}"

    def decide(self, ctx: DecisionContext) -> str:
        z = float(ctx.features.get(f"ma_dist_{self.window}", np.nan))
        if np.isnan(z):
            return FLAT
        if ctx.position.is_open:
            if abs(z) <= self.exit_z:
                return FLAT
            return LONG if ctx.want_long else SHORT
        if z <= -self.entry_z:
            return LONG  # price far below MA -> expect reversion up
        if z >= self.entry_z:
            return SHORT
        return FLAT


class SupervisedPolicy:
    """Trades the supervised model's P(up) with a no-trade zone."""

    def __init__(self, model, prob_low: float = 0.55, prob_high: float = 0.60):
        self.model = model
        self.prob_low = prob_low
        self.prob_high = prob_high
        self.name = "supervised"
        self._feature_names = list(model.meta.feature_names)

    def decide(self, ctx: DecisionContext) -> str:
        row = ctx.features.reindex(self._feature_names).astype(float)
        if row.isna().any():
            return FLAT
        prob_up = float(self.model.predict_proba(pd.DataFrame([row]))[0])
        if prob_up >= self.prob_high:
            return LONG
        if prob_up <= self.prob_low:
            return SHORT
        return FLAT


class RLPolicy:
    """Wraps a trained stable-baselines3 policy for use inside the backtester."""

    def __init__(self, model, window: int = 24, feature_names: list[str] | None = None):
        self.model = model
        self.window = window
        self.name = "rl"
        self.feature_names = feature_names or []
        self._history: list[np.ndarray] = []

    def decide(self, ctx: DecisionContext) -> str:
        if self.feature_names:
            values = ctx.features.reindex(self.feature_names).astype(float).to_numpy()
        else:
            values = ctx.features.astype(float).to_numpy()
        self._history.append(values)
        if len(self._history) < self.window:
            return FLAT
        window = np.vstack(self._history[-self.window :]).astype(np.float32)
        action, _ = self.model.predict(window, deterministic=True)
        mapping = {0: FLAT, 1: LONG, 2: SHORT}
        return mapping[int(action)]


class HybridPolicy:
    """Documented hybrid decision rules (configs/default.yaml -> hybrid).

    1. RL == FLAT                       -> FLAT (RL veto)
    2. supervised P(up) >= prob_high    -> LONG
       supervised P(up) <= prob_low     -> SHORT
       else                             -> FLAT (no-trade zone)
    3. if require_direction_agreement and RL direction != supervised direction -> FLAT
    4. Risk manager has final veto (applied outside this class).
    """

    def __init__(
        self,
        supervised_model,
        rl_policy: RLPolicy,
        prob_low: float = 0.55,
        prob_high: float = 0.60,
        require_direction_agreement: bool = True,
        min_confidence: float = 0.0,
    ):
        self.supervised = supervised_model
        self.rl = rl_policy
        self.prob_low = prob_low
        self.prob_high = prob_high
        self.require_agreement = require_direction_agreement
        self.min_confidence = min_confidence
        self.name = "hybrid"
        self._feature_names = list(supervised_model.meta.feature_names)

    def decide(self, ctx: DecisionContext) -> str:
        rl_action = self.rl.decide(ctx)
        if rl_action == FLAT:
            return FLAT
        row = ctx.features.reindex(self._feature_names).astype(float)
        if row.isna().any():
            return FLAT
        prob_up = float(self.supervised.predict_proba(pd.DataFrame([row]))[0])
        if prob_up >= self.prob_high:
            sup_action = LONG
        elif prob_up <= self.prob_low:
            sup_action = SHORT
        else:
            return FLAT
        if self.require_agreement and sup_action != rl_action:
            return FLAT
        confidence = abs(prob_up - 0.5)
        if confidence < self.min_confidence:
            return FLAT
        return sup_action


def features_with_close(features: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["_close"] = bars["close"].to_numpy()
    return out
