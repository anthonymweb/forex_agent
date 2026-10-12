"""Custom Gymnasium environment for H1 EUR/USD research (simulated money).

Design notes
------------
* Observation: rolling window of *causal, train-normalised* feature rows.
  Row ``t`` of the window contains only information available at close of bar t.
* Actions: Discrete(3) = HOLD FLAT / HOLD LONG / HOLD SHORT.
* Transition: choosing an action at bar ``t`` applies it for bar ``t+1``.
  Reward for that transition = mark-to-market P&L of the position over
  bar t+1 (close[t] -> close[t+1]) minus the flip cost when the position
  changes. This is a close-to-close approximation of costs; final performance
  claims must come from the backtester (next-open fills), not from training
  reward.
* Reward never rewards merely trading: staying flat earns exactly 0.
* Episodes are contiguous chronological segments; data is never shuffled.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

FLAT, LONG, SHORT = 0, 1, 2


class ForexTradingEnv(gym.Env):
    """Deterministic under fixed seeds; CPU-only friendly."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,
        closes: pd.Series,
        *,
        window: int = 24,
        episode_bars: int = 1008,
        units: int = 100_000,
        cost_pips_per_flip: float = 1.0,
        pip_size: float = 0.0001,
        reward_scale: float = 1000.0,
        stats_mean: np.ndarray | None = None,
        stats_std: np.ndarray | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        if len(features) != len(closes):
            raise ValueError("features and closes must be aligned")
        if window < 2:
            raise ValueError("window must be >= 2")

        self.X = features.to_numpy(dtype=np.float32)
        self.closes = closes.to_numpy(dtype=np.float64)
        self.window = int(window)
        self.episode_bars = int(episode_bars)
        self.units = int(units)
        self.flip_cost = float(cost_pips_per_flip) * pip_size * units
        self.reward_scale = float(reward_scale)

        self._mean = stats_mean if stats_mean is not None else np.nanmean(self.X, axis=0)
        self._std = stats_std if stats_std is not None else np.nanstd(self.X, axis=0)
        self._std = np.where(self._std < 1e-8, 1.0, self._std)

        n_features = self.X.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window, n_features), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._idx = 0
        self._episode_end = 0
        self._position = FLAT
        self._rng = np.random.default_rng(seed)

    # ---- helpers ------------------------------------------------------
    def _normalise(self, arr: np.ndarray) -> np.ndarray:
        return ((arr - self._mean) / self._std).astype(np.float32)

    def _obs(self, at: int) -> np.ndarray:
        lo = at - self.window + 1
        window = self.X[lo : at + 1]
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        return self._normalise(window)

    def _available_starts(self) -> np.ndarray:
        first = self.window - 1
        last = len(self.closes) - 2  # need at least one step after start
        return np.arange(first, max(first, last - self.episode_bars + 1))

    # ---- gym API ------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        starts = self._available_starts()
        if len(starts) == 0:
            raise RuntimeError("dataset too short for requested episode length")
        start = int(self._rng.choice(starts)) if options is None else int(options["start"])
        self._idx = start
        self._episode_end = min(len(self.closes) - 1, start + self.episode_bars)
        self._position = FLAT
        return self._obs(self._idx), {}

    def step(self, action: int):
        action = int(action)
        if action not in (FLAT, LONG, SHORT):
            raise ValueError(f"invalid action {action}")

        i = self._idx
        reward = 0.0
        # flip cost when the held position changes
        if action != self._position:
            if self._position != FLAT:
                reward -= self.flip_cost
            self._position = action

        # mark-to-market over the next bar (close[i] -> close[i+1])
        nxt = i + 1
        if nxt < len(self.closes):
            move = self.closes[nxt] - self.closes[i]
            if self._position == LONG:
                reward += self.units * move
            elif self._position == SHORT:
                reward -= self.units * move
        reward *= self.reward_scale

        self._idx = nxt
        terminated = nxt >= self._episode_end
        truncated = False
        obs = self._obs(self._idx) if not terminated else np.zeros(
            (self.window, self.X.shape[1]), dtype=np.float32
        )
        info = {"position": self._position, "bar_index": self._idx}
        return obs, float(reward), terminated, truncated, info
