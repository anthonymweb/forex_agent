"""RL environment determinism, reward semantics, and seeding."""

from __future__ import annotations

import numpy as np
from tests.conftest import make_ohlc

from fxagent.features.base import build_features
from fxagent.models.rl.env import FLAT, ForexTradingEnv


def _setup(n=900, window=24, episode_bars=200):
    h1 = make_ohlc(n, seed=21)
    feats = build_features(h1).fillna(0.0)
    env = ForexTradingEnv(
        feats, h1["close"], window=window, episode_bars=episode_bars, seed=42
    )
    return env, h1


def test_reset_seeded_deterministic():
    env, _ = _setup()
    env2, _ = _setup()
    obs1, _ = env.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)


def test_obs_shape():
    env, _ = _setup(window=17)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (17, env.X.shape[1])
    assert obs.dtype == np.float32


def test_same_action_sequence_same_rewards():
    env1, _ = _setup()
    env2, _ = _setup()
    env1.reset(seed=7)
    env2.reset(seed=7)
    for action in [1, 1, 2, 0, 2, 1, 1, 0]:
        o1, r1, t1, _, _ = env1.step(action)
        o2, r2, t2, _, _ = env2.step(action)
        assert r1 == r2
        np.testing.assert_array_equal(o1, o2)
        assert t1 == t2


def test_flat_always_zero_reward():
    env, _ = _setup()
    env.reset(seed=3)
    for _ in range(30):
        _, reward, term, _, info = env.step(FLAT)
        assert reward == 0.0
        if term:
            break


def test_long_reward_follows_price():
    h1 = make_ohlc(300, seed=5)
    feats = build_features(h1).fillna(0.0)
    env = ForexTradingEnv(
        feats, h1["close"], window=10, episode_bars=280, seed=1,
        cost_pips_per_flip=0.0, reward_scale=1.0, units=100_000,
    )
    env.reset(options={"start": 10})
    # hold long across a known up bar
    i = env._idx
    move = h1["close"].iloc[i + 1] - h1["close"].iloc[i]
    _, reward, _, _, info = env.step(1)  # LONG
    assert info["position"] == 1
    assert abs(reward - 100_000 * move) < 1e-6


def test_flip_cost_charged():
    env, _ = _setup()
    env.reset(options={"start": 30})
    env.step(1)  # go long
    _, reward, _, _, _ = env.step(0)  # flatten -> pay flip cost (scaled)
    assert reward < 0


def test_terminates_at_episode_end():
    env, _ = _setup(episode_bars=50)
    env.reset(options={"start": 30})
    term = False
    steps = 0
    while not term and steps < 500:
        _, _, term, trunc, _ = env.step(1)
        steps += 1
    assert term
    assert not trunc


def test_invalid_action_raises():
    env, _ = _setup()
    env.reset(seed=1)
    try:
        env.step(7)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_too_short_dataset_raises():
    h1 = make_ohlc(30, seed=2)
    feats = build_features(h1).fillna(0.0)
    env = ForexTradingEnv(feats, h1["close"], window=24, episode_bars=100)
    try:
        env.reset(seed=1)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
