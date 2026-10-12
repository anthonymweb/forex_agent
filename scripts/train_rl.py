#!/usr/bin/env python3
"""Train a PPO agent inside the custom Gymnasium forex environment.

Training uses ONLY the training split. Evaluation is performed separately by
scripts/backtest.py on validation/test using the next-open execution model —
training reward is a close-to-close approximation and is never reported as
trading performance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxagent.config import Config  # noqa: E402
from fxagent.models.rl.env import ForexTradingEnv  # noqa: E402
from fxagent.pipeline import prepare_dataset  # noqa: E402

log = logging.getLogger("train_rl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--timesteps", type=int, default=None, help="override rl.ppo.total_timesteps")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("stable-baselines3 + torch are required for RL training") from exc

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    seed = int(cfg.get("project.seed", 42))

    data = prepare_dataset(cfg)
    train_bars, X_train, y_train = data.clean_split("train")
    val_bars, X_val, _ = data.clean_split("val")
    log.info("train bars=%d  val bars=%d", len(X_train), len(val_bars))

    env_cfg = cfg.get("rl.env", {}) or {}
    ppo_cfg = dict(cfg.get("rl.ppo", {}) or {})
    total_timesteps = args.timesteps or int(ppo_cfg.pop("total_timesteps", 200_000))
    n_eval_episodes = int(ppo_cfg.pop("n_eval_episodes", 10))
    ppo_cfg.pop("seed", None)

    window = int(env_cfg.get("window", 24))
    episode_bars = int(env_cfg.get("episode_bars", 1008))
    units = int(cfg.get("position.default_units", 100_000))
    pip_size = float(cfg.get("instrument.pip_size", 0.0001))

    make_env = lambda df, closes, s: ForexTradingEnv(  # noqa: E731
        df,
        closes,
        window=window,
        episode_bars=episode_bars,
        units=units,
        cost_pips_per_flip=float(env_cfg.get("cost_pips_per_flip", 1.0)),
        pip_size=pip_size,
        reward_scale=float(env_cfg.get("reward_scale", 1000.0)),
        seed=s,
    )

    train_env = make_env(X_train, train_bars["close"], seed)
    eval_env = make_env(X_val, val_bars["close"], seed + 1)

    model = PPO(
        "MlpPolicy",
        train_env,
        seed=seed,
        verbose=1,
        **ppo_cfg,
    )

    models_dir = cfg.path("project.paths.models_dir")
    version_dir = models_dir / "rl_agent" / args.version
    version_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(version_dir / "best"),
        log_path=str(version_dir / "logs"),
        eval_freq=max(10_000, total_timesteps // 10),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    callbacks.append(eval_cb)

    log.info("starting PPO training for %d timesteps (CPU)", total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)

    final_path = version_dir / "model.zip"
    model.save(str(final_path))
    meta = {
        "algorithm": "PPO",
        "total_timesteps": total_timesteps,
        "window": window,
        "episode_bars": episode_bars,
        "units": units,
        "cost_pips_per_flip": float(env_cfg.get("cost_pips_per_flip", 1.0)),
        "seed": seed,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "feature_names": list(X_train.columns),
        "best_mean_reward": float(getattr(eval_cb, "best_mean_reward", float("nan"))),
        "note": (
            "Training reward is a close-to-close approximation with per-flip costs. "
            "It is NOT trading performance; use scripts/backtest.py for evaluation."
        ),
    }
    (version_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("saved RL model to %s", final_path)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
