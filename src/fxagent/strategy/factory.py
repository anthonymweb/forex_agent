"""Strategy factory: build configured strategies with trained artefacts."""

from __future__ import annotations

from pathlib import Path

from fxagent.config import Config
from fxagent.models.supervised.core import SupervisedDirectionModel
from fxagent.strategy.strategies import (
    AlwaysFlat,
    AlwaysLong,
    HybridPolicy,
    MeanReversion,
    RLPolicy,
    SmaCross,
    SupervisedPolicy,
)


def load_supervised(cfg: Config, version: str = "v1") -> SupervisedDirectionModel:
    return SupervisedDirectionModel.load(
        cfg.path("project.paths.models_dir"), name="supervised_direction", version=version
    )


def load_rl_policy(cfg: Config, version: str = "v1", window: int | None = None) -> RLPolicy:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("stable-baselines3 required for RL policy") from exc
    version_dir = Path(cfg.path("project.paths.models_dir")) / "rl_agent" / version
    path = version_dir / "model.zip"
    if not path.exists():
        raise FileNotFoundError(f"RL model not found: {path} — run scripts/train_rl.py")
    model = PPO.load(str(path))
    window = window or int(cfg.get("rl.env.window", 24))
    feature_names: list[str] | None = None
    meta_path = version_dir / "meta.json"
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_names = meta.get("feature_names") or None
    return RLPolicy(model, window=window, feature_names=feature_names)


def build_strategy(
    cfg: Config,
    name: str,
    supervised_version: str = "v1",
    rl_version: str = "v1",
):
    """Instantiate a strategy by name. Model-backed strategies load artefacts."""
    if name == "always_flat":
        return AlwaysFlat()
    if name == "always_long":
        return AlwaysLong()
    if name == "sma_cross":
        return SmaCross(fast=8, slow=48)
    if name == "mean_reversion":
        return MeanReversion(window=24)
    if name == "supervised":
        model = load_supervised(cfg, supervised_version)
        policy = SupervisedPolicy(
            model,
            prob_low=float(cfg.get("supervised.prob_low", 0.55)),
            prob_high=float(cfg.get("supervised.prob_high", 0.60)),
        )
        policy.model_versions = {"supervised": supervised_version}
        return policy
    if name == "rl":
        rl = load_rl_policy(cfg, rl_version)
        rl.model_versions = {"rl": rl_version}
        return rl
    if name == "hybrid":
        model = load_supervised(cfg, supervised_version)
        rl = load_rl_policy(cfg, rl_version)
        policy = HybridPolicy(
            model,
            rl,
            prob_low=float(cfg.get("supervised.prob_low", 0.55)),
            prob_high=float(cfg.get("supervised.prob_high", 0.60)),
            require_direction_agreement=bool(cfg.get("hybrid.require_direction_agreement", True)),
            min_confidence=float(cfg.get("hybrid.min_confidence", 0.0)),
        )
        policy.model_versions = {"supervised": supervised_version, "rl": rl_version}
        return policy
    raise ValueError(f"unknown strategy: {name}")
