"""Configuration loading and path resolution.

Configuration is YAML + optional environment overrides. No secrets live here.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


@dataclass
class Config:
    """Thin wrapper over the merged configuration dictionary."""

    data: dict[str, Any] = field(default_factory=dict)
    root: Path = PROJECT_ROOT

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not cfg_path.is_absolute():
            cfg_path = PROJECT_ROOT / cfg_path
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(data=raw, root=PROJECT_ROOT)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # ---- path helpers -------------------------------------------------
    def path(self, dotted: str) -> Path:
        rel = self.get(dotted)
        if rel is None:
            raise KeyError(f"missing config path: {dotted}")
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    @property
    def db_path(self) -> Path:
        env = os.environ.get("FXAGENT_DB_PATH")
        if env:
            return Path(env)
        return self.path("project.paths.db_path")

    def ensure_dirs(self) -> None:
        for key in ("raw_dir", "processed_dir", "models_dir", "logs_dir", "reports_dir"):
            self.path(f"project.paths.{key}").mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


def periods_per_year_from_config(cfg: Config) -> int:
    """Config may contain a YAML expression like ``252 * 24``; evaluate safely."""
    raw = cfg.get("evaluation.periods_per_year", 6048)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        allowed = set("0123456789+*() ")
        if raw.strip() and set(raw) <= allowed:
            return int(eval(raw, {"__builtins__": {}}, {}))  # noqa: S307
    raise ValueError(f"invalid evaluation.periods_per_year: {raw!r}")
