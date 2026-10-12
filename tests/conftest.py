"""Shared fixtures: deterministic synthetic OHLC data, tmp config, tmp DB."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fxagent.config import Config  # noqa: E402


def make_ohlc(
    n: int = 500,
    start: str = "2024-01-01 00:00:00",
    seed: int = 7,
    start_price: float = 1.10,
    vol: float = 0.0008,
) -> pd.DataFrame:
    """Deterministic geometric random-walk H1 candles (synthetic TEST data)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, vol, size=n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 3, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 3, n)))
    ts = pd.date_range(start=start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close}
    )


@pytest.fixture
def ohlc() -> pd.DataFrame:
    return make_ohlc(600)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def tmp_cfg(tmp_path: Path, tmp_db: Path) -> Config:
    cfg = Config.load(ROOT / "configs" / "default.yaml")
    cfg.data = {
        **cfg.data,
        "project": {
            **cfg.data["project"],
            "paths": {
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
                "models_dir": str(tmp_path / "models"),
                "logs_dir": str(tmp_path / "logs"),
                "reports_dir": str(tmp_path / "reports"),
                "db_path": str(tmp_db),
            },
        },
        "risk": {**cfg.data.get("risk", {}), "emergency_stop_file": str(tmp_path / "STOP")},
    }
    return cfg
