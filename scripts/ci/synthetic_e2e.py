#!/usr/bin/env python3
"""SYNTHETIC end-to-end pipeline check for CI (NOT real market data).

Generates deterministic synthetic H1 bars, runs a short historical_replay
experiment with stop-loss exits through the real ExperimentController +
PaperBroker + RiskManager, then generates a final report from the resulting
SQLite records. Everything is labelled synthetic.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fxagent.config import Config  # noqa: E402
from fxagent.runner.experiment import ExperimentController  # noqa: E402


def synthetic_bars(n: int = 200) -> pd.DataFrame:
    """SYNTHETIC fixture: deterministic zig-zag, not market data."""
    base = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    rows = []
    price = 1.1000
    for i in range(n):
        drift = 0.0004 if (i // 12) % 2 == 0 else -0.0004
        open_ = price
        close = price + drift
        high = max(open_, close) + 0.0003
        low = min(open_, close) - 0.0003
        rows.append({
            "timestamp": base + pd.Timedelta(hours=i),
            "open": round(open_, 5), "high": round(high, 5),
            "low": round(low, 5), "close": round(close, 5),
        })
        price = close
    return pd.DataFrame(rows)


class ZigZagLong:
    """SYNTHETIC strategy: long every 6th bar, flat otherwise."""

    name = "synthetic_zigzag_long"
    model_versions = {"synthetic": "ci"}

    def __init__(self) -> None:
        self._i = 0

    def decide(self, ctx):  # noqa: ARG002
        self._i += 1
        return "long" if self._i % 6 == 1 else "flat"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="fxagent-ci-"))
    try:
        processed = tmp / "processed"
        processed.mkdir()
        df = synthetic_bars(200)
        df.to_csv(processed / "EURUSD_H1.csv", index=False)

        cfg = Config(data={
            "project": {"name": "ci-synthetic", "seed": 42, "base_currency": "USD",
                        "paths": {"raw_dir": str(tmp / "raw"),
                                  "processed_dir": str(processed),
                                  "models_dir": str(tmp / "models"),
                                  "logs_dir": str(tmp / "logs"),
                                  "reports_dir": str(tmp / "reports"),
                                  "db_path": str(tmp / "ci.db")}},
            "instrument": {"symbol": "EURUSD", "pip_size": 0.0001, "point_size": 0.00001},
            "costs": {"spread_pips": 0.8, "slippage_pips": 0.2, "commission_per_million": 3.0},
            "position": {"default_units": 100000, "max_units": 300000},
            "risk": {"max_position_units": 300000, "max_open_positions": 1,
                     "max_daily_loss_usd": 500.0, "max_drawdown_pct": 60.0,
                     "max_spread_pips": 3.0, "stale_data_bars": 3,
                     "max_consecutive_errors": 5,
                     "emergency_stop_file": str(tmp / "raw" / "EMERGENCY_STOP")},
            "paper_broker": {"initial_balance_usd": 10000.0},
            "runner": {"heartbeat_seconds": 30},
            "exits": {"stop_loss_pips": 8, "take_profit_pips": 16, "max_hold_bars": 24},
            "split": {"test_start": "2026-01-01"},
            "features": {"return_lags": [1, 2, 3], "vol_windows": [4, 8],
                         "ma_windows": [4, 8], "range_windows": [4],
                         "momentum_windows": [4], "use_time_of_week": False},
        })

        ctl = ExperimentController(
            cfg, ZigZagLong(), mode="historical_replay",
            experiment_id="ci-synthetic-e2e", duration_days=0.0,
        )
        summary = ctl.run()
        print("experiment summary:", json.dumps(summary, indent=2, default=str))

        assert summary["status"] == "data_exhausted", summary
        assert summary["bars_processed"] > 100, summary
        assert summary["reconcile"]["balanced"] is True, summary
        assert Path(cfg.db_path).exists()

        out = ROOT / "ci-artifacts"
        out.mkdir(exist_ok=True)
        shutil.copy(cfg.db_path, out / "ci.db")
        print("SYNTHETIC E2E OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
