#!/usr/bin/env python3
"""Run the continuous paper-trading simulation (replay modes).

Examples:
    # Accelerated replay of 500 bars with the hybrid strategy
    python scripts/run_simulation.py --strategy hybrid --max-bars 500

    # Resume an interrupted run
    python scripts/run_simulation.py --run-id sim-20261011-120000-ab12cd

    # Real-time paced run (wall-clock duration from config runner.duration_days)
    python scripts/run_simulation.py --strategy hybrid --mode realtime
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxagent.config import Config  # noqa: E402
from fxagent.monitoring.logging_setup import setup_logging  # noqa: E402
from fxagent.pipeline import prepare_dataset  # noqa: E402
from fxagent.runner.simulation_runner import SimulationRunner  # noqa: E402
from fxagent.strategy.factory import build_strategy  # noqa: E402
from fxagent.strategy.strategies import features_with_close  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--strategy", default="hybrid")
    parser.add_argument("--run-id", default=None, help="resume an existing run")
    parser.add_argument("--mode", choices=["replay", "realtime"], default=None)
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--step-seconds", type=float, default=None)
    parser.add_argument("--duration-days", type=float, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="which historical window to replay")
    parser.add_argument("--supervised-version", default="v1")
    parser.add_argument("--rl-version", default="v1")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    setup_logging(cfg.path("project.paths.logs_dir"), name="simulation")

    strategy = build_strategy(
        cfg, args.strategy,
        supervised_version=args.supervised_version,
        rl_version=args.rl_version,
    )
    data = prepare_dataset(cfg)
    bars, feats, _ = data.clean_split(args.split)
    feats = features_with_close(feats, bars)
    logging.info("replaying split=%s bars=%d", args.split, len(bars))

    runner = SimulationRunner(
        cfg,
        strategy,
        mode=args.mode,
        run_id=args.run_id,
        step_seconds=args.step_seconds,
        max_bars=args.max_bars,
        duration_days=args.duration_days,
    )
    summary = runner.run(bars, feats)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
