#!/usr/bin/env python3
"""Seven-day paper-trading experiment CLI (SIMULATED MONEY ONLY).

Commands:
  start   --mode delayed_data_paper|historical_replay [--duration-days 7]
  resume  --experiment-id <id>
  status  [--experiment-id <id>]
  pause / resume-flag / stop   (control-file based; safe while running)

Examples:
  python scripts/experiment.py start --mode delayed_data_paper --strategy hybrid
  python scripts/experiment.py status
  python scripts/experiment.py pause
  python scripts/experiment.py stop
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxagent.config import Config  # noqa: E402
from fxagent.runner.experiment import ExperimentController  # noqa: E402
from fxagent.storage.db import Repository, session  # noqa: E402
from fxagent.strategy.factory import build_strategy  # noqa: E402


def _controller(args, cfg: Config, strategy=None) -> ExperimentController:
    strat = strategy or build_strategy(cfg, args.strategy)
    return ExperimentController(
        cfg, strat,
        mode=args.mode,
        experiment_id=args.experiment_id,
        duration_days=args.duration_days,
        poll_seconds=args.poll_seconds,
        flatten_on_stop=not args.keep_positions,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["start", "resume", "status", "pause", "stop"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--strategy", default="hybrid")
    parser.add_argument("--mode", choices=["historical_replay", "delayed_data_paper"],
                        default="delayed_data_paper")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--duration-days", type=float, default=7.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--keep-positions", action="store_true",
                        help="do not flatten at experiment end")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.load(args.config)
    data_dir = Path(cfg.path("project.paths.raw_dir")).parent

    if args.command == "status":
        exp_id = args.experiment_id
        with session(cfg.db_path) as conn:
            repo = Repository(conn)
            row = repo.get_experiment(exp_id) if exp_id else repo.latest_experiment()
        print(json.dumps({
            "experiment": row,
            "pause_flag": (data_dir / "EXPERIMENT_PAUSE").exists(),
            "stop_flag": (data_dir / "EXPERIMENT_STOP").exists(),
            "emergency_stop": (data_dir / "EMERGENCY_STOP").exists()
            or Path(cfg.get("risk.emergency_stop_file", "data/EMERGENCY_STOP")).exists(),
        }, indent=2, default=str))
        return 0

    if args.command == "pause":
        (data_dir / "EXPERIMENT_PAUSE").write_text("paused\n")
        print(f"pause flag written to {data_dir / 'EXPERIMENT_PAUSE'}")
        return 0

    if args.command == "stop":
        (data_dir / "EXPERIMENT_STOP").write_text("stop\n")
        print(f"stop flag written to {data_dir / 'EXPERIMENT_STOP'}")
        return 0

    if args.command == "resume":
        if not args.experiment_id:
            parser.error("resume requires --experiment-id")
        # resume-flag: clear pause, clear stop, run with same id
        (data_dir / "EXPERIMENT_PAUSE").unlink(missing_ok=True)
        (data_dir / "EXPERIMENT_STOP").unlink(missing_ok=True)
        with session(cfg.db_path) as conn:
            row = Repository(conn).get_experiment(args.experiment_id)
        if row is None:
            print(f"experiment {args.experiment_id} not found", file=sys.stderr)
            return 1
        args.mode = row["mode"]
        args.strategy = row["strategy"]

    # start/resume
    (data_dir / "EXPERIMENT_STOP").unlink(missing_ok=True)
    (data_dir / "EXPERIMENT_PAUSE").unlink(missing_ok=True)
    ctl = _controller(args, cfg)
    summary = ctl.run()
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("status") in (
        "stopped", "duration_reached", "data_exhausted", "completed"
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
