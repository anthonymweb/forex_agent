#!/usr/bin/env python3
"""Download EURUSD M1 data from HistData.com (free, personal-use).

Usage:
    python scripts/download_data.py --start 2018-01 --end 2026-10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxagent.config import Config  # noqa: E402
from fxagent.data.histdata_client import HistDataClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--start", default=None, help="YYYY-MM (default from config)")
    parser.add_argument("--end", default=None, help="YYYY-MM (default from config)")
    parser.add_argument("--force", action="store_true", help="re-download existing months")
    parser.add_argument("--rounds", type=int, default=3, help="M1 retry rounds before tick fallback")
    parser.add_argument("--round-delay", type=float, default=60.0)
    parser.add_argument("--no-tick-fallback", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    start = args.start or cfg.get("data.start_year_month")
    end = args.end or cfg.get("data.end_year_month")
    raw_dir = cfg.path("project.paths.raw_dir")

    client = HistDataClient(
        raw_dir=raw_dir,
        pair=cfg.get("instrument.symbol", "EURUSD"),
        delay_seconds=float(cfg.get("data.request_delay_seconds", 1.2)),
        timeout=int(cfg.get("data.http_timeout_seconds", 120)),
    )
    paths = client.download_range(
        start,
        end,
        force=args.force,
        rounds=args.rounds,
        round_delay=args.round_delay,
        tick_fallback=not args.no_tick_fallback,
    )
    print(f"Downloaded/present archives: {len(paths)} months ({start} .. {end})")
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
