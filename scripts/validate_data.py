#!/usr/bin/env python3
"""Validate the processed dataset and print a provenance-aware report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxagent.config import Config  # noqa: E402
from fxagent.data.candles import validate_ohlc  # noqa: E402
from fxagent.data.store import latest_provenance, load_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    df = load_dataset(cfg.path("project.paths.processed_dir"))
    report = validate_ohlc(df, max_gap="6h")
    prov = latest_provenance(cfg.db_path)
    payload = {"validation": report.to_dict(), "provenance": prov}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"rows={report.n_rows}  span={report.start_ts} .. {report.end_ts}")
        print(f"ok={report.ok}  duplicates={report.n_duplicates}  gaps_flagged={report.n_gap_flags}")
        for p in report.problems:
            print(f"PROBLEM: {p}")
        if prov:
            print(f"provenance: {prov['source']} retrieved {prov['retrieved_at']} checksum={prov['checksum']}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
