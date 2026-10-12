#!/usr/bin/env python3
"""Build the processed H1 dataset from downloaded HistData M1 archives.

Steps: load zips -> validate -> resample to H1 -> re-validate -> save CSV
       -> record provenance in SQLite.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxagent.config import Config  # noqa: E402
from fxagent.data.candles import (  # noqa: E402
    clean_ohlc,
    resample_ohlc,
    tick_spread_stats,
    ticks_to_ohlc,
    validate_ohlc,
)
from fxagent.data.histdata_client import zip_to_dataframe, zip_to_ticks  # noqa: E402
from fxagent.data.store import dataset_checksum, record_provenance, save_dataset  # noqa: E402

log = logging.getLogger("build_dataset")


def load_all_h1(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Build H1 bars from M1 archives and (fallback) ASCII tick archives.

    M1 months are bid OHLC bars. Tick months are aggregated from bid ticks so
    the price basis is consistent across the dataset.
    """
    m1_zips = sorted(p for p in raw_dir.rglob("*.zip") if "/M1/" in str(p))
    tick_zips = sorted(p for p in raw_dir.rglob("*.zip") if "/T/" in str(p))
    if not m1_zips and not tick_zips:
        raise FileNotFoundError(
            f"no archives under {raw_dir} — run scripts/download_data.py first"
        )
    frames = []
    stats = {"m1_months": 0, "tick_months": 0, "m1_rows": 0, "tick_spreads": []}
    for zp in m1_zips:
        try:
            m1 = clean_ohlc(zip_to_dataframe(zp))
            frames.append(resample_ohlc(m1, rule="1h"))
            stats["m1_months"] += 1
            stats["m1_rows"] += len(m1)
        except Exception as exc:
            log.warning("failed to read M1 %s: %s", zp, exc)
    for zp in tick_zips:
        try:
            ticks = zip_to_ticks(zp)
            stats["tick_months"] += 1
            spread = tick_spread_stats(ticks)
            if spread:
                stats["tick_spreads"].append(spread["median_spread"])
            frames.append(ticks_to_ohlc(ticks, rule="1h", price_col="bid"))
        except Exception as exc:
            log.warning("failed to read ticks %s: %s", zp, exc)
    if not frames:
        raise RuntimeError("no CSVs could be read from archives")
    h1 = pd.concat(frames, ignore_index=True)
    h1 = clean_ohlc(h1)
    return h1, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--strict", action="store_true", help="fail on any validation problem")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    raw_dir = cfg.path("project.paths.raw_dir")
    processed_dir = cfg.path("project.paths.processed_dir")

    gap = cfg.get("data.max_gap_hours", 6)
    gap_str = gap if isinstance(gap, str) else f"{gap}h"

    log.info("aggregating archives under %s", raw_dir)
    h1, stats = load_all_h1(raw_dir)
    log.info(
        "combined H1 rows=%d from m1_months=%d (%d M1 rows) + tick_months=%d",
        len(h1), stats["m1_months"], stats["m1_rows"], stats["tick_months"],
    )
    if stats["tick_spreads"]:
        med = sorted(stats["tick_spreads"])[len(stats["tick_spreads"]) // 2]
        log.info("median observed tick spread across sample months: %.5f (%.1f pips)",
                 med, med / cfg.get("instrument.pip_size", 0.0001))

    report = validate_ohlc(h1, max_gap=gap_str)
    log.info(
        "H1 validation: ok=%s rows=%d gaps_flagged=%d problems=%s",
        report.ok, report.n_rows, report.n_gap_flags, report.problems,
    )
    if not report.ok and args.strict:
        raise SystemExit("H1 validation failed in strict mode")

    path = save_dataset(h1, processed_dir)
    record_provenance(
        cfg.db_path,
        symbol=cfg.get("instrument.symbol", "EURUSD"),
        timeframe="H1",
        source="HistData.com ASCII M1 (bid) + ASCII tick (bid) fallback",
        source_url="https://www.histdata.com/download-free-forex-historical-data/",
        license_note=(
            "Free for personal use; timestamps treated as UTC; bid-side prices. "
            "Ask side approximated via configured spread in trading sims."
        ),
        n_rows=len(h1),
        start_ts=str(h1["timestamp"].min()),
        end_ts=str(h1["timestamp"].max()),
        checksum=dataset_checksum(h1),
        notes=(
            f"m1_months={stats['m1_months']} tick_months={stats['tick_months']} "
            f"gaps_flagged={report.n_gap_flags}"
        ),
    )
    print(f"Built {path} with {len(h1)} H1 bars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
