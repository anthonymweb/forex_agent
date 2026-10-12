#!/usr/bin/env python3
"""Backtest one or more strategies on validation and/or test splits.

Usage:
    python scripts/backtest.py --strategies always_flat always_long sma_cross
    python scripts/backtest.py --strategies supervised rl hybrid --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxagent.config import Config, periods_per_year_from_config  # noqa: E402
from fxagent.data.store import latest_provenance  # noqa: E402
from fxagent.pipeline import prepare_dataset  # noqa: E402
from fxagent.reporting.metrics import annotate_exposure  # noqa: E402
from fxagent.reporting.report import write_report  # noqa: E402
from fxagent.simulation.accounting import CostModel  # noqa: E402
from fxagent.simulation.backtest import run_backtest  # noqa: E402
from fxagent.strategy.factory import build_strategy  # noqa: E402
from fxagent.strategy.strategies import features_with_close  # noqa: E402

log = logging.getLogger("backtest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["always_flat", "always_long", "sma_cross"],
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--supervised-version", default="v1")
    parser.add_argument("--rl-version", default="v1")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    data = prepare_dataset(cfg)
    cost = CostModel(
        spread_pips=float(cfg.get("costs.spread_pips", 0.8)),
        slippage_pips=float(cfg.get("costs.slippage_pips", 0.2)),
        commission_per_mm=float(cfg.get("costs.commission_per_million", 3.0)),
        pip_size=float(cfg.get("instrument.pip_size", 0.0001)),
    )
    units = int(cfg.get("position.default_units", 100_000))
    balance = float(cfg.get("paper_broker.initial_balance_usd", 10_000.0))
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    results: dict = {}
    for split_name in splits:
        bars, feats, _ = data.clean_split(split_name)
        if len(bars) < 10:
            log.warning("split %s too small (%d rows); skipping", split_name, len(bars))
            continue
        feats = features_with_close(feats, bars)
        for strat_name in args.strategies:
            try:
                strategy = build_strategy(
                    cfg, strat_name,
                    supervised_version=args.supervised_version,
                    rl_version=args.rl_version,
                )
            except (FileNotFoundError, RuntimeError) as exc:
                log.warning("skip strategy %s: %s", strat_name, exc)
                continue
            res = run_backtest(bars, feats, strategy, cost, units=units, initial_balance=balance)
            annotate_exposure(res)
            key = f"{split_name}:{strat_name}"
            results[key] = res
            print(res.summary(), f"[{split_name}]")

    if not results:
        print("No results produced.")
        return 1

    prov = latest_provenance(cfg.db_path)
    dataset_info = {
        "source": prov["source"] if prov else "unknown",
        "retrieved_at": prov["retrieved_at"] if prov else "unknown",
        "checksum": prov["checksum"] if prov else "unknown",
        "rows": prov["n_rows"] if prov else "unknown",
        "costs": (
            f"spread={cfg.get('costs.spread_pips')}pips "
            f"slippage={cfg.get('costs.slippage_pips')}pips "
            f"commission={cfg.get('costs.commission_per_million')}$/MM round-turn"
        ),
        "splits": json.dumps(cfg.get("split")),
    }
    paths = write_report(
        results,
        cfg.path("project.paths.reports_dir"),
        dataset_info=dataset_info,
        extra={"periods_per_year": periods_per_year_from_config(cfg)},
        name="backtest",
    )
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
