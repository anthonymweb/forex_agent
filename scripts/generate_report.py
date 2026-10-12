#!/usr/bin/env python3
"""Generate an evaluation report from stored backtest results (or re-run them).

If no fresh results exist, this re-executes the standard comparison set.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxagent.config import Config  # noqa: E402
from fxagent.data.store import latest_provenance  # noqa: E402
from fxagent.pipeline import prepare_dataset  # noqa: E402
from fxagent.reporting.metrics import annotate_exposure  # noqa: E402
from fxagent.reporting.report import write_report  # noqa: E402
from fxagent.simulation.accounting import CostModel  # noqa: E402
from fxagent.simulation.backtest import run_backtest  # noqa: E402
from fxagent.strategy.factory import build_strategy  # noqa: E402
from fxagent.strategy.strategies import features_with_close  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--strategies", nargs="+",
                        default=["always_flat", "always_long", "sma_cross", "mean_reversion"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = Config.load(args.config)
    data = prepare_dataset(cfg)
    cost = CostModel(
        spread_pips=float(cfg.get("costs.spread_pips", 0.8)),
        slippage_pips=float(cfg.get("costs.slippage_pips", 0.2)),
        commission_per_mm=float(cfg.get("costs.commission_per_million", 3.0)),
        pip_size=float(cfg.get("instrument.pip_size", 0.0001)),
    )
    units = int(cfg.get("position.default_units", 100_000))
    balance = float(cfg.get("paper_broker.initial_balance_usd", 10_000.0))

    bars, feats, _ = data.clean_split(args.split)
    feats = features_with_close(feats, bars)
    results = {}
    for name in args.strategies:
        try:
            strategy = build_strategy(cfg, name)
        except (FileNotFoundError, RuntimeError) as exc:
            logging.warning("skip %s: %s", name, exc)
            continue
        res = run_backtest(bars, feats, strategy, cost, units=units, initial_balance=balance)
        annotate_exposure(res)
        results[f"{args.split}:{name}"] = res

    if not results:
        raise SystemExit("no strategies could be evaluated")

    prov = latest_provenance(cfg.db_path)
    paths = write_report(
        results,
        cfg.path("project.paths.reports_dir"),
        dataset_info={
            "source": prov["source"] if prov else "unknown",
            "checksum": prov["checksum"] if prov else "unknown",
            "split": args.split,
            "bars": len(bars),
            "span": f"{bars['timestamp'].min()} .. {bars['timestamp'].max()}",
        },
        name="final_evaluation",
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
