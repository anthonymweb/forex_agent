#!/usr/bin/env python3
"""Train the supervised direction model on the training split only.

Evaluates on validation and (optionally) the final test split, logs metrics,
persists the model and writes prediction logs to SQLite.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxagent.config import Config  # noqa: E402
from fxagent.models.supervised.core import (  # noqa: E402
    NaiveBaselines,
    SupervisedDirectionModel,
    binary_direction_metrics,
)
from fxagent.pipeline import prepare_dataset  # noqa: E402
from fxagent.storage.db import Repository, session  # noqa: E402

log = logging.getLogger("train_supervised")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--eval-test", action="store_true", help="also report final test metrics")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    seed = int(cfg.get("project.seed", 42))

    data = prepare_dataset(cfg)
    train_bars, X_train, y_train = data.clean_split("train")
    val_bars, X_val, y_val = data.clean_split("val")
    log.info("train rows=%d  val rows=%d", len(X_train), len(X_val))

    model_type = cfg.get("supervised.model_type", "gradient_boosting")
    params = cfg.get(f"supervised.{model_type}", {}) or {}
    model = SupervisedDirectionModel(model_type=model_type, params=params, seed=seed)
    meta = model.fit(X_train, y_train["label"], version=args.version)
    log.info("trained %s (%d rows)", model_type, meta.n_train_rows)

    val_proba = model.predict_proba(X_val)
    val_metrics = binary_direction_metrics(y_val["label"].to_numpy(dtype=int), val_proba)
    log.info("validation metrics: %s", json.dumps(val_metrics, indent=2))

    # Honest baselines on the same validation rows.
    naive = NaiveBaselines.majority_class(y_train["label"])
    majority_acc = float((naive.predict(X_val) == y_val["label"].to_numpy(dtype=int)).mean())
    log.info("baseline majority-class accuracy: %.4f", majority_acc)

    test_metrics = None
    if args.eval_test:
        test_bars, X_test, y_test = data.clean_split("test")
        test_proba = model.predict_proba(X_test)
        test_metrics = binary_direction_metrics(y_test["label"].to_numpy(dtype=int), test_proba)
        log.info("TEST metrics (final, reported once): %s", json.dumps(test_metrics, indent=2))

    out_dir = model.save(cfg.path("project.paths.models_dir"), name="supervised_direction")
    log.info("saved model to %s", out_dir)

    # prediction log for validation rows
    with session(cfg.db_path) as conn:
        repo = Repository(conn)
        pred_df = pd.DataFrame(
            {
                "ts": val_bars["timestamp"].astype(str),
                "prob_up": val_proba,
                "pred_class": (val_proba >= 0.5).astype(int),
            }
        )
        for row in pred_df.itertuples(index=False):
            repo.insert_prediction(
                run_id=None,
                ts=row.ts,
                model_name="supervised_direction",
                model_version=args.version,
                horizon=int(cfg.get("labels.horizon_bars", 4)),
                prob_up=float(row.prob_up),
                pred_class=int(row.pred_class),
                details={"split": "val"},
            )
        metrics_payload = {
            "val": val_metrics,
            "baseline_majority_accuracy": majority_acc,
            "train_rows": meta.n_train_rows,
            "val_rows": len(X_val),
        }
        if test_metrics:
            metrics_payload["test"] = test_metrics
        (out_dir / "eval_metrics.json").write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )

    print(json.dumps(metrics_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
