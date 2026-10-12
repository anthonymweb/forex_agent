"""Structured logging setup + health checks."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path


def setup_logging(logs_dir: str | Path, name: str = "fxagent", level: int = logging.INFO) -> Path:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}_{datetime.now(UTC).strftime('%Y%m%d')}.log"

    root = logging.getLogger()
    root.setLevel(level)
    # avoid duplicate handlers when re-invoked (e.g. tests)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    return log_path


class JsonLogFilterer:
    """Tiny helper to emit structured JSON events into the std logging stream."""

    def __init__(self, logger_name: str = "fxagent.events"):
        self.log = logging.getLogger(logger_name)

    def emit(self, event: str, **fields) -> None:
        payload = {"event": event, "ts": datetime.now(UTC).isoformat(timespec="seconds"), **fields}
        self.log.info(json.dumps(payload, sort_keys=True, default=str))
