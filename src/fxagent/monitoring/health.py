"""Health checks for the monitoring layer."""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path


def disk_free_gb(path: str | Path) -> float:
    usage = shutil.disk_usage(Path(path))
    return usage.free / (1024**3)


def check_disk(path: str | Path, min_free_gb: float = 2.0) -> dict:
    free = disk_free_gb(path)
    return {
        "check": "disk_space",
        "ok": free >= min_free_gb,
        "free_gb": round(free, 2),
        "min_free_gb": min_free_gb,
    }


def check_db_writable(db_path: str | Path) -> dict:
    path = Path(db_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.with_suffix(path.suffix + ".probe")
        probe.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
        probe.unlink(missing_ok=True)
        ok = True
        detail = ""
    except OSError as exc:
        ok = False
        detail = str(exc)
    return {"check": "db_writable", "ok": ok, "detail": detail}


def check_pid_alive(pid: int) -> dict:
    try:
        os.kill(pid, 0)
        alive = True
    except (OSError, ValueError):
        alive = False
    return {"check": "process_alive", "ok": alive, "pid": pid}


def run_all_checks(db_path: str | Path, data_dir: str | Path) -> list[dict]:
    return [check_db_writable(db_path), check_disk(data_dir)]
