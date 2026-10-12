#!/usr/bin/env bash
# Relaunch the tick downloader until 2018-01..2024-12 are complete.
set -u
cd "$(dirname "$0")/.."
for round in $(seq 1 12); do
  missing=$(.venv/bin/python - <<'PY'
from pathlib import Path
root = Path("data/raw/histdata/EURUSD/T")
n = 0
for y in range(2018, 2025):
    for m in range(1, 13):
        if y == 2024 and m > 12:
            break
        d = root / f"{y:04d}" / f"{m:02d}"
        if not list(d.glob("*.zip")) if d.exists() else True:
            n += 1
print(n)
PY
)
  echo "[loop round $round] missing months: $missing"
  if [ "$missing" -eq 0 ]; then
    echo "[loop] all months present"
    exit 0
  fi
  .venv/bin/python scripts/download_data.py --start 2018-01 --end 2024-12 --rounds 1 --round-delay 2 \
    >> logs/download_tick.log 2>&1 || true
  sleep 5
done
echo "[loop] gave up after 12 rounds"
exit 1
