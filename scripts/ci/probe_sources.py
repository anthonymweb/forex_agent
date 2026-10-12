#!/usr/bin/env python3
"""Probe free market-data sources and print a Markdown summary.

Used by the data-health workflow and runnable locally:
  python scripts/ci/probe_sources.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fxagent.data.live_source import YahooHourlySource  # noqa: E402


def main() -> int:
    print("## Market-data source health probe")
    print()
    results: list[tuple[str, bool, str]] = []

    src = YahooHourlySource()
    health = src.health()
    ok = bool(health.get("ok"))
    results.append(("yahoo_hourly (delayed FX, free)", ok,
                    health.get("error") or f"last={health.get('last_observation_ts')}"))
    print(f"- **yahoo_hourly**: {'OK' if ok else 'FAIL'} — "
          f"{health.get('error') or health.get('last_observation_ts')}")
    print(f"  - mode: {health.get('mode')}, basis: {health.get('price_basis')}")

    try:
        import urllib.request
        url = "https://api.frankfurter.dev/v1/latest?base=EUR&symbols=USD"
        req = urllib.request.Request(url, headers={"User-Agent": "fxagent-ci"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read(300).decode()
        results.append(("frankfurter (ECB daily, free)", True, body[:120]))
        print(f"- **frankfurter**: OK — {body[:120]}")
    except Exception as exc:
        results.append(("frankfurter (ECB daily, free)", False, str(exc)))
        print(f"- **frankfurter**: FAIL — {exc}")

    healthy = sum(1 for _, ok, _ in results if ok)
    print()
    if healthy == 0:
        print("**OVERALL: FAIL** — no source reachable")
        return 1
    if healthy < len(results):
        print(f"**OVERALL: WARN** — {healthy}/{len(results)} sources healthy")
    else:
        print(f"**OVERALL: OK** — {healthy}/{len(results)} sources healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
