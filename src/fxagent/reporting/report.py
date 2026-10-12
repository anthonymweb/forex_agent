"""Evaluation report generation (Markdown + JSON + CSV)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fxagent.reporting.metrics import interpret_metrics, strategy_comparison
from fxagent.simulation.backtest import BacktestResult


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != v:  # NaN
            return "n/a"
        if abs(v) == float("inf"):
            return "inf"
        return f"{v:,.2f}"
    return str(v)


def write_report(
    results: dict[str, BacktestResult],
    reports_dir: str | Path,
    *,
    dataset_info: dict | None = None,
    extra: dict | None = None,
    name: str = "evaluation",
) -> dict[str, Path]:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    comparison = strategy_comparison(results)
    csv_path = reports_dir / f"{name}_comparison_{stamp}.csv"
    comparison.to_csv(csv_path, index=False)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": dataset_info or {},
        "results": {k: v.metrics for k, v in results.items()},
        "notes": {k: interpret_metrics(v.metrics) for k, v in results.items()},
        "extra": extra or {},
    }
    json_path = reports_dir / f"{name}_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md = [f"# Evaluation report — {stamp}", ""]
    if dataset_info:
        md.append("## Dataset")
        for k, v in dataset_info.items():
            md.append(f"- **{k}**: {v}")
        md.append("")
    md.append("## Strategy comparison")
    md.append("")
    if comparison.empty:
        md.append("_No results._")
    else:
        cols = [
            "strategy", "n_trades", "net_pnl_usd", "total_return_pct",
            "max_drawdown_pct", "win_rate_pct", "profit_factor",
            "estimated_costs_usd", "avg_bars_held", "exposure_pct", "n_bars",
        ]
        cols = [c for c in cols if c in comparison.columns]
        md.append("| " + " | ".join(cols) + " |")
        md.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, row in comparison.iterrows():
            md.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    md.append("")
    for name_, res in results.items():
        md.append(f"## {name_}")
        md.append("")
        for k, v in res.metrics.items():
            md.append(f"- {k}: {_fmt(v)}")
        md.append("")
        md.append("**Interpretation caveats**")
        for note in interpret_metrics(res.metrics):
            md.append(f"- {note}")
        md.append("")
    md_path = reports_dir / f"{name}_{stamp}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    return {"csv": csv_path, "json": json_path, "markdown": md_path}
