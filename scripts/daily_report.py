#!/usr/bin/env python3
"""Generate DAILY and FINAL experiment reports from stored SQLite results.

Only real persisted data is used — nothing is synthesised. If a metric
cannot be computed from stored records it is omitted or reported as n/a.

Outputs (under reports/):
  daily_report_<experiment>_<YYYYMMDD>.{md,json,csv}
  final_report_<experiment>.{md,json,csv}

Usage:
  python scripts/daily_report.py                       # latest experiment, today
  python scripts/daily_report.py --experiment-id exp-x --day 2026-10-13
  python scripts/daily_report.py --final               # final evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxagent.config import Config  # noqa: E402
from fxagent.storage.db import Repository, session  # noqa: E402

SAMPLE_SIZE_WARNING = 20  # trades below this are flagged as underpowered


def _fetch_run_data(conn, run_id: str) -> dict:
    decisions = [dict(r) for r in conn.execute(
        "SELECT * FROM decisions WHERE run_id=? ORDER BY ts", (run_id,))]
    equity = [dict(r) for r in conn.execute(
        "SELECT * FROM equity_snapshots WHERE run_id=? ORDER BY ts", (run_id,))]
    fills = [dict(r) for r in conn.execute(
        "SELECT * FROM fills WHERE run_id=? ORDER BY ts", (run_id,))]
    positions = [dict(r) for r in conn.execute(
        "SELECT * FROM positions WHERE run_id=? ORDER BY ts_opened", (run_id,))]
    risk_events = [dict(r) for r in conn.execute(
        "SELECT * FROM risk_events WHERE run_id=? ORDER BY ts", (run_id,))]
    heartbeats = [dict(r) for r in conn.execute(
        "SELECT * FROM heartbeats WHERE run_id=? ORDER BY ts DESC LIMIT 5", (run_id,))]
    run = dict(conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone() or {})
    return {
        "run": run, "decisions": decisions, "equity": equity, "fills": fills,
        "positions": positions, "risk_events": risk_events, "heartbeats": heartbeats,
    }


def _closed_positions_metrics(positions: list[dict]) -> dict:
    closed = [p for p in positions if p.get("status") == "closed" and p.get("realized_pnl") is not None]
    n = len(closed)
    if n == 0:
        return {"n_trades": 0}
    pnls = [float(p["realized_pnl"]) for p in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses) if losses else 0.0
    return {
        "n_trades": n,
        "win_rate_pct": 100.0 * len(wins) / n,
        "avg_win_usd": gross_win / len(wins) if wins else 0.0,
        "avg_loss_usd": -gross_loss / len(losses) if losses else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else math.inf,
        "net_pnl_usd": sum(pnls),
        "best_trade_usd": max(pnls),
        "worst_trade_usd": min(pnls),
        "avg_bars_held": sum(int(p.get("bars_held") or 0) for p in closed) / n,
        "exit_reasons": _count_by(closed, "exit_reason"),
    }


def _count_by(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key) or "unset")
        out[k] = out.get(k, 0) + 1
    return out


def _equity_metrics(equity: list[dict]) -> dict:
    if not equity:
        return {}
    eq = [float(r["equity"]) for r in equity]
    peak = eq[0]
    max_dd = 0.0
    for v in eq:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak * 100.0)
    start, end = eq[0], eq[-1]
    return {
        "starting_equity": start,
        "final_equity": end,
        "net_return_pct": (end - start) / start * 100.0 if start else 0.0,
        "max_drawdown_pct": max_dd,
        "peak_equity": max(eq),
        "trough_equity": min(eq),
    }


def _costs(fills: list[dict]) -> dict:
    commission = sum(float(f.get("commission") or 0.0) for f in fills)
    slippage = sum(float(f.get("slippage") or 0.0) * int(f.get("units") or 0) for f in fills)
    return {"commission_usd": commission, "slippage_usd": slippage,
            "n_fills": len(fills)}


def _daily_slice(data: dict, day: str) -> dict:
    """Filter records to one UTC calendar day."""
    def keep(ts: str | None) -> bool:
        return bool(ts) and str(ts).startswith(day)
    return {
        "decisions": [d for d in data["decisions"] if keep(d["ts"])],
        "equity": [e for e in data["equity"] if keep(e["ts"])],
        "fills": [f for f in data["fills"] if keep(f["ts"])],
        "positions": [p for p in data["positions"]
                      if keep(p.get("ts_opened")) or keep(p.get("ts_closed"))],
        "risk_events": [r for r in data["risk_events"] if keep(r["ts"])],
    }


def _build_report(conn, repo: Repository, experiment_id: str | None, day: str | None,
                  final: bool, cfg: Config) -> dict:
    exp = repo.get_experiment(experiment_id) if experiment_id else repo.latest_experiment()
    if exp is None:
        raise SystemExit("no experiment found in database — run scripts/experiment.py first")
    run_id = exp["run_id"]
    data = _fetch_run_data(conn, run_id)

    scope = data
    label = "full"
    if day and not final:
        scope = _daily_slice(data, day)
        label = day

    metrics = {}
    metrics.update(_closed_positions_metrics(scope["positions"]))
    metrics.update(_equity_metrics(scope["equity"]))
    metrics.update(_costs(scope["fills"]))
    metrics["n_decisions"] = len(scope["decisions"])
    metrics["actions"] = _count_by(scope["decisions"], "action")
    metrics["risk_rejections"] = sum(
        1 for d in scope["decisions"] if d.get("risk_verdict") == "reject")
    metrics["risk_rejection_reasons"] = {}
    for d in scope["decisions"]:
        if d.get("risk_verdict") == "reject":
            for reason in json.loads(d.get("risk_reasons") or "[]"):
                metrics["risk_rejection_reasons"][reason] = \
                    metrics["risk_rejection_reasons"].get(reason, 0) + 1
    metrics["stale_or_error_events"] = sum(
        1 for r in scope["risk_events"] if r.get("action_taken") in
        ("reject_order", "safe_stop"))
    metrics["sample_size_warning"] = (
        metrics.get("n_trades", 0) < SAMPLE_SIZE_WARNING
    ) if final or day else False

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "report_type": "final" if final else "daily",
        "scope": label,
        "experiment": {k: exp[k] for k in exp if k != "details"},
        "experiment_details": json.loads(exp.get("details") or "{}"),
        "mode": exp["mode"],
        "data_source": exp["data_source"],
        "last_observation_ts": exp.get("last_observation_ts"),
        "run_status": data["run"].get("status"),
        "metrics": metrics,
        "recent_heartbeats": [
            {"ts": h["ts"], "status": h["status"], "last_bar_ts": h["last_bar_ts"],
             "bars_processed": h["bars_processed"]}
            for h in data["heartbeats"][:5]
        ],
        "equity_curve": [
            {"ts": e["ts"], "equity": e["equity"], "balance": e["balance"]}
            for e in (scope["equity"] if len(scope["equity"]) <= 5000
                      else scope["equity"][:: max(1, len(scope["equity"]) // 5000)])
        ],
        "limitations": _limitations(exp, metrics, final),
    }
    return payload


def _limitations(exp: dict, metrics: dict, final: bool) -> list[str]:
    out = [
        "SIMULATED MONEY ONLY — no real orders were placed.",
    ]
    if exp["mode"] == "historical_replay":
        out.append("Mode is historical_replay: results are NOT live-market observation.")
    if exp["mode"] == "delayed_data_paper":
        out.append("Mode is delayed_data_paper: free delayed quotes (~1-2 min), not a real-time feed.")
        out.append("Price basis is approximately mid; fills still pay the configured spread assumption.")
    if metrics.get("sample_size_warning"):
        out.append(
            f"WARNING: only {metrics.get('n_trades', 0)} closed trades — "
            "sample too small for statistical conclusions."
        )
    if exp.get("status") in ("stopped", "safe_stop_data_errors"):
        out.append(f"Experiment ended early with status={exp['status']}.")
    if final:
        days = None
        try:
            start = datetime.fromisoformat(exp["started_at"])
            end = datetime.fromisoformat(exp["ended_at"]) if exp.get("ended_at") else datetime.now(UTC)
            days = (end - start).total_seconds() / 86400.0
        except (ValueError, TypeError):
            pass
        if days is not None and days < 7.0 and exp["mode"] != "historical_replay":
            out.append(
                f"WARNING: only {days:.2f} of 7 calendar days elapsed — "
                "this experiment is NOT complete."
            )
    return out


def _write(payload: dict, reports_dir: Path, stem: str) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    csv_path = reports_dir / f"{stem}.csv"
    metrics = payload["metrics"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, json.dumps(v) if isinstance(v, (dict, list)) else v])

    m = payload["metrics"]
    md = [
        f"# {'Final' if payload['report_type']=='final' else 'Daily'} report — {payload['scope']}",
        "",
        f"- **Experiment**: `{payload['experiment']['experiment_id']}`",
        f"- **Mode**: `{payload['mode']}` (source: {payload['data_source']})",
        f"- **Status**: {payload['run_status']}",
        f"- **Last observation**: {payload.get('last_observation_ts') or 'n/a'}",
        f"- **Generated**: {payload['generated_at']}",
        "",
        "> SIMULATED MONEY ONLY. Not investment advice. Past performance does not",
        "> predict future results. Historical replay is not live-market observation.",
        "",
        "## Performance (stored records only)",
        "",
        f"- Net return: {m.get('net_return_pct', float('nan')):.2f}%" if "net_return_pct" in m else "- Net return: n/a (no equity snapshots)",
        f"- Final equity: ${m.get('final_equity', float('nan')):,.2f}" if "final_equity" in m else "- Final equity: n/a",
        f"- Max drawdown: {m.get('max_drawdown_pct', float('nan')):.2f}%" if "max_drawdown_pct" in m else "- Max drawdown: n/a",
        f"- Closed trades: {m.get('n_trades', 0)}",
        f"- Win rate: {m.get('win_rate_pct', float('nan')):.1f}%" if "win_rate_pct" in m else "- Win rate: n/a",
        f"- Profit factor: {m.get('profit_factor', float('nan')):.2f}" if isinstance(m.get("profit_factor"), float) and math.isfinite(m.get("profit_factor", math.nan)) else "- Profit factor: n/a or inf",
        f"- Commission paid: ${m.get('commission_usd', 0.0):,.2f}",
        f"- Decisions: {m.get('n_decisions', 0)} (risk rejections: {m.get('risk_rejections', 0)})",
        f"- Actions: {json.dumps(m.get('actions', {}))}",
        f"- Exit reasons: {json.dumps(m.get('exit_reasons', {}))}",
        "",
    ]
    if m.get("sample_size_warning"):
        md += ["⚠️ **Sample size too small for conclusions.**", ""]
    md.append("## Limitations")
    for lim in payload["limitations"]:
        md.append(f"- {lim}")
    md.append("")
    md.append("## Recent heartbeats")
    for h in payload["recent_heartbeats"]:
        md.append(f"- `{h['ts']}` status={h['status']} bars={h['bars_processed']} last_bar={h['last_bar_ts']}")
    md_path = reports_dir / f"{stem}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--day", default=None, help="UTC day YYYY-MM-DD for daily report")
    parser.add_argument("--final", action="store_true", help="full final evaluation")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    reports_dir = Path(cfg.path("project.paths.reports_dir"))

    with session(cfg.db_path) as conn:
        repo = Repository(conn)
        payload = _build_report(conn, repo, args.experiment_id, args.day, args.final, cfg)

    exp_id = payload["experiment"]["experiment_id"]
    if args.final:
        stem = f"final_report_{exp_id}"
    else:
        day = args.day or datetime.now(UTC).strftime("%Y-%m-%d")
        stem = f"daily_report_{exp_id}_{day.replace('-', '')}"
    paths = _write(payload, reports_dir, stem)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
