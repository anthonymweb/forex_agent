"""Evaluation metrics shared by backtests, reports and the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxagent.simulation.backtest import BacktestResult


def exposure_pct(decisions: pd.DataFrame, position_col: str = "position_after") -> float:
    if decisions.empty:
        return 0.0
    exposed = (decisions[position_col] != "flat").sum()
    return float(exposed / len(decisions) * 100.0)


def annotate_exposure(result: BacktestResult) -> BacktestResult:
    if not result.decisions.empty:
        result.metrics["exposure_pct"] = exposure_pct(result.decisions)
    return result


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity - peak) / peak.replace(0, np.nan)


def strategy_comparison(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        m = dict(res.metrics)
        m["strategy"] = name
        rows.append(m)
    df = pd.DataFrame(rows)
    if not df.empty:
        cols = ["strategy"] + [c for c in df.columns if c != "strategy"]
        df = df[cols]
    return df


def interpret_metrics(m: dict) -> list[str]:
    """Plain-language caveats — shown in reports so numbers are not oversold."""
    notes = [
        "Historical simulation with assumed costs; not a forecast of future returns.",
        "EURUSD bid-side M1 data resampled to H1; ask side synthesised via fixed spread.",
    ]
    if m.get("n_trades", 0) < 30:
        notes.append(
            f"Very small sample ({m.get('n_trades', 0)} trades): statistics are unreliable."
        )
    if m.get("forced_final_close"):
        notes.append("Final open position was force-closed at the last close.")
    pf = m.get("profit_factor")
    if isinstance(pf, float) and np.isinf(pf):
        notes.append("Profit factor undefined (no losing trades) — inspect sample size.")
    if m.get("win_rate_pct", 0) > 0 and m.get("avg_win_usd", 0) < abs(m.get("avg_loss_usd", 0)):
        notes.append("Win rate is positive but average loss exceeds average win.")
    return notes
