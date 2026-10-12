"""fxagent Streamlit dashboard.

Reads ONLY real stored results (SQLite + processed CSV + report files).
Empty states are explicit — this dashboard never fabricates performance.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fxagent.config import Config  # noqa: E402
from fxagent.reporting.metrics import drawdown_series  # noqa: E402
from fxagent.storage.db import connect, migrate  # noqa: E402

st.set_page_config(
    page_title="fxagent — Forex Research Dashboard",
    page_icon="📈",
    layout="wide",
)

CFG = Config.load()
DB = CFG.db_path


@st.cache_data(ttl=15)
def load_query(sql: str) -> pd.DataFrame:
    if not DB.exists():
        return pd.DataFrame()
    conn = connect(DB)
    try:
        migrate(conn)
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


@st.cache_data(ttl=15)
def load_dataset() -> pd.DataFrame:
    try:
        from fxagent.data.store import load_dataset as _load

        return _load(CFG.path("project.paths.processed_dir"))
    except FileNotFoundError:
        return pd.DataFrame()


def empty_state(msg: str) -> None:
    st.info(msg)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("fxagent")
st.sidebar.caption("Hybrid AI forex research — simulated money only")
section = st.sidebar.radio(
    "Section",
    ["Overview", "Experiment", "Equity & P&L", "Trades", "Predictions", "Comparisons", "Risk & Health", "Seven-day test", "Reports"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"DB: `{DB.name}` exists={DB.exists()}\n\n"
    f"Refreshed: {datetime.now(UTC).strftime('%H:%M:%S UTC')}"
)
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()

# ---------------------------------------------------------------- overview
if section == "Overview":
    st.title("System overview")
    runs = load_query("SELECT * FROM runs ORDER BY started_at DESC LIMIT 10")
    if runs.empty:
        empty_state(
            "No runs recorded yet. Train models (`scripts/train_supervised.py`, "
            "`scripts/train_rl.py`) and start a simulation (`scripts/run_simulation.py`)."
        )
    else:
        latest = runs.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest run", latest["run_id"][:24])
        c2.metric("Status", latest["status"])
        c3.metric("Mode", latest["mode"])
        c4.metric("Strategy", latest["strategy"])
        st.subheader("Recent runs")
        st.dataframe(runs[["run_id", "strategy", "mode", "status", "started_at", "ended_at"]], use_container_width=True)

    st.markdown("---")
    st.subheader("Dataset")
    prov = load_query("SELECT * FROM datasets ORDER BY created_at DESC LIMIT 1")
    bars = load_dataset()
    c1, c2 = st.columns(2)
    with c1:
        if prov.empty:
            empty_state("No dataset provenance recorded. Run scripts/build_dataset.py.")
        else:
            p = prov.iloc[0]
            st.markdown(
                f"- **Source**: {p['source']}\n"
                f"- **Timeframe**: {p['timeframe']}\n"
                f"- **Retrieved**: {p['retrieved_at']}\n"
                f"- **Rows**: {p['n_rows']}\n"
                f"- **Span**: {p['start_ts']} → {p['end_ts']}\n"
                f"- **Checksum**: `{p['checksum']}`\n"
                f"- **Licence/notes**: {p['license']}"
            )
    with c2:
        if bars.empty:
            empty_state("No processed candles found (`data/processed/EURUSD_H1.csv`).")
        else:
            st.markdown(
                f"- **Local bars**: {len(bars):,}\n"
                f"- **Last bar**: {bars['timestamp'].max()}\n"
                f"- **First bar**: {bars['timestamp'].min()}"
            )
            fig_data = bars.tail(24 * 90)
            st.line_chart(fig_data.set_index("timestamp")["close"], height=220)

    st.subheader("Model versions on disk")
    models_dir = CFG.path("project.paths.models_dir")
    rows = []
    for meta in sorted(models_dir.rglob("meta.json")):
        try:
            payload = json.loads(meta.read_text())
            rows.append(
                {
                    "model": meta.parent.parent.name,
                    "version": meta.parent.name,
                    "trained_at": payload.get("trained_at", ""),
                    "type": payload.get("model_type", payload.get("algorithm", "")),
                }
            )
        except (OSError, json.JSONDecodeError):
            continue
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        empty_state("No trained models found in `models/`.")

# ---------------------------------------------------------------- experiment
elif section == "Experiment":
    st.title("Paper-trading experiment")
    st.caption("Simulated money only. Mode and data source are recorded per run — never guessed.")
    exps = load_query("SELECT * FROM experiments ORDER BY started_at DESC")
    if exps.empty:
        empty_state(
            "No experiment recorded yet. Start one with:\n\n"
            "`python scripts/experiment.py start --mode delayed_data_paper --strategy hybrid`"
        )
    else:
        exp = exps.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Experiment", str(exp["experiment_id"])[:28])
        c2.metric("Status", exp["status"])
        c3.metric("Mode", exp["mode"])
        c4.metric("Data source", exp["data_source"])
        st.markdown(
            f"- **Started**: {exp['started_at']}\n"
            f"- **Ends (planned)**: {exp.get('ends_at') or '—'}\n"
            f"- **Ended**: {exp.get('ended_at') or '—'}\n"
            f"- **Last observation**: {exp.get('last_observation_ts') or 'n/a'}"
        )
        if exp["mode"] == "historical_replay":
            st.warning("historical_replay: accelerated/paced replay of LOCAL data — NOT live-market observation.")
        elif exp["mode"] == "delayed_data_paper":
            st.info("delayed_data_paper: free delayed hourly quotes (~1-2 min), not a real-time feed.")

        run_id = exp["run_id"]
        hb = load_query(
            f"SELECT * FROM heartbeats WHERE run_id='{run_id}' ORDER BY ts DESC LIMIT 200"
        )
        if not hb.empty:
            latest = hb.iloc[0]
            try:
                age_s = (datetime.now(UTC) - pd.to_datetime(latest["ts"]).tz_convert("UTC")).total_seconds()
                st.subheader("Freshness")
                st.markdown(
                    f"- **Latest heartbeat**: {latest['ts']} ({age_s/60:.1f} min ago)\n"
                    f"- **Heartbeat status**: {latest['status']}\n"
                    f"- **Bars processed**: {int(latest['bars_processed']):,}"
                )
            except (TypeError, ValueError):
                pass

        eq = load_query(
            f"SELECT * FROM equity_snapshots WHERE run_id='{run_id}' ORDER BY ts"
        )
        if not eq.empty:
            eq["ts"] = pd.to_datetime(eq["ts"])
            start_eq = float(eq["equity"].iloc[0])
            last_eq = float(eq["equity"].iloc[-1])
            peak = eq["equity"].cummax()
            dd = ((peak - eq["equity"]) / peak * 100).max()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Starting equity", f"${start_eq:,.2f}")
            c2.metric("Current equity", f"${last_eq:,.2f}",
                      delta=f"{(last_eq-start_eq)/start_eq*100:.2f}%" if start_eq else None)
            c3.metric("Max drawdown", f"{dd:.2f}%")
            c4.metric("Snapshots", f"{len(eq):,}")
            st.subheader("Equity curve")
            st.line_chart(eq.set_index("ts")["equity"], height=240)
        else:
            empty_state("No equity snapshots yet for this experiment.")

        pos = load_query(
            f"""SELECT status, exit_reason, COUNT(*) AS n, AVG(realized_pnl) AS avg_pnl
                FROM positions WHERE run_id='{run_id}' GROUP BY status, exit_reason"""
        )
        st.subheader("Positions by exit reason")
        if pos.empty:
            empty_state("No positions recorded yet.")
        else:
            st.dataframe(pos, use_container_width=True, hide_index=True)

        dec = load_query(
            f"""SELECT action, risk_verdict, COUNT(*) AS n FROM decisions
                WHERE run_id='{run_id}' GROUP BY action, risk_verdict"""
        )
        st.subheader("Decisions by action / risk verdict")
        if dec.empty:
            empty_state("No decisions recorded yet.")
        else:
            st.dataframe(dec, use_container_width=True, hide_index=True)

        events = load_query(
            f"SELECT * FROM experiment_events WHERE experiment_id='{exp['experiment_id']}' ORDER BY ts DESC LIMIT 50"
        )
        st.subheader("Experiment events")
        if events.empty:
            st.caption("No lifecycle events recorded.")
        else:
            st.dataframe(events, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- equity
elif section == "Equity & P&L":
    st.title("Equity, P&L and drawdown")
    runs = load_query("SELECT run_id FROM runs ORDER BY started_at DESC")
    if runs.empty:
        empty_state("No runs recorded yet.")
    else:
        run_id = st.selectbox("Run", runs["run_id"].tolist())
        eq = load_query(
            f"SELECT * FROM equity_snapshots WHERE run_id='{run_id}' ORDER BY ts"
        )
        if eq.empty:
            empty_state("No equity snapshots for this run.")
        else:
            eq["ts"] = pd.to_datetime(eq["ts"])
            eq = eq.set_index("ts")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Final equity", f"${eq['equity'].iloc[-1]:,.2f}")
            c2.metric("Final balance", f"${eq['balance'].iloc[-1]:,.2f}")
            dd = drawdown_series(eq["equity"])
            c3.metric("Max drawdown", f"{dd.min()*100:.2f}%")
            c4.metric("Snapshots", f"{len(eq):,}")
            st.subheader("Equity curve")
            st.line_chart(eq["equity"], height=260)
            st.subheader("Drawdown")
            st.area_chart(dd * 100, height=180)
            daily = eq["equity"].resample("1D").last().dropna()
            if len(daily) > 1:
                st.subheader("Daily P&L")
                daily_pnl = daily.diff().dropna()
                st.bar_chart(daily_pnl, height=200)

# ---------------------------------------------------------------- trades
elif section == "Trades":
    st.title("Trade history")
    trades = load_query(
        """SELECT p.*, (p.realized_pnl) AS net_pnl FROM positions p
           WHERE p.status='closed' ORDER BY p.ts_closed DESC LIMIT 500"""
    )
    fills = load_query("SELECT * FROM fills ORDER BY ts DESC LIMIT 500")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Closed positions")
        if trades.empty:
            empty_state("No closed trades recorded.")
        else:
            st.dataframe(trades, use_container_width=True, hide_index=True)
            st.download_button("Export trades CSV", trades.to_csv(index=False), "trades.csv")
    with c2:
        st.subheader("Fills")
        if fills.empty:
            empty_state("No fills recorded.")
        else:
            st.dataframe(fills, use_container_width=True, hide_index=True)

    orders = load_query("SELECT * FROM orders ORDER BY ts_created DESC LIMIT 500")
    st.subheader("Orders (incl. rejections)")
    if orders.empty:
        empty_state("No orders recorded.")
    else:
        st.dataframe(orders, use_container_width=True, hide_index=True)
        if "status" in orders.columns:
            st.caption(f"Rejected orders: {(orders['status'] == 'rejected').sum()}")

# ---------------------------------------------------------------- predictions
elif section == "Predictions":
    st.title("Prediction history")
    preds = load_query("SELECT * FROM predictions ORDER BY ts DESC LIMIT 2000")
    if preds.empty:
        empty_state(
            "No predictions logged. They are written during backtests/runs "
            "and by `scripts/train_supervised.py`."
        )
    else:
        st.dataframe(preds.head(200), use_container_width=True, hide_index=True)
        preds["ts"] = pd.to_datetime(preds["ts"])
        by_model = preds.groupby("model_name")["prob_up"].describe()
        st.subheader("P(up) distribution stats")
        st.dataframe(by_model, use_container_width=True)
        st.download_button("Export predictions CSV", preds.to_csv(index=False), "predictions.csv")

# ---------------------------------------------------------------- comparisons
elif section == "Comparisons":
    st.title("Strategy comparison")
    reports_dir = CFG.path("project.paths.reports_dir")
    csvs = sorted(reports_dir.glob("*_comparison_*.csv"), reverse=True)
    if not csvs:
        empty_state(
            "No comparison reports yet. Run "
            "`python scripts/backtest.py --split all --strategies always_flat always_long sma_cross`."
        )
    else:
        choice = st.selectbox("Report", [p.name for p in csvs])
        comp = pd.read_csv(reports_dir / choice)
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.download_button("Export this comparison", comp.to_csv(index=False), choice)
        if "net_pnl_usd" in comp.columns:
            st.subheader("Net P&L by strategy")
            st.bar_chart(comp.set_index("strategy")["net_pnl_usd"])
        if "max_drawdown_pct" in comp.columns:
            st.subheader("Max drawdown (%) by strategy")
            st.bar_chart(comp.set_index("strategy")["max_drawdown_pct"])

# ---------------------------------------------------------------- risk & health
elif section == "Risk & Health":
    st.title("Risk controls and system health")
    events = load_query("SELECT * FROM risk_events ORDER BY ts DESC LIMIT 200")
    st.subheader("Risk-limit events")
    if events.empty:
        st.success("No risk-limit events recorded.")
    else:
        st.dataframe(events, use_container_width=True, hide_index=True)
        st.bar_chart(events.groupby("limit_name").size())

    checks = load_query("SELECT * FROM health_checks ORDER BY ts DESC LIMIT 50")
    st.subheader("Health checks")
    if checks.empty:
        empty_state("No health checks recorded (they run when a simulation starts).")
    else:
        st.dataframe(checks, use_container_width=True, hide_index=True)

    hb = load_query("SELECT * FROM heartbeats ORDER BY ts DESC LIMIT 20")
    st.subheader("Heartbeats (latest)")
    if hb.empty:
        empty_state("No heartbeats recorded.")
    else:
        st.dataframe(hb, use_container_width=True, hide_index=True)
        latest = hb.iloc[0]
        age = (datetime.now(UTC) - pd.to_datetime(latest["ts"]).tz_convert("UTC")).total_seconds()
        if age > 120 and latest["status"] == "running":
            st.warning(f"Latest heartbeat is {age/60:.1f} minutes old while status=running.")
        else:
            st.caption(f"Latest heartbeat age: {age:.0f}s")

    orders = load_query("SELECT status, COUNT(*) AS n FROM orders GROUP BY status")
    st.subheader("Order outcomes")
    if orders.empty:
        empty_state("No orders recorded.")
    else:
        st.dataframe(orders, use_container_width=True, hide_index=True)

    stop_file = CFG.path(CFG.get("risk.emergency_stop_file", "data/EMERGENCY_STOP"))
    st.subheader("Emergency stop")
    if stop_file.exists():
        st.error(f"EMERGENCY STOP ENGAGED — {stop_file.read_text().strip()}")
    else:
        st.success("Emergency stop is clear.")

# ---------------------------------------------------------------- 7-day
elif section == "Seven-day test":
    st.title("Seven-day continuous test")
    runs = load_query("SELECT * FROM runs ORDER BY started_at DESC LIMIT 5")
    if runs.empty:
        empty_state(
            "No runs recorded. Start the seven-day experiment with:\n\n"
            "`python scripts/experiment.py start --mode delayed_data_paper --strategy hybrid --duration-days 7`"
        )
    else:
        run = runs.iloc[0]
        st.markdown(
            f"- **Run**: `{run['run_id']}`\n"
            f"- **Status**: {run['status']}\n"
            f"- **Mode**: {run['mode']}\n"
            f"- **Started**: {run['started_at']}\n"
            f"- **Ended**: {run['ended_at'] or '—'}"
        )
        hb = load_query(
            f"SELECT * FROM heartbeats WHERE run_id='{run['run_id']}' ORDER BY ts DESC LIMIT 1000"
        )
        if not hb.empty:
            hb["ts"] = pd.to_datetime(hb["ts"])
            st.subheader("Progress")
            st.line_chart(hb.sort_values("ts").set_index("ts")["bars_processed"], height=200)
            st.caption(f"Bars processed: {hb['bars_processed'].max():,}")
        eq = load_query(
            f"SELECT * FROM equity_snapshots WHERE run_id='{run['run_id']}' ORDER BY ts"
        )
        if not eq.empty:
            eq["ts"] = pd.to_datetime(eq["ts"])
            st.subheader("Equity during run")
            st.line_chart(eq.set_index("ts")["equity"], height=220)

        st.markdown(
            "**Reminder**: a seven-day run demonstrates operational reliability "
            "(restarts, persistence, risk gating) — not profitability."
        )

# ---------------------------------------------------------------- reports
elif section == "Reports":
    st.title("Reports and exports")
    reports_dir = CFG.path("project.paths.reports_dir")
    files = sorted(reports_dir.glob("*"), reverse=True)
    files = [f for f in files if f.is_file()]
    if not files:
        empty_state("No reports generated yet. Run `python scripts/generate_report.py`.")
    else:
        for f in files[:30]:
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{f.name}** — {f.stat().st_size:,} bytes")
            if f.suffix in {".md", ".json", ".csv"}:
                c2.download_button("Download", f.read_bytes(), file_name=f.name, key=f.name)
            if f.suffix == ".md" and c1.checkbox(f"Preview {f.name}", key=f"p{f.name}"):
                st.markdown(f.read_text(encoding="utf-8")[:8000])

st.sidebar.markdown("---")
st.sidebar.caption(
    "Historical simulation with assumed costs. Not investment advice. "
    "No live trading endpoint exists in this system."
)
