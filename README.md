# fxagent — Hybrid AI Forex Research Agent (Simulated Money)

Research-only system for studying whether a **hybrid of supervised ML and
reinforcement learning** adds value over simple trading baselines on intraday
EUR/USD. Everything runs in **paper/simulation mode with fake money** — there is
no live-broker integration, and code actively refuses live endpoints.

> **Safety**: this is an experiment, not investment advice. Historical
> backtests do not predict future results. Do not connect this to real money.

## What it does

- Downloads free EUR/USD tick + M1 history from HistData.com (tick data is
  aggregated to H1 bid bars; the M1 page for 2018–2024 is served with broken
  download tokens, so the tick product is used as fallback for those years).
- Builds a validated, provenance-tracked H1 dataset (SQLite + CSV).
- Computes **causal** features and forward-return labels with an explicit
  timing contract: decide at bar close, execute at next open + spread/slippage.
- Trains supervised direction models (logistic / gradient boosting) with
  metrics beyond accuracy (AUC, log-loss, Brier).
- Trains a PPO agent in a custom Gymnasium environment (deterministic under
  seed, cost-aware reward, flat earns exactly zero).
- Combines both via a **hybrid policy** (RL veto + supervised no-trade zone +
  direction agreement), with per-decision audit rows.
- Backtests every strategy with identical costs; reports vs naive baselines.
- Runs a resumable paper-trading simulation (replay or realtime pacing) with
  idempotent orders, heartbeats, restart recovery, and a risk manager that has
  final veto on every order.
- Streamlit dashboard that renders **only real stored results** (never fabricates
  performance) with explicit empty states.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-dev.txt   # pytest, ruff
```

CPU-only is fully supported (torch CPU wheels).

## Quick start

```bash
# 1. Data (M1 where available, ASCII tick fallback for 2018-2024)
.venv/bin/python scripts/download_data.py --start 2018-01 --end 2026-10

# 2. Build + validate the H1 dataset (writes CSV + provenance to SQLite)
.venv/bin/python scripts/build_dataset.py
.venv/bin/python scripts/validate_data.py

# 3. Train supervised baseline, then RL (PPO)
.venv/bin/python scripts/train_supervised.py
.venv/bin/python scripts/train_rl.py

# 4. Backtests: baselines + supervised + RL + hybrid, one report
.venv/bin/python scripts/backtest.py

# 5. Paper simulation (replay a historical window or pace realtime)
.venv/bin/python scripts/run_simulation.py --mode replay --max-bars 500

# 6. Report + dashboard
.venv/bin/python scripts/generate_report.py
bash scripts/run_dashboard.sh          # http://localhost:8501

# 7. Seven-day paper-trading experiment (SIMULATED MONEY; see docs/EXPERIMENT.md)
.venv/bin/python scripts/experiment.py start --mode delayed_data_paper --strategy hybrid
.venv/bin/python scripts/experiment.py status
.venv/bin/python scripts/daily_report.py          # daily report from real records
.venv/bin/python scripts/daily_report.py --final  # final evaluation
```

All knobs live in `configs/default.yaml` (splits, costs, risk limits, hybrid
thresholds, exits, runner pacing).

## Experiment modes

- `historical_replay` — local dataset replay (accelerated or wall-clock).
  **Not live-market observation.**
- `delayed_data_paper` — free delayed hourly quotes (Yahoo chart API, no key,
  ~1–2 min delay, personal educational use). Each completed hour is traded
  exactly once. Simulated fills only.
- `current_data_paper` — not available (requires a licensed real-time feed).

## GitHub

CI (`.github/workflows/`): tests+ruff, 6-hourly free-source health probe,
synthetic end-to-end pipeline check, daily-report and manual final-eval
wrappers. Least-privilege `contents: read`; no secrets required. The
experiment itself runs on this machine — GitHub runners are ephemeral and
cannot resume SQLite state (documented in `docs/EXPERIMENT.md`).

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m ruff check src/ tests/ scripts/
```

The suite covers data validation, resampling, feature leakage (invariance to
future bars), split integrity, supervised persistence, RL env determinism and
reward semantics, every risk limit, paper-broker idempotency/restart recovery,
runner crash-resume without duplicate orders, hybrid decision rules, dashboard
smoke, and an end-to-end synthetic pipeline.

## Layout

```
src/fxagent/
  data/        HistData client (M1 + tick), candles, store, provenance,
               live_source (historical_replay | delayed_data_paper)
  features/    causal features, labels, chronological splits
  models/      supervised sklearn wrapper; Gym env + PPO training
  strategy/    baselines, supervised/RL policies, hybrid policy
  simulation/  cost model, next-open backtester, metrics, exits
  execution/   paper broker (SQLite), live-endpoint guard
  risk/        RiskManager, limits, emergency-stop file flag
  storage/     SQLite schema (v2: exits + experiments) + repository
  runner/      resumable simulation loop + 7-day experiment controller
  reporting/   metrics, comparison, markdown/json/csv reports
  dashboard/   Streamlit app (read-only over real results)
scripts/       entry points (download, build, train, backtest, sim, experiment,
               daily_report, report)
tests/         pytest suite
docs/          project plan, experiment ops guide, session reports
```

## Data source notes

- HistData.com ASCII M1: `YYYYMMDD HHMMSS;O;H;L;C;V`, UTC, bid-only.
- HistData.com ASCII ticks: `YYYYMMDD HHMMSSmmm,bid,ask,volume`.
- The H1 dataset used for research is **bid-side**; the ask is approximated in
  trading sims via the configured spread (default 0.8 pips).
- Free for personal use; check HistData terms before redistribution.

## License / use

Research and education only. No warranty. Simulated money only.
