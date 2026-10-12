# Changelog

All notable changes to fxagent. Simulated money only.

## [0.2.0] — 2026-10-12 — Phase 2: seven-day experiment upgrade

### Added
- `src/fxagent/simulation/exits.py` — configurable simulated stop-loss,
  take-profit and time-stop (worst-case intra-bar ordering; tested).
- `src/fxagent/data/live_source.py` — market-data source interface with
  explicit modes: `historical_replay`, `delayed_data_paper` (Yahoo hourly,
  free, no key, ~1–2 min delay), `current_data_paper` (deliberately
  unimplemented). Validation, retries/backoff, staleness and weekend
  detection.
- `src/fxagent/runner/experiment.py` — seven-day experiment controller:
  start/pause/resume/stop (control files + signals), validated SQLite
  checkpoints, idempotent per-bar processing, heartbeats, stale-data
  no-trade, safe-stop on consecutive errors, experiment metadata +
  append-only lifecycle events.
- `scripts/experiment.py` — CLI (start/resume/status/pause/stop).
- `scripts/daily_report.py` — daily and final reports (Markdown/JSON/CSV)
  computed only from stored SQLite records; sample-size and incomplete-
  experiment warnings.
- GitHub Actions: `tests`, `data-health` (6-hourly source probe),
  `pipeline-check` (synthetic end-to-end, labelled), `daily-report`,
  `final-eval` (manual). Least-privilege permissions, concurrency groups.
- `scripts/ci/probe_sources.py`, `scripts/ci/synthetic_e2e.py`.
- `docs/EXPERIMENT.md` — operations guide.
- DB schema v2: position exit columns (`stop_price`, `target_price`,
  `max_hold_bars`, `bars_held`, `exit_reason`), `experiments` and
  `experiment_events` tables.
- Dashboard "Experiment" section (mode, freshness, equity, exits, events).

### Changed
- Canonical train/val/test splits restored after full 2018-01..2024-12 tick
  download completed (84/84 months); dataset rebuilt to 53,799 H1 bars
  (gap 2022-04..2024-12 filled).
- Models retrained on complete data (supervised + PPO).
- `PaperBroker.close_position` accepts `exit_reason` (persisted).

## [0.1.0] — 2026-10-11 — Phase 1: research system

Full supervised + RL hybrid research stack: HistData ingestion (M1 + tick
fallback), causal features, chronological splits, next-open backtester with
costs, risk manager, idempotent paper broker, resumable runner, Streamlit
dashboard, 97-test suite. First honest evaluation: supervised AUC ≈0.52–0.53;
hybrid risk-gate reduced supervised's −169% test loss to −11%.
