# Project Plan — Hybrid AI Forex Research Agent

Status legend: `TODO` / `WIP` / `DONE` / `BLOCKED`

## 1. Mission Summary

Build a hybrid supervised-ML + reinforcement-learning forex research agent that runs on
simulated money only. Primary instrument: EUR/USD, hourly (H1) candles derived from
HistData.com free M1 data. Includes backtesting with realistic costs, an independent
risk controller, a local paper broker, a resumable 7-day-capable simulation runner, a
Streamlit dashboard, tests and documentation.

**Non-goals:** live trading, real funds, paid data/APIs, LLM control of risk/orders,
multiple currency pairs (until EUR/USD works end to end).

## 2. Phase Zero — Environment Assessment (completed 2026-10-11)

| Item | Finding |
|---|---|
| OS / CPU / RAM | Ubuntu 24.04.4 LTS, 4 cores, 23 GiB RAM, 63 GB free disk |
| Python | 3.12.3 (`/usr/bin/python3`), `venv` available |
| Git | 2.43.0 available |
| Network | PyPI, GitHub reachable |
| Existing repo | None — project created fresh at `~/Downloads/forex-research-agent` |
| Prior user work | None overwritten; Downloads only contains unrelated documents |

### Data-source research (verified by direct request on 2026-10-11)

| Source | Result | Decision |
|---|---|---|
| **HistData.com** free ASCII M1 | Working. Per-month page token → POST `get.php` returns real ZIP. EURUSD M1 available 2000-05 → present. | **Selected** as primary source |
| Yahoo Finance (`EURUSD=X`) | HTTP 429 on every request from this network | Blocked; not used |
| Dukascopy datafeed | HTTP 503 on tick files | Blocked; not used |
| Stooq CSV | Browser JS proof-of-work challenge | Not used (anti-bot measure) |
| TwelveData `demo` key | Works for EUR/USD 1h but demo tier is heavily rate-limited and recent-only | Backup only; insufficient volume |
| HistData FTP / Google Drive updates | Paid add-ons ($7/mo) | Not used (budget: none) |

**Assumption:** HistData M1 ASCII files are **bid-price** OHLC with UTC timestamps
(documented on their FAQ). Ask-side prices are synthesised for spread modelling via a
configurable constant spread; this is an assumption, not observed data.

**Known risk:** HistData year-level download pages are HTML-cached and can serve a stale
`tk` token (empty POST response). The month-level pages produce fresh tokens. The
downloader therefore iterates month by month with a 1.2 s politeness delay.

## 3. Architecture

```
scripts/  (CLI entry points)
   └── src/fxagent/
         config.py          YAML config + env overlay, path resolution
         data/              histdata_client (download), candles (M1→H1),
                            validation, store (SQLite + provenance)
         features/          causal feature builder, horizon labels
         models/supervised/ naive baselines, sklearn classifier, persistence
         models/rl/         Gymnasium H1 trading env, PPO training/eval
         strategy/          Strategy protocol; baseline / supervised / RL / hybrid
         simulation/        accounting (equity, P&L, costs), backtest engine
         execution/         ExecutionBroker protocol, local paper broker, live-guard
         risk/              RiskManager + limits (independent of models)
         storage/           SQLite migrations + repositories (runs, decisions,
                            trades, heartbeats, predictions)
         monitoring/        structured logging, health checks
         reporting/         metrics, evaluation report (md/json/csv)
         runner/            resumable continuous simulation runner
         dashboard/         Streamlit app reading real stored results only
```

### Data flow

1. `download_data.py` → `data/raw/histdata/EURUSD/M1/<year>/<month>/*.zip` (+ CSV)
2. `build_dataset.py` → validate → resample to H1 → `data/processed/EURUSD_H1.parquet`
   (fallback CSV) + provenance rows in SQLite
3. Chronological split (config): train / validation / final-test — never shuffled
4. `train_supervised.py` → features+labels from train only → model in `models/`
5. `train_rl.py` → PPO in custom env on train period → model in `models/`
6. `backtest.py` → each strategy (baseline / supervised / RL / hybrid) on val & test
   with identical cost assumptions → metrics + reports
7. `run_simulation.py` → paper broker + risk + chosen strategy, persistent state,
   resumable, 7-day capable
8. `dashboard/app.py` → reads SQLite/CSV only; empty states honest

### Timing model (anti-lookahead contract)

- Feature row at bar `t` uses information up to and including H1 close of `t`.
- Decision is produced at the close of bar `t`.
- Execution is simulated at the **open of bar `t+1`** plus half-spread (buy) or
  minus half-spread (sell), plus configurable slippage and per-side commission.
- P&L is marked to market on closes; open positions are force-closed at the last
  available close in the evaluation window and reported explicitly.
- Forward labels use the same future window the backtest would experience
  (`open[t+1] → close[t+h]`), so label and trading horizon agree.

## 4. Milestones and Acceptance Criteria

| # | Milestone | Acceptance criteria | Status |
|---|---|---|---|
| 0 | Env inspection + plan | This document; assumptions recorded | DONE |
| 1 | Skeleton + venv + deps | venv installs; `python -c "import fxagent"` works | DONE |
| 2 | Data pipeline | downloader pulls real EURUSD M1; validator catches dupes/gaps/bad prices; H1 dataset + provenance persisted; tests pass | WIP |
| 3 | Features + labels | causal features; leakage tests fail on shift-0 misuse; split integrity test | DONE |
| 4 | Supervised baseline | naive + sklearn trained reproducibly; prediction log written; metrics beyond accuracy | DONE |
| 5 | RL env + PPO | env deterministic under seed; PPO trains; cost-aware reward; eval on unseen period | DONE |
| 6 | Hybrid engine | documented decision rules; per-decision audit rows; honest comparison vs simpler strategies | DONE |
| 7 | Backtester | costs + next-open execution; all required metrics; reproducible | DONE |
| 8 | Paper broker + risk | order lifecycle, idempotency, restart recovery, every risk limit tested | DONE |
| 9 | Runner | resume after kill without duplicate orders; heartbeat; stale-data detection; market-closure skip | DONE |
| 10 | Dashboard | loads real DB results only; empty states; CSV export | DONE |
| 11 | Docs | README + 9 docs + CHANGELOG | WIP |
| 12 | Quality gate | full pytest green; ruff clean; end-to-end sim executed; report generated | DONE |

## 5. Resource / Budget Constraints

- CPU only; no GPU. PPO uses a small MLP; training measured in minutes, not hours.
- All data free (HistData). No paid APIs. No cloud. SQLite on local disk.
- Optional LLM explanation layer: **not implemented** in this release (core must work
  without it; it may never control risk or orders). Documented in KNOWN_LIMITATIONS.

## 6. Security / Safety Controls

- `execution/guards.py` refuses any live host/endpoint; paper mode is explicit and
  verified at startup; order submission disabled unless mode == `paper`.
- No credentials in source; `.env.example` lists names only; `.gitignore` excludes
  secrets, data, models, logs, DB files.
- Risk manager cannot be disabled by model code; emergency stop is a file flag +
  config; every rejection is logged with a reason code.

## 7. Current Status

- Phase 0–1 complete; venv + locked deps installed and import-verified.
- **91 pytest tests green; ruff clean** (2026-10-11).
- Data: M1 archives for 2025-01..2026-10 (22 months) downloaded. HistData serves
  2018–2024 M1 pages from cache with stripped download tokens (not retryable), so a
  **tick-data fallback** was implemented: ASCII tick months (bid+ask) aggregated to
  H1 bid bars. Tick download in progress (was 48/84 months at session end).
- Backtester, paper broker, risk manager, supervised model, RL env, hybrid policy,
  runner (resumable, idempotent) all implemented and unit/integration tested on
  synthetic data.
- Runner performance fix: WAL `synchronous=NORMAL` + one long-lived SQLite
  connection per run (300 bars: ~360s → 3.8s).

## 8. Next Concrete Actions

1. Finish tick download (2018–2024); run `scripts/build_dataset.py` +
   `scripts/validate_data.py` on the real combined H1 set.
2. Train supervised baseline (`train_supervised.py`), train PPO (`train_rl.py`),
   run backtests (`backtest.py`) incl. hybrid vs baselines comparison.
3. Execute end-to-end 7-day (or full replay) simulation; generate report; smoke-test
   dashboard against the real DB.
4. Write README + docs/ pages + CHANGELOG (milestone 11); final quality gate (12).

## 9. Session Log

- **2026-10-11 (morning)** — Phase 0 inspection, data-source verification (HistData
  selected; Yahoo/Dukascopy/Stooq blocked), skeleton + configs created, plan written.
- **2026-10-11 (day)** — Full source tree implemented (data, features, supervised,
  RL env, strategies, simulation, execution, risk, storage, runner, reporting,
  dashboard). Deps installed and locked. HistData M1 downloader working; discovered
  2018–2024 M1 pages served with empty download tokens (stale cache). Verified tick
  pages still carry tokens; implemented tick-data fallback (fetch_tick_month,
  ticks_to_ohlc, build_dataset tick path). Fixed M1 CSV parser (semicolon separator).
- **2026-10-11 (afternoon)** — Wrote 91 tests across 13 files covering validation,
  candles, leakage, splits, backtest+accounting, supervised, RL env, risk, paper
  broker, runner recovery, hybrid rules, end-to-end. Fixed real bugs found by tests:
  tz-aware timestamp crash in backtester (PositionState.opened_index), account_state
  hiding positions without a mark price, runner SQLite-per-bar fsync bottleneck,
  WAL synchronous mode. Ruff autofixes applied; suite green.

**Knowledge gained**
- HistData M1 ASCII: `YYYYMMDD HHMMSS;O;H;L;C;V` (semicolon, 6 fields, UTC, bid-only).
- HistData tick ASCII: `YYYYMMDD HHMMSSmmm,bid,ask,volume` (comma); ~8–19 MB/month;
  separate page cache from M1 — survives the M1 token-poisoning.
- Download token is month-page-specific; reuse across months fails (`tk` mismatch).
- SQLite on this host: full fsync ≈ 100ms; batch commits + WAL NORMAL = 100× speedup.
- Test dir names can trip `assert_not_live` markers ("live"/"real" substrings).

- **2026-10-11 (evening)** — Completed first full real-data run-through:
  dataset build (37,450 H1 bars, 2018-01→2026-10 with 2022-04→2024-12 gap while
  throttled download continues), supervised training (test AUC 0.523), PPO 200k
  steps (~220s CPU), backtests for all 7 strategies on val+test, full paper
  replay of the test split (10,625 bars, 72s, reconciled), final evaluation
  report + dashboard smoke tests. Fixed RL feature_names mismatch (30 vs 31
  columns with `_close`) by persisting feature names in meta.json. Ruff clean,
  97 tests green.

**Honest first results (test split 2025-01→2026-10, costs 0.8+0.2 pips, 3$/mm)**
- always_long +83.4% (one buy-and-hold trade; strong EURUSD uptrend regime)
- sma_cross_8_48 +55.8%
- rl +92.4% (single long held — learned buy-and-hold in this regime)
- supervised −168.6% (1,397 trades; 55.7% win rate but negative skew — costs + chop)
- hybrid −10.9% (449 trades; RL veto + no-trade zone cut the supervised blowup)
- val split (2021-10→2022-03 downtrend): supervised +14.9%, hybrid −5.8%,
  always_long −51.6%. Edges do NOT transfer across regimes — exactly what the
  project set out to test. No profitability claims are made.

**Remaining issues**
- Tick download for 2022-04..2024-12 incomplete (HistData throttling ~25 min/month);
  background loop `scripts/download_tick_loop.sh` keeps retrying. After it
  completes: rebuild dataset, restore canonical splits in configs/default.yaml
  (train→2022-12, val→2023-01..2024-12), retrain, re-report.
- PPO with 200k steps is a minimal run; longer training + hyperparameter sweep
  would be the next research step (still CPU-bounded).
- Docs milestone 11: README done; remaining docs/ pages + CHANGELOG pending.
- No committed git history yet (files untracked).
