# Phase 2 Implementation Report
## Seven-day autonomous forex paper-trading experiment

**Date:** 2026-10-12
**Project:** `~/Downloads/forex-research-agent`
**Commit:** `4b8a998` + follow-ups on `main`
**Verdict:** All in-scope Phase 2 work implemented and verified. The 7-day
experiment is **RUNNING** (not complete — it ends 2026-10-19T11:08Z).

> SIMULATED MONEY ONLY. No live trading exists or was added. The live-endpoint
> guard is untouched. Not investment advice.

---

## 1. Implementation summary

| Area | What was done |
|---|---|
| **Baseline (Phase A)** | Inspected git (no commits), ran 97 tests + ruff (green), found tick download **complete (84/84)**, dataset still had the 2022–24 gap, models trained on interim splits. Timestamped backup created (`backup/pre_phase2_20261012/`, 12 MB) **before** any migration. |
| **Data migration** | Canonical splits restored in `configs/default.yaml`; dataset rebuilt: **37,450 → 53,799 H1 bars**, gap 2022-04..2024-12 filled, max gap now 73h (weekends only). Models retrained on complete data. |
| **Market data (Phase B)** | New `src/fxagent/data/live_source.py`: explicit modes `historical_replay` (local CSV), `delayed_data_paper` (Yahoo hourly chart API — verified working from this machine **and** compatible with GitHub runners; free, no key, ~1–2 min delay, approximately-mid quotes, personal-educational use), `current_data_paper` **deliberately unimplemented**. OHLC validation (NaN, high<low, bounds, duplicates, positivity), retries with backoff, stale detection, weekend/market-closure check. Null-OHLC Yahoo rows are dropped, never imputed. |
| **Experiment controller (Phase C)** | New `src/fxagent/runner/experiment.py` + `scripts/experiment.py`: 7-day duration, unique experiment ID + SQLite metadata + append-only `experiment_events`, start/pause/resume/stop (control files `EXPERIMENT_PAUSE`/`EXPERIMENT_STOP` + SIGTERM/SIGINT), validated versioned checkpoints (incompatible → refused), idempotent bar processing (`run:bar_ts:strategy` keys), UTC throughout, heartbeats, stale-data no-trade, market-closure idle, exponential backoff on fetch errors, safe-stop after max consecutive errors, start/end reconciliation, only newly completed bars traded (never the in-progress hour), clean $10,000 balance per experiment. |
| **Exits (Phase D)** | New `src/fxagent/simulation/exits.py`: configurable stop-loss / take-profit / time-stop; worst-case intra-bar ordering (stop first when both inside one bar); persisted on positions (`stop_price`, `target_price`, `max_hold_bars`, `bars_held`, `exit_reason`); applied **before** any new signal; documented in config with "tune on train/val only" note. Defaults are **off** (no test-period tuning performed). |
| **GitHub (Phase E)** | 5 workflows (see §5). Repo hygiene: `.gitignore` excludes DB/raw/models/logs/backups; `.env.example` placeholders only; `requirements.lock.txt` already pinned. Cross-run persistence: **local SQLite is the source of truth**; documented limitation of ephemeral runners; no infinite processes in Actions; no secrets required. |
| **Reports (Phase F)** | `scripts/daily_report.py`: daily + final reports (Markdown/JSON/CSV) computed **only** from stored records; metrics block (return, DD, trades, win rate, PF, costs, exposure, exits, risk rejections, actions); sample-size warning (<20 trades); final report warns if <7 days elapsed in wall-clock mode; limitations section always present. Dashboard: new "Experiment" section (mode banner, freshness/heartbeat, equity, exits, decisions, events); existing sections intact; smoke-tested. |
| **Tests (Phase G)** | New `tests/test_phase2_experiment.py` (25 tests): exits semantics, worst-case ordering, validator, NaN/duplicates/positivity, staleness, weekend, Yahoo completed-bar filter + NaN drop (synthetic payloads), checkpoint incompatibility, migration v2, controller idempotent resume, stop flag, pause flag, exit persistence, reconciliation. Existing 97 tests still pass. |

## 2. Files created and modified

**Created**
- `src/fxagent/simulation/exits.py`
- `src/fxagent/data/live_source.py`
- `src/fxagent/runner/experiment.py`
- `scripts/experiment.py`
- `scripts/daily_report.py`
- `scripts/ci/probe_sources.py`
- `scripts/ci/synthetic_e2e.py`
- `.github/workflows/{tests,data-health,pipeline-check,daily-report,final-eval}.yml`
- `tests/test_phase2_experiment.py`
- `docs/EXPERIMENT.md`, `docs/PHASE2_IMPLEMENTATION_REPORT.md`, `CHANGELOG.md`
- `backup/pre_phase2_20261012/` (local only, git-ignored)

**Modified**
- `configs/default.yaml` — canonical splits; `exits:` section
- `src/fxagent/storage/db.py` — schema v2 (exit columns, experiments, experiment_events)
- `src/fxagent/execution/paper.py` — `exit_reason` on closes; `reset_balance()`
- `src/fxagent/dashboard/app.py` — Experiment section
- `README.md`, `docs/PROJECT_PLAN.md`
- `data/processed/EURUSD_H1.csv` — rebuilt (53,799 bars)
- `models/**` — retrained on complete data

## 3. Test and lint results

```
ruff check src/ tests/ scripts/     → All checks passed!
pytest tests/ -q                    → 122 passed (97 existing + 25 new)
dashboard AppTest smoke             → OK, 9 sections incl. Experiment
daily_report.py (live experiment)   → real metrics, limitations populated
```

## 4. Data source and experiment mode verified

- **historical_replay**: local `data/processed/EURUSD_H1.csv` (HistData M1 +
  tick fallback, bid basis, provenance in SQLite). Labelled "NOT live".
- **delayed_data_paper**: `https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X?interval=1h&range=5d`
  — verified reachable from this machine; free; no API key; last completed
  hour only; ~1–2 min delay; approximately-mid quotes (spread assumption
  still applied at fills); personal educational use, no redistribution.
- **current_data_paper**: not implemented (no verified licensed real-time feed).
- Backup/reference source probed: Frankfurter ECB daily (OK, daily cadence only).

## 5. GitHub workflows and status

| Workflow | Trigger | Permissions | Status |
|---|---|---|---|
| `tests` | push/PR/manual | `contents: read` | files committed; **runs after push to GitHub** |
| `data-health` | cron `17 */6 * * *` + manual | `contents: read` | same |
| `pipeline-check` | PR (runner paths) + manual | `contents: read` | synthetic e2e, labelled |
| `daily-report` | cron `30 22 * * *` + manual | `contents: read` | documents local-first path; uses artifact if provided |
| `final-eval` | **manual only** | `contents: read` | requires DB artifact or runs locally |

All pin `actions/checkout@v4`, `actions/setup-python@v5`,
`actions/upload-artifact@v4`; concurrency groups prevent overlaps; no secrets.

**Current status: committed locally on `main`; NOT yet pushed — requires your
GitHub repository (see §7).**

## 6. Exact commands

**Local**
```bash
cd ~/Downloads/forex-research-agent
.venv/bin/python -m pytest tests/ -q          # 122 green
.venv/bin/python -m ruff check src/ tests/ scripts/
.venv/bin/python scripts/experiment.py status
.venv/bin/python scripts/daily_report.py       # daily
.venv/bin/python scripts/daily_report.py --final
bash scripts/run_dashboard.sh
```

**GitHub (after push)**
- Push triggers `tests` automatically.
- Actions tab → `data-health` → *Run workflow*.
- Actions tab → `pipeline-check` → *Run workflow*.
- Actions tab → `final-eval` → *Run workflow* (after 7 days + DB artifact upload).

## 7. Manual GitHub setup required (your action)

```bash
cd ~/Downloads/forex-research-agent
# create an empty repo on github.com (no README/LICENSE to avoid conflicts), then:
git remote add origin git@github.com:<you>/<repo>.git
git push -u origin main
```
No secrets, tokens or paid plans needed. Student/GitHub Free is sufficient.

## 8. Remaining blockers and limitations

1. **7 days have not elapsed** — the experiment ends 2026-10-19T11:08Z. The
   final report must not be labelled complete before then (the script warns).
2. **Local-first persistence** — the experiment cannot resume across GitHub
   runners (ephemeral FS). Local Ubuntu box is the runner; GitHub is CI only.
3. **Exit parameters left at defaults (off)** — deliberately: tuning them on
   the test period would contaminate evaluation. Tune on train/val later.
4. **Yahoo free tier** — rate limits possible; controller backs off and
   safe-stops after 5 consecutive failures; not a licensed real-time feed.
5. **First 5 days of Yahoo history are processed as catch-up at start** —
   labelled as such in equity (they precede wall-clock day 1).
6. `sqlite3` CLI absent on this machine (use Python); no other blockers.

## 9. Start command (already executed — experiment RUNNING)

```bash
.venv/bin/python scripts/experiment.py start \
    --mode delayed_data_paper --strategy hybrid --duration-days 7
```
- Experiment ID: **`exp-20261012-110843-9792`**
- Ends: 2026-10-19T11:08:43Z · Pause: `scripts/experiment.py pause` ·
  Stop: `scripts/experiment.py stop` · Emergency: `touch data/EMERGENCY_STOP`

## 10. Final report command

```bash
.venv/bin/python scripts/daily_report.py --final
```
Outputs `reports/final_report_<experiment_id>.{md,json,csv}` with metrics,
limitations, incomplete-run warnings and heartbeat history — all from stored
records only.
