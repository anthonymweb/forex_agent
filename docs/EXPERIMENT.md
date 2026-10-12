# Seven-day paper-trading experiment — operations guide

> **SIMULATED MONEY ONLY.** No live-broker integration exists; the code
> refuses live endpoints. This experiment cannot place real orders. Not
> investment advice. Past performance does not predict future results.

## Operating modes (explicit — never silently switched)

| Mode | Meaning | Live? |
|---|---|---|
| `historical_replay` | Replay of the local HistData-derived H1 dataset. Can be accelerated (`--duration-days 0`) or wall-clock paced. | **No — not live-market observation** |
| `delayed_data_paper` | Polls the free Yahoo Finance hourly chart API (~1–2 min delay, no API key, personal educational use) and trades each newly completed hour exactly once. | Delayed observations, paper fills only |
| `current_data_paper` | **Not available.** Requires a verified licensed real-time feed; deliberately unimplemented. | — |

Every run and report records its mode, data source, last observation
timestamp and data-quality status.

## Commands

```bash
cd ~/Downloads/forex-research-agent

# Start a 7-day delayed-data paper experiment (hybrid strategy)
.venv/bin/python scripts/experiment.py start \
    --mode delayed_data_paper --strategy hybrid --duration-days 7

# Accelerated historical replay of the test split (finishes in minutes; NOT live)
.venv/bin/python scripts/experiment.py start \
    --mode historical_replay --strategy hybrid --duration-days 0

# Observe
.venv/bin/python scripts/experiment.py status

# Pause (idles; processes nothing) / resume
.venv/bin/python scripts/experiment.py pause
.venv/bin/python scripts/experiment.py resume --experiment-id <id>

# Clean stop (closes nothing until finish; flatten happens at end by default)
.venv/bin/python scripts/experiment.py stop

# Emergency stop — IMMEDIATE halt of new orders (independent of experiment flags)
touch data/EMERGENCY_STOP

# Daily and final reports (from real stored records only)
.venv/bin/python scripts/daily_report.py
.venv/bin/python scripts/daily_report.py --day 2026-10-13
.venv/bin/python scripts/daily_report.py --final

# Dashboard
bash scripts/run_dashboard.sh   # http://127.0.0.1:8501
```

## How it works

- **Timing contract** (same as the backtester): decision at close of bar *t*,
  simulated fill at open of bar *t+1*, paying half-spread + slippage +
  commission. In delayed mode the "next open" is the open of the newly
  completed hourly bar fetched from the source.
- **Idempotency**: every decision has key `run:bar_ts:strategy`. Kill the
  process at any point; on resume no bar is processed twice, no order is
  duplicated (verified by tests).
- **Exits** (`exits:` in `configs/default.yaml`): optional stop-loss,
  take-profit and time-stop. If stop and target both lie inside one bar,
  the stop fills first (worst case). Tune on train/validation only.
- **Risk manager** is independent and has final veto: position caps, daily
  loss, drawdown, stale data, spread, consecutive errors, emergency-stop
  file. Nothing in the experiment controller can override it.
- **Staleness**: if no completed bar arrives within `max_age_hours`
  (default 3) during market hours, heartbeats report `stale_data` and no
  trading occurs. Fetch errors back off exponentially and safe-stop after
  `max_consecutive_errors`.
- **Market closure**: weekend polling idles (`market_closed` heartbeats);
  no synthetic bars are ever generated.
- **State**: checkpoint in SQLite (`kv` table) + experiment row +
  append-only `experiment_events`. Checkpoints carry a schema version and
  are validated on restore; incompatible checkpoints are refused.

## Persistence & GitHub Actions (honest limits)

- The experiment runs **on this Ubuntu machine** where the SQLite database
  and models live. GitHub-hosted runners are ephemeral and cannot resume
  the experiment between runs; cross-run state is therefore local.
- GitHub provides: CI tests+ruff, scheduled free-source health checks, a
  synthetic end-to-end pipeline check, and manual final-eval packaging.
- Daily reports are generated locally (cron/systemd timer below); the
  `daily-report` workflow documents the same command for manual runs with
  an attached DB artifact.

### Optional local daily-report timer

```bash
crontab -e
# Every day 22:35 UTC: generate daily report
35 22 * * * cd $HOME/Downloads/forex-research-agent && .venv/bin/python scripts/daily_report.py >> logs/daily_report.log 2>&1
```

## GitHub setup (user action required)

```bash
cd ~/Downloads/forex-research-agent
git remote add origin git@github.com:<you>/<repo>.git   # or https URL
git push -u origin main
```

Workflows need no secrets. They request only `contents: read`.

## Completing the experiment

The experiment is **not complete until 7 calendar days have elapsed** in
`delayed_data_paper` wall-clock mode (or you explicitly stop early and the
final report says so). Generate the final report with:

```bash
.venv/bin/python scripts/daily_report.py --final
```

## Troubleshooting

| Symptom | Action |
|---|---|
| Heartbeats show `stale_data` | Source may be down or market closed; check `python scripts/ci/probe_sources.py` |
| `data_error` heartbeats | Yahoo rate-limiting; controller backs off and retries, safe-stops after 5 consecutive failures |
| Process killed / rebooted | `python scripts/experiment.py resume --experiment-id <id>` — no bars reprocessed |
| New orders must stop NOW | `touch data/EMERGENCY_STOP` (closes still allowed); remove file to re-arm after review |
| Pause processing | `python scripts/experiment.py pause` (flag file); resume clears it |
| Dashboard shows nothing | Ensure `data/fxagent.db` exists; restart `run_dashboard.sh` |
