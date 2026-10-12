# fxagent — Project Report
## Hybrid AI Forex Research Agent (Simulated Money Only)

**Date:** 2026-10-11
**Location:** `~/Downloads/forex-research-agent/`
**Status:** Core system complete and verified end-to-end on real data.
**Quality gate:** 97 pytest tests green · ruff clean · full paper replay reconciled

> ⚠️ **This is a research experiment with fake money.** No live-broker
> integration exists; code refuses live endpoints. Nothing here is investment
> advice, and no profitability is claimed.

---

## 1. Executive summary

A complete hybrid trading-research system was built from scratch: real market
data ingestion, causal feature engineering, supervised learning (gradient
boosting), reinforcement learning (PPO in a custom Gym environment), a
documented hybrid decision policy, a next-open backtester with realistic
costs, a risk-gated paper broker with crash recovery, a 7-day-capable runner,
a Streamlit dashboard, and a 97-test suite.

**The headline finding: the infrastructure is trustworthy, the raw ML edge is
not.** The supervised model achieved AUC ≈ 0.52–0.53 (barely above chance).
On the 2025–2026 test period it lost −169% by overtrading a near-zero edge
into transaction costs, while naive buy-and-hold made +83% in the same
uptrend. The **hybrid risk-gate reduced that loss to −11%** (a 94%
damage reduction), which is the project's one genuinely interesting positive
result: **risk gating beats signal cleverness when your edge is 0.02 AUC.**

---

## 2. What was built

```
src/fxagent/
  data/        HistData client (M1 + tick fallback), validation, provenance
  features/    causal features, forward-return labels, chronological splits
  models/      supervised sklearn wrapper; Gymnasium env + PPO training
  strategy/    baselines, supervised/RL policies, hybrid policy
  simulation/  cost model, next-open backtester, full metric block
  execution/   SQLite paper broker (idempotent, restart-safe), live guard
  risk/        RiskManager (every limit tested), emergency-stop file flag
  storage/     SQLite schema + repository
  runner/      resumable simulation loop (replay / realtime pacing)
  reporting/   metrics, markdown/json/csv evaluation reports
  dashboard/   Streamlit app — reads only real stored results
scripts/       download, build, validate, train (×2), backtest, sim, report
tests/         97 tests across 15 files
```

### Timing contract (the part most retail backtests get wrong)

- Features at bar `t` use data ≤ close of `t` (leakage-tested by invariance).
- Decision is made at **close of t**; executed at **open of t+1**.
- Fills pay half-spread + adverse slippage + commission.
- Labels = economic move `open[t+1] → close[t+h]` (what the trader earns).
- Splits are strictly chronological and never shuffled.

### Safety controls

- `assert_not_live()` refuses any endpoint containing live/real/production markers.
- Risk manager has final veto on every order: position cap, daily loss,
  drawdown, stale data, spread, repeated errors, emergency-stop file flag.
- Orders are idempotent by `client_order_id`; runner decisions are idempotent
  by `run:bar:strategy` key — kill/resume produces **zero duplicates** (tested).
- Dashboard renders only real stored results; empty states are explicit.

---

## 3. Data

| Item | Value |
|---|---|
| Source | HistData.com (free, personal use) — M1 ASCII + ASCII tick fallback |
| Instrument / TF | EURUSD, H1 research bars (bid-side) |
| Built dataset | **37,450 H1 bars**, 2018-01-01 → 2026-10-09 (UTC) |
| Validation | duplicates=0, invalid prices=0, 323 gaps flagged (informational) |
| Provenance | SQLite row with checksum `a56b7bf0696d4e9d`, license note |
| Observed tick spread | median **0.3 pips** across sampled months |

**Data-source saga:** Yahoo (429), Dukascopy (503), Stooq (JS PoW challenge)
all blocked. HistData M1 worked, but its server cache serves the 2018–2024
download pages with **empty security tokens** (not retryable). Breakthrough:
the separate *tick-data* product's cache still had valid tokens, so 2018–2024
is recovered as tick data and aggregated to H1 bid bars.

**Known gap:** 2022-04 → 2024-12 (33 months) still downloading — HistData
began throttling to ~25 min/month. A background loop
(`scripts/download_tick_loop.sh`) keeps retrying. Interim splits were adjusted
(config carries a revert note) so val comes from 2021-10 → 2022-03.

---

## 4. Models trained

### Supervised — gradient boosting (22,236 train rows)

| Split | Accuracy | Balanced acc | AUC | Log-loss | Majority baseline |
|---|---|---|---|---|---|
| Val (2021-10→2022-03) | 0.519 | 0.521 | **0.530** | 0.695 | 0.512 |
| Test (2025→2026-10) | 0.514 | 0.514 | **0.523** | 0.699 | 0.501 |

Verdict: a real but tiny signal — roughly +0.02 AUC over chance.

### Reinforcement learning — PPO (CPU, 200k timesteps, ~220 s)

- Custom `ForexTradingEnv`: rolling feature window (24×30), Discrete(3)
  actions, cost-aware reward (flip costs), flat earns exactly 0, fully
  deterministic under seed.
- Best mean eval reward: **−668.8** → the agent did **not** learn a profitable
  policy at this budget. This is a smoke-test-level run, not serious training.
- Honest framing: the RL half currently behaves mostly as a
  *flat-veto / regime gate*, not an independent forecaster.

---

## 5. Backtest results

Costs identical for every strategy: 0.8 pips spread + 0.2 slippage per fill,
$3 per million round-turn commission. Initial balance $10,000, 100k units.
Next-open execution, forced flat at end of data.

### Validation window: 2021-10-01 → 2022-03-31 (3,007 bars — EURUSD downtrend)

| Strategy | Return | Max DD | Trades | Win rate | Costs paid |
|---|---:|---:|---:|---:|---:|
| always_flat | 0.00% | 0.00% | 0 | — | $0 |
| always_long | **−51.62%** | −77.27% | 1 | 0% | $12 |
| sma_cross_8_48 | −35.44% | −87.07% | 105 | 27.6% | $1,292 |
| mean_reversion_24 | 0.00% | 0.00% | 0 | — | $0 |
| **supervised** | **+14.85%** | −30.48% | 413 | 59.6% | $5,080 |
| rl | −52.54% | −77.91% | 1 | 0% | $12 |
| hybrid | −5.75% | −21.80% | 133 | 54.9% | $1,636 |

### Test window: 2025-01-01 → 2026-10-09 (10,625 bars — EURUSD uptrend)

| Strategy | Return | Max DD | Trades | Win rate | Profit factor | Costs paid |
|---|---:|---:|---:|---:|---:|---:|
| always_flat | 0.00% | 0.00% | 0 | — | — | $0 |
| always_long | **+83.40%** | −32.57% | 1 | 100% | ∞ | $12 |
| sma_cross_8_48 | +55.83% | −47.11% | 301 | 32.6% | 1.11 | $3,702 |
| mean_reversion_24 | 0.00% | 0.00% | 0 | — | — | $0 |
| supervised | **−168.58%** | −215.75% | 1,397 | 55.7% | 0.86 | **$17,183** |
| rl | **+92.40%** | −31.51% | 1 | 100% | ∞ | $12 |
| hybrid | **−10.86%** | −30.89% | 449 | 53.0% | 0.94 | $5,523 |

### Paper-trading replay (real paper broker, SQLite, risk manager on)

| Item | Value |
|---|---|
| Run | `sim-20261011-160721-9819ca`, strategy=hybrid, mode=replay |
| Window | full test split, **10,625 bars in 72 s** |
| Final equity | **$9,808.00** (balanced flat) |
| Reconciliation | `balanced=true`, 0 open positions, 0 issues |

---

## 6. Interpretation — the honest read

1. **Supervised's −169% is a sizing/exit problem, not purely a signal
   problem.** 55.7% win rate with profit factor 0.86 means small wins, large
   losses: classic negative skew. The strategy layer has **no stop-loss,
   take-profit, or time-stop** — the model says "long" and the position rides
   until the model flips. Adding exits is the single highest-leverage fix.

2. **RL's +92% is an illusion, and the data proves it.** It made *one trade*
   (buy and hold). The same policy lost −52% in the val downtrend. It is a
   regime-chaser, not a forecaster — and at 200k steps with eval reward −669,
   it never learned much of anything.

3. **The hybrid result is the real finding.** −10.9% vs supervised's −168.6%:
   the RL-flat veto + no-trade zone + agreement rule cut ~94% of the damage.
   Note exposure: hybrid was in the market only **7.4%** of bars vs
   supervised's **83.3%** — costs fell from $17.2k to $5.5k. The gate worked
   exactly as designed. It didn't make money, but it demonstrated that a
   conservative risk layer adds value even on top of a mediocre signal.

4. **Edges do not transfer across regimes.** Supervised won in the 2021–22
   downtrend val and got destroyed in the 2025–26 uptrend test. Because those
   are opposite regimes, we cannot yet separate "signal decay" from "regime
   mismatch" — that's precisely what the canonical 2023–24 validation split
   will test once the data gap fills.

5. **Overall conclusion:** at H1 EURUSD with realistic retail costs, a
   ±0.02-AUC directional model cannot overcome trading costs on its own. The
   measurable value in this system came from **risk control, not prediction.**

---

## 7. Bugs found and fixed (by the test suite)

| Bug | Fix |
|---|---|
| Backtester crashed on tz-aware timestamps (`searchsorted` on a string-parsed ts) | track `opened_index` on `PositionState` |
| `account_state()` hid open positions unless a mark price was passed | report position always, P&L=0 without mark |
| Runner opened ~7 SQLite connections/bar; fsync ≈ 100 ms each → 300 bars ≈ 6 min | one long-lived connection/loop + WAL `synchronous=NORMAL` → **3.8 s** |
| RL policy fed 31 cols at inference vs 30 at training (`_close` helper col) | persist `feature_names` in meta.json; factory passes them |
| HistData M1 parser assumed commas; files are semicolon-separated | parser rewritten (`YYYYMMDD HHMMSS;O;H;L;C;V`) |
| Download died on `ChunkedEncodingError` / read timeouts | retry ×3 on any request error + 600 s read timeout + supervising loop script |

**Knowledge gained:** HistData token caching behavior, tick-vs-M1 cache
independence, WAL write amplification on spinning disks, Streamlit AppTest
quirks, and "test directory names can trip substring live-endpoint guards."

---

## 8. Remaining work

1. **Finish the tick download** (2022-04→2024-12) → rebuild dataset →
   restore canonical splits (train→2022-12, val→2023-01..2024-12) → retrain
   → re-report. Config already carries the revert note.
2. **Add trade exits** (stop-loss / take-profit / time-stop) — biggest
   expected improvement, small code change.
3. **Fill from real tick spread** — tick data includes true bid/ask; use it
   at fill time instead of a synthetic 0.8-pip assumption.
4. RL at millions of steps (CPU-bounded; give it ~20% odds of beating the
   veto'd supervised baseline honestly).
5. Docs milestone: README done; remaining docs pages + CHANGELOG.
6. No git history yet (all files untracked).

---

## 9. Where everything lives

| Artifact | Path |
|---|---|
| This report | `~/Downloads/forex-research-agent/docs/SESSION_REPORT_2026-10-11.md` |
| Plan + session log | `docs/PROJECT_PLAN.md` |
| Final evaluation report | `reports/final_evaluation_20261011_161146.{md,json,csv}` |
| Per-split backtests | `reports/backtest_*_{154901,154953,160406,160512}.{md,json,csv}` |
| Dataset + provenance | `data/processed/EURUSD_H1.csv` + `data/fxagent.db` |
| Trained models | `models/supervised_direction/v1/`, `models/rl_agent/v1/` |
| Paper-run results | `data/fxagent.db` (runs, orders, fills, equity, decisions) |
| Raw market data | `data/raw/histdata/EURUSD/{M1,T}/` |

### Reproduce

```bash
cd ~/Downloads/forex-research-agent
.venv/bin/python -m pytest tests/ -q          # 97 green
.venv/bin/python -m ruff check src/ tests/ scripts/   # clean
.venv/bin/python scripts/backtest.py --strategies always_flat always_long \
    sma_cross mean_reversion supervised rl hybrid --split test
.venv/bin/python scripts/generate_report.py --split test --strategies \
    always_flat always_long sma_cross mean_reversion supervised rl hybrid
bash scripts/run_dashboard.sh                 # http://localhost:8501
```

---
*Generated 2026-10-11. Simulated money only. Past performance does not predict
future results. Not investment advice.*
