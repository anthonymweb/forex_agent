"""Resumable continuous simulation runner.

Operational test harness for the seven-day evaluation. Modes:

* ``replay``     — accelerated historical replay (``step_seconds`` per bar).
* ``realtime``   — historical replay paced at wall-clock speed; runs for
                   ``duration_days``. Data exhaustion is reported, never hidden.

Safety properties:

* Every order is risk-gated (PaperBroker -> RiskManager).
* Emergency-stop file halts new orders immediately (closes are still allowed).
* State (cursor, counters) is persisted after every committed bar, so a
  process kill and restart never re-executes already-processed bars
  (idempotency keys + cursor are both honoured).
* Heartbeats are written every ``heartbeat_seconds`` and at every bar close
  batch so the dashboard can detect stalls.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from fxagent.config import Config
from fxagent.execution.paper import PaperBroker
from fxagent.monitoring.health import run_all_checks
from fxagent.risk.controls import RiskLimits, RiskManager
from fxagent.simulation.accounting import CostModel
from fxagent.simulation.backtest import FLAT, LONG, SHORT, DecisionContext, PositionState
from fxagent.storage.db import Repository, connect, migrate, session

log = logging.getLogger("runner")

ACTION_MAP = {"flat": FLAT, "long": LONG, "short": SHORT}


@dataclass
class RunnerState:
    run_id: str
    cursor: int = -1          # index of last committed bar
    bars_processed: int = 0
    orders_placed: int = 0
    orders_rejected: int = 0
    status: str = "starting"
    last_bar_ts: str = ""
    daily_realised_pnl: float = 0.0
    peak_equity: float = 0.0
    details: dict = field(default_factory=dict)


class SimulationRunner:
    def __init__(
        self,
        cfg: Config,
        strategy,
        *,
        mode: str | None = None,
        run_id: str | None = None,
        step_seconds: float | None = None,
        max_bars: int | None = None,
        duration_days: float | None = None,
        initial_balance: float | None = None,
    ):
        self.cfg = cfg
        self.strategy = strategy
        self.mode = mode or cfg.get("runner.mode", "replay")
        self.step_seconds = (
            step_seconds if step_seconds is not None else float(cfg.get("runner.step_seconds", 0.05))
        )
        self.max_bars = max_bars if max_bars is not None else int(cfg.get("runner.max_bars", 0))
        self.duration_days = (
            duration_days if duration_days is not None else float(cfg.get("runner.duration_days", 7))
        )
        self.run_id = run_id or f"sim-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        cost = CostModel(
            spread_pips=float(cfg.get("costs.spread_pips", 0.8)),
            slippage_pips=float(cfg.get("costs.slippage_pips", 0.2)),
            commission_per_mm=float(cfg.get("costs.commission_per_million", 3.0)),
            pip_size=float(cfg.get("instrument.pip_size", 0.0001)),
        )
        limits = RiskLimits.from_config(cfg.get("risk", {}) or {}, pip_size=cost.pip_size)
        self.risk = RiskManager(limits)
        self.broker = PaperBroker(
            db_path=cfg.db_path,
            cost=cost,
            risk=self.risk,
            initial_balance=initial_balance
            or float(cfg.get("paper_broker.initial_balance_usd", 10_000.0)),
            run_id=self.run_id,
        )
        self.cost = cost
        self.units = int(cfg.get("position.default_units", 100_000))
        self.prob_low = float(cfg.get("supervised.prob_low", 0.55))
        self.prob_high = float(cfg.get("supervised.prob_high", 0.60))
        self.heartbeat_seconds = float(cfg.get("runner.heartbeat_seconds", 30))

    # ---- state persistence --------------------------------------------
    def _state_key(self) -> str:
        return f"runner_state:{self.run_id}"

    def _load_state(self) -> RunnerState | None:
        with session(self.cfg.db_path) as conn:
            row = conn.execute("SELECT value FROM kv WHERE key=?", (self._state_key(),)).fetchone()
            if row is None:
                return None
            raw = json.loads(row["value"])
            return RunnerState(**raw)

    def _save_state(self, state: RunnerState, conn=None) -> None:
        payload = json.dumps(state.__dict__, default=str)
        if conn is not None:
            conn.execute(
                "INSERT INTO kv(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (self._state_key(), payload),
            )
            return
        with session(self.cfg.db_path) as conn:
            conn.execute(
                "INSERT INTO kv(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (self._state_key(), payload),
            )

    # ---- run ----------------------------------------------------------
    def run(self, bars: pd.DataFrame, features: pd.DataFrame) -> dict:
        """Execute the simulation loop. Returns a summary dict."""
        if len(bars) != len(features):
            raise ValueError("bars/features length mismatch")
        started_wall = time.monotonic()
        state = self._load_state()
        if state is None:
            state = RunnerState(run_id=self.run_id, status="running")
            self.broker.init_balance()
            with session(self.cfg.db_path) as conn:
                repo = Repository(conn)
                if repo.get_run(self.run_id) is None:
                    repo.create_run(
                        self.run_id,
                        strategy=self.strategy.name,
                        mode=self.mode,
                        model_versions=getattr(self.strategy, "model_versions", None),
                        config_snapshot=self.cfg.data,
                    )
        else:
            state.status = "running"
            log.info(
                "resuming run %s from cursor=%d (%d bars processed)",
                self.run_id, state.cursor, state.bars_processed,
            )

        self._save_state(state)
        checks = run_all_checks(self.cfg.db_path, self.cfg.path("project.paths.raw_dir"))
        for chk in checks:
            with session(self.cfg.db_path) as conn:
                Repository(conn).insert_health(chk["check"], chk["ok"], chk)

        feature_rows = features.to_dict(orient="records")
        open_ts = bars["timestamp"].to_numpy()
        opens = bars["open"].to_numpy(dtype=float)
        closes = bars["close"].to_numpy(dtype=float)

        start_idx = state.cursor + 1
        end_idx = len(bars) if self.max_bars <= 0 else min(len(bars), start_idx + self.max_bars)
        last_heartbeat = 0.0
        total_planned = max(1, end_idx - start_idx)
        if self.mode == "realtime":
            # Spread the planned bars evenly across the wall-clock duration.
            bar_sleep = (self.duration_days * 86400.0) / total_planned
            wall_deadline = time.monotonic() + self.duration_days * 86400.0
        else:
            bar_sleep = max(0.0, self.step_seconds)
            wall_deadline = None
        current_day = ""

        log.info(
            "run %s mode=%s bars [%d, %d) strategy=%s",
            self.run_id, self.mode, start_idx, end_idx, self.strategy.name,
        )

        # One long-lived connection for the whole loop: opening a fresh
        # SQLite session per bar dominated wall time on slow disks.
        db = connect(self.cfg.db_path)
        migrate(db)
        try:
            for i in range(start_idx, end_idx):
                if wall_deadline is not None and time.monotonic() >= wall_deadline:
                    state.status = "duration_reached"
                    log.info("wall-clock duration reached (%.2f days)", self.duration_days)
                    break

                now = pd.Timestamp(open_ts[i])
                day_key = str(now.date())

                # Build decision context at close of bar i-1 (we are AT bar i's open).
                prev = i - 1
                account = self.broker.account_state(mark_price=float(closes[prev]))
                pos = self._position_state(account)
                ctx = DecisionContext(
                    index=prev,
                    timestamp=pd.Timestamp(open_ts[prev]),
                    features=pd.Series(feature_rows[prev]),
                    position=pos,
                    bars_seen=prev,
                )
                target = ACTION_MAP[self.strategy.decide(ctx)]

                desired_units = 0
                if target == LONG:
                    desired_units = self.units
                elif target == SHORT:
                    desired_units = -self.units

                bar_ts = str(now)
                idem = f"{self.run_id}:{bar_ts}:{self.strategy.name}"

                # Idempotency: if this bar was already committed, skip execution.
                dup = db.execute(
                    "SELECT id, executed FROM decisions WHERE idempotency_key=?", (idem,)
                ).fetchone()
                if dup is not None:
                    state.cursor = i
                    state.bars_processed += 1
                    self._save_state(state, conn=db)
                    db.commit()
                    continue

                risk_ctx_drawdown = self._drawdown_pct(account["equity"], state)
                daily_pnl = state.daily_realised_pnl
                executed = False
                order_id = None
                risk_verdict = "allow"
                risk_reasons: list[str] = []

                if desired_units != 0 or pos.is_open:
                    if target == FLAT:
                        if pos.is_open:
                            res = self.broker.close_position(
                                ts=bar_ts, mid_price=float(opens[i]),
                                client_order_id=f"{idem}-close",
                            )
                            executed = res.ok
                            order_id = res.client_order_id
                            if not res.ok:
                                risk_verdict = "reject"
                                risk_reasons = [res.reject_reason or "close_failed"]
                        # flat with no position: nothing to do
                    else:
                        res = self.broker.submit_market_order(
                            side=target,
                            units=self.units,
                            ts=bar_ts,
                            mid_price=float(opens[i]),
                            client_order_id=idem,
                            daily_pnl=daily_pnl,
                            drawdown_pct=risk_ctx_drawdown,
                            bars_stale=0,
                            spread_pips=self.cost.spread_pips,
                        )
                        executed = res.ok
                        order_id = res.client_order_id
                        if not res.ok:
                            risk_verdict = "reject"
                            risk_reasons = [res.reject_reason or "rejected"]
                        else:
                            state.orders_placed += 1

                repo = Repository(db)
                repo.insert_decision(
                    run_id=self.run_id,
                    ts=bar_ts,
                    strategy=self.strategy.name,
                    action=target,
                    units=abs(desired_units) if desired_units else 0,
                    confidence=None,
                    supervised_prob=getattr(self.strategy, "last_prob_up", None),
                    rl_action=getattr(self.strategy, "last_rl_action", None),
                    risk_verdict=risk_verdict,
                    risk_reasons=risk_reasons,
                    model_versions=getattr(self.strategy, "model_versions", None),
                    executed=executed,
                    order_id=order_id,
                    idempotency_key=idem,
                )

                # Equity snapshot at bar close
                account = self.broker.account_state(mark_price=float(closes[i]))
                open_pnl = account["open_position"]["unrealised_pnl"] if account["open_position"] else 0.0
                repo.insert_equity(
                    self.run_id, bar_ts, account["balance"], account["equity"], open_pnl
                )

                state.peak_equity = max(state.peak_equity, account["equity"])
                if day_key != current_day:
                    current_day = day_key
                    log.info(
                        "day %s start: equity=%.2f balance=%.2f processed=%d",
                        day_key, account["equity"], account["balance"], state.bars_processed,
                    )

                state.cursor = i
                state.bars_processed += 1
                state.last_bar_ts = bar_ts
                state.status = "running"
                self._save_state(state, conn=db)

                now_mono = time.monotonic()
                if now_mono - last_heartbeat >= self.heartbeat_seconds or self.step_seconds >= 1.0:
                    repo.insert_heartbeat(
                        self.run_id, self.mode, bar_ts, state.bars_processed, "running",
                        {"equity": account["equity"], "cursor": i},
                    )
                    last_heartbeat = now_mono

                db.commit()

                if self.step_seconds > 0 or self.mode == "realtime":
                    time.sleep(bar_sleep)

        except KeyboardInterrupt:
            state.status = "interrupted"
            log.warning("interrupted by user")
        except Exception:
            state.status = "failed"
            self._save_state(state)
            log.exception("runner failed")
            with session(self.cfg.db_path) as conn:
                Repository(conn).finish_run(self.run_id, status="failed")
            raise
        finally:
            db.close()

        # End-of-run: optionally close any open position.
        final_price = float(closes[min(len(closes) - 1, max(0, state.cursor))])
        if self.cfg.get("runner.close_positions_at_end", True):
            open_pos = self.broker.account_state(mark_price=final_price)["open_position"]
            if open_pos:
                self.broker.close_position(
                    ts=state.last_bar_ts or str(pd.Timestamp(open_ts[-1])),
                    mid_price=final_price,
                    client_order_id=f"{self.run_id}-final-close",
                )

        account = self.broker.account_state(mark_price=final_price)
        elapsed = time.monotonic() - started_wall
        if state.status == "running":
            state.status = "completed"
        self._save_state(state)

        with session(self.cfg.db_path) as conn:
            repo = Repository(conn)
            repo.insert_heartbeat(
                self.run_id, self.mode, state.last_bar_ts, state.bars_processed,
                state.status, {"equity": account["equity"]},
            )
            repo.finish_run(self.run_id, status=state.status)

        summary = {
            "run_id": self.run_id,
            "strategy": self.strategy.name,
            "mode": self.mode,
            "status": state.status,
            "bars_processed": state.bars_processed,
            "orders_placed": state.orders_placed,
            "final_equity": account["equity"],
            "final_balance": account["balance"],
            "wall_seconds": round(elapsed, 2),
            "reconcile": self.broker.reconcile(),
        }
        log.info("run finished: %s", json.dumps(summary, indent=2))
        return summary

    # ---- helpers ------------------------------------------------------
    @staticmethod
    def _position_state(account: dict) -> PositionState:
        op = account.get("open_position")
        if not op:
            return PositionState()
        return PositionState(
            side=op["side"], units=op["units"], entry_price=op["entry_price"],
            opened_ts=op["opened_ts"],
        )

    @staticmethod
    def _drawdown_pct(equity: float, state: RunnerState) -> float:
        if state.peak_equity <= 0:
            return 0.0
        return max(0.0, (state.peak_equity - equity) / state.peak_equity * 100.0)
