"""Seven-day autonomous experiment controller (paper trading only).

Modes
-----
``historical_replay``
    Accelerated or wall-clock-paced replay of the local processed dataset.
    Explicitly NOT live-market observation; every report says so.

``delayed_data_paper``
    Polls a free delayed data source (Yahoo hourly FX by default) and
    processes each newly COMPLETED hourly bar exactly once. Decisions are
    made at the close of the previous bar and executed at the new bar's
    open (same timing contract as the backtester).

``current_data_paper``
    Unimplemented on purpose — requires a verified licensed real-time feed.

Control files (under the configured data directory, same folder as the
emergency-stop flag):

* ``EXPERIMENT_PAUSE`` — while present, the loop idles and processes nothing.
* ``EXPERIMENT_STOP``  — clean shutdown at the next loop iteration.

State (cursor, counters, last observation) is checkpointed to SQLite after
every processed bar, so killing the process and resuming re-processes
nothing. The risk manager's emergency-stop file remains fully independent
and is never overridden by this controller.
"""

from __future__ import annotations

import contextlib
import json
import logging
import signal
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from fxagent.config import Config
from fxagent.data.live_source import (
    BarSource,
    HistoricalReplaySource,
    YahooHourlySource,
    is_stale,
    market_closed_now,
)
from fxagent.execution.paper import PaperBroker
from fxagent.features.base import build_features, feature_warmup_bars
from fxagent.risk.controls import RiskLimits, RiskManager
from fxagent.simulation.accounting import CostModel
from fxagent.simulation.backtest import FLAT, LONG, SHORT, DecisionContext, PositionState
from fxagent.simulation.exits import ExitRules, check_exits, exit_levels
from fxagent.storage.db import Repository, connect, migrate, session

log = logging.getLogger("experiment")

ACTION_MAP = {"flat": FLAT, "long": LONG, "short": SHORT}
CHECKPOINT_VERSION = 1

MODES = ("historical_replay", "delayed_data_paper")
# current_data_paper intentionally absent — see module docstring.


@dataclass
class ExperimentCheckpoint:
    """Validated on restore; incompatible checkpoints are refused, not guessed."""

    version: int = CHECKPOINT_VERSION
    experiment_id: str = ""
    run_id: str = ""
    mode: str = ""
    bars_processed: int = 0
    orders_placed: int = 0
    orders_rejected: int = 0
    exit_events: int = 0
    last_bar_ts: str = ""
    last_decision: str = "flat"
    daily_realised_pnl: float = 0.0
    peak_equity: float = 0.0
    stale_events: int = 0
    consecutive_errors: int = 0
    started_at: str = ""
    ends_at: str = ""
    data_source: str = ""
    details: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.version != CHECKPOINT_VERSION:
            raise ValueError(
                f"incompatible checkpoint version {self.version} "
                f"(expected {CHECKPOINT_VERSION})"
            )
        if not self.experiment_id or not self.run_id:
            raise ValueError("checkpoint missing experiment_id/run_id")
        if self.mode not in MODES:
            raise ValueError(f"checkpoint mode {self.mode!r} not in {MODES}")


@dataclass
class ExitPolicy:
    """Simulated exits; defaults from config section `exits`."""

    rules: ExitRules

    @classmethod
    def from_config(cls, cfg: Config) -> ExitPolicy:
        return cls(ExitRules.from_config(cfg.get("exits", {}) or {}))


def build_source(cfg: Config, mode: str) -> BarSource:
    if mode == "historical_replay":
        csv_path = str(Path(cfg.path("project.paths.processed_dir")) / "EURUSD_H1.csv")
        return HistoricalReplaySource(csv_path)
    if mode == "delayed_data_paper":
        symbol = f"{cfg.get('instrument.symbol', 'EURUSD')}=X"
        return YahooHourlySource(symbol=symbol)
    raise ValueError(f"unsupported mode {mode!r}; current_data_paper is not available")


class ExperimentController:
    """Orchestrates one paper-trading experiment end to end."""

    def __init__(
        self,
        cfg: Config,
        strategy,
        *,
        mode: str = "delayed_data_paper",
        experiment_id: str | None = None,
        duration_days: float = 7.0,
        poll_seconds: float = 60.0,
        max_age_hours: float = 3.0,
        history_bars: int = 2500,
        exit_policy: ExitPolicy | None = None,
        source: BarSource | None = None,
        flatten_on_stop: bool = True,
    ):
        if mode not in MODES:
            raise ValueError(f"unsupported mode {mode!r}")
        self.cfg = cfg
        self.strategy = strategy
        self.mode = mode
        self.experiment_id = experiment_id or (
            f"exp-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        )
        self.duration_days = float(duration_days)
        self.poll_seconds = float(poll_seconds)
        self.max_age_hours = float(max_age_hours)
        self.history_bars = int(history_bars)
        self.exit_policy = exit_policy or ExitPolicy.from_config(cfg)
        self.source = source or build_source(cfg, mode)
        self.flatten_on_stop = bool(flatten_on_stop)

        self.data_dir = Path(cfg.path("project.paths.raw_dir")).parent
        self.pause_flag = self.data_dir / "EXPERIMENT_PAUSE"
        self.stop_flag = self.data_dir / "EXPERIMENT_STOP"

        cost = CostModel(
            spread_pips=float(cfg.get("costs.spread_pips", 0.8)),
            slippage_pips=float(cfg.get("costs.slippage_pips", 0.2)),
            commission_per_mm=float(cfg.get("costs.commission_per_million", 3.0)),
            pip_size=float(cfg.get("instrument.pip_size", 0.0001)),
        )
        limits = RiskLimits.from_config(cfg.get("risk", {}) or {}, pip_size=cost.pip_size)
        self.risk = RiskManager(limits)
        self.cost = cost
        self.units = int(cfg.get("position.default_units", 100_000))
        self.pip_size = cost.pip_size
        self._ckpt: ExperimentCheckpoint | None = None
        self._broker: PaperBroker | None = None
        self._conn = None
        self._history: pd.DataFrame | None = None
        self._features: pd.DataFrame | None = None
        self._shutdown = False
        self._replay_idx = 0  # cursor into historical frames

    # ---- broker / db ---------------------------------------------------
    def _get_broker(self) -> PaperBroker:
        if self._broker is None:
            self._broker = PaperBroker(
                db_path=self.cfg.db_path,
                cost=self.cost,
                risk=self.risk,
                initial_balance=float(self.cfg.get("paper_broker.initial_balance_usd", 10_000.0)),
                run_id=self._ckpt.run_id if self._ckpt else self.experiment_id,
            )
        return self._broker

    def _ensure_db(self):
        if self._conn is None:
            self._conn = connect(self.cfg.db_path)
            migrate(self._conn)
        return self._conn

    # ---- checkpoint ----------------------------------------------------
    def _ckpt_key(self) -> str:
        return f"experiment_checkpoint:{self.experiment_id}"

    def _load_checkpoint(self) -> ExperimentCheckpoint | None:
        with session(self.cfg.db_path) as conn:
            row = conn.execute("SELECT value FROM kv WHERE key=?", (self._ckpt_key(),)).fetchone()
        if row is None:
            return None
        ckpt = ExperimentCheckpoint(**json.loads(row["value"]))
        ckpt.validate()
        return ckpt

    def _save_checkpoint(self) -> None:
        assert self._ckpt is not None
        payload = json.dumps(asdict(self._ckpt), default=str)
        with session(self.cfg.db_path) as conn:
            conn.execute(
                "INSERT INTO kv(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (self._ckpt_key(), payload),
            )

    # ---- lifecycle -----------------------------------------------------
    def status(self) -> dict:
        ckpt = self._load_checkpoint()
        broker_state = None
        with contextlib.suppress(Exception):
            broker_state = self._get_broker().account_state()
        with session(self.cfg.db_path) as conn:
            exp = Repository(conn).get_experiment(self.experiment_id)
        return {
            "experiment_id": self.experiment_id,
            "mode": self.mode,
            "data_source": self.source.name,
            "checkpoint": asdict(ckpt) if ckpt else None,
            "experiment_row": exp,
            "account": broker_state,
            "pause_flag_present": self.pause_flag.exists(),
            "stop_flag_present": self.stop_flag.exists(),
        }

    def request_stop(self) -> None:
        self.stop_flag.parent.mkdir(parents=True, exist_ok=True)
        self.stop_flag.write_text(datetime.now(UTC).isoformat())
        self._shutdown = True

    def _install_signal_handlers(self) -> None:
        def _handler(signum, frame):  # noqa: ARG001
            log.warning("signal %s received — clean stop", signum)
            self._shutdown = True

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    # ---- history / features -------------------------------------------
    def _seed_history(self) -> None:
        csv_path = Path(self.cfg.path("project.paths.processed_dir")) / "EURUSD_H1.csv"
        frame = pd.read_csv(csv_path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        if self.mode == "delayed_data_paper":
            frame = frame.tail(self.history_bars).reset_index(drop=True)
        self._history = frame[["timestamp", "open", "high", "low", "close"]]
        self._recompute_features()

    def _recompute_features(self) -> None:
        assert self._history is not None
        feats = build_features(self._history, self.cfg.get("features", {}) or {})
        self._features = feats

    def _append_bar(self, bar: pd.Series) -> None:
        assert self._history is not None
        row = pd.DataFrame([{
            "timestamp": pd.Timestamp(bar["timestamp"]),
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
        }])
        self._history = (
            pd.concat([self._history, row], ignore_index=True)
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if len(self._history) > self.history_bars:
            self._history = self._history.tail(self.history_bars).reset_index(drop=True)
        self._recompute_features()

    # ---- single-bar processing ----------------------------------------
    def _process_bar(self, bar: pd.Series) -> None:
        """Idempotent: decision row keyed by run:bar_ts:strategy."""
        assert self._ckpt is not None
        conn = self._ensure_db()
        repo = Repository(conn)
        broker = self._get_broker()
        bar_ts = str(pd.Timestamp(bar["timestamp"]))
        idem = f"{self._ckpt.run_id}:{bar_ts}:{self.strategy.name}"

        dup = conn.execute(
            "SELECT id FROM decisions WHERE idempotency_key=?", (idem,)
        ).fetchone()
        if dup is not None:
            self._ckpt.bars_processed += 1
            self._ckpt.last_bar_ts = bar_ts
            self._save_checkpoint()
            return

        account_prev = broker.account_state(mark_price=float(bar["open"]))
        open_pos = account_prev["open_position"]

        # 1) exits take precedence over any new signal on this bar
        exit_reason = None
        if open_pos is not None:
            pos_row = repo.open_positions(self._ckpt.run_id)
            bars_held = 0
            if pos_row:
                row = conn.execute(
                    "SELECT COALESCE(bars_held, 0) AS bh FROM positions WHERE id=?",
                    (pos_row[0]["id"],),
                ).fetchone()
                bars_held = int(row["bh"]) + 1 if row else 1
                conn.execute(
                    "UPDATE positions SET bars_held=? WHERE id=?", (bars_held, pos_row[0]["id"])
                )
                # Release the write lock before the broker's own session writes.
                conn.commit()
                exit_reason = check_exits(
                    side=open_pos["side"],
                    entry_price=open_pos["entry_price"],
                    bar_high=float(bar["high"]),
                    bar_low=float(bar["low"]),
                    bars_held=bars_held,
                    rules=self.exit_policy.rules,
                    pip_size=self.pip_size,
                )

        if exit_reason:
            # Release any open read/write txn on the long-lived connection so
            # the broker's own SQLite session can write (WAL allows one writer).
            conn.commit()
            res = broker.close_position(
                ts=bar_ts,
                mid_price=float(bar["close"]),
                client_order_id=f"{idem}-exit-{exit_reason}",
                exit_reason=exit_reason,
            )
            repo.insert_decision(
                run_id=self._ckpt.run_id, ts=bar_ts, strategy=self.strategy.name,
                action="flat", units=0, risk_verdict="allow",
                executed=res.ok, order_id=res.client_order_id,
                idempotency_key=idem, model_versions=getattr(self.strategy, "model_versions", None),
            )
            self._ckpt.exit_events += 1
            self._ckpt.last_decision = f"exit:{exit_reason}"
            log.info("exit fired at %s: %s", bar_ts, exit_reason)

        else:
            # 2) strategy decision at close of PREVIOUS bar -> fill at this open
            prev_idx = len(self._history) - 2
            if prev_idx >= feature_warmup_bars(self.cfg.get("features", {}) or {}):
                prev_close_ts = self._history["timestamp"].iloc[prev_idx]
                prev_feats = self._features.iloc[prev_idx]
                pos = self._position_state(broker.account_state(mark_price=float(bar["open"])))
                ctx = DecisionContext(
                    index=prev_idx, timestamp=pd.Timestamp(prev_close_ts),
                    features=pd.Series(prev_feats), position=pos, bars_seen=prev_idx,
                )
                action_str = self.strategy.decide(ctx)
            else:
                action_str = "flat"
            target = ACTION_MAP[action_str]
            desired_units = self.units if target == LONG else -self.units if target == SHORT else 0

            executed = False
            order_id = None
            risk_verdict = "allow"
            risk_reasons: list[str] = []
            if desired_units != 0:
                conn.commit()  # release txn before broker's own session writes
                res = broker.submit_market_order(
                    side=target, units=self.units, ts=bar_ts,
                    mid_price=float(bar["open"]), client_order_id=idem,
                    daily_pnl=self._ckpt.daily_realised_pnl,
                    drawdown_pct=self._drawdown_pct(
                        broker.account_state(mark_price=float(bar["close"]))["equity"],
                        self._ckpt,
                    ),
                    bars_stale=0, spread_pips=self.cost.spread_pips,
                )
                executed = res.ok
                order_id = res.client_order_id
                if res.ok:
                    self._ckpt.orders_placed += 1
                    # attach exit levels to the freshly opened position
                    pos_rows = repo.open_positions(self._ckpt.run_id)
                    if pos_rows:
                        stop, target_px = exit_levels(
                            target, float(res.fill_price or bar["open"]),
                            self.exit_policy.rules, self.pip_size,
                        )
                        repo.set_position_exit_levels(
                            pos_rows[0]["id"], stop_price=stop, target_price=target_px,
                            max_hold_bars=self.exit_policy.rules.max_hold_bars or None,
                        )
                else:
                    self._ckpt.orders_rejected += 1
                    risk_verdict = "reject"
                    risk_reasons = [res.reject_reason or "rejected"]

            # entry-bar exit check (filled at this bar's open; rest of bar applies)
            if executed and desired_units != 0 and self.exit_policy.rules.enabled():
                pos_rows = repo.open_positions(self._ckpt.run_id)
                if pos_rows:
                    entry = float(pos_rows[0]["entry_price"])
                    reason = check_exits(
                        side="long" if target == LONG else "short",
                        entry_price=entry, bar_high=float(bar["high"]),
                        bar_low=float(bar["low"]), bars_held=1,
                        rules=self.exit_policy.rules, pip_size=self.pip_size,
                    )
                    if reason:
                        conn.commit()
                        close_res = broker.close_position(
                            ts=bar_ts, mid_price=float(bar["close"]),
                            client_order_id=f"{idem}-exit-{reason}", exit_reason=reason,
                        )
                        if close_res.ok:
                            self._ckpt.exit_events += 1
                            exit_reason = reason

            repo.insert_decision(
                run_id=self._ckpt.run_id, ts=bar_ts, strategy=self.strategy.name,
                action="flat" if exit_reason else action_str,
                units=0 if exit_reason else abs(desired_units),
                supervised_prob=getattr(self.strategy, "last_prob_up", None),
                rl_action=getattr(self.strategy, "last_rl_action", None),
                risk_verdict=risk_verdict, risk_reasons=risk_reasons,
                model_versions=getattr(self.strategy, "model_versions", None),
                executed=executed, order_id=order_id, idempotency_key=idem,
            )
            self._ckpt.last_decision = action_str

        # 3) equity snapshot + counters
        account = broker.account_state(mark_price=float(bar["close"]))
        open_pnl = account["open_position"]["unrealised_pnl"] if account["open_position"] else 0.0
        repo.insert_equity(self._ckpt.run_id, bar_ts, account["balance"], account["equity"], open_pnl)
        self._ckpt.peak_equity = max(self._ckpt.peak_equity, account["equity"])
        self._ckpt.bars_processed += 1
        self._ckpt.last_bar_ts = bar_ts
        conn.commit()
        self._save_checkpoint()
        self._append_bar(bar)

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
    def _drawdown_pct(equity: float, ckpt: ExperimentCheckpoint) -> float:
        if ckpt.peak_equity <= 0:
            return 0.0
        return max(0.0, (ckpt.peak_equity - equity) / ckpt.peak_equity * 100.0)

    # ---- main loop -----------------------------------------------------
    def run(self) -> dict:
        """Start or resume the experiment and run until stop/duration/data-end."""
        self._install_signal_handlers()
        ckpt = self._load_checkpoint()
        if ckpt is None:
            run_id = f"exp-run-{self.experiment_id}"
            now = datetime.now(UTC)
            ckpt = ExperimentCheckpoint(
                experiment_id=self.experiment_id, run_id=run_id, mode=self.mode,
                started_at=now.isoformat(timespec="seconds"),
                ends_at=(now + timedelta(days=self.duration_days)).isoformat(timespec="seconds"),
                data_source=self.source.name,
            )
            self._ckpt = ckpt
            broker = self._get_broker()
            broker.init_balance()
            conn = self._ensure_db()
            repo = Repository(conn)
            if repo.get_run(run_id) is None:
                repo.create_run(
                    run_id, strategy=self.strategy.name, mode=self.mode,
                    model_versions=getattr(self.strategy, "model_versions", None),
                    config_snapshot=self.cfg.data,
                    notes=f"experiment {self.experiment_id}; simulated money only",
                )
            repo.upsert_experiment(
                experiment_id=self.experiment_id, mode=self.mode,
                strategy=self.strategy.name, run_id=run_id, status="running",
                data_source=self.source.name, started_at=ckpt.started_at,
                ends_at=ckpt.ends_at,
            )
            repo.log_experiment_event(self.experiment_id, "started", {"mode": self.mode})
            conn.commit()
        else:
            self._ckpt = ckpt
            log.info(
                "resuming experiment %s (bars_processed=%d, last_bar=%s)",
                self.experiment_id, ckpt.bars_processed, ckpt.last_bar_ts or "none",
            )
            self._broker = PaperBroker(
                db_path=self.cfg.db_path, cost=self.cost, risk=self.risk,
                initial_balance=float(self.cfg.get("paper_broker.initial_balance_usd", 10_000.0)),
                run_id=ckpt.run_id,
            )
            conn = self._ensure_db()
            repo = Repository(conn)
            repo.upsert_experiment(
                experiment_id=self.experiment_id, mode=self.mode,
                strategy=self.strategy.name, run_id=ckpt.run_id, status="resumed",
                data_source=ckpt.data_source or self.source.name,
            )
            repo.log_experiment_event(self.experiment_id, "resumed", {"bars": ckpt.bars_processed})
            conn.commit()

        self._seed_history()
        recon = self._get_broker().reconcile()
        log.info("start reconciliation: %s", recon)

        if self.mode == "historical_replay":
            summary = self._run_historical_replay()
        else:
            summary = self._run_delayed_loop()
        return summary

    # -- mode: historical replay ----------------------------------------
    def _run_historical_replay(self) -> dict:
        assert self._ckpt is not None
        csv_path = Path(self.cfg.path("project.paths.processed_dir")) / "EURUSD_H1.csv"
        frame = pd.read_csv(csv_path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        test_start = pd.Timestamp(self.cfg.get("split.test_start"), tz="UTC")
        frame = frame.loc[frame["timestamp"] >= test_start].reset_index(drop=True)
        total = len(frame)
        start_idx = min(self._replay_idx + int(self._ckpt.bars_processed), total)
        remaining = total - start_idx
        if remaining <= 0:
            return self._finish("data_exhausted")
        # duration_days <= 0 => accelerated replay (no pacing, no deadline).
        # Otherwise pace remaining bars evenly across the wall-clock duration.
        if self.duration_days > 0:
            wall_span = self.duration_days * 86400.0
            bar_sleep = wall_span / remaining
            deadline: float | None = time.monotonic() + wall_span
        else:
            bar_sleep = 0.0
            deadline = None
        log.info(
            "historical_replay: %d bars, pacing %.2fs/bar over %s (NOT live)",
            remaining, bar_sleep, f"{self.duration_days}d" if self.duration_days > 0 else "accelerated",
        )
        for i in range(start_idx, total):
            if self._shutdown or self.stop_flag.exists():
                return self._finish("stopped")
            if deadline is not None and time.monotonic() >= deadline:
                return self._finish("duration_reached")
            while self.pause_flag.exists() and not self._shutdown:
                self._heartbeat("paused")
                time.sleep(min(self.poll_seconds, 30.0))
            if self._shutdown or self.stop_flag.exists():
                return self._finish("stopped")
            self._process_bar(frame.iloc[i])
            if bar_sleep > 0:
                time.sleep(bar_sleep)
        return self._finish("data_exhausted")

    # -- mode: delayed data paper ----------------------------------------
    def _run_delayed_loop(self) -> dict:
        assert self._ckpt is not None
        ends_at = pd.Timestamp(self._ckpt.ends_at)
        consecutive_errors = 0
        while not self._shutdown:
            if self.stop_flag.exists():
                return self._finish("stopped")
            now = pd.Timestamp.now(tz="UTC")
            if now >= ends_at:
                return self._finish("duration_reached")
            while self.pause_flag.exists() and not self._shutdown:
                self._heartbeat("paused")
                time.sleep(min(self.poll_seconds, 30.0))
            if market_closed_now(now):
                self._heartbeat("market_closed")
                time.sleep(self.poll_seconds)
                continue
            try:
                since = pd.Timestamp(self._ckpt.last_bar_ts) if self._ckpt.last_bar_ts else None
                new_bars = self.source.fetch_completed_bars(since=since)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                log.warning("data fetch failed (%d/%d): %s",
                            consecutive_errors, self.risk.limits.max_consecutive_errors, exc)
                self._heartbeat("data_error", details={"error": str(exc)})
                if consecutive_errors >= int(self.risk.limits.max_consecutive_errors):
                    log.error("max consecutive data errors — safe stop")
                    self._heartbeat("safe_stop_data_errors")
                    return self._finish("safe_stop_data_errors")
                time.sleep(self.poll_seconds * min(consecutive_errors, 5))
                continue
            if new_bars.empty:
                if is_stale(
                    pd.Timestamp(self._ckpt.last_bar_ts) if self._ckpt.last_bar_ts else None,
                    max_age_hours=self.max_age_hours, now=now,
                ):
                    self._ckpt.stale_events += 1
                    self._heartbeat("stale_data")
                else:
                    self._heartbeat("waiting_for_bar")
                time.sleep(self.poll_seconds)
                continue
            for _, bar in new_bars.iterrows():
                if self._shutdown or self.stop_flag.exists():
                    return self._finish("stopped")
                self._process_bar(bar)
            self._heartbeat("running")
        return self._finish("stopped")

    # ---- helpers -------------------------------------------------------
    def _heartbeat(self, status: str, details: dict | None = None) -> None:
        assert self._ckpt is not None
        conn = self._ensure_db()
        repo = Repository(conn)
        account = self._get_broker().account_state()
        repo.insert_heartbeat(
            self._ckpt.run_id, self.mode, self._ckpt.last_bar_ts or None,
            self._ckpt.bars_processed, status,
            {"equity": account["equity"], "experiment": self.experiment_id,
             "last_decision": self._ckpt.last_decision, **(details or {})},
        )
        repo.upsert_experiment(
            experiment_id=self.experiment_id, mode=self.mode,
            strategy=self.strategy.name, run_id=self._ckpt.run_id, status=status,
            data_source=self._ckpt.data_source,
            last_observation_ts=self._ckpt.last_bar_ts or None,
        )
        conn.commit()

    def _finish(self, reason: str) -> dict:
        assert self._ckpt is not None
        if self.flatten_on_stop:
            try:
                account = self._get_broker().account_state()
                if account.get("open_position"):
                    price = account["open_position"]["entry_price"]
                    self._get_broker().close_position(
                        ts=self._ckpt.last_bar_ts or datetime.now(UTC).isoformat(),
                        mid_price=price,
                        client_order_id=f"{self._ckpt.run_id}-flatten-{uuid.uuid4().hex[:6]}",
                        exit_reason="experiment_finish",
                    )
                    log.info("flattened open position at experiment end")
            except Exception:
                log.exception("flatten at end failed (kept open)")
        recon = self._get_broker().reconcile()
        account = self._get_broker().account_state()
        conn = self._ensure_db()
        repo = Repository(conn)
        repo.upsert_experiment(
            experiment_id=self.experiment_id, mode=self.mode,
            strategy=self.strategy.name, run_id=self._ckpt.run_id, status=reason,
            data_source=self._ckpt.data_source, ended_at=datetime.now(UTC).isoformat(timespec="seconds"),
            last_observation_ts=self._ckpt.last_bar_ts or None,
        )
        repo.log_experiment_event(self.experiment_id, reason, {"bars": self._ckpt.bars_processed})
        repo.insert_heartbeat(
            self._ckpt.run_id, self.mode, self._ckpt.last_bar_ts or None,
            self._ckpt.bars_processed, reason, {"equity": account["equity"]},
        )
        repo.finish_run(self._ckpt.run_id, status=reason)
        conn.commit()
        summary = {
            "experiment_id": self.experiment_id,
            "mode": self.mode,
            "data_source": self._ckpt.data_source,
            "status": reason,
            "bars_processed": self._ckpt.bars_processed,
            "orders_placed": self._ckpt.orders_placed,
            "orders_rejected": self._ckpt.orders_rejected,
            "exit_events": self._ckpt.exit_events,
            "stale_events": self._ckpt.stale_events,
            "last_bar_ts": self._ckpt.last_bar_ts,
            "final_equity": account["equity"],
            "final_balance": account["balance"],
            "reconcile": recon,
            "note": (
                "SIMULATED MONEY. historical_replay is NOT live observation."
                if self.mode == "historical_replay"
                else "SIMULATED MONEY. delayed_data_paper uses delayed free quotes, not a real-time feed."
            ),
        }
        log.info("experiment finished: %s", json.dumps(summary, indent=2, default=str))
        return summary
