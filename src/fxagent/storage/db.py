"""SQLite persistence: connection management, migrations, repositories."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        source TEXT NOT NULL,
        source_url TEXT,
        license TEXT,
        retrieved_at TEXT NOT NULL,
        start_ts TEXT,
        end_ts TEXT,
        n_rows INTEGER,
        checksum TEXT,
        notes TEXT,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        strategy TEXT NOT NULL,
        mode TEXT NOT NULL,
        model_versions TEXT,
        config_snapshot TEXT,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        status TEXT NOT NULL DEFAULT 'running',
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        ts TEXT NOT NULL,
        strategy TEXT NOT NULL,
        action TEXT NOT NULL,
        units INTEGER NOT NULL DEFAULT 0,
        confidence REAL,
        supervised_prob REAL,
        rl_action TEXT,
        risk_verdict TEXT NOT NULL,
        risk_reasons TEXT,
        model_versions TEXT,
        executed INTEGER NOT NULL DEFAULT 0,
        order_id TEXT,
        idempotency_key TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    );

    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        client_order_id TEXT NOT NULL UNIQUE,
        ts_created TEXT NOT NULL,
        ts_updated TEXT NOT NULL,
        side TEXT NOT NULL,
        units INTEGER NOT NULL,
        order_type TEXT NOT NULL DEFAULT 'market',
        status TEXT NOT NULL,
        reject_reason TEXT,
        fill_price REAL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS fills (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER NOT NULL,
        run_id TEXT NOT NULL,
        ts TEXT NOT NULL,
        side TEXT NOT NULL,
        units INTEGER NOT NULL,
        price REAL NOT NULL,
        commission REAL NOT NULL DEFAULT 0,
        slippage REAL NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        ts_opened TEXT NOT NULL,
        ts_closed TEXT,
        side TEXT NOT NULL,
        units INTEGER NOT NULL,
        entry_price REAL NOT NULL,
        exit_price REAL,
        realized_pnl REAL,
        commission REAL NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'open',
        source_order_id INTEGER
    );

    CREATE TABLE IF NOT EXISTS equity_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        ts TEXT NOT NULL,
        balance REAL NOT NULL,
        equity REAL NOT NULL,
        open_position_pnl REAL NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS heartbeats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        ts TEXT NOT NULL,
        mode TEXT NOT NULL,
        last_bar_ts TEXT,
        bars_processed INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        details TEXT
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        ts TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        horizon INTEGER,
        prob_up REAL,
        pred_class INTEGER,
        details TEXT
    );

    CREATE TABLE IF NOT EXISTS risk_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        ts TEXT NOT NULL,
        limit_name TEXT NOT NULL,
        detail TEXT,
        action_taken TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS health_checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        check_name TEXT NOT NULL,
        ok INTEGER NOT NULL,
        details TEXT
    );

    CREATE TABLE IF NOT EXISTS kv (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_decisions_run_ts ON decisions(run_id, ts);
    CREATE INDEX IF NOT EXISTS idx_positions_run ON positions(run_id, status);
    CREATE INDEX IF NOT EXISTS idx_equity_run_ts ON equity_snapshots(run_id, ts);
    CREATE INDEX IF NOT EXISTS idx_heartbeats_run_ts ON heartbeats(run_id, ts);
    CREATE INDEX IF NOT EXISTS idx_orders_client ON orders(client_order_id);
    CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts);
    CREATE INDEX IF NOT EXISTS idx_fills_run ON fills(run_id, ts);
    """,
    2: """
    ALTER TABLE positions ADD COLUMN stop_price REAL;
    ALTER TABLE positions ADD COLUMN target_price REAL;
    ALTER TABLE positions ADD COLUMN max_hold_bars INTEGER;
    ALTER TABLE positions ADD COLUMN opened_bar_index INTEGER;
    ALTER TABLE positions ADD COLUMN bars_held INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE positions ADD COLUMN exit_reason TEXT;

    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        mode TEXT NOT NULL,
        strategy TEXT NOT NULL,
        run_id TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        ends_at TEXT,
        ended_at TEXT,
        data_source TEXT NOT NULL,
        last_observation_ts TEXT,
        details TEXT
    );

    CREATE TABLE IF NOT EXISTS experiment_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT NOT NULL,
        ts TEXT NOT NULL,
        event TEXT NOT NULL,
        details TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_exp_events ON experiment_events(experiment_id, ts);
    """,
}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # WAL + NORMAL is durable at the OS level and avoids a full fsync per
    # commit, which dominates latency on spinning disks (~100ms/commit).
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
    )
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    current = int(row["v"] or 0) if row else 0
    for version in sorted(MIGRATIONS):
        if version > current:
            conn.executescript(MIGRATIONS[version])
            conn.execute("INSERT INTO schema_version(version) VALUES (?)", (version,))
            conn.commit()
            current = version


@contextmanager
def session(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


class Repository:
    """Small typed access layer over the SQLite event store."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---- runs ---------------------------------------------------------
    def create_run(
        self,
        run_id: str,
        strategy: str,
        mode: str,
        model_versions: dict | None = None,
        config_snapshot: dict | None = None,
        notes: str | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO runs(run_id, strategy, mode, model_versions,
               config_snapshot, started_at, status, notes)
               VALUES (?,?,?,?,?,?, 'running', ?)""",
            (
                run_id,
                strategy,
                mode,
                _json(model_versions or {}),
                _json(config_snapshot or {}),
                utc_now_iso(),
                notes,
            ),
        )

    def finish_run(self, run_id: str, status: str = "completed", notes: str | None = None):
        self.conn.execute(
            "UPDATE runs SET ended_at=?, status=?, notes=COALESCE(?, notes) WHERE run_id=?",
            (utc_now_iso(), status, notes, run_id),
        )

    def get_run(self, run_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def latest_run(self) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    # ---- decisions ----------------------------------------------------
    def insert_decision(
        self,
        *,
        run_id: str,
        ts: str,
        strategy: str,
        action: str,
        units: int = 0,
        confidence: float | None = None,
        supervised_prob: float | None = None,
        rl_action: str | None = None,
        risk_verdict: str = "allow",
        risk_reasons: list | dict | None = None,
        model_versions: dict | None = None,
        executed: bool = False,
        order_id: str | None = None,
        idempotency_key: str,
    ) -> bool:
        """Insert a decision. Returns False if the idempotency key already exists."""
        try:
            self.conn.execute(
                """INSERT INTO decisions(run_id, ts, strategy, action, units, confidence,
                   supervised_prob, rl_action, risk_verdict, risk_reasons, model_versions,
                   executed, order_id, idempotency_key, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    ts,
                    strategy,
                    action,
                    units,
                    confidence,
                    supervised_prob,
                    rl_action,
                    risk_verdict,
                    _json(risk_reasons or []),
                    _json(model_versions or {}),
                    int(executed),
                    order_id,
                    idempotency_key,
                    utc_now_iso(),
                ),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    # ---- orders / fills / positions -----------------------------------
    def insert_order(
        self,
        *,
        run_id: str,
        client_order_id: str,
        side: str,
        units: int,
        status: str,
        order_type: str = "market",
        reject_reason: str | None = None,
        fill_price: float | None = None,
    ) -> int | None:
        try:
            cur = self.conn.execute(
                """INSERT INTO orders(run_id, client_order_id, ts_created, ts_updated,
                   side, units, order_type, status, reject_reason, fill_price, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    client_order_id,
                    utc_now_iso(),
                    utc_now_iso(),
                    side,
                    units,
                    order_type,
                    status,
                    reject_reason,
                    fill_price,
                    utc_now_iso(),
                ),
            )
            return int(cur.lastrowid)
        except sqlite3.IntegrityError:
            row = self.conn.execute(
                "SELECT id FROM orders WHERE client_order_id=?", (client_order_id,)
            ).fetchone()
            return int(row["id"]) if row else None

    def insert_fill(
        self,
        *,
        order_id: int,
        run_id: str,
        ts: str,
        side: str,
        units: int,
        price: float,
        commission: float,
        slippage: float,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO fills(order_id, run_id, ts, side, units, price, commission, slippage)
               VALUES (?,?,?,?,?,?,?,?)""",
            (order_id, run_id, ts, side, units, price, commission, slippage),
        )
        return int(cur.lastrowid)

    def insert_position(
        self,
        *,
        run_id: str,
        ts_opened: str,
        side: str,
        units: int,
        entry_price: float,
        commission: float,
        status: str = "open",
        source_order_id: int | None = None,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO positions(run_id, ts_opened, side, units, entry_price,
               commission, status, source_order_id) VALUES (?,?,?,?,?,?,?,?)""",
            (run_id, ts_opened, side, units, entry_price, commission, status, source_order_id),
        )
        return int(cur.lastrowid)

    def close_position(
        self, position_id: int, ts_closed: str, exit_price: float, realized_pnl: float,
        exit_reason: str | None = None,
    ) -> None:
        self.conn.execute(
            """UPDATE positions SET ts_closed=?, exit_price=?, realized_pnl=?, status='closed',
               exit_reason=COALESCE(?, exit_reason) WHERE id=?""",
            (ts_closed, exit_price, realized_pnl, exit_reason, position_id),
        )

    def open_positions(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM positions WHERE run_id=? AND status='open' ORDER BY ts_opened",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- snapshots / heartbeat ----------------------------------------
    def insert_equity(
        self, run_id: str, ts: str, balance: float, equity: float, open_pnl: float = 0.0
    ) -> None:
        self.conn.execute(
            """INSERT INTO equity_snapshots(run_id, ts, balance, equity, open_position_pnl)
               VALUES (?,?,?,?,?)""",
            (run_id, ts, balance, equity, open_pnl),
        )

    def insert_heartbeat(
        self,
        run_id: str,
        mode: str,
        last_bar_ts: str | None,
        bars_processed: int,
        status: str,
        details: dict | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO heartbeats(run_id, ts, mode, last_bar_ts, bars_processed, status, details)
               VALUES (?,?,?,?,?,?,?)""",
            (run_id, utc_now_iso(), mode, last_bar_ts, bars_processed, status, _json(details or {})),
        )

    def insert_risk_event(
        self, run_id: str | None, limit_name: str, detail: str, action_taken: str
    ) -> None:
        self.conn.execute(
            """INSERT INTO risk_events(run_id, ts, limit_name, detail, action_taken)
               VALUES (?,?,?,?,?)""",
            (run_id, utc_now_iso(), limit_name, detail, action_taken),
        )

    def insert_prediction(
        self,
        *,
        run_id: str | None,
        ts: str,
        model_name: str,
        model_version: str,
        horizon: int | None = None,
        prob_up: float | None = None,
        pred_class: int | None = None,
        details: dict | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO predictions(run_id, ts, model_name, model_version, horizon,
               prob_up, pred_class, details) VALUES (?,?,?,?,?,?,?,?)""",
            (run_id, ts, model_name, model_version, horizon, prob_up, pred_class, _json(details or {})),
        )

    def insert_health(self, check_name: str, ok: bool, details: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO health_checks(ts, check_name, ok, details) VALUES (?,?,?,?)",
            (utc_now_iso(), check_name, int(ok), _json(details or {})),
        )

    # ---- exit levels / experiment metadata -----------------------------
    def set_position_exit_levels(
        self,
        position_id: int,
        *,
        stop_price: float | None,
        target_price: float | None,
        max_hold_bars: int | None,
        opened_bar_index: int | None = None,
    ) -> None:
        self.conn.execute(
            """UPDATE positions SET stop_price=?, target_price=?, max_hold_bars=?,
               opened_bar_index=COALESCE(?, opened_bar_index), bars_held=0 WHERE id=?""",
            (stop_price, target_price, max_hold_bars, opened_bar_index, position_id),
        )

    def bump_position_bars_held(self, position_id: int) -> int:
        self.conn.execute(
            "UPDATE positions SET bars_held = bars_held + 1 WHERE id=?", (position_id,)
        )
        row = self.conn.execute(
            "SELECT bars_held FROM positions WHERE id=?", (position_id,)
        ).fetchone()
        return int(row["bars_held"]) if row else 0

    def upsert_experiment(
        self,
        *,
        experiment_id: str,
        mode: str,
        strategy: str,
        run_id: str,
        status: str,
        data_source: str,
        started_at: str | None = None,
        ends_at: str | None = None,
        ended_at: str | None = None,
        last_observation_ts: str | None = None,
        details: dict | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO experiments(experiment_id, mode, strategy, run_id, status,
               started_at, ends_at, ended_at, data_source, last_observation_ts, details)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(experiment_id) DO UPDATE SET
                 status=excluded.status,
                 ends_at=COALESCE(excluded.ends_at, experiments.ends_at),
                 ended_at=COALESCE(excluded.ended_at, experiments.ended_at),
                 last_observation_ts=COALESCE(excluded.last_observation_ts,
                                              experiments.last_observation_ts),
                 details=COALESCE(excluded.details, experiments.details)""",
            (
                experiment_id, mode, strategy, run_id, status,
                started_at or utc_now_iso(), ends_at, ended_at,
                data_source, last_observation_ts, _json(details or {}),
            ),
        )

    def get_experiment(self, experiment_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)
        ).fetchone()
        return dict(row) if row else None

    def latest_experiment(self) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM experiments ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def log_experiment_event(self, experiment_id: str, event: str, details: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO experiment_events(experiment_id, ts, event, details) VALUES (?,?,?,?)",
            (experiment_id, utc_now_iso(), event, _json(details or {})),
        )
