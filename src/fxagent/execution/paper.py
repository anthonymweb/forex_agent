"""Execution layer: provider-independent interface + local paper broker.

SAFETY: the application refuses live endpoints. Only the local paper broker
is available in this release; a real broker-DEMO adapter may be added after
the provider's documentation, eligibility and demo access are verified.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from fxagent.risk.controls import RiskContext, RiskDecision, RiskManager
from fxagent.simulation.accounting import CostModel, position_pnl
from fxagent.storage.db import Repository, session

log = logging.getLogger(__name__)

LIVE_HOST_MARKERS = ("live", "api-live", "live-", "-live", "real", "production")


class LiveEndpointBlocked(RuntimeError):
    """Raised when code attempts to configure a live trading endpoint."""


def assert_not_live(endpoint: str) -> None:
    """Refuse anything that looks like a live/real-money endpoint."""
    lowered = str(endpoint).lower()
    if not lowered:
        return
    for marker in LIVE_HOST_MARKERS:
        if marker in lowered:
            raise LiveEndpointBlocked(
                f"endpoint {endpoint!r} looks like a LIVE environment; "
                "fxagent only supports simulated/demo execution"
            )


@dataclass
class OrderResult:
    ok: bool
    client_order_id: str
    status: str
    fill_price: float | None = None
    reject_reason: str | None = None
    position_id: int | None = None
    order_id: int | None = None


class ExecutionBroker(Protocol):
    mode: str

    def submit_market_order(self, side: str, units: int, ts: str) -> OrderResult: ...

    def close_position(self, ts: str) -> OrderResult: ...

    def account_state(self) -> dict: ...


class PaperBroker:
    """Deterministic local paper broker with SQLite-persisted state.

    Features: simulated balance/positions, fill simulation at provided mid
    prices, idempotent order handling via ``client_order_id``, persistent
    trade history, restart reconciliation and an emergency-stop hook.
    """

    mode = "paper"

    def __init__(
        self,
        db_path: str | Path,
        cost: CostModel,
        risk: RiskManager,
        initial_balance: float = 10_000.0,
        run_id: str = "paper",
        units_default: int = 100_000,
    ):
        assert_not_live(str(db_path))  # path sanity; main guard is below
        self.db_path = Path(db_path)
        self.cost = cost
        self.risk = risk
        self.initial_balance = initial_balance
        self.run_id = run_id
        self.units_default = units_default
        self._pending_mid: dict[str, float] = {}  # side -> mid used for next fill

    # ---- context ------------------------------------------------------
    def _ctx(self, desired_units: int, current_units: int, daily_pnl: float,
             drawdown_pct: float, bars_stale: int = 0, spread_pips: float | None = None) -> RiskContext:
        return RiskContext(
            desired_units=desired_units,
            current_units=current_units,
            current_drawdown_pct=drawdown_pct,
            daily_pnl_usd=daily_pnl,
            bars_since_last_data=bars_stale,
            current_spread_pips=spread_pips if spread_pips is not None else self.cost.spread_pips,
            consecutive_errors=self.risk.consecutive_errors,
            account_state_valid=True,
        )

    # ---- orders -------------------------------------------------------
    def submit_market_order(
        self,
        side: str,
        units: int,
        ts: str,
        mid_price: float,
        *,
        client_order_id: str | None = None,
        daily_pnl: float = 0.0,
        drawdown_pct: float = 0.0,
        bars_stale: int = 0,
        spread_pips: float | None = None,
    ) -> OrderResult:
        if side not in ("long", "short", "close"):
            raise ValueError(f"invalid side: {side}")
        cid = client_order_id or f"{self.run_id}-{uuid.uuid4().hex[:16]}"

        with session(self.db_path) as conn:
            repo = Repository(conn)
            existing = conn.execute(
                "SELECT id, status, fill_price FROM orders WHERE client_order_id=?", (cid,)
            ).fetchone()
            if existing is not None:
                log.info("idempotent replay of order %s", cid)
                return OrderResult(
                    ok=existing["status"] == "filled",
                    client_order_id=cid,
                    status=existing["status"],
                    fill_price=existing["fill_price"],
                    order_id=int(existing["id"]),
                )

            open_pos = self._open_position(conn)
            current_units = self._signed_units(open_pos)

            if side == "close":
                if open_pos is None:
                    oid = repo.insert_order(
                        run_id=self.run_id, client_order_id=cid, side="close",
                        units=0, status="rejected", reject_reason="no_open_position",
                    )
                    self.risk.record_error()
                    return OrderResult(False, cid, "rejected", reject_reason="no_open_position", order_id=oid)
                return self._close_existing(conn, repo, open_pos, ts, mid_price, cid)

            desired = units if side == "long" else -units
            decision = self._approve(repo, desired, current_units, daily_pnl, drawdown_pct, bars_stale, spread_pips)
            if not decision.allowed:
                oid = repo.insert_order(
                    run_id=self.run_id, client_order_id=cid, side=side, units=units,
                    status="rejected", reject_reason=";".join(decision.reasons),
                )
                self.risk.record_error()
                return OrderResult(False, cid, "rejected", reject_reason=";".join(decision.reasons), order_id=oid)

            fill_units = decision.adjusted_units if decision.adjusted_units is not None else desired
            abs_units = abs(fill_units)

            # Close existing opposite position first (netting).
            if open_pos is not None:
                self._close_existing(conn, repo, open_pos, ts, mid_price, f"{cid}-close")
                open_pos = None

            fill_price = self.cost.buy_price(mid_price) if fill_units > 0 else self.cost.sell_price(mid_price)
            commission = self.cost.commission_usd(abs_units) / 2.0
            oid = repo.insert_order(
                run_id=self.run_id, client_order_id=cid, side=side, units=abs_units,
                status="filled", fill_price=fill_price,
            )
            if oid is None:  # pragma: no cover - defensive
                raise RuntimeError("failed to persist order")
            repo.insert_fill(
                order_id=oid, run_id=self.run_id, ts=ts, side=side,
                units=abs_units, price=fill_price, commission=commission,
                slippage=self.cost.slippage_pips * self.cost.pip_size,
            )
            pos_id = repo.insert_position(
                run_id=self.run_id, ts_opened=ts, side=side, units=abs_units,
                entry_price=fill_price, commission=commission, source_order_id=oid,
            )
            balance = self._balance(conn) - commission
            self._set_balance(conn, balance)
            self.risk.record_success()
            return OrderResult(True, cid, "filled", fill_price=fill_price, position_id=pos_id, order_id=oid)

    def close_position(
        self, ts: str, mid_price: float, *, client_order_id: str | None = None,
        exit_reason: str | None = None,
    ) -> OrderResult:
        cid = client_order_id or f"{self.run_id}-close-{uuid.uuid4().hex[:12]}"
        with session(self.db_path) as conn:
            repo = Repository(conn)
            existing = conn.execute(
                "SELECT id, status, fill_price FROM orders WHERE client_order_id=?", (cid,)
            ).fetchone()
            if existing is not None:
                return OrderResult(
                    ok=existing["status"] in ("filled", "rejected"),
                    client_order_id=cid, status=existing["status"],
                    fill_price=existing["fill_price"], order_id=int(existing["id"]),
                )
            open_pos = self._open_position(conn)
            if open_pos is None:
                oid = repo.insert_order(
                    run_id=self.run_id, client_order_id=cid, side="close", units=0,
                    status="rejected", reject_reason="no_open_position",
                )
                return OrderResult(False, cid, "rejected", reject_reason="no_open_position", order_id=oid)
            return self._close_existing(conn, repo, open_pos, ts, mid_price, cid, exit_reason=exit_reason)

    # ---- internals ----------------------------------------------------
    def _approve(self, repo: Repository, desired: int, current: int, daily_pnl: float,
                 drawdown: float, bars_stale: int, spread_pips: float | None) -> RiskDecision:
        ctx = self._ctx(desired, current, daily_pnl, drawdown, bars_stale, spread_pips)
        decision = self.risk.evaluate(ctx)
        if not decision.allowed:
            repo.insert_risk_event(
                self.run_id, "order_gate", ";".join(decision.reasons), "reject_order"
            )
        return decision

    def _close_existing(
        self, conn, repo: Repository, pos_row, ts: str, mid: float, cid: str,
        *, exit_reason: str | None = None,
    ) -> OrderResult:
        side = pos_row["side"]
        units = int(pos_row["units"])
        exit_price = self.cost.sell_price(mid) if side == "long" else self.cost.buy_price(mid)
        commission = self.cost.commission_usd(units) / 2.0
        gross = position_pnl(side, units, float(pos_row["entry_price"]), exit_price)
        net = gross - commission
        repo.close_position(int(pos_row["id"]), ts, exit_price, net, exit_reason=exit_reason)
        oid = repo.insert_order(
            run_id=self.run_id, client_order_id=cid, side="close", units=units,
            status="filled", fill_price=exit_price,
        )
        if oid is not None:
            repo.insert_fill(
                order_id=oid, run_id=self.run_id, ts=ts, side="close", units=units,
                price=exit_price, commission=commission,
                slippage=self.cost.slippage_pips * self.cost.pip_size,
            )
        balance = self._balance(conn) + net
        self._set_balance(conn, balance)
        self.risk.record_success()
        return OrderResult(True, cid, "filled", fill_price=exit_price, position_id=int(pos_row["id"]), order_id=oid)

    @staticmethod
    def _open_position(conn):
        row = conn.execute(
            "SELECT * FROM positions WHERE status='open' ORDER BY ts_opened DESC LIMIT 1"
        ).fetchone()
        return row

    @staticmethod
    def _signed_units(pos_row) -> int:
        if pos_row is None:
            return 0
        return int(pos_row["units"]) if pos_row["side"] == "long" else -int(pos_row["units"])

    @staticmethod
    def _balance(conn) -> float:
        row = conn.execute("SELECT value FROM kv WHERE key='balance'").fetchone()
        if row is None:
            return 0.0
        return float(row["value"])

    @staticmethod
    def _set_balance(conn, value: float) -> None:
        conn.execute(
            "INSERT INTO kv(key, value) VALUES('balance', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(float(value)),),
        )

    # ---- account ------------------------------------------------------
    def account_state(self, mark_price: float | None = None) -> dict:
        with session(self.db_path) as conn:
            balance = self._balance(conn)
            pos = self._open_position(conn)
            open_pnl = 0.0
            position = None
            if pos is not None:
                if mark_price is not None:
                    open_pnl = position_pnl(
                        pos["side"], int(pos["units"]), float(pos["entry_price"]), mark_price
                    )
                position = {
                    "side": pos["side"],
                    "units": int(pos["units"]),
                    "entry_price": float(pos["entry_price"]),
                    "unrealised_pnl": open_pnl,
                    "opened_ts": pos["ts_opened"],
                }
            return {
                "mode": self.mode,
                "balance": balance,
                "equity": balance + open_pnl,
                "open_position": position,
                "emergency_stop": self.risk.stop.engaged,
                "consecutive_errors": self.risk.consecutive_errors,
            }

    def reconcile(self) -> dict:
        """Verify recorded state is consistent; returns a report."""
        with session(self.db_path) as conn:
            open_rows = conn.execute("SELECT * FROM positions WHERE status='open'").fetchall()
            orders = conn.execute("SELECT COUNT(*) AS n FROM orders").fetchone()["n"]
            fills = conn.execute("SELECT COUNT(*) AS n FROM fills").fetchone()["n"]
            # netting invariant: at most ONE open position per run for this instrument
            report = {
                "open_positions": len(open_rows),
                "orders": int(orders),
                "fills": int(fills),
                "balanced": len(open_rows) <= 1,
                "issues": [],
            }
            if len(open_rows) > 1:
                report["issues"].append("multiple_open_positions")
            # every open position must have at least one fill
            for row in open_rows:
                n = conn.execute(
                    "SELECT COUNT(*) AS n FROM fills WHERE run_id=? AND units=?", 
                    (row["run_id"], row["units"]),
                ).fetchone()["n"]
                if n == 0:
                    report["issues"].append(f"open_position_{row['id']}_without_fills")
                    report["balanced"] = False
            return report

    def init_balance(self) -> None:
        with session(self.db_path) as conn:
            row = conn.execute("SELECT value FROM kv WHERE key='balance'").fetchone()
            if row is None:
                self._set_balance(conn, self.initial_balance)


def new_order_id(run_id: str) -> str:
    return f"{run_id}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
