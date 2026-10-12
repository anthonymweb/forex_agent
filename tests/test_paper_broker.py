"""Paper broker: lifecycle, idempotency, restart recovery, reconciliation, no-live guard."""

from __future__ import annotations

import pytest

from fxagent.execution.paper import (
    LiveEndpointBlocked,
    PaperBroker,
    assert_not_live,
)
from fxagent.risk.controls import RiskLimits, RiskManager
from fxagent.simulation.accounting import CostModel
from fxagent.storage.db import session

COST = CostModel(spread_pips=0.8, slippage_pips=0.2, commission_per_mm=3.0)


def make_broker(db_path, tmp_path, run_id="t1", units=100_000) -> PaperBroker:
    limits = RiskLimits(
        max_position_units=300_000,
        emergency_stop_file=str(tmp_path / "STOP"),
    )
    broker = PaperBroker(
        db_path=db_path, cost=COST, risk=RiskManager(limits),
        initial_balance=10_000.0, run_id=run_id, units_default=units,
    )
    broker.init_balance()
    return broker


def test_open_long_position(tmp_db, tmp_path):
    b = make_broker(tmp_db, tmp_path)
    res = b.submit_market_order("long", 100_000, "2024-06-03T10:00:00Z", 1.10, client_order_id="o1")
    assert res.ok
    st = b.account_state(mark_price=1.10)
    assert st["open_position"]["side"] == "long"
    assert st["open_position"]["units"] == 100_000
    assert st["balance"] <= 10_000.0  # commission deducted


def test_idempotent_same_client_order_id(tmp_db, tmp_path):
    b = make_broker(tmp_db, tmp_path)
    r1 = b.submit_market_order("long", 100_000, "2024-06-03T10:00:00Z", 1.10, client_order_id="dup")
    r2 = b.submit_market_order("long", 100_000, "2024-06-03T10:00:00Z", 1.10, client_order_id="dup")
    assert r1.ok and r2.ok
    assert r1.order_id == r2.order_id
    st = b.account_state()
    assert st["open_position"]["units"] == 100_000  # not doubled


def test_close_position(tmp_db, tmp_path):
    b = make_broker(tmp_db, tmp_path)
    b.submit_market_order("long", 100_000, "2024-06-03T10:00:00Z", 1.10, client_order_id="o1")
    res = b.close_position("2024-06-03T11:00:00Z", 1.1010, client_order_id="c1")
    assert res.ok
    st = b.account_state(mark_price=1.1010)
    assert st["open_position"] is None
    assert st["balance"] > 10_000.0  # profitable round trip, costs small


def test_reject_when_risk_breached(tmp_db, tmp_path):
    limits = RiskLimits(
        max_position_units=50_000, emergency_stop_file=str(tmp_path / "STOP")
    )
    b = PaperBroker(db_path=tmp_db, cost=COST, risk=RiskManager(limits),
                    initial_balance=10_000.0, run_id="t2")
    b.init_balance()
    res = b.submit_market_order("long", 100_000, "2024-06-03T10:00:00Z", 1.10, client_order_id="big")
    assert not res.ok
    assert res.status == "rejected"
    assert "max_position_units" in (res.reject_reason or "")
    assert b.account_state()["open_position"] is None


def test_restart_recovery(tmp_db, tmp_path):
    """Fresh broker instance on the same DB recovers the open position."""
    b1 = make_broker(tmp_db, tmp_path, run_id="before")
    b1.submit_market_order("long", 100_000, "2024-06-03T10:00:00Z", 1.10, client_order_id="o1")
    b2 = make_broker(tmp_db, tmp_path, run_id="after")
    st = b2.account_state(mark_price=1.10)
    assert st["open_position"] is not None
    assert st["open_position"]["side"] == "long"
    # closing from the new instance works
    res = b2.close_position("2024-06-03T12:00:00Z", 1.1020, client_order_id="c-after")
    assert res.ok


def test_reconcile_reports_consistent(tmp_db, tmp_path):
    b = make_broker(tmp_db, tmp_path)
    b.submit_market_order("short", 100_000, "2024-06-03T10:00:00Z", 1.10, client_order_id="s1")
    rec = b.reconcile()
    assert rec["balanced"] is True
    assert rec["open_positions"] <= 1


def test_orders_are_persisted(tmp_db, tmp_path):
    b = make_broker(tmp_db, tmp_path)
    b.submit_market_order("long", 100_000, "2024-06-03T10:00:00Z", 1.10, client_order_id="p1")
    with session(tmp_db) as conn:
        rows = conn.execute("SELECT * FROM orders WHERE client_order_id='p1'").fetchall()
        fills = conn.execute("SELECT * FROM fills").fetchall()
    assert len(rows) == 1
    assert len(fills) == 1


def test_assert_not_live_blocks_live_endpoints():
    for bad in ("live", "real", "production", "oanda-live", "LIVE"):
        with pytest.raises(LiveEndpointBlocked):
            assert_not_live(bad)


def test_guard_allows_paper_paths(tmp_db):
    assert_not_live(str(tmp_db))
    assert_not_live("data/paper.db")


def test_close_without_position_rejected(tmp_db, tmp_path):
    b = make_broker(tmp_db, tmp_path)
    res = b.close_position("2024-06-03T10:00:00Z", 1.10, client_order_id="cx")
    assert not res.ok
    assert res.reject_reason == "no_open_position"


def test_flip_long_to_short_netting(tmp_db, tmp_path):
    b = make_broker(tmp_db, tmp_path)
    b.submit_market_order("long", 100_000, "2024-06-03T10:00:00Z", 1.10, client_order_id="f1")
    res = b.submit_market_order("short", 100_000, "2024-06-03T11:00:00Z", 1.10, client_order_id="f2")
    assert res.ok
    st = b.account_state()
    assert st["open_position"]["side"] == "short"
