"""Risk manager: every limit rejects; emergency stop is absolute."""

from __future__ import annotations

from fxagent.risk.controls import (
    EmergencyStop,
    RiskContext,
    RiskLimits,
    RiskManager,
)


def _ctx(**kw) -> RiskContext:
    base = dict(
        desired_units=100_000,
        current_units=0,
        current_drawdown_pct=0.0,
        daily_pnl_usd=0.0,
        bars_since_last_data=0,
        current_spread_pips=0.8,
        consecutive_errors=0,
        account_state_valid=True,
    )
    base.update(kw)
    return RiskContext(**base)


def _mgr(tmp_path, **limit_kw) -> RiskManager:
    limits = RiskLimits(emergency_stop_file=str(tmp_path / "STOP"), **limit_kw)
    return RiskManager(limits)


def test_baseline_allows(tmp_path):
    m = _mgr(tmp_path)
    d = m.evaluate(_ctx())
    assert d.allowed
    assert d.adjusted_units == 100_000


def test_max_position_units_rejects(tmp_path):
    m = _mgr(tmp_path, max_position_units=50_000)
    d = m.evaluate(_ctx(desired_units=100_000))
    assert not d.allowed
    assert any("max_position_units" in r for r in d.reasons)


def test_daily_loss_rejects(tmp_path):
    m = _mgr(tmp_path, max_daily_loss_usd=200.0)
    d = m.evaluate(_ctx(daily_pnl_usd=-250.0))
    assert not d.allowed
    assert any("max_daily_loss" in r for r in d.reasons)


def test_drawdown_rejects(tmp_path):
    m = _mgr(tmp_path, max_drawdown_pct=10.0)
    d = m.evaluate(_ctx(current_drawdown_pct=12.0))
    assert not d.allowed
    assert any("max_drawdown" in r for r in d.reasons)


def test_stale_data_rejects(tmp_path):
    m = _mgr(tmp_path, stale_data_bars=3)
    d = m.evaluate(_ctx(bars_since_last_data=5))
    assert not d.allowed
    assert any("stale_data" in r for r in d.reasons)


def test_excessive_spread_rejects(tmp_path):
    m = _mgr(tmp_path, max_spread_pips=3.0)
    d = m.evaluate(_ctx(current_spread_pips=4.5))
    assert not d.allowed
    assert any("excessive_spread" in r for r in d.reasons)


def test_repeated_errors_reject(tmp_path):
    m = _mgr(tmp_path, max_consecutive_errors=5)
    for _ in range(5):
        m.record_error()
    d = m.evaluate(_ctx())
    assert not d.allowed
    assert any("repeated_errors" in r for r in d.reasons)


def test_invalid_account_state_rejects(tmp_path):
    m = _mgr(tmp_path)
    d = m.evaluate(_ctx(account_state_valid=False))
    assert not d.allowed
    assert any("invalid_account_state" in r for r in d.reasons)


def test_emergency_stop_rejects_everything(tmp_path):
    m = _mgr(tmp_path)
    m.stop.engage("unit-test")
    assert m.stop.engaged
    d = m.evaluate(_ctx(desired_units=100_000))
    assert not d.allowed
    assert any("emergency_stop" in r for r in d.reasons)
    # even flattening is blocked while stopped? flattening allowed unless other reasons
    flat = m.evaluate(_ctx(desired_units=0))
    # safety: flatten blocked when stop engaged per implementation
    assert not flat.allowed
    m.stop.release()
    assert not m.stop.engaged
    assert m.evaluate(_ctx()).allowed


def test_emergency_stop_file_presence_detected(tmp_path):
    path = tmp_path / "STOP"
    path.write_text("x")
    stop = EmergencyStop(path)
    assert stop.engaged


def test_error_counter_resets_on_success(tmp_path):
    m = _mgr(tmp_path, max_consecutive_errors=5)
    for _ in range(3):
        m.record_error()
    m.record_success()
    assert m.consecutive_errors == 0
    assert m.evaluate(_ctx()).allowed


def test_breach_callback_invoked(tmp_path):
    calls = []

    def on_breach(ctx, decision):
        calls.append(decision)

    limits = RiskLimits(max_spread_pips=1.0, emergency_stop_file=str(tmp_path / "S"))
    m = RiskManager(limits, on_breach=on_breach)
    m.evaluate(_ctx(current_spread_pips=9.0))
    assert len(calls) == 1


def test_from_config_mapping(tmp_path):
    limits = RiskLimits.from_config(
        {"max_position_units": 123, "max_daily_loss_usd": 55.0,
         "emergency_stop_file": str(tmp_path / "X")},
        pip_size=0.0001,
    )
    assert limits.max_position_units == 123
    assert limits.max_daily_loss_usd == 55.0
    assert limits.pip_size == 0.0001


def test_cannot_bypass_with_zero_and_stop(tmp_path):
    """Zero-desired with stop engaged must not be silently allowed."""
    m = _mgr(tmp_path)
    m.stop.engage("kill")
    d = m.evaluate(_ctx(desired_units=0))
    assert not d.allowed
