"""Independent risk controls.

The risk manager sits BETWEEN strategies and execution. Model code cannot
disable it: the paper broker refuses any order that does not carry a valid
approval from a RiskManager instance, and approval checks cannot be turned
off by configuration (the emergency stop only ever *tightens* behaviour).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    max_position_units: int = 300_000
    max_open_positions: int = 1
    max_daily_loss_usd: float = 200.0
    max_drawdown_pct: float = 10.0
    max_spread_pips: float = 3.0
    stale_data_bars: int = 3
    max_consecutive_errors: int = 5
    emergency_stop_file: str = "data/EMERGENCY_STOP"
    pip_size: float = 0.0001

    @classmethod
    def from_config(cls, cfg: dict, pip_size: float = 0.0001) -> RiskLimits:
        return cls(
            max_position_units=int(cfg.get("max_position_units", 300_000)),
            max_open_positions=int(cfg.get("max_open_positions", 1)),
            max_daily_loss_usd=float(cfg.get("max_daily_loss_usd", 200.0)),
            max_drawdown_pct=float(cfg.get("max_drawdown_pct", 10.0)),
            max_spread_pips=float(cfg.get("max_spread_pips", 3.0)),
            stale_data_bars=int(cfg.get("stale_data_bars", 3)),
            max_consecutive_errors=int(cfg.get("max_consecutive_errors", 5)),
            emergency_stop_file=str(cfg.get("emergency_stop_file", "data/EMERGENCY_STOP")),
            pip_size=pip_size,
        )


@dataclass
class RiskContext:
    """Snapshot the risk manager evaluates before approving an order."""

    desired_units: int  # signed: +long / -short / 0 flat
    current_units: int  # signed exposure right now
    current_drawdown_pct: float
    daily_pnl_usd: float
    bars_since_last_data: int = 0
    current_spread_pips: float = 0.8
    consecutive_errors: int = 0
    account_state_valid: bool = True


@dataclass
class RiskDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    adjusted_units: int | None = None

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return "ALLOW" if self.allowed else f"REJECT({';'.join(self.reasons)})"


class EmergencyStop:
    """File-flag based global kill switch. Presence of the file == stopped."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def engage(self, reason: str = "manual") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"{datetime.now(UTC).isoformat()} {reason}\n", encoding="utf-8")
        log.critical("EMERGENCY STOP engaged: %s", reason)

    def release(self) -> None:
        if self.path.exists():
            self.path.unlink()
            log.warning("EMERGENCY STOP released")

    @property
    def engaged(self) -> bool:
        return self.path.exists()

    def reason(self) -> str:
        if not self.engaged:
            return ""
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return "stop file present (unreadable)"


class RiskManager:
    """Independent, always-on order gate."""

    def __init__(self, limits: RiskLimits, on_breach=None):
        self.limits = limits
        self.stop = EmergencyStop(limits.emergency_stop_file)
        self._on_breach = on_breach
        self._consecutive_errors = 0

    # ---- error tracking ----------------------------------------------
    def record_error(self) -> None:
        self._consecutive_errors += 1

    def record_success(self) -> None:
        self._consecutive_errors = 0

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors

    # ---- main gate ----------------------------------------------------
    def evaluate(self, ctx: RiskContext) -> RiskDecision:
        limits = self.limits
        reasons: list[str] = []

        if self.stop.engaged:
            reasons.append(f"emergency_stop:{self.stop.reason()}")

        if not ctx.account_state_valid:
            reasons.append("invalid_account_state")

        if ctx.desired_units == 0:
            # Flattening is always allowed (except when stopped? closing is safety)
            if not reasons:
                return RiskDecision(allowed=True, reasons=[])
            return RiskDecision(allowed=False, reasons=reasons)

        if abs(ctx.desired_units) > limits.max_position_units:
            reasons.append(
                f"max_position_units:{abs(ctx.desired_units)}>{limits.max_position_units}"
            )

        # Number of open positions (1 instrument, netted) — treat non-flat as 1.
        open_positions = 0 if ctx.current_units == 0 else 1
        opening_new = ctx.current_units == 0 or (ctx.current_units > 0) != (ctx.desired_units > 0)
        if opening_new and open_positions >= limits.max_open_positions and ctx.current_units != 0:
            # flip: closes then opens — allowed only if flip keeps within max positions
            pass
        if opening_new and open_positions > limits.max_open_positions:
            reasons.append(f"max_open_positions:{open_positions}>{limits.max_open_positions}")

        if ctx.daily_pnl_usd <= -limits.max_daily_loss_usd:
            reasons.append(
                f"max_daily_loss:{ctx.daily_pnl_usd:.2f}<=-{limits.max_daily_loss_usd:.2f}"
            )

        if ctx.current_drawdown_pct >= limits.max_drawdown_pct:
            reasons.append(
                f"max_drawdown:{ctx.current_drawdown_pct:.2f}%>={limits.max_drawdown_pct}%"
            )

        if ctx.bars_since_last_data > limits.stale_data_bars:
            reasons.append(
                f"stale_data:{ctx.bars_since_last_data}>{limits.stale_data_bars} bars"
            )

        if ctx.current_spread_pips > limits.max_spread_pips:
            reasons.append(
                f"excessive_spread:{ctx.current_spread_pips}>{limits.max_spread_pips} pips"
            )

        errors = max(ctx.consecutive_errors, self._consecutive_errors)
        if errors >= limits.max_consecutive_errors:
            reasons.append(f"repeated_errors:{errors}>={limits.max_consecutive_errors}")

        if reasons:
            decision = RiskDecision(allowed=False, reasons=reasons)
            self._notify(ctx, decision)
            return decision

        # Optionally shrink (never enlarge) the order to the cap.
        units = ctx.desired_units
        if abs(units) > limits.max_position_units:
            units = int(np_sign(units) * limits.max_position_units)
        return RiskDecision(allowed=True, adjusted_units=units)

    # ---- helpers ------------------------------------------------------
    def _notify(self, ctx: RiskContext, decision: RiskDecision) -> None:
        log.warning("risk rejection: %s (desired=%s)", ";".join(decision.reasons), ctx.desired_units)
        if self._on_breach:
            try:
                self._on_breach(ctx, decision)
            except Exception:  # pragma: no cover - notifier must never break the gate
                log.exception("risk breach notifier failed")


def np_sign(x: int) -> int:
    return 1 if x > 0 else -1
