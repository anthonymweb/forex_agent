"""Configurable simulated trade exits: stop-loss, take-profit, time-stop.

Applied inside the runner/backtester BEFORE the strategy decision for the
bar, so an exit always takes precedence over a new signal in the same bar.

Intra-bar worst-case convention: if both stop-loss and take-profit levels
lie inside a bar's [low, high] range, the stop-loss is assumed to fill
first (conservative). Time-stop fires when ``bars_held >= max_hold_bars``.

Exit levels are stored on the open position and persisted so a restart
resumes with the same levels. All levels are computed from the entry fill
price; they are SIMULATED only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExitRules:
    """Exit configuration. A value of 0 (or negative) disables that rule."""

    stop_loss_pips: float = 0.0
    take_profit_pips: float = 0.0
    max_hold_bars: int = 0

    def enabled(self) -> bool:
        return self.stop_loss_pips > 0 or self.take_profit_pips > 0 or self.max_hold_bars > 0

    @classmethod
    def from_config(cls, cfg: dict | None) -> ExitRules:
        cfg = cfg or {}
        return cls(
            stop_loss_pips=float(cfg.get("stop_loss_pips", 0.0) or 0.0),
            take_profit_pips=float(cfg.get("take_profit_pips", 0.0) or 0.0),
            max_hold_bars=int(cfg.get("max_hold_bars", 0) or 0),
        )


def exit_levels(
    side: str, entry_price: float, rules: ExitRules, pip_size: float
) -> tuple[float | None, float | None]:
    """Return (stop_price, target_price) for a position, None if disabled."""
    stop = target = None
    if rules.stop_loss_pips > 0:
        stop = (
            entry_price - rules.stop_loss_pips * pip_size
            if side == "long"
            else entry_price + rules.stop_loss_pips * pip_size
        )
    if rules.take_profit_pips > 0:
        target = (
            entry_price + rules.take_profit_pips * pip_size
            if side == "long"
            else entry_price - rules.take_profit_pips * pip_size
        )
    return stop, target


def check_exits(
    *,
    side: str,
    entry_price: float,
    bar_high: float,
    bar_low: float,
    bars_held: int,
    rules: ExitRules,
    pip_size: float,
) -> str | None:
    """Return exit reason ('stop_loss' | 'take_profit' | 'time_stop') or None.

    ``bars_held`` counts bars since entry; a position opened during bar t
    has bars_held=1 when bar t+1's high/low are evaluated against it.
    """
    if side not in ("long", "short"):
        raise ValueError(f"invalid side: {side}")

    stop, target = exit_levels(side, entry_price, rules, pip_size)

    # Time-stop first: it is unambiguous (no intra-bar ordering assumption).
    if rules.max_hold_bars > 0 and bars_held >= rules.max_hold_bars:
        return "time_stop"

    if stop is not None and target is not None:
        stop_hit = bar_low <= stop if side == "long" else bar_high >= stop
        target_hit = bar_high >= target if side == "long" else bar_low <= target
        if stop_hit:
            return "stop_loss"          # worst-case ordering when both hit
        if target_hit:
            return "take_profit"
        return None

    if stop is not None:
        if side == "long" and bar_low <= stop:
            return "stop_loss"
        if side == "short" and bar_high >= stop:
            return "stop_loss"
    if target is not None:
        if side == "long" and bar_high >= target:
            return "take_profit"
        if side == "short" and bar_low <= target:
            return "take_profit"
    return None
