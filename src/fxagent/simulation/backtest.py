"""Bar-by-bar backtesting engine with honest execution timing.

Execution model
---------------
* Decisions are requested at the close of bar ``t`` (the strategy only sees
  information up to and including bar ``t``).
* The resulting order is filled at the OPEN of bar ``t+1`` with spread and
  slippage applied against the trader.
* Equity is marked to market on every bar close.
* At the end of the run any open position is force-closed at the final close
  and reported explicitly (it is never silently ignored).

This engine is deterministic and reproducible given identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from fxagent.simulation.accounting import CostModel, position_pnl, unrealised_pnl

FLAT, LONG, SHORT = "flat", "long", "short"


@dataclass
class PositionState:
    side: str | None = None  # None = flat
    units: int = 0
    entry_price: float = 0.0
    opened_ts: str = ""
    opened_index: int | None = None  # bar index where the fill happened

    @property
    def is_open(self) -> bool:
        return self.side is not None and self.units > 0


@dataclass
class DecisionContext:
    index: int
    timestamp: pd.Timestamp
    features: pd.Series
    position: PositionState
    bars_seen: int

    @property
    def want_long(self) -> bool:
        return self.position.side == LONG

    @property
    def want_short(self) -> bool:
        return self.position.side == SHORT

    @property
    def is_flat(self) -> bool:
        return not self.position.is_open


class Strategy(Protocol):
    name: str

    def decide(self, ctx: DecisionContext) -> str:
        """Return one of 'flat' | 'long' | 'short' (desired end-of-next-bar-hold)."""
        ...


@dataclass
class TradeRecord:
    side: str
    units: int
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    gross_pnl: float
    commission: float
    net_pnl: float
    bars_held: int
    cost_estimate: float


@dataclass
class BacktestResult:
    strategy: str
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]
    decisions: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        m = self.metrics
        return (
            f"[{self.strategy}] net_pnl={m['net_pnl_usd']:.2f} "
            f"ret={m['total_return_pct']:.2f}% dd={m['max_drawdown_pct']:.2f}% "
            f"trades={m['n_trades']} win_rate={m['win_rate_pct']:.1f}%"
        )


def _force_close_cost(cost: CostModel, units: int) -> float:
    return abs(units) * (cost.spread_pips + 2 * cost.slippage_pips) * cost.pip_size / 2 + cost.commission_usd(units)


def run_backtest(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    strategy: Strategy,
    cost: CostModel,
    units: int = 100_000,
    initial_balance: float = 10_000.0,
) -> BacktestResult:
    """Run one strategy over ``bars`` using precomputed causal ``features``.

    ``bars`` and ``features`` must share the same index and timestamps.
    """
    if len(bars) != len(features):
        raise ValueError("bars and features length mismatch")
    if len(bars) < 3:
        raise ValueError("need at least 3 bars")

    ts = bars["timestamp"].to_numpy()
    opens = bars["open"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)

    pos = PositionState()
    balance = initial_balance
    trades: list[TradeRecord] = []
    equity_rows: list[tuple] = []
    decision_rows: list[dict] = []

    feature_rows = features.to_dict(orient="records")

    def close_position(at_index: int, price: float, reason: str) -> None:
        nonlocal balance, pos
        gross = position_pnl(pos.side, pos.units, pos.entry_price, price)
        commission = cost.commission_usd(pos.units) / 2.0  # half at entry, half at exit
        # commission already charged at open; charge the exit half now
        net = gross - commission
        balance += net
        exit_ts = str(pd.Timestamp(ts[at_index]))
        if pos.opened_index is not None:
            entry_idx = pos.opened_index
        else:
            entry_idx = int(np.searchsorted(ts, np.datetime64(pd.Timestamp(pos.opened_ts))))
        trades.append(
            TradeRecord(
                side=pos.side,
                units=pos.units,
                entry_ts=pos.opened_ts,
                exit_ts=exit_ts,
                entry_price=pos.entry_price,
                exit_price=price,
                gross_pnl=gross,
                commission=commission,
                net_pnl=net,
                bars_held=max(0, at_index - entry_idx),
                cost_estimate=cost.round_turn_cost_usd(pos.units, pos.entry_price, price),
            )
        )
        pos = PositionState()

    # Walk bars: decision made on bar i is executed at open of bar i+1.
    for i in range(len(bars) - 1):
        now_ts = pd.Timestamp(ts[i])
        ctx = DecisionContext(
            index=i,
            timestamp=now_ts,
            features=pd.Series(feature_rows[i]),
            position=pos,
            bars_seen=i,
        )
        target = strategy.decide(ctx)
        if target not in (FLAT, LONG, SHORT):
            raise ValueError(f"strategy returned invalid action: {target!r}")

        open_next = float(opens[i + 1])
        entry_ts = str(pd.Timestamp(ts[i + 1]))

        # execute at next open
        if target != pos.side:
            if pos.is_open:
                close_position(i + 1, _effective_exit(price=open_next, side=pos.side, cost=cost), "flip")
            if target != FLAT:
                fill = cost.buy_price(open_next) if target == LONG else cost.sell_price(open_next)
                commission = cost.commission_usd(units) / 2.0
                balance -= commission
                pos = PositionState(
                    side=target, units=units, entry_price=fill,
                    opened_ts=entry_ts, opened_index=i + 1,
                )

        mark = float(closes[i])
        open_pnl = unrealised_pnl(pos.side, pos.units, pos.entry_price, mark) if pos.is_open else 0.0
        equity_rows.append((now_ts, balance, balance + open_pnl, open_pnl))
        decision_rows.append(
            {
                "timestamp": now_ts,
                "action": target,
                "position_after": pos.side or FLAT,
                "balance": balance,
                "equity": balance + open_pnl,
            }
        )

    # Final bar: force-close at last close.
    last_i = len(bars) - 1
    last_close = float(closes[last_i])
    forced_close = False
    if pos.is_open:
        close_position(last_i, _effective_exit(price=last_close, side=pos.side, cost=cost), "end_of_data")
        forced_close = True
    equity_rows.append(
        (pd.Timestamp(ts[last_i]), balance, balance, 0.0)
    )

    equity_df = pd.DataFrame(
        equity_rows, columns=["timestamp", "balance", "equity", "open_position_pnl"]
    )
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    metrics = compute_metrics(equity_df, trades_df, cost, initial_balance, forced_close)
    return BacktestResult(
        strategy=strategy.name,
        equity_curve=equity_df,
        trades=trades_df,
        metrics=metrics,
        decisions=pd.DataFrame(decision_rows),
    )


def _effective_exit(*, price: float, side: str, cost: CostModel) -> float:
    """Closing a long hits the bid; closing a short hits the ask, plus slippage."""
    if side == LONG:
        return cost.sell_price(price)
    return cost.buy_price(price)


def compute_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    cost: CostModel,
    initial_balance: float,
    forced_close: bool = False,
) -> dict[str, Any]:
    """Full metric block required by the evaluation spec."""
    end_equity = float(equity["equity"].iloc[-1])
    net_pnl = end_equity - initial_balance
    total_return_pct = net_pnl / initial_balance * 100.0

    # Drawdown on marked equity
    eq = equity["equity"]
    peak = eq.cummax()
    dd = (eq - peak) / peak.replace(0, np.nan)
    max_dd_pct = float(dd.min() * 100.0) if len(dd) else 0.0

    n_trades = int(len(trades))
    if n_trades:
        wins = trades.loc[trades["net_pnl"] > 0, "net_pnl"]
        losses = trades.loc[trades["net_pnl"] < 0, "net_pnl"]
        win_rate_pct = float(len(wins) / n_trades * 100.0)
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(-losses.sum()) if len(losses) else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_bars_held = float(trades["bars_held"].mean())
        total_costs = float(trades["cost_estimate"].sum())
    else:
        win_rate_pct = avg_win = avg_loss = 0.0
        profit_factor = float("nan")
        avg_bars_held = 0.0
        total_costs = 0.0

    # Exposure: fraction of bars with an open position (approx from decisions)
    exposure_pct = float("nan")
    return {
        "initial_balance": initial_balance,
        "ending_equity": end_equity,
        "net_pnl_usd": net_pnl,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct,
        "n_trades": n_trades,
        "win_rate_pct": win_rate_pct,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "profit_factor": profit_factor,
        "estimated_costs_usd": total_costs,
        "avg_bars_held": avg_bars_held,
        "forced_final_close": forced_close,
        "exposure_pct": exposure_pct,
        "n_bars": int(len(equity)),
        "period_start": str(equity["timestamp"].iloc[0]),
        "period_end": str(equity["timestamp"].iloc[-1]),
    }
