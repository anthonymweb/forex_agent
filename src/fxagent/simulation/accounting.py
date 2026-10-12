"""Trading cost model and P&L accounting.

All amounts are SIMULATED. Costs are applied symmetrically to every strategy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Costs expressed in price units for an instrument quoted like EURUSD.

    spread_pips       : total bid/ask spread; half is paid at each fill
    slippage_pips     : adverse slippage applied per fill (always against the trader)
    commission_per_mm : USD per 1,000,000 base units, round-turn
    pip_size          : instrument pip size (0.0001 for EURUSD)
    """

    spread_pips: float = 0.8
    slippage_pips: float = 0.2
    commission_per_mm: float = 3.0
    pip_size: float = 0.0001

    def buy_price(self, mid: float) -> float:
        return mid + self.spread_pips * self.pip_size / 2 + self.slippage_pips * self.pip_size

    def sell_price(self, mid: float) -> float:
        return mid - self.spread_pips * self.pip_size / 2 - self.slippage_pips * self.pip_size

    def commission_usd(self, units: int) -> float:
        return abs(units) / 1_000_000.0 * self.commission_per_mm

    def round_turn_cost_usd(self, units: int, entry_price: float, exit_price: float) -> float:
        """Estimated total friction for one complete long or short round trip."""
        spread_cost = abs(units) * self.spread_pips * self.pip_size
        slip_cost = abs(units) * 2 * self.slippage_pips * self.pip_size
        return spread_cost + slip_cost + self.commission_usd(units)


def position_pnl(side: str, units: int, entry: float, exit_price: float) -> float:
    """Realised P&L in quote currency (USD for EURUSD) for a completed position."""
    direction = 1 if side == "long" else -1
    return direction * units * (exit_price - entry)


def unrealised_pnl(side: str, units: int, entry: float, mark: float) -> float:
    return position_pnl(side, units, entry, mark)


def pips(price_move: float, pip_size: float = 0.0001) -> float:
    return price_move / pip_size
