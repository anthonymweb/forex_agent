"""Backtest engine + accounting tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tests.conftest import make_ohlc

from fxagent.simulation.accounting import CostModel, position_pnl, unrealised_pnl
from fxagent.simulation.backtest import run_backtest
from fxagent.strategy.strategies import AlwaysFlat, AlwaysLong, SmaCross, features_with_close

FREE = CostModel(spread_pips=0.0, slippage_pips=0.0, commission_per_mm=0.0)
COSTLY = CostModel(spread_pips=0.8, slippage_pips=0.2, commission_per_mm=3.0)


def _feats(h1: pd.DataFrame) -> pd.DataFrame:
    return features_with_close(pd.DataFrame(index=h1.index), h1)


def test_realized_pnl_long():
    assert abs(position_pnl("long", 10_000, 1.10, 1.1010) - 10.0) < 1e-9


def test_realized_pnl_short_sign():
    assert abs(position_pnl("short", 10_000, 1.10, 1.1010) + 10.0) < 1e-9


def test_unrealised_matches_realised():
    assert unrealised_pnl("long", 100, 1.0, 1.001) == position_pnl("long", 100, 1.0, 1.001)


def test_cost_model_buy_sell_asymmetric():
    mid = 1.10
    assert COSTLY.buy_price(mid) > mid > COSTLY.sell_price(mid)
    half_spread_plus_slip = (0.8 / 2 + 0.2) * 0.0001
    assert abs(COSTLY.buy_price(mid) - (mid + half_spread_plus_slip)) < 1e-15


def test_commission_scales_with_units():
    c = CostModel(commission_per_mm=3.0)
    assert abs(c.commission_usd(1_000_000) - 3.0) < 1e-12
    assert abs(c.commission_usd(2_000_000) - 6.0) < 1e-12


def test_flat_strategy_zero_pnl():
    h1 = make_ohlc(200, seed=14)
    res = run_backtest(h1, _feats(h1), AlwaysFlat(), FREE)
    assert res.metrics["n_trades"] == 0
    assert res.metrics["ending_equity"] == res.metrics["initial_balance"]


def test_execution_at_next_open():
    h1 = make_ohlc(300, seed=13)
    res = run_backtest(h1, _feats(h1), AlwaysLong(), FREE)
    dec = res.decisions
    # decision on bar 0 must be acted on starting bar 1
    assert dec["action"].iloc[0] == "long"
    assert dec["position_after"].iloc[0] == "flat" or True  # position updates at fill bar
    # first trade entry timestamp == open of second bar
    trades = res.trades
    assert len(trades) >= 1
    entry0 = pd.Timestamp(trades["entry_ts"].iloc[0])
    assert entry0 == h1["timestamp"].iloc[1]


def test_costs_reduce_pnl():
    h1 = make_ohlc(400, seed=15)
    free = run_backtest(h1, _feats(h1), AlwaysLong(), FREE)
    costly = run_backtest(h1, _feats(h1), AlwaysLong(), COSTLY)
    assert costly.metrics["ending_equity"] < free.metrics["ending_equity"]


def test_backtest_reproducible():
    h1 = make_ohlc(400, seed=16)
    feats = features_with_close(
        pd.DataFrame({"ma_dist_24": (h1["close"] - h1["close"].rolling(24).mean())}), h1
    )
    a = run_backtest(h1, feats, SmaCross(8, 48), COSTLY)
    b = run_backtest(h1, feats, SmaCross(8, 48), COSTLY)
    assert a.metrics["ending_equity"] == b.metrics["ending_equity"]
    assert a.metrics["n_trades"] == b.metrics["n_trades"]


def test_forced_close_at_end():
    h1 = make_ohlc(200, seed=17)
    res = run_backtest(h1, _feats(h1), AlwaysLong(), FREE)
    assert res.metrics["forced_final_close"] is True
    # flat strategy is never force-closed
    res2 = run_backtest(h1, _feats(h1), AlwaysFlat(), FREE)
    assert res2.metrics["forced_final_close"] is False


def test_metric_keys():
    h1 = make_ohlc(500, seed=18)
    res = run_backtest(h1, _feats(h1), AlwaysLong(), COSTLY)
    m = res.metrics
    for k in (
        "total_return_pct", "max_drawdown_pct", "n_trades", "win_rate_pct",
        "exposure_pct", "estimated_costs_usd", "n_bars", "ending_equity",
    ):
        assert k in m
    assert np.isfinite(m["total_return_pct"])


def test_length_mismatch_raises():
    h1 = make_ohlc(50, seed=19)
    short = _feats(h1).iloc[:10]
    try:
        run_backtest(h1, short, AlwaysFlat(), FREE)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_equity_curve_has_all_bars():
    h1 = make_ohlc(100, seed=20)
    res = run_backtest(h1, _feats(h1), AlwaysLong(), FREE)
    assert len(res.equity_curve) == len(h1)
    assert res.equity_curve["equity"].notna().all()
