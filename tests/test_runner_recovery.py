"""Runner recovery: interrupted runs resume without duplicate orders."""

from __future__ import annotations

import pandas as pd
from tests.conftest import make_ohlc

from fxagent.runner.simulation_runner import SimulationRunner
from fxagent.storage.db import Repository, session
from fxagent.strategy.strategies import AlwaysLong, SmaCross, features_with_close


def _cfg_for_runner(tmp_cfg, tmp_path):
    cfg = tmp_cfg
    cfg.ensure_dirs()
    cfg.data.setdefault("runner", {})
    cfg.data["runner"].update({"mode": "replay", "step_seconds": 0, "max_bars": 0})
    cfg.data.setdefault("paper_broker", {})
    cfg.data["paper_broker"]["initial_balance_usd"] = 10_000.0
    cfg.data.setdefault("position", {})
    cfg.data["position"]["default_units"] = 100_000
    cfg.data.setdefault("costs", {})
    cfg.data["costs"].update(
        {"spread_pips": 0.8, "slippage_pips": 0.2, "commission_per_million": 3.0}
    )
    return cfg


def test_replay_run_completes(tmp_cfg, tmp_path):
    cfg = _cfg_for_runner(tmp_cfg, tmp_path)
    h1 = make_ohlc(300, seed=31)
    feats = features_with_close(pd.DataFrame(index=h1.index), h1)
    runner = SimulationRunner(cfg, AlwaysLong(), mode="replay", run_id="r1", step_seconds=0)
    summary = runner.run(h1, feats)
    assert summary["status"] == "completed"
    assert summary["bars_processed"] == len(h1)
    assert summary["reconcile"]["balanced"] is True


def test_resumed_run_has_no_duplicate_orders(tmp_cfg, tmp_path):
    """Simulate a crash after N bars, then resume with same run_id."""
    cfg = _cfg_for_runner(tmp_cfg, tmp_path)
    h1 = make_ohlc(200, seed=32)
    feats = features_with_close(pd.DataFrame(index=h1.index), h1)
    run_id = "crashy"

    # First pass: process only 50 bars (acts like an interrupted run).
    r1 = SimulationRunner(cfg, AlwaysLong(), mode="replay", run_id=run_id,
                          step_seconds=0, max_bars=50)
    s1 = r1.run(h1, feats)
    assert s1["bars_processed"] == 50

    with session(cfg.db_path) as conn:
        orders1 = conn.execute("SELECT COUNT(*) c FROM orders").fetchone()["c"]

    # Second pass: same run_id, full data — resumes from cursor.
    r2 = SimulationRunner(cfg, AlwaysLong(), mode="replay", run_id=run_id, step_seconds=0)
    s2 = r2.run(h1, feats)
    assert s2["status"] == "completed"
    # total bars = 50 (first pass) + 150 resumed = 200 = len(h1)
    assert s2["bars_processed"] == len(h1)

    with session(cfg.db_path) as conn:
        dup = conn.execute(
            "SELECT idempotency_key, COUNT(*) c FROM decisions "
            "GROUP BY idempotency_key HAVING c > 1"
        ).fetchall()
        orders2 = conn.execute("SELECT COUNT(*) c FROM orders").fetchone()["c"]
    assert len(dup) == 0
    # new decisions were added for the resumed range, not re-recorded for bars 0-49
    with session(cfg.db_path) as conn:
        decisions2 = conn.execute("SELECT COUNT(*) c FROM decisions").fetchone()["c"]
    assert decisions2 == len(h1)
    assert orders2 >= orders1


def test_idempotency_keys_unique_per_bar(tmp_cfg, tmp_path):
    cfg = _cfg_for_runner(tmp_cfg, tmp_path)
    h1 = make_ohlc(100, seed=33)
    feats = features_with_close(pd.DataFrame(index=h1.index), h1)
    SimulationRunner(cfg, SmaCross(5, 20), mode="replay", run_id="idem",
                     step_seconds=0, max_bars=40).run(h1, feats)
    SimulationRunner(cfg, SmaCross(5, 20), mode="replay", run_id="idem",
                     step_seconds=0).run(h1, feats)
    with session(cfg.db_path) as conn:
        rows = conn.execute(
            "SELECT COUNT(*) c FROM (SELECT idempotency_key FROM decisions "
            "GROUP BY idempotency_key HAVING COUNT(*) > 1)"
        ).fetchone()
    assert rows["c"] == 0


def test_runner_persists_heartbeats_and_run_row(tmp_cfg, tmp_path):
    cfg = _cfg_for_runner(tmp_cfg, tmp_path)
    h1 = make_ohlc(80, seed=34)
    feats = features_with_close(pd.DataFrame(index=h1.index), h1)
    SimulationRunner(cfg, AlwaysLong(), mode="replay", run_id="hb", step_seconds=0).run(h1, feats)
    with session(cfg.db_path) as conn:
        repo = Repository(conn)
        run = repo.get_run("hb")
        hb = conn.execute("SELECT COUNT(*) c FROM heartbeats WHERE run_id='hb'").fetchone()
        eq = conn.execute("SELECT COUNT(*) c FROM equity_snapshots WHERE run_id='hb'").fetchone()
    assert run is not None
    assert hb["c"] >= 1
    assert eq["c"] == len(h1)  # one equity snapshot per bar


def test_runner_records_decisions(tmp_cfg, tmp_path):
    cfg = _cfg_for_runner(tmp_cfg, tmp_path)
    h1 = make_ohlc(60, seed=35)
    feats = features_with_close(pd.DataFrame(index=h1.index), h1)
    SimulationRunner(cfg, AlwaysLong(), mode="replay", run_id="dec", step_seconds=0).run(h1, feats)
    with session(cfg.db_path) as conn:
        row = conn.execute(
            "SELECT action, risk_verdict FROM decisions WHERE run_id='dec' LIMIT 1"
        ).fetchone()
    assert row["action"] == "long"
    assert row["risk_verdict"] == "allow"
