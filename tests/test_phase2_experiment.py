"""Tests for simulated exits, data-source validation, and the experiment
controller (checkpointing, idempotency, pause/stop flags, staleness).

SYNTHETIC FIXTURES are labelled as such; integration tests against genuine
historical data live in test_end_to_end.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fxagent.data.live_source import (
    BAR_COLUMNS,
    HistoricalReplaySource,
    YahooHourlySource,
    _validate_bars,
    is_stale,
    market_closed_now,
)
from fxagent.simulation.exits import ExitRules, check_exits, exit_levels
from fxagent.storage.db import connect, migrate, session

# ---- exits ---------------------------------------------------------------

class TestExits:
    def test_stop_loss_long(self):
        rules = ExitRules(stop_loss_pips=10)
        assert check_exits(
            side="long", entry_price=1.1000, bar_high=1.1005, bar_low=1.0985,
            bars_held=1, rules=rules, pip_size=0.0001,
        ) == "stop_loss"

    def test_stop_loss_short(self):
        rules = ExitRules(stop_loss_pips=10)
        assert check_exits(
            side="short", entry_price=1.1000, bar_high=1.1015, bar_low=1.0999,
            bars_held=1, rules=rules, pip_size=0.0001,
        ) == "stop_loss"

    def test_take_profit_long(self):
        rules = ExitRules(take_profit_pips=20)
        assert check_exits(
            side="long", entry_price=1.1000, bar_high=1.1021, bar_low=1.1000,
            bars_held=1, rules=rules, pip_size=0.0001,
        ) == "take_profit"

    def test_worst_case_stop_before_target_same_bar(self):
        rules = ExitRules(stop_loss_pips=10, take_profit_pips=10)
        assert check_exits(
            side="long", entry_price=1.1000, bar_high=1.1015, bar_low=1.0985,
            bars_held=1, rules=rules, pip_size=0.0001,
        ) == "stop_loss"

    def test_time_stop(self):
        rules = ExitRules(max_hold_bars=5)
        assert check_exits(
            side="long", entry_price=1.1, bar_high=1.11, bar_low=1.09,
            bars_held=5, rules=rules, pip_size=0.0001,
        ) == "time_stop"
        assert check_exits(
            side="long", entry_price=1.1, bar_high=1.11, bar_low=1.09,
            bars_held=4, rules=rules, pip_size=0.0001,
        ) is None

    def test_no_exit_when_disabled(self):
        rules = ExitRules()
        assert not rules.enabled()
        assert check_exits(
            side="long", entry_price=1.1, bar_high=2.0, bar_low=0.5,
            bars_held=1000, rules=rules, pip_size=0.0001,
        ) is None

    def test_levels_direction(self):
        stop, target = exit_levels("short", 1.1000, ExitRules(10, 20), 0.0001)
        assert stop == pytest.approx(1.1010)
        assert target == pytest.approx(1.0980)

    def test_from_config_and_invalid_side(self):
        rules = ExitRules.from_config({"stop_loss_pips": 12, "take_profit_pips": 24, "max_hold_bars": 48})
        assert rules == ExitRules(12.0, 24.0, 48)
        with pytest.raises(ValueError):
            check_exits(side="buy", entry_price=1, bar_high=1, bar_low=1,
                        bars_held=1, rules=rules, pip_size=0.0001)


# ---- data source validation ---------------------------------------------

def _synthetic_bars(n=5) -> pd.DataFrame:
    """SYNTHETIC fixture: generated OHLC, not market data."""
    base = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    rows = []
    price = 1.10
    for i in range(n):
        rows.append({
            "timestamp": base + pd.Timedelta(hours=i),
            "open": price, "high": price + 0.0005, "low": price - 0.0005,
            "close": price + 0.0002,
        })
        price += 0.0002
    return pd.DataFrame(rows)


class TestValidateBars:
    def test_valid(self):
        out = _validate_bars(_synthetic_bars(), "synthetic")
        assert list(out.columns) == BAR_COLUMNS

    def test_rejects_high_lt_low(self):
        df = _synthetic_bars()
        df.loc[1, "high"] = df.loc[1, "low"] - 0.001
        with pytest.raises(Exception, match="high < low"):
            _validate_bars(df, "synthetic")

    def test_rejects_close_outside_range(self):
        df = _synthetic_bars()
        df.loc[0, "close"] = df.loc[0, "high"] + 0.001
        with pytest.raises(Exception, match="outside"):
            _validate_bars(df, "synthetic")

    def test_rejects_nan_and_duplicates(self):
        df = _synthetic_bars()
        df.loc[2, "close"] = float("nan")
        with pytest.raises(Exception, match="NaN"):
            _validate_bars(df, "synthetic")
        df2 = pd.concat([_synthetic_bars(2), _synthetic_bars(2).head(1)], ignore_index=True)
        with pytest.raises(Exception, match="duplicate"):
            _validate_bars(df2, "synthetic")

    def test_naive_timestamps_are_localized_to_utc(self):
        df = _synthetic_bars()
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        out = _validate_bars(df, "synthetic")
        assert str(out["timestamp"].dtype).startswith("datetime64[ns, UTC]")


class TestStaleness:
    def test_is_stale(self):
        now = pd.Timestamp("2026-01-06 12:00", tz="UTC")
        fresh = now - pd.Timedelta(hours=1)
        old = now - pd.Timedelta(hours=5)
        assert not is_stale(fresh, max_age_hours=3, now=now)
        assert is_stale(old, max_age_hours=3, now=now)
        assert is_stale(None, max_age_hours=3, now=now)

    def test_market_closed_weekend(self):
        sat = pd.Timestamp("2026-01-03 12:00", tz="UTC")  # Saturday
        assert market_closed_now(sat)
        assert market_closed_now(pd.Timestamp("2026-01-02 23:00", tz="UTC"))  # Friday late
        assert market_closed_now(pd.Timestamp("2026-01-04 10:00", tz="UTC"))   # Sunday early
        assert not market_closed_now(pd.Timestamp("2026-01-05 10:00", tz="UTC"))  # Tuesday


class TestHistoricalSource:
    def test_fetch_and_since_filter(self, tmp_path):
        df = _synthetic_bars(10)
        csv = tmp_path / "bars.csv"
        df.to_csv(csv, index=False)
        src = HistoricalReplaySource(str(csv))
        all_bars = src.fetch_completed_bars()
        assert len(all_bars) == 10
        since = df["timestamp"].iloc[6]
        newer = src.fetch_completed_bars(since=since)
        assert len(newer) == 3
        assert newer["timestamp"].min() > since
        health = src.health()
        assert health["ok"] and health["mode"] == "historical_replay"
        assert "NOT live" in health["delay"]


class TestYahooSourceParsing:
    def test_completed_bar_filter(self, monkeypatch):
        """SYNTHETIC API payload: only completed hours must survive."""
        now = pd.Timestamp.now(tz="UTC")
        ts = [
            int((now.floor("1h") - pd.Timedelta(hours=3)).timestamp()),
            int((now.floor("1h") - pd.Timedelta(hours=2)).timestamp()),
            int(now.floor("1h").timestamp()),  # in-progress hour
        ]
        payload = {
            "chart": {"result": [{
                "timestamp": ts,
                "indicators": {"quote": [{
                    "open": [1.1, 1.1, 1.1],
                    "high": [1.11, 1.11, 1.11],
                    "low": [1.09, 1.09, 1.09],
                    "close": [1.105, 1.105, 1.105],
                }]},
            }], "error": None},
        }
        src = YahooHourlySource()
        monkeypatch.setattr(src, "_get", lambda: payload["chart"]["result"][0])
        bars = src.fetch_completed_bars()
        assert len(bars) == 2
        assert bars["timestamp"].max() < now.floor("1h")

    def test_nan_rows_dropped_not_imputed(self, monkeypatch):
        """SYNTHETIC API payload with a null-OHLC row: must be dropped."""
        now = pd.Timestamp.now(tz="UTC")
        ts = [
            int((now.floor("1h") - pd.Timedelta(hours=2)).timestamp()),
            int((now.floor("1h") - pd.Timedelta(hours=1)).timestamp()),
        ]
        result = {
            "timestamp": ts,
            "indicators": {"quote": [{
                "open": [1.1, None],
                "high": [1.11, None],
                "low": [1.09, None],
                "close": [1.105, None],
            }]},
        }
        src = YahooHourlySource()
        monkeypatch.setattr(src, "_get", lambda: result)
        bars = src.fetch_completed_bars()
        assert len(bars) == 1
        assert not bars[["open", "high", "low", "close"]].isna().any().any()

    def test_source_failure_propagates(self, monkeypatch):
        """Retry/backoff lives inside _get; a failing _get must surface an error."""
        src = YahooHourlySource(retries=2, backoff_seconds=0.01)

        def boom():
            raise TimeoutError("synthetic failure")

        monkeypatch.setattr(src, "_get", boom)
        with pytest.raises(Exception, match="synthetic failure"):
            src.fetch_completed_bars()


# ---- schema migration ----------------------------------------------------

class TestMigrationV2:
    def test_positions_exit_columns_and_experiment_tables(self, tmp_path):
        conn = connect(tmp_path / "m.db")
        migrate(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(positions)")}
        assert {"stop_price", "target_price", "max_hold_bars", "bars_held", "exit_reason"} <= cols
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "experiments" in tables and "experiment_events" in tables
        conn.close()

    def test_incompatible_checkpoint_refused(self):
        from fxagent.runner.experiment import ExperimentCheckpoint
        ckpt = ExperimentCheckpoint(version=99, experiment_id="e", run_id="r", mode="delayed_data_paper")
        with pytest.raises(ValueError, match="incompatible checkpoint version"):
            ckpt.validate()
        ckpt2 = ExperimentCheckpoint(version=1, experiment_id="", run_id="r", mode="delayed_data_paper")
        with pytest.raises(ValueError, match="missing"):
            ckpt2.validate()
        ckpt3 = ExperimentCheckpoint(version=1, experiment_id="e", run_id="r", mode="live_money")
        with pytest.raises(ValueError, match="not in"):
            ckpt3.validate()


# ---- experiment controller (synthetic bars, no network) -------------------

@pytest.fixture()
def synthetic_cfg(tmp_path):
    """SYNTHETIC environment: temp DB + temp processed CSV, no models needed."""
    from fxagent.config import Config
    df = _synthetic_bars(60)
    processed = tmp_path / "processed"
    processed.mkdir()
    df.to_csv(processed / "EURUSD_H1.csv", index=False)
    cfg = {
        "project": {"name": "test", "seed": 42, "base_currency": "USD",
                    "paths": {"raw_dir": str(tmp_path / "raw"), "processed_dir": str(processed),
                              "models_dir": str(tmp_path / "models"), "logs_dir": str(tmp_path / "logs"),
                              "reports_dir": str(tmp_path / "reports"), "db_path": str(tmp_path / "t.db")}},
        "instrument": {"symbol": "EURUSD", "pip_size": 0.0001, "point_size": 0.00001},
        "costs": {"spread_pips": 0.8, "slippage_pips": 0.2, "commission_per_million": 3.0},
        "position": {"default_units": 100000, "max_units": 300000},
        "risk": {"max_position_units": 300000, "max_open_positions": 1,
                 "max_daily_loss_usd": 200.0, "max_drawdown_pct": 10.0,
                 "max_spread_pips": 3.0, "stale_data_bars": 3,
                 "max_consecutive_errors": 5,
                 "emergency_stop_file": str(tmp_path / "raw" / "EMERGENCY_STOP")},
        "paper_broker": {"initial_balance_usd": 10000.0},
        "runner": {"heartbeat_seconds": 30},
        "exits": {"stop_loss_pips": 0, "take_profit_pips": 0, "max_hold_bars": 0},
        "split": {"test_start": "2026-01-01"},
        "features": {"return_lags": [1, 2], "vol_windows": [3, 5], "ma_windows": [3, 5],
                     "range_windows": [3], "momentum_windows": [3], "use_time_of_week": False},
    }
    return Config(cfg)


class _AlwaysLong:
    name = "always_long"
    model_versions = {}

    def decide(self, ctx):
        return "long"


class _AlwaysFlat:
    name = "always_flat"
    model_versions = {}

    def decide(self, ctx):
        return "flat"


class TestExperimentController:
    def test_replay_idempotent_and_resumable(self, synthetic_cfg):
        from fxagent.runner.experiment import ExperimentController
        ctl = ExperimentController(
            synthetic_cfg, _AlwaysLong(), mode="historical_replay",
            experiment_id="exp-test-1", duration_days=0.0, poll_seconds=0.01,
        )
        summary = ctl.run()
        assert summary["status"] == "data_exhausted"
        assert summary["bars_processed"] > 0
        assert summary["reconcile"]["balanced"] is True
        # resume: no duplicate decisions
        with session(synthetic_cfg.db_path) as conn:
            n1 = conn.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()["n"]
            idem = conn.execute(
                "SELECT COUNT(DISTINCT idempotency_key) AS n FROM decisions").fetchone()["n"]
        assert n1 == idem
        ctl2 = ExperimentController(
            synthetic_cfg, _AlwaysLong(), mode="historical_replay",
            experiment_id="exp-test-1", duration_days=0.0,
        )
        summary2 = ctl2.run()
        with session(synthetic_cfg.db_path) as conn:
            n2 = conn.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()["n"]
        assert n2 == n1  # nothing reprocessed
        assert summary2["bars_processed"] == summary["bars_processed"]

    def test_stop_flag_halts(self, synthetic_cfg):
        from fxagent.runner.experiment import ExperimentController
        ctl = ExperimentController(
            synthetic_cfg, _AlwaysFlat(), mode="historical_replay",
            experiment_id="exp-stop", duration_days=0.0,
        )
        ctl.request_stop()  # flag already present before loop
        summary = ctl.run()
        assert summary["status"] == "stopped"

    def test_exit_fires_and_persists_reason(self, synthetic_cfg):
        from fxagent.runner.experiment import ExitPolicy, ExperimentController
        from fxagent.simulation.exits import ExitRules
        synthetic_cfg.data["exits"] = {"stop_loss_pips": 1, "take_profit_pips": 0, "max_hold_bars": 0}
        ctl = ExperimentController(
            synthetic_cfg, _AlwaysLong(), mode="historical_replay",
            experiment_id="exp-exit", duration_days=0.0,
            exit_policy=ExitPolicy(ExitRules(stop_loss_pips=1.0)),
        )
        summary = ctl.run()
        assert summary["exit_events"] > 0
        with session(synthetic_cfg.db_path) as conn:
            row = conn.execute(
                "SELECT exit_reason FROM positions WHERE exit_reason IS NOT NULL LIMIT 1"
            ).fetchone()
        assert row is not None and row["exit_reason"] in ("stop_loss", "take_profit", "time_stop", "experiment_finish")

    def test_pause_flag_idles(self, synthetic_cfg):
        import threading

        from fxagent.runner.experiment import ExperimentController
        ctl = ExperimentController(
            synthetic_cfg, _AlwaysFlat(), mode="historical_replay",
            experiment_id="exp-pause", duration_days=0.0,
        )
        ctl.pause_flag.write_text("pause")
        # stop after a short moment so the loop exits
        def _stopper():
            import time as _t
            _t.sleep(0.5)
            ctl.request_stop()
        threading.Thread(target=_stopper, daemon=True).start()
        summary = ctl.run()
        assert summary["status"] == "stopped"
        assert summary["bars_processed"] == 0  # nothing processed while paused

    def test_reconcile_balanced_after_run(self, synthetic_cfg):
        from fxagent.runner.experiment import ExperimentController
        ctl = ExperimentController(
            synthetic_cfg, _AlwaysLong(), mode="historical_replay",
            experiment_id="exp-recon", duration_days=0.0,
        )
        summary = ctl.run()
        rec = summary["reconcile"]
        assert rec["balanced"] and rec["open_positions"] == 0  # flattened at end
