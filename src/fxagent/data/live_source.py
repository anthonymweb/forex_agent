"""Market-data source interface for the 7-day paper-trading experiment.

Operating modes (explicit, never silently switched):

* ``historical_replay``   — the local processed H1 dataset (HistData-derived).
* ``delayed_data_paper``  — free delayed hourly observations (Yahoo Finance
                            chart API, ~1-2 min delay, no API key). Personal
                            educational use only; quotes are NOT redistributed.
* ``current_data_paper``  — NOT available in this release. Requires a
                            verified real-time licensed feed; deliberately
                            unimplemented so nothing can masquerade as live.

Every source exposes ``fetch_completed_bars(since)`` which returns ONLY
completed hourly bars (the in-progress hour is always dropped), validated
for OHLC sanity and sorted UTC timestamps.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Protocol

import pandas as pd

log = logging.getLogger(__name__)

BAR_COLUMNS = ["timestamp", "open", "high", "low", "close"]

# Yahoo FX quotes are consolidated mid-ish prices with ~1-2 min delay.
YAHOO_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?interval=1h&range={range_}"
)


class DataSourceError(RuntimeError):
    """Raised when a data source fails after retries."""


class DataSourceStale(DataSourceError):
    """Raised when observations are older than the staleness threshold."""


class BarSource(Protocol):
    name: str
    mode: str  # historical_replay | delayed_data_paper

    def fetch_completed_bars(self, since: pd.Timestamp | None = None) -> pd.DataFrame: ...

    def health(self) -> dict: ...


def _validate_bars(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Sanity-check an OHLC frame; raise DataSourceError on invalid data."""
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise DataSourceError(f"{source}: missing columns {missing}")
    out = df[BAR_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out.empty:
        return out
    if out[BAR_COLUMNS[1:]].isna().any().any():
        raise DataSourceError(f"{source}: NaN prices in feed")
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataSourceError(f"{source}: non-positive prices in feed")
    bad_hl = out["high"] < out["low"]
    if bad_hl.any():
        raise DataSourceError(f"{source}: high < low in {int(bad_hl.sum())} bars")
    bad_ohlc = (out["close"] > out["high"] + 1e-9) | (out["close"] < out["low"] - 1e-9)
    bad_ohlc |= (out["open"] > out["high"] + 1e-9) | (out["open"] < out["low"] - 1e-9)
    if bad_ohlc.any():
        raise DataSourceError(f"{source}: open/close outside [low, high] in {int(bad_ohlc.sum())} bars")
    if out["timestamp"].duplicated().any():
        raise DataSourceError(f"{source}: duplicate timestamps")
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


class YahooHourlySource:
    """Free delayed hourly FX bars from the Yahoo Finance chart API.

    * No API key, no account. Rate limits: be polite (min_interval_seconds).
    * Delay: ~1-2 minutes behind mid-market; the in-progress hour is dropped,
      so the newest bar is the last COMPLETED hour.
    * Price basis: consolidated quote (approximately mid). Bid/ask are not
      provided — the configured spread assumption still applies at fills.
    * Terms: for personal educational use; we do not redistribute the data.
    """

    name = "yahoo_hourly"
    mode = "delayed_data_paper"

    def __init__(
        self,
        symbol: str = "EURUSD=X",
        range_: str = "5d",
        timeout_seconds: float = 20.0,
        retries: int = 3,
        backoff_seconds: float = 5.0,
        min_interval_seconds: float = 2.0,
    ):
        self.symbol = symbol
        self.range = range_
        self.timeout = timeout_seconds
        self.retries = retries
        self.backoff = backoff_seconds
        self.min_interval = min_interval_seconds
        self._last_fetch_monotonic = 0.0
        self._last_ok_at: datetime | None = None

    def _get(self) -> dict:
        url = YAHOO_URL.format(symbol=self.symbol, range_=self.range)
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_fetch_monotonic)
            if wait > 0:
                time.sleep(wait)
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0 (educational research; contact: local)"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    self._last_fetch_monotonic = time.monotonic()
                    payload = json.loads(resp.read().decode("utf-8"))
                chart = payload.get("chart") or {}
                if chart.get("error"):
                    raise DataSourceError(f"yahoo error: {chart['error']}")
                results = chart.get("result") or []
                if not results:
                    raise DataSourceError("yahoo: empty result")
                return results[0]
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, DataSourceError) as exc:
                last_exc = exc
                sleep_for = self.backoff * attempt
                log.warning("yahoo fetch attempt %d failed (%s); retry in %.1fs", attempt, exc, sleep_for)
                time.sleep(sleep_for)
        raise DataSourceError(f"yahoo fetch failed after {self.retries} attempts: {last_exc}")

    def fetch_completed_bars(self, since: pd.Timestamp | None = None) -> pd.DataFrame:
        result = self._get()
        ts = result.get("timestamp") or []
        if not ts:
            return pd.DataFrame(columns=BAR_COLUMNS)
        quote = (result.get("indicators") or {}).get("quote") or [{}]
        q = quote[0]
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts, unit="s", utc=True),
                "open": q.get("open"),
                "high": q.get("high"),
                "low": q.get("low"),
                "close": q.get("close"),
            }
        )
        # Drop the in-progress hour: a bar is completed once its hour has ended.
        now = pd.Timestamp.now(tz="UTC")
        hour_end = frame["timestamp"].dt.floor("1h") + pd.Timedelta(hours=1)
        frame = frame.loc[hour_end <= now]
        frame = _validate_bars(frame, self.name)
        if since is not None:
            since_ts = pd.Timestamp(since)
            if since_ts.tzinfo is None:
                since_ts = since_ts.tz_localize("UTC")
            frame = frame.loc[frame["timestamp"] > since_ts].reset_index(drop=True)
        if not frame.empty:
            self._last_ok_at = datetime.now(UTC)
        return frame

    def health(self) -> dict:
        try:
            bars = self.fetch_completed_bars()
            last_ts = bars["timestamp"].max() if not bars.empty else None
            return {
                "source": self.name,
                "mode": self.mode,
                "ok": not bars.empty,
                "last_observation_ts": str(last_ts) if last_ts is not None else None,
                "bars_fetched": int(len(bars)),
                "last_success_at": self._last_ok_at.isoformat() if self._last_ok_at else None,
                "price_basis": "consolidated_quote_approx_mid",
                "delay": "~1-2min; only completed hours returned",
            }
        except Exception as exc:  # health must never raise
            return {"source": self.name, "mode": self.mode, "ok": False, "error": str(exc)}


class HistoricalReplaySource:
    """The local processed dataset — explicit ``historical_replay`` mode."""

    name = "histdata_local"
    mode = "historical_replay"

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._frame: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._frame is None:
            raw = pd.read_csv(self.csv_path)
            self._frame = _validate_bars(raw, self.name)
        return self._frame

    def fetch_completed_bars(self, since: pd.Timestamp | None = None) -> pd.DataFrame:
        frame = self._load()
        if since is None:
            return frame.copy()
        since_ts = pd.Timestamp(since)
        if since_ts.tzinfo is None:
            since_ts = since_ts.tz_localize("UTC")
        return frame.loc[frame["timestamp"] > since_ts].reset_index(drop=True)

    def health(self) -> dict:
        frame = self._load()
        return {
            "source": self.name,
            "mode": self.mode,
            "ok": len(frame) > 0,
            "last_observation_ts": str(frame["timestamp"].max()) if len(frame) else None,
            "bars_fetched": int(len(frame)),
            "last_success_at": datetime.now(UTC).isoformat(),
            "price_basis": "bid (ask synthesized via configured spread)",
            "delay": "historical — NOT live market observation",
        }


def staleness_seconds(last_ts: pd.Timestamp | None, now: pd.Timestamp | None = None) -> float | None:
    if last_ts is None:
        return None
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    last = pd.Timestamp(last_ts)
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    return max(0.0, (now - last).total_seconds())


def is_stale(
    last_ts: pd.Timestamp | None,
    *,
    max_age_hours: float = 3.0,
    now: pd.Timestamp | None = None,
) -> bool:
    """True when no fresh completed bar has arrived within ``max_age_hours``.

    Callers should treat stale data as NO-TRADE, never as a crash.
    """
    age = staleness_seconds(last_ts, now)
    if age is None:
        return True
    return age > max_age_hours * 3600.0


def market_closed_now(now: pd.Timestamp | None = None) -> bool:
    """Weekend check for FX (closed Fri 22:00 UTC -> Sun 22:00 UTC).

    Approximation used only to skip no-trade polling, never to fabricate
    bars: Sunday-open and Friday-close hours are treated as open.
    """
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    weekday = now.dayofweek  # Mon=0 .. Sun=6
    hour = now.hour
    if weekday == 5:  # Saturday
        return True
    if weekday == 4 and hour >= 22:  # Friday after 22:00
        return True
    return weekday == 6 and hour < 22  # Sunday before 22:00


def next_bar_due(last_ts: pd.Timestamp | None, now: datetime | None = None) -> bool:
    """True when at least one full hour has elapsed since the last bar."""
    if last_ts is None:
        return True
    now = now or datetime.now(UTC)
    elapsed = now - pd.Timestamp(last_ts).to_pydatetime()
    return elapsed >= timedelta(hours=1)
