"""HistData.com free-data client.

Verified working on 2026-10-11 from Ubuntu:

* Year-level pages are HTML-cached and can carry a *stale* ``tk`` token,
  which makes ``get.php`` answer with an empty body.
* Month-level pages mint a fresh token per request, so the client always
  walks month by month with a politeness delay.

Flow per month:
  1. GET  /download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/<pair>/<y>/<m>
  2. parse hidden form fields (tk, date, datemonth, platform, timeframe, fxpair)
  3. POST /get.php  -> ZIP attachment (HISTDATA_COM_ASCII_<PAIR>_M1<YYYYMM>.zip)

Terms: data is free for personal use; HistData asks that heavy users consider
their paid FTP to offset traffic costs. We stay politely rate-limited.
"""

from __future__ import annotations

import io
import logging
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

log = logging.getLogger(__name__)

BASE = "https://www.histdata.com"
PAGE_URL = (
    BASE + "/download-free-forex-historical-data/?/{product}/1-minute-bar-quotes/{pair}/{year}/{month}"
)
TICK_PAGE_URL = (
    BASE + "/download-free-forex-historical-data/?/ascii/tick-data-quotes/{pair}/{year}/{month}"
)
POST_URL = BASE + "/get.php"

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

_FIELD_RE = re.compile(
    r'name="(?P<name>tk|date|datemonth|platform|timeframe|fxpair)"[^>]*value="(?P<value>[^"]*)"'
)


class HistDataError(RuntimeError):
    """Raised when HistData refuses or returns an unusable payload."""


@dataclass(frozen=True)
class MonthSpec:
    year: int
    month: int

    @property
    def key(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.key


def iter_months(start: str, end: str) -> list[MonthSpec]:
    """Inclusive list of months from ``YYYY-MM`` to ``YYYY-MM``."""
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    months: list[MonthSpec] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(MonthSpec(y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def parse_form(html: str) -> dict[str, str]:
    fields = {m.group("name"): m.group("value") for m in _FIELD_RE.finditer(html)}
    required = {"tk", "date", "datemonth", "platform", "timeframe", "fxpair"}
    missing = required - fields.keys()
    if missing:
        raise HistDataError(f"download form not found (missing: {sorted(missing)})")
    if not fields.get("tk"):
        # Server-side page cache sometimes serves the download page with the
        # token stripped. Those entries expire; retrying later usually works.
        raise HistDataError("stale_cache_empty_token")
    return fields


class HistDataClient:
    def __init__(
        self,
        raw_dir: str | Path,
        pair: str = "EURUSD",
        delay_seconds: float = 1.2,
        timeout: float | tuple[float, float] = (10.0, 600.0),
        session: requests.Session | None = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.pair = pair.upper()
        self.delay = max(0.5, float(delay_seconds))
        self.timeout = timeout
        self.http = session or requests.Session()
        self.http.headers.update({"User-Agent": UA})
        self._last_request = 0.0

    # ---- low level ----------------------------------------------------
    def _throttle(self) -> None:
        wait = self.delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _get(self, url: str, **kw) -> requests.Response:
        self._throttle()
        resp = self.http.get(url, timeout=self.timeout, **kw)
        resp.raise_for_status()
        return resp

    # ---- month download ----------------------------------------------
    def _post_form(self, page: str, fields: dict[str, str]) -> bytes:
        payload: bytes | None = None
        last_exc: Exception | None = None
        for attempt in range(3):
            self._throttle()
            try:
                resp = self.http.post(
                    POST_URL,
                    data=fields,
                    headers={
                        "Origin": BASE,
                        "Referer": page,
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "DNT": "1",
                        "Upgrade-Insecure-Requests": "1",
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                payload = resp.content
                break
            except (requests.RequestException, OSError) as exc:
                last_exc = exc
                backoff = 5.0 * (attempt + 1)
                log.warning("download attempt %d failed (%s); retrying in %.0fs",
                            attempt + 1, type(exc).__name__, backoff)
                time.sleep(backoff)
        if payload is None:
            raise HistDataError(f"download failed after retries: {last_exc}")
        if not payload.startswith(b"PK"):
            raise HistDataError(
                f"empty/invalid archive ({fields.get('datemonth')}) bytes={len(payload)}"
            )
        return payload

    def fetch_month(self, month: MonthSpec) -> bytes:
        """Return the raw M1 ZIP bytes for one month (raises HistDataError)."""
        page = PAGE_URL.format(
            product="ascii", pair=self.pair.lower(), year=month.year, month=month.month
        )
        html = self._get(page).text
        fields = parse_form(html)
        expected = f"{month.year}{month.month:02d}"
        if fields.get("datemonth") != expected:
            raise HistDataError(
                f"stale/cached page for {month.key}: datemonth={fields.get('datemonth')}"
            )
        return self._post_form(page, fields)

    def fetch_tick_month(self, month: MonthSpec) -> bytes:
        """Return raw ASCII tick-data ZIP bytes for one month.

        Used as a fallback when the M1 page's cached entry has a stripped
        token. Tick data includes bid AND ask, and its page cache is separate
        from the M1 page cache.
        """
        page = TICK_PAGE_URL.format(pair=self.pair.lower(), year=month.year, month=month.month)
        html = self._get(page).text
        fields = parse_form(html)
        expected = f"{month.year}{month.month:02d}"
        if fields.get("datemonth") != expected:
            raise HistDataError(
                f"stale/cached tick page for {month.key}: datemonth={fields.get('datemonth')}"
            )
        return self._post_form(page, fields)

    def download_month(self, month: MonthSpec, force: bool = False) -> Path | None:
        """Download one M1 month into raw_dir; skip when already present."""
        target = self.raw_dir / self.pair / "M1" / f"{month.year:04d}" / f"{month.month:02d}"
        target.mkdir(parents=True, exist_ok=True)
        existing = list(target.glob("*.zip"))
        if existing and not force:
            log.info("skip %s (already present)", month.key)
            return existing[0]
        payload = self.fetch_month(month)
        name = f"HISTDATA_COM_ASCII_{self.pair}_M1{month.year:04d}{month.month:02d}.zip"
        path = target / name
        path.write_bytes(payload)
        log.info("saved %s (%.1f KB)", path.name, len(payload) / 1024)
        return path

    def download_tick_month(self, month: MonthSpec, force: bool = False) -> Path | None:
        """Download one ASCII tick month (bid+ask) into raw_dir."""
        target = self.raw_dir / self.pair / "T" / f"{month.year:04d}" / f"{month.month:02d}"
        target.mkdir(parents=True, exist_ok=True)
        existing = list(target.glob("*.zip"))
        if existing and not force:
            log.info("skip tick %s (already present)", month.key)
            return existing[0]
        payload = self.fetch_tick_month(month)
        name = f"HISTDATA_COM_ASCII_{self.pair}_T{month.year:04d}{month.month:02d}.zip"
        path = target / name
        path.write_bytes(payload)
        log.info("saved %s (%.1f MB)", path.name, len(payload) / 1024 / 1024)
        return path

    def missing_m1_months(self, start: str, end: str) -> list[MonthSpec]:
        out = []
        for month in iter_months(start, end):
            target = self.raw_dir / self.pair / "M1" / f"{month.year:04d}" / f"{month.month:02d}"
            if not list(target.glob("*.zip")):
                out.append(month)
        return out

    def download_range(
        self,
        start: str,
        end: str,
        force: bool = False,
        rounds: int = 3,
        round_delay: float = 60.0,
        tick_fallback: bool = True,
    ) -> list[Path]:
        """Download every month in [start, end].

        HistData's page cache sometimes serves M1 download pages with the
        token stripped. Those months are retried, and — if still failing —
        recovered through the ASCII *tick data* product (bid+ask), whose page
        cache is separate. Tick months are aggregated to H1 at build time.
        """
        import time as _time

        pending = iter_months(start, end)
        out: list[Path] = []
        for rnd in range(1, max(1, rounds) + 1):
            if not pending:
                break
            still: list[MonthSpec] = []
            log.info("M1 download round %d/%d (%d months pending)", rnd, rounds, len(pending))
            for month in pending:
                try:
                    p = self.download_month(month, force=force)
                    if p:
                        out.append(p)
                except HistDataError as exc:
                    if "stale_cache_empty_token" in str(exc):
                        still.append(month)
                        log.debug("cached M1 page without token for %s; will retry", month.key)
                    else:
                        log.warning("skip %s: %s", month.key, exc)
            pending = still
            if pending and rnd < rounds:
                log.info("waiting %.0fs before retrying %d months", round_delay, len(pending))
                _time.sleep(round_delay)

        if tick_fallback and pending:
            log.info("M1 fallback: downloading ASCII tick data for %d months", len(pending))
            for month in pending:
                try:
                    p = self.download_tick_month(month, force=force)
                    if p:
                        out.append(p)
                except HistDataError as exc:
                    log.warning("tick fallback failed for %s: %s", month.key, exc)
        elif pending:
            log.warning("%d months still unavailable after %d rounds", len(pending), rounds)
        return out


    # ---- extraction ---------------------------------------------------
    def extract_csvs(self, zip_paths: list[Path], csv_dir: Path) -> list[Path]:
        """Unzip all members to ``csv_dir``; returns written CSV paths."""
        csv_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for zp in zip_paths:
            with zipfile.ZipFile(zp) as zf:
                for name in zf.namelist():
                    if not name.lower().endswith(".csv"):
                        continue
                    data = zf.read(name)
                    out = csv_dir / Path(name).name
                    if out.exists() and out.stat().st_size == len(data):
                        written.append(out)
                        continue
                    out.write_bytes(data)
                    written.append(out)
        return written


def retrieve_utc_now() -> str:  # pragma: no cover - helper
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_m1_csv(path_or_buf) -> pd.DataFrame:
    """Read a HistData ASCII M1 CSV.

    Format: ``YYYYMMDD HHMMSS;OPEN;HIGH;LOW;CLOSE;VOLUME`` (semicolon, no header).
    Timestamps are treated as UTC.
    """
    import pandas as pd

    df = pd.read_csv(
        path_or_buf,
        sep=";",
        header=None,
        names=["dt", "open", "high", "low", "close", "volume"],
        dtype={"dt": str},
    )
    ts = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S", utc=True)
    df = df.drop(columns=["dt", "volume"])
    df.insert(0, "timestamp", ts)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype("float64")
    return df[["timestamp", "open", "high", "low", "close"]]


def read_tick_csv(path_or_buf) -> pd.DataFrame:
    """Read a HistData ASCII tick CSV.

    Format: ``YYYYMMDD HHMMSSmmm,BID,ASK,VOLUME`` (comma, no header).
    """
    import pandas as pd

    df = pd.read_csv(
        path_or_buf,
        sep=",",
        header=None,
        names=["dt", "bid", "ask", "volume"],
        dtype={"dt": str},
    )
    ts = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S%f", utc=True)
    df = df.drop(columns=["dt", "volume"])
    df.insert(0, "timestamp", ts)
    df["bid"] = df["bid"].astype("float64")
    df["ask"] = df["ask"].astype("float64")
    return df[["timestamp", "bid", "ask"]]


def zip_to_dataframe(path: str | Path):
    """Load every M1 CSV inside a HistData M1 ZIP into one DataFrame."""
    import pandas as pd

    frames = []
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".csv"):
                continue
            with zf.open(name) as fh:
                frames.append(read_m1_csv(io.TextIOWrapper(fh, encoding="utf-8")))
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    return pd.concat(frames, ignore_index=True)


def zip_to_ticks(path: str | Path):
    """Load every tick CSV inside a HistData ASCII tick ZIP into one DataFrame."""
    import pandas as pd

    frames = []
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".csv"):
                continue
            with zf.open(name) as fh:
                frames.append(read_tick_csv(io.TextIOWrapper(fh, encoding="utf-8")))
    if not frames:
        return pd.DataFrame(columns=["timestamp", "bid", "ask"])
    return pd.concat(frames, ignore_index=True)
