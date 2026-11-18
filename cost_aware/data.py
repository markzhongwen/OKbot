"""Public OKX market-data acquisition and deterministic OHLCV preparation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Callable, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd


USER_AGENT = "cost-aware-okx-research/1.0"
ARCHIVE_URL = (
    "https://static.okx.com/cdn/okex/traderecords/candlesticks/daily/"
    "{compact_date}/{instrument}-candlesticks-{date}.zip"
)
FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate-history"
OHLCV_COLUMNS = ("open_time", "open", "high", "low", "close", "vol")
Opener = Callable[..., object]


@dataclass(frozen=True)
class ArchiveFetchResult:
    files: tuple[Path, ...]
    failures: tuple[dict[str, str], ...]


def select_universe(
    tickers: Iterable[dict[str, object]],
    instruments: Iterable[dict[str, object]],
    *,
    limit: int = 20,
    forced: Sequence[str] = (
        "BTC-USDT-SWAP",
        "ETH-USDT-SWAP",
        "SOL-USDT-SWAP",
    ),
) -> list[str]:
    """Select active USDT swaps by estimated 24-hour quote volume."""
    if limit < len(forced):
        raise ValueError("limit must accommodate all forced instruments")

    active = {
        str(item["instId"])
        for item in instruments
        if item.get("state") == "live"
        and item.get("settleCcy") == "USDT"
        and str(item.get("instId", "")).endswith("-USDT-SWAP")
    }
    missing = [instrument for instrument in forced if instrument not in active]
    if missing:
        raise ValueError(f"forced instruments are not active USDT swaps: {missing}")

    quote_volume: dict[str, float] = {}
    for ticker in tickers:
        instrument = str(ticker.get("instId", ""))
        if instrument not in active:
            continue
        try:
            quote_volume[instrument] = float(ticker["last"]) * float(
                ticker["volCcy24h"]
            )
        except (KeyError, TypeError, ValueError):
            continue

    ranked = sorted(
        (instrument for instrument in active if instrument not in forced),
        key=lambda instrument: (-quote_volume.get(instrument, 0.0), instrument),
    )
    selected = ranked[: limit - len(forced)]
    selected.extend(forced)
    return selected


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate one-minute OHLCV without manufacturing rows."""
    missing = set(OHLCV_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"incomplete OHLCV columns: {sorted(missing)}")

    result = frame.loc[:, OHLCV_COLUMNS].copy()
    result["open_time"] = pd.to_datetime(result["open_time"], utc=True, errors="coerce")
    for column in OHLCV_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if result.loc[:, OHLCV_COLUMNS].isna().any(axis=None):
        raise ValueError("incomplete OHLCV row")
    if result["open_time"].duplicated().any():
        raise ValueError("duplicate OHLCV timestamp")
    if (result[["open", "high", "low", "close"]] <= 0).any(axis=None):
        raise ValueError("OHLC prices must be positive")
    if (result["vol"] < 0).any():
        raise ValueError("volume cannot be negative")

    upper = result[["open", "close", "low"]].max(axis=1)
    lower = result[["open", "close", "high"]].min(axis=1)
    if (result["high"] < upper).any() or (result["low"] > lower).any():
        raise ValueError("OHLC bounds are invalid")

    result = result.sort_values("open_time", kind="stable").reset_index(drop=True)
    if (result["open_time"] != result["open_time"].dt.floor("min")).any():
        raise ValueError("open_time must align to UTC minute boundaries")
    return result


def resample_ohlcv(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Resample 1m OHLCV and discard any target bucket missing source minutes."""
    source = validate_ohlcv(frame)
    target = pd.Timedelta(frequency)
    minute = pd.Timedelta(minutes=1)
    if target < minute or target % minute:
        raise ValueError("frequency must be a whole number of minutes")
    expected = int(target / minute)

    indexed = source.set_index("open_time")
    grouped = indexed.resample(
        frequency, origin="epoch", label="left", closed="left"
    )
    result = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        vol=("vol", "sum"),
    )
    counts = grouped["open"].count()
    result = result.loc[counts == expected].dropna().reset_index()
    return result.loc[:, OHLCV_COLUMNS]


def _request_bytes(
    url: str,
    *,
    opener: Opener,
    timeout: float,
    retries: int,
) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with opener(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as error:
            if error.code == 404 or error.code < 500 or attempt == retries:
                raise
        except (URLError, TimeoutError, OSError):
            if attempt == retries:
                raise
        time.sleep(0.25 * (2**attempt))
    raise RuntimeError("unreachable retry state")


def _archive_frame(payload: bytes) -> pd.DataFrame:
    with ZipFile(BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("daily archive must contain exactly one CSV")
        with archive.open(names[0]) as source:
            raw = pd.read_csv(source)

    if "confirm" in raw.columns:
        raw = raw.loc[pd.to_numeric(raw["confirm"], errors="coerce") == 1]
    if "open_time" not in raw.columns:
        raise ValueError("daily archive is missing open_time")
    numeric_time = pd.to_numeric(raw["open_time"], errors="coerce")
    raw = raw.assign(open_time=pd.to_datetime(numeric_time, unit="ms", utc=True))
    return validate_ohlcv(raw)


def _dates(start_date: date, end_date: date) -> list[date]:
    if end_date < start_date:
        raise ValueError("end_date cannot precede start_date")
    return [
        start_date + timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
    ]


def fetch_daily_archives(
    instruments: Sequence[str],
    start_date: date,
    end_date: date,
    cache_dir: Path | str,
    *,
    opener: Opener = urlopen,
    max_workers: int = 6,
    retries: int = 3,
    timeout: float = 30,
) -> ArchiveFetchResult:
    """Download inclusive daily archives into a validated 1m parquet cache."""
    root = Path(cache_dir) / "raw_1m"
    jobs = [
        (instrument, day)
        for instrument in instruments
        for day in _dates(start_date, end_date)
    ]

    def fetch_one(instrument: str, day: date) -> tuple[Path | None, dict | None]:
        destination = root / instrument / f"{day.isoformat()}.parquet"
        if destination.exists():
            try:
                validate_ohlcv(pd.read_parquet(destination))
                return destination, None
            except (ValueError, OSError):
                destination.unlink(missing_ok=True)

        url = ARCHIVE_URL.format(
            compact_date=day.strftime("%Y%m%d"),
            instrument=instrument,
            date=day.isoformat(),
        )
        try:
            payload = _request_bytes(
                url,
                opener=opener,
                timeout=timeout,
                retries=retries,
            )
            frame = _archive_frame(payload)
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(destination, index=False)
            return destination, None
        except HTTPError as error:
            message = f"HTTP {error.code}"
        except Exception as error:  # one bad daily file must not abort the batch
            message = f"{type(error).__name__}: {error}"
        return None, {
            "instrument": instrument,
            "date": day.isoformat(),
            "error": message,
        }

    files: list[Path] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_one, instrument, day): (instrument, day)
            for instrument, day in jobs
        }
        for future in as_completed(futures):
            file, failure = future.result()
            if file is not None:
                files.append(file)
            if failure is not None:
                failures.append(failure)

    files.sort()
    failures.sort(key=lambda item: (item["instrument"], item["date"]))
    return ArchiveFetchResult(tuple(files), tuple(failures))


def fetch_funding_history(
    instrument: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_path: Path | str,
    *,
    opener: Opener = urlopen,
    page_limit: int = 100,
    retries: int = 3,
    timeout: float = 30,
) -> pd.DataFrame:
    """Fetch realized funding in ``[start, end)`` using OKX cursor pagination."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
    if end <= start:
        raise ValueError("funding end must be after start")

    start_ms = int(start.timestamp() * 1000)
    cursor = int(end.timestamp() * 1000)
    records: list[dict[str, object]] = []
    seen_cursors: set[int] = set()
    while cursor > start_ms and cursor not in seen_cursors:
        seen_cursors.add(cursor)
        query = urlencode(
            {
                "instId": instrument,
                "before": start_ms,
                "after": cursor,
                "limit": page_limit,
            }
        )
        payload = json.loads(
            _request_bytes(
                f"{FUNDING_URL}?{query}",
                opener=opener,
                timeout=timeout,
                retries=retries,
            )
        )
        if payload.get("code") != "0":
            raise RuntimeError(f"OKX funding API error: {payload}")
        page = payload.get("data", [])
        if not page:
            break
        records.extend(page)
        next_cursor = min(int(item["fundingTime"]) for item in page)
        if next_cursor >= cursor:
            raise RuntimeError("OKX funding cursor did not move backwards")
        cursor = next_cursor

    rows = []
    for item in records:
        timestamp = pd.to_datetime(int(item["fundingTime"]), unit="ms", utc=True)
        if not start <= timestamp < end:
            continue
        realized = item.get("realizedRate")
        value = realized if realized not in (None, "") else item.get("fundingRate")
        rows.append(
            {
                "symbol": str(item.get("instId", instrument)),
                "timestamp": timestamp,
                "rate": float(value),
            }
        )
    frame = pd.DataFrame(rows, columns=["symbol", "timestamp", "rate"])
    if not frame.empty:
        frame = (
            frame.drop_duplicates(["symbol", "timestamp"])
            .sort_values(["timestamp", "symbol"], kind="stable")
            .reset_index(drop=True)
        )
    destination = Path(cache_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return frame
