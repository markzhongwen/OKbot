from datetime import date
from io import BytesIO
import json
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile

import pandas as pd
import pytest

from cost_aware.data import (
    fetch_daily_archives,
    fetch_funding_history,
    resample_ohlcv,
    select_universe,
    validate_ohlcv,
)
from scripts.fetch_okx_history import complete_days


def candles(minutes: list[int]) -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [f"2026-01-01 00:{minute:02d}:00+00:00" for minute in minutes],
        utc=True,
    )
    return pd.DataFrame(
        {
            "open_time": timestamps,
            "open": [100.0 + minute for minute in minutes],
            "high": [101.0 + minute for minute in minutes],
            "low": [99.0 + minute for minute in minutes],
            "close": [100.5 + minute for minute in minutes],
            "vol": [10.0 + minute for minute in minutes],
        }
    )


def test_select_universe_ranks_active_usdt_swaps_by_quote_volume_and_forces_majors():
    instruments = [
        {"instId": "BTC-USDT-SWAP", "state": "live", "settleCcy": "USDT"},
        {"instId": "ETH-USDT-SWAP", "state": "live", "settleCcy": "USDT"},
        {"instId": "SOL-USDT-SWAP", "state": "live", "settleCcy": "USDT"},
        {"instId": "DOGE-USDT-SWAP", "state": "live", "settleCcy": "USDT"},
        {"instId": "XRP-USDT-SWAP", "state": "suspend", "settleCcy": "USDT"},
        {"instId": "BTC-USDC-SWAP", "state": "live", "settleCcy": "USDC"},
    ]
    tickers = [
        {"instId": "BTC-USDT-SWAP", "last": "100", "volCcy24h": "1"},
        {"instId": "ETH-USDT-SWAP", "last": "10", "volCcy24h": "2"},
        {"instId": "SOL-USDT-SWAP", "last": "5", "volCcy24h": "3"},
        {"instId": "DOGE-USDT-SWAP", "last": "2", "volCcy24h": "1000"},
        {"instId": "XRP-USDT-SWAP", "last": "3", "volCcy24h": "1000"},
        {"instId": "BTC-USDC-SWAP", "last": "100", "volCcy24h": "1000"},
    ]

    selected = select_universe(tickers, instruments, limit=4)

    assert selected == [
        "DOGE-USDT-SWAP",
        "BTC-USDT-SWAP",
        "ETH-USDT-SWAP",
        "SOL-USDT-SWAP",
    ]


def test_validate_ohlcv_normalizes_open_time_to_utc_and_sorts():
    frame = candles([1, 0])
    frame["open_time"] = frame["open_time"].dt.tz_convert("America/New_York")

    validated = validate_ohlcv(frame)

    assert str(validated["open_time"].dt.tz) == "UTC"
    assert validated["open_time"].tolist() == sorted(validated["open_time"].tolist())


def test_validate_ohlcv_rejects_duplicate_timestamps():
    frame = pd.concat([candles([0]), candles([0])], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        validate_ohlcv(frame)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("open", 0, "positive"),
        ("vol", -1, "volume"),
        ("high", 99, "OHLC bounds"),
        ("low", 101, "OHLC bounds"),
        ("close", float("nan"), "incomplete"),
    ],
)
def test_validate_ohlcv_rejects_invalid_rows(column, value, message):
    frame = candles([0])
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_ohlcv(frame)


def test_resample_ohlcv_uses_utc_boundaries_and_ohlcv_aggregation():
    frame = candles([0, 1, 2])

    result = resample_ohlcv(frame, "3min")

    assert result.to_dict("records") == [
        {
            "open_time": pd.Timestamp("2026-01-01 00:00:00+00:00"),
            "open": 100.0,
            "high": 103.0,
            "low": 99.0,
            "close": 102.5,
            "vol": 33.0,
        }
    ]


def test_resample_ohlcv_drops_bucket_with_missing_source_minute_without_filling():
    frame = candles([0, 2, 3, 4, 5])

    result = resample_ohlcv(frame, "3min")

    assert result["open_time"].tolist() == [
        pd.Timestamp("2026-01-01 00:03:00+00:00")
    ]
    assert result.iloc[0]["open"] == 103.0


def test_complete_days_combines_utc_day_split_across_okx_archive_files(tmp_path):
    full_day = pd.DataFrame(
        {
            "open_time": pd.date_range(
                "2026-01-01 00:00:00+00:00", periods=1440, freq="min"
            ),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "vol": 1.0,
        }
    )
    paths = [tmp_path / "archive-a.parquet", tmp_path / "archive-b.parquet"]
    full_day.iloc[:960].to_parquet(paths[0], index=False)
    full_day.iloc[960:].to_parquet(paths[1], index=False)

    assert complete_days(paths) == ["2026-01-01"]


class Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self.payload


def zipped_csv(text: str) -> bytes:
    payload = BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("BTC-USDT-SWAP.csv", text)
    return payload.getvalue()


def test_fetch_daily_archives_caches_validated_parquet_and_records_404(tmp_path):
    requests = []
    archive = zipped_csv(
        "open_time,open,high,low,close,vol\n"
        "1767225600000,100,101,99,100.5,12\n"
    )

    def opener(request, timeout):
        requests.append((request, timeout))
        if "ETH-USDT-SWAP" in request.full_url:
            raise HTTPError(request.full_url, 404, "not found", {}, None)
        return Response(archive)

    result = fetch_daily_archives(
        ["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
        date(2026, 1, 1),
        date(2026, 1, 1),
        tmp_path,
        opener=opener,
        max_workers=1,
    )

    assert len(result.files) == 1
    assert pd.read_parquet(result.files[0])["open_time"].iloc[0] == pd.Timestamp(
        "2026-01-01 00:00:00+00:00"
    )
    assert result.failures == (
        {
            "instrument": "ETH-USDT-SWAP",
            "date": "2026-01-01",
            "error": "HTTP 404",
        },
    )
    assert all(request.headers["User-agent"] for request, _ in requests)


def test_fetch_funding_history_paginates_with_bounds_and_prefers_realized_rate(
    tmp_path,
):
    calls = []
    pages = [
        {
            "code": "0",
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "fundingTime": "1767312000000",
                    "realizedRate": "0.001",
                    "fundingRate": "0.009",
                },
                {
                    "instId": "BTC-USDT-SWAP",
                    "fundingTime": "1767283200000",
                    "realizedRate": "",
                    "fundingRate": "0.002",
                },
            ],
        },
        {"code": "0", "data": []},
    ]

    def opener(request, timeout):
        calls.append((request, timeout))
        return Response(json.dumps(pages.pop(0)).encode())

    result = fetch_funding_history(
        "BTC-USDT-SWAP",
        pd.Timestamp("2026-01-01 00:00:00+00:00"),
        pd.Timestamp("2026-01-03 00:00:00+00:00"),
        tmp_path / "funding.parquet",
        opener=opener,
        page_limit=2,
    )

    assert result["rate"].tolist() == [0.002, 0.001]
    first_query = parse_qs(urlparse(calls[0][0].full_url).query)
    second_query = parse_qs(urlparse(calls[1][0].full_url).query)
    assert first_query["before"] == ["1767225600000"]
    assert first_query["after"] == ["1767398400000"]
    assert second_query["before"] == ["1767225600000"]
    assert second_query["after"] == ["1767283200000"]
    assert all(request.headers["User-agent"] for request, _ in calls)
    assert pd.read_parquet(tmp_path / "funding.parquet").equals(result)
