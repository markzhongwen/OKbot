from datetime import date
from io import BytesIO
import json
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile

import pandas as pd
import pytest

from cost_aware.data import (
    _request_bytes,
    fetch_daily_archives,
    fetch_funding_history,
    resample_ohlcv,
    select_universe,
    validate_ohlcv,
)
from scripts.fetch_okx_history import (
    choose_covered_universe,
    complete_days,
    ticker_snapshot,
)


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


def test_choose_covered_universe_replaces_nonforced_without_squeezing_forced():
    coverage = {
        "LOW-USDT-SWAP": 89,
        "BACKUP-USDT-SWAP": 90,
        "BTC-USDT-SWAP": 90,
        "ETH-USDT-SWAP": 91,
        "SOL-USDT-SWAP": 90,
    }

    selected = choose_covered_universe(
        [
            "LOW-USDT-SWAP",
            "BTC-USDT-SWAP",
            "ETH-USDT-SWAP",
            "SOL-USDT-SWAP",
        ],
        ["BACKUP-USDT-SWAP"],
        coverage,
        limit=4,
    )

    assert selected == [
        "BACKUP-USDT-SWAP",
        "BTC-USDT-SWAP",
        "ETH-USDT-SWAP",
        "SOL-USDT-SWAP",
    ]


def test_choose_covered_universe_fails_when_forced_instrument_lacks_coverage():
    coverage = {
        "BACKUP-USDT-SWAP": 91,
        "BTC-USDT-SWAP": 91,
        "ETH-USDT-SWAP": 89,
        "SOL-USDT-SWAP": 91,
    }

    with pytest.raises(ValueError, match="forced.*ETH-USDT-SWAP.*89"):
        choose_covered_universe(
            ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"],
            ["BACKUP-USDT-SWAP"],
            coverage,
            limit=4,
        )


def test_ticker_snapshot_records_quote_volume_and_market_rank():
    snapshot = ticker_snapshot(
        ["BTC-USDT-SWAP", "SOL-USDT-SWAP"],
        [
            {"instId": "SOL-USDT-SWAP", "last": "5", "volCcy24h": "3"},
            {"instId": "BTC-USDT-SWAP", "last": "100", "volCcy24h": "2"},
            {"instId": "ETH-USDT-SWAP", "last": "10", "volCcy24h": "3"},
        ],
    )

    assert snapshot == {
        "BTC-USDT-SWAP": {
            "last": 100.0,
            "volCcy24h": 2.0,
            "estimated_quote_volume": 200.0,
            "rank": 1,
        },
        "SOL-USDT-SWAP": {
            "last": 5.0,
            "volCcy24h": 3.0,
            "estimated_quote_volume": 15.0,
            "rank": 3,
        },
    }


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


@pytest.mark.parametrize("status", [429, 500, 503])
def test_request_bytes_retries_rate_limits_and_server_errors(
    status, monkeypatch
):
    calls = 0
    sleeps = []

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(
                request.full_url,
                status,
                "retry",
                {"Retry-After": "0"},
                None,
            )
        return Response(b"ok")

    monkeypatch.setattr("cost_aware.data.time.sleep", sleeps.append)

    result = _request_bytes(
        "https://example.test/data",
        opener=opener,
        timeout=1,
        retries=1,
    )

    assert result == b"ok"
    assert calls == 2
    assert sleeps == [0.0]


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
    start = pd.Timestamp("2026-01-01 00:00:00+00:00")
    end = pd.Timestamp("2026-01-03 00:00:00+00:00")

    def ms(timestamp):
        return str(int(pd.Timestamp(timestamp).timestamp() * 1000))

    pages = [
        {
            "code": "0",
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "fundingTime": ms("2026-01-02 16:00:00+00:00"),
                    "realizedRate": "0.001",
                    "fundingRate": "0.009",
                },
                {
                    "instId": "BTC-USDT-SWAP",
                    "fundingTime": ms("2026-01-02 08:00:00+00:00"),
                    "realizedRate": "",
                    "fundingRate": "0.002",
                },
            ],
        },
        {
            "code": "0",
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "fundingTime": ms("2026-01-01 16:00:00+00:00"),
                    "realizedRate": "0.003",
                    "fundingRate": "0.009",
                },
                {
                    "instId": "BTC-USDT-SWAP",
                    "fundingTime": ms("2026-01-01 08:00:00+00:00"),
                    "realizedRate": "0.004",
                    "fundingRate": "0.009",
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
        start,
        end,
        tmp_path / "funding.parquet",
        opener=opener,
        page_limit=2,
    )

    assert result["rate"].tolist() == [0.004, 0.003, 0.002, 0.001]
    first_query = parse_qs(urlparse(calls[0][0].full_url).query)
    second_query = parse_qs(urlparse(calls[1][0].full_url).query)
    assert first_query["before"] == [ms(start)]
    assert first_query["after"] == [ms(end)]
    assert second_query["before"] == [ms(start)]
    assert second_query["after"] == [ms("2026-01-02 08:00:00+00:00")]
    assert all(request.headers["User-agent"] for request, _ in calls)
    assert pd.read_parquet(tmp_path / "funding.parquet").equals(result)


def test_fetch_funding_history_rejects_empty_successful_response(tmp_path):
    def opener(_request, _timeout=None, **_kwargs):
        return Response(b'{"code":"0","data":[]}')

    with pytest.raises(ValueError, match="empty funding history"):
        fetch_funding_history(
            "BTC-USDT-SWAP",
            pd.Timestamp("2026-01-01 00:00:00+00:00"),
            pd.Timestamp("2026-01-03 00:00:00+00:00"),
            tmp_path / "funding.parquet",
            opener=opener,
        )


def test_fetch_funding_history_rejects_truncated_window(tmp_path):
    pages = [
        {
            "code": "0",
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "fundingTime": str(
                        int(
                            pd.Timestamp(
                                "2026-01-02 08:00:00+00:00"
                            ).timestamp()
                            * 1000
                        )
                    ),
                    "realizedRate": "0.001",
                }
            ],
        },
        {"code": "0", "data": []},
    ]

    def opener(_request, _timeout=None, **_kwargs):
        return Response(json.dumps(pages.pop(0)).encode())

    with pytest.raises(ValueError, match="does not cover requested window"):
        fetch_funding_history(
            "BTC-USDT-SWAP",
            pd.Timestamp("2026-01-01 00:00:00+00:00"),
            pd.Timestamp("2026-01-03 00:00:00+00:00"),
            tmp_path / "funding.parquet",
            opener=opener,
        )


def test_fetch_funding_history_accepts_four_hour_settlement_boundaries(tmp_path):
    start = pd.Timestamp("2026-01-01 00:00:00+00:00")
    end = pd.Timestamp("2026-01-02 00:00:00+00:00")
    timestamps = pd.date_range(start + pd.Timedelta(hours=4), end, freq="4h")
    page = {
        "code": "0",
        "data": [
            {
                "instId": "BTC-USDT-SWAP",
                "fundingTime": str(int(timestamp.timestamp() * 1000)),
                "realizedRate": "0.001",
            }
            for timestamp in reversed(timestamps[:-1])
        ],
    }
    pages = [page, {"code": "0", "data": []}]

    def opener(_request, _timeout=None, **_kwargs):
        return Response(json.dumps(pages.pop(0)).encode())

    result = fetch_funding_history(
        "BTC-USDT-SWAP",
        start,
        end,
        tmp_path / "funding.parquet",
        opener=opener,
    )

    assert result["timestamp"].iloc[0] == start + pd.Timedelta(hours=4)
    assert result["timestamp"].iloc[-1] == end - pd.Timedelta(hours=4)


def test_fetch_funding_history_retries_http_200_business_error(tmp_path):
    start = pd.Timestamp("2026-01-01 00:00:00+00:00")
    end = pd.Timestamp("2026-01-02 00:00:00+00:00")
    responses = [
        {"code": "50011", "msg": "rate limit", "data": []},
        {
            "code": "0",
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "fundingTime": str(
                        int((start + pd.Timedelta(hours=8)).timestamp() * 1000)
                    ),
                    "realizedRate": "0.001",
                },
                {
                    "instId": "BTC-USDT-SWAP",
                    "fundingTime": str(
                        int((end - pd.Timedelta(hours=8)).timestamp() * 1000)
                    ),
                    "realizedRate": "0.002",
                },
            ],
        },
        {"code": "0", "data": []},
    ]

    def opener(_request, _timeout=None, **_kwargs):
        return Response(json.dumps(responses.pop(0)).encode())

    result = fetch_funding_history(
        "BTC-USDT-SWAP",
        start,
        end,
        tmp_path / "funding.parquet",
        opener=opener,
        retries=1,
    )

    assert len(result) == 2
