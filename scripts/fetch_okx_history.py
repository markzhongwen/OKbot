#!/usr/bin/env python3
"""Download and prepare public OKX history for cost-aware research."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_aware.data import (  # noqa: E402
    _request_bytes,
    fetch_daily_archives,
    fetch_funding_history,
    resample_ohlcv,
    select_universe,
    validate_ohlcv,
)


INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments"
TICKERS_URL = "https://www.okx.com/api/v5/market/tickers"
FORCED = ("BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP")


def public_data(url: str, parameters: dict[str, object]) -> list[dict[str, object]]:
    payload = json.loads(
        _request_bytes(
            f"{url}?{urlencode(parameters)}",
            opener=urlopen,
            timeout=30,
            retries=3,
        )
    )
    if payload.get("code") != "0":
        raise RuntimeError(f"OKX public API error: {payload}")
    return payload["data"]


def daily_files(cache_root: Path, instrument: str, start, end) -> list[Path]:
    directory = cache_root / "raw_1m" / instrument
    return [
        directory / f"{day.date().isoformat()}.parquet"
        for day in pd.date_range(start, end, freq="D", tz="UTC")
        if (directory / f"{day.date().isoformat()}.parquet").exists()
    ]


def load_source(files: list[Path]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame(columns=("open_time", "open", "high", "low", "close", "vol"))
    return validate_ohlcv(
        pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    )


def complete_days(
    files: list[Path],
    start_date=None,
    end_date=None,
) -> list[str]:
    source = load_source(files)
    if source.empty:
        return []
    complete = []
    for day, frame in source.groupby(source["open_time"].dt.floor("D"), sort=True):
        if start_date is not None and day.date() < start_date:
            continue
        if end_date is not None and day.date() > end_date:
            continue
        expected = pd.date_range(day, periods=1440, freq="min")
        if len(frame) == 1440 and frame["open_time"].tolist() == expected.tolist():
            complete.append(day.date().isoformat())
    return complete


def frame_coverage(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"rows": 0, "first": None, "last": None}
    return {
        "rows": len(frame),
        "first": frame["open_time"].iloc[0].isoformat(),
        "last": frame["open_time"].iloc[-1].isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=91)
    parser.add_argument("--universe-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "data" / "okx_cost_aware",
    )
    args = parser.parse_args()
    if args.days < 91:
        parser.error("--days must be at least 91")
    if args.universe_size < len(FORCED):
        parser.error("--universe-size must accommodate BTC, ETH, and SOL")

    now = datetime.now(timezone.utc)
    # OKX archive dates use an UTC+8 boundary. Yesterday's archive supplies
    # the final eight hours of the day before yesterday in UTC.
    end_date = (now - timedelta(days=2)).date()
    start_date = end_date - timedelta(days=args.days - 1)
    archive_end_date = end_date + timedelta(days=1)
    start = pd.Timestamp(start_date, tz="UTC")
    end_exclusive = pd.Timestamp(end_date + timedelta(days=1), tz="UTC")
    args.cache_root.mkdir(parents=True, exist_ok=True)

    instruments = public_data(INSTRUMENTS_URL, {"instType": "SWAP"})
    tickers = public_data(TICKERS_URL, {"instType": "SWAP"})
    candidate_limit = min(
        len(instruments),
        max(args.universe_size + 15, args.universe_size),
    )
    candidates = select_universe(
        tickers,
        instruments,
        limit=candidate_limit,
        forced=FORCED,
    )
    primary = candidates[: args.universe_size - len(FORCED)] + list(FORCED)
    backups = [
        instrument
        for instrument in candidates
        if instrument not in primary and instrument not in FORCED
    ]

    archive_failures: list[dict[str, str]] = []
    requested: list[str] = []

    def download(batch: list[str]) -> None:
        if not batch:
            return
        requested.extend(batch)
        result = fetch_daily_archives(
            batch,
            start_date,
            archive_end_date,
            args.cache_root,
            max_workers=args.workers,
        )
        archive_failures.extend(result.failures)
        print(
            f"archives: {len(result.files)} files cached, "
            f"{len(result.failures)} failures for {', '.join(batch)}",
            flush=True,
        )

    download(primary)
    coverage_days: dict[str, list[str]] = {}
    selected: list[str] = []
    rejected: list[dict[str, object]] = []
    for instrument in primary:
        days = complete_days(
            daily_files(args.cache_root, instrument, start_date, archive_end_date),
            start_date,
            end_date,
        )
        coverage_days[instrument] = days
        if len(days) >= 90:
            selected.append(instrument)
        else:
            rejected.append(
                {
                    "instrument": instrument,
                    "complete_days": len(days),
                    "reason": "fewer than 90 complete UTC days",
                }
            )

    for backup in backups:
        if len(selected) >= args.universe_size:
            break
        download([backup])
        days = complete_days(
            daily_files(args.cache_root, backup, start_date, archive_end_date),
            start_date,
            end_date,
        )
        coverage_days[backup] = days
        if len(days) >= 90:
            selected.append(backup)
        else:
            rejected.append(
                {
                    "instrument": backup,
                    "complete_days": len(days),
                    "reason": "fewer than 90 complete UTC days",
                }
            )

    # Forced instruments stay mandatory; a coverage failure is reported, not hidden.
    selected = list(dict.fromkeys([*selected, *FORCED]))
    selected = selected[: args.universe_size]

    output_coverage: dict[str, object] = {}
    funding_failures: list[dict[str, str]] = []
    for sequence, instrument in enumerate(selected, start=1):
        paths = daily_files(
            args.cache_root, instrument, start_date, archive_end_date
        )
        source = load_source(paths)
        source = source.loc[
            (source["open_time"] >= start)
            & (source["open_time"] < end_exclusive)
        ].reset_index(drop=True)
        instrument_coverage: dict[str, object] = {
            "complete_utc_days": len(
                complete_days(paths, start_date, end_date)
            ),
            "requested_days": args.days,
            "1m": frame_coverage(source),
        }
        for frequency, directory_name in (
            ("3min", "3m"),
            ("15min", "15m"),
            ("1h", "1h"),
            ("4h", "4h"),
        ):
            bars = resample_ohlcv(source, frequency)
            destination = (
                args.cache_root / "bars" / directory_name / f"{instrument}.parquet"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            bars.to_parquet(destination, index=False)
            instrument_coverage[directory_name] = frame_coverage(bars)

        try:
            funding = fetch_funding_history(
                instrument,
                start,
                end_exclusive,
                args.cache_root / "funding" / f"{instrument}.parquet",
            )
            instrument_coverage["funding"] = {
                "rows": len(funding),
                "first": (
                    funding["timestamp"].iloc[0].isoformat()
                    if not funding.empty
                    else None
                ),
                "last": (
                    funding["timestamp"].iloc[-1].isoformat()
                    if not funding.empty
                    else None
                ),
            }
        except Exception as error:
            funding_failures.append(
                {
                    "instrument": instrument,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            instrument_coverage["funding"] = {"error": str(error)}

        output_coverage[instrument] = instrument_coverage
        print(
            f"prepared {sequence}/{len(selected)} {instrument}: "
            f"{instrument_coverage['complete_utc_days']} complete days",
            flush=True,
        )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "OKX public tickers, instruments, daily archives, and funding API",
        "requested_start": start_date.isoformat(),
        "requested_end": end_date.isoformat(),
        "requested_days": args.days,
        "universe_size_requested": args.universe_size,
        "universe": selected,
        "forced_instruments": list(FORCED),
        "downloaded_candidates": requested,
        "rejected_candidates": rejected,
        "coverage": output_coverage,
        "archive_failures": archive_failures,
        "funding_failures": funding_failures,
    }
    metadata_path = args.cache_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"metadata: {metadata_path}", flush=True)

    concerns = (
        len(selected) < args.universe_size
        or any(
            details["complete_utc_days"] < 90
            for details in output_coverage.values()
        )
        or bool(funding_failures)
    )
    return 1 if concerns else 0


if __name__ == "__main__":
    raise SystemExit(main())
