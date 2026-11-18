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
    fetch_daily_archives,
    fetch_funding_history,
    request_okx_json,
    resample_ohlcv,
    select_universe,
    validate_ohlcv,
)


INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments"
TICKERS_URL = "https://www.okx.com/api/v5/market/tickers"
FORCED = ("BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP")


def choose_covered_universe(
    primary: list[str],
    backups: list[str],
    coverage: dict[str, int],
    funding_complete: set[str],
    *,
    limit: int,
    forced: tuple[str, ...] = FORCED,
    minimum_days: int = 90,
) -> list[str]:
    insufficient_forced = [
        f"{instrument} ({coverage.get(instrument, 0)} days)"
        for instrument in forced
        if coverage.get(instrument, 0) < minimum_days
    ]
    if insufficient_forced:
        raise ValueError(
            "forced instrument coverage below "
            f"{minimum_days} days: {', '.join(insufficient_forced)}"
        )
    incomplete_forced = [
        instrument for instrument in forced if instrument not in funding_complete
    ]
    if incomplete_forced:
        raise ValueError(
            "forced instrument "
            f"{', '.join(incomplete_forced)} funding incomplete"
        )

    ranked = list(dict.fromkeys([*primary, *backups]))
    nonforced = [
        instrument
        for instrument in ranked
        if instrument not in forced
        and coverage.get(instrument, 0) >= minimum_days
        and instrument in funding_complete
    ]
    needed = limit - len(forced)
    if len(nonforced) < needed:
        raise ValueError(
            f"only {len(nonforced) + len(forced)} instruments have "
            f"{minimum_days} complete days; {limit} required"
        )
    return [*nonforced[:needed], *forced]


def ticker_snapshot(
    selected: list[str],
    tickers: list[dict[str, object]],
) -> dict[str, dict[str, float | int | str]]:
    rows = []
    for ticker in tickers:
        try:
            last = float(ticker["last"])
            volume = float(ticker["volCcy24h"])
            timestamp = str(ticker["ts"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(
            (
                str(ticker.get("instId", "")),
                last,
                volume,
                last * volume,
                timestamp,
            )
        )
    rows.sort(key=lambda row: (-row[3], row[0]))
    by_instrument = {
        instrument: {
            "last": last,
            "volCcy24h": volume,
            "estimated_quote_volume": quote_volume,
            "rank": rank,
            "ts": timestamp,
        }
        for rank, (
            instrument,
            last,
            volume,
            quote_volume,
            timestamp,
        ) in enumerate(
            rows, start=1
        )
    }
    missing = [instrument for instrument in selected if instrument not in by_instrument]
    if missing:
        raise ValueError(f"missing ticker snapshot for: {', '.join(missing)}")
    return {
        instrument: by_instrument[instrument]
        for instrument in selected
        if instrument in by_instrument
    }


def public_data(
    url: str, parameters: dict[str, object]
) -> dict[str, object]:
    return request_okx_json(
        f"{url}?{urlencode(parameters)}",
        opener=urlopen,
        timeout=30,
        retries=3,
    )


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

    instruments_payload = public_data(INSTRUMENTS_URL, {"instType": "SWAP"})
    tickers_payload = public_data(TICKERS_URL, {"instType": "SWAP"})
    instruments = instruments_payload["data"]
    tickers = tickers_payload["data"]
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

    coverage_counts: dict[str, int] = {}
    funding_complete: set[str] = set()
    funding_frames: dict[str, pd.DataFrame] = {}
    rejected: list[dict[str, object]] = []
    candidate_funding_failures: list[dict[str, str]] = []

    def assess_ohlcv(instrument: str) -> bool:
        days = complete_days(
            daily_files(args.cache_root, instrument, start_date, archive_end_date),
            start_date,
            end_date,
        )
        coverage_counts[instrument] = len(days)
        if len(days) < 90:
            rejected.append(
                {
                    "instrument": instrument,
                    "complete_days": len(days),
                    "reason": "fewer than 90 complete UTC days",
                }
            )
            return False
        return True

    def assess_funding(instrument: str) -> bool:
        try:
            frame = fetch_funding_history(
                instrument,
                start,
                end_exclusive,
                args.cache_root / "funding" / f"{instrument}.parquet",
            )
        except Exception as error:
            failure = {
                "instrument": instrument,
                "error": f"{type(error).__name__}: {error}",
            }
            candidate_funding_failures.append(failure)
            rejected.append(
                {
                    "instrument": instrument,
                    "complete_days": coverage_counts.get(instrument, 0),
                    "reason": f"incomplete funding: {failure['error']}",
                }
            )
            return False
        funding_complete.add(instrument)
        funding_frames[instrument] = frame
        return True

    download(primary)
    for instrument in primary:
        if assess_ohlcv(instrument):
            assess_funding(instrument)

    try:
        choose_covered_universe(
            list(FORCED),
            [],
            coverage_counts,
            funding_complete,
            limit=len(FORCED),
        )
    except ValueError as error:
        print(f"universe selection failed: {error}", file=sys.stderr)
        return 1

    evaluated_backups: list[str] = []
    required_nonforced = args.universe_size - len(FORCED)
    for backup in backups:
        eligible_nonforced = sum(
            instrument not in FORCED
            and days >= 90
            and instrument in funding_complete
            for instrument, days in coverage_counts.items()
        )
        if eligible_nonforced >= required_nonforced:
            break
        download([backup])
        evaluated_backups.append(backup)
        if assess_ohlcv(backup):
            assess_funding(backup)

    try:
        selected = choose_covered_universe(
            primary,
            evaluated_backups,
            coverage_counts,
            funding_complete,
            limit=args.universe_size,
        )
    except ValueError as error:
        print(f"universe selection failed: {error}", file=sys.stderr)
        return 1

    output_coverage: dict[str, object] = {}
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

        funding = funding_frames[instrument]
        instrument_coverage["funding"] = {
            "rows": len(funding),
            "first": funding["timestamp"].iloc[0].isoformat(),
            "last": funding["timestamp"].iloc[-1].isoformat(),
            "max_gap": funding["timestamp"].diff().max().isoformat(),
        }

        output_coverage[instrument] = instrument_coverage
        print(
            f"prepared {sequence}/{len(selected)} {instrument}: "
            f"{instrument_coverage['complete_utc_days']} complete days",
            flush=True,
        )

    active_usdt = {
        str(item.get("instId", ""))
        for item in instruments
        if item.get("state") == "live"
        and item.get("settleCcy") == "USDT"
        and str(item.get("instId", "")).endswith("-USDT-SWAP")
    }
    selected_tickers = [
        ticker for ticker in tickers if str(ticker.get("instId", "")) in active_usdt
    ]
    snapshots = ticker_snapshot(selected, selected_tickers)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "OKX public tickers, instruments, daily archives, and funding API",
        "requested_start": start_date.isoformat(),
        "requested_end": end_date.isoformat(),
        "requested_days": args.days,
        "universe_size_requested": args.universe_size,
        "universe": selected,
        "ticker_snapshot": snapshots,
        "api_timestamps": {
            "instruments": instruments_payload.get("ts"),
            "tickers": max(item["ts"] for item in snapshots.values()),
        },
        "forced_instruments": list(FORCED),
        "downloaded_candidates": requested,
        "rejected_candidates": rejected,
        "coverage": output_coverage,
        "archive_failures": archive_failures,
        "funding_failures": [],
        "candidate_funding_failures": candidate_funding_failures,
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
    )
    return 1 if concerns else 0


if __name__ == "__main__":
    raise SystemExit(main())
