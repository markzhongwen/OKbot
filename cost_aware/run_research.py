"""Execute the reproducible cost-aware OKX walk-forward research.

The default command consumes the cached final universe under
``data/okx_cost_aware`` and evaluates every bounded strategy grid on all four
timeframes.  Checkpoints are written after each family/timeframe result, while
the small final JSON summary and both human-readable reports are written only
after the complete run.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .config import ExecutionConfig, SizingConfig
from .engine import (
    BacktestResult,
    PreparedCandles,
    prepare_candles,
    run_backtest_result,
)
from .report import validate_summary, write_reports
from .strategies import (
    STRATEGY_SPECS,
    StrategySpec,
    generate_signals,
    neighbor_params,
    parameter_grid,
)
from .walkforward import (
    CandidateEvaluation,
    Fold,
    Metrics,
    build_folds,
    compute_metrics,
    evaluate_candidate,
    passes_research_gates,
)


TIMEFRAMES: dict[str, str] = {
    "3m": "3min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
}
COST_ON_EXECUTION = ExecutionConfig(
    signal_lag_bars=1,
    fee_per_side=0.0005,
    slippage_per_side=0.0002,
)
COST_OFF_EXECUTION = ExecutionConfig(
    signal_lag_bars=1,
    fee_per_side=0.0,
    slippage_per_side=0.0,
)
PRIMARY_SIZING = SizingConfig(
    initial_equity=100_000.0,
    margin_fraction=0.10,
    leverage=8.0,
    max_positions=3,
)
REFERENCE_SIZING = SizingConfig(
    initial_equity=100_000.0,
    margin_fraction=0.05,
    leverage=5.0,
    max_positions=3,
)
USER_CALIBRATION_RETURNS = {"7d": -0.066, "30d": -0.127, "56d": -0.181}
CALIBRATION_PARAMS = {
    "lookback": 20,
    "atr_window": 14,
    "stop_atr": 3.5,
    "take_profit_atr": 4.0,
    "time_stop_bars": 24,
    "cooldown_bars": 6,
}


@dataclass(frozen=True)
class ResearchData:
    metadata: dict[str, object]
    bars: dict[str, pd.DataFrame]
    funding: pd.DataFrame


def _canonical_params(params: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(params.items()))


def _timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def write_json_atomic(path: Path | str, value: object) -> None:
    """Write stable UTF-8 JSON without exposing a partially written file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload = json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(destination)


def search_space_manifest() -> dict[str, object]:
    """Describe the complete deterministic grid before any result is known."""
    grids = {
        name: parameter_grid(spec)
        for name, spec in STRATEGY_SPECS.items()
    }
    grid_sizes = {name: len(grid) for name, grid in grids.items()}
    configurations = sum(grid_sizes.values()) * len(TIMEFRAMES)
    return {
        "strategies": list(STRATEGY_SPECS),
        "timeframes": list(TIMEFRAMES),
        "family_timeframe_candidates": len(STRATEGY_SPECS) * len(TIMEFRAMES),
        "grid_sizes": grid_sizes,
        "grid_configurations": configurations,
        "is_fold_evaluations": configurations * 3,
        "grids": grids,
    }


def build_research_folds(metadata: Mapping[str, object]) -> tuple[Fold, ...]:
    """Build the required 45d IS + 15d OOS × 3 schedule from metadata."""
    if "requested_start" not in metadata or "requested_end" not in metadata:
        raise ValueError("metadata requires requested_start and requested_end")
    start = _timestamp(metadata["requested_start"]).normalize()
    requested_end = _timestamp(metadata["requested_end"]).normalize()
    folds = build_folds(start, is_days=45, oos_days=15, n_folds=3)
    if folds[-1].oos_end > requested_end + pd.Timedelta(days=1):
        raise ValueError("metadata coverage is too short for the required folds")
    return folds


def _prepare_engine_bars(
    candles: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cadence: pd.Timedelta,
    signal_lag_bars: int,
) -> pd.DataFrame:
    timestamp_column = "timestamp" if "timestamp" in candles else "open_time"
    earliest = start - cadence * signal_lag_bars
    timestamps = pd.to_datetime(candles[timestamp_column], utc=True)
    selected = candles.loc[
        (timestamps >= earliest) & (timestamps < end),
        ["symbol", timestamp_column, "open", "high", "low", "close"],
    ].copy()
    return selected.rename(columns={timestamp_column: "timestamp"})


def _signals_with_exact_entry_bar(
    signals: pd.DataFrame,
    engine_bars: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cadence: pd.Timedelta,
    signal_lag_bars: int,
) -> pd.DataFrame:
    earliest = start - cadence * signal_lag_bars
    selected = signals.loc[
        (signals["timestamp"] >= earliest) & (signals["timestamp"] < end)
    ].copy()
    if selected.empty:
        return selected
    available = pd.MultiIndex.from_frame(
        engine_bars[["symbol", "timestamp"]].assign(
            timestamp=lambda frame: frame["timestamp"]
        )
    )
    entry_keys = pd.MultiIndex.from_arrays(
        [
            selected["symbol"],
            selected["timestamp"] + cadence * signal_lag_bars,
        ]
    )
    return selected.loc[entry_keys.isin(available)].reset_index(drop=True)


@dataclass(frozen=True)
class _PreparedWindow:
    engine_bars: pd.DataFrame
    prepared_candles: PreparedCandles
    funding_records: tuple[dict[str, object], ...]


def _materialize_window(
    candles: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cadence: pd.Timedelta,
    signal_lag_bars: int,
) -> _PreparedWindow:
    engine_bars = _prepare_engine_bars(
        candles,
        start=start,
        end=end,
        cadence=cadence,
        signal_lag_bars=signal_lag_bars,
    )
    funding_timestamps = pd.to_datetime(funding["timestamp"], utc=True)
    funding_window = funding.loc[
        (funding_timestamps >= start) & (funding_timestamps < end)
    ]
    candle_records = tuple(engine_bars.to_dict("records"))
    return _PreparedWindow(
        engine_bars=engine_bars,
        prepared_candles=prepare_candles(candle_records),
        funding_records=tuple(funding_window.to_dict("records")),
    )


def _run_window_from_signals(
    candles: pd.DataFrame,
    signals: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    start: object,
    end: object,
    expected_cadence: pd.Timedelta | str,
    execution: ExecutionConfig,
    sizing: SizingConfig,
    include_funding: bool,
    prepared_window: _PreparedWindow | None = None,
) -> BacktestResult:
    start_time = _timestamp(start)
    end_time = _timestamp(end)
    cadence = pd.Timedelta(expected_cadence)
    window = prepared_window or _materialize_window(
        candles,
        funding,
        start=start_time,
        end=end_time,
        cadence=cadence,
        signal_lag_bars=execution.signal_lag_bars,
    )
    scheduled_signals = _signals_with_exact_entry_bar(
        signals,
        window.engine_bars,
        start=start_time,
        end=end_time,
        cadence=cadence,
        signal_lag_bars=execution.signal_lag_bars,
    )
    return run_backtest_result(
        window.prepared_candles,
        scheduled_signals.to_dict("records"),
        window.funding_records if include_funding else (),
        execution,
        sizing,
        start_time=start_time,
        end_time=end_time,
        expected_cadence=cadence,
        force_close_at_end=True,
    )


def run_strategy_window(
    candles: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    spec: StrategySpec,
    params: Mapping[str, object],
    start: object,
    end: object,
    expected_cadence: pd.Timedelta | str,
    execution: ExecutionConfig,
    sizing: SizingConfig,
    include_funding: bool = True,
) -> BacktestResult:
    """Run one flat-start window while retaining pre-window indicator history."""
    end_time = _timestamp(end)
    timestamp_column = "timestamp" if "timestamp" in candles else "open_time"
    timestamps = pd.to_datetime(candles[timestamp_column], utc=True)
    indicator_history = candles.loc[timestamps < end_time]
    signals = generate_signals(
        indicator_history,
        spec,
        params,
        expected_cadence=expected_cadence,
    )
    return _run_window_from_signals(
        indicator_history,
        signals,
        funding,
        start=start,
        end=end,
        expected_cadence=expected_cadence,
        execution=execution,
        sizing=sizing,
        include_funding=include_funding,
    )


def _validate_loaded_bars(
    frame: pd.DataFrame,
    *,
    symbol: str,
    cadence: pd.Timedelta,
) -> pd.DataFrame:
    required = {"open_time", "open", "high", "low", "close", "vol"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{symbol}: cached bars missing {sorted(missing)}")
    result = frame.loc[
        :,
        ["open_time", "open", "high", "low", "close", "vol"],
    ].copy()
    result["open_time"] = pd.to_datetime(result["open_time"], utc=True)
    if result["open_time"].duplicated().any():
        raise ValueError(f"{symbol}: duplicate cached bar timestamps")
    if (
        result["open_time"].dt.floor(cadence)
        != result["open_time"]
    ).any():
        raise ValueError(f"{symbol}: cached bars are not cadence-aligned")
    result.insert(0, "symbol", symbol)
    return result


def load_research_data(cache_dir: Path | str) -> ResearchData:
    """Load and validate the final metadata universe, four bars, and funding."""
    root = Path(cache_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    universe = list(metadata.get("universe", []))
    if len(universe) != 20 or len(set(universe)) != 20:
        raise ValueError("metadata must identify the final 20-instrument universe")

    bars: dict[str, pd.DataFrame] = {}
    for timeframe, cadence_text in TIMEFRAMES.items():
        cadence = pd.Timedelta(cadence_text)
        frames = []
        for symbol in universe:
            path = root / "bars" / timeframe / f"{symbol}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"missing cached bars: {path}")
            frames.append(
                _validate_loaded_bars(
                    pd.read_parquet(path),
                    symbol=symbol,
                    cadence=cadence,
                )
            )
        bars[timeframe] = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["open_time", "symbol"], kind="stable")
            .reset_index(drop=True)
        )

    funding_frames = []
    for symbol in universe:
        path = root / "funding" / f"{symbol}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing cached funding: {path}")
        frame = pd.read_parquet(path)
        required = {"symbol", "timestamp", "rate"}
        if missing := required.difference(frame.columns):
            raise ValueError(f"{symbol}: cached funding missing {sorted(missing)}")
        frame = frame.loc[:, ["symbol", "timestamp", "rate"]].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        funding_frames.append(frame)
    funding = (
        pd.concat(funding_frames, ignore_index=True)
        .sort_values(["timestamp", "symbol"], kind="stable")
        .reset_index(drop=True)
    )
    return ResearchData(metadata=metadata, bars=bars, funding=funding)


def _equity_series(result: BacktestResult) -> pd.Series:
    index = pd.MultiIndex.from_arrays(
        [
            [event.timestamp for event in result.equity_events],
            [event.sequence for event in result.equity_events],
        ],
        names=["timestamp", "sequence"],
    )
    return pd.Series(
        [event.equity for event in result.equity_events],
        index=index,
        dtype=float,
    )


def _chain_results(
    results: Sequence[BacktestResult],
) -> tuple[pd.Series, list[object], tuple[float, ...]]:
    values: list[float] = []
    timestamps: list[pd.Timestamp] = []
    level = 1.0
    trades: list[object] = []
    fold_returns: list[float] = []
    previous_end: pd.Timestamp | None = None
    for result in results:
        equity = _equity_series(result)
        raw_timestamps = pd.DatetimeIndex(equity.index.get_level_values(0))
        fold_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        fold_returns.append(fold_return)
        normalized = equity / float(equity.iloc[0]) * level
        if previous_end is not None and raw_timestamps[0] == previous_end:
            normalized = normalized.iloc[1:]
            raw_timestamps = raw_timestamps[1:]
        values.extend(float(value) for value in normalized)
        timestamps.extend(raw_timestamps)
        level *= 1.0 + fold_return
        previous_end = pd.DatetimeIndex(
            equity.index.get_level_values(0)
        )[-1]
        trades.extend(result.trades)
    index = pd.MultiIndex.from_arrays(
        [timestamps, range(len(values))],
        names=["timestamp", "sequence"],
    )
    return pd.Series(values, index=index), trades, tuple(fold_returns)


def _metrics_for_results(results: Sequence[BacktestResult]) -> Metrics:
    equity, trades, fold_returns = _chain_results(results)
    return compute_metrics(equity, trades, fold_returns=fold_returns)


def _metrics_dict(metrics: Metrics, *, folds: bool = False) -> dict[str, object]:
    result: dict[str, object] = {
        "return": float(metrics.total_return),
        "sharpe": float(metrics.sharpe),
        "max_drawdown": float(metrics.max_drawdown),
        "trades": int(metrics.trades),
        "win_rate": float(metrics.win_rate),
    }
    if folds:
        result["fold_returns"] = [float(value) for value in metrics.fold_returns]
    return result


class ResearchRunner:
    """Shared data, signal cache, and bounded backtest-result cache."""

    def __init__(self, data: ResearchData, *, result_cache_size: int = 32):
        self.data = data
        self.result_cache_size = result_cache_size
        self._signal_cache: dict[
            tuple[str, str, tuple[tuple[str, object], ...]],
            pd.DataFrame,
        ] = {}
        self._result_cache: OrderedDict[tuple[object, ...], BacktestResult] = (
            OrderedDict()
        )
        self._window_cache: OrderedDict[
            tuple[object, ...],
            _PreparedWindow,
        ] = OrderedDict()
        self.signal_generations = 0
        self.backtests = 0
        self.cache_hits = 0
        self.window_materializations = 0

    def signals(
        self,
        spec: StrategySpec,
        timeframe: str,
        params: Mapping[str, object],
    ) -> pd.DataFrame:
        key = (spec.name, timeframe, _canonical_params(params))
        if key not in self._signal_cache:
            self._signal_cache[key] = generate_signals(
                self.data.bars[timeframe],
                spec,
                params,
                expected_cadence=TIMEFRAMES[timeframe],
            )
            self.signal_generations += 1
        return self._signal_cache[key]

    def run(
        self,
        spec: StrategySpec,
        timeframe: str,
        params: Mapping[str, object],
        *,
        start: object,
        end: object,
        sizing: SizingConfig,
        execution: ExecutionConfig = COST_ON_EXECUTION,
        include_funding: bool = True,
    ) -> BacktestResult:
        start_time = _timestamp(start)
        end_time = _timestamp(end)
        key = (
            spec.name,
            timeframe,
            _canonical_params(params),
            start_time.value,
            end_time.value,
            sizing,
            execution,
            include_funding,
        )
        cached = self._result_cache.get(key)
        if cached is not None:
            self._result_cache.move_to_end(key)
            self.cache_hits += 1
            return cached
        window_key = (
            timeframe,
            start_time.value,
            end_time.value,
            execution.signal_lag_bars,
        )
        prepared_window = self._window_cache.get(window_key)
        if prepared_window is None:
            prepared_window = _materialize_window(
                self.data.bars[timeframe],
                self.data.funding,
                start=start_time,
                end=end_time,
                cadence=pd.Timedelta(TIMEFRAMES[timeframe]),
                signal_lag_bars=execution.signal_lag_bars,
            )
            self.window_materializations += 1
            self._window_cache[window_key] = prepared_window
        self._window_cache.move_to_end(window_key)
        while len(self._window_cache) > 1:
            self._window_cache.popitem(last=False)
        result = _run_window_from_signals(
            self.data.bars[timeframe],
            self.signals(spec, timeframe, params),
            self.data.funding,
            start=start_time,
            end=end_time,
            expected_cadence=TIMEFRAMES[timeframe],
            execution=execution,
            sizing=sizing,
            include_funding=include_funding,
            prepared_window=prepared_window,
        )
        self.backtests += 1
        self._result_cache[key] = result
        self._result_cache.move_to_end(key)
        while len(self._result_cache) > self.result_cache_size:
            self._result_cache.popitem(last=False)
        return result

    def clear_candidate_caches(self) -> None:
        self._signal_cache.clear()
        self._result_cache.clear()
        self._window_cache.clear()


def _fold_window(fold: Fold, phase: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if phase == "is":
        return fold.is_start, fold.is_end
    return fold.oos_start, fold.oos_end


def _main_evaluation(
    runner: ResearchRunner,
    spec: StrategySpec,
    timeframe: str,
    folds: Sequence[Fold],
) -> CandidateEvaluation:
    def callback(
        params: Mapping[str, object],
        fold: Fold,
        sizing: object,
        phase: str,
    ) -> BacktestResult:
        start, end = _fold_window(fold, phase)
        return runner.run(
            spec,
            timeframe,
            params,
            start=start,
            end=end,
            sizing=sizing,
        )

    return evaluate_candidate(
        parameter_grid(spec),
        folds,
        sizing=PRIMARY_SIZING,
        callback=callback,
        neighbors=lambda params: neighbor_params(spec, params),
    )


def _frozen_variant_metrics(
    runner: ResearchRunner,
    spec: StrategySpec,
    timeframe: str,
    folds: Sequence[Fold],
    selected_params: Sequence[Mapping[str, object]],
) -> tuple[Metrics, Metrics]:
    cost_off_results: list[BacktestResult] = []
    reference_results: list[BacktestResult] = []
    for fold, params in zip(folds, selected_params, strict=True):
        cost_off_results.append(
            runner.run(
                spec,
                timeframe,
                params,
                start=fold.oos_start,
                end=fold.oos_end,
                sizing=PRIMARY_SIZING,
                execution=COST_OFF_EXECUTION,
                include_funding=False,
            )
        )
        reference_results.append(
            runner.run(
                spec,
                timeframe,
                params,
                start=fold.oos_start,
                end=fold.oos_end,
                sizing=REFERENCE_SIZING,
            )
        )
    return (
        _metrics_for_results(cost_off_results),
        _metrics_for_results(reference_results),
    )


def _calendar_returns(
    runner: ResearchRunner,
    spec: StrategySpec,
    timeframe: str,
    params: Mapping[str, object],
    *,
    end: pd.Timestamp,
    windows: Iterable[int],
    sizings: Sequence[SizingConfig],
) -> list[dict[str, float]]:
    output = [dict() for _ in sizings]
    for days in windows:
        start = end - pd.Timedelta(days=days)
        for index, sizing in enumerate(sizings):
            result = runner.run(
                spec,
                timeframe,
                params,
                start=start,
                end=end,
                sizing=sizing,
            )
            metrics = compute_metrics(_equity_series(result), result.trades)
            output[index][f"{days}d"] = float(metrics.total_return)
    return output


def _candidate_result(
    runner: ResearchRunner,
    spec: StrategySpec,
    timeframe: str,
    folds: Sequence[Fold],
    *,
    calendar_end: pd.Timestamp,
) -> dict[str, object]:
    evaluation = _main_evaluation(runner, spec, timeframe, folds)
    selected_params = [
        result.selected_params
        for result in evaluation.fold_results
    ]
    if len(selected_params) != 3 or any(params is None for params in selected_params):
        raise RuntimeError(
            f"{spec.name}/{timeframe}: no valid exact selection in every fold"
        )
    exact_params = [dict(params) for params in selected_params if params is not None]
    cost_off, reference = _frozen_variant_metrics(
        runner,
        spec,
        timeframe,
        folds,
        exact_params,
    )
    primary_calendar, reference_calendar = _calendar_returns(
        runner,
        spec,
        timeframe,
        exact_params[-1],
        end=calendar_end,
        windows=(7, 30, 90),
        sizings=(PRIMARY_SIZING, REFERENCE_SIZING),
    )
    return {
        "strategy": spec.name,
        "family": spec.family,
        "timeframe": timeframe,
        "passed": passes_research_gates(evaluation),
        "adaptive_params": [
            {"fold": index, "params": deepcopy(params)}
            for index, params in enumerate(exact_params)
        ],
        "latest_params": deepcopy(exact_params[-1]),
        "oos": _metrics_dict(evaluation.metrics, folds=True),
        "neighbor": {
            "positive": int(evaluation.neighbor_positive),
            "total": int(evaluation.neighbor_valid),
            "robustness": float(evaluation.neighbor_robustness),
        },
        "cost_off": _metrics_dict(cost_off),
        "reference": {
            "oos": _metrics_dict(reference),
            "calendar_returns": reference_calendar,
        },
        "calendar_returns": primary_calendar,
    }


def _coverage_text(metadata: Mapping[str, object]) -> str:
    coverage = _require_mapping(metadata.get("coverage"), "metadata.coverage")
    days = [
        int(_require_mapping(item, f"coverage.{symbol}")["complete_utc_days"])
        for symbol, item in coverage.items()
        if symbol in metadata["universe"]
    ]
    return (
        f"{len(days)} instruments; {min(days)}-{max(days)} complete UTC days; "
        "missing source bars were not forward-filled"
    )


def _require_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _fingerprint(data: ResearchData) -> str:
    manifest = search_space_manifest()
    payload = {
        "requested_start": data.metadata["requested_start"],
        "requested_end": data.metadata["requested_end"],
        "universe": data.metadata["universe"],
        "grids": manifest["grids"],
        "timeframes": TIMEFRAMES,
        "cost_on": asdict(COST_ON_EXECUTION),
        "cost_off": asdict(COST_OFF_EXECUTION),
        "primary": asdict(PRIMARY_SIZING),
        "reference": asdict(REFERENCE_SIZING),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_checkpoint(
    path: Path,
    *,
    fingerprint: str,
) -> dict[tuple[str, str], dict[str, object]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        raise ValueError(
            f"checkpoint fingerprint mismatch: remove {path} or use --fresh"
        )
    results = {}
    for result in payload.get("results", []):
        key = (str(result["strategy"]), str(result["timeframe"]))
        results[key] = result
    return results


def _checkpoint_payload(
    *,
    fingerprint: str,
    started_at: str,
    results: Mapping[tuple[str, str], Mapping[str, object]],
    elapsed_seconds: float,
) -> dict[str, object]:
    ordered = [
        results[(strategy, timeframe)]
        for strategy in STRATEGY_SPECS
        for timeframe in TIMEFRAMES
        if (strategy, timeframe) in results
    ]
    return {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "started_at": started_at,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "completed": len(ordered),
        "expected": len(STRATEGY_SPECS) * len(TIMEFRAMES),
        "results": ordered,
    }


def _calibration(
    runner: ResearchRunner,
    *,
    calendar_end: pd.Timestamp,
    universe_size: int,
) -> dict[str, object]:
    legacy_sizing = SizingConfig(
        initial_equity=100_000.0,
        margin_fraction=1.0 / universe_size,
        leverage=1.0,
        max_positions=universe_size,
    )
    primary_returns, legacy_returns = _calendar_returns(
        runner,
        STRATEGY_SPECS["donchian"],
        "15m",
        CALIBRATION_PARAMS,
        end=calendar_end,
        windows=(7, 30, 56),
        sizings=(PRIMARY_SIZING, legacy_sizing),
    )
    return {
        "strategy": "donchian",
        "timeframe": "15m",
        "direction": "both",
        "params": dict(CALIBRATION_PARAMS),
        "primary_returns": primary_returns,
        "legacy_equal_weight_returns": legacy_returns,
        "legacy_method": (
            "engine-compatible approximation: 1/N current-equity margin, "
            "1x leverage, max N simultaneous positions"
        ),
        "user_reference_returns": dict(USER_CALIBRATION_RETURNS),
    }


def run_full_research(
    *,
    cache_dir: Path | str,
    checkpoint_path: Path | str,
    summary_path: Path | str,
    markdown_path: Path | str,
    html_path: Path | str,
    fresh: bool = False,
) -> dict[str, object]:
    """Run or resume all 28 family/timeframe walk-forward candidates."""
    started_clock = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    data = load_research_data(cache_dir)
    folds = build_research_folds(data.metadata)
    calendar_end = _timestamp(data.metadata["requested_end"]).normalize() + pd.Timedelta(
        days=1
    )
    runner = ResearchRunner(data)
    fingerprint = _fingerprint(data)
    checkpoint = Path(checkpoint_path)
    if fresh:
        checkpoint.unlink(missing_ok=True)
    completed = _load_checkpoint(checkpoint, fingerprint=fingerprint)
    expected = len(STRATEGY_SPECS) * len(TIMEFRAMES)

    for strategy_name, spec in STRATEGY_SPECS.items():
        for timeframe in TIMEFRAMES:
            key = (strategy_name, timeframe)
            ordinal = (
                list(STRATEGY_SPECS).index(strategy_name) * len(TIMEFRAMES)
                + list(TIMEFRAMES).index(timeframe)
                + 1
            )
            if key in completed:
                print(
                    f"[{ordinal:02d}/{expected}] {strategy_name}/{timeframe}: "
                    "checkpoint hit",
                    flush=True,
                )
                continue
            result_started = time.perf_counter()
            completed[key] = _candidate_result(
                runner,
                spec,
                timeframe,
                folds,
                calendar_end=calendar_end,
            )
            result_elapsed = time.perf_counter() - result_started
            print(
                f"[{ordinal:02d}/{expected}] {strategy_name}/{timeframe}: "
                f"return={completed[key]['oos']['return']:.6f} "
                f"sharpe={completed[key]['oos']['sharpe']:.3f} "
                f"pass={completed[key]['passed']} elapsed={result_elapsed:.1f}s "
                f"backtests={runner.backtests} cache_hits={runner.cache_hits}",
                flush=True,
            )
            write_json_atomic(
                checkpoint,
                _checkpoint_payload(
                    fingerprint=fingerprint,
                    started_at=started_at,
                    results=completed,
                    elapsed_seconds=time.perf_counter() - started_clock,
                ),
            )
            runner.clear_candidate_caches()

    candidates = [deepcopy(value) for value in completed.values()]
    candidates.sort(
        key=lambda candidate: (
            -float(candidate["oos"]["sharpe"]),
            -float(candidate["oos"]["return"]),
            str(candidate["strategy"]),
            str(candidate["timeframe"]),
        )
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    calibration = _calibration(
        runner,
        calendar_end=calendar_end,
        universe_size=len(data.metadata["universe"]),
    )
    elapsed = time.perf_counter() - started_clock
    manifest = search_space_manifest()
    search_counts = {
        key: value
        for key, value in manifest.items()
        if key not in {"grids", "strategies", "timeframes"}
    }
    search_counts.update(
        {
            "strategies": len(STRATEGY_SPECS),
            "timeframes": len(TIMEFRAMES),
            "actual_backtests": runner.backtests,
            "backtest_cache_hits": runner.cache_hits,
            "signal_generations": runner.signal_generations,
        }
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "metadata_generated_at": data.metadata.get("generated_at"),
            "source": data.metadata.get("source"),
            "requested_start": data.metadata["requested_start"],
            "requested_end": data.metadata["requested_end"],
            "universe": list(data.metadata["universe"]),
            "timeframes": list(TIMEFRAMES),
            "coverage": _coverage_text(data.metadata),
        },
        "research": {
            "cost_model": {
                **asdict(COST_ON_EXECUTION),
                "funding": "realized OKX funding at the published timestamp",
                "cost_off": "fee=0, slippage=0, funding removed",
            },
            "primary_sizing": asdict(PRIMARY_SIZING),
            "reference_sizing": asdict(REFERENCE_SIZING),
            "folds": [
                {
                    "fold": fold.index,
                    "is_start": fold.is_start.isoformat(),
                    "is_end": fold.is_end.isoformat(),
                    "oos_start": fold.oos_start.isoformat(),
                    "oos_end": fold.oos_end.isoformat(),
                }
                for fold in folds
            ],
            "calendar_window_end": calendar_end.isoformat(),
            "calendar_window_definition": (
                "last complete UTC calendar days; latest fold exact params; "
                "continuous cached history for indicators; flat at window start"
            ),
            "neighbor_rule": (
                "unique legal one-step grid neighbors; positive compounded "
                "OOS return divided by all unique neighbors, with invalid, "
                "flat, and losing neighbors nonpositive"
            ),
            "ranking": "OOS daily Sharpe descending, then OOS return descending",
            "passing_gates": {
                "positive_oos_folds": "strict majority: at least 2 of exactly 3",
                "minimum_oos_trades": 30,
                "maximum_drawdown_floor": -0.30,
                "minimum_neighbor_robustness": 0.70,
            },
            "search_counts": search_counts,
            "elapsed_seconds": round(elapsed, 6),
            "checkpoint": str(checkpoint),
        },
        "candidates": candidates,
        "passing_count": sum(bool(candidate["passed"]) for candidate in candidates),
        "calibration": calibration,
        "warnings": [
            (
                "Multiple testing across complete bounded grids can overstate "
                "apparent robustness; no multiplicity correction was applied."
            ),
            (
                "The selected universe reflects one current liquidity snapshot "
                "and may introduce survivorship and selection bias."
            ),
            (
                "Ninety days and three OOS folds are limited evidence; regime "
                "changes, latency, liquidity, and liquidation are not modeled."
            ),
            (
                "Research results are not evidence that a strategy is ready "
                "for live trading; independent validation and paper trading "
                "are required."
            ),
        ],
    }
    validate_summary(summary)
    write_json_atomic(summary_path, summary)
    write_reports(summary, markdown_path, html_path)
    write_json_atomic(
        checkpoint,
        _checkpoint_payload(
            fingerprint=fingerprint,
            started_at=started_at,
            results=completed,
            elapsed_seconds=elapsed,
        ),
    )
    return summary


def run_smoke(cache_dir: Path | str) -> dict[str, object]:
    """Run one real-data calibration window before starting the full search."""
    data = load_research_data(cache_dir)
    runner = ResearchRunner(data)
    end = _timestamp(data.metadata["requested_end"]).normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=2)
    result = runner.run(
        STRATEGY_SPECS["donchian"],
        "15m",
        CALIBRATION_PARAMS,
        start=start,
        end=end,
        sizing=PRIMARY_SIZING,
    )
    metrics = compute_metrics(_equity_series(result), result.trades)
    return {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "metrics": _metrics_dict(metrics),
        "backtests": runner.backtests,
        "signal_generations": runner.signal_generations,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/okx_cost_aware"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/okx_cost_aware/research_checkpoint.json"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/cost_aware_summary.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("reports/cost_aware_results.md"),
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("reports/cost_aware_results.html"),
    )
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.smoke:
        print(
            json.dumps(run_smoke(arguments.cache_dir), indent=2, sort_keys=True),
            flush=True,
        )
        return 0
    summary = run_full_research(
        cache_dir=arguments.cache_dir,
        checkpoint_path=arguments.checkpoint,
        summary_path=arguments.summary,
        markdown_path=arguments.markdown,
        html_path=arguments.html,
        fresh=arguments.fresh,
    )
    print(
        json.dumps(
            {
                "passing_count": summary["passing_count"],
                "candidate_count": len(summary["candidates"]),
                "summary": str(arguments.summary),
                "markdown": str(arguments.markdown),
                "html": str(arguments.html),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
