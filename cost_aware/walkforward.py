"""Leakage-resistant walk-forward selection and event-equity metrics.

The backtest callback is deliberately independent of data loading.  It receives
``(params, fold, sizing, phase)`` where phase is ``"is"`` or ``"oos"``, and
returns either ``BacktestResult`` or ``(event_equity, trades)``.  Equity must
contain the actual event-time account value, including values at both window
boundaries; metrics are never rebuilt from exit-day trade PnL.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Real
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd


Params = Mapping[str, object]
Phase = Literal["is", "oos"]
BacktestOutput = tuple[object, Sequence[object]]
BacktestCallback = Callable[
    [Params, "Fold", object, Phase],
    BacktestOutput | None,
]
NeighborCallback = Callable[[Params], Iterable[Params]]
_NEIGHBOR_DATA_ERRORS = (
    TypeError,
    ValueError,
    AttributeError,
    KeyError,
    IndexError,
)


@dataclass(frozen=True)
class Fold:
    """One rolling in-sample window and its immediately adjacent OOS window.

    Window boundaries are half-open for data selection: ``[start, end)``.
    An equity observation at ``end`` is allowed because it can represent the
    account value after the final interval in the window.
    """

    index: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp

    def __post_init__(self) -> None:
        boundaries = (
            self.is_start,
            self.is_end,
            self.oos_start,
            self.oos_end,
        )
        if self.index < 0:
            raise ValueError("fold index cannot be negative")
        if any(not isinstance(value, pd.Timestamp) for value in boundaries):
            raise TypeError("fold boundaries must be pandas timestamps")
        if any(value.tzinfo is None for value in boundaries):
            raise ValueError("fold boundaries must be timezone-aware")
        if any(
            value.tz_convert("UTC") != value.tz_convert("UTC").normalize()
            for value in boundaries
        ):
            raise ValueError("fold boundaries must be aligned to UTC midnight")
        if not self.is_start < self.is_end:
            raise ValueError("in-sample window must have positive duration")
        if self.is_end != self.oos_start:
            raise ValueError("OOS must immediately follow in-sample")
        if not self.oos_start < self.oos_end:
            raise ValueError("OOS window must have positive duration")


@dataclass(frozen=True)
class Metrics:
    """Metrics derived from event equity and closed trades."""

    total_return: float
    sharpe: float
    max_drawdown: float
    trades: int
    win_rate: float
    fold_returns: tuple[float, ...] = ()


@dataclass(frozen=True)
class FoldEvaluation:
    """The exact IS choice frozen before its adjacent OOS evaluation."""

    fold: Fold
    selected_params: dict[str, object] | None
    is_metrics: Metrics | None
    oos_metrics: Metrics | None


@dataclass(frozen=True)
class CandidateEvaluation:
    """Walk-forward OOS result plus first-neighbor robustness."""

    fold_results: tuple[FoldEvaluation, ...]
    metrics: Metrics
    neighbor_robustness: float
    neighbor_positive: int
    neighbor_valid: int


def build_folds(
    start: object,
    *,
    is_days: int = 45,
    oos_days: int = 15,
    n_folds: int = 3,
    step_days: int | None = None,
) -> tuple[Fold, ...]:
    """Build fixed-length rolling IS windows with non-overlapping OOS windows.

    Defaults produce three 15-day OOS folds after rolling 45-day IS windows,
    spanning 90 days in total. ``step_days`` must equal ``oos_days`` so all OOS
    evidence is contiguous as well as non-overlapping.
    """

    values = {
        "is_days": is_days,
        "oos_days": oos_days,
        "n_folds": n_folds,
    }
    if step_days is not None:
        values["step_days"] = step_days
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    step = oos_days if step_days is None else step_days
    if step != oos_days:
        raise ValueError(
            "step_days must equal oos_days so OOS folds remain adjacent"
        )

    origin = pd.Timestamp(start)
    if origin.tzinfo is None:
        origin = origin.tz_localize("UTC")
    else:
        origin = origin.tz_convert("UTC")
    if origin != origin.normalize():
        raise ValueError("fold start must be aligned to UTC midnight")
    is_delta = pd.Timedelta(days=is_days)
    oos_delta = pd.Timedelta(days=oos_days)
    step_delta = pd.Timedelta(days=step)
    folds = []
    for index in range(n_folds):
        is_start = origin + index * step_delta
        is_end = is_start + is_delta
        folds.append(
            Fold(
                index=index,
                is_start=is_start,
                is_end=is_end,
                oos_start=is_end,
                oos_end=is_end + oos_delta,
            )
        )
    return tuple(folds)


def compute_metrics(
    equity: pd.Series,
    trades: Sequence[object],
    *,
    fold_returns: Iterable[float] = (),
) -> Metrics:
    """Compute return, daily Sharpe, drawdown, and trade statistics.

    Account values are sampled at consecutive UTC day boundaries and forward
    filled.  Consequently quiet days contribute zero daily returns instead of
    disappearing from the Sharpe sample.  Intraday event values remain in the
    drawdown path.
    """

    event_equity = _coerce_equity(equity)
    closed_trades = list(trades)
    normalized_fold_returns = tuple(float(value) for value in fold_returns)
    if any(not isfinite(value) for value in normalized_fold_returns):
        raise ValueError("fold returns must be finite")

    total_return = float(event_equity.iloc[-1] / event_equity.iloc[0] - 1.0)
    running_peak = event_equity.cummax()
    max_drawdown = float((event_equity / running_peak - 1.0).min())

    if not closed_trades:
        sharpe = 0.0
        win_rate = 0.0
    else:
        daily_returns = _daily_returns(event_equity)
        standard_deviation = float(daily_returns.std(ddof=1))
        if (
            len(daily_returns) < 2
            or not isfinite(standard_deviation)
            or standard_deviation == 0.0
        ):
            sharpe = 0.0
        else:
            sharpe = float(
                daily_returns.mean() / standard_deviation * sqrt(365.0)
            )
        wins = sum(_trade_net_pnl(trade) > 0.0 for trade in closed_trades)
        win_rate = wins / len(closed_trades)

    return Metrics(
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        trades=len(closed_trades),
        win_rate=float(win_rate),
        fold_returns=normalized_fold_returns,
    )


def evaluate_candidate(
    parameter_grid: Iterable[Params],
    folds: Iterable[Fold],
    *,
    sizing: object,
    callback: BacktestCallback,
    neighbors: NeighborCallback,
) -> CandidateEvaluation:
    """Select independently on each IS fold, freeze, and evaluate adjacent OOS.

    IS candidates rank by annualized daily Sharpe, then compounded return.
    Input grid order is the deterministic final tie-break.  Only the chosen
    parameters are sent to that fold's OOS callback.  First neighbors of each
    frozen choice are evaluated on that same OOS window.

    A callback may return ``None`` to mark an unavailable/invalid evaluation.
    Neighbor results fail closed: every unique legal neighbor is in the
    denominator, while ``None``, malformed, losing, and flat results are
    nonpositive. Repeated canonical neighbors are chained across their OOS
    folds before their compounded return is classified.
    """

    params = [deepcopy(dict(candidate)) for candidate in parameter_grid]
    if not params:
        raise ValueError("parameter_grid cannot be empty")
    ordered_folds = tuple(folds)
    _validate_folds(ordered_folds)

    fold_results: list[FoldEvaluation] = []
    oos_equities: list[pd.Series] = []
    oos_trades: list[object] = []
    fold_returns: list[float] = []
    neighbor_results: dict[
        object,
        dict[str, object],
    ] = {}

    for fold in ordered_folds:
        best_params: dict[str, object] | None = None
        best_metrics: Metrics | None = None
        best_score: tuple[float, float] | None = None

        for candidate in params:
            output = callback(deepcopy(candidate), fold, sizing, "is")
            resolved = _resolve_output(output, fold, "is")
            if resolved is None:
                continue
            equity, candidate_trades = resolved
            candidate_metrics = compute_metrics(equity, candidate_trades)
            score = (candidate_metrics.sharpe, candidate_metrics.total_return)
            if best_score is None or score > best_score:
                best_params = deepcopy(candidate)
                best_metrics = candidate_metrics
                best_score = score

        if best_params is None:
            fold_results.append(
                FoldEvaluation(
                    fold=fold,
                    selected_params=None,
                    is_metrics=None,
                    oos_metrics=None,
                )
            )
            continue

        frozen_params = deepcopy(best_params)
        output = callback(deepcopy(frozen_params), fold, sizing, "oos")
        resolved_oos = _resolve_output(output, fold, "oos")
        if resolved_oos is None:
            oos_metrics = None
        else:
            equity, trades = resolved_oos
            oos_metrics = compute_metrics(equity, trades)
            oos_equities.append(equity)
            oos_trades.extend(trades)
            fold_returns.append(oos_metrics.total_return)

        seen_neighbors: set[object] = set()
        selected_key = _params_key(frozen_params)
        for neighbor in neighbors(deepcopy(frozen_params)):
            neighbor_copy = deepcopy(dict(neighbor))
            key = _params_key(neighbor_copy)
            if key == selected_key or key in seen_neighbors:
                continue
            seen_neighbors.add(key)
            aggregate = neighbor_results.setdefault(
                key,
                {
                    "equities": [],
                    "trades": [],
                    "valid": True,
                },
            )
            try:
                neighbor_output = callback(
                    neighbor_copy,
                    fold,
                    sizing,
                    "oos",
                )
                resolved_neighbor = _resolve_output(
                    neighbor_output,
                    fold,
                    "oos",
                )
            except _NEIGHBOR_DATA_ERRORS:
                aggregate["valid"] = False
                continue
            if resolved_neighbor is None:
                aggregate["valid"] = False
                continue
            neighbor_equity, neighbor_trades = resolved_neighbor
            aggregate["equities"].append(neighbor_equity)
            aggregate["trades"].extend(neighbor_trades)

        fold_results.append(
            FoldEvaluation(
                fold=fold,
                selected_params=deepcopy(frozen_params),
                is_metrics=best_metrics,
                oos_metrics=oos_metrics,
            )
        )

    if oos_equities:
        combined_equity = _chain_equity(oos_equities)
        metrics = compute_metrics(
            combined_equity,
            oos_trades,
            fold_returns=fold_returns,
        )
    else:
        metrics = _empty_metrics()
    neighbor_positive = 0
    for aggregate in neighbor_results.values():
        equities = aggregate["equities"]
        if not aggregate["valid"] or not equities:
            continue
        try:
            neighbor_equity = _chain_equity(equities)
            neighbor_metrics = compute_metrics(
                neighbor_equity,
                aggregate["trades"],
            )
        except _NEIGHBOR_DATA_ERRORS:
            continue
        if neighbor_metrics.total_return > 0.0:
            neighbor_positive += 1
    neighbor_valid = len(neighbor_results)
    robustness = neighbor_positive / neighbor_valid if neighbor_valid else 0.0
    return CandidateEvaluation(
        fold_results=tuple(fold_results),
        metrics=metrics,
        neighbor_robustness=float(robustness),
        neighbor_positive=neighbor_positive,
        neighbor_valid=neighbor_valid,
    )


def passes_research_gates(
    evaluation: CandidateEvaluation | Metrics,
    *,
    neighbor_robustness: float | None = None,
) -> bool:
    """Apply the final OOS gates while retaining Sharpe as a reported metric."""

    if isinstance(evaluation, CandidateEvaluation):
        metrics = evaluation.metrics
        robustness = evaluation.neighbor_robustness
        fold_results = evaluation.fold_results
        if neighbor_robustness is not None:
            raise ValueError(
                "neighbor_robustness is already part of CandidateEvaluation"
            )
        if len(fold_results) != 3:
            return False
        try:
            _validate_folds(tuple(result.fold for result in fold_results))
        except ValueError:
            return False
    elif isinstance(evaluation, Metrics):
        metrics = evaluation
        if neighbor_robustness is None:
            raise ValueError("neighbor_robustness is required with Metrics")
        robustness = float(neighbor_robustness)
    else:
        raise TypeError("evaluation must be CandidateEvaluation or Metrics")

    usable_folds = len(metrics.fold_returns)
    positive_folds = sum(value > 0.0 for value in metrics.fold_returns)
    return (
        usable_folds == 3
        and positive_folds > usable_folds / 2
        and metrics.trades >= 30
        and metrics.max_drawdown >= -0.30
        and robustness >= 0.70
    )


def _coerce_equity(equity: pd.Series) -> pd.Series:
    if not isinstance(equity, pd.Series):
        raise TypeError("equity must be a pandas Series")
    if equity.empty:
        raise ValueError("equity cannot be empty")
    values = pd.to_numeric(equity, errors="coerce").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("equity values must be finite")
    if (values <= 0.0).any():
        raise ValueError("equity values must be positive")

    if isinstance(values.index, pd.DatetimeIndex):
        timestamps = values.index
        if timestamps.tz is None:
            raise ValueError("equity timestamps must be timezone-aware")
        if timestamps.has_duplicates:
            raise ValueError(
                "duplicate timestamps require a timestamp/sequence MultiIndex"
            )
        values = values.sort_index(kind="stable")
        timestamps = values.index.tz_convert("UTC")
        sequences = range(len(values))
    elif isinstance(values.index, pd.MultiIndex) and values.index.nlevels == 2:
        timestamps = pd.DatetimeIndex(values.index.get_level_values(0))
        if timestamps.tz is None:
            raise ValueError("equity timestamps must be timezone-aware")
        raw_sequences = list(values.index.get_level_values(1))
        if any(
            isinstance(sequence, bool)
            or not isinstance(sequence, (int, np.integer))
            for sequence in raw_sequences
        ):
            raise TypeError("event sequence must contain integers")
        if any(
            right <= left
            for left, right in zip(raw_sequences, raw_sequences[1:])
        ):
            raise ValueError("event sequence must be strictly increasing")
        if not timestamps.is_monotonic_increasing:
            raise ValueError("event timestamps must be nondecreasing")
        timestamps = timestamps.tz_convert("UTC")
        sequences = raw_sequences
    else:
        raise TypeError(
            "equity index must be a DatetimeIndex or timestamp/sequence MultiIndex"
        )

    values.index = pd.MultiIndex.from_arrays(
        [timestamps, sequences],
        names=["timestamp", "sequence"],
    )
    return values


def _trade_net_pnl(trade: object) -> float:
    if isinstance(trade, Mapping):
        value = trade.get("net_pnl")
    else:
        value = getattr(trade, "net_pnl", None)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("each trade must expose numeric net_pnl")
    result = float(value)
    if not isfinite(result):
        raise ValueError("trade net_pnl must be finite")
    return result


def _equity_timestamps(event_equity: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(event_equity.index.get_level_values("timestamp"))


def _series_from_events(events: Iterable[object]) -> pd.Series:
    records = list(events)
    index = pd.MultiIndex.from_arrays(
        [
            [getattr(event, "timestamp") for event in records],
            [getattr(event, "sequence") for event in records],
        ],
        names=["timestamp", "sequence"],
    )
    return pd.Series(
        [getattr(event, "equity") for event in records],
        index=index,
        dtype=float,
    )


def _trade_field(trade: object, name: str) -> object:
    if isinstance(trade, Mapping):
        if name not in trade:
            raise TypeError(f"each trade must expose {name}")
        return trade[name]
    if not hasattr(trade, name):
        raise TypeError(f"each trade must expose {name}")
    return getattr(trade, name)


def _validate_trades(
    trades: Sequence[object],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    for trade in trades:
        entry_time = _trade_field(trade, "entry_time")
        exit_time = _trade_field(trade, "exit_time")
        exit_reason = _trade_field(trade, "exit_reason")
        if not start <= entry_time < end:
            raise ValueError("trade entry_time lies outside the fold window")
        if exit_time == end:
            if exit_reason != "force_close":
                raise ValueError(
                    "trade exit_time at fold end requires force_close"
                )
        elif not start <= exit_time < end:
            raise ValueError("trade exit_time lies outside the fold window")
        if exit_time < entry_time:
            raise ValueError("trade exit_time precedes entry_time")


def _daily_returns(event_equity: pd.Series) -> pd.Series:
    event_timestamps = _equity_timestamps(event_equity)
    start = event_timestamps[0]
    end = event_timestamps[-1]
    initial_equity = float(event_equity.iloc[0])
    timestamp_equity = pd.Series(
        event_equity.to_numpy(),
        index=event_timestamps,
    ).groupby(level=0, sort=False).last()
    first_midnight = start.normalize()
    if first_midnight <= start:
        first_midnight += pd.Timedelta(days=1)
    interior = pd.date_range(
        first_midnight,
        end,
        freq="1D",
        inclusive="left",
    )
    later_boundaries = pd.DatetimeIndex([*interior, end])
    later_values = (
        timestamp_equity.reindex(
            timestamp_equity.index.union(later_boundaries)
        )
        .sort_index(kind="stable")
        .ffill()
        .reindex(later_boundaries)
    )
    values = pd.Series(
        [initial_equity, *later_values.to_list()],
        index=pd.DatetimeIndex([start, *later_boundaries]),
    )
    return values.pct_change(fill_method=None).dropna()


def _validate_folds(folds: tuple[Fold, ...]) -> None:
    if not folds:
        raise ValueError("folds cannot be empty")
    if len({fold.index for fold in folds}) != len(folds):
        raise ValueError("fold indexes must be unique")
    for left, right in zip(folds, folds[1:]):
        if left.oos_start > right.oos_start:
            raise ValueError("folds must be ordered by OOS start")
        if left.oos_end > right.oos_start:
            raise ValueError("OOS folds cannot overlap")
        if left.oos_end != right.oos_start:
            raise ValueError("OOS folds must be adjacent")


def _resolve_output(
    output: BacktestOutput | None,
    fold: Fold,
    phase: Phase,
) -> tuple[pd.Series, list[object]] | None:
    if output is None:
        return None
    if hasattr(output, "equity_events") and hasattr(output, "trades"):
        equity = _series_from_events(output.equity_events)
        trades = output.trades
    elif isinstance(output, tuple) and len(output) == 2:
        equity, trades = output
    else:
        raise TypeError(
            "callback must return BacktestResult, (event_equity, trades), or None"
        )
    resolved_equity = _coerce_equity(equity)
    start, end = (
        (fold.is_start, fold.is_end)
        if phase == "is"
        else (fold.oos_start, fold.oos_end)
    )
    timestamps = _equity_timestamps(resolved_equity)
    if timestamps[0] != start or timestamps[-1] != end:
        raise ValueError(
            f"{phase} equity must cover both fold window boundaries"
        )
    resolved_trades = list(trades)
    _validate_trades(resolved_trades, start, end)
    return resolved_equity, resolved_trades


def _chain_equity(segments: Sequence[pd.Series]) -> pd.Series:
    chained_values: list[float] = []
    chained_timestamps: list[pd.Timestamp] = []
    level = 1.0
    previous_end: pd.Timestamp | None = None
    for raw_segment in segments:
        segment = _coerce_equity(raw_segment)
        timestamps = _equity_timestamps(segment)
        if previous_end is not None and timestamps[0] < previous_end:
            raise ValueError("OOS equity segments overlap")
        normalized = segment / float(segment.iloc[0]) * level
        if previous_end is not None and timestamps[0] == previous_end:
            normalized = normalized.iloc[1:]
            timestamps = timestamps[1:]
        chained_values.extend(float(value) for value in normalized)
        chained_timestamps.extend(timestamps)
        level *= float(segment.iloc[-1] / segment.iloc[0])
        previous_end = _equity_timestamps(segment)[-1]
    if not chained_values:
        first = _coerce_equity(segments[0])
        chained_values = [1.0]
        chained_timestamps = [_equity_timestamps(first)[0]]
    index = pd.MultiIndex.from_arrays(
        [chained_timestamps, range(len(chained_values))],
        names=["timestamp", "sequence"],
    )
    return pd.Series(chained_values, index=index)


def _params_key(params: Params) -> object:
    try:
        return tuple(sorted(params.items()))
    except TypeError:
        return repr(sorted((key, repr(value)) for key, value in params.items()))


def _empty_metrics() -> Metrics:
    return Metrics(
        total_return=0.0,
        sharpe=0.0,
        max_drawdown=0.0,
        trades=0,
        win_rate=0.0,
        fold_returns=(),
    )
