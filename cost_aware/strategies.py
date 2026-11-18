"""Deterministic closed-candle strategy signals for cost-aware research.

Every emitted row is an instruction for :mod:`cost_aware.engine`: its timestamp
is the fully closed signal bar, while the engine applies its configured
``signal_lag_bars`` before filling at a later open.  Time stops and cooldowns
are intentionally absent because the current engine has no interface for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Real
from typing import Mapping

import numpy as np
import pandas as pd


Parameter = int | float | str
ParameterSet = Mapping[str, Parameter]
SIGNAL_COLUMNS = ["symbol", "timestamp", "side", "stop", "take_profit"]


@dataclass(frozen=True)
class StrategySpec:
    """A named strategy and its ordered, finite research parameter space."""

    name: str
    family: str
    parameter_space: tuple[tuple[str, tuple[Parameter, ...]], ...]

    def __post_init__(self) -> None:
        if not self.name or not self.family:
            raise ValueError("strategy name and family cannot be empty")
        names = [name for name, _ in self.parameter_space]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: duplicate parameter name")
        if any(not values for _, values in self.parameter_space):
            raise ValueError(f"{self.name}: parameter choices cannot be empty")
        grid_size = int(np.prod([len(values) for _, values in self.parameter_space]))
        if grid_size > 16:
            raise ValueError(f"{self.name}: parameter grid exceeds research bound")


STRATEGY_SPECS: dict[str, StrategySpec] = {
    "donchian": StrategySpec(
        name="donchian",
        family="trend_breakout",
        parameter_space=(
            ("lookback", (20, 40)),
            ("atr_window", (14,)),
            ("stop_atr", (2.0, 3.5)),
            ("take_profit_atr", (3.0, 4.0)),
        ),
    ),
    "keltner": StrategySpec(
        name="keltner",
        family="trend_breakout",
        parameter_space=(
            ("ema_window", (20, 40)),
            ("atr_window", (14,)),
            ("band_atr", (1.5, 2.0)),
            ("stop_atr", (2.0, 3.5)),
            ("take_profit_atr", (4.0,)),
        ),
    ),
    "rsi_mean_reversion": StrategySpec(
        name="rsi_mean_reversion",
        family="mean_reversion",
        parameter_space=(
            ("rsi_window", (7, 14)),
            ("threshold", (25.0, 30.0)),
            ("atr_window", (14,)),
            ("stop_atr", (2.0, 3.5)),
            ("take_profit_atr", (3.0,)),
        ),
    ),
    "vwap_mean_reversion": StrategySpec(
        name="vwap_mean_reversion",
        family="mean_reversion",
        parameter_space=(
            ("vwap_window", (20, 40)),
            ("atr_window", (14,)),
            ("deviation_atr", (1.0, 1.5)),
            ("stop_atr", (2.0, 3.5)),
            ("take_profit_atr", (3.0,)),
        ),
    ),
    "extreme_reversal": StrategySpec(
        name="extreme_reversal",
        family="extreme_reversal",
        parameter_space=(
            ("return_lookback", (1,)),
            ("return_threshold", (0.02, 0.04)),
            ("volume_window", (20, 40)),
            ("volume_multiplier", (1.5, 2.0)),
            ("direction", ("both", "short")),
            ("atr_window", (14,)),
            ("stop_atr", (3.5,)),
            ("take_profit_atr", (4.0,)),
        ),
    ),
    "cross_sectional_momentum": StrategySpec(
        name="cross_sectional_momentum",
        family="cross_sectional_rotation",
        parameter_space=(
            ("lookback", (12, 24)),
            ("selection_count", (1, 2)),
            ("atr_window", (14,)),
            ("stop_atr", (2.0, 3.5)),
            ("take_profit_atr", (4.0,)),
        ),
    ),
    "regime_gated": StrategySpec(
        name="regime_gated",
        family="regime_gated",
        parameter_space=(
            ("fast_window", (20, 40)),
            ("slow_window", (50, 100)),
            ("breakout_lookback", (10, 20)),
            ("atr_window", (14,)),
            ("stop_atr", (3.5,)),
            ("take_profit_atr", (4.0,)),
        ),
    ),
}


def parameter_grid(spec: StrategySpec | str) -> list[dict[str, Parameter]]:
    """Return the ordered Cartesian product for a bounded strategy spec."""
    resolved = _resolve_spec(spec)
    names = [name for name, _ in resolved.parameter_space]
    return [
        dict(zip(names, combination, strict=True))
        for combination in product(
            *(values for _, values in resolved.parameter_space)
        )
    ]


def neighbor_params(
    spec: StrategySpec | str,
    params: ParameterSet,
) -> list[dict[str, Parameter]]:
    """Return configurations one adjacent grid step away in one dimension."""
    resolved = _resolve_spec(spec)
    current = _validate_parameter_keys(resolved, params)
    values_by_name = dict(resolved.parameter_space)
    if any(current[name] not in values for name, values in values_by_name.items()):
        raise ValueError(f"{resolved.name}: parameters must belong to its grid")

    neighbors: list[dict[str, Parameter]] = []
    for name, values in resolved.parameter_space:
        index = values.index(current[name])
        for neighbor_index in (index - 1, index + 1):
            if 0 <= neighbor_index < len(values):
                neighbor = dict(current)
                neighbor[name] = values[neighbor_index]
                neighbors.append(neighbor)
    return neighbors


def generate_signals(
    candles: pd.DataFrame,
    spec: StrategySpec | str,
    params: ParameterSet,
) -> pd.DataFrame:
    """Generate causal signal rows from OHLCV bars.

    Input accepts the cached parquet timestamp name ``open_time`` or the
    engine-facing name ``timestamp``. Indicators are grouped by symbol and use
    only each row and earlier rows; prior-window breakouts explicitly shift the
    channel by one bar.
    """
    resolved = _resolve_spec(spec)
    values = _validate_parameter_keys(resolved, params)
    frame = _prepare_candles(candles)
    atr_window = _positive_int(values, "atr_window")
    frame["atr"] = _group_indicator(
        frame, lambda group: _average_true_range(group, atr_window)
    )

    if resolved.name == "donchian":
        long_mask, short_mask = _donchian(frame, values)
        return _directional_signals(frame, long_mask, short_mask, values)
    if resolved.name == "keltner":
        long_mask, short_mask = _keltner(frame, values)
        return _directional_signals(frame, long_mask, short_mask, values)
    if resolved.name == "rsi_mean_reversion":
        long_mask, short_mask = _rsi_mean_reversion(frame, values)
        return _directional_signals(frame, long_mask, short_mask, values)
    if resolved.name == "vwap_mean_reversion":
        long_mask, short_mask = _vwap_mean_reversion(frame, values)
        return _directional_signals(frame, long_mask, short_mask, values)
    if resolved.name == "extreme_reversal":
        long_mask, short_mask = _extreme_reversal(frame, values)
        return _directional_signals(frame, long_mask, short_mask, values)
    if resolved.name == "cross_sectional_momentum":
        return _cross_sectional_momentum(frame, values)
    if resolved.name == "regime_gated":
        long_mask, short_mask = _regime_gated(frame, values)
        return _directional_signals(frame, long_mask, short_mask, values)
    raise ValueError(f"unsupported strategy: {resolved.name}")


def _resolve_spec(spec: StrategySpec | str) -> StrategySpec:
    if isinstance(spec, StrategySpec):
        return spec
    try:
        return STRATEGY_SPECS[spec]
    except KeyError as error:
        raise ValueError(f"unknown strategy: {spec}") from error


def _validate_parameter_keys(
    spec: StrategySpec,
    params: ParameterSet,
) -> dict[str, Parameter]:
    expected = [name for name, _ in spec.parameter_space]
    missing = set(expected).difference(params)
    extra = set(params).difference(expected)
    if missing or extra:
        raise ValueError(
            f"{spec.name}: parameter mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return {name: params[name] for name in expected}


def _positive_int(params: ParameterSet, name: str) -> int:
    value = params[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(params: ParameterSet, name: str) -> float:
    value = params[name]
    if isinstance(value, bool) or not isinstance(value, Real) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _prepare_candles(candles: pd.DataFrame) -> pd.DataFrame:
    timestamp_column = "timestamp" if "timestamp" in candles else "open_time"
    required = {
        "symbol",
        timestamp_column,
        "open",
        "high",
        "low",
        "close",
        "vol",
    }
    missing = required.difference(candles.columns)
    if missing:
        raise ValueError(f"incomplete strategy candles: {sorted(missing)}")

    frame = candles.loc[
        :,
        ["symbol", timestamp_column, "open", "high", "low", "close", "vol"],
    ].copy()
    frame = frame.rename(columns={timestamp_column: "timestamp"})
    frame["symbol"] = frame["symbol"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "vol"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.isna().any(axis=None):
        raise ValueError("strategy candles contain missing or invalid values")
    if frame.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("duplicate symbol/timestamp strategy candle")
    if (frame[["open", "high", "low", "close"]] <= 0).any(axis=None):
        raise ValueError("strategy candle prices must be positive")
    if (frame["vol"] < 0).any():
        raise ValueError("strategy candle volume cannot be negative")
    upper = frame[["open", "low", "close"]].max(axis=1)
    lower = frame[["open", "high", "close"]].min(axis=1)
    if (frame["high"] < upper).any() or (frame["low"] > lower).any():
        raise ValueError("strategy candle OHLC bounds are invalid")
    return frame.sort_values(["symbol", "timestamp"], kind="stable").reset_index(
        drop=True
    )


def _group_indicator(frame: pd.DataFrame, calculate) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby("symbol", sort=False):
        result.loc[group.index] = calculate(group).to_numpy()
    return result


def _average_true_range(group: pd.DataFrame, window: int) -> pd.Series:
    previous_close = group["close"].shift()
    true_range = pd.concat(
        [
            group["high"] - group["low"],
            (group["high"] - previous_close).abs(),
            (group["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=window).mean()


def _donchian(
    frame: pd.DataFrame, params: ParameterSet
) -> tuple[pd.Series, pd.Series]:
    lookback = _positive_int(params, "lookback")
    prior_high = _group_indicator(
        frame,
        lambda group: group["high"]
        .rolling(lookback, min_periods=lookback)
        .max()
        .shift(1),
    )
    prior_low = _group_indicator(
        frame,
        lambda group: group["low"]
        .rolling(lookback, min_periods=lookback)
        .min()
        .shift(1),
    )
    return frame["close"] > prior_high, frame["close"] < prior_low


def _keltner(
    frame: pd.DataFrame, params: ParameterSet
) -> tuple[pd.Series, pd.Series]:
    window = _positive_int(params, "ema_window")
    multiplier = _positive_number(params, "band_atr")
    middle = _group_indicator(
        frame,
        lambda group: group["close"].ewm(
            span=window, adjust=False, min_periods=window
        ).mean(),
    )
    return (
        frame["close"] > middle + multiplier * frame["atr"],
        frame["close"] < middle - multiplier * frame["atr"],
    )


def _rsi_mean_reversion(
    frame: pd.DataFrame, params: ParameterSet
) -> tuple[pd.Series, pd.Series]:
    window = _positive_int(params, "rsi_window")
    threshold = _positive_number(params, "threshold")
    if threshold >= 50:
        raise ValueError("threshold must be below 50")

    def calculate(group: pd.DataFrame) -> pd.Series:
        change = group["close"].diff()
        average_gain = change.clip(lower=0).rolling(
            window, min_periods=window
        ).mean()
        average_loss = (-change.clip(upper=0)).rolling(
            window, min_periods=window
        ).mean()
        denominator = average_gain + average_loss
        rsi = 100 * average_gain / denominator
        return rsi.mask(denominator == 0, 50.0)

    rsi = _group_indicator(frame, calculate)
    return rsi <= threshold, rsi >= 100 - threshold


def _vwap_mean_reversion(
    frame: pd.DataFrame, params: ParameterSet
) -> tuple[pd.Series, pd.Series]:
    window = _positive_int(params, "vwap_window")
    deviation = _positive_number(params, "deviation_atr")

    def calculate(group: pd.DataFrame) -> pd.Series:
        typical = (group["high"] + group["low"] + group["close"]) / 3
        numerator = (typical * group["vol"]).rolling(
            window, min_periods=window
        ).sum()
        denominator = group["vol"].rolling(window, min_periods=window).sum()
        return numerator / denominator.replace(0, np.nan)

    rolling_vwap = _group_indicator(frame, calculate)
    return (
        frame["close"] < rolling_vwap - deviation * frame["atr"],
        frame["close"] > rolling_vwap + deviation * frame["atr"],
    )


def _extreme_reversal(
    frame: pd.DataFrame, params: ParameterSet
) -> tuple[pd.Series, pd.Series]:
    return_lookback = _positive_int(params, "return_lookback")
    volume_window = _positive_int(params, "volume_window")
    threshold = _positive_number(params, "return_threshold")
    volume_multiplier = _positive_number(params, "volume_multiplier")
    direction = params["direction"]
    if direction not in {"both", "short"}:
        raise ValueError("direction must be 'both' or 'short'")

    returns = _group_indicator(
        frame, lambda group: group["close"].pct_change(return_lookback)
    )
    prior_volume = _group_indicator(
        frame,
        lambda group: group["vol"]
        .rolling(volume_window, min_periods=volume_window)
        .mean()
        .shift(1),
    )
    volume_extreme = frame["vol"] >= volume_multiplier * prior_volume
    long_mask = (returns <= -threshold) & volume_extreme
    if direction == "short":
        long_mask &= False
    return long_mask, (returns >= threshold) & volume_extreme


def _cross_sectional_momentum(
    frame: pd.DataFrame, params: ParameterSet
) -> pd.DataFrame:
    lookback = _positive_int(params, "lookback")
    selection_count = _positive_int(params, "selection_count")
    momentum = _group_indicator(
        frame, lambda group: group["close"].pct_change(lookback)
    )
    long_mask = pd.Series(False, index=frame.index)
    short_mask = pd.Series(False, index=frame.index)
    ranked = frame.assign(momentum=momentum).dropna(subset=["momentum", "atr"])
    for _, group in ranked.groupby("timestamp", sort=True):
        if len(group) < 2 * selection_count:
            continue
        ordered = group.sort_values(
            ["momentum", "symbol"],
            ascending=[False, True],
            kind="stable",
        )
        long_mask.loc[ordered.index[:selection_count]] = True
        short_mask.loc[ordered.index[-selection_count:]] = True
    return _directional_signals(frame, long_mask, short_mask, params)


def _regime_gated(
    frame: pd.DataFrame, params: ParameterSet
) -> tuple[pd.Series, pd.Series]:
    fast_window = _positive_int(params, "fast_window")
    slow_window = _positive_int(params, "slow_window")
    breakout_lookback = _positive_int(params, "breakout_lookback")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be below slow_window")

    fast = _group_indicator(
        frame,
        lambda group: group["close"].ewm(
            span=fast_window,
            adjust=False,
            min_periods=fast_window,
        ).mean(),
    )
    slow = _group_indicator(
        frame,
        lambda group: group["close"].ewm(
            span=slow_window,
            adjust=False,
            min_periods=slow_window,
        ).mean(),
    )
    prior_high = _group_indicator(
        frame,
        lambda group: group["high"]
        .rolling(breakout_lookback, min_periods=breakout_lookback)
        .max()
        .shift(1),
    )
    prior_low = _group_indicator(
        frame,
        lambda group: group["low"]
        .rolling(breakout_lookback, min_periods=breakout_lookback)
        .min()
        .shift(1),
    )
    return (
        (fast > slow) & (frame["close"] > prior_high),
        (fast < slow) & (frame["close"] < prior_low),
    )


def _directional_signals(
    frame: pd.DataFrame,
    long_mask: pd.Series,
    short_mask: pd.Series,
    params: ParameterSet,
) -> pd.DataFrame:
    stop_atr = _positive_number(params, "stop_atr")
    take_profit_atr = _positive_number(params, "take_profit_atr")
    records: list[dict[str, object]] = []
    for side, mask in (("long", long_mask), ("short", short_mask)):
        for row in frame.loc[mask & frame["atr"].notna()].itertuples(index=False):
            if side == "long":
                stop = row.close - stop_atr * row.atr
                take_profit = row.close + take_profit_atr * row.atr
            else:
                stop = row.close + stop_atr * row.atr
                take_profit = row.close - take_profit_atr * row.atr
            if stop <= 0 or take_profit <= 0:
                continue
            records.append(
                {
                    "symbol": row.symbol,
                    "timestamp": row.timestamp,
                    "side": side,
                    "stop": float(stop),
                    "take_profit": float(take_profit),
                }
            )
    if not records:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    return (
        pd.DataFrame.from_records(records, columns=SIGNAL_COLUMNS)
        .sort_values(["timestamp", "symbol", "side"], kind="stable")
        .reset_index(drop=True)
    )
