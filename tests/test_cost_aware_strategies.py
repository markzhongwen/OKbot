import itertools

import pandas as pd
import pytest

from cost_aware.strategies import (
    STRATEGY_SPECS,
    StrategySpec,
    generate_signals,
    neighbor_params,
    parameter_grid,
)


REQUIRED_SIGNAL_COLUMNS = [
    "symbol",
    "timestamp",
    "side",
    "stop",
    "take_profit",
]


def bars(
    closes_by_symbol: dict[str, list[float]],
    *,
    volumes_by_symbol: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    rows = []
    for symbol, closes in closes_by_symbol.items():
        volumes = (
            volumes_by_symbol[symbol]
            if volumes_by_symbol is not None
            else [10.0] * len(closes)
        )
        for timestamp, close, volume in zip(
            pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC"),
            closes,
            volumes,
            strict=True,
        ):
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": timestamp,
                    "open": close,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "vol": volume,
                }
            )
    return pd.DataFrame(rows)


CASES = {
    "donchian": (
        bars({"BTC": [100.0] * 20 + [110.0]}),
        {
            "lookback": 20,
            "atr_window": 3,
            "stop_atr": 2.0,
            "take_profit_atr": 3.0,
        },
        {("BTC", "long")},
    ),
    "keltner": (
        bars({"BTC": [100.0] * 20 + [110.0]}),
        {
            "ema_window": 5,
            "atr_window": 3,
            "band_atr": 1.0,
            "stop_atr": 2.0,
            "take_profit_atr": 3.0,
        },
        {("BTC", "long")},
    ),
    "rsi_mean_reversion": (
        bars({"BTC": [100.0] * 10 + [99.0, 98.0, 97.0]}),
        {
            "rsi_window": 3,
            "threshold": 30.0,
            "atr_window": 3,
            "stop_atr": 2.0,
            "take_profit_atr": 3.0,
        },
        {("BTC", "long")},
    ),
    "vwap_mean_reversion": (
        bars({"BTC": [100.0] * 10 + [90.0]}),
        {
            "vwap_window": 5,
            "atr_window": 3,
            "deviation_atr": 0.5,
            "stop_atr": 2.0,
            "take_profit_atr": 3.0,
        },
        {("BTC", "long")},
    ),
    "extreme_reversal": (
        bars(
            {"BTC": [100.0] * 20 + [110.0]},
            volumes_by_symbol={"BTC": [10.0] * 20 + [30.0]},
        ),
        {
            "return_lookback": 1,
            "return_threshold": 0.05,
            "volume_window": 10,
            "volume_multiplier": 2.0,
            "direction": "short",
            "atr_window": 3,
            "stop_atr": 2.0,
            "take_profit_atr": 3.0,
        },
        {("BTC", "short")},
    ),
    "cross_sectional_momentum": (
        bars(
            {
                "WIN": [100.0, 100.0, 120.0],
                "MID": [100.0, 100.0, 100.0],
                "LOSE": [100.0, 100.0, 80.0],
            }
        ),
        {
            "lookback": 2,
            "selection_count": 1,
            "atr_window": 2,
            "stop_atr": 2.0,
            "take_profit_atr": 3.0,
        },
        {("WIN", "long"), ("LOSE", "short")},
    ),
    "regime_gated": (
        bars(
            {
                "UP": [100.0] * 10 + [101.0, 102.0, 104.0],
                "DOWN": [100.0] * 10 + [99.0, 98.0, 96.0],
            }
        ),
        {
            "fast_window": 2,
            "slow_window": 4,
            "breakout_lookback": 3,
            "atr_window": 3,
            "stop_atr": 2.0,
            "take_profit_atr": 3.0,
        },
        {("UP", "long"), ("DOWN", "short")},
    ),
}


def test_strategy_specs_have_deterministic_bounded_parameter_grids():
    assert set(STRATEGY_SPECS) == set(CASES)
    assert all(isinstance(spec, StrategySpec) for spec in STRATEGY_SPECS.values())

    for spec in STRATEGY_SPECS.values():
        first = parameter_grid(spec)
        second = parameter_grid(spec)

        assert first == second
        assert 1 <= len(first) <= 16
        assert len({tuple(params.items()) for params in first}) == len(first)
        assert all(
            key not in {"time_stop", "time_stop_bars", "cooldown", "cooldown_bars"}
            for params in first
            for key in params
        )


def test_neighbor_params_changes_exactly_one_parameter_by_one_grid_step():
    for spec in STRATEGY_SPECS.values():
        grid = parameter_grid(spec)
        current = grid[len(grid) // 2]
        neighbors = neighbor_params(spec, current)
        values_by_name = dict(spec.parameter_space)

        assert neighbors
        assert neighbors == neighbor_params(spec, current)
        for neighbor in neighbors:
            changed = [key for key in current if neighbor[key] != current[key]]
            assert len(changed) == 1
            key = changed[0]
            values = values_by_name[key]
            assert abs(values.index(neighbor[key]) - values.index(current[key])) == 1
            assert neighbor in grid


@pytest.mark.parametrize("strategy_name", CASES)
def test_each_strategy_family_emits_expected_closed_bar_signals(strategy_name):
    frame, params, expected = CASES[strategy_name]

    signals = generate_signals(frame, STRATEGY_SPECS[strategy_name], params)

    assert list(signals.columns) == REQUIRED_SIGNAL_COLUMNS
    assert expected.issubset(set(zip(signals["symbol"], signals["side"])))
    assert set(signals["side"]) <= {"long", "short"}
    assert signals[REQUIRED_SIGNAL_COLUMNS].notna().all(axis=None)


def test_extreme_reversal_short_only_never_emits_longs():
    frame = bars(
        {"BTC": [100.0] * 20 + [90.0, 110.0]},
        volumes_by_symbol={"BTC": [10.0] * 20 + [30.0, 30.0]},
    )
    params = CASES["extreme_reversal"][1]

    signals = generate_signals(frame, STRATEGY_SPECS["extreme_reversal"], params)

    assert not signals.empty
    assert set(signals["side"]) == {"short"}


def test_stops_and_targets_use_signal_bar_atr():
    frame, params, _ = CASES["donchian"]

    signal = generate_signals(
        frame, STRATEGY_SPECS["donchian"], params
    ).iloc[-1]
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - frame["close"].shift()).abs(),
            (frame["low"] - frame["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    signal_atr = true_range.rolling(params["atr_window"]).mean().iloc[-1]

    assert signal["timestamp"] == frame["open_time"].iloc[-1]
    assert signal["stop"] == pytest.approx(
        frame["close"].iloc[-1] - params["stop_atr"] * signal_atr
    )
    assert signal["take_profit"] == pytest.approx(
        frame["close"].iloc[-1] + params["take_profit_atr"] * signal_atr
    )


@pytest.mark.parametrize("strategy_name", CASES)
def test_modifying_future_bars_cannot_change_prior_signals(strategy_name):
    frame, params, _ = CASES[strategy_name]
    last_timestamp = frame["open_time"].max()
    extended = []
    mutated = []
    for symbol, symbol_frame in frame.groupby("symbol", sort=False):
        final = symbol_frame.iloc[-1].copy()
        final["open_time"] = final["open_time"] + pd.Timedelta(hours=1)
        extended.append(final)
        changed = final.copy()
        changed[["open", "high", "low", "close"]] = [500.0, 501.0, 499.0, 500.0]
        changed["vol"] = 1_000_000.0
        mutated.append(changed)

    baseline_frame = pd.concat(
        [frame, pd.DataFrame(extended)], ignore_index=True
    )
    mutated_frame = pd.concat(
        [frame, pd.DataFrame(mutated)], ignore_index=True
    )
    baseline = generate_signals(
        baseline_frame, STRATEGY_SPECS[strategy_name], params
    )
    changed = generate_signals(
        mutated_frame, STRATEGY_SPECS[strategy_name], params
    )
    baseline_prior = baseline.loc[baseline["timestamp"] <= last_timestamp]
    changed_prior = changed.loc[changed["timestamp"] <= last_timestamp]

    assert not baseline_prior.empty
    pd.testing.assert_frame_equal(
        baseline_prior.reset_index(drop=True),
        changed_prior.reset_index(drop=True),
    )


def test_parameter_grid_matches_cartesian_product_order():
    spec = STRATEGY_SPECS["donchian"]
    names = [name for name, _ in spec.parameter_space]
    expected = [
        dict(zip(names, values, strict=True))
        for values in itertools.product(
            *(values for _, values in spec.parameter_space)
        )
    ]

    assert parameter_grid(spec) == expected
