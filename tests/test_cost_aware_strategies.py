import itertools
import math

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
    "time_stop_bars",
    "cooldown_bars",
    "priority",
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
            "time_stop_bars": 24,
            "cooldown_bars": 6,
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
            "time_stop_bars": 24,
            "cooldown_bars": 6,
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
            "time_stop_bars": 24,
            "cooldown_bars": 6,
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
            "time_stop_bars": 24,
            "cooldown_bars": 6,
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
            "time_stop_bars": 24,
            "cooldown_bars": 6,
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
            "time_stop_bars": 24,
            "cooldown_bars": 6,
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
            "time_stop_bars": 24,
            "cooldown_bars": 6,
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
        assert all(params["time_stop_bars"] == 24 for params in first)
        assert all(params["cooldown_bars"] == 6 for params in first)


def test_donchian_grid_contains_supplied_calibration_without_grid_explosion():
    calibration = {
        "lookback": 20,
        "atr_window": 14,
        "stop_atr": 3.5,
        "take_profit_atr": 4.0,
        "time_stop_bars": 24,
        "cooldown_bars": 6,
    }

    assert calibration in parameter_grid(STRATEGY_SPECS["donchian"])
    assert len(parameter_grid(STRATEGY_SPECS["donchian"])) == 8


def test_cross_sectional_grid_rotates_one_winner_and_one_loser():
    grid = parameter_grid(STRATEGY_SPECS["cross_sectional_momentum"])

    assert {params["selection_count"] for params in grid} == {1}
    assert len(grid) == 4


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
    assert (signals["priority"] >= 0).all()
    assert set(signals["time_stop_bars"]) == {24}
    assert set(signals["cooldown_bars"]) == {6}


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


def long_tail_bars(periods: int = 240) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    rows = []
    formulas = {
        "ALPHA": lambda index: 100 + 0.08 * index + 5 * math.sin(index / 4),
        "BETA": lambda index: 140 - 0.05 * index + 6 * math.cos(index / 5),
        "GAMMA": lambda index: 90 + 4 * math.sin(index / 3),
        "DELTA": lambda index: 110 + 0.03 * index - 5 * math.cos(index / 6),
    }
    for symbol, formula in formulas.items():
        for index, timestamp in enumerate(timestamps):
            close = formula(index)
            shock = index > 0 and index % 41 == 0
            if shock:
                close *= 1.08 if symbol in {"ALPHA", "DELTA"} else 0.92
            spread = max(0.5, close * 0.002)
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": timestamp,
                    "open": close,
                    "high": close + spread,
                    "low": close - spread,
                    "close": close,
                    "vol": 100.0 if shock else 10.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("strategy_name", CASES)
def test_actual_grid_params_are_prefix_invariant_at_multiple_long_tail_cutoffs(
    strategy_name,
):
    frame = long_tail_bars()
    spec = STRATEGY_SPECS[strategy_name]
    params = parameter_grid(spec)[0]
    full = generate_signals(frame, spec, params)

    assert not full.empty
    timestamps = sorted(frame["open_time"].unique())
    for cutoff_index in (130, 180, 220):
        cutoff = timestamps[cutoff_index]
        prefix = frame.loc[frame["open_time"] <= cutoff]
        prefix_signals = generate_signals(prefix, spec, params)
        expected = full.loc[full["timestamp"] <= cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            prefix_signals.reset_index(drop=True),
            expected,
        )

        mutated = frame.copy()
        future = mutated["open_time"] > cutoff
        mutated.loc[future, ["open", "high", "low", "close"]] *= 1.7
        mutated.loc[future, "vol"] = 1_000_000.0
        mutated_prior = generate_signals(mutated, spec, params)
        mutated_prior = mutated_prior.loc[
            mutated_prior["timestamp"] <= cutoff
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(mutated_prior, expected)


def test_extreme_return_after_missing_bar_is_not_treated_as_one_bar_return():
    frame, params, _ = CASES["extreme_reversal"]
    frame = frame.copy()
    frame.loc[frame.index[-1], "open_time"] += pd.Timedelta(hours=1)
    params = {**params, "direction": "both"}

    signals = generate_signals(frame, STRATEGY_SPECS["extreme_reversal"], params)

    assert signals.empty


def test_bar_interval_inference_does_not_use_future_timestamp_cadence():
    frame = bars(
        {"BTC": [100.0] * 11 + [110.0]},
        volumes_by_symbol={"BTC": [10.0] * 11 + [30.0]},
    )
    frame["open_time"] = pd.date_range(
        "2026-01-01", periods=len(frame), freq="2h", tz="UTC"
    )
    future = []
    final = frame.iloc[-1].copy()
    for offset in range(1, 25):
        row = final.copy()
        row["open_time"] = final["open_time"] + pd.Timedelta(hours=offset)
        row[["open", "high", "low", "close"]] = [110.0, 110.5, 109.5, 110.0]
        row["vol"] = 10.0
        future.append(row)
    full_frame = pd.concat([frame, pd.DataFrame(future)], ignore_index=True)
    params = {
        **CASES["extreme_reversal"][1],
        "volume_window": 5,
        "direction": "both",
    }

    prefix_signals = generate_signals(
        frame, STRATEGY_SPECS["extreme_reversal"], params
    )
    full_signals = generate_signals(
        full_frame, STRATEGY_SPECS["extreme_reversal"], params
    )
    full_prior = full_signals.loc[
        full_signals["timestamp"] <= frame["open_time"].max()
    ].reset_index(drop=True)

    assert not prefix_signals.empty
    pd.testing.assert_frame_equal(
        prefix_signals.reset_index(drop=True),
        full_prior,
    )


def test_cross_sectional_momentum_excludes_symbol_whose_lookback_crosses_gap():
    frame = bars(
        {
            "WIN": [100.0] * 11 + [120.0],
            "RUNNER": [100.0] * 11 + [110.0],
            "MID": [100.0] * 12,
            "LOSE": [100.0] * 11 + [90.0],
        }
    )
    missing_timestamp = pd.Timestamp("2026-01-01 10:00:00+00:00")
    frame = frame.loc[
        ~(
            (frame["symbol"] == "WIN")
            & (frame["open_time"] == missing_timestamp)
        )
    ]
    params = {
        **CASES["cross_sectional_momentum"][1],
        "lookback": 1,
    }

    signals = generate_signals(
        frame, STRATEGY_SPECS["cross_sectional_momentum"], params
    )
    final = signals.loc[
        signals["timestamp"] == pd.Timestamp("2026-01-01 11:00:00+00:00")
    ]

    assert set(zip(final["symbol"], final["side"])) == {
        ("RUNNER", "long"),
        ("LOSE", "short"),
    }


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
