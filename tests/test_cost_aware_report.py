import json
import math
from pathlib import Path

import pandas as pd
import pytest

from cost_aware.config import ExecutionConfig, SizingConfig
from cost_aware.engine import BacktestResult
from cost_aware.report import (
    NO_PASS_STATEMENT,
    render_html,
    render_markdown,
    validate_summary,
    write_reports,
)
from cost_aware.run_research import (
    COST_OFF_EXECUTION,
    COST_ON_EXECUTION,
    PRIMARY_SIZING,
    REFERENCE_SIZING,
    ResearchData,
    ResearchRunner,
    _candidate_result,
    _checkpoint_payload,
    _equity_series,
    _fingerprint,
    _fingerprint_payload,
    build_research_folds,
    run_strategy_window,
    search_space_manifest,
    write_json_atomic,
)
from cost_aware.strategies import STRATEGY_SPECS, StrategySpec, parameter_grid
from cost_aware.walkforward import build_folds, compute_metrics


CALIBRATION_PARAMS = {
    "lookback": 20,
    "atr_window": 14,
    "stop_atr": 3.5,
    "take_profit_atr": 4.0,
    "time_stop_bars": 24,
    "cooldown_bars": 6,
}


def _summary(*, passing: bool = True) -> dict[str, object]:
    adaptive_params = [
        {"fold": 0, "params": {**CALIBRATION_PARAMS, "lookback": 20}},
        {"fold": 1, "params": {**CALIBRATION_PARAMS, "lookback": 40}},
        {"fold": 2, "params": {**CALIBRATION_PARAMS, "lookback": 20}},
    ]
    candidate = {
        "rank": 1,
        "pass_rank": 1 if passing else None,
        "strategy": "donchian<script>",
        "family": "trend_breakout",
        "timeframe": "15m|unsafe",
        "passed": passing,
        "adaptive_params": adaptive_params,
        "latest_params": adaptive_params[-1]["params"],
        "oos": {
            "return": 0.1234,
            "sharpe": 1.234,
            "max_drawdown": -0.1567,
            "trades": 42,
            "win_rate": 0.5432,
            "fold_returns": [0.01, -0.02, 0.14],
        },
        "neighbor": {"positive": 7, "total": 10, "robustness": 0.70},
        "cost_off": {
            "return": 0.2345,
            "sharpe": 1.8,
            "max_drawdown": -0.12,
            "trades": 42,
            "win_rate": 0.57,
        },
        "reference": {
            "oos": {
                "return": 0.052,
                "sharpe": 0.98,
                "max_drawdown": -0.09,
                "trades": 42,
                "win_rate": 0.52,
            },
            "calendar_returns": {"7d": 0.01, "30d": 0.02, "90d": 0.03},
        },
        "calendar_returns": {"7d": 0.04, "30d": 0.05, "90d": 0.06},
        "start_sensitivity": (
            {
                "nominal_window": "30d",
                "start_shift_days": 7,
                "actual_window": "23d",
                "days": 23,
                "start": "2026-10-25T00:00:00+00:00",
                "end": "2026-11-17T00:00:00+00:00",
                "primary_return": 0.031,
                "reference_return": 0.012,
            }
            if passing
            else None
        ),
    }
    if not passing:
        candidate["oos"] = {
            **candidate["oos"],
            "return": -0.01,
            "fold_returns": [0.01, -0.02, -0.01],
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "generated_at": "2026-11-18T14:00:00+00:00",
        "data": {
            "metadata_generated_at": "2026-11-18T13:19:56+00:00",
            "source": "OKX public data",
            "requested_start": "2026-08-18",
            "requested_end": "2026-11-16",
            "universe": ["BTC-USDT-SWAP", "<ETH&>"],
            "timeframes": ["3m", "15m", "1h", "4h"],
            "coverage": "20 instruments; 90-91 complete UTC days",
            "source_gaps": [
                {
                    "instrument": "SOL-USDT-SWAP",
                    "date": "2026-09-01",
                    "timestamp": "2026-09-01T10:33:00+00:00",
                    "missing_minutes": 1,
                }
            ],
        },
        "research": {
            "cost_model": {
                "fee_per_side": 0.0005,
                "slippage_per_side": 0.0002,
                "funding": "realized OKX funding at published timestamp",
                "signal_lag_bars": 1,
            },
            "primary_sizing": {
                "margin_fraction": 0.10,
                "leverage": 8,
                "max_positions": 3,
            },
            "reference_sizing": {
                "margin_fraction": 0.05,
                "leverage": 5,
                "max_positions": 3,
            },
            "folds": [
                {
                    "fold": index,
                    "is_start": f"2026-0{8 + index}-18",
                    "is_end": f"2026-1{index}-02",
                    "oos_start": f"2026-1{index}-02",
                    "oos_end": f"2026-1{index}-17",
                }
                for index in range(3)
            ],
            "calendar_window_end": "2026-11-17T00:00:00+00:00",
            "calendar_window_definition": (
                "last complete UTC calendar days using latest exact params"
            ),
            "calendar_overlap_disclosure": {
                "parameter_source": "latest fold exact params",
                "last_30d_latest_is_overlap_days": 14,
                "last_90d_latest_is_overlap_days": 45,
                "last_90d_includes_pre_latest_is_history": True,
                "independent_oos": False,
                "start_sensitive": True,
                "sensitivity_window": "23d (30d start shifted +7d)",
            },
            "neighbor_rule": (
                "unique legal one-step grid neighbors; positive compounded "
                "OOS return / all unique neighbors"
            ),
            "passing_gates": {
                "positive_oos_folds": "at least 2 of 3",
                "minimum_total_oos_return_exclusive": 0.0,
                "minimum_oos_trades": 30,
                "maximum_drawdown_floor": -0.30,
                "minimum_neighbor_robustness": 0.70,
            },
            "search_counts": {
                "strategies": 7,
                "timeframes": 4,
                "family_timeframe_candidates": 1,
                "grid_configurations": 240,
                "is_fold_evaluations": 720,
                "fresh_backtests": 99,
                "fresh_candidates": 1,
                "checkpoint_reused_candidates": 0,
            },
        },
        "candidates": [candidate],
        "passing_count": int(passing),
        "calibration": {
            "strategy": "donchian",
            "timeframe": "15m",
            "direction": "both",
            "params": CALIBRATION_PARAMS,
            "primary_returns": {"7d": -0.01, "30d": -0.02, "56d": -0.03},
            "legacy_equal_weight_returns": {
                "7d": -0.066,
                "30d": -0.127,
                "56d": -0.181,
            },
            "legacy_method": "1/N x 1x approximation",
            "user_reference_returns": {
                "7d": -0.066,
                "30d": -0.127,
                "56d": -0.181,
            },
        },
        "warnings": [
            "Multiple testing can overstate apparent robustness.",
            (
                "Last 30d/90d use latest fold params, overlap its IS by "
                "14/45 days, are not independent OOS, and are start sensitive."
            ),
            "Research results are not evidence that a strategy is ready for live trading.",
        ],
    }


def _window_bars() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01 00:00:00+00:00",
        periods=23,
        freq="1h",
    )
    rows = []
    for index, timestamp in enumerate(timestamps):
        close = 100.0 if index < 20 else 110.0
        if index == 21:
            close = 120.0
        rows.append(
            {
                "symbol": "BTC-USDT-SWAP",
                "open_time": timestamp,
                "open": close,
                "high": close + 0.25,
                "low": close - 0.25,
                "close": close,
                "vol": 10.0,
            }
        )
    return pd.DataFrame(rows)


def _integration_data() -> ResearchData:
    timestamps = pd.date_range(
        "2025-12-20 00:00:00+00:00",
        "2026-04-01 20:00:00+00:00",
        freq="4h",
    )
    rows: list[dict[str, object]] = []
    symbols = ("ALPHA-USDT-SWAP", "BETA-USDT-SWAP", "GAMMA-USDT-SWAP")
    for symbol_index, symbol in enumerate(symbols):
        previous = 100.0 + symbol_index * 10.0
        for index, timestamp in enumerate(timestamps):
            close = (
                100.0
                + symbol_index * 10.0
                + index * (0.018 + symbol_index * 0.002)
                + 3.5 * math.sin(index / (3.0 + symbol_index) + symbol_index)
            )
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": timestamp,
                    "open": previous,
                    "high": max(previous, close) + 0.4,
                    "low": min(previous, close) - 0.4,
                    "close": close,
                    "vol": 100.0,
                }
            )
            previous = close
    funding = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "timestamp": timestamp,
                "rate": 0.0003,
            }
            for symbol in symbols
            for timestamp in pd.date_range(
                "2026-01-01 08:00:00+00:00",
                "2026-04-01 16:00:00+00:00",
                freq="8h",
            )
        ]
    )
    return ResearchData(
        metadata={
            "requested_start": "2026-01-01",
            "requested_end": "2026-04-01",
            "universe": list(symbols),
            "generated_at": "2026-04-02T00:00:00+00:00",
        },
        bars={"4h": pd.DataFrame(rows)},
        funding=funding,
    )


SMALL_DONCHIAN = StrategySpec(
    name="donchian",
    family="synthetic_integration",
    parameter_space=(
        ("lookback", (3, 5)),
        ("atr_window", (3,)),
        ("stop_atr", (1.5,)),
        ("take_profit_atr", (2.0,)),
        ("time_stop_bars", (4,)),
        ("cooldown_bars", (0,)),
    ),
)


class _RecordingRunner(ResearchRunner):
    def __init__(self, data: ResearchData):
        super().__init__(data)
        self.calls: list[dict[str, object]] = []

    def run(self, spec, timeframe, params, **kwargs):
        result = super().run(spec, timeframe, params, **kwargs)
        self.calls.append(
            {
                "params": dict(params),
                "timeframe": timeframe,
                "execution": kwargs.get("execution", COST_ON_EXECUTION),
                "include_funding": kwargs.get("include_funding", True),
                **kwargs,
                "result": result,
            }
        )
        return result


def test_search_manifest_runs_every_family_timeframe_grid():
    manifest = search_space_manifest()

    assert manifest["strategies"] == list(STRATEGY_SPECS)
    assert manifest["timeframes"] == ["3m", "15m", "1h", "4h"]
    assert manifest["family_timeframe_candidates"] == 28
    assert manifest["grid_configurations"] == 240
    assert manifest["is_fold_evaluations"] == 720
    assert manifest["grid_sizes"] == {
        name: len(list(manifest["grids"][name]))
        for name in STRATEGY_SPECS
    }


def test_research_folds_come_from_metadata_start_and_fit_requested_coverage():
    metadata = {
        "requested_start": "2026-08-18",
        "requested_end": "2026-11-16",
    }

    folds = build_research_folds(metadata)

    assert len(folds) == 3
    assert folds[0].is_start == pd.Timestamp("2026-08-18", tz="UTC")
    assert folds[0].is_end == pd.Timestamp("2026-10-02", tz="UTC")
    assert folds[-1].oos_end == pd.Timestamp("2026-11-16", tz="UTC")


def test_window_uses_prefold_signal_for_first_open_but_starts_flat():
    bars = _window_bars()
    start = pd.Timestamp("2026-01-01 21:00:00+00:00")
    end = pd.Timestamp("2026-01-01 23:00:00+00:00")

    result = run_strategy_window(
        bars,
        pd.DataFrame(columns=["symbol", "timestamp", "rate"]),
        spec=STRATEGY_SPECS["donchian"],
        params=CALIBRATION_PARAMS,
        start=start,
        end=end,
        expected_cadence="1h",
        execution=ExecutionConfig(1, 0.0, 0.0),
        sizing=SizingConfig(1_000, 0.1, 1, 3),
    )

    assert result.equity_events[0].timestamp == start
    assert result.equity_events[0].kind == "window_start"
    assert result.trades
    assert result.trades[0].entry_time == start
    assert all(trade.entry_time >= start for trade in result.trades)


def test_cost_off_is_a_real_zero_cost_zero_funding_configuration():
    assert COST_ON_EXECUTION == ExecutionConfig(
        signal_lag_bars=1,
        fee_per_side=0.0005,
        slippage_per_side=0.0002,
    )
    assert COST_OFF_EXECUTION == ExecutionConfig(
        signal_lag_bars=1,
        fee_per_side=0.0,
        slippage_per_side=0.0,
    )
    assert PRIMARY_SIZING == SizingConfig(100_000.0, 0.10, 8.0, 3)
    assert REFERENCE_SIZING == SizingConfig(100_000.0, 0.05, 5.0, 3)

    bars = _window_bars()
    start = pd.Timestamp("2026-01-01 21:00:00+00:00")
    end = pd.Timestamp("2026-01-01 23:00:00+00:00")
    funding = pd.DataFrame(
        [
            {
                "symbol": "BTC-USDT-SWAP",
                "timestamp": pd.Timestamp("2026-01-01 22:00:00+00:00"),
                "rate": 0.01,
            }
        ]
    )

    result = run_strategy_window(
        bars,
        funding,
        spec=STRATEGY_SPECS["donchian"],
        params=CALIBRATION_PARAMS,
        start=start,
        end=end,
        expected_cadence="1h",
        execution=COST_OFF_EXECUTION,
        sizing=SizingConfig(1_000, 0.1, 1, 3),
        include_funding=False,
    )

    assert result.trades
    assert all(trade.fees == 0.0 for trade in result.trades)
    assert all(trade.funding_pnl == 0.0 for trade in result.trades)


def test_runner_materializes_shared_window_once_for_distinct_grid_params():
    bars = _window_bars()
    data = ResearchData(
        metadata={},
        bars={"1h": bars},
        funding=pd.DataFrame(columns=["symbol", "timestamp", "rate"]),
    )
    runner = ResearchRunner(data)
    start = pd.Timestamp("2026-01-01 21:00:00+00:00")
    end = pd.Timestamp("2026-01-01 23:00:00+00:00")

    for lookback in (20, 40):
        runner.run(
            STRATEGY_SPECS["donchian"],
            "1h",
            {**CALIBRATION_PARAMS, "lookback": lookback},
            start=start,
            end=end,
            sizing=SizingConfig(1_000, 0.1, 1, 3),
        )

    assert runner.backtests == 2
    assert runner.window_materializations == 1


def test_runner_cache_key_distinguishes_execution_sizing_and_funding():
    data = _integration_data()
    runner = ResearchRunner(data)
    params = next(iter(parameter_grid(SMALL_DONCHIAN)))
    start = pd.Timestamp("2026-03-01", tz="UTC")
    end = pd.Timestamp("2026-03-03", tz="UTC")
    variants = (
        (PRIMARY_SIZING, COST_ON_EXECUTION, True),
        (PRIMARY_SIZING, COST_ON_EXECUTION, False),
        (PRIMARY_SIZING, COST_OFF_EXECUTION, False),
        (REFERENCE_SIZING, COST_ON_EXECUTION, True),
    )

    results = [
        runner.run(
            SMALL_DONCHIAN,
            "4h",
            params,
            start=start,
            end=end,
            sizing=sizing,
            execution=execution,
            include_funding=include_funding,
        )
        for sizing, execution, include_funding in variants
    ]

    assert runner.backtests == len(variants)
    assert len({id(result) for result in results}) == len(variants)
    repeated = runner.run(
        SMALL_DONCHIAN,
        "4h",
        params,
        start=start,
        end=end,
        sizing=PRIMARY_SIZING,
        execution=COST_ON_EXECUTION,
        include_funding=True,
    )
    assert repeated is results[0]
    assert runner.cache_hits == 1


def test_checkpoint_fingerprint_covers_sources_metadata_and_research_contract():
    data = _integration_data()
    sources = {
        "engine.py": b"engine-v1",
        "strategies.py": b"strategies-v1",
        "walkforward.py": b"walkforward-v1",
        "run_research.py": b"runner-v1",
        "config.py": b"config-v1",
        "data.py": b"data-v1",
        "report.py": b"report-v1",
    }

    payload = _fingerprint_payload(data, source_contents=sources)
    first = _fingerprint(data, source_contents=sources)
    changed_source = _fingerprint(
        data,
        source_contents={**sources, "engine.py": b"engine-v2"},
    )
    changed_config = _fingerprint(
        data,
        source_contents={**sources, "config.py": b"config-v2"},
    )
    changed_metadata = _fingerprint(
        ResearchData(
            metadata={**data.metadata, "generated_at": "2026-04-03T00:00:00Z"},
            bars=data.bars,
            funding=data.funding,
        ),
        source_contents=sources,
    )

    assert set(payload["production_source_sha256"]) == set(sources)
    assert len(payload["metadata_sha256"]) == 64
    assert payload["fold_config"] == {
        "count": 3,
        "is_days": 45,
        "oos_days": 15,
    }
    assert payload["gate_thresholds"]["minimum_total_oos_return_exclusive"] == 0.0
    assert payload["calendar"]["latest_param_windows_days"] == [7, 30, 90]
    assert payload["calendar"]["start_sensitivity_days"] == 23
    assert payload["execution"]["cost_on"] == {
        "signal_lag_bars": 1,
        "fee_per_side": 0.0005,
        "slippage_per_side": 0.0002,
    }
    assert payload["sizing"]["primary"]["margin_fraction"] == 0.10
    assert first != changed_source
    assert first != changed_config
    assert first != changed_metadata


def test_checkpoint_payload_distinguishes_fresh_and_reused_work():
    payload = _checkpoint_payload(
        fingerprint="fingerprint",
        started_at="2026-01-01T00:00:00Z",
        results={},
        elapsed_seconds=1.25,
        run_stats={
            "fresh_backtests": 12,
            "fresh_candidates": 2,
            "checkpoint_reused_candidates": 5,
        },
    )

    assert payload["run_stats"] == {
        "fresh_backtests": 12,
        "fresh_candidates": 2,
        "checkpoint_reused_candidates": 5,
    }


def test_candidate_integration_runs_full_small_grid_and_independent_variants(
    monkeypatch,
):
    import cost_aware.run_research as research_module

    data = _integration_data()
    runner = _RecordingRunner(data)
    folds = build_folds(pd.Timestamp("2026-01-01", tz="UTC"), n_folds=3)
    calendar_end = pd.Timestamp("2026-04-01", tz="UTC")
    engine_calls: list[dict[str, object]] = []
    original = research_module.run_backtest_result

    def observing_run(candles, signals, funding, execution, sizing, **kwargs):
        funding_rows = tuple(funding)
        engine_calls.append(
            {
                "funding_rows": len(funding_rows),
                "execution": execution,
                "sizing": sizing,
                **kwargs,
            }
        )
        return original(
            candles,
            signals,
            funding_rows,
            execution,
            sizing,
            **kwargs,
        )

    monkeypatch.setattr(research_module, "run_backtest_result", observing_run)
    result = _candidate_result(
        runner,
        SMALL_DONCHIAN,
        "4h",
        folds,
        calendar_end=calendar_end,
        gate=lambda _evaluation: True,
    )

    grid = list(parameter_grid(SMALL_DONCHIAN))
    assert len(grid) == 2
    assert result["passed"] is True
    assert len(result["adaptive_params"]) == 3

    for fold, adaptive in zip(folds, result["adaptive_params"], strict=True):
        is_calls = [
            call
            for call in runner.calls
            if call["start"] == fold.is_start
            and call["end"] == fold.is_end
            and call["sizing"] == PRIMARY_SIZING
            and call["execution"] == COST_ON_EXECUTION
            and call["include_funding"] is True
        ]
        assert len(is_calls) == len(grid)
        scored = [
            (
                compute_metrics(
                    _equity_series(call["result"]),
                    call["result"].trades,
                ),
                call["params"],
            )
            for call in is_calls
        ]
        expected = max(
            scored,
            key=lambda item: (item[0].sharpe, item[0].total_return),
        )[1]
        assert adaptive["params"] == expected
        assert any(
            call["params"] == adaptive["params"]
            and call["start"] == fold.oos_start
            and call["end"] == fold.oos_end
            and call["sizing"] == PRIMARY_SIZING
            and call["execution"] == COST_ON_EXECUTION
            and call["include_funding"] is True
            for call in runner.calls
        )

    assert sum(
        call["start"] == fold.is_start and call["end"] == fold.is_end
        for fold in folds
        for call in runner.calls
        if call["sizing"] == PRIMARY_SIZING
        and call["execution"] == COST_ON_EXECUTION
        and call["include_funding"] is True
    ) == len(grid) * len(folds)

    cost_off_calls = [
        call
        for call in runner.calls
        if call["execution"] == COST_OFF_EXECUTION
        and call["sizing"] == PRIMARY_SIZING
    ]
    reference_calls = [
        call
        for call in runner.calls
        if call["execution"] == COST_ON_EXECUTION
        and call["sizing"] == REFERENCE_SIZING
        and any(
            call["start"] == fold.oos_start and call["end"] == fold.oos_end
            for fold in folds
        )
    ]
    assert len(cost_off_calls) == 3
    assert len(reference_calls) == 3
    assert all(call["include_funding"] is False for call in cost_off_calls)
    assert all(
        call["funding_rows"] == 0
        for call in engine_calls
        if call["execution"] == COST_OFF_EXECUTION
    )
    assert all(
        cost_off["result"] is not reference["result"]
        for cost_off, reference in zip(
            cost_off_calls,
            reference_calls,
            strict=True,
        )
    )

    sensitivity = result["start_sensitivity"]
    assert sensitivity["nominal_window"] == "30d"
    assert sensitivity["start_shift_days"] == 7
    assert sensitivity["actual_window"] == "23d"
    assert sensitivity["days"] == 23
    assert sensitivity["end"] == calendar_end.isoformat()
    expected_start = (calendar_end - pd.Timedelta(days=23)).isoformat()
    assert sensitivity["start"] == expected_start
    assert sensitivity["primary_return"] == pytest.approx(
        sensitivity["primary_return"]
    )
    assert sensitivity["reference_return"] == pytest.approx(
        sensitivity["reference_return"]
    )
    calendar_calls = [
        call
        for call in runner.calls
        if call["end"] == calendar_end
        and (calendar_end - call["start"]).days in {7, 23, 30, 90}
        and call["execution"] == COST_ON_EXECUTION
        and call["include_funding"] is True
    ]
    assert {
        (calendar_end - call["start"]).days
        for call in calendar_calls
        if call["sizing"] == PRIMARY_SIZING
    } == {7, 23, 30, 90}
    assert {
        (calendar_end - call["start"]).days
        for call in calendar_calls
        if call["sizing"] == REFERENCE_SIZING
    } == {7, 23, 30, 90}
    for call in calendar_calls:
        events = call["result"].equity_events
        assert events[0].timestamp == call["start"]
        assert events[0].kind == "window_start"
        assert events[-1].timestamp == call["end"]
        assert events[-1].kind == "window_end"


def test_passing_report_has_exact_params_all_metrics_and_consistent_numbers():
    summary = _summary()

    validate_summary(summary)
    markdown = render_markdown(summary)
    html = render_html(summary)

    for label in (
        "Adaptive OOS exact params",
        "Latest exact params",
        "OOS return",
        "Last 7d",
        "Last 30d",
        "Last 90d",
        "Sharpe",
        "Max drawdown",
        "Trades",
        "Win rate",
        "Cost-off return",
        "5% x 5x OOS return",
        "5% x 5x Last 7d",
        "5% x 5x Last 30d",
        "5% x 5x Last 90d",
        "5% x 5x Sharpe",
        "5% x 5x Max drawdown",
        "5% x 5x Trades",
        "5% x 5x Win rate",
    ):
        assert label in markdown
        assert label in html
    assert '"lookback": 40' in markdown
    assert "12.34%" in markdown
    assert "12.34%" in html
    assert "fee_per_side" in markdown
    assert "signal_lag_bars" in markdown
    assert "unique legal one-step" in markdown
    assert "Multiple testing" in markdown
    assert "not evidence" in markdown
    assert "not tuned" in markdown
    assert "not tuned" in html
    assert "Data snapshot generated" in markdown
    assert "Calendar return windows" in markdown
    assert "minimum_oos_trades" in markdown
    assert "| OOS Sharpe | OOS Max drawdown | OOS Trades | OOS Win rate |" in markdown
    assert "23d (30d start +7d) return" in markdown
    assert "2026-10-25T00:00:00+00:00" in markdown
    assert "2026-11-17T00:00:00+00:00" in markdown
    assert "3.10%" in markdown
    assert "latest fold exact params" in markdown
    assert "14 days" in markdown
    assert "45 days" in markdown
    assert "not independent OOS" in markdown
    assert "start-date sensitive" in markdown
    assert "SOL-USDT-SWAP" in markdown
    assert "2026-09-01T10:33:00+00:00" in markdown

    outcome = markdown.split("## Outcome", 1)[1].split("## Data", 1)[0]
    assert "Pass rank 1" in outcome
    assert "donchian<script>" in outcome
    assert "15m\\|unsafe" in outcome


def test_no_pass_report_is_explicit_and_still_lists_all_candidates():
    summary = _summary(passing=False)
    summary["candidates"].append(
        {
            **summary["candidates"][0],
            "rank": 2,
            "strategy": "keltner",
            "passed": False,
        }
    )
    summary["research"]["search_counts"]["family_timeframe_candidates"] = 2
    summary["research"]["search_counts"]["fresh_candidates"] = 2

    validate_summary(summary)
    markdown = render_markdown(summary)
    html = render_html(summary)

    assert NO_PASS_STATEMENT in markdown
    assert NO_PASS_STATEMENT in html
    assert "donchian" in markdown
    assert "keltner" in markdown


def test_outcome_lists_only_passers_in_pass_rank_order_not_overall_rank():
    summary = _summary()
    passing = summary["candidates"][0]
    passing["rank"] = 2
    failed = {
        **passing,
        "rank": 1,
        "pass_rank": None,
        "strategy": "failed_high_sharpe",
        "passed": False,
        "oos": {
            **passing["oos"],
            "return": -0.01,
            "sharpe": 9.0,
            "fold_returns": [0.01, -0.01, -0.01],
        },
        "start_sensitivity": None,
    }
    summary["candidates"] = [failed, passing]
    summary["research"]["search_counts"]["family_timeframe_candidates"] = 2
    summary["research"]["search_counts"]["fresh_candidates"] = 2

    validate_summary(summary)
    markdown = render_markdown(summary)
    outcome = markdown.split("## Outcome", 1)[1].split("## Data", 1)[0]

    assert "Pass rank 1" in outcome
    assert "donchian<script>" in outcome
    assert "failed_high_sharpe" not in outcome
    assert "overall rank does not imply passing status" in markdown


def test_summary_validation_rejects_tampered_overall_rank_order():
    summary = _summary()
    passing = summary["candidates"][0]
    weaker = {
        **passing,
        "rank": 1,
        "pass_rank": None,
        "strategy": "weaker_first",
        "passed": False,
        "oos": {
            **passing["oos"],
            "return": -0.5,
            "sharpe": 0.25,
            "fold_returns": [0.01, -0.02, -0.01],
        },
        "start_sensitivity": None,
    }
    stronger = {
        **passing,
        "rank": 2,
        "pass_rank": 1,
        "strategy": "stronger_second",
        "oos": {
            **passing["oos"],
            "return": 0.5,
            "sharpe": 2.5,
            "fold_returns": [0.01, 0.02, 0.14],
        },
    }
    summary["candidates"] = [weaker, stronger]
    summary["research"]["search_counts"]["family_timeframe_candidates"] = 2
    summary["research"]["search_counts"]["fresh_candidates"] = 2

    with pytest.raises(ValueError, match="overall order"):
        validate_summary(summary)


def test_summary_validation_recomputes_gates_pass_order_and_calibration_contract():
    mismatched_gate = _summary()
    mismatched_gate["candidates"][0]["passed"] = False
    mismatched_gate["candidates"][0]["pass_rank"] = None
    mismatched_gate["passing_count"] = 0
    with pytest.raises(ValueError, match="does not match recomputed gates"):
        validate_summary(mismatched_gate)

    bad_pass_rank = _summary()
    bad_pass_rank["candidates"][0]["pass_rank"] = 2
    with pytest.raises(ValueError, match="pass_rank"):
        validate_summary(bad_pass_rank)

    missing_calibration = _summary()
    del missing_calibration["calibration"]["direction"]
    with pytest.raises(ValueError, match="calibration.direction"):
        validate_summary(missing_calibration)

    missing_legacy_method = _summary()
    del missing_legacy_method["calibration"]["legacy_method"]
    with pytest.raises(ValueError, match="calibration.legacy_method"):
        validate_summary(missing_legacy_method)


def test_summary_validation_enforces_passer_sensitivity_and_overlap_disclosure():
    missing_sensitivity = _summary()
    missing_sensitivity["candidates"][0]["start_sensitivity"] = None
    with pytest.raises(ValueError, match="start_sensitivity"):
        validate_summary(missing_sensitivity)

    mislabeled_sensitivity = _summary()
    mislabeled_sensitivity["candidates"][0]["start_sensitivity"][
        "actual_window"
    ] = "30d"
    with pytest.raises(ValueError, match="actual_window"):
        validate_summary(mislabeled_sensitivity)

    missing_overlap = _summary()
    del missing_overlap["research"]["calendar_overlap_disclosure"]
    with pytest.raises(ValueError, match="calendar_overlap_disclosure"):
        validate_summary(missing_overlap)


def test_html_report_remains_explicit_and_escaped_for_review_fields():
    html = render_html(_summary())

    assert "OOS Sharpe" in html
    assert "OOS Max drawdown" in html
    assert "OOS Trades" in html
    assert "OOS Win rate" in html
    assert "23d (30d start +7d) return" in html
    assert "2026-10-25T00:00:00+00:00" in html
    assert "14 days" in html
    assert "45 days" in html
    assert "not independent OOS" in html
    assert "start-date sensitive" in html
    assert "donchian&lt;script&gt;" in html


def test_committed_html_report_is_unignored():
    gitignore = (Path(__file__).parents[1] / ".gitignore").read_text()

    assert "reports/*.html" in gitignore
    assert "!reports/cost_aware_results.html" in gitignore


def test_html_and_markdown_escape_untrusted_labels_and_are_standalone():
    summary = _summary()

    markdown = render_markdown(summary)
    html = render_html(summary)

    assert "15m\\|unsafe" in markdown
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;ETH&amp;&gt;" in html
    assert html.lower().startswith("<!doctype html>")
    assert "<style>" in html


def test_summary_validation_rejects_missing_pass_fields_and_numeric_mismatch():
    missing = _summary()
    del missing["candidates"][0]["cost_off"]
    with pytest.raises(ValueError, match="cost_off"):
        validate_summary(missing)

    mismatch = _summary()
    mismatch["passing_count"] = 0
    with pytest.raises(ValueError, match="passing_count"):
        validate_summary(mismatch)


def test_reports_and_machine_json_are_written_deterministically(tmp_path):
    summary = _summary()
    markdown_path = tmp_path / "results.md"
    html_path = tmp_path / "results.html"
    json_path = tmp_path / "summary.json"

    write_reports(summary, markdown_path, html_path)
    write_json_atomic(json_path, summary)
    first = json_path.read_text()
    write_json_atomic(json_path, summary)

    assert markdown_path.read_text() == render_markdown(summary)
    assert html_path.read_text() == render_html(summary)
    assert json.loads(first) == summary
    assert json_path.read_text() == first
    assert not list(Path(tmp_path).glob("*.tmp"))
