import json
from pathlib import Path

import pandas as pd
import pytest

from cost_aware.config import ExecutionConfig, SizingConfig
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
    build_research_folds,
    run_strategy_window,
    search_space_manifest,
    write_json_atomic,
)
from cost_aware.strategies import STRATEGY_SPECS


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
            "neighbor_rule": (
                "unique legal one-step grid neighbors; positive compounded "
                "OOS return / all unique neighbors"
            ),
            "passing_gates": {
                "positive_oos_folds": "at least 2 of 3",
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
            },
        },
        "candidates": [candidate],
        "passing_count": int(passing),
        "calibration": {
            "strategy": "donchian",
            "timeframe": "15m",
            "params": CALIBRATION_PARAMS,
            "primary_returns": {"7d": -0.01, "30d": -0.02, "56d": -0.03},
            "legacy_equal_weight_returns": {
                "7d": -0.066,
                "30d": -0.127,
                "56d": -0.181,
            },
            "user_reference_returns": {
                "7d": -0.066,
                "30d": -0.127,
                "56d": -0.181,
            },
        },
        "warnings": [
            "Multiple testing can overstate apparent robustness.",
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

    validate_summary(summary)
    markdown = render_markdown(summary)
    html = render_html(summary)

    assert NO_PASS_STATEMENT in markdown
    assert NO_PASS_STATEMENT in html
    assert "donchian" in markdown
    assert "keltner" in markdown


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
