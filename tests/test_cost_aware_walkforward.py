from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cost_aware.walkforward import (
    CandidateEvaluation,
    Fold,
    Metrics,
    build_folds,
    compute_metrics,
    evaluate_candidate,
    passes_research_gates,
)


def _equity(
    start: pd.Timestamp,
    end: pd.Timestamp,
    total_return: float,
) -> pd.Series:
    return pd.Series(
        [100.0, 100.0 * (1.0 + total_return)],
        index=pd.DatetimeIndex([start, end]),
        dtype=float,
    )


def _trades(count: int, *, wins: int | None = None):
    win_count = count if wins is None else wins
    return [
        SimpleNamespace(net_pnl=1.0 if index < win_count else -1.0)
        for index in range(count)
    ]


def _window(fold: Fold, phase: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if phase == "is":
        return fold.is_start, fold.is_end
    return fold.oos_start, fold.oos_end


def test_default_folds_cover_90_days_with_rolling_is_and_nonoverlapping_oos():
    start = pd.Timestamp("2026-01-01", tz="UTC")

    folds = build_folds(start)

    assert len(folds) == 3
    assert folds[0] == Fold(
        index=0,
        is_start=start,
        is_end=start + pd.Timedelta(days=45),
        oos_start=start + pd.Timedelta(days=45),
        oos_end=start + pd.Timedelta(days=60),
    )
    assert folds[-1].oos_end == start + pd.Timedelta(days=90)
    assert all(
        fold.oos_start == fold.is_end
        and fold.is_end - fold.is_start == pd.Timedelta(days=45)
        and fold.oos_end - fold.oos_start == pd.Timedelta(days=15)
        for fold in folds
    )
    assert all(
        left.oos_end <= right.oos_start
        for left, right in zip(folds, folds[1:])
    )


def test_build_folds_has_configurable_rolling_and_oos_lengths():
    start = pd.Timestamp("2026-01-01", tz="UTC")

    folds = build_folds(
        start,
        is_days=20,
        oos_days=5,
        n_folds=4,
        step_days=5,
    )

    assert len(folds) == 4
    assert folds[2].is_start == start + pd.Timedelta(days=10)
    assert folds[2].is_end == start + pd.Timedelta(days=30)
    assert folds[2].oos_end == start + pd.Timedelta(days=35)


def test_metrics_use_compounded_event_equity_and_keep_no_trade_calendar_days():
    index = pd.to_datetime(
        [
            "2026-01-01 00:00Z",
            "2026-01-02 00:00Z",
            "2026-01-02 12:00Z",
            "2026-01-02 23:00Z",
            "2026-01-04 00:00Z",
            "2026-01-05 00:00Z",
        ]
    )
    equity = pd.Series([100.0, 110.0, 88.0, 110.0, 110.0, 121.0], index=index)
    trades = [
        SimpleNamespace(net_pnl=10_000.0),
        SimpleNamespace(net_pnl=-1.0),
    ]
    daily_returns = pd.Series([0.10, 0.0, 0.0, 0.10])
    expected_sharpe = (
        daily_returns.mean()
        / daily_returns.std(ddof=1)
        * np.sqrt(365.0)
    )

    metrics = compute_metrics(equity, trades, fold_returns=(0.10, -0.02))

    assert metrics.total_return == pytest.approx(0.21)
    assert metrics.sharpe == pytest.approx(expected_sharpe)
    assert metrics.max_drawdown == pytest.approx(-0.20)
    assert metrics.trades == 2
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.fold_returns == pytest.approx((0.10, -0.02))


def test_daily_sharpe_includes_return_earned_during_first_calendar_day():
    equity = pd.Series(
        [100.0, 110.0, 110.0],
        index=pd.to_datetime(
            [
                "2026-01-01 00:00Z",
                "2026-01-01 12:00Z",
                "2026-01-03 00:00Z",
            ]
        ),
    )
    daily_returns = pd.Series([0.10, 0.0])
    expected_sharpe = (
        daily_returns.mean()
        / daily_returns.std(ddof=1)
        * np.sqrt(365.0)
    )

    metrics = compute_metrics(equity, _trades(1))

    assert metrics.sharpe == pytest.approx(expected_sharpe)


def test_no_trades_force_sharpe_and_win_rate_to_zero():
    equity = pd.Series(
        [100.0, 110.0],
        index=pd.to_datetime(["2026-01-01", "2026-01-03"], utc=True),
    )

    metrics = compute_metrics(equity, [])

    assert metrics.total_return == pytest.approx(0.10)
    assert metrics.sharpe == 0.0
    assert metrics.win_rate == 0.0


def test_fold_equity_must_cover_both_boundaries_so_quiet_tail_is_counted():
    folds = build_folds(
        pd.Timestamp("2026-01-01", tz="UTC"),
        n_folds=1,
    )

    def callback(param, fold, sizing, phase):
        del param, sizing
        start, end = _window(fold, phase)
        return _equity(start, end - pd.Timedelta(days=1), 0.01), _trades(1)

    with pytest.raises(ValueError, match="cover.*window boundaries"):
        evaluate_candidate(
            [{"lookback": 20}],
            folds,
            sizing=object(),
            callback=callback,
            neighbors=lambda selected: (),
        )


def test_each_fold_selects_on_its_own_is_only_and_freezes_exact_params():
    folds = build_folds(pd.Timestamp("2026-01-01", tz="UTC"))
    params = [{"name": "alpha"}, {"name": "beta"}]
    winners = {0: "alpha", 1: "beta", 2: "alpha"}
    calls: list[tuple[int, str, str]] = []

    def callback(param, fold, sizing, phase):
        del sizing
        calls.append((fold.index, phase, param["name"]))
        start, end = _window(fold, phase)
        if phase == "is":
            if param["name"] == winners[fold.index]:
                timestamps = pd.date_range(start, end, freq="1D", tz="UTC")
                values = 100.0 * np.power(1.001, np.arange(len(timestamps)))
                return pd.Series(values, index=timestamps), _trades(1)
            midpoint = start + (end - start) / 2
            return (
                pd.Series(
                    [100.0, 80.0, 101.0],
                    index=pd.DatetimeIndex([start, midpoint, end]),
                ),
                _trades(1),
            )
        future_return = 10.0 if param["name"] == "beta" else 0.01
        return _equity(start, end, future_return), _trades(1)

    evaluation = evaluate_candidate(
        params,
        folds,
        sizing=object(),
        callback=callback,
        neighbors=lambda selected: (),
    )

    assert [result.selected_params for result in evaluation.fold_results] == [
        {"name": "alpha"},
        {"name": "beta"},
        {"name": "alpha"},
    ]
    assert not any(
        phase == "oos" and name != winners[index]
        for index, phase, name in calls
    )


def test_evaluate_candidate_compounds_nonoverlapping_oos_equity_and_fold_returns():
    folds = build_folds(pd.Timestamp("2026-01-01", tz="UTC"))

    def callback(param, fold, sizing, phase):
        del param, sizing
        start, end = _window(fold, phase)
        return _equity(start, end, 0.01 if phase == "is" else 0.10), _trades(2)

    evaluation = evaluate_candidate(
        [{"lookback": 20}],
        folds,
        sizing=object(),
        callback=callback,
        neighbors=lambda selected: (),
    )

    assert evaluation.metrics.total_return == pytest.approx(1.1**3 - 1)
    assert evaluation.metrics.fold_returns == pytest.approx((0.10, 0.10, 0.10))
    assert evaluation.metrics.trades == 6


def test_neighbor_robustness_excludes_invalid_neighbors_but_keeps_losses():
    folds = build_folds(
        pd.Timestamp("2026-01-01", tz="UTC"),
        n_folds=1,
    )
    neighbors = [
        *({"neighbor": f"good-{index}"} for index in range(7)),
        *({"neighbor": f"loss-{index}"} for index in range(3)),
        {"neighbor": "invalid"},
    ]

    def callback(param, fold, sizing, phase):
        del sizing
        start, end = _window(fold, phase)
        name = param.get("neighbor", "selected")
        if phase == "oos" and name == "invalid":
            return None
        result = -0.01 if name.startswith("loss-") else 0.01
        return _equity(start, end, result), _trades(1)

    evaluation = evaluate_candidate(
        [{"neighbor": "selected"}],
        folds,
        sizing=object(),
        callback=callback,
        neighbors=lambda selected: neighbors,
    )

    assert evaluation.neighbor_positive == 7
    assert evaluation.neighbor_valid == 10
    assert evaluation.neighbor_robustness == pytest.approx(0.70)


def test_research_gate_boundaries_and_strict_majority():
    metrics = Metrics(
        total_return=0.10,
        sharpe=1.25,
        max_drawdown=-0.30,
        trades=30,
        win_rate=0.50,
        fold_returns=(0.01, 0.02, -0.01),
    )
    evaluation = CandidateEvaluation(
        fold_results=(),
        metrics=metrics,
        neighbor_robustness=0.70,
        neighbor_positive=7,
        neighbor_valid=10,
    )

    assert passes_research_gates(evaluation)
    assert not passes_research_gates(
        replace(evaluation, metrics=replace(metrics, trades=29))
    )
    assert not passes_research_gates(
        replace(evaluation, metrics=replace(metrics, max_drawdown=-0.300001))
    )
    assert not passes_research_gates(
        replace(evaluation, neighbor_robustness=0.699999)
    )
    assert not passes_research_gates(
        replace(metrics, fold_returns=(0.01, -0.01, -0.02)),
        neighbor_robustness=0.70,
    )
    assert not passes_research_gates(
        replace(metrics, fold_returns=(0.01, 0.02, -0.01, -0.02)),
        neighbor_robustness=0.70,
    )
