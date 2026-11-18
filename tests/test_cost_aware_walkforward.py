from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cost_aware.engine import BacktestResult, EquityEvent
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


def _window_trades(
    count: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    wins: int | None = None,
    exit_at_end: bool = False,
):
    win_count = count if wins is None else wins
    return [
        SimpleNamespace(
            net_pnl=1.0 if index < win_count else -1.0,
            entry_time=start,
            exit_time=end if exit_at_end else start + (end - start) / 2,
            exit_reason="force_close" if exit_at_end else "stop",
        )
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


def test_build_folds_rejects_start_not_aligned_to_utc_midnight():
    with pytest.raises(ValueError, match="UTC midnight"):
        build_folds(pd.Timestamp("2026-01-01 00:01", tz="UTC"))


def test_manual_fold_rejects_any_boundary_not_aligned_to_utc_midnight():
    start = pd.Timestamp("2026-01-01", tz="UTC")

    with pytest.raises(ValueError, match="UTC midnight"):
        Fold(
            index=0,
            is_start=start + pd.Timedelta(minutes=1),
            is_end=start + pd.Timedelta(days=45),
            oos_start=start + pd.Timedelta(days=45),
            oos_end=start + pd.Timedelta(days=60),
        )


def test_evaluate_candidate_rejects_gaps_between_manual_oos_folds():
    start = pd.Timestamp("2026-01-01", tz="UTC")
    folds = (
        Fold(
            index=0,
            is_start=start,
            is_end=start + pd.Timedelta(days=45),
            oos_start=start + pd.Timedelta(days=45),
            oos_end=start + pd.Timedelta(days=60),
        ),
        Fold(
            index=1,
            is_start=start + pd.Timedelta(days=16),
            is_end=start + pd.Timedelta(days=61),
            oos_start=start + pd.Timedelta(days=61),
            oos_end=start + pd.Timedelta(days=76),
        ),
    )

    with pytest.raises(ValueError, match="adjacent"):
        evaluate_candidate(
            [{"lookback": 20}],
            folds,
            sizing=object(),
            callback=lambda *args: None,
            neighbors=lambda selected: (),
        )


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


def test_daily_sharpe_preserves_first_boundary_before_same_timestamp_costs():
    start = pd.Timestamp("2026-01-01", tz="UTC")
    index = pd.MultiIndex.from_tuples(
        [
            (start, 0),
            (start, 1),
            (start + pd.Timedelta(days=1), 2),
            (start + pd.Timedelta(days=2), 3),
        ],
        names=["timestamp", "sequence"],
    )
    equity = pd.Series([100.0, 99.0, 99.0, 99.0], index=index)
    daily_returns = pd.Series([-0.01, 0.0])
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


def test_metrics_accept_same_timestamp_events_ordered_by_strict_sequence():
    timestamp = pd.Timestamp("2026-01-02", tz="UTC")
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-01", tz="UTC"), 0),
            (timestamp, 1),
            (timestamp, 2),
            (pd.Timestamp("2026-01-03", tz="UTC"), 3),
        ],
        names=["timestamp", "sequence"],
    )
    equity = pd.Series([100.0, 110.0, 80.0, 100.0], index=index)

    metrics = compute_metrics(equity, _trades(1))

    assert metrics.total_return == pytest.approx(0.0)
    assert metrics.max_drawdown == pytest.approx(80 / 110 - 1)


def test_metrics_reject_nonincreasing_event_sequence():
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-01", tz="UTC"), 2),
            (pd.Timestamp("2026-01-02", tz="UTC"), 1),
        ],
        names=["timestamp", "sequence"],
    )

    with pytest.raises(ValueError, match="sequence.*strictly increasing"):
        compute_metrics(pd.Series([100.0, 101.0], index=index), _trades(1))


def test_fold_equity_must_cover_both_boundaries_so_quiet_tail_is_counted():
    folds = build_folds(
        pd.Timestamp("2026-01-01", tz="UTC"),
        n_folds=1,
    )

    def callback(param, fold, sizing, phase):
        del param, sizing
        start, end = _window(fold, phase)
        return (
            _equity(start, end - pd.Timedelta(days=1), 0.01),
            _window_trades(1, start, end),
        )

    with pytest.raises(ValueError, match="cover.*window boundaries"):
        evaluate_candidate(
            [{"lookback": 20}],
            folds,
            sizing=object(),
            callback=callback,
            neighbors=lambda selected: (),
        )


@pytest.mark.parametrize(
    ("entry_offset", "exit_at_end", "exit_reason", "message"),
    [
        (pd.Timedelta(days=-1), False, "stop", "entry_time"),
        (pd.Timedelta(0), True, "stop", "exit_time"),
    ],
)
def test_fold_trade_validation_rejects_out_of_window_trades(
    entry_offset,
    exit_at_end,
    exit_reason,
    message,
):
    folds = build_folds(pd.Timestamp("2026-01-01", tz="UTC"), n_folds=1)

    def callback(param, fold, sizing, phase):
        del param, sizing
        start, end = _window(fold, phase)
        trade = SimpleNamespace(
            net_pnl=1.0,
            entry_time=start + entry_offset,
            exit_time=end if exit_at_end else start + pd.Timedelta(days=1),
            exit_reason=exit_reason,
        )
        return _equity(start, end, 0.01), [trade]

    with pytest.raises(ValueError, match=message):
        evaluate_candidate(
            [{"lookback": 20}],
            folds,
            sizing=object(),
            callback=callback,
            neighbors=lambda selected: (),
        )


def test_force_close_trade_at_fold_end_is_explicitly_allowed():
    folds = build_folds(pd.Timestamp("2026-01-01", tz="UTC"), n_folds=1)

    def callback(param, fold, sizing, phase):
        del param, sizing
        start, end = _window(fold, phase)
        return (
            _equity(start, end, 0.01),
            _window_trades(1, start, end, exit_at_end=True),
        )

    evaluation = evaluate_candidate(
        [{"lookback": 20}],
        folds,
        sizing=object(),
        callback=callback,
        neighbors=lambda selected: (),
    )

    assert evaluation.metrics.trades == 1


def test_evaluate_candidate_accepts_engine_backtest_result_directly():
    folds = build_folds(pd.Timestamp("2026-01-01", tz="UTC"), n_folds=1)

    def callback(param, fold, sizing, phase):
        del param, sizing
        start, end = _window(fold, phase)
        trade = _window_trades(1, start, end, exit_at_end=True)[0]
        return BacktestResult(
            trades=(trade,),
            equity_events=(
                EquityEvent(start, 0, 100.0, 100.0, "window_start"),
                EquityEvent(end, 1, 101.0, 101.0, "window_end"),
            ),
        )

    evaluation = evaluate_candidate(
        [{"lookback": 20}],
        folds,
        sizing=object(),
        callback=callback,
        neighbors=lambda selected: (),
    )

    assert evaluation.metrics.total_return == pytest.approx(0.01)
    assert evaluation.metrics.trades == 1


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
                return (
                    pd.Series(values, index=timestamps),
                    _window_trades(1, start, end),
                )
            midpoint = start + (end - start) / 2
            return (
                pd.Series(
                    [100.0, 80.0, 101.0],
                    index=pd.DatetimeIndex([start, midpoint, end]),
                ),
                _window_trades(1, start, end),
            )
        future_return = 10.0 if param["name"] == "beta" else 0.01
        return (
            _equity(start, end, future_return),
            _window_trades(1, start, end),
        )

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
        return (
            _equity(start, end, 0.01 if phase == "is" else 0.10),
            _window_trades(2, start, end),
        )

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


def test_neighbor_robustness_counts_invalid_and_losing_neighbors_as_nonpositive():
    folds = build_folds(
        pd.Timestamp("2026-01-01", tz="UTC"),
        n_folds=1,
    )
    neighbors = [
        *({"neighbor": f"good-{index}"} for index in range(7)),
        *({"neighbor": f"loss-{index}"} for index in range(3)),
        {"neighbor": "invalid"},
        {"neighbor": "invalid-output"},
        {"neighbor": "raises-key"},
        {"neighbor": "bad-event"},
        {"neighbor": "bad-pnl"},
    ]

    def callback(param, fold, sizing, phase):
        del sizing
        start, end = _window(fold, phase)
        name = param.get("neighbor", "selected")
        if phase == "oos" and name == "invalid":
            return None
        if phase == "oos" and name == "invalid-output":
            return (
                _equity(start, end - pd.Timedelta(days=1), 0.01),
                _window_trades(1, start, end),
            )
        if phase == "oos" and name == "raises-key":
            raise KeyError("malformed neighbor data")
        if phase == "oos" and name == "bad-event":
            return BacktestResult(trades=(), equity_events=(object(),))
        if phase == "oos" and name == "bad-pnl":
            trade = _window_trades(1, start, end)[0]
            trade.net_pnl = "not-numeric"
            return _equity(start, end, 0.01), [trade]
        result = -0.01 if name.startswith("loss-") else 0.01
        return _equity(start, end, result), _window_trades(1, start, end)

    evaluation = evaluate_candidate(
        [{"neighbor": "selected"}],
        folds,
        sizing=object(),
        callback=callback,
        neighbors=lambda selected: neighbors,
    )

    assert evaluation.neighbor_positive == 7
    assert evaluation.neighbor_valid == 15
    assert evaluation.neighbor_robustness == pytest.approx(7 / 15)


def test_neighbor_robustness_compounds_each_unique_config_across_oos_folds():
    folds = build_folds(pd.Timestamp("2026-01-01", tz="UTC"))
    compounded_returns = (0.10, -0.01, -0.01)

    def callback(param, fold, sizing, phase):
        del sizing
        start, end = _window(fold, phase)
        name = param["name"]
        if phase == "is" or name == "selected":
            result = 0.01
        elif name == "compound":
            result = compounded_returns[fold.index]
        elif name == "loss":
            result = -0.02
        else:
            return None
        return _equity(start, end, result), _window_trades(1, start, end)

    evaluation = evaluate_candidate(
        [{"name": "selected"}],
        folds,
        sizing=object(),
        callback=callback,
        neighbors=lambda selected: [
            {"name": "compound"},
            {"name": "loss"},
            {"name": "invalid"},
        ],
    )

    assert np.prod([1.10, 0.99, 0.99]) - 1 > 0
    assert evaluation.neighbor_positive == 1
    assert evaluation.neighbor_valid == 3
    assert evaluation.neighbor_robustness == pytest.approx(1 / 3)


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

    assert passes_research_gates(metrics, neighbor_robustness=0.70)
    assert not passes_research_gates(evaluation)
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
        replace(metrics, fold_returns=(0.01, 0.02, 0.03, -0.02)),
        neighbor_robustness=0.70,
    )
    assert not passes_research_gates(
        replace(metrics, fold_returns=(0.01, 0.02)),
        neighbor_robustness=0.70,
    )
