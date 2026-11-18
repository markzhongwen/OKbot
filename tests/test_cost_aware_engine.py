from datetime import datetime, timezone

import pytest

from cost_aware.config import ExecutionConfig, SizingConfig
from cost_aware.engine import BacktestResult, Trade, run_backtest, run_backtest_result


UTC = timezone.utc


def ts(hour: int) -> datetime:
    return datetime(2026, 1, 1, hour, tzinfo=UTC)


def candle(symbol: str, hour: int, open_: float, high: float, low: float, close: float):
    return {
        "symbol": symbol,
        "timestamp": ts(hour),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }


def signal(
    symbol: str,
    hour: int,
    side: str,
    stop: float | None,
    take_profit: float | None,
    **lifecycle,
):
    result = {
        "symbol": symbol,
        "timestamp": ts(hour),
        "side": side,
        **lifecycle,
    }
    if stop is not None:
        result["stop"] = stop
    if take_profit is not None:
        result["take_profit"] = take_profit
    return result


def configs(*, max_positions: int = 3):
    return (
        ExecutionConfig(
            signal_lag_bars=1,
            fee_per_side=0.0005,
            slippage_per_side=0.0002,
        ),
        SizingConfig(
            initial_equity=1_000,
            margin_fraction=0.1,
            leverage=1,
            max_positions=max_positions,
        ),
    )


def test_enters_at_next_open_with_adverse_long_entry_and_exit_slippage():
    execution, sizing = configs()
    candles = [
        candle("BTC", 0, 90, 91, 89, 90),
        candle("BTC", 1, 100, 111, 99, 108),
    ]

    trades = run_backtest(
        candles,
        [signal("BTC", 0, "long", stop=95, take_profit=110)],
        [],
        execution,
        sizing,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert isinstance(trade, Trade)
    assert trade.entry_time == ts(1)
    assert trade.entry_price == pytest.approx(100 * 1.0002)
    assert trade.exit_price == pytest.approx(110 * 0.9998)


def test_short_entry_and_exit_slippage_are_adverse():
    execution, sizing = configs()
    candles = [
        candle("ETH", 0, 110, 111, 109, 110),
        candle("ETH", 1, 100, 101, 89, 92),
    ]

    trade = run_backtest(
        candles,
        [signal("ETH", 0, "short", stop=105, take_profit=90)],
        [],
        execution,
        sizing,
    )[0]

    assert trade.entry_price == pytest.approx(100 * 0.9998)
    assert trade.exit_price == pytest.approx(90 * 1.0002)


def test_charges_fees_on_both_filled_sides():
    execution, sizing = configs()
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 101, 99, 100),
        candle("BTC", 2, 100, 110, 90, 100),
    ]

    trade = run_backtest(
        candles,
        [signal("BTC", 0, "long", stop=100, take_profit=200)],
        [],
        execution,
        sizing,
    )[0]

    assert trade.entry_fee == pytest.approx(trade.quantity * trade.entry_price * 0.0005)
    assert trade.exit_fee == pytest.approx(trade.quantity * trade.exit_price * 0.0005)
    assert trade.fees == pytest.approx(trade.entry_fee + trade.exit_fee)
    assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.fees + trade.funding_pnl)


@pytest.mark.parametrize(
    ("side", "expected_sign"),
    [("long", -1), ("short", 1)],
)
def test_positive_funding_is_paid_by_longs_and_received_by_shorts(side, expected_sign):
    execution, sizing = configs()
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 101, 99, 100),
        candle("BTC", 2, 100, 110, 90, 100),
    ]
    stop = 95 if side == "long" else 200
    take_profit = 200 if side == "long" else 95

    trade = run_backtest(
        candles,
        [signal("BTC", 0, side, stop=stop, take_profit=take_profit)],
        [{"symbol": "BTC", "timestamp": ts(2), "rate": 0.01}],
        execution,
        sizing,
    )[0]

    assert trade.funding_pnl * expected_sign > 0
    assert abs(trade.funding_pnl) == pytest.approx(
        trade.quantity * candles[2]["open"] * 0.01
    )


def test_funding_at_entry_timestamp_is_not_applied_but_later_boundary_is():
    execution, sizing = configs()
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 101, 99, 100),
        candle("BTC", 2, 100, 110, 90, 100),
    ]

    trade = run_backtest(
        candles,
        [signal("BTC", 0, "long", stop=95, take_profit=200)],
        [
            {"symbol": "BTC", "timestamp": ts(1), "rate": 0.02},
            {"symbol": "BTC", "timestamp": ts(2), "rate": 0.01},
        ],
        execution,
        sizing,
    )[0]

    assert trade.funding_pnl == pytest.approx(
        -(trade.quantity * candles[2]["open"] * 0.01)
    )


@pytest.mark.parametrize(
    ("side", "stop", "take_profit"),
    [("long", 95, 105), ("short", 105, 95)],
)
def test_same_candle_stop_and_take_profit_uses_stop(side, stop, take_profit):
    execution, sizing = configs()
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 106, 94, 100),
    ]

    trade = run_backtest(
        candles,
        [signal("BTC", 0, side, stop=stop, take_profit=take_profit)],
        [],
        execution,
        sizing,
    )[0]

    expected = stop * (0.9998 if side == "long" else 1.0002)
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(expected)


def test_rejects_entries_beyond_concurrent_position_limit():
    execution, sizing = configs(max_positions=2)
    candles = []
    signals = []
    for symbol in ("BTC", "ETH", "SOL"):
        candles.extend(
            [
                candle(symbol, 0, 100, 100, 100, 100),
                candle(symbol, 1, 100, 101, 99, 100),
                candle(symbol, 2, 100, 110, 90, 100),
            ]
        )
        signals.append(signal(symbol, 0, "long", stop=100, take_profit=200))

    trades = run_backtest(candles, signals, [], execution, sizing)

    assert len(trades) == 2
    assert {trade.symbol for trade in trades} == {"BTC", "ETH"}


def test_entry_fee_and_funding_immediately_reduce_equity_used_for_next_entry():
    execution, sizing = configs()
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 101, 99, 100),
        candle("BTC", 2, 100, 101, 99, 100),
        candle("BTC", 3, 100, 101, 89, 90),
        candle("ETH", 1, 100, 100, 100, 100),
        candle("ETH", 2, 100, 101, 98, 99),
    ]

    trades = run_backtest(
        candles,
        [
            signal("BTC", 0, "long", stop=90, take_profit=200),
            signal("ETH", 1, "long", stop=99, take_profit=200),
        ],
        [{"symbol": "BTC", "timestamp": ts(2), "rate": 0.01}],
        execution,
        sizing,
    )

    btc_trade = next(trade for trade in trades if trade.symbol == "BTC")
    eth_trade = next(trade for trade in trades if trade.symbol == "ETH")
    equity_before_eth = 1_000 - btc_trade.entry_fee + btc_trade.funding_pnl
    expected_eth_notional = equity_before_eth * sizing.margin_fraction * sizing.leverage
    assert eth_trade.quantity * eth_trade.entry_price == pytest.approx(
        expected_eth_notional
    )


def test_existing_gap_exit_releases_slot_before_same_open_new_entry():
    execution, sizing = configs(max_positions=1)
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 101, 99, 100),
        candle("BTC", 2, 90, 91, 89, 90),
        candle("ETH", 1, 100, 100, 100, 100),
        candle("ETH", 2, 100, 101, 99, 100),
        candle("ETH", 3, 100, 101, 94, 95),
    ]

    trades = run_backtest(
        candles,
        [
            signal("BTC", 0, "long", stop=95, take_profit=200),
            signal("ETH", 1, "long", stop=95, take_profit=200),
        ],
        [],
        execution,
        sizing,
    )

    assert [trade.symbol for trade in trades] == ["BTC", "ETH"]
    assert trades[0].exit_time == trades[1].entry_time == ts(2)


def test_intrabar_exit_does_not_retroactively_free_slot_for_same_open_entry():
    execution, sizing = configs(max_positions=1)
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 101, 99, 100),
        candle("BTC", 2, 100, 101, 94, 95),
        candle("ETH", 1, 100, 100, 100, 100),
        candle("ETH", 2, 100, 101, 99, 100),
        candle("ETH", 3, 100, 101, 94, 95),
    ]

    trades = run_backtest(
        candles,
        [
            signal("BTC", 0, "long", stop=95, take_profit=200),
            signal("ETH", 1, "long", stop=95, take_profit=200),
        ],
        [],
        execution,
        sizing,
    )

    assert [trade.symbol for trade in trades] == ["BTC"]


def test_time_stop_one_exits_at_next_bar_close_not_entry_bar_close():
    execution, sizing = configs()
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 101, 99, 100),
        candle("BTC", 2, 101, 103, 100, 102),
    ]

    trade = run_backtest(
        candles,
        [
            signal(
                "BTC",
                0,
                "long",
                stop=90,
                take_profit=200,
                time_stop_bars=1,
            )
        ],
        [],
        execution,
        sizing,
    )[0]

    assert trade.exit_time == ts(2)
    assert trade.exit_reason == "time_stop"
    assert trade.exit_price == pytest.approx(102 * 0.9998)


@pytest.mark.parametrize(
    ("side", "low", "high", "expected_reference", "slippage_multiplier"),
    [
        ("long", 114, 121, 115, 0.9998),
        ("short", 119, 126, 125, 1.0002),
    ],
)
def test_distance_risk_is_anchored_to_unadjusted_next_open_and_takes_precedence(
    side,
    low,
    high,
    expected_reference,
    slippage_multiplier,
):
    execution, sizing = configs()
    candles = [
        candle("BTC", 0, 100, 101, 99, 100),
        candle("BTC", 1, 120, high, low, 120),
    ]

    trade = run_backtest(
        candles,
        [
            signal(
                "BTC",
                0,
                side,
                stop=50,
                take_profit=250,
                stop_distance=5,
                take_profit_distance=10,
            )
        ],
        [],
        execution,
        sizing,
    )[0]

    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(
        expected_reference * slippage_multiplier
    )


def test_intrabar_stop_precedes_time_stop_on_same_bar():
    execution, sizing = configs()
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 101, 99, 100),
        candle("BTC", 2, 100, 101, 94, 98),
    ]

    trade = run_backtest(
        candles,
        [
            signal(
                "BTC",
                0,
                "long",
                stop=95,
                take_profit=200,
                time_stop_bars=2,
            )
        ],
        [],
        execution,
        sizing,
    )[0]

    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(95 * 0.9998)


def test_time_stop_at_close_does_not_release_slot_for_same_bar_open_entry():
    execution, sizing = configs(max_positions=1)
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 101, 99, 100),
        candle("BTC", 2, 100, 101, 99, 100),
        candle("ETH", 1, 100, 100, 100, 100),
        candle("ETH", 2, 100, 101, 99, 100),
        candle("ETH", 3, 100, 101, 94, 95),
    ]

    trades = run_backtest(
        candles,
        [
            signal(
                "BTC",
                0,
                "long",
                stop=90,
                take_profit=200,
                time_stop_bars=1,
            ),
            signal("ETH", 1, "long", stop=95, take_profit=200),
        ],
        [],
        execution,
        sizing,
    )

    assert [trade.symbol for trade in trades] == ["BTC"]
    assert trades[0].exit_reason == "time_stop"


def test_cooldown_uses_symbol_bar_indexes_and_allows_exact_boundary():
    execution, sizing = configs()
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 101, 99, 100),
        candle("BTC", 2, 100, 101, 94, 98),
        candle("BTC", 3, 100, 101, 99, 100),
        candle("BTC", 4, 100, 101, 99, 100),
        candle("BTC", 5, 100, 101, 94, 98),
    ]

    trades = run_backtest(
        candles,
        [
            signal(
                "BTC", 0, "long", 95, 200, cooldown_bars=2
            ),
            signal(
                "BTC", 2, "long", 95, 200, cooldown_bars=2
            ),
            signal(
                "BTC", 3, "long", 95, 200, cooldown_bars=2
            ),
        ],
        [],
        execution,
        sizing,
    )

    assert [trade.entry_time for trade in trades] == [ts(1), ts(4)]
    assert all(trade.exit_reason == "stop" for trade in trades)


def test_entry_capacity_prefers_priority_then_symbol_over_input_order():
    execution, sizing = configs(max_positions=2)
    candles = []
    priorities = {"BTC": 1.0, "XRP": 2.0, "SOL": 2.0, "ETH": 3.0}
    signals = []
    for symbol in ("BTC", "XRP", "SOL", "ETH"):
        candles.extend(
            [
                candle(symbol, 0, 100, 100, 100, 100),
                candle(symbol, 1, 100, 101, 99, 100),
                candle(symbol, 2, 100, 101, 94, 95),
            ]
        )
        signals.append(
            signal(
                symbol,
                0,
                "long",
                stop=95,
                take_profit=200,
                priority=priorities[symbol],
            )
        )

    trades = run_backtest(candles, signals, [], execution, sizing)

    assert {trade.symbol for trade in trades} == {"ETH", "SOL"}


def test_backtest_result_records_boundaries_and_same_timestamp_event_order():
    execution, sizing = configs()
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 101, 99, 100),
        candle("BTC", 2, 100, 101, 94, 96),
    ]

    result = run_backtest_result(
        candles,
        [signal("BTC", 0, "long", stop=95, take_profit=200)],
        [{"symbol": "BTC", "timestamp": ts(2), "rate": 0.01}],
        execution,
        sizing,
        start_time=ts(0),
        end_time=ts(3),
    )

    assert isinstance(result, BacktestResult)
    assert result.equity_events[0].kind == "window_start"
    assert result.equity_events[-1].kind == "window_end"
    assert [event.sequence for event in result.equity_events] == list(
        range(len(result.equity_events))
    )
    assert [event.kind for event in result.equity_events if event.timestamp == ts(2)] == [
        "funding",
        "exit",
        "bar_mark",
    ]
    funding_event = next(
        event for event in result.equity_events if event.kind == "funding"
    )
    assert funding_event.realized_equity < sizing.initial_equity


def test_bar_marks_include_unrealized_pnl_for_event_time_drawdown():
    execution, sizing = configs()
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 101, 99, 100),
        candle("BTC", 2, 80, 81, 79, 80),
    ]

    result = run_backtest_result(
        candles,
        [signal("BTC", 0, "long", stop=50, take_profit=200)],
        [],
        execution,
        sizing,
        start_time=ts(0),
        end_time=ts(3),
        force_close_at_end=False,
    )

    entry_event = next(
        event for event in result.equity_events if event.kind == "entry_fee"
    )
    final_mark = [
        event for event in result.equity_events if event.kind == "bar_mark"
    ][-1]
    assert entry_event.realized_equity == pytest.approx(999.95)
    assert final_mark.realized_equity == pytest.approx(entry_event.realized_equity)
    assert final_mark.equity < final_mark.realized_equity
    assert min(event.equity for event in result.equity_events) == pytest.approx(
        final_mark.equity
    )


def test_force_close_at_end_uses_last_close_adverse_slippage_and_exit_fee():
    execution, sizing = configs()
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 101, 99, 100),
        candle("BTC", 2, 105, 106, 104, 105),
    ]

    result = run_backtest_result(
        candles,
        [signal("BTC", 0, "long", stop=50, take_profit=200)],
        [],
        execution,
        sizing,
        start_time=ts(0),
        end_time=ts(3),
        force_close_at_end=True,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_time == ts(3)
    assert trade.exit_reason == "force_close"
    assert trade.exit_price == pytest.approx(105 * 0.9998)
    assert trade.exit_fee == pytest.approx(
        trade.quantity * trade.exit_price * execution.fee_per_side
    )
    assert result.equity_events[-2].kind == "force_close"
    assert result.equity_events[-1].equity == pytest.approx(
        sizing.initial_equity + trade.net_pnl
    )


def test_legacy_run_backtest_remains_a_trade_list_wrapper():
    execution, sizing = configs()
    candles = [
        candle("BTC", 0, 100, 100, 100, 100),
        candle("BTC", 1, 100, 101, 94, 95),
    ]
    signals = [signal("BTC", 0, "long", stop=95, take_profit=200)]

    legacy = run_backtest(candles, signals, [], execution, sizing)

    assert isinstance(legacy, list)
    assert len(legacy) == 1
