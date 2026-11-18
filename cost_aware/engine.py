from dataclasses import dataclass
from typing import Iterable, Mapping

from .config import ExecutionConfig, SizingConfig


Row = Mapping[str, object]


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: str
    entry_time: object
    exit_time: object
    entry_price: float
    exit_price: float
    quantity: float
    entry_fee: float
    exit_fee: float
    funding_pnl: float
    exit_reason: str

    @property
    def fees(self) -> float:
        return self.entry_fee + self.exit_fee

    @property
    def gross_pnl(self) -> float:
        direction = 1 if self.side == "long" else -1
        return direction * self.quantity * (self.exit_price - self.entry_price)

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees + self.funding_pnl


@dataclass
class _Position:
    symbol: str
    side: str
    entry_time: object
    entry_price: float
    quantity: float
    entry_fee: float
    stop: float
    take_profit: float
    entry_bar_index: int
    time_stop_bars: int
    funding_pnl: float = 0.0


def _adverse_fill(reference: float, action: str, slippage: float) -> float:
    multiplier = 1 + slippage if action == "buy" else 1 - slippage
    return reference * multiplier


def _exit_event(position: _Position, bar: Row) -> tuple[float, str] | None:
    open_price = float(bar["open"])
    high = float(bar["high"])
    low = float(bar["low"])

    if position.side == "long":
        if low <= position.stop:
            return min(open_price, position.stop), "stop"
        if high >= position.take_profit:
            return max(open_price, position.take_profit), "take_profit"
    else:
        if high >= position.stop:
            return max(open_price, position.stop), "stop"
        if low <= position.take_profit:
            return min(open_price, position.take_profit), "take_profit"
    return None


def _open_exit_event(position: _Position, bar: Row) -> tuple[float, str] | None:
    open_price = float(bar["open"])
    if position.side == "long":
        if open_price <= position.stop:
            return open_price, "stop"
        if open_price >= position.take_profit:
            return open_price, "take_profit"
    else:
        if open_price >= position.stop:
            return open_price, "stop"
        if open_price <= position.take_profit:
            return open_price, "take_profit"
    return None


def run_backtest(
    candles: Iterable[Row],
    signals: Iterable[Row],
    funding: Iterable[Row],
    execution: ExecutionConfig,
    sizing: SizingConfig,
) -> list[Trade]:
    """Run deterministic fill, cost, and funding events without external I/O."""
    bars = list(candles)
    bars_by_symbol: dict[str, list[Row]] = {}
    for bar in bars:
        bars_by_symbol.setdefault(str(bar["symbol"]), []).append(bar)
    bar_indexes: dict[tuple[str, object], int] = {}
    for symbol, symbol_bars in bars_by_symbol.items():
        symbol_bars.sort(key=lambda row: row["timestamp"])
        for index, bar in enumerate(symbol_bars):
            bar_indexes[(symbol, bar["timestamp"])] = index

    entries: dict[tuple[object, str], list[Row]] = {}
    for signal in signals:
        symbol = str(signal["symbol"])
        symbol_bars = bars_by_symbol.get(symbol, [])
        signal_index = bar_indexes.get((symbol, signal["timestamp"]))
        if signal_index is None:
            continue
        entry_index = signal_index + execution.signal_lag_bars
        if entry_index < len(symbol_bars):
            entry_time = symbol_bars[entry_index]["timestamp"]
            entries.setdefault((entry_time, symbol), []).append(signal)

    funding_by_event: dict[object, list[tuple[str, float]]] = {}
    event_times = {bar["timestamp"] for bar in bars}
    for payment in funding:
        funding_by_event.setdefault(payment["timestamp"], []).append(
            (str(payment["symbol"]), float(payment["rate"]))
        )
        event_times.add(payment["timestamp"])

    bars_by_event: dict[object, list[tuple[int, Row]]] = {}
    for sequence, bar in enumerate(bars):
        bars_by_event.setdefault(bar["timestamp"], []).append((sequence, bar))

    positions: dict[str, _Position] = {}
    last_exit_indexes: dict[str, int] = {}
    last_price: dict[str, float] = {}
    equity = sizing.initial_equity
    trades: list[Trade] = []

    def close_on_bar(
        symbol: str,
        bar: Row,
        event_time: object,
        bar_index: int,
        *,
        open_only: bool = False,
        close_reason: str | None = None,
    ) -> bool:
        nonlocal equity
        position = positions.get(symbol)
        if position is None:
            return False
        if close_reason is not None:
            exit_event = (float(bar["close"]), close_reason)
        else:
            exit_event = (
                _open_exit_event(position, bar)
                if open_only
                else _exit_event(position, bar)
            )
        if exit_event is None:
            return False
        reference, reason = exit_event
        action = "sell" if position.side == "long" else "buy"
        exit_price = _adverse_fill(reference, action, execution.slippage_per_side)
        trade = Trade(
            symbol=symbol,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=event_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_fee=position.entry_fee,
            exit_fee=position.quantity * exit_price * execution.fee_per_side,
            funding_pnl=position.funding_pnl,
            exit_reason=reason,
        )
        trades.append(trade)
        # Entry fees and funding were booked when their events occurred.
        equity += trade.gross_pnl - trade.exit_fee
        del positions[symbol]
        last_exit_indexes[symbol] = bar_index
        return True

    for event_time in sorted(event_times):
        current_bars = sorted(bars_by_event.get(event_time, []))
        bar_for_symbol = {str(bar["symbol"]): bar for _, bar in current_bars}

        # Funding precedes fills at the same timestamp: a new position was not
        # already open at the published settlement boundary.
        for symbol, rate in funding_by_event.get(event_time, []):
            if symbol not in positions:
                continue
            mark = (
                float(bar_for_symbol[symbol]["open"])
                if symbol in bar_for_symbol
                else last_price.get(symbol)
            )
            if mark is None:
                continue
            position = positions[symbol]
            direction = 1 if position.side == "long" else -1
            payment = -direction * position.quantity * mark * rate
            position.funding_pnl += payment
            equity += payment

        # Only exits knowable at the open can free capacity for open entries.
        existing_symbols = set(positions)
        for _, bar in current_bars:
            symbol = str(bar["symbol"])
            if symbol in existing_symbols:
                close_on_bar(
                    symbol,
                    bar,
                    event_time,
                    bar_indexes[(symbol, event_time)],
                    open_only=True,
                )

        # Rank simultaneous candidates independently of candle/signal input order.
        candidates: list[tuple[float, str, Row, Row]] = []
        for _, bar in current_bars:
            symbol = str(bar["symbol"])
            for signal in entries.get((event_time, symbol), []):
                priority = float(signal.get("priority", 0.0))
                candidates.append((-priority, symbol, signal, bar))

        for _, symbol, signal, bar in sorted(
            candidates, key=lambda candidate: (candidate[0], candidate[1])
        ):
            if symbol in positions or len(positions) >= sizing.max_positions:
                continue
            side = str(signal["side"])
            if side not in {"long", "short"}:
                raise ValueError(f"unsupported side: {side}")
            bar_index = bar_indexes[(symbol, event_time)]
            cooldown_bars = int(signal.get("cooldown_bars", 0))
            if cooldown_bars < 0:
                raise ValueError("cooldown_bars cannot be negative")
            last_exit_index = last_exit_indexes.get(symbol)
            if (
                last_exit_index is not None
                and bar_index - last_exit_index < cooldown_bars
            ):
                continue
            time_stop_bars = int(signal.get("time_stop_bars", 0))
            if time_stop_bars < 0:
                raise ValueError("time_stop_bars cannot be negative")
            reference_open = float(bar["open"])
            has_stop_distance = "stop_distance" in signal
            has_take_profit_distance = "take_profit_distance" in signal
            if has_stop_distance or has_take_profit_distance:
                # Distance fields deliberately take precedence when a caller
                # also supplies backward-compatible absolute levels.
                if not (has_stop_distance and has_take_profit_distance):
                    raise ValueError(
                        "stop_distance and take_profit_distance must be paired"
                    )
                stop_distance = float(signal["stop_distance"])
                take_profit_distance = float(signal["take_profit_distance"])
                if stop_distance <= 0 or take_profit_distance <= 0:
                    raise ValueError("risk distances must be positive")
                if side == "long":
                    stop = reference_open - stop_distance
                    take_profit = reference_open + take_profit_distance
                else:
                    stop = reference_open + stop_distance
                    take_profit = reference_open - take_profit_distance
            else:
                # Backward-compatible handwritten signals use absolute levels.
                stop = float(signal["stop"])
                take_profit = float(signal["take_profit"])
            action = "buy" if side == "long" else "sell"
            entry_price = _adverse_fill(
                reference_open, action, execution.slippage_per_side
            )
            notional = equity * sizing.margin_fraction * sizing.leverage
            quantity = notional / entry_price
            entry_fee = quantity * entry_price * execution.fee_per_side
            positions[symbol] = _Position(
                symbol=symbol,
                side=side,
                entry_time=event_time,
                entry_price=entry_price,
                quantity=quantity,
                entry_fee=entry_fee,
                stop=stop,
                take_profit=take_profit,
                entry_bar_index=bar_index,
                time_stop_bars=time_stop_bars,
            )
            equity -= entry_fee

        # Intrabar ranges are now known. They apply to both carried and newly
        # opened positions, but cannot retroactively change open admission.
        for _, bar in current_bars:
            symbol = str(bar["symbol"])
            if symbol in positions:
                close_on_bar(
                    symbol,
                    bar,
                    event_time,
                    bar_indexes[(symbol, event_time)],
                )

        # Time stops execute at the close, after all open and intrabar events.
        # Consequently they cannot release an entry slot at this bar's open.
        for _, bar in current_bars:
            symbol = str(bar["symbol"])
            position = positions.get(symbol)
            if position is None or position.time_stop_bars == 0:
                continue
            bar_index = bar_indexes[(symbol, event_time)]
            held_bars = bar_index - position.entry_bar_index
            if held_bars >= position.time_stop_bars:
                close_on_bar(
                    symbol,
                    bar,
                    event_time,
                    bar_index,
                    close_reason="time_stop",
                )

        for _, bar in current_bars:
            symbol = str(bar["symbol"])
            last_price[symbol] = float(bar["close"])

    return trades
