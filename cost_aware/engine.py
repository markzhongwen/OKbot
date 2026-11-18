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
    for symbol_bars in bars_by_symbol.values():
        symbol_bars.sort(key=lambda row: row["timestamp"])

    entries: dict[tuple[object, str], list[Row]] = {}
    for signal in signals:
        symbol = str(signal["symbol"])
        symbol_bars = bars_by_symbol.get(symbol, [])
        signal_index = next(
            (
                index
                for index, bar in enumerate(symbol_bars)
                if bar["timestamp"] == signal["timestamp"]
            ),
            None,
        )
        if signal_index is None:
            continue
        entry_index = signal_index + execution.signal_lag_bars
        if entry_index < len(symbol_bars):
            entry_time = symbol_bars[entry_index]["timestamp"]
            entries.setdefault((entry_time, symbol), []).append(signal)

    funding_by_event: dict[tuple[object, str], list[float]] = {}
    event_times = {bar["timestamp"] for bar in bars}
    for payment in funding:
        key = (payment["timestamp"], str(payment["symbol"]))
        funding_by_event.setdefault(key, []).append(float(payment["rate"]))
        event_times.add(payment["timestamp"])

    bars_by_event: dict[object, list[tuple[int, Row]]] = {}
    for sequence, bar in enumerate(bars):
        bars_by_event.setdefault(bar["timestamp"], []).append((sequence, bar))

    positions: dict[str, _Position] = {}
    last_price: dict[str, float] = {}
    equity = sizing.initial_equity
    trades: list[Trade] = []

    for event_time in sorted(event_times):
        current_bars = bars_by_event.get(event_time, [])
        bar_for_symbol = {str(bar["symbol"]): bar for _, bar in current_bars}

        # Funding precedes fills at the same timestamp: a new position was not
        # already open at the published settlement boundary.
        for (timestamp, symbol), rates in funding_by_event.items():
            if timestamp != event_time or symbol not in positions:
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
            for rate in rates:
                position.funding_pnl -= direction * position.quantity * mark * rate

        # Preserve input order to make position-cap selection deterministic.
        for _, bar in sorted(current_bars):
            symbol = str(bar["symbol"])
            for signal in entries.get((event_time, symbol), []):
                if symbol in positions or len(positions) >= sizing.max_positions:
                    continue
                side = str(signal["side"])
                if side not in {"long", "short"}:
                    raise ValueError(f"unsupported side: {side}")
                action = "buy" if side == "long" else "sell"
                entry_price = _adverse_fill(
                    float(bar["open"]), action, execution.slippage_per_side
                )
                notional = equity * sizing.margin_fraction * sizing.leverage
                quantity = notional / entry_price
                positions[symbol] = _Position(
                    symbol=symbol,
                    side=side,
                    entry_time=event_time,
                    entry_price=entry_price,
                    quantity=quantity,
                    entry_fee=quantity * entry_price * execution.fee_per_side,
                    stop=float(signal["stop"]),
                    take_profit=float(signal["take_profit"]),
                )

        for _, bar in sorted(current_bars):
            symbol = str(bar["symbol"])
            position = positions.get(symbol)
            if position is None:
                last_price[symbol] = float(bar["close"])
                continue
            exit_event = _exit_event(position, bar)
            if exit_event is not None:
                reference, reason = exit_event
                action = "sell" if position.side == "long" else "buy"
                exit_price = _adverse_fill(
                    reference, action, execution.slippage_per_side
                )
                trade = Trade(
                    symbol=symbol,
                    side=position.side,
                    entry_time=position.entry_time,
                    exit_time=event_time,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    quantity=position.quantity,
                    entry_fee=position.entry_fee,
                    exit_fee=(
                        position.quantity * exit_price * execution.fee_per_side
                    ),
                    funding_pnl=position.funding_pnl,
                    exit_reason=reason,
                )
                trades.append(trade)
                equity += trade.net_pnl
                del positions[symbol]
            last_price[symbol] = float(bar["close"])

    return trades
