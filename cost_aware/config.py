from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionConfig:
    signal_lag_bars: int = 1
    fee_per_side: float = 0.0005
    slippage_per_side: float = 0.0002

    def __post_init__(self) -> None:
        if self.signal_lag_bars < 1:
            raise ValueError("signal_lag_bars must be at least one")
        if self.fee_per_side < 0 or self.slippage_per_side < 0:
            raise ValueError("execution costs cannot be negative")


@dataclass(frozen=True)
class SizingConfig:
    initial_equity: float
    margin_fraction: float
    leverage: float
    max_positions: int

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if not 0 < self.margin_fraction <= 1:
            raise ValueError("margin_fraction must be in (0, 1]")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least one")
