"""Limit Order Book Order Flow Imbalance (OFI) Calculation Engine.

Implements the Cont, Kukanov, and Stoikov (2014) formulation of Order Flow Imbalance
across multi-level depth book updates with rolling z-score normalization and
conditional volatility scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

from context_scout.trading_strategies.volatility_garch import GARCH11Model


@dataclass(frozen=True)
class LOBLevel:
    """Individual price level in a limit order book."""

    price: float
    size: float

    def __post_init__(self) -> None:
        if self.price <= 0.0:
            raise ValueError(f"Price must be positive, got {self.price}")
        if self.size < 0.0:
            raise ValueError(f"Size cannot be negative, got {self.size}")


@dataclass
class OrderBookSnapshot:
    """Multi-level limit order book state at discrete time t."""

    timestamp_ms: int
    symbol: str
    bids: list[LOBLevel] = field(default_factory=list)
    asks: list[LOBLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> LOBLevel:
        if not self.bids:
            raise ValueError(f"No bid levels available for {self.symbol}")
        return self.bids[0]

    @property
    def best_ask(self) -> LOBLevel:
        if not self.asks:
            raise ValueError(f"No ask levels available for {self.symbol}")
        return self.asks[0]

    @property
    def mid_price(self) -> float:
        return (self.best_bid.price + self.best_ask.price) / 2.0

    @property
    def spread(self) -> float:
        return self.best_ask.price - self.best_bid.price


@dataclass(frozen=True)
class OFISignalUpdate:
    """Emitted OFI step calculation and normalized trading signal."""

    timestamp_ms: int
    symbol: str
    level_1_ofi: float
    multi_level_ofi: float
    rolling_zscore: float
    volatility_adjusted_signal: float
    current_mid_price: float


class OrderFlowImbalanceSignal:
    """Computes multi-level Order Flow Imbalance across sequential book snapshots.

    Cont-Kukanov-Stoikov Level 1 formulation:
        Delta W_t = I_{p_t^b >= p_{t-1}^b} * q_t^b - I_{p_t^b <= p_{t-1}^b} * q_{t-1}^b
                  - I_{p_t^a <= p_{t-1}^a} * q_t^a + I_{p_t^a >= p_{t-1}^a} * q_{t-1}^a
    """

    DEFAULT_LEVEL_DECAY: Final[float] = 0.5
    DEFAULT_ROLLING_WINDOW: Final[int] = 50

    def __init__(
        self,
        symbol: str,
        depth_levels: int = 5,
        level_decay: float = DEFAULT_LEVEL_DECAY,
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
        garch_model: GARCH11Model | None = None,
    ) -> None:
        if depth_levels < 1:
            raise ValueError(f"Depth levels must be >= 1, got {depth_levels}")
        if level_decay < 0.0 or level_decay > 1.0:
            raise ValueError(f"Level decay must be in [0, 1], got {level_decay}")
        if rolling_window < 5:
            raise ValueError(f"Rolling window must be >= 5, got {rolling_window}")

        self.symbol = symbol
        self.depth_levels = depth_levels
        self.level_decay = level_decay
        self.rolling_window = rolling_window
        self.garch_model = garch_model or GARCH11Model()

        self._prev_snapshot: OrderBookSnapshot | None = None
        self._history_ofi: list[float] = []

    def compute_single_level_ofi(
        self,
        prev_bid: LOBLevel,
        curr_bid: LOBLevel,
        prev_ask: LOBLevel,
        curr_ask: LOBLevel,
    ) -> float:
        """Calculates Cont-Kukanov-Stoikov OFI for a single price tier."""
        # Bid side contribution
        if curr_bid.price > prev_bid.price:
            delta_bid = curr_bid.size
        elif curr_bid.price == prev_bid.price:
            delta_bid = curr_bid.size - prev_bid.size
        else:
            delta_bid = -prev_bid.size

        # Ask side contribution
        if curr_ask.price < prev_ask.price:
            delta_ask = curr_ask.size
        elif curr_ask.price == prev_ask.price:
            delta_ask = curr_ask.size - prev_ask.size
        else:
            delta_ask = -prev_ask.size

        return delta_bid - delta_ask

    def process_snapshot(self, snapshot: OrderBookSnapshot) -> OFISignalUpdate:
        """Processes an incoming order book update and generates normalized OFI signal."""
        if snapshot.symbol != self.symbol:
            raise ValueError(
                f"Snapshot symbol '{snapshot.symbol}' does not match engine symbol '{self.symbol}'"
            )

        if self._prev_snapshot is None:
            self._prev_snapshot = snapshot
            return OFISignalUpdate(
                timestamp_ms=snapshot.timestamp_ms,
                symbol=self.symbol,
                level_1_ofi=0.0,
                multi_level_ofi=0.0,
                rolling_zscore=0.0,
                volatility_adjusted_signal=0.0,
                current_mid_price=snapshot.mid_price,
            )

        prev = self._prev_snapshot
        max_levels = min(
            self.depth_levels,
            len(curr_b := snapshot.bids),
            len(prev_b := prev.bids),
            len(curr_a := snapshot.asks),
            len(prev_a := prev.asks),
        )

        level_1_val = 0.0
        multi_level_val = 0.0

        for k in range(max_levels):
            weight = math.exp(-self.level_decay * k)
            ofi_k = self.compute_single_level_ofi(
                prev_bid=prev_b[k],
                curr_bid=curr_b[k],
                prev_ask=prev_a[k],
                curr_ask=curr_a[k],
            )
            if k == 0:
                level_1_val = ofi_k
            multi_level_val += weight * ofi_k

        self._history_ofi.append(multi_level_val)
        if len(self._history_ofi) > self.rolling_window * 3:
            self._history_ofi = self._history_ofi[-self.rolling_window :]

        # Rolling z-score normalization
        window_vals = self._history_ofi[-self.rolling_window :]
        mean_val = sum(window_vals) / float(len(window_vals))
        deg_free = float(max(1, len(window_vals) - 1))
        var_val = sum((x - mean_val) ** 2 for x in window_vals) / deg_free
        std_val = math.sqrt(var_val) if var_val > 1e-12 else 1.0

        z_score = (multi_level_val - mean_val) / std_val

        # Volatility scaling via GARCH conditional variance
        cond_vol = math.sqrt(self.garch_model.unconditional_variance)
        vol_adj_signal = z_score / max(cond_vol, 1e-4)

        self._prev_snapshot = snapshot

        return OFISignalUpdate(
            timestamp_ms=snapshot.timestamp_ms,
            symbol=self.symbol,
            level_1_ofi=round(level_1_val, 4),
            multi_level_ofi=round(multi_level_val, 4),
            rolling_zscore=round(z_score, 4),
            volatility_adjusted_signal=round(vol_adj_signal, 4),
            current_mid_price=round(snapshot.mid_price, 4),
        )
