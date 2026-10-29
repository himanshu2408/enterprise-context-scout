"""Multi-file algorithmic trading strategies package.

Provides production-grade quantitative models for order flow imbalance,
cointegrated pairs trading with Kalman filters, Almgren-Chriss execution,
GARCH(1,1) volatility forecasting, and fractional Kelly risk management.
"""

from context_scout.trading_strategies.almgren_chriss_execution import (
    AlmgrenChrissExecutionOptimizer,
    ExecutionStep,
    ExecutionTrajectory,
    MarketImpactParameters,
)
from context_scout.trading_strategies.kelly_risk_manager import (
    KellyRiskManager,
    PortfolioAllocationResult,
)
from context_scout.trading_strategies.order_flow_imbalance import (
    LOBLevel,
    OFISignalUpdate,
    OrderBookSnapshot,
    OrderFlowImbalanceSignal,
)
from context_scout.trading_strategies.pairs_trading_kalman import (
    KalmanHedgeEstimator,
    KalmanState,
    PairsSpreadSignal,
    PairsTradingSignal,
)
from context_scout.trading_strategies.volatility_garch import (
    GARCH11Model,
    VolatilityForecastResult,
)

__all__ = [
    "AlmgrenChrissExecutionOptimizer",
    "ExecutionStep",
    "ExecutionTrajectory",
    "GARCH11Model",
    "KalmanHedgeEstimator",
    "KalmanState",
    "KellyRiskManager",
    "LOBLevel",
    "MarketImpactParameters",
    "OFISignalUpdate",
    "OrderBookSnapshot",
    "OrderFlowImbalanceSignal",
    "PairsSpreadSignal",
    "PairsTradingSignal",
    "PortfolioAllocationResult",
    "VolatilityForecastResult",
]
