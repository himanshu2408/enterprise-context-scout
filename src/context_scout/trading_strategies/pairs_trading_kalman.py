"""Pairs Trading Cointegration Engine with Kalman Filter Dynamic Hedge Ratio.

Implements online recursive state-space estimation of time-varying hedge ratios
between cointegrated asset pairs with Joseph-form covariance stabilization and
fractional Kelly position sizing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from context_scout.trading_strategies.kelly_risk_manager import KellyRiskManager


@dataclass(frozen=True)
class KalmanState:
    """Estimated state vector and covariance of dynamic linear hedge."""

    hedge_ratio_beta: float
    intercept_alpha: float
    beta_variance: float
    alpha_variance: float
    covariance: float
    innovation: float
    innovation_variance: float


@dataclass(frozen=True)
class PairsTradingSignal:
    """Real-time statistical arbitrage signal emitted for an asset pair."""

    symbol_y: str
    symbol_x: str
    current_price_y: float
    current_price_x: float
    dynamic_hedge_ratio: float
    spread: float
    spread_zscore: float
    target_weight_y: float
    target_weight_x: float
    signal_regime: str


class KalmanHedgeEstimator:
    """Recursive Kalman filter estimator for time-varying linear cointegration.

    State transition:
        [beta_t, alpha_t]^T = [beta_{t-1}, alpha_{t-1}]^T + eta_t,  eta_t ~ N(0, Q)

    Observation:
        y_t = [x_t, 1.0] * [beta_t, alpha_t]^T + epsilon_t,  epsilon_t ~ N(0, R)
    """

    DEFAULT_DELTA: Final[float] = 1e-4
    DEFAULT_R: Final[float] = 1e-3

    def __init__(
        self,
        process_noise_delta: float = DEFAULT_DELTA,
        measurement_noise_r: float = DEFAULT_R,
        initial_beta: float = 1.0,
        initial_alpha: float = 0.0,
    ) -> None:
        if process_noise_delta <= 0.0:
            raise ValueError(f"Process noise delta must be positive, got {process_noise_delta}")
        if measurement_noise_r <= 0.0:
            raise ValueError(f"Measurement noise R must be positive, got {measurement_noise_r}")

        self.delta = process_noise_delta
        self.r = measurement_noise_r

        # State vector theta = [beta, alpha]^T
        self.theta = np.array([initial_beta, initial_alpha], dtype=np.float64)

        # State covariance P
        self.p = np.eye(2, dtype=np.float64) * 1.0

        # Process noise covariance Q = delta / (1 - delta) * I
        self.q = (self.delta / (1.0 - self.delta)) * np.eye(2, dtype=np.float64)

    def update(self, price_y: float, price_x: float) -> KalmanState:
        """Executes one predict-update cycle given concurrent asset observations."""
        # Observation vector H_t = [x_t, 1.0]
        h = np.array([price_x, 1.0], dtype=np.float64)

        # 1. State prediction: theta_{t|t-1} = theta_{t-1}
        theta_pred = self.theta

        # 2. Covariance prediction: P_{t|t-1} = P_{t-1} + Q
        p_pred = self.p + self.q

        # 3. Innovation: y_t - H * theta_{t|t-1}
        y_pred = float(np.dot(h, theta_pred))
        innovation = price_y - y_pred

        # 4. Innovation variance: F_t = H * P_{t|t-1} * H^T + R
        f_val = float(np.dot(h, np.dot(p_pred, h)) + self.r)
        f_val = max(f_val, 1e-8)

        # 5. Kalman gain: K_t = P_{t|t-1} * H^T / F_t
        k_gain = np.dot(p_pred, h) / f_val

        # 6. State update: theta_{t|t} = theta_{t|t-1} + K_t * innovation
        self.theta = theta_pred + k_gain * innovation

        # 7. Covariance update using Joseph stabilized form:
        # P = (I - K*H) * P_pred * (I - K*H)^T + K * R * K^T
        eye_2 = np.eye(2, dtype=np.float64)
        i_kh = eye_2 - np.outer(k_gain, h)
        p_updated = np.dot(np.dot(i_kh, p_pred), i_kh.T) + self.r * np.outer(k_gain, k_gain)

        # Enforce exact symmetry
        self.p = 0.5 * (p_updated + p_updated.T)

        return KalmanState(
            hedge_ratio_beta=round(float(self.theta[0]), 6),
            intercept_alpha=round(float(self.theta[1]), 6),
            beta_variance=round(float(self.p[0, 0]), 8),
            alpha_variance=round(float(self.p[1, 1]), 8),
            covariance=round(float(self.p[0, 1]), 8),
            innovation=round(float(innovation), 6),
            innovation_variance=round(float(f_val), 8),
        )


class PairsSpreadSignal:
    """Monitors dynamic spread between cointegrated pairs and issues trading signals."""

    ZSCORE_ENTRY_THRESHOLD: Final[float] = 2.0
    ZSCORE_EXIT_THRESHOLD: Final[float] = 0.5

    def __init__(
        self,
        symbol_y: str,
        symbol_x: str,
        estimator: KalmanHedgeEstimator | None = None,
        risk_manager: KellyRiskManager | None = None,
        rolling_window: int = 60,
    ) -> None:
        self.symbol_y = symbol_y
        self.symbol_x = symbol_x
        self.estimator = estimator or KalmanHedgeEstimator()
        self.risk_manager = risk_manager or KellyRiskManager()
        self.rolling_window = rolling_window
        self._spread_history: list[float] = []

    def compute_spread(self, price_y: float, price_x: float) -> float:
        """Calculates instantaneous spread under active Kalman hedge ratio."""
        beta = float(self.estimator.theta[0])
        alpha = float(self.estimator.theta[1])
        return price_y - beta * price_x - alpha

    def update_and_signal(
        self,
        price_y: float,
        price_x: float,
    ) -> PairsTradingSignal:
        """Ingests asset ticks, updates Kalman estimator, and calculates pair weights."""
        state = self.estimator.update(price_y, price_x)
        spread = price_y - state.hedge_ratio_beta * price_x - state.intercept_alpha

        self._spread_history.append(spread)
        if len(self._spread_history) > self.rolling_window * 3:
            self._spread_history = self._spread_history[-self.rolling_window :]

        window_slice = self._spread_history[-self.rolling_window :]
        mean_s = sum(window_slice) / float(len(window_slice))
        var_s = sum((s - mean_s) ** 2 for s in window_slice) / float(max(1, len(window_slice) - 1))
        std_s = math.sqrt(var_s) if var_s > 1e-12 else 1.0

        z_score = (spread - mean_s) / std_s

        # Signal determination
        if z_score > self.ZSCORE_ENTRY_THRESHOLD:
            # Spread is too wide: Short Y, Long X
            regime = "SHORT_SPREAD"
            direction_y = -1.0
            direction_x = state.hedge_ratio_beta
        elif z_score < -self.ZSCORE_ENTRY_THRESHOLD:
            # Spread is too narrow: Long Y, Short X
            regime = "LONG_SPREAD"
            direction_y = 1.0
            direction_x = -state.hedge_ratio_beta
        elif abs(z_score) < self.ZSCORE_EXIT_THRESHOLD:
            regime = "NEUTRAL"
            direction_y = 0.0
            direction_x = 0.0
        else:
            regime = "HOLD"
            direction_y = 0.0
            direction_x = 0.0

        # Fractional Kelly sizing projection
        base_alloc = self.risk_manager.max_single_position * self.risk_manager.fractional_multiplier
        target_w_y = direction_y * base_alloc
        target_w_x = direction_x * base_alloc

        return PairsTradingSignal(
            symbol_y=self.symbol_y,
            symbol_x=self.symbol_x,
            current_price_y=round(price_y, 4),
            current_price_x=round(price_x, 4),
            dynamic_hedge_ratio=state.hedge_ratio_beta,
            spread=round(spread, 6),
            spread_zscore=round(z_score, 4),
            target_weight_y=round(target_w_y, 4),
            target_weight_x=round(target_w_x, 4),
            signal_regime=regime,
        )
