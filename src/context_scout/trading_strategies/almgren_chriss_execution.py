"""Almgren-Chriss Optimal Trade Execution Engine with Market Impact.

Implements the classical Almgren and Chriss (2000) optimal liquidation trajectory
incorporating permanent price impact, temporary price impact, asset volatility,
and risk aversion parameterization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from context_scout.trading_strategies.volatility_garch import VolatilityForecastResult


@dataclass(frozen=True)
class MarketImpactParameters:
    """Microstructure friction parameters for optimal execution scheduling."""

    temporary_impact_eta: float
    permanent_impact_gamma: float
    annualized_volatility_sigma: float
    risk_aversion_lambda: float

    def __post_init__(self) -> None:
        if self.temporary_impact_eta <= 0.0:
            raise ValueError(
                f"Temporary impact eta must be positive, got {self.temporary_impact_eta}"
            )
        if self.permanent_impact_gamma < 0.0:
            raise ValueError(
                f"Permanent impact gamma cannot be negative, got {self.permanent_impact_gamma}"
            )
        if self.annualized_volatility_sigma <= 0.0:
            raise ValueError(
                f"Volatility sigma must be positive, got {self.annualized_volatility_sigma}"
            )


@dataclass(frozen=True)
class ExecutionStep:
    """Individual trade step within optimal trajectory."""

    step_index: int
    elapsed_time_hours: float
    holdings_remaining: float
    shares_to_trade: float
    trading_rate_shares_per_hour: float


@dataclass(frozen=True)
class ExecutionTrajectory:
    """Optimal multi-period trade trajectory and expected friction costs."""

    total_shares: float
    time_horizon_hours: float
    num_steps: int
    steps: list[ExecutionStep]
    urgency_parameter_kappa: float
    expected_execution_cost: float
    variance_of_cost: float
    utility_cost: float
    half_life_hours: float


class AlmgrenChrissExecutionOptimizer:
    """Optimizes discrete-time trade liquidation trajectories under Almgren-Chriss (2000).

    Trajectory:
        x_j = sinh(kappa * (T - t_j)) / sinh(kappa * T) * X_0

    Urgency parameter:
        kappa_tilde = acosh( (tau^2 * lambda * sigma^2 / eta + 2) / 2 ) / tau
    """

    HOURS_PER_TRADING_DAY: Final[float] = 6.5
    DAYS_PER_YEAR: Final[float] = 252.0

    def __init__(
        self,
        parameters: MarketImpactParameters | None = None,
    ) -> None:
        self.params = parameters or MarketImpactParameters(
            temporary_impact_eta=2.5e-6,
            permanent_impact_gamma=1.0e-7,
            annualized_volatility_sigma=0.25,
            risk_aversion_lambda=1e-5,
        )

    def update_volatility_from_garch(self, forecast: VolatilityForecastResult) -> None:
        """Dynamically recalibrates asset volatility parameter using GARCH(1,1) forecast."""
        self.params = MarketImpactParameters(
            temporary_impact_eta=self.params.temporary_impact_eta,
            permanent_impact_gamma=self.params.permanent_impact_gamma,
            annualized_volatility_sigma=forecast.current_annualized_volatility,
            risk_aversion_lambda=self.params.risk_aversion_lambda,
        )

    def compute_optimal_trajectory(
        self,
        total_shares: float,
        time_horizon_hours: float = 6.5,
        num_steps: int = 13,
    ) -> ExecutionTrajectory:
        """Calculates discrete trajectory minimizing expected cost + risk penalty."""
        if total_shares <= 0.0:
            raise ValueError(f"Total shares must be positive, got {total_shares}")
        if time_horizon_hours <= 0.0:
            raise ValueError(f"Time horizon must be positive, got {time_horizon_hours}")
        if num_steps < 2:
            raise ValueError(f"Number of steps must be >= 2, got {num_steps}")

        tau = time_horizon_hours / float(num_steps)
        total_time = time_horizon_hours

        # Convert annualized volatility to hourly volatility
        trading_hours_per_year = self.DAYS_PER_YEAR * self.HOURS_PER_TRADING_DAY
        hourly_sigma = self.params.annualized_volatility_sigma / math.sqrt(trading_hours_per_year)

        # Dimensionless risk-adjusted impact parameter
        lambda_val = self.params.risk_aversion_lambda
        eta = self.params.temporary_impact_eta
        gamma = self.params.permanent_impact_gamma

        steps: list[ExecutionStep] = []
        holdings = [0.0] * (num_steps + 1)
        trades = [0.0] * num_steps

        if lambda_val <= 0.0:
            kappa = 0.0
            for j in range(num_steps + 1):
                t_j = j * tau
                holdings[j] = total_shares * (1.0 - (t_j / total_time))
        else:
            tilde_param = (tau**2 * lambda_val * (hourly_sigma**2)) / eta
            cosh_arg = (tilde_param + 2.0) / 2.0

            if cosh_arg >= 1.0:
                kappa = math.acosh(cosh_arg) / tau
            else:
                kappa = math.sqrt(max(0.0, (lambda_val * (hourly_sigma**2)) / eta))

            if kappa <= 1e-9:
                kappa = 0.0
                for j in range(num_steps + 1):
                    t_j = j * tau
                    holdings[j] = total_shares * (1.0 - (t_j / total_time))
            else:
                sinh_kappa_t = math.sinh(kappa * total_time)
                if sinh_kappa_t == 0.0:
                    sinh_kappa_t = 1e-12

                for j in range(num_steps + 1):
                    t_j = j * tau
                    num = total_shares * math.sinh(kappa * (total_time - t_j))
                    holdings[j] = num / sinh_kappa_t

        for j in range(num_steps):
            t_j = j * tau
            shares_j = holdings[j] - holdings[j + 1]
            trades[j] = shares_j
            rate = shares_j / tau
            steps.append(
                ExecutionStep(
                    step_index=j + 1,
                    elapsed_time_hours=round(t_j + tau, 3),
                    holdings_remaining=round(holdings[j + 1], 2),
                    shares_to_trade=round(shares_j, 2),
                    trading_rate_shares_per_hour=round(rate, 2),
                )
            )

        # Expected execution cost: 0.5 * gamma * X^2 + eta * sum(n_j^2 / tau)
        perm_cost = 0.5 * gamma * (total_shares**2)
        temp_cost = eta * sum((n_j**2) / tau for n_j in trades)
        expected_cost = perm_cost + temp_cost

        # Cost variance: (sigma_hourly)^2 * sum(tau * x_j^2)
        cost_variance = (hourly_sigma**2) * sum(tau * (holdings[j] ** 2) for j in range(num_steps))

        utility_cost = expected_cost + lambda_val * cost_variance
        half_life = math.log(2.0) / kappa if kappa > 0.0 else float("inf")

        return ExecutionTrajectory(
            total_shares=round(total_shares, 2),
            time_horizon_hours=round(time_horizon_hours, 2),
            num_steps=num_steps,
            steps=steps,
            urgency_parameter_kappa=round(kappa, 6),
            expected_execution_cost=round(expected_cost, 4),
            variance_of_cost=round(cost_variance, 4),
            utility_cost=round(utility_cost, 4),
            half_life_hours=round(half_life, 3),
        )
