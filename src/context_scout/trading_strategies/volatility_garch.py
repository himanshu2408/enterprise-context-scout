"""GARCH(1,1) Volatility Forecaster with Quasi-Maximum Likelihood Calibration.

Implements the classical Bollerslev (1986) generalized autoregressive conditional
heteroskedasticity model with stationarity enforcement and rolling variance forecasting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class VolatilityForecastResult:
    """Quantitative forecast metrics emitted by GARCH(1,1)."""

    current_conditional_variance: float
    current_annualized_volatility: float
    long_run_unconditional_variance: float
    shock_half_life_periods: float
    forecast_variance_steps: list[float]
    forecast_annualized_volatility_steps: list[float]
    stationarity_sum: float


class GARCH11Model:
    """Univariate GARCH(1,1) conditional volatility forecaster.

    Variance evolution:
        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

    Stationarity requires:
        alpha + beta < 1.0
    """

    MIN_OMEGA: Final[float] = 1e-6
    ANNUALIZATION_FACTOR: Final[float] = 252.0

    def __init__(
        self,
        omega: float = 1e-5,
        alpha: float = 0.08,
        beta: float = 0.90,
    ) -> None:
        if omega <= 0.0:
            raise ValueError(f"Omega must be strictly positive, got {omega}")
        if alpha < 0.0 or beta < 0.0:
            raise ValueError(f"Parameters alpha ({alpha}) and beta ({beta}) must be non-negative")
        if (alpha + beta) >= 1.0:
            raise ValueError(
                f"Stationarity violated: alpha + beta must be < 1.0, got {alpha + beta:.4f}"
            )

        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self._fitted_variance_path: list[float] = []

    @property
    def unconditional_variance(self) -> float:
        """Computes the long-run stationary unconditional variance."""
        persistence = self.alpha + self.beta
        if persistence >= 1.0:
            return float("inf")
        return self.omega / (1.0 - persistence)

    @property
    def half_life_periods(self) -> float:
        """Computes the half-life of volatility shocks in periods."""
        persistence = self.alpha + self.beta
        if persistence <= 0.0 or persistence >= 1.0:
            return float("inf")
        return math.log(0.5) / math.log(persistence)

    def compute_variance_path(
        self,
        returns: list[float],
        initial_variance: float | None = None,
    ) -> list[float]:
        """Calculates conditional variance path sigma_t^2 given a return series."""
        if not returns:
            return []

        n = len(returns)
        variance_path = [0.0] * n
        init_var = initial_variance if initial_variance is not None else self.unconditional_variance
        variance_path[0] = max(init_var, self.MIN_OMEGA)

        for t in range(1, n):
            eps_prev_sq = returns[t - 1] ** 2
            var_t = self.omega + self.alpha * eps_prev_sq + self.beta * variance_path[t - 1]
            variance_path[t] = max(var_t, self.MIN_OMEGA)

        self._fitted_variance_path = list(variance_path)
        return variance_path

    def fit_quasi_mle(self, returns: list[float]) -> GARCH11Model:
        """Calibrates omega, alpha, and beta via quasi-maximum likelihood estimation."""
        if len(returns) < 30:
            raise ValueError(
                f"GARCH calibration requires at least 30 observations, got {len(returns)}"
            )

        ret_arr = np.array(returns, dtype=np.float64)
        sample_var = float(np.var(ret_arr))
        init_omega = max(sample_var * 0.05, self.MIN_OMEGA)
        init_params = np.array([init_omega, 0.08, 0.88], dtype=np.float64)

        def _neg_log_likelihood(params: np.ndarray) -> float:
            omega_p, alpha_p, beta_p = float(params[0]), float(params[1]), float(params[2])
            if omega_p <= 0.0 or alpha_p < 0.0 or beta_p < 0.0 or (alpha_p + beta_p) >= 0.9999:
                return 1e12

            n_samples = len(ret_arr)
            var_t = sample_var
            llh = 0.0

            for i in range(n_samples):
                ret_sq = ret_arr[i] ** 2
                prev_ret_sq = ret_arr[i - 1] ** 2 if i > 0 else sample_var
                var_t = omega_p + alpha_p * prev_ret_sq + beta_p * var_t
                var_t = max(var_t, 1e-8)
                llh += math.log(var_t) + (ret_sq / var_t)

            return 0.5 * llh

        bounds = [
            (self.MIN_OMEGA, sample_var * 0.5),
            (0.01, 0.40),
            (0.50, 0.98),
        ]
        res = minimize(
            _neg_log_likelihood,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
        )

        if res.success:
            opt_omega, opt_alpha, opt_beta = (
                float(res.x[0]),
                float(res.x[1]),
                float(res.x[2]),
            )
            if (opt_alpha + opt_beta) < 1.0 and opt_omega > 0.0:
                self.omega = opt_omega
                self.alpha = opt_alpha
                self.beta = opt_beta

        self.compute_variance_path(returns)
        return self

    def forecast(
        self,
        horizon_steps: int = 5,
        latest_return: float | None = None,
        latest_variance: float | None = None,
    ) -> VolatilityForecastResult:
        """Projects conditional variance over horizon_steps into the future."""
        if horizon_steps < 1:
            raise ValueError(f"Forecast horizon must be >= 1, got {horizon_steps}")

        if latest_variance is not None:
            curr_var = latest_variance
        elif self._fitted_variance_path:
            curr_var = self._fitted_variance_path[-1]
        else:
            curr_var = self.unconditional_variance

        if latest_return is not None:
            one_step_var = self.omega + self.alpha * (latest_return**2) + self.beta * curr_var
        else:
            one_step_var = curr_var

        persistence = self.alpha + self.beta
        v_long = self.unconditional_variance

        var_steps: list[float] = []
        vol_steps: list[float] = []

        for k in range(horizon_steps):
            if k == 0:
                v_k = max(one_step_var, self.MIN_OMEGA)
            else:
                v_k = v_long + (persistence ** k) * (one_step_var - v_long)
                v_k = max(v_k, self.MIN_OMEGA)

            ann_vol = math.sqrt(v_k * self.ANNUALIZATION_FACTOR)
            var_steps.append(round(v_k, 8))
            vol_steps.append(round(ann_vol, 6))

        curr_ann_vol = math.sqrt(curr_var * self.ANNUALIZATION_FACTOR)
        return VolatilityForecastResult(
            current_conditional_variance=round(curr_var, 8),
            current_annualized_volatility=round(curr_ann_vol, 6),
            long_run_unconditional_variance=round(v_long, 8),
            shock_half_life_periods=round(self.half_life_periods, 2),
            forecast_variance_steps=var_steps,
            forecast_annualized_volatility_steps=vol_steps,
            stationarity_sum=round(persistence, 6),
        )
