"""Multi-Asset Fractional Kelly Criterion with Ledoit-Wolf Covariance Shrinkage.

Implements optimal growth portfolio allocation under continuous log-utility with
Ledoit-Wolf linear shrinkage for sample covariance matrix stabilization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np


@dataclass(frozen=True)
class PortfolioAllocationResult:
    """Optimal portfolio weights and growth characteristics emitted by Kelly optimizer."""

    raw_unconstrained_weights: list[float]
    bounded_fractional_weights: list[float]
    fractional_multiplier: float
    gross_leverage: float
    net_exposure: float
    expected_growth_rate: float
    expected_portfolio_volatility: float
    shrinkage_intensity: float


class KellyRiskManager:
    """Multi-asset portfolio sizer utilizing the fractional Kelly criterion.

    Computes:
        f* = c * Sigma^{-1} (mu - r_f * 1)
    where c is the fractional Kelly parameter (e.g. 0.5 for half-Kelly).
    """

    MIN_EIGENVALUE_FLOOR: Final[float] = 1e-6
    DEFAULT_FRACTIONAL_MULTIPLIER: Final[float] = 0.5

    def __init__(
        self,
        fractional_multiplier: float = DEFAULT_FRACTIONAL_MULTIPLIER,
        max_gross_leverage: float = 2.0,
        max_single_position: float = 0.40,
    ) -> None:
        if fractional_multiplier <= 0.0 or fractional_multiplier > 1.0:
            raise ValueError(
                f"Fractional multiplier must be in (0, 1], got {fractional_multiplier}"
            )
        if max_gross_leverage <= 0.0:
            raise ValueError(f"Max leverage must be positive, got {max_gross_leverage}")
        if max_single_position <= 0.0 or max_single_position > 1.0:
            raise ValueError(
                f"Max single position must be in (0, 1], got {max_single_position}"
            )

        self.fractional_multiplier = fractional_multiplier
        self.max_gross_leverage = max_gross_leverage
        self.max_single_position = max_single_position

    def ledoit_wolf_shrinkage(
        self,
        returns_history: list[list[float]],
    ) -> tuple[np.ndarray, float]:
        """Performs linear covariance shrinkage towards constant correlation target.

        Sigma_shrunk = (1 - delta) * S + delta * F
        """
        arr = np.array(returns_history, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D returns array (T x N), got shape {arr.shape}")

        t_samples, n_assets = arr.shape
        if t_samples < 4 or n_assets < 1:
            raise ValueError(
                f"Returns matrix requires T >= 4 and N >= 1, got shape ({t_samples}, {n_assets})"
            )

        # De-mean returns
        mean_vec = np.mean(arr, axis=0)
        x_centered = arr - mean_vec

        # Sample covariance matrix S
        sample_cov = np.dot(x_centered.T, x_centered) / float(t_samples - 1)
        var_diag = np.diag(sample_cov)
        std_diag = np.sqrt(np.maximum(var_diag, self.MIN_EIGENVALUE_FLOOR))

        if n_assets == 1:
            return sample_cov, 0.0

        # Correlation matrix
        corr_matrix = sample_cov / np.outer(std_diag, std_diag)
        np.fill_diagonal(corr_matrix, 1.0)

        # Average correlation target r_bar
        upper_tri = np.triu(corr_matrix, k=1)
        k_pairs = n_assets * (n_assets - 1) / 2.0
        r_bar = float(np.sum(upper_tri) / k_pairs) if k_pairs > 0 else 0.0

        # Prior target F
        target_f = r_bar * np.outer(std_diag, std_diag)
        np.fill_diagonal(target_f, var_diag)

        # Asymptotic shrinkage intensity delta estimation
        z = np.zeros((t_samples, n_assets, n_assets), dtype=np.float64)
        for t in range(t_samples):
            row = x_centered[t : t + 1, :]
            z[t] = np.dot(row.T, row)

        phi_mat = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                phi_mat[i, j] = float(np.var(z[:, i, j]))

        pi_hat = float(np.sum(phi_mat))

        # Gamma calculation (Frobenius distance ||S - F||^2)
        gamma_hat = float(np.sum((sample_cov - target_f) ** 2))

        if gamma_hat <= 1e-12:
            delta = 0.0
        else:
            delta = max(0.0, min(1.0, (pi_hat / float(t_samples)) / gamma_hat))

        shrunk_cov = (1.0 - delta) * sample_cov + delta * target_f
        return shrunk_cov, round(delta, 4)

    def compute_optimal_weights(
        self,
        expected_returns: list[float],
        covariance_matrix: list[list[float]],
        risk_free_rate: float = 0.0,
    ) -> PortfolioAllocationResult:
        """Solves for unconstrained and bounded fractional Kelly portfolio weights."""
        n_assets = len(expected_returns)
        if n_assets == 0:
            raise ValueError("Expected returns vector cannot be empty")

        mu = np.array(expected_returns, dtype=np.float64)
        cov = np.array(covariance_matrix, dtype=np.float64)

        if cov.shape != (n_assets, n_assets):
            raise ValueError(
                f"Covariance shape {cov.shape} does not match return length {n_assets}"
            )

        # Eigenvalue regularization for numerical stability
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        eig_vals_clipped = np.maximum(eig_vals, self.MIN_EIGENVALUE_FLOOR)
        regularized_cov = np.dot(eig_vecs * eig_vals_clipped, eig_vecs.T)

        excess_returns = mu - risk_free_rate

        # Unconstrained Kelly allocation: f* = Sigma^{-1} * excess_returns
        try:
            unconstrained_w = np.linalg.solve(regularized_cov, excess_returns)
        except np.linalg.LinAlgError:
            unconstrained_w = np.linalg.pinv(regularized_cov) @ excess_returns

        # Scale by fractional multiplier c
        fractional_w = unconstrained_w * self.fractional_multiplier

        # Apply single-asset position caps
        clipped_w = np.clip(
            fractional_w,
            -self.max_single_position,
            self.max_single_position,
        )

        # Apply gross leverage budget constraint
        gross_leverage = float(np.sum(np.abs(clipped_w)))
        if gross_leverage > self.max_gross_leverage and gross_leverage > 0.0:
            scaling = self.max_gross_leverage / gross_leverage
            final_w = clipped_w * scaling
        else:
            final_w = clipped_w

        net_exposure = float(np.sum(final_w))
        final_gross = float(np.sum(np.abs(final_w)))

        # Growth rate: g(w) = r_f + w^T(mu - r_f) - 0.5 * w^T * Sigma * w
        port_var = float(final_w.T @ regularized_cov @ final_w)
        port_vol = math.sqrt(max(port_var, 0.0))
        expected_growth = risk_free_rate + float(final_w @ excess_returns) - 0.5 * port_var

        return PortfolioAllocationResult(
            raw_unconstrained_weights=[round(float(x), 6) for x in unconstrained_w],
            bounded_fractional_weights=[round(float(x), 6) for x in final_w],
            fractional_multiplier=self.fractional_multiplier,
            gross_leverage=round(final_gross, 6),
            net_exposure=round(net_exposure, 6),
            expected_growth_rate=round(expected_growth, 6),
            expected_portfolio_volatility=round(port_vol, 6),
            shrinkage_intensity=0.0,
        )
