"""Statistical Evaluation Engine: Clopper-Pearson, Rule of Three, and Wilcoxon A/B Testing."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

DEFAULT_CONFIDENCE_LEVEL: Final[float] = 0.95
DEFAULT_ALPHA: Final[float] = 0.05


class ClopperPearsonBound(BaseModel):
    """Exact Clopper-Pearson binomial confidence bounds."""

    model_config = ConfigDict(frozen=True)

    k_failures: int = Field(ge=0)
    n_trials: int = Field(gt=0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    p_lower: float = Field(ge=0.0, le=1.0)
    p_upper: float = Field(ge=0.0, le=1.0)
    is_one_sided: bool

    @property
    def upper_percentage(self) -> float:
        return round(self.p_upper * 100.0, 4)

    @property
    def lower_percentage(self) -> float:
        return round(self.p_lower * 100.0, 4)


class WilcoxonABResult(BaseModel):
    """Paired Wilcoxon signed-rank test result comparing Arm A (no scout) vs Arm B (with scout)."""

    model_config = ConfigDict(frozen=True)

    n_tasks: int = Field(gt=0)
    statistic: float
    p_value: float
    median_arm_a_tokens: float
    median_arm_b_tokens: float
    median_savings_pct: float
    iqr_savings_pct: float
    cohens_d: float
    is_significant: bool
    target_reduction_met: bool
    target_reduction_p_value: float = 1.0
    is_target_reduction_significant: bool = False


def compute_rule_of_three(n_trials: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Computes the Rule of Three upper bound for 0 failures in n trials: p_U = -ln(alpha) / n."""
    if n_trials <= 0:
        raise ValueError(f"n_trials must be > 0, got {n_trials}")
    return float(-math.log(alpha) / n_trials)


def compute_clopper_pearson(
    k_failures: int,
    n_trials: int,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    one_sided: bool = True,
) -> ClopperPearsonBound:
    """Computes exact Clopper-Pearson confidence interval using Beta distribution quantiles."""
    if n_trials <= 0:
        raise ValueError(f"n_trials must be > 0, got {n_trials}")
    if k_failures < 0 or k_failures > n_trials:
        raise ValueError(f"k_failures must be in [0, {n_trials}], got {k_failures}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    alpha = 1.0 - confidence_level
    tail_alpha = alpha if one_sided else alpha / 2.0

    # Lower bound
    if k_failures == 0:
        p_lower = 0.0
    else:
        p_lower = float(stats.beta.ppf(tail_alpha, k_failures, n_trials - k_failures + 1))

    # Upper bound
    if k_failures == n_trials:
        p_upper = 1.0
    else:
        p_upper = float(stats.beta.ppf(1.0 - tail_alpha, k_failures + 1, n_trials - k_failures))

    return ClopperPearsonBound(
        k_failures=k_failures,
        n_trials=n_trials,
        confidence_level=confidence_level,
        p_lower=p_lower,
        p_upper=p_upper,
        is_one_sided=one_sided,
    )


def compute_token_ab_significance(
    arm_a_tokens: list[int],
    arm_b_tokens: list[int],
    alpha: float = 0.01,
    target_reduction_pct: float = 80.0,
) -> WilcoxonABResult:
    """Computes Wilcoxon signed-rank test and Cohen's d effect size for paired token consumption."""
    if len(arm_a_tokens) != len(arm_b_tokens) or not arm_a_tokens:
        raise ValueError("Arms must contain equal non-empty samples")

    n = len(arm_a_tokens)
    a_arr = np.array(arm_a_tokens, dtype=float)
    b_arr = np.array(arm_b_tokens, dtype=float)

    diffs = a_arr - b_arr
    savings_pcts = (diffs / a_arr) * 100.0

    median_a = float(np.median(a_arr))
    median_b = float(np.median(b_arr))
    median_savings = float(np.median(savings_pcts))
    q75, q25 = np.percentile(savings_pcts, [75, 25])
    iqr_savings = float(q75 - q25)

    # Cohen's d for paired samples: mean(diff) / std(diff)
    diff_std = float(np.std(diffs, ddof=1)) if n > 1 else 1.0
    cohens_d = float(np.mean(diffs) / diff_std) if diff_std > 0 else 0.0

    # Wilcoxon signed-rank test
    if np.all(diffs == 0):
        stat = 0.0
        p_val = 1.0
    else:
        wilc = stats.wilcoxon(a_arr, b_arr, alternative="greater")
        stat = float(wilc.statistic)
        p_val = float(wilc.pvalue)

    # One-sided test that token consumption reduction exceeds target_reduction_pct
    threshold_fraction = 1.0 - (target_reduction_pct / 100.0)
    target_excess = (a_arr * threshold_fraction) - b_arr
    if np.all(np.isclose(target_excess, 0.0, atol=1e-9)):
        target_p_val = 1.0
    else:
        target_wilc = stats.wilcoxon(target_excess, alternative="greater")
        target_p_val = float(target_wilc.pvalue)

    return WilcoxonABResult(
        n_tasks=n,
        statistic=round(stat, 4),
        p_value=p_val,
        median_arm_a_tokens=round(median_a, 1),
        median_arm_b_tokens=round(median_b, 1),
        median_savings_pct=round(median_savings, 2),
        iqr_savings_pct=round(iqr_savings, 2),
        cohens_d=round(cohens_d, 4),
        is_significant=bool(p_val < alpha),
        target_reduction_met=bool(median_savings >= target_reduction_pct),
        target_reduction_p_value=target_p_val,
        is_target_reduction_significant=bool(target_p_val < alpha),
    )
