"""Domain Plugins (F18)."""

from context_scout.plugins.domains.finance import (
    FinanceDomainPlugin,
    FinancialEvaluationResult,
    ScepticismEvaluationResult,
    calculate_effective_friction_floor,
    estimate_net_sharpe,
)

__all__ = [
    "FinanceDomainPlugin",
    "FinancialEvaluationResult",
    "ScepticismEvaluationResult",
    "calculate_effective_friction_floor",
    "estimate_net_sharpe",
]
