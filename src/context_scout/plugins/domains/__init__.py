"""Domain Plugins (F18)."""

from context_scout.plugins.domains.data_engineering import (
    DOMAIN_ID_DATA_ENG,
    DataEngineeringDomainPlugin,
    DataEngineeringGateResult,
    DataEngineeringRealismGate,
)
from context_scout.plugins.domains.finance import (
    FinanceDomainPlugin,
    FinancialEvaluationResult,
    ScepticismEvaluationResult,
    calculate_effective_friction_floor,
)

__all__ = [
    "DOMAIN_ID_DATA_ENG",
    "DataEngineeringDomainPlugin",
    "DataEngineeringGateResult",
    "DataEngineeringRealismGate",
    "FinanceDomainPlugin",
    "FinancialEvaluationResult",
    "ScepticismEvaluationResult",
    "calculate_effective_friction_floor",
]
