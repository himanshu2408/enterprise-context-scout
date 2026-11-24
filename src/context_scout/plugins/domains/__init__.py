"""Domain Plugins (F18)."""

from context_scout.plugins.domains.data_engineering import (
    DOMAIN_ID_DATA_ENG,
    DataEngineeringDomainPlugin,
    DataEngineeringGateResult,
    DataEngineeringRealismGate,
)
from context_scout.plugins.domains.enterprise_core import (
    DOMAIN_ID_CODE_ARCH,
    DOMAIN_ID_ENTERPRISE_DOCS,
    DOMAIN_ID_FEDERATED_ALL,
    EnterpriseCoreDomainPlugin,
)

__all__ = [
    "DOMAIN_ID_CODE_ARCH",
    "DOMAIN_ID_DATA_ENG",
    "DOMAIN_ID_ENTERPRISE_DOCS",
    "DOMAIN_ID_FEDERATED_ALL",
    "DataEngineeringDomainPlugin",
    "DataEngineeringGateResult",
    "DataEngineeringRealismGate",
    "EnterpriseCoreDomainPlugin",
]
