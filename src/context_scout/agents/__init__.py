"""Agent orchestration, DomainPack registry, and AdapterPool management."""

from context_scout.agents.domain_registry import AdapterPool, AdapterSlot, DomainRegistry
from context_scout.agents.gemini_executor import (
    DEFAULT_FAST_MODEL_ID,
    FRONTIER_REASONING_MODEL_ID,
    BudgetExceededError,
    GeminiImplementationExecutor,
    GitWorktreeSandbox,
)
from context_scout.agents.preflight_gate import (
    ContextPackValidator,
    PreFlightValidator,
    create_insufficient_context_diagnostic,
)
from context_scout.agents.scout_workflow import (
    DEFAULT_LOCAL_GEMMA_MODEL,
    MAX_SAFE_KV_SIZE,
    FederatedScoutCoordinator,
    LocalGemmaClient,
    ScoutWorkflow,
)

__all__ = [
    "DEFAULT_FAST_MODEL_ID",
    "DEFAULT_LOCAL_GEMMA_MODEL",
    "FRONTIER_REASONING_MODEL_ID",
    "MAX_SAFE_KV_SIZE",
    "AdapterPool",
    "AdapterSlot",
    "BudgetExceededError",
    "ContextPackValidator",
    "DomainRegistry",
    "FederatedScoutCoordinator",
    "GeminiImplementationExecutor",
    "GitWorktreeSandbox",
    "LocalGemmaClient",
    "PreFlightValidator",
    "ScoutWorkflow",
    "create_insufficient_context_diagnostic",
]

