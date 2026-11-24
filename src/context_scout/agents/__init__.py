"""Agent orchestration, DomainPack registry, and AdapterPool management."""

from context_scout.agents.domain_registry import AdapterPool, AdapterSlot, DomainRegistry
from context_scout.agents.gemini_executor import (
    DEFAULT_FAST_MODEL_ID,
    FRONTIER_REASONING_MODEL_ID,
    BudgetExceededError,
    GeminiImplementationExecutor,
    GitWorktreeSandbox,
)
from context_scout.agents.llm_backend import (
    AdapterRouteDecision,
    AdapterRouter,
    GPUOverloadError,
    GPUPriorityClass,
    LLMBackendSPI,
    RedisGPUQuotaManager,
    compute_adaptive_complexity_budget,
)
from context_scout.agents.open_weights_vllm_executor import (
    ModelArchitectureAdapter,
    OpenWeightsVLLMExecutor,
    compute_prefix_cache_salt,
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
    ModelUnavailableError,
    ModelUnavailableException,
    ScoutWorkflow,
)
from context_scout.agents.session_compactor import ScoutSessionManager, SessionPreflightResult

__all__ = [
    "DEFAULT_FAST_MODEL_ID",
    "DEFAULT_LOCAL_GEMMA_MODEL",
    "FRONTIER_REASONING_MODEL_ID",
    "MAX_SAFE_KV_SIZE",
    "AdapterPool",
    "AdapterRouteDecision",
    "AdapterRouter",
    "AdapterSlot",
    "BudgetExceededError",
    "ContextPackValidator",
    "DomainRegistry",
    "FederatedScoutCoordinator",
    "GPUOverloadError",
    "GPUPriorityClass",
    "GeminiImplementationExecutor",
    "GitWorktreeSandbox",
    "LLMBackendSPI",
    "LocalGemmaClient",
    "ModelArchitectureAdapter",
    "ModelUnavailableError",
    "ModelUnavailableException",
    "OpenWeightsVLLMExecutor",
    "PreFlightValidator",
    "RedisGPUQuotaManager",
    "ScoutSessionManager",
    "ScoutWorkflow",
    "SessionPreflightResult",
    "compute_adaptive_complexity_budget",
    "compute_prefix_cache_salt",
    "create_insufficient_context_diagnostic",
]
