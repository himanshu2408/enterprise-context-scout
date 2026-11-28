"""Agent orchestration, DomainPack registry, and AdapterPool management."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_LAZY_EXPORTS: dict[str, str] = {
    "AdapterPool": "context_scout.agents.domain_registry",
    "AdapterSlot": "context_scout.agents.domain_registry",
    "DomainRegistry": "context_scout.agents.domain_registry",
    "DEFAULT_FAST_MODEL_ID": "context_scout.agents.gemini_executor",
    "FRONTIER_REASONING_MODEL_ID": "context_scout.agents.gemini_executor",
    "BudgetExceededError": "context_scout.agents.gemini_executor",
    "GeminiImplementationExecutor": "context_scout.agents.gemini_executor",
    "GitWorktreeSandbox": "context_scout.agents.gemini_executor",
    "AdapterRouteDecision": "context_scout.agents.llm_backend",
    "AdapterRouter": "context_scout.agents.llm_backend",
    "GPUOverloadError": "context_scout.agents.llm_backend",
    "GPUPriorityClass": "context_scout.agents.llm_backend",
    "LLMBackendSPI": "context_scout.agents.llm_backend",
    "RedisGPUQuotaManager": "context_scout.agents.llm_backend",
    "compute_adaptive_complexity_budget": "context_scout.agents.llm_backend",
    "ModelArchitectureAdapter": "context_scout.agents.open_weights_vllm_executor",
    "OpenWeightsVLLMExecutor": "context_scout.agents.open_weights_vllm_executor",
    "compute_prefix_cache_salt": "context_scout.agents.open_weights_vllm_executor",
    "ContextPackValidator": "context_scout.agents.preflight_gate",
    "PreFlightValidator": "context_scout.agents.preflight_gate",
    "create_insufficient_context_diagnostic": "context_scout.agents.preflight_gate",
    "DEFAULT_LOCAL_GEMMA_MODEL": "context_scout.agents.scout_workflow",
    "MAX_SAFE_KV_SIZE": "context_scout.agents.scout_workflow",
    "FederatedScoutCoordinator": "context_scout.agents.scout_workflow",
    "LocalGemmaClient": "context_scout.agents.scout_workflow",
    "ModelUnavailableError": "context_scout.agents.scout_workflow",
    "ModelUnavailableException": "context_scout.agents.scout_workflow",
    "ScoutWorkflow": "context_scout.agents.scout_workflow",
    "ScoutSessionManager": "context_scout.agents.session_compactor",
    "SessionPreflightResult": "context_scout.agents.session_compactor",
}

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


def __getattr__(name: str) -> Any:
    mod_path = _LAZY_EXPORTS.get(name)
    if mod_path is not None:
        mod = importlib.import_module(mod_path)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
