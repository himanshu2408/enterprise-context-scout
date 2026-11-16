"""Backend-driven workspace configuration endpoint for dynamic UI & client discovery."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.api.dependencies import (
    DEFAULT_ROLE_SCOPES,
    get_domain_registry,
    get_principal_context,
    get_repo_root,
)
from context_scout.api.routes.domains import build_domain_details
from context_scout.api.routes.health import probe_runtime_environment
from context_scout.api.schemas import (
    PipelineStageSpec,
    WorkspaceConfigResponse,
    WorkspaceRolePreset,
)
from context_scout.config import ScoutSettings
from context_scout.schemas.contracts import PrincipalContext

router = APIRouter(tags=["Workspace"])

_PIPELINE_STAGE_SPECS: tuple[PipelineStageSpec, ...] = (
    PipelineStageSpec(
        stage_id=1,
        number="01",
        title="Auth, Guardrail & Query Plan",
        summary=(
            "Enforce principal RBAC scopes, ShieldGemma prompt-injection screen, "
            "and structured QueryPlan decomposition."
        ),
        what_happens=(
            "Binds PrincipalContext to PostgreSQL RLS scopes, screens untrusted query text, "
            "decomposes multi-part concepts into facets, and selects candidate DomainPacks."
        ),
        gate_condition="Fail closed on security violation or empty authorized scopes ($0).",
    ),
    PipelineStageSpec(
        stage_id=2,
        number="02",
        title="Multi-Domain Hybrid Retrieval & AST Closure",
        summary=(
            "Run calibrated PG17 RRF (pgvector + BM25 + pg_trgm) "
            "and 1-2 hop AST dependency expansion."
        ),
        what_happens=(
            "Probes authorized DomainPacks with cosine + BM25 + trigram floor filtering, "
            "drops below-floor noise, and expands AST call/type edges for code hits."
        ),
        gate_condition="Reject hits below relevance floor; rank domains by calibrated evidence.",
    ),
    PipelineStageSpec(
        stage_id=3,
        number="03",
        title="Local Gemma-4 Synthesis & Facet Grounding",
        summary=(
            "Synthesize full-chunk evidence with base Gemma-4 or domain QLoRA "
            "and verify facet grounding."
        ),
        what_happens=(
            "Feeds full retrieved chunks (not snippets) to local Gemma-4 or extractive "
            "synthesizer, scoring cohesive per-facet grounding and sufficiency_verdict."
        ),
        gate_condition="Verify sufficiency_verdict and cohesive same-chunk facet grounding.",
    ),
    PipelineStageSpec(
        stage_id=4,
        number="04",
        title="Pre-Flight Sufficiency Gate",
        summary=(
            "Deterministic verification gate: $0 abstention on insufficient context "
            "or verified frontier handoff."
        ),
        what_happens=(
            "Audits symbol/file provenance, line hashes, calibrated retrieval confidence, "
            "sufficiency_verdict, and facet coverage before emitting context_pack_ready."
        ),
        gate_condition=(
            "Abstain with $0 paid_tokens_spent if insufficient or below relevance floor."
        ),
    ),
)


@router.get("/workspace-config", response_model=WorkspaceConfigResponse)
def get_workspace_config(
    registry: Annotated[DomainRegistry, Depends(get_domain_registry)],
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
) -> WorkspaceConfigResponse:
    """Returns dynamic DomainPacks, RLS counts, role presets, stages, and runtime status."""
    repo_root = get_repo_root()
    settings = ScoutSettings()
    domains = build_domain_details(registry=registry, principal=principal)
    probed = probe_runtime_environment(registry=registry, repo_root=repo_root, settings=settings)

    role_presets = [
        WorkspaceRolePreset(
            role=role,
            label=role.value.replace("_", " ").title(),
            default_scopes=list(scopes),
        )
        for role, scopes in DEFAULT_ROLE_SCOPES.items()
    ]

    return WorkspaceConfigResponse(
        domains=domains,
        federated_option={
            "domain_id": "federated_all",
            "display_name": "All Authorized Domains (Auto-Route)",
            "description": (
                "Dynamically probes all authorized DomainPacks in PostgreSQL 17, routes to the "
                "best domain(s), or abstains at $0 if evidence is below the relevance floor."
            ),
        },
        role_presets=role_presets,
        pipeline_stages=list(_PIPELINE_STAGE_SPECS),
        source_filters=["all", "code", "wiki", "jira"],
        hardware_profile=probed["hardware_profile"],
        database_connected=probed["database_connected"],
        database_status=probed["database_status"],
        runtime_mode=probed["runtime_mode"],
        timestamp=datetime.now(UTC).isoformat(),
    )
