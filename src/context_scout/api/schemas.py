"""Pydantic v2 API request, response, and SSE event schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from context_scout.schemas.contracts import (
    DocCitationSpan,
    DomainPackManifest,
    ExecutionTelemetryRecord,
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    MissingContextRequest,
    PreFlightValidationReport,
    PrincipalContext,
    QuantStrategyBrief,
    RoleType,
    StageConfidence,
    StageExecutionTrace,
)


class AuthSessionRequest(BaseModel):
    """Non-production session exchange request for minting signed Bearer tokens."""

    user_id: str = Field(default="demo_engineer", min_length=1, max_length=128)
    role: RoleType = Field(default=RoleType.ENGINEER)
    allowed_scopes: list[str] | None = None
    max_session_paid_tokens: int = Field(default=50_000, ge=1_000, le=500_000)


class AuthSessionResponse(BaseModel):
    """Signed Bearer token session response."""

    access_token: str
    token_type: Literal["Bearer"] = "Bearer"
    expires_in_seconds: int = 3600
    principal: PrincipalContext


class HealthHardwareProfile(BaseModel):
    """Hardware and memory profile specification."""

    device: str = "Unknown"
    unified_memory: str = "0.0 GB"
    memory_gb: float = 0.0
    arch: str = "unknown"
    os_name: str = "unknown"
    max_kv_tokens: int = 32768
    quantization: str = "4-bit (MLX)"
    lora_slots_allocated: int = 0
    lora_slots_capacity: int = 4
    postgres_version: str | None = None
    runtime_mode: str = "offline_deterministic"


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    database_connected: bool
    database_status: str
    registered_domains: list[str]
    adapter_pool_status: dict[str, Any]
    hardware_profile: dict[str, Any]
    runtime_models: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DomainDetail(BaseModel):
    """DomainPack detail with adapter verification, corpus counts, and training metrics."""

    domain_id: str
    display_name: str
    description: str = ""
    modalities: list[Literal["code", "document", "issues"]] = Field(default_factory=list)
    sample_queries: list[str] = Field(default_factory=list)
    lora_enabled: bool = True
    default_scopes: list[str] = Field(default_factory=list)
    doc_chunks_count: int = 0
    documents_count: int = 0
    code_files_count: int = 0
    ast_symbols_count: int = 0
    manifest: DomainPackManifest
    adapter_verification: dict[str, Any]
    training_metrics: dict[str, Any] | None = None


class DomainsResponse(BaseModel):
    """Response containing all registered domain packs."""

    domains: list[DomainDetail]
    count: int


class WorkspaceRolePreset(BaseModel):
    """Principal role preset with default RBAC scopes exposed by workspace-config."""

    role: RoleType
    label: str
    default_scopes: list[str] = Field(default_factory=list)


class PipelineStageSpec(BaseModel):
    """Metadata specification for one of the 4 Context Scout pipeline stages."""

    stage_id: int = Field(ge=1, le=4)
    number: str
    title: str
    summary: str
    what_happens: str
    gate_condition: str


class WorkspaceConfigResponse(BaseModel):
    """Dynamic workspace configuration consumed by UI and API clients."""

    domains: list[DomainDetail]
    federated_option: dict[str, Any]
    role_presets: list[WorkspaceRolePreset]
    pipeline_stages: list[PipelineStageSpec]
    source_filters: list[str]
    hardware_profile: dict[str, Any]
    database_connected: bool
    database_status: str
    runtime_mode: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ScoutStreamRequest(BaseModel):
    """Request payload for scout query or SSE stream."""

    query: str = Field(min_length=1, max_length=16_000)
    domain_id: str = Field(default="default")
    trace_id: str | None = None
    normalized_rrf_score: float | None = Field(default=None, ge=0.0, le=1.0)
    max_bounce_backs: int = Field(default=1, ge=0, le=3)
    source_filter: Literal["all", "code", "wiki", "jira"] = "all"


class ScoutTokenSavingsBreakdown(BaseModel):
    """Token savings and ROI breakdown."""

    local_tokens_explored: int
    pack_tokens_sent: int
    tokens_saved: int
    estimated_usd_saved: float
    frontier_savings_ratio_pct: float
    qlora_overhead_eliminated_tokens: float = 1634.0


class ScoutQueryResponse(BaseModel):
    """Synchronous scout query response."""

    trace_id: str
    domain_id: str
    passed: bool
    context_pack: ImplementationContextPack | None = None
    diagnostic: InsufficientContextDiagnostic | None = None
    citations: list[DocCitationSpan] = Field(default_factory=list)
    preflight_report: PreFlightValidationReport | None = None
    quant_strategy_spec: QuantStrategyBrief | None = None
    token_savings: ScoutTokenSavingsBreakdown
    trace_attributes: dict[str, Any] = Field(default_factory=dict)
    stage_traces: list[StageExecutionTrace] = Field(default_factory=list)


class SSEStageEvent(BaseModel):
    """Stage transition event emitted over Server-Sent Events."""

    event_type: Literal["stage_update"] = "stage_update"
    stage: int
    stage_name: str
    status: Literal["in_progress", "completed", "failed"] = "in_progress"
    details: str
    confidence: StageConfidence | None = None
    stage_trace: StageExecutionTrace | None = None
    elapsed_ms: float | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SSEBounceBackEvent(BaseModel):
    """$0 local bounce-back self-correction loop event."""

    event_type: Literal["bounce_back"] = "bounce_back"
    stage: int = 3
    status: Literal["bounce_back_triggered"] = "bounce_back_triggered"
    feedback: str
    bounce_count: int = 1
    remediation_action: str = "Re-querying local model with deterministic pre-flight feedback"
    paid_tokens_spent: int = 0
    cost_usd: float = 0.0
    details: str = "$0 local bounce-back loop triggered to resolve missing symbols"
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SSEContextPackReadyEvent(BaseModel):
    """Completion event containing final verified ContextPack."""

    event_type: Literal["context_pack_ready"] = "context_pack_ready"
    status: Literal["ready"] = "ready"
    pack: ImplementationContextPack | None = None
    quant_strategy_spec: QuantStrategyBrief | None = None
    token_savings: ScoutTokenSavingsBreakdown
    stage_traces: list[StageExecutionTrace] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SSEErrorEvent(BaseModel):
    """Error or blocked security event emitted over SSE."""

    event_type: Literal["error"] = "error"
    stage: str
    status: Literal["blocked", "error"] = "blocked"
    category: str
    rationale: str
    diagnostic: InsufficientContextDiagnostic | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class HandoffPreviewRequest(BaseModel):
    """Request payload to preview implementation patch."""

    pack: ImplementationContextPack
    model_id: str = Field(default="gemini-3.8-flash")
    custom_patch: str | None = None
    missing_context_requests: list[MissingContextRequest] | None = None


class HandoffPreviewResponse(BaseModel):
    """Handoff preview response with confirmation token."""

    confirmation_token: str
    patch_diff: str
    target_files: list[str]
    model_id: str
    local_tokens_consumed: int
    pack_token_count: int
    estimated_paid_cost_usd: float
    estimated_paid_savings_usd: float
    cost_low_usd: float | None = None
    cost_expected_usd: float | None = None
    cost_high_usd: float | None = None
    pricing_source: str | None = None
    requires_human_confirmation: Literal[True] = True
    expires_in_seconds: int = 3600
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class HandoffApplyRequest(BaseModel):
    """Request payload to apply and verify patch in worktree sandbox."""

    confirmation_token: str
    confirmed: bool
    custom_patch: str | None = None


class HandoffApplyResponse(BaseModel):
    """Handoff apply verification result."""

    success: bool
    confirmed: bool
    target_files: list[str]
    lint_passed: bool
    tests_passed: bool
    verification_passed: bool
    details: str
    telemetry: ExecutionTelemetryRecord


class AggregateROISummary(BaseModel):
    """Aggregated ROI and token savings metrics."""

    total_queries: int
    total_local_tokens_explored: int
    total_pack_tokens_sent: int
    total_paid_tokens_saved: int
    total_estimated_usd_saved: float
    frontier_savings_ratio_pct: float
    qlora_overhead_eliminated_per_query: float = 1634.0
    total_qlora_overhead_eliminated: float
    telemetry_source: Literal["postgresql_live", "in_memory_session", "m6_eval_baseline"] = (
        "postgresql_live"
    )


class ROIMetricsResponse(BaseModel):
    """ROI and evaluation benchmark reports response."""

    summary: AggregateROISummary
    m6_evaluation_report: dict[str, Any] | None = None
    m7_qlora_benchmark_report: dict[str, Any] | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
