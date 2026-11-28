"""Database schema definitions, records, and domain models for Enterprise Context Scout."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Final, Literal
from uuid import UUID

from context_scout.schemas.contracts import DEFAULT_TENANT_ID, DocumentScope

CANONICAL_PG17_TABLES: Final[frozenset[str]] = frozenset(
    {
        "domain_packs",
        "code_files",
        "ast_symbols",
        "ast_edges",
        "doc_chunks",
        "execution_telemetry",
        "repositories",
        "repo_commits",
        "confluence_pages",
        "jira_issues",
        "indexing_queue",
        "indexing_dead_letter",
        "embedding_spaces",
        "domain_sync_manifests",
        "ref_snapshots",
        "overlay_file_deltas",
        "side_effects",
        "connector_sync_watermarks",
        "scout_sessions",
        "model_artifacts",
        "audit_events",
        "scim_users",
        "scim_groups",
    }
)

SymbolKind = Literal[
    "function",
    "class",
    "method",
    "interface",
    "type_alias",
    "hook",
    "slice",
    "selector",
    "reducer",
    "story",
    "tf_resource",
    "tf_module",
    "gcp_resource",
    "component",
    "system",
    "team",
    "service",
    "module",
    "api",
    "domain",
]

EdgeKind = Literal[
    "CALLS",
    "IMPORTS",
    "USES_TYPE",
    "INHERITS",
    "TESTED_BY",
    "PROVISIONED_BY",
    "DISPATCHES",
    "SELECTS",
    "DOCUMENTS_COMPONENT",
    "OWNED_BY_TEAM",
    "PART_OF_SYSTEM",
    "PUBLISHES_EVENT",
    "SUBSCRIBES_EVENT",
    "READS_TABLE",
    "WRITES_TABLE",
    "RENDERS_TEMPLATE",
    "IMPLEMENTS_SPEC",
    "IMPLEMENTS_ISSUE",
    "BINDS_SERVICE_ACCOUNT",
    "ACCESSES_SECRET",
]

EffectCategory = Literal[
    "email",
    "sms",
    "push_notification",
    "webhook",
    "pubsub_kafka_event",
    "cron_job",
    "feature_flag",
    "db_write",
]

SideEffectSeverity = Literal[
    "external_comms",
    "external_io",
    "async_messaging",
    "scheduled_job",
    "runtime_toggle",
    "state_mutation",
]


@dataclass(frozen=True)
class RepositoryRecord:
    repo_id: UUID
    domain_id: str
    repo_name: str
    source_type: Literal["gitlab", "github", "local_git", "airgap_bundle"]
    remote_url: str
    default_branch: str
    allowed_scopes: list[str]
    head_commit_sha: str = ""
    metadata_json: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class RepoCommitRecord:
    commit_id: UUID
    repo_id: UUID
    commit_sha: str
    parent_shas: list[str]
    author_name: str
    author_email: str
    commit_message: str
    committed_at: datetime
    indexed_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class ConfluencePageRecord:
    page_id: UUID
    domain_id: str
    confluence_page_id: str
    space_key: str
    title: str
    version_number: int
    author_username: str
    source_url: str
    allowed_scopes: list[str]
    last_modified_at: datetime
    indexed_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class JiraIssueRecord:
    issue_id: UUID
    domain_id: str
    issue_key: str
    project_key: str
    summary: str
    issue_type: str
    status: str
    source_url: str
    allowed_scopes: list[str]
    updated_at: datetime
    resolution: str | None = None
    assignee_username: str | None = None
    indexed_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID
    epic_key: str | None = None
    parent_issue_key: str | None = None
    linked_issues: list[dict[str, Any]] = field(default_factory=list)
    linked_mrs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CodeFileRecord:
    file_id: UUID
    domain_id: str
    repo_name: str
    file_path: str
    git_commit_sha: str
    git_blob_sha: str
    language: str
    allowed_scopes: list[str]
    raw_content: str
    line_count: int
    repo_id: UUID | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID
    ranking_weight: float = 1.0
    is_generated: bool = False
    is_vendored: bool = False


@dataclass(frozen=True)
class RefSnapshotRecord:
    snapshot_id: UUID
    domain_id: str
    repo_name: str
    ref_name: str
    ref_kind: Literal["base_branch", "feature_branch", "merge_request", "pull_request"]
    base_commit_sha: str
    head_commit_sha: str
    allowed_scopes: list[str]
    repo_id: UUID | None = None
    mr_iid: str | None = None
    base_snapshot_id: UUID | None = None
    state: Literal["active", "open", "merged", "closed"] = "open"
    metadata_json: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class OverlayFileDeltaRecord:
    delta_id: UUID
    snapshot_id: UUID
    domain_id: str
    repo_name: str
    file_path: str
    change_type: Literal["added", "modified", "deleted", "renamed"]
    allowed_scopes: list[str]
    previous_file_path: str | None = None
    base_blob_sha: str | None = None
    overlay_blob_sha: str | None = None
    language: str = "text"
    raw_content: str | None = None
    line_count: int = 0
    ranking_weight: float = 1.0
    is_generated: bool = False
    is_vendored: bool = False
    symbols_json: list[dict[str, Any]] = field(default_factory=list)
    edges_json: list[dict[str, Any]] = field(default_factory=list)
    indexed_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class ResolvedOverlayFileRecord:
    domain_id: str
    repo_name: str
    file_path: str
    git_commit_sha: str
    git_blob_sha: str
    language: str
    allowed_scopes: list[str]
    raw_content: str
    line_count: int
    source_layer: Literal["base_snapshot", "overlay"]
    snapshot_id: UUID | None = None
    base_file_id: UUID | None = None
    delta_id: UUID | None = None
    change_type: str | None = None
    ranking_weight: float = 1.0
    is_generated: bool = False
    is_vendored: bool = False
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class ASTSymbolRecord:
    symbol_id: UUID
    file_id: UUID
    domain_id: str
    qualified_name: str
    symbol_kind: str
    start_line: int
    end_line: int
    signature_skeleton: str
    contextual_summary: str
    allowed_scopes: list[str]
    embedding: list[float] = field(default_factory=list)
    repo_id: UUID | None = None
    repo_qualified_id: str | None = None
    bare_name: str | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID
    embedding_space_id: str = "embeddinggemma-2-768"


@dataclass(frozen=True)
class ASTEdgeRecord:
    edge_id: UUID
    domain_id: str
    source_symbol_id: UUID
    target_qualified_name: str
    target_symbol_id: UUID | None
    edge_kind: str
    allowed_scopes: list[str]
    target_repo_qualified_id: str | None = None
    is_cross_repo: bool = False
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class DocChunkRecord:
    chunk_id: UUID
    domain_id: str
    doc_id: str
    doc_title: str
    source_uri: str
    market_scope: DocumentScope | str | None
    published_date: date | None
    authority_weight: float
    section_heading: str
    contextual_header: str
    verbatim_content: str
    metadata_json: dict[str, object]
    allowed_scopes: list[str]
    embedding: list[float]
    source_type: str = "enterprise_doc"
    version_str: str = "1"
    is_stale: bool = False
    stale_reason: str | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID
    embedding_space_id: str = "embeddinggemma-2-768"


@dataclass(frozen=True)
class EmbeddingSpaceRecord:
    domain_id: str
    space_id: str
    model_id: str
    dimensions: int = 768
    provider: str = "sentence-transformers"
    quantization: Literal["halfvec", "vector"] = "halfvec"
    status: Literal["active", "shadow", "retired"] = "shadow"
    is_active: bool = False
    index_operator_class: str = "halfvec_cosine_ops"
    tenant_id: UUID = DEFAULT_TENANT_ID
    created_at: datetime | None = None
    activated_at: datetime | None = None


@dataclass(frozen=True)
class DomainSyncManifestRecord:
    domain_id: str
    kind: str
    source_dir: str = ""
    repo_commit_shas: dict[str, str] = field(default_factory=dict)
    empty_doc_shas: dict[str, str] = field(default_factory=dict)
    code_files_indexed: int = 0
    ast_symbols_indexed: int = 0
    ast_edges_indexed: int = 0
    doc_chunks_indexed: int = 0
    tenant_id: UUID = DEFAULT_TENANT_ID
    s3_manifest_uri: str | None = None
    manifest_sha256: str | None = None
    synced_at: datetime | None = None


@dataclass(frozen=True)
class ASTNeighborhoodRecord:
    """Resolved 1-hop AST neighbor record returned by recursive CTE or memory traversal."""

    intent_role: Literal["imported_type", "direct_caller", "test_fixture"]
    edge_kind: str
    depth: int
    symbol: ASTSymbolRecord
    file: CodeFileRecord


@dataclass(frozen=True)
class SideEffectRecord:
    side_effect_id: UUID
    domain_id: str
    repo_name: str
    file_path: str
    qualified_symbol: str
    start_line: int
    end_line: int
    effect_category: EffectCategory
    target_identifier: str
    allowed_scopes: list[str]
    file_id: UUID | None = None
    symbol_id: UUID | None = None
    provider_or_driver: str = ""
    trigger_condition: str = ""
    template_or_schema_ref: str | None = None
    severity_tier: SideEffectSeverity = "external_io"
    confidence: float = 1.0
    metadata_json: dict[str, Any] = field(default_factory=dict)
    indexed_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class ConnectorWatermarkRecord:
    domain_id: str
    source_id: str
    container_key: str
    watermark_cursor: str = ""
    sync_tier: Literal["webhook", "watermark", "reconcile", "bundle"] = "watermark"
    items_synced: int = 0
    last_webhook_at: datetime | None = None
    last_watermark_sync_at: datetime | None = None
    last_reconciliation_at: datetime | None = None
    metadata_json: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class ScoutSessionRecord:
    session_id: str
    domain_id: str
    principal_id: str
    allowed_scopes: list[str]
    turn_count: int = 0
    compacted_summary: str = ""
    recent_turns_json: list[dict[str, Any]] = field(default_factory=list)
    pinned_file_blobs_json: dict[str, dict[str, str]] = field(default_factory=dict)
    pinned_symbols_json: list[str] = field(default_factory=list)
    total_tokens_compacted: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


ModelArtifactKind = Literal["lora_adapter", "bi_encoder", "merged_checkpoint"]
ModelArtifactStatus = Literal["staged", "active", "retired", "revoked"]


@dataclass(frozen=True)
class ModelArtifactRecord:
    domain_id: str
    artifact_id: str
    base_model_id: str
    base_model_revision_sha256: str
    storage_uri: str
    weights_object_key: str
    config_object_key: str
    weights_sha256: str
    training_scope_ceiling: list[str]
    allowed_scopes: list[str]
    artifact_kind: ModelArtifactKind = "lora_adapter"
    signature_object_key: str = ""
    cosign_signature_b64: str = ""
    cosign_key_id: str = "default"
    lora_rank: int = 16
    lora_alpha: int = 32
    use_rslora: bool = True
    use_dora: bool = False
    canary_leak_count: int = 0
    status: ModelArtifactStatus = "staged"
    is_active: bool = False
    metadata_json: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    activated_at: datetime | None = None
    updated_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class DomainPackRecord:
    domain_id: str
    display_name: str
    lora_adapter_id: str
    primary_truth_source: Literal["code", "document"]
    config_json: dict[str, Any] = field(default_factory=dict)
    freshness_half_life_days: int = 180
    s3_bundle_uri: str | None = None
    bundle_sha256: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class ExecutionTelemetryRecord:
    trace_id: UUID
    domain_id: str
    principal_id: str
    principal_role: str
    execution_mode: Literal["chat_qa", "gemini_handoff"]
    local_tokens_read: int
    context_pack_tokens_sent: int
    paid_tokens_saved: int
    estimated_usd_saved: float
    preflight_passed_first_try: bool
    latency_ms: int
    zero_touch_sufficient: bool | None = None
    missing_context_requests: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class IndexingQueueRecord:
    task_id: UUID
    domain_id: str
    event_type: str
    payload_json: dict[str, Any] = field(default_factory=dict)
    priority: int = 10
    status: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"] = "PENDING"
    worker_id: str | None = None
    attempts: int = 0
    max_attempts: int = 3
    error_message: str | None = None
    scheduled_at: datetime | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    processed_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class IndexingDeadLetterRecord:
    dlq_id: UUID
    task_id: UUID
    domain_id: str
    event_type: str
    payload_json: dict[str, Any]
    failure_reason: str
    stack_trace: str
    attempts: int
    failed_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class AuditEventRecord:
    seq_no: int
    event_id: UUID
    actor_id: str
    actor_role: str
    event_type: str
    prev_hash_sha256: str
    event_hash_sha256: str
    domain_id: str = "default"
    severity: str = "INFO"
    resource_uri: str = ""
    outcome: str = "allow"
    trace_id: str = ""
    span_id: str = ""
    payload_json: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class ScimUserRecord:
    scim_user_id: str
    user_name: str
    domain_id: str = "default"
    external_id: str = ""
    email: str = ""
    display_name: str = ""
    active: bool = True
    role_override: str | None = None
    custom_scopes: list[str] = field(default_factory=list)
    metadata_json: dict[str, Any] = field(default_factory=dict)
    emails_json: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class ScimGroupRecord:
    scim_group_id: str
    display_name: str
    domain_id: str = "default"
    external_id: str = ""
    member_user_ids: list[str] = field(default_factory=list)
    member_group_ids: list[str] = field(default_factory=list)
    mapped_scopes: list[str] = field(default_factory=list)
    mapped_role: str | None = None
    metadata_json: dict[str, Any] = field(default_factory=dict)
    members_json: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class RankedSearchHit:
    item_id: UUID
    qualified_name_or_title: str
    normalized_rrf_score: float
    trigram_rank: int | None
    bm25_rank: int | None
    vector_rank: int | None


CANONICAL_TABLE_RECORD_CLASSES: Final[dict[str, type[Any]]] = {
    "domain_packs": DomainPackRecord,
    "code_files": CodeFileRecord,
    "ast_symbols": ASTSymbolRecord,
    "ast_edges": ASTEdgeRecord,
    "doc_chunks": DocChunkRecord,
    "execution_telemetry": ExecutionTelemetryRecord,
    "repositories": RepositoryRecord,
    "repo_commits": RepoCommitRecord,
    "confluence_pages": ConfluencePageRecord,
    "jira_issues": JiraIssueRecord,
    "indexing_queue": IndexingQueueRecord,
    "indexing_dead_letter": IndexingDeadLetterRecord,
    "embedding_spaces": EmbeddingSpaceRecord,
    "domain_sync_manifests": DomainSyncManifestRecord,
    "ref_snapshots": RefSnapshotRecord,
    "overlay_file_deltas": OverlayFileDeltaRecord,
    "side_effects": SideEffectRecord,
    "connector_sync_watermarks": ConnectorWatermarkRecord,
    "scout_sessions": ScoutSessionRecord,
    "model_artifacts": ModelArtifactRecord,
    "audit_events": AuditEventRecord,
    "scim_users": ScimUserRecord,
    "scim_groups": ScimGroupRecord,
}
