"""Data models and constants for Post-PITR multi-store state reconciliation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Final

_DOMAIN_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9_]+$")
_ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"gs", "s3", "memory"})
_FORBIDDEN_ADMIN_DB_ROLES: Final[frozenset[str]] = frozenset(
    {"postgres", "postgres_admin", "cloudsqlsuperuser", "rds_superuser", "admin", "root"}
)
_GIT_SOURCE_IDS: Final[frozenset[str]] = frozenset({"gitlab", "github", "local_git"})


class StateReconcilerError(RuntimeError):
    """Raised when post-PITR state reconciliation encounters a non-recoverable contract breach."""


@dataclass(frozen=True)
class CommitWatermarkDrift:
    """Detected commit / connector watermark discrepancy for a repository."""

    repo_name: str
    pg_commit_sha: str
    manifest_commit_sha: str | None
    object_store_commit_sha: str | None
    connector_watermark_cursor: str | None
    action_taken: str


@dataclass(frozen=True)
class EmbeddingWatermarkDrift:
    """Summary of embedding space drift & halfvec(D) healing for a domain."""

    active_space_id: str | None
    orphaned_shadow_rows_cleared: int = 0
    drifted_symbol_rows_found: int = 0
    drifted_chunk_rows_found: int = 0
    symbol_rows_reembedded: int = 0
    chunk_rows_reembedded: int = 0
    unhealed_rows: int = 0


@dataclass(frozen=True)
class AdapterWatermarkDrift:
    """Summary of model_artifacts vs GCS/S3 & vLLM pod reconciliation."""

    active_artifact_id_before: str | None
    active_artifact_id_after: str | None
    retired_broken_artifacts: list[str] = field(default_factory=list)
    promoted_artifact_id: str | None = None
    orphaned_object_keys: list[str] = field(default_factory=list)
    pruned_object_keys: list[str] = field(default_factory=list)
    sidecar_synced_pods: int = 0


@dataclass(frozen=True)
class StateReconciliationReport:
    """Structured audit report produced by StateReconciler.reconcile()."""

    domain_id: str
    tenant_id: str
    drift_detected: bool
    healed: bool
    commit_drifts: list[CommitWatermarkDrift]
    zoekt_purged_repos: int
    zoekt_purged_files: int
    zoekt_reindexed_files: int
    watermarks_rewound: int
    reconcile_tasks_enqueued: int
    stale_tasks_reclaimed: int
    object_manifests_repaired: list[str]
    embedding_drift: EmbeddingWatermarkDrift
    adapter_drift: AdapterWatermarkDrift
    permission_cache_invalidated: bool
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
