"""RLS-enforced repositories and 3-way Normalized RRF search (M1 Foundation)."""

from __future__ import annotations

from context_scout.db.repos.in_memory_repo import InMemoryRLSRepository as InMemoryRLSRepository
from context_scout.db.repos.postgres_repo import PostgresRLSRepository as PostgresRLSRepository
from context_scout.db.repos.types_and_helpers import (
    SYSTEM_CONNECTOR_INGEST_SCOPE as SYSTEM_CONNECTOR_INGEST_SCOPE,
)
from context_scout.db.repos.types_and_helpers import (
    CalibratedRankedSearchHit as CalibratedRankedSearchHit,
)
from context_scout.db.repos.types_and_helpers import (
    NormalizedPurgeSelectors as NormalizedPurgeSelectors,
)
from context_scout.db.repos.types_and_helpers import (
    OwnershipEdgeRecord as OwnershipEdgeRecord,
)
from context_scout.db.repos.types_and_helpers import (
    _apply_authority_score as _apply_authority_score,
)
from context_scout.db.repos.types_and_helpers import (
    _as_dt as _as_dt,
)
from context_scout.db.repos.types_and_helpers import (
    _as_opt_dt as _as_opt_dt,
)
from context_scout.db.repos.types_and_helpers import (
    _compute_calibrated_relevance as _compute_calibrated_relevance,
)
from context_scout.db.repos.types_and_helpers import (
    _connector_ingest_or_scopes_intersect as _connector_ingest_or_scopes_intersect,
)
from context_scout.db.repos.types_and_helpers import (
    _cosine_similarity as _cosine_similarity,
)
from context_scout.db.repos.types_and_helpers import (
    _format_db_vector_and_halfvec as _format_db_vector_and_halfvec,
)
from context_scout.db.repos.types_and_helpers import (
    _infer_market_scopes_from_query as _infer_market_scopes_from_query,
)
from context_scout.db.repos.types_and_helpers import (
    _market_scope_aliases as _market_scope_aliases,
)
from context_scout.db.repos.types_and_helpers import (
    _parse_market_scope as _parse_market_scope,
)
from context_scout.db.repos.types_and_helpers import (
    _scopes_intersect as _scopes_intersect,
)
from context_scout.db.repos.types_and_helpers import (
    _serialize_market_scope as _serialize_market_scope,
)
from context_scout.db.repos.types_and_helpers import (
    _trigram_similarity as _trigram_similarity,
)
from context_scout.db.repos.types_and_helpers import (
    normalize_purge_selectors as normalize_purge_selectors,
)
from context_scout.db.schema import (
    ASTEdgeRecord as ASTEdgeRecord,
)
from context_scout.db.schema import (
    ASTNeighborhoodRecord as ASTNeighborhoodRecord,
)
from context_scout.db.schema import (
    ASTSymbolRecord as ASTSymbolRecord,
)
from context_scout.db.schema import (
    CodeFileRecord as CodeFileRecord,
)
from context_scout.db.schema import (
    ConfluencePageRecord as ConfluencePageRecord,
)
from context_scout.db.schema import (
    ConnectorWatermarkRecord as ConnectorWatermarkRecord,
)
from context_scout.db.schema import (
    DocChunkRecord as DocChunkRecord,
)
from context_scout.db.schema import (
    DomainSyncManifestRecord as DomainSyncManifestRecord,
)
from context_scout.db.schema import (
    EffectCategory as EffectCategory,
)
from context_scout.db.schema import (
    EmbeddingSpaceRecord as EmbeddingSpaceRecord,
)
from context_scout.db.schema import (
    JiraIssueRecord as JiraIssueRecord,
)
from context_scout.db.schema import (
    ModelArtifactRecord as ModelArtifactRecord,
)
from context_scout.db.schema import (
    OverlayFileDeltaRecord as OverlayFileDeltaRecord,
)
from context_scout.db.schema import (
    RankedSearchHit as RankedSearchHit,
)
from context_scout.db.schema import (
    RefSnapshotRecord as RefSnapshotRecord,
)
from context_scout.db.schema import (
    RepoCommitRecord as RepoCommitRecord,
)
from context_scout.db.schema import (
    RepositoryRecord as RepositoryRecord,
)
from context_scout.db.schema import (
    ResolvedOverlayFileRecord as ResolvedOverlayFileRecord,
)
from context_scout.db.schema import (
    ScoutSessionRecord as ScoutSessionRecord,
)
from context_scout.db.schema import (
    SideEffectRecord as SideEffectRecord,
)
from context_scout.db.schema import (
    SideEffectSeverity as SideEffectSeverity,
)

__all__ = [
    "SYSTEM_CONNECTOR_INGEST_SCOPE",
    "ASTEdgeRecord",
    "ASTNeighborhoodRecord",
    "ASTSymbolRecord",
    "CalibratedRankedSearchHit",
    "CodeFileRecord",
    "ConfluencePageRecord",
    "ConnectorWatermarkRecord",
    "DocChunkRecord",
    "DomainSyncManifestRecord",
    "EffectCategory",
    "EmbeddingSpaceRecord",
    "InMemoryRLSRepository",
    "JiraIssueRecord",
    "ModelArtifactRecord",
    "NormalizedPurgeSelectors",
    "OverlayFileDeltaRecord",
    "OwnershipEdgeRecord",
    "PostgresRLSRepository",
    "RankedSearchHit",
    "RefSnapshotRecord",
    "RepoCommitRecord",
    "RepositoryRecord",
    "ResolvedOverlayFileRecord",
    "ScoutSessionRecord",
    "SideEffectRecord",
    "SideEffectSeverity",
    "normalize_purge_selectors",
]
