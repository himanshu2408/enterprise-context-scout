"""Database session management and RLS-enforced repositories."""

from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    DomainSyncManifestRecord,
    InMemoryRLSRepository,
    PostgresRLSRepository,
    RankedSearchHit,
    ScoutSessionRecord,
)
from context_scout.db.session import (
    PoolRole,
    RLSConnectionManager,
    apply_schema_ddl,
    get_ingestion_pool,
    get_query_pool,
)

__all__ = [
    "ASTEdgeRecord",
    "ASTSymbolRecord",
    "CodeFileRecord",
    "DocChunkRecord",
    "DomainSyncManifestRecord",
    "InMemoryRLSRepository",
    "PoolRole",
    "PostgresRLSRepository",
    "RLSConnectionManager",
    "RankedSearchHit",
    "ScoutSessionRecord",
    "apply_schema_ddl",
    "get_ingestion_pool",
    "get_query_pool",
]
