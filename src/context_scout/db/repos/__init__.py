"""Bounded domain mixins for PostgreSQL 17 and In-Memory RLS repositories."""

from __future__ import annotations

from context_scout.db.repos.in_memory_repo import InMemoryRLSRepository
from context_scout.db.repos.postgres_repo import PostgresRLSRepository
from context_scout.db.repos.types_and_helpers import (
    SYSTEM_CONNECTOR_INGEST_SCOPE,
    CalibratedRankedSearchHit,
    NormalizedPurgeSelectors,
    OwnershipEdgeRecord,
    normalize_purge_selectors,
)

__all__ = [
    "SYSTEM_CONNECTOR_INGEST_SCOPE",
    "CalibratedRankedSearchHit",
    "InMemoryRLSRepository",
    "NormalizedPurgeSelectors",
    "OwnershipEdgeRecord",
    "PostgresRLSRepository",
    "normalize_purge_selectors",
]
