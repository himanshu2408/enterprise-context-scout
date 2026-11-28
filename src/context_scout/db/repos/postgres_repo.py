"""RLS-enforced PostgreSQL 17 repository facade composed from domain mixins."""

from __future__ import annotations

from context_scout.db.repos.pg_catalog_code import _PgCatalogAndCodeMixin
from context_scout.db.repos.pg_docs_connectors import _PgDocsAndConnectorsMixin
from context_scout.db.repos.pg_embedding_purge import _PgEmbeddingAndPurgeMixin
from context_scout.db.repos.pg_graph_search import _PgGraphSearchMixin
from context_scout.db.repos.pg_overlays_effects import _PgOverlaysAndEffectsMixin
from context_scout.db.repos.pg_sessions_models import _PgSessionsAndModelsMixin


class PostgresRLSRepository(
    _PgCatalogAndCodeMixin,
    _PgGraphSearchMixin,
    _PgOverlaysAndEffectsMixin,
    _PgDocsAndConnectorsMixin,
    _PgEmbeddingAndPurgeMixin,
    _PgSessionsAndModelsMixin,
):
    """Executes parameterized SQL inside an RLS-enforced PostgreSQL 17 transaction."""


__all__ = ["PostgresRLSRepository"]
