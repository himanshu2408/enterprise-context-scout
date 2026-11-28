"""In-memory RLS repository twin composed from domain mixins."""

from __future__ import annotations

from context_scout.db.repos.in_memory_code_search import _InMemoryCodeAndSearchMixin
from context_scout.db.repos.in_memory_docs_connectors import _InMemoryDocsAndConnectorsMixin
from context_scout.db.repos.in_memory_ops_purge import _InMemoryOpsAndPurgeMixin
from context_scout.db.repos.in_memory_overlays_effects import _InMemoryOverlaysAndEffectsMixin


class InMemoryRLSRepository(
    _InMemoryCodeAndSearchMixin,
    _InMemoryOverlaysAndEffectsMixin,
    _InMemoryDocsAndConnectorsMixin,
    _InMemoryOpsAndPurgeMixin,
):
    """Deterministic in-memory twin enforcing identical RLS scope intersection and 3-way RRF.

    Used for fast unit testing and offline verification when a Postgres 17 daemon is not active.
    """


__all__ = ["InMemoryRLSRepository"]
