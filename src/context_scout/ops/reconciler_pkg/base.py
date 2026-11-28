"""Typed state base for StateReconciler mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import httpx

from context_scout.agents.llm_backend import AdapterRouter
from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
from context_scout.db.session import RLSConnectionManager
from context_scout.queue.task_queue import TaskQueueProtocol
from context_scout.retrieval.embedder import EmbeddingProviderSPI
from context_scout.serving.adapter_sync_sidecar import AdapterSyncSidecar
from context_scout.storage.object_store import ArtifactObjectStore


class _StateReconcilerBase:
    """Shared typed attributes for StateReconciler domain mixins."""

    _settings: ScoutSettings
    repository: PostgresRLSRepository | InMemoryRLSRepository
    connection_manager: RLSConnectionManager | None
    zoekt_clients: list[Any]
    _default_object_store: ArtifactObjectStore | None
    _stores_by_scheme: dict[str, ArtifactObjectStore]
    _dynamic_bucket_stores: dict[tuple[str, str], ArtifactObjectStore]
    adapter_sidecar: AdapterSyncSidecar | None
    adapter_router: AdapterRouter | None
    task_queue: TaskQueueProtocol | None
    embedder: EmbeddingProviderSPI | None
    permission_cache: Any | None
    promote_last_known_good_adapter: bool
    prune_orphan_objects: bool
    stale_task_seconds: int
    reconcile_batch_size: int
    _http_client: httpx.Client | None

    if TYPE_CHECKING:

        def _validate_domain_id(self, domain_id: str) -> str: ...

        @staticmethod
        def _infer_store_scheme(store: ArtifactObjectStore) -> str: ...

        def _assert_pg_session_non_bypass_rls(self, conn: Any) -> None: ...

        @staticmethod
        def _assert_store_bucket_matches(
            store: ArtifactObjectStore, expected_bucket: str, storage_uri: str
        ) -> ArtifactObjectStore: ...

        def _resolve_object_store_for_uri(self, storage_uri: str) -> ArtifactObjectStore: ...

        def _target_stores(self) -> list[tuple[str, ArtifactObjectStore]]: ...

        @staticmethod
        def _extract_key_from_uri(uri: str, expected_tenant_id: UUID) -> str | None: ...
