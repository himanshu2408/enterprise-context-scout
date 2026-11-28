"""Typed state base for InMemoryRLSRepository preserving exact internal dict attributes."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING
from uuid import UUID

from context_scout.db.schema import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    ConfluencePageRecord,
    ConnectorWatermarkRecord,
    DocChunkRecord,
    DomainSyncManifestRecord,
    EmbeddingSpaceRecord,
    JiraIssueRecord,
    ModelArtifactRecord,
    OverlayFileDeltaRecord,
    RefSnapshotRecord,
    RepoCommitRecord,
    RepositoryRecord,
    ScoutSessionRecord,
    SideEffectRecord,
)
from context_scout.schemas.contracts import (
    DocumentScope,
    ExecutionTelemetryRecord,
    PrincipalContext,
)

logger = logging.getLogger(__name__)


class _InMemoryStateBase:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._files: dict[UUID, CodeFileRecord] = {}
        self._symbols: dict[UUID, ASTSymbolRecord] = {}
        self._edges: dict[UUID, ASTEdgeRecord] = {}
        self._docs: dict[UUID, DocChunkRecord] = {}
        self._telemetry: dict[UUID, ExecutionTelemetryRecord] = {}
        self._repos: dict[UUID, RepositoryRecord] = {}
        self._commits: dict[UUID, RepoCommitRecord] = {}
        self._confluence: dict[UUID, ConfluencePageRecord] = {}
        self._deleted_confluence: set[tuple[UUID, str, str]] = set()
        self._jira: dict[UUID, JiraIssueRecord] = {}
        self._embedding_spaces: dict[tuple[UUID, str, str], EmbeddingSpaceRecord] = {}
        self._shadow_symbol_vecs: dict[tuple[UUID, UUID, str], list[float]] = {}
        self._shadow_doc_vecs: dict[tuple[UUID, UUID, str], list[float]] = {}
        self._ref_snapshots: dict[UUID, RefSnapshotRecord] = {}
        self._overlay_deltas: dict[tuple[UUID, UUID, str], OverlayFileDeltaRecord] = {}
        self._side_effects: dict[UUID, SideEffectRecord] = {}
        self._watermarks: dict[tuple[UUID, str, str, str], ConnectorWatermarkRecord] = {}
        self._sessions: dict[tuple[UUID, str], ScoutSessionRecord] = {}
        self._model_artifacts: dict[tuple[UUID, str, str], ModelArtifactRecord] = {}
        self._sync_manifests: dict[tuple[UUID, str], DomainSyncManifestRecord] = {}

    if TYPE_CHECKING:

        def list_visible_code_files(
            self, principal: PrincipalContext, domain_id: str
        ) -> list[CodeFileRecord]: ...

        def list_visible_ast_symbols(
            self, principal: PrincipalContext, domain_id: str
        ) -> list[ASTSymbolRecord]: ...

        def list_visible_ast_edges(
            self, principal: PrincipalContext, domain_id: str
        ) -> list[ASTEdgeRecord]: ...

        def list_visible_doc_chunks(
            self,
            principal: PrincipalContext,
            domain_id: str,
            market_scope: DocumentScope | str | None = None,
        ) -> list[DocChunkRecord]: ...
