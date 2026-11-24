"""Post-PITR Multi-Store State Reconciler (PG17/Cloud SQL + Zoekt + GCS gs:// & S3 s3:// + vLLM).

Implements PITR cross-store reconciliation (`L9`):
1. Reconciles PostgreSQL 17 / Cloud SQL commit watermarks (`repositories`, `repo_commits`,
   `code_files`, `domain_sync_manifests`, `connector_sync_watermarks`) against Zoekt
   trigram shards (`LocalTrigramZoektIndex` & `ZoektGrpcJsonClient`) and `ArtifactObjectStore`
   sync manifests across both keyless GCP `gs://` (`GCSObjectStore`) and `s3://` S3/MinIO
   (`S3CompatibleObjectStore`), rewinding drifted watermarks and enqueuing catch-up tasks.
2. Reconciles embedding watermarks (`embedding_spaces` vs `ast_symbols` & `doc_chunks`),
   clearing orphaned shadow vectors from interrupted blue/green cutovers and re-embedding
   drifted rows with real non-zero `halfvec(D)` vectors (never `[0.0]*D` or fake hashes).
3. Reconciles governed LoRA/model adapter watermarks (`model_artifacts` vs `gs://` & `s3://`
   object stores & `AdapterSyncSidecar`), failing closed on corrupted active cartridges,
   optionally promoting the latest verified intact cartridge, pruning orphaned objects, and
   sweeping stranded post-PITR adapters across all `vLLM` pods.
4. Enforces `NOBYPASSRLS` (`scout_app_user` only; refuses `postgres`/superuser/`BYPASSRLS`),
   reclaims stale `PROCESSING` queue tasks, invalidates stale permission caches, and records
   tamper-evident audit events in `audit_events`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Final
from urllib.parse import urlparse
from uuid import UUID

import httpx

from context_scout.agents.llm_backend import AdapterRouter
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    InMemoryRLSRepository,
    PostgresRLSRepository,
    _format_db_vector_and_halfvec,
)
from context_scout.db.schema import (
    CodeFileRecord,
    ConnectorWatermarkRecord,
    DomainSyncManifestRecord,
    EmbeddingSpaceRecord,
    ModelArtifactRecord,
    RepositoryRecord,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.queue.task_queue import PostgresTaskQueue, TaskQueueProtocol
from context_scout.retrieval.embedder import (
    DEFAULT_EMBEDDING_SPACE_ID,
    DEFAULT_GOOGLE_EMBEDDING_MODEL,
    EmbeddingProviderSPI,
    get_embedding_provider,
    validate_non_zero_embedding,
)
from context_scout.schemas.contracts import DomainPackManifest, PrincipalContext
from context_scout.security.audit_chain import get_audit_chain_service
from context_scout.serving.adapter_sync_sidecar import AdapterSyncSidecar
from context_scout.storage.object_store import (
    ArtifactObjectStore,
    GCSObjectStore,
    InMemoryObjectStore,
    S3CompatibleObjectStore,
    build_tenant_domain_key,
    validate_object_key,
    validate_object_prefix,
)

logger = logging.getLogger(__name__)

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


class StateReconciler:
    """Reconciles PG17/Cloud SQL state against Zoekt, GCS/S3 object stores, Redis, and vLLM."""

    def __init__(
        self,
        *,
        repository: PostgresRLSRepository | InMemoryRLSRepository | None = None,
        connection_manager: RLSConnectionManager | None = None,
        zoekt_index: Any | None = None,
        zoekt_clients: Sequence[Any] = (),
        default_object_store: ArtifactObjectStore | None = None,
        object_stores_by_scheme: Mapping[str, ArtifactObjectStore] | None = None,
        adapter_sidecar: AdapterSyncSidecar | None = None,
        adapter_router: AdapterRouter | None = None,
        task_queue: TaskQueueProtocol | None = None,
        embedder: EmbeddingProviderSPI | None = None,
        permission_cache: Any | None = None,
        settings: ScoutSettings | None = None,
        promote_last_known_good_adapter: bool = True,
        prune_orphan_objects: bool = False,
        stale_task_seconds: int = 0,
    ) -> None:
        self._settings = settings or ScoutSettings()
        if connection_manager is not None:
            self._validate_non_admin_rls_manager(connection_manager, self._settings)

        if repository is None:
            self.repository: PostgresRLSRepository | InMemoryRLSRepository = PostgresRLSRepository()
        else:
            self.repository = repository

        self.connection_manager = connection_manager
        if isinstance(self.repository, PostgresRLSRepository) and self.connection_manager is None:
            raise ValueError(
                "StateReconciler with PostgresRLSRepository requires RLSConnectionManager "
                "(NOBYPASSRLS runtime pool)."
            )

        merged_zoekt: list[Any] = []
        seen_zoekt_ids: set[int] = set()
        for candidate in [zoekt_index, *zoekt_clients]:
            if candidate is not None and id(candidate) not in seen_zoekt_ids:
                merged_zoekt.append(candidate)
                seen_zoekt_ids.add(id(candidate))
        self.zoekt_clients: list[Any] = merged_zoekt

        self._default_object_store = default_object_store
        self._stores_by_scheme: dict[str, ArtifactObjectStore] = {
            k.strip().lower().rstrip(":/"): v for k, v in (object_stores_by_scheme or {}).items()
        }
        self._dynamic_bucket_stores: dict[tuple[str, str], ArtifactObjectStore] = {}
        self.adapter_sidecar = adapter_sidecar
        self.adapter_router = adapter_router
        self.task_queue = task_queue
        self.embedder = embedder
        self.permission_cache = permission_cache
        self.promote_last_known_good_adapter = bool(promote_last_known_good_adapter)
        self.prune_orphan_objects = bool(prune_orphan_objects)
        self.stale_task_seconds = max(0, int(stale_task_seconds))

        if isinstance(self.task_queue, PostgresTaskQueue):
            self._validate_non_admin_rls_manager(self.task_queue._conn_mgr, self._settings)

    @staticmethod
    def _validate_non_admin_rls_manager(
        mgr: RLSConnectionManager,
        settings: ScoutSettings | None = None,
    ) -> None:
        raw_dsn = str(
            getattr(mgr, "_app_db_url", "")
            or getattr(mgr, "_conninfo", "")
            or getattr(mgr, "db_url", "")
            or ""
        ).strip()
        if not raw_dsn:
            return
        parsed = urlparse(raw_dsn)
        user = (parsed.username or "").strip().lower()
        if user in _FORBIDDEN_ADMIN_DB_ROLES or user.startswith("postgres"):
            raise PermissionError(
                f"StateReconciler MUST NOT connect as privileged DB role '{user}'; "
                "use RLS runtime pool ('scout_app_user' with NOBYPASSRLS)."
            )
        if settings is not None:
            admin_parsed = urlparse(settings.db_admin_url.strip())
            if raw_dsn == settings.db_admin_url.strip() or (
                parsed.hostname == admin_parsed.hostname
                and parsed.port == admin_parsed.port
                and parsed.path == admin_parsed.path
                and user
                and user == (admin_parsed.username or "").strip().lower()
            ):
                raise PermissionError(
                    "StateReconciler MUST NOT use db_admin_url; use RLS runtime pool (NOBYPASSRLS)."
                )

    @staticmethod
    def _assert_pg_session_non_bypass_rls(conn: Any) -> None:
        row = conn.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user;"
        ).fetchone()
        if row is not None and (bool(row[0]) or bool(row[1])):
            raise PermissionError(
                "StateReconciler refuses execution on PostgreSQL role with SUPERUSER or BYPASSRLS."
            )

    @staticmethod
    def _validate_domain_id(domain_id: str) -> str:
        cleaned = domain_id.strip()
        if not cleaned or not _DOMAIN_SLUG_RE.fullmatch(cleaned):
            raise ValueError(
                f"Invalid domain_id {domain_id!r}; "
                f"expected lowercase slug {_DOMAIN_SLUG_RE.pattern}."
            )
        return cleaned

    @staticmethod
    def _infer_store_scheme(store: ArtifactObjectStore) -> str:
        if isinstance(store, GCSObjectStore):
            return "gs"
        if isinstance(store, S3CompatibleObjectStore):
            return "s3"
        if isinstance(store, InMemoryObjectStore):
            return "memory"
        cls_name = type(store).__name__.lower()
        if "gcs" in cls_name:
            return "gs"
        if "s3" in cls_name:
            return "s3"
        return "memory"

    @staticmethod
    def _assert_store_bucket_matches(
        store: ArtifactObjectStore,
        expected_bucket: str,
        storage_uri: str,
    ) -> ArtifactObjectStore:
        configured_bucket = str(
            getattr(store, "bucket", None) or getattr(store, "_bucket", None) or ""
        ).strip()
        if configured_bucket and configured_bucket != expected_bucket:
            raise StateReconcilerError(
                f"Storage bucket mismatch for {storage_uri!r}: store configured for "
                f"'{configured_bucket}', URI requested '{expected_bucket}'."
            )
        return store

    def _resolve_object_store_for_uri(self, storage_uri: str) -> ArtifactObjectStore:
        parsed = urlparse(storage_uri.strip())
        scheme = parsed.scheme.lower()
        bucket = parsed.netloc.strip()
        if scheme not in _ALLOWED_SCHEMES:
            raise StateReconcilerError(
                f"Unsupported storage_uri scheme '{scheme}://' for {storage_uri!r}; "
                "expected 'gs://' (GCSObjectStore) or 's3://' (S3CompatibleObjectStore)."
            )
        if not bucket or "/" in bucket or "\\" in bucket or bucket in {".", ".."}:
            raise StateReconcilerError(f"Missing or invalid bucket in storage_uri: {storage_uri!r}")

        if scheme in self._stores_by_scheme:
            return self._assert_store_bucket_matches(
                self._stores_by_scheme[scheme], bucket, storage_uri
            )

        if self._default_object_store is not None:
            if scheme == "gs" and isinstance(self._default_object_store, GCSObjectStore):
                return self._assert_store_bucket_matches(
                    self._default_object_store, bucket, storage_uri
                )
            if scheme == "s3" and isinstance(self._default_object_store, S3CompatibleObjectStore):
                return self._assert_store_bucket_matches(
                    self._default_object_store, bucket, storage_uri
                )
            if scheme == "memory" and isinstance(self._default_object_store, InMemoryObjectStore):
                return self._assert_store_bucket_matches(
                    self._default_object_store, bucket, storage_uri
                )

        cache_key = (scheme, bucket)
        if cache_key in self._dynamic_bucket_stores:
            return self._dynamic_bucket_stores[cache_key]

        if scheme == "gs":
            gcs_store = GCSObjectStore(
                bucket=bucket,
                project=self._settings.gcs_project or self._settings.gcp_project,
                emulator_host=self._settings.gcs_emulator_host,
            )
            self._dynamic_bucket_stores[cache_key] = gcs_store
            return gcs_store

        if scheme == "s3":
            s3_store = S3CompatibleObjectStore(
                bucket=bucket,
                endpoint_url=self._settings.s3_endpoint_url,
                region_name=self._settings.s3_region,
                aws_access_key_id=self._settings.s3_access_key_id,
                aws_secret_access_key=self._settings.s3_secret_access_key,
                force_path_style=self._settings.s3_force_path_style,
            )
            self._dynamic_bucket_stores[cache_key] = s3_store
            return s3_store

        raise StateReconcilerError(
            f"Unconfigured storage_uri scheme '{scheme}://' for {storage_uri!r}."
        )

    def _target_stores(self) -> list[tuple[str, ArtifactObjectStore]]:
        stores: list[tuple[str, ArtifactObjectStore]] = []
        seen_ids: set[int] = set()
        for scheme, store in sorted(self._stores_by_scheme.items()):
            if id(store) not in seen_ids:
                stores.append((scheme, store))
                seen_ids.add(id(store))
        if (
            self._default_object_store is not None
            and id(self._default_object_store) not in seen_ids
        ):
            stores.append(
                (self._infer_store_scheme(self._default_object_store), self._default_object_store)
            )
            seen_ids.add(id(self._default_object_store))
        return stores

    @staticmethod
    def _extract_key_from_uri(uri: str, expected_tenant_id: UUID) -> str | None:
        parsed = urlparse(uri.strip())
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            return None
        rel = parsed.path.lstrip("/")
        if not rel:
            return None
        return validate_object_key(rel, expected_tenant_id=expected_tenant_id)

    def _read_external_manifest_payload(
        self,
        store: ArtifactObjectStore,
        key: str,
        *,
        tenant_id: UUID,
        warnings: list[str],
        label: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            if not store.exists(key, expected_tenant_id=tenant_id):
                return None, None
            raw = store.get_bytes(key, expected_tenant_id=tenant_id)
        except Exception as exc:
            warnings.append(f"Failed reading {label} manifest '{key}': {type(exc).__name__}: {exc}")
            return None, None

        digest = hashlib.sha256(raw).hexdigest()
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            warnings.append(f"Corrupted JSON in {label} manifest '{key}': {type(exc).__name__}")
            return None, digest
        if not isinstance(parsed, dict):
            warnings.append(f"Unexpected root type in {label} manifest '{key}'.")
            return None, digest
        return parsed, digest

    def _reconcile_zoekt_client(
        self,
        zoekt: Any,
        *,
        principal: PrincipalContext,
        domain_id: str,
        active_repos: dict[str, RepositoryRecord],
        pg_files_by_key: dict[tuple[str, str], CodeFileRecord],
        warnings: list[str],
    ) -> tuple[int, int, int]:
        purged_repos = 0
        purged_files = 0
        reindexed_files = 0

        # Case 1: In-process LocalTrigramZoektIndex
        if hasattr(zoekt, "_docs") and hasattr(zoekt, "_lock"):
            with zoekt._lock:
                shard_docs = [
                    doc
                    for doc in zoekt._docs.values()
                    if doc.tenant_id == principal.tenant_id and doc.domain_id == domain_id
                ]
                shard_repos = {doc.repo_name for doc in shard_docs}
                acl_repos = {
                    rname
                    for (tid, did, rname) in getattr(zoekt.acl_registry, "_entries_by_key", {})
                    if tid == principal.tenant_id and did == domain_id
                }
                extra_repos = sorted((shard_repos | acl_repos) - set(active_repos))
                for extra_repo in extra_repos:
                    deleted = int(
                        zoekt.purge_repo(
                            tenant_id=principal.tenant_id,
                            domain_id=domain_id,
                            repo_name=extra_repo,
                        )
                    )
                    purged_repos += 1
                    purged_files += deleted

                for rname, repo_rec in sorted(active_repos.items()):
                    existing_rid = zoekt.acl_registry.get_repo_id(
                        principal.tenant_id, domain_id, rname
                    )
                    if existing_rid is None:
                        zoekt.register_repository(
                            tenant_id=principal.tenant_id,
                            domain_id=domain_id,
                            repo_name=rname,
                            allowed_scopes=repo_rec.allowed_scopes,
                        )
                    else:
                        entry = zoekt.acl_registry._entries_by_id.get(existing_rid)
                        if entry is not None and entry.allowed_scopes != frozenset(
                            repo_rec.allowed_scopes
                        ):
                            zoekt.acl_registry.update_repo_scopes(
                                principal.tenant_id,
                                domain_id,
                                rname,
                                repo_rec.allowed_scopes,
                            )

                existing_by_key: dict[tuple[str, str], Any] = {}
                extra_file_keys: set[tuple[str, str]] = set()
                for doc in shard_docs:
                    if doc.repo_name not in active_repos:
                        continue
                    key = (doc.repo_name, doc.file_path)
                    if key not in pg_files_by_key:
                        extra_file_keys.add(key)
                    elif doc.branch == "HEAD" or key not in existing_by_key:
                        existing_by_key[key] = doc

                for rname, fpath in sorted(extra_file_keys):
                    removed = int(
                        zoekt.delete_file_all_branches(
                            tenant_id=principal.tenant_id,
                            domain_id=domain_id,
                            repo_name=rname,
                            file_path=fpath,
                        )
                    )
                    purged_files += removed

                for key, cf in sorted(pg_files_by_key.items()):
                    existing_doc = existing_by_key.get(key)
                    needs_reindex = (
                        existing_doc is None
                        or existing_doc.checksum != cf.git_blob_sha
                        or existing_doc.content != cf.raw_content
                        or existing_doc.allowed_scopes != frozenset(cf.allowed_scopes)
                        or abs(float(existing_doc.ranking_weight) - float(cf.ranking_weight)) > 1e-6
                    )
                    if needs_reindex:
                        zoekt.index_file(
                            tenant_id=principal.tenant_id,
                            domain_id=domain_id,
                            repo_name=cf.repo_name,
                            file_path=cf.file_path,
                            content=cf.raw_content,
                            allowed_scopes=cf.allowed_scopes,
                            language=cf.language,
                            branch="HEAD",
                            ranking_weight=cf.ranking_weight,
                            is_generated=cf.is_generated,
                            is_vendored=cf.is_vendored,
                            checksum=cf.git_blob_sha,
                        )
                        reindexed_files += 1

            return purged_repos, purged_files, reindexed_files

        # Case 2: Remote ZoektGrpcJsonClient (HTTP /api/list, /api/admin/purge, /api/admin/reindex)
        if hasattr(zoekt, "endpoint_url") and hasattr(zoekt, "_client"):
            remote_repos: set[str] = set()
            remote_files: dict[tuple[str, str], str] = {}
            list_supported = False
            try:
                list_resp = zoekt._client.post(
                    f"{zoekt.endpoint_url}/api/list",
                    json={"tenant_id": str(principal.tenant_id), "domain_id": domain_id},
                )
                if 200 <= list_resp.status_code < 300:
                    list_supported = True
                    payload = list_resp.json() if list_resp.content else {}
                    for r in payload.get("repos") or []:
                        if str(r).strip():
                            remote_repos.add(str(r).strip())
                    for item in payload.get("files") or []:
                        if not isinstance(item, Mapping):
                            continue
                        rname = str(item.get("repo_name") or item.get("Repository") or "").strip()
                        fpath = str(item.get("file_path") or item.get("FileName") or "").strip()
                        csum = str(item.get("checksum") or item.get("Checksum") or "").strip()
                        if rname and fpath:
                            remote_repos.add(rname)
                            remote_files[(rname, fpath)] = csum
                elif list_resp.status_code != 404:
                    warnings.append(
                        f"Remote Zoekt /api/list returned HTTP {list_resp.status_code}."
                    )
            except httpx.HTTPError as exc:
                warnings.append(f"Remote Zoekt /api/list failed ({type(exc).__name__}): {exc}")

            acl_repos = {
                rname
                for (tid, did, rname) in getattr(zoekt.acl_registry, "_entries_by_key", {})
                if tid == principal.tenant_id and did == domain_id
            }
            for extra_repo in sorted((remote_repos | acl_repos) - set(active_repos)):
                zoekt.purge_repo(
                    tenant_id=principal.tenant_id,
                    domain_id=domain_id,
                    repo_name=extra_repo,
                )
                purged_repos += 1
                purged_files += sum(1 for (rn, _fp) in remote_files if rn == extra_repo)

            for rname, repo_rec in sorted(active_repos.items()):
                existing_rid = zoekt.acl_registry.get_repo_id(principal.tenant_id, domain_id, rname)
                if existing_rid is None:
                    zoekt.acl_registry.register_repo(
                        principal.tenant_id, domain_id, rname, repo_rec.allowed_scopes
                    )
                else:
                    zoekt.acl_registry.update_repo_scopes(
                        principal.tenant_id, domain_id, rname, repo_rec.allowed_scopes
                    )

            extra_remote_files = [
                (rn, fp)
                for (rn, fp) in sorted(remote_files)
                if rn in active_repos and (rn, fp) not in pg_files_by_key
            ]
            files_to_upsert = [
                cf
                for (rn, fp), cf in sorted(pg_files_by_key.items())
                if (rn, fp) not in remote_files or remote_files[(rn, fp)] != cf.git_blob_sha
            ]
            if (extra_remote_files or files_to_upsert) and (list_supported or pg_files_by_key):
                try:
                    reindex_resp = zoekt._client.post(
                        f"{zoekt.endpoint_url}/api/admin/reindex",
                        json={
                            "tenant_id": str(principal.tenant_id),
                            "domain_id": domain_id,
                            "active_repos": sorted(active_repos),
                            "delete_files": [
                                {"repo_name": rn, "file_path": fp} for rn, fp in extra_remote_files
                            ],
                            "upsert_files": [
                                {
                                    "repo_name": cf.repo_name,
                                    "file_path": cf.file_path,
                                    "checksum": cf.git_blob_sha,
                                    "content": cf.raw_content,
                                    "language": cf.language,
                                    "allowed_scopes": cf.allowed_scopes,
                                    "ranking_weight": cf.ranking_weight,
                                }
                                for cf in files_to_upsert
                            ],
                        },
                    )
                    if 200 <= reindex_resp.status_code < 300:
                        purged_files += len(extra_remote_files)
                        reindexed_files += len(files_to_upsert)
                    else:
                        warnings.append(
                            f"Remote Zoekt /api/admin/reindex returned HTTP "
                            f"{reindex_resp.status_code}."
                        )
                except httpx.HTTPError as exc:
                    warnings.append(
                        f"Remote Zoekt /api/admin/reindex failed ({type(exc).__name__}): {exc}"
                    )

        return purged_repos, purged_files, reindexed_files

    def _resolve_embedder_for_space(self, space: EmbeddingSpaceRecord) -> EmbeddingProviderSPI:
        if self.embedder is not None:
            return self.embedder
        return get_embedding_provider(
            settings=self._settings,
            provider_backend=space.provider,
            model_id=space.model_id,
            dimensions=space.dimensions,
        )

    def _reconcile_embeddings_pg(
        self,
        conn: Any,
        *,
        principal: PrincipalContext,
        domain_id: str,
        warnings: list[str],
    ) -> EmbeddingWatermarkDrift:
        pg_repo = self.repository
        assert isinstance(pg_repo, PostgresRLSRepository)

        spaces = pg_repo.list_embedding_spaces(conn, domain_id)
        registered_ids = [s.space_id for s in spaces]
        active_space = pg_repo.get_active_embedding_space(conn, domain_id)

        # 1. Clear orphaned shadow vectors pointing to embedding_spaces absent after PITR
        orphaned_cleared = 0
        for tbl in ("ast_symbols", "doc_chunks"):
            if registered_ids:
                res = conn.execute(
                    f"""
                    UPDATE {tbl}
                    SET shadow_embedding_space_id = NULL,
                        shadow_embedding_halfvec = NULL
                    WHERE domain_id = %s
                      AND shadow_embedding_space_id IS NOT NULL
                      AND NOT (shadow_embedding_space_id = ANY(%s::text[]));
                    """,
                    (domain_id, registered_ids),
                )
            else:
                res = conn.execute(
                    f"""
                    UPDATE {tbl}
                    SET shadow_embedding_space_id = NULL,
                        shadow_embedding_halfvec = NULL
                    WHERE domain_id = %s
                      AND shadow_embedding_space_id IS NOT NULL;
                    """,
                    (domain_id,),
                )
            orphaned_cleared += max(int(res.rowcount or 0), 0)

        # Ensure an active space exists if vector rows exist
        if active_space is None:
            sym_cnt_row = conn.execute(
                "SELECT COUNT(*) FROM ast_symbols WHERE domain_id = %s;", (domain_id,)
            ).fetchone()
            chk_cnt_row = conn.execute(
                "SELECT COUNT(*) FROM doc_chunks WHERE domain_id = %s AND NOT is_stale;",
                (domain_id,),
            ).fetchone()
            has_rows = int(str((sym_cnt_row or (0,))[0])) + int(str((chk_cnt_row or (0,))[0])) > 0
            if not has_rows:
                return EmbeddingWatermarkDrift(
                    active_space_id=None,
                    orphaned_shadow_rows_cleared=orphaned_cleared,
                )
            if spaces:
                active_space = pg_repo.register_embedding_space(
                    conn,
                    EmbeddingSpaceRecord(
                        tenant_id=principal.tenant_id,
                        domain_id=domain_id,
                        space_id=spaces[0].space_id,
                        model_id=spaces[0].model_id,
                        provider=spaces[0].provider,
                        dimensions=spaces[0].dimensions,
                        quantization=spaces[0].quantization,
                        status="active",
                        is_active=True,
                        index_operator_class=spaces[0].index_operator_class,
                    ),
                )
            else:
                default_dim = (
                    self.embedder.dimensions
                    if self.embedder is not None
                    else self._settings.embedding_dimensions
                )
                default_model = (
                    self.embedder.model_id
                    if self.embedder is not None
                    else DEFAULT_GOOGLE_EMBEDDING_MODEL
                )
                active_space = pg_repo.register_embedding_space(
                    conn,
                    EmbeddingSpaceRecord(
                        tenant_id=principal.tenant_id,
                        domain_id=domain_id,
                        space_id=DEFAULT_EMBEDDING_SPACE_ID,
                        model_id=default_model,
                        dimensions=default_dim,
                        status="active",
                        is_active=True,
                    ),
                )

        # 2. Detect drifted rows (embedding_space_id mismatch, NULL halfvec, or wrong dims)
        sym_rows = conn.execute(
            """
            SELECT symbol_id, qualified_name, signature_skeleton, contextual_summary
            FROM ast_symbols
            WHERE domain_id = %s
              AND (
                  COALESCE(embedding_space_id, '') <> %s
                  OR embedding_halfvec IS NULL
                  OR vector_dims(embedding_halfvec) <> %s
              )
            ORDER BY symbol_id;
            """,
            (domain_id, active_space.space_id, active_space.dimensions),
        ).fetchall()

        chk_rows = conn.execute(
            """
            SELECT chunk_id, doc_title, section_heading, contextual_header, verbatim_content
            FROM doc_chunks
            WHERE domain_id = %s
              AND NOT is_stale
              AND (
                  COALESCE(embedding_space_id, '') <> %s
                  OR embedding_halfvec IS NULL
                  OR vector_dims(embedding_halfvec) <> %s
              )
            ORDER BY chunk_id;
            """,
            (domain_id, active_space.space_id, active_space.dimensions),
        ).fetchall()

        if not sym_rows and not chk_rows:
            return EmbeddingWatermarkDrift(
                active_space_id=active_space.space_id,
                orphaned_shadow_rows_cleared=orphaned_cleared,
            )

        drifted_syms = len(sym_rows)
        drifted_chks = len(chk_rows)
        sym_reembedded = 0
        chk_reembedded = 0
        unhealed = 0

        try:
            embedder = self._resolve_embedder_for_space(active_space)
            if sym_rows:
                sym_texts = [f"{r[1]} {r[2]} {r[3]}".strip() for r in sym_rows]
                sym_vecs = embedder.embed_batch(sym_texts, is_query=False)
                if len(sym_vecs) != len(sym_rows):
                    raise RuntimeError(
                        f"Embedder returned {len(sym_vecs)} vectors for {len(sym_rows)} symbols."
                    )
                sym_updates: list[tuple[str, str, str, str, UUID]] = []
                for row, vec in zip(sym_rows, sym_vecs, strict=True):
                    checked = validate_non_zero_embedding(vec, expected_dim=active_space.dimensions)
                    vec768_lit, hv_lit = _format_db_vector_and_halfvec(checked)
                    sym_updates.append(
                        (vec768_lit, active_space.space_id, hv_lit, domain_id, UUID(str(row[0])))
                    )
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        UPDATE ast_symbols
                        SET embedding = %s::vector,
                            embedding_space_id = %s,
                            embedding_halfvec = %s::halfvec
                        WHERE domain_id = %s AND symbol_id = %s;
                        """,
                        sym_updates,
                    )
                    sym_reembedded = max(int(cur.rowcount or 0), len(sym_updates))

            if chk_rows:
                chk_texts = [f"{r[1]} {r[2]} {r[3]} {r[4]}".strip() for r in chk_rows]
                chk_vecs = embedder.embed_batch(chk_texts, is_query=False)
                if len(chk_vecs) != len(chk_rows):
                    raise RuntimeError(
                        f"Embedder returned {len(chk_vecs)} vectors for {len(chk_rows)} chunks."
                    )
                chk_updates: list[tuple[str, str, str, str, UUID]] = []
                for row, vec in zip(chk_rows, chk_vecs, strict=True):
                    checked = validate_non_zero_embedding(vec, expected_dim=active_space.dimensions)
                    vec768_lit, hv_lit = _format_db_vector_and_halfvec(checked)
                    chk_updates.append(
                        (vec768_lit, active_space.space_id, hv_lit, domain_id, UUID(str(row[0])))
                    )
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        UPDATE doc_chunks
                        SET embedding = %s::vector,
                            embedding_space_id = %s,
                            embedding_halfvec = %s::halfvec
                        WHERE domain_id = %s AND chunk_id = %s;
                        """,
                        chk_updates,
                    )
                    chk_reembedded = max(int(cur.rowcount or 0), len(chk_updates))
        except Exception as exc:
            unhealed = (drifted_syms - sym_reembedded) + (drifted_chks - chk_reembedded)
            warnings.append(
                f"Fail-closed embedding reconciliation deferred ({type(exc).__name__}): {exc}"
            )
            if self.task_queue is not None:
                self.task_queue.enqueue_task(
                    domain_id=domain_id,
                    event_type="reconcile_sweep",
                    payload={
                        "reason": "pitr_reembed_domain",
                        "domain_id": domain_id,
                        "space_id": active_space.space_id,
                        "unhealed_rows": unhealed,
                    },
                    priority=20,
                    tenant_id=principal.tenant_id,
                )

        return EmbeddingWatermarkDrift(
            active_space_id=active_space.space_id,
            orphaned_shadow_rows_cleared=orphaned_cleared,
            drifted_symbol_rows_found=drifted_syms,
            drifted_chunk_rows_found=drifted_chks,
            symbol_rows_reembedded=sym_reembedded,
            chunk_rows_reembedded=chk_reembedded,
            unhealed_rows=unhealed,
        )

    def _reconcile_embeddings_inmemory(
        self,
        *,
        principal: PrincipalContext,
        domain_id: str,
        warnings: list[str],
    ) -> EmbeddingWatermarkDrift:
        import dataclasses

        mem_repo = self.repository
        assert isinstance(mem_repo, InMemoryRLSRepository)

        spaces = mem_repo.list_embedding_spaces(principal, domain_id)
        registered_ids = {s.space_id for s in spaces}
        active_space = mem_repo.get_active_embedding_space(principal, domain_id)

        orphaned_cleared = 0
        with mem_repo._lock:
            domain_sym_ids = {
                s.symbol_id
                for s in mem_repo._symbols.values()
                if s.tenant_id == principal.tenant_id and s.domain_id == domain_id
            }
            domain_doc_ids = {
                d.chunk_id
                for d in mem_repo._docs.values()
                if d.tenant_id == principal.tenant_id and d.domain_id == domain_id
            }
            for key in list(mem_repo._shadow_symbol_vecs):
                tid, sid, sp_id = key
                if (
                    tid == principal.tenant_id
                    and sid in domain_sym_ids
                    and sp_id not in registered_ids
                ):
                    mem_repo._shadow_symbol_vecs.pop(key, None)
                    orphaned_cleared += 1
            for key in list(mem_repo._shadow_doc_vecs):
                tid, cid, sp_id = key
                if (
                    tid == principal.tenant_id
                    and cid in domain_doc_ids
                    and sp_id not in registered_ids
                ):
                    mem_repo._shadow_doc_vecs.pop(key, None)
                    orphaned_cleared += 1

        syms = mem_repo.list_visible_ast_symbols(principal, domain_id)
        chks = [c for c in mem_repo.list_visible_doc_chunks(principal, domain_id) if not c.is_stale]
        if active_space is None:
            if not syms and not chks:
                return EmbeddingWatermarkDrift(
                    active_space_id=None,
                    orphaned_shadow_rows_cleared=orphaned_cleared,
                )
            if spaces:
                active_space = mem_repo.register_embedding_space(
                    principal,
                    dataclasses.replace(spaces[0], status="active", is_active=True),
                )
            else:
                active_space = mem_repo.register_embedding_space(
                    principal,
                    EmbeddingSpaceRecord(
                        tenant_id=principal.tenant_id,
                        domain_id=domain_id,
                        space_id=DEFAULT_EMBEDDING_SPACE_ID,
                        model_id=(
                            self.embedder.model_id
                            if self.embedder is not None
                            else DEFAULT_GOOGLE_EMBEDDING_MODEL
                        ),
                        dimensions=(
                            self.embedder.dimensions
                            if self.embedder is not None
                            else self._settings.embedding_dimensions
                        ),
                        status="active",
                        is_active=True,
                    ),
                )

        def _is_drifted_vec(vec: Sequence[float], space: EmbeddingSpaceRecord) -> bool:
            return len(vec) != space.dimensions or sum(float(x) * float(x) for x in vec) <= 1e-9

        drifted_sym_list = [
            s
            for s in syms
            if s.embedding_space_id != active_space.space_id
            or _is_drifted_vec(s.embedding, active_space)
        ]
        drifted_chk_list = [
            c
            for c in chks
            if c.embedding_space_id != active_space.space_id
            or _is_drifted_vec(c.embedding, active_space)
        ]
        if not drifted_sym_list and not drifted_chk_list:
            return EmbeddingWatermarkDrift(
                active_space_id=active_space.space_id,
                orphaned_shadow_rows_cleared=orphaned_cleared,
            )

        sym_reembedded = 0
        chk_reembedded = 0
        unhealed = 0
        try:
            embedder = self._resolve_embedder_for_space(active_space)
            if drifted_sym_list:
                sym_vecs = embedder.embed_batch(
                    [
                        f"{s.qualified_name} {s.signature_skeleton} {s.contextual_summary}".strip()
                        for s in drifted_sym_list
                    ],
                    is_query=False,
                )
                with mem_repo._lock:
                    for sym, vec in zip(drifted_sym_list, sym_vecs, strict=True):
                        checked = validate_non_zero_embedding(
                            vec, expected_dim=active_space.dimensions
                        )
                        mem_repo._symbols[sym.symbol_id] = dataclasses.replace(
                            sym,
                            embedding=checked,
                            embedding_space_id=active_space.space_id,
                        )
                        sym_reembedded += 1
            if drifted_chk_list:
                chk_vecs = embedder.embed_batch(
                    [
                        (
                            f"{c.doc_title} {c.section_heading} "
                            f"{c.contextual_header} {c.verbatim_content}"
                        ).strip()
                        for c in drifted_chk_list
                    ],
                    is_query=False,
                )
                with mem_repo._lock:
                    for chk, vec in zip(drifted_chk_list, chk_vecs, strict=True):
                        checked = validate_non_zero_embedding(
                            vec, expected_dim=active_space.dimensions
                        )
                        mem_repo._docs[chk.chunk_id] = dataclasses.replace(
                            chk,
                            embedding=checked,
                            embedding_space_id=active_space.space_id,
                        )
                        chk_reembedded += 1
        except Exception as exc:
            unhealed = (len(drifted_sym_list) - sym_reembedded) + (
                len(drifted_chk_list) - chk_reembedded
            )
            warnings.append(
                f"Fail-closed embedding reconciliation deferred ({type(exc).__name__}): {exc}"
            )
            if self.task_queue is not None:
                self.task_queue.enqueue_task(
                    domain_id=domain_id,
                    event_type="reconcile_sweep",
                    payload={
                        "reason": "pitr_reembed_domain",
                        "domain_id": domain_id,
                        "space_id": active_space.space_id,
                        "unhealed_rows": unhealed,
                    },
                    priority=20,
                    tenant_id=principal.tenant_id,
                )

        return EmbeddingWatermarkDrift(
            active_space_id=active_space.space_id,
            orphaned_shadow_rows_cleared=orphaned_cleared,
            drifted_symbol_rows_found=len(drifted_sym_list),
            drifted_chunk_rows_found=len(drifted_chk_list),
            symbol_rows_reembedded=sym_reembedded,
            chunk_rows_reembedded=chk_reembedded,
            unhealed_rows=unhealed,
        )

    def _verify_model_artifact_integrity(
        self,
        artifact: ModelArtifactRecord,
        *,
        principal: PrincipalContext,
    ) -> tuple[bool, str]:
        if artifact.tenant_id != principal.tenant_id:
            return False, "cross_tenant_artifact"
        try:
            weights_key = validate_object_key(
                artifact.weights_object_key, expected_tenant_id=principal.tenant_id
            )
            config_key = validate_object_key(
                artifact.config_object_key, expected_tenant_id=principal.tenant_id
            )
            sig_key = (
                validate_object_key(
                    artifact.signature_object_key, expected_tenant_id=principal.tenant_id
                )
                if artifact.signature_object_key.strip()
                else ""
            )
            store = self._resolve_object_store_for_uri(artifact.storage_uri)
            if not store.exists(weights_key, expected_tenant_id=principal.tenant_id):
                return False, f"missing_weights:{weights_key}"
            if not store.exists(config_key, expected_tenant_id=principal.tenant_id):
                return False, f"missing_config:{config_key}"
            if sig_key and not store.exists(sig_key, expected_tenant_id=principal.tenant_id):
                return False, f"missing_signature:{sig_key}"

            weights_bytes = store.get_bytes(weights_key, expected_tenant_id=principal.tenant_id)
            actual_sha = hashlib.sha256(weights_bytes).hexdigest()
            if actual_sha != artifact.weights_sha256.strip().lower():
                return False, f"weights_sha256_mismatch:{actual_sha}"

            cfg_bytes = store.get_bytes(config_key, expected_tenant_id=principal.tenant_id)
            expected_cfg_sha = (
                str(artifact.metadata_json.get("config_sha256") or "").strip().lower()
            )
            if expected_cfg_sha and hashlib.sha256(cfg_bytes).hexdigest() != expected_cfg_sha:
                return False, "config_sha256_mismatch"
            cfg_obj = json.loads(cfg_bytes.decode("utf-8"))
            if not isinstance(cfg_obj, Mapping):
                return False, "invalid_config_json"

            if self.adapter_sidecar is not None:
                sig_text = artifact.cosign_signature_b64.strip()
                if sig_key:
                    sig_text = (
                        store.get_bytes(sig_key, expected_tenant_id=principal.tenant_id)
                        .decode("utf-8")
                        .strip()
                    )
                if not sig_text:
                    return False, "missing_cosign_signature"
                self.adapter_sidecar.cosign_verifier.verify(
                    weights_bytes,
                    sig_text,
                    key_id=artifact.cosign_key_id or "default",
                )
            return True, "ok"
        except Exception as exc:
            return False, f"{type(exc).__name__}:{exc}"

    def _sweep_stranded_pod_adapters(
        self,
        *,
        sidecar: AdapterSyncSidecar,
        valid_active_ids: set[str],
        warnings: list[str],
    ) -> int:
        """Unloads any post-PITR stranded adapters on vLLM pods absent from active DB state."""
        unloaded_count = 0
        headers = sidecar._vllm_headers()
        stranded_names: set[str] = set()
        for pod_v1 in sidecar.vllm_pod_endpoints:
            try:
                resp = sidecar._http_client.get(f"{pod_v1}/models", headers=headers)
                if resp.status_code != 200:
                    continue
                payload = resp.json() if resp.content else {}
                for item in payload.get("data") or []:
                    if not isinstance(item, Mapping):
                        continue
                    mid = str(item.get("id") or "").strip()
                    if not mid or mid == sidecar.expected_base_model_id:
                        continue
                    if mid not in valid_active_ids:
                        stranded_names.add(mid)
            except Exception as exc:
                warnings.append(
                    f"Failed querying vLLM pod '{pod_v1}/models' ({type(exc).__name__}): {exc}"
                )

        for lora_name in sorted(stranded_names):
            sidecar._unload_across_all_vllm_pods(lora_name)
            unloaded_count += 1
        return unloaded_count

    def _reconcile_adapters(
        self,
        *,
        principal: PrincipalContext,
        domain_id: str,
        is_pg: bool,
        warnings: list[str],
    ) -> AdapterWatermarkDrift:
        def _fetch_all_domain_artifacts() -> list[ModelArtifactRecord]:
            if is_pg:
                assert self.connection_manager is not None
                pg_repo = self.repository
                assert isinstance(pg_repo, PostgresRLSRepository)
                with self.connection_manager.scoped_transaction(
                    principal, enable_hnsw_iterative_scan=False
                ) as conn:
                    return pg_repo.list_model_artifacts(conn, domain_id)
            mem_repo = self.repository
            assert isinstance(mem_repo, InMemoryRLSRepository)
            return mem_repo.list_model_artifacts(principal, domain_id)

        all_artifacts = _fetch_all_domain_artifacts()
        active_before = [a for a in all_artifacts if a.is_active]
        active_before_id = active_before[0].artifact_id if active_before else None

        retired_broken: list[str] = []
        promoted_id: str | None = None

        for active_art in active_before:
            ok, reason = self._verify_model_artifact_integrity(active_art, principal=principal)
            if ok:
                continue
            warnings.append(
                f"Active model_artifact '{active_art.artifact_id}' failed integrity check "
                f"({reason}); retiring fail-closed."
            )
            if is_pg:
                assert self.connection_manager is not None
                pg_repo = self.repository
                assert isinstance(pg_repo, PostgresRLSRepository)
                with self.connection_manager.scoped_transaction(
                    principal, enable_hnsw_iterative_scan=False
                ) as conn:
                    pg_repo.retire_model_artifact(conn, domain_id, active_art.artifact_id)
            else:
                mem_repo = self.repository
                assert isinstance(mem_repo, InMemoryRLSRepository)
                mem_repo.retire_model_artifact(principal, domain_id, active_art.artifact_id)
            retired_broken.append(active_art.artifact_id)

            if self.promote_last_known_good_adapter:
                for cand in all_artifacts:
                    if (
                        cand.artifact_id == active_art.artifact_id
                        or cand.artifact_kind != active_art.artifact_kind
                        or cand.status == "revoked"
                    ):
                        continue
                    cand_ok, _cand_reason = self._verify_model_artifact_integrity(
                        cand, principal=principal
                    )
                    if cand_ok:
                        if is_pg:
                            assert self.connection_manager is not None
                            pg_repo = self.repository
                            assert isinstance(pg_repo, PostgresRLSRepository)
                            with self.connection_manager.scoped_transaction(
                                principal, enable_hnsw_iterative_scan=False
                            ) as conn:
                                pg_repo.activate_model_artifact(
                                    conn,
                                    domain_id,
                                    cand.artifact_id,
                                    artifact_kind=cand.artifact_kind,
                                )
                        else:
                            mem_repo = self.repository
                            assert isinstance(mem_repo, InMemoryRLSRepository)
                            mem_repo.activate_model_artifact(
                                principal,
                                domain_id,
                                cand.artifact_id,
                                artifact_kind=cand.artifact_kind,
                            )
                        promoted_id = cand.artifact_id
                        break

        refreshed_artifacts = _fetch_all_domain_artifacts()
        active_after = [a for a in refreshed_artifacts if a.is_active]
        active_after_id = active_after[0].artifact_id if active_after else None

        if not active_after and self.adapter_router is not None and retired_broken:
            self.adapter_router.unregister_specialization(domain_id, tenant_id=principal.tenant_id)

        referenced_keys: set[str] = set()
        for art in refreshed_artifacts:
            if art.status == "revoked":
                continue
            for raw_k in (art.weights_object_key, art.config_object_key, art.signature_object_key):
                if raw_k and raw_k.strip():
                    referenced_keys.add(raw_k.strip())

        adapters_prefix = validate_object_prefix(
            f"tenants/{principal.tenant_id}/domains/{domain_id}/adapters/",
            expected_tenant_id=principal.tenant_id,
        )
        orphaned_keys: list[str] = []
        pruned_keys: list[str] = []

        target_stores = list(self._target_stores())
        seen_store_ids = {id(s) for _, s in target_stores}
        for art in refreshed_artifacts:
            try:
                art_store = self._resolve_object_store_for_uri(art.storage_uri)
                if id(art_store) not in seen_store_ids:
                    target_stores.append((self._infer_store_scheme(art_store), art_store))
                    seen_store_ids.add(id(art_store))
            except Exception as exc:
                logger.debug(
                    "Skipping dynamic store discovery for '%s' (%s): %s",
                    art.artifact_id,
                    type(exc).__name__,
                    exc,
                )

        for scheme_label, store in target_stores:
            try:
                keys = store.list_keys(adapters_prefix, expected_tenant_id=principal.tenant_id)
            except Exception as exc:
                warnings.append(
                    f"Failed listing adapter keys on {scheme_label} ({type(exc).__name__}): {exc}"
                )
                continue
            for obj_key in keys:
                if obj_key not in referenced_keys:
                    orphaned_keys.append(obj_key)
                    if self.prune_orphan_objects:
                        try:
                            store.delete(obj_key, expected_tenant_id=principal.tenant_id)
                            pruned_keys.append(obj_key)
                        except Exception as exc:
                            warnings.append(
                                f"Failed pruning orphan '{obj_key}' on {scheme_label}: {exc}"
                            )

        synced_pods = 0
        if self.adapter_sidecar is not None:
            try:
                sync_rep = self.adapter_sidecar.reconcile_once()
                for failed_aid, err_msg in sorted(sync_rep.failed_artifacts.items()):
                    warnings.append(f"AdapterSyncSidecar failed '{failed_aid}': {err_msg}")
                active_tenant_records = self.adapter_sidecar._fetch_active_artifacts_from_db()
                valid_active_ids = {r.artifact_id for r in active_tenant_records}
                self._sweep_stranded_pod_adapters(
                    sidecar=self.adapter_sidecar,
                    valid_active_ids=valid_active_ids,
                    warnings=warnings,
                )
                parity = self.adapter_sidecar.verify_multi_pod_parity()
                synced_pods = len(parity)
            except Exception as exc:
                warnings.append(
                    f"AdapterSyncSidecar reconciliation failed ({type(exc).__name__}): {exc}"
                )

        return AdapterWatermarkDrift(
            active_artifact_id_before=active_before_id,
            active_artifact_id_after=active_after_id,
            retired_broken_artifacts=retired_broken,
            promoted_artifact_id=promoted_id,
            orphaned_object_keys=sorted(set(orphaned_keys)),
            pruned_object_keys=sorted(set(pruned_keys)),
            sidecar_synced_pods=synced_pods,
        )

    def _compute_authoritative_pg_commits(
        self,
        *,
        repos: Sequence[RepositoryRecord],
        code_files: Sequence[CodeFileRecord],
        repo_commits_map: Mapping[UUID, Sequence[Any]],
    ) -> dict[str, str]:
        files_by_repo: dict[str, list[CodeFileRecord]] = {}
        for cf in code_files:
            files_by_repo.setdefault(cf.repo_name, []).append(cf)

        pg_repo_shas: dict[str, str] = {}
        for r in repos:
            if not r.is_active:
                continue
            commits = list(repo_commits_map.get(r.repo_id) or [])
            commit_shas_known = {str(c.commit_sha) for c in commits if getattr(c, "commit_sha", "")}
            r_files = files_by_repo.get(r.repo_name, [])
            file_commit_shas = [cf.git_commit_sha for cf in r_files if cf.git_commit_sha]

            resolved_sha = r.head_commit_sha.strip()
            if commits:
                latest_commit_sha = str(commits[0].commit_sha).strip()
                if not resolved_sha or (
                    resolved_sha not in commit_shas_known and resolved_sha not in file_commit_shas
                ):
                    resolved_sha = latest_commit_sha
            elif file_commit_shas:
                if not resolved_sha or resolved_sha not in file_commit_shas:
                    resolved_sha = file_commit_shas[0]
            pg_repo_shas[r.repo_name] = resolved_sha
        return pg_repo_shas

    def reconcile(
        self,
        principal: PrincipalContext,
        domain_id: str,
    ) -> StateReconciliationReport:
        """Reconciles PG17 state, Zoekt shards, GCS/S3 manifests, embeddings, and adapters."""
        t0 = time.perf_counter()
        clean_domain = self._validate_domain_id(domain_id)
        warnings: list[str] = []

        is_pg = isinstance(self.repository, PostgresRLSRepository)
        if is_pg:
            assert self.connection_manager is not None
            self._validate_non_admin_rls_manager(self.connection_manager, self._settings)

        # ------------------------------------------------------------------
        # Plane 1a: Load authoritative PG17 / InMemory domain state
        # ------------------------------------------------------------------
        repos: list[RepositoryRecord] = []
        repo_commits_map: dict[UUID, list[Any]] = {}
        code_files: list[CodeFileRecord] = []
        watermarks: list[ConnectorWatermarkRecord] = []
        existing_manifest: DomainSyncManifestRecord | None = None
        domain_pack: DomainPackManifest | None = None
        ast_symbols_count = 0
        ast_edges_count = 0
        doc_chunks_count = 0

        if is_pg:
            assert self.connection_manager is not None
            pg_repo = self.repository
            assert isinstance(pg_repo, PostgresRLSRepository)
            with self.connection_manager.scoped_transaction(
                principal, enable_hnsw_iterative_scan=False
            ) as conn:
                self._assert_pg_session_non_bypass_rls(conn)
                domain_pack = pg_repo.get_domain_pack(conn, clean_domain)
                repos = pg_repo.list_repositories(conn, clean_domain)
                for r in repos:
                    repo_commits_map[r.repo_id] = pg_repo.get_repo_commits(conn, r.repo_id)
                code_files = pg_repo.list_visible_code_file_records(conn, clean_domain, limit=10000)
                watermarks = pg_repo.list_connector_watermarks(conn, clean_domain)
                existing_manifest = pg_repo.get_domain_sync_manifest(conn, clean_domain)
                sym_r = conn.execute(
                    "SELECT COUNT(*) FROM ast_symbols WHERE domain_id = %s;", (clean_domain,)
                ).fetchone()
                edge_r = conn.execute(
                    "SELECT COUNT(*) FROM ast_edges WHERE domain_id = %s;", (clean_domain,)
                ).fetchone()
                doc_r = conn.execute(
                    "SELECT COUNT(*) FROM doc_chunks WHERE domain_id = %s AND NOT is_stale;",
                    (clean_domain,),
                ).fetchone()
                ast_symbols_count = int(str((sym_r or (0,))[0]))
                ast_edges_count = int(str((edge_r or (0,))[0]))
                doc_chunks_count = int(str((doc_r or (0,))[0]))
        else:
            mem_repo = self.repository
            assert isinstance(mem_repo, InMemoryRLSRepository)
            repos = mem_repo.list_repositories(principal, clean_domain)
            for r in repos:
                repo_commits_map[r.repo_id] = mem_repo.get_repo_commits(principal, r.repo_id)
            code_files = mem_repo.list_visible_code_file_records(
                principal, clean_domain, limit=10000
            )
            watermarks = mem_repo.list_connector_watermarks(principal, clean_domain)
            existing_manifest = mem_repo.get_domain_sync_manifest(principal, clean_domain)
            ast_symbols_count = len(mem_repo.list_visible_ast_symbols(principal, clean_domain))
            ast_edges_count = len(mem_repo.list_visible_ast_edges(principal, clean_domain))
            doc_chunks_count = len(
                [
                    d
                    for d in mem_repo.list_visible_doc_chunks(principal, clean_domain)
                    if not d.is_stale
                ]
            )

        active_repos = {r.repo_name: r for r in repos if r.is_active}
        pg_files_by_key: dict[tuple[str, str], CodeFileRecord] = {
            (cf.repo_name, cf.file_path): cf for cf in code_files if cf.repo_name in active_repos
        }
        pg_repo_shas = self._compute_authoritative_pg_commits(
            repos=repos,
            code_files=code_files,
            repo_commits_map=repo_commits_map,
        )

        # Align repositories.head_commit_sha if it pointed past surviving repo_commits after PITR
        for rname, expected_sha in sorted(pg_repo_shas.items()):
            rec = active_repos[rname]
            if expected_sha and rec.head_commit_sha != expected_sha:
                import dataclasses

                fixed_repo = dataclasses.replace(rec, head_commit_sha=expected_sha)
                if is_pg:
                    assert self.connection_manager is not None
                    assert isinstance(self.repository, PostgresRLSRepository)
                    with self.connection_manager.scoped_transaction(
                        principal, enable_hnsw_iterative_scan=False
                    ) as conn:
                        self.repository.upsert_repository(conn, fixed_repo)
                else:
                    assert isinstance(self.repository, InMemoryRLSRepository)
                    self.repository.upsert_repository(principal, fixed_repo)
                active_repos[rname] = fixed_repo

        # ------------------------------------------------------------------
        # Plane 1b: Compare & repair GCS/S3 sync manifests & connector watermarks
        # ------------------------------------------------------------------
        manifest_rel_key = build_tenant_domain_key(
            principal.tenant_id, clean_domain, "manifests/sync_manifest.json"
        )
        target_stores = self._target_stores()
        ext_commits_by_store: dict[str, dict[str, str]] = {}
        ext_digests_by_store: dict[str, str | None] = {}
        for scheme_label, store in target_stores:
            parsed_m, sha_m = self._read_external_manifest_payload(
                store,
                manifest_rel_key,
                tenant_id=principal.tenant_id,
                warnings=warnings,
                label=scheme_label,
            )
            ext_digests_by_store[scheme_label] = sha_m
            if parsed_m and isinstance(parsed_m.get("repo_commit_shas"), Mapping):
                ext_commits_by_store[scheme_label] = {
                    str(k): str(v) for k, v in parsed_m["repo_commit_shas"].items()
                }
            else:
                ext_commits_by_store[scheme_label] = {}

        if existing_manifest is not None and existing_manifest.s3_manifest_uri:
            custom_key = self._extract_key_from_uri(
                existing_manifest.s3_manifest_uri, principal.tenant_id
            )
            if custom_key and custom_key != manifest_rel_key:
                try:
                    c_store = self._resolve_object_store_for_uri(existing_manifest.s3_manifest_uri)
                    parsed_c, _ = self._read_external_manifest_payload(
                        c_store,
                        custom_key,
                        tenant_id=principal.tenant_id,
                        warnings=warnings,
                        label="registered_manifest",
                    )
                    if parsed_c and isinstance(parsed_c.get("repo_commit_shas"), Mapping):
                        ext_commits_by_store["registered_manifest"] = {
                            str(k): str(v) for k, v in parsed_c["repo_commit_shas"].items()
                        }
                except Exception as exc:
                    warnings.append(
                        f"Failed inspecting registered s3_manifest_uri "
                        f"'{existing_manifest.s3_manifest_uri}': {exc}"
                    )

        manifest_repo_shas = (
            dict(existing_manifest.repo_commit_shas) if existing_manifest is not None else {}
        )
        git_watermarks_by_repo: dict[str, list[ConnectorWatermarkRecord]] = {}
        for wm in watermarks:
            if wm.source_id in _GIT_SOURCE_IDS or wm.container_key in active_repos:
                git_watermarks_by_repo.setdefault(wm.container_key, []).append(wm)

        all_candidate_repos = sorted(
            set(pg_repo_shas)
            | set(manifest_repo_shas)
            | set(git_watermarks_by_repo)
            | {rn for commits_d in ext_commits_by_store.values() for rn in commits_d}
        )

        commit_drifts: list[CommitWatermarkDrift] = []
        watermarks_rewound = 0
        reconcile_tasks_enqueued = 0
        now_utc = datetime.now(UTC)
        files_count_by_repo: dict[str, int] = {}
        for cf in code_files:
            files_count_by_repo[cf.repo_name] = files_count_by_repo.get(cf.repo_name, 0) + 1

        for rname in all_candidate_repos:
            pg_sha = pg_repo_shas.get(rname, "")
            man_sha = manifest_repo_shas.get(rname)
            ext_shas = [
                commits_d[rname]
                for commits_d in ext_commits_by_store.values()
                if commits_d.get(rname)
            ]
            ext_sha = next((s for s in ext_shas if s != pg_sha), ext_shas[0] if ext_shas else None)
            repo_wms = git_watermarks_by_repo.get(rname, [])

            wm_cursors = [w.watermark_cursor for w in repo_wms]
            has_wm_mismatch = any(c != pg_sha for c in wm_cursors)
            has_manifest_mismatch = man_sha is not None and man_sha != pg_sha
            has_ext_mismatch = ext_sha is not None and ext_sha != pg_sha

            if not (has_wm_mismatch or has_manifest_mismatch or has_ext_mismatch):
                continue

            if repo_wms:
                for wm in repo_wms:
                    if wm.watermark_cursor != pg_sha or has_ext_mismatch:
                        rewound_rec = ConnectorWatermarkRecord(
                            tenant_id=principal.tenant_id,
                            domain_id=clean_domain,
                            source_id=wm.source_id,
                            container_key=wm.container_key,
                            watermark_cursor=pg_sha,
                            sync_tier="reconcile",
                            items_synced=files_count_by_repo.get(rname, 0),
                            last_webhook_at=wm.last_webhook_at,
                            last_watermark_sync_at=wm.last_watermark_sync_at,
                            last_reconciliation_at=now_utc,
                            metadata_json={
                                **dict(wm.metadata_json),
                                "pitr_reconciled": True,
                                "previous_cursor": wm.watermark_cursor,
                                "external_commit_sha": ext_sha,
                            },
                            updated_at=now_utc,
                        )
                        if is_pg:
                            assert self.connection_manager is not None
                            assert isinstance(self.repository, PostgresRLSRepository)
                            with self.connection_manager.scoped_transaction(
                                principal, enable_hnsw_iterative_scan=False
                            ) as conn:
                                self.repository.reset_connector_watermark(conn, rewound_rec)
                        else:
                            assert isinstance(self.repository, InMemoryRLSRepository)
                            self.repository.reset_connector_watermark(principal, rewound_rec)
                        watermarks_rewound += 1
            else:
                # Synthesize reconcile watermark anchor if drift existed in external manifest
                source_hint = (
                    active_repos[rname].source_type
                    if rname in active_repos and active_repos[rname].source_type in _GIT_SOURCE_IDS
                    else "local_git"
                )
                rewound_rec = ConnectorWatermarkRecord(
                    tenant_id=principal.tenant_id,
                    domain_id=clean_domain,
                    source_id=source_hint,
                    container_key=rname,
                    watermark_cursor=pg_sha,
                    sync_tier="reconcile",
                    items_synced=files_count_by_repo.get(rname, 0),
                    last_reconciliation_at=now_utc,
                    metadata_json={
                        "pitr_reconciled": True,
                        "previous_cursor": man_sha or "",
                        "external_commit_sha": ext_sha,
                    },
                    updated_at=now_utc,
                )
                if is_pg:
                    assert self.connection_manager is not None
                    assert isinstance(self.repository, PostgresRLSRepository)
                    with self.connection_manager.scoped_transaction(
                        principal, enable_hnsw_iterative_scan=False
                    ) as conn:
                        self.repository.reset_connector_watermark(conn, rewound_rec)
                else:
                    assert isinstance(self.repository, InMemoryRLSRepository)
                    self.repository.reset_connector_watermark(principal, rewound_rec)
                watermarks_rewound += 1

            enqueued_msg = ""
            if self.task_queue is not None:
                self.task_queue.enqueue_task(
                    domain_id=clean_domain,
                    event_type="reconcile_sweep",
                    payload={
                        "reason": "pitr_commit_drift",
                        "domain_id": clean_domain,
                        "repo_name": rname,
                        "pg_commit_sha": pg_sha,
                        "external_commit_sha": ext_sha,
                        "manifest_commit_sha": man_sha,
                    },
                    priority=20,
                    tenant_id=principal.tenant_id,
                )
                reconcile_tasks_enqueued += 1
                enqueued_msg = "_and_enqueued_reconcile"

            action_label = (
                f"watermark_rewound_to_{pg_sha or 'empty'}{enqueued_msg}"
                if rname in active_repos
                else f"watermark_rewound_orphan_repo{enqueued_msg}"
            )
            commit_drifts.append(
                CommitWatermarkDrift(
                    repo_name=rname,
                    pg_commit_sha=pg_sha,
                    manifest_commit_sha=man_sha,
                    object_store_commit_sha=ext_sha,
                    connector_watermark_cursor=wm_cursors[0] if wm_cursors else None,
                    action_taken=action_label,
                )
            )

        # Build canonical post-PITR sync_manifest.json and repair PG17 + gs:// + s3:// if drifted
        kind_val = (
            existing_manifest.kind
            if existing_manifest is not None
            else (
                domain_pack.authority_rules.primary_truth_source
                if domain_pack is not None
                else "code"
            )
        )
        source_dir_val = existing_manifest.source_dir if existing_manifest is not None else ""
        empty_docs_val = (
            dict(existing_manifest.empty_doc_shas) if existing_manifest is not None else {}
        )
        canonical_manifest_payload = {
            "domain_id": clean_domain,
            "tenant_id": str(principal.tenant_id),
            "kind": kind_val,
            "source_dir": source_dir_val,
            "repo_commit_shas": dict(sorted(pg_repo_shas.items())),
            "empty_doc_shas": empty_docs_val,
            "code_files_indexed": len(code_files),
            "ast_symbols_indexed": ast_symbols_count,
            "ast_edges_indexed": ast_edges_count,
            "doc_chunks_indexed": doc_chunks_count,
        }
        canonical_bytes = json.dumps(
            canonical_manifest_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        canonical_sha = hashlib.sha256(canonical_bytes).hexdigest()

        needs_pg_manifest_repair = (
            existing_manifest is None
            or existing_manifest.repo_commit_shas != pg_repo_shas
            or existing_manifest.code_files_indexed != len(code_files)
            or existing_manifest.ast_symbols_indexed != ast_symbols_count
            or existing_manifest.ast_edges_indexed != ast_edges_count
            or existing_manifest.doc_chunks_indexed != doc_chunks_count
            or existing_manifest.manifest_sha256 != canonical_sha
        )
        stores_needing_repair = [
            (scheme_label, store)
            for scheme_label, store in target_stores
            if ext_digests_by_store.get(scheme_label) != canonical_sha
        ]

        object_manifests_repaired: list[str] = []
        primary_manifest_uri: str | None = (
            existing_manifest.s3_manifest_uri if existing_manifest is not None else None
        )
        if stores_needing_repair or (needs_pg_manifest_repair and target_stores):
            for _scheme_label, store in target_stores:
                if (
                    ext_digests_by_store.get(_scheme_label) != canonical_sha
                    or needs_pg_manifest_repair
                ):
                    try:
                        meta = store.put_bytes(
                            manifest_rel_key,
                            canonical_bytes,
                            content_type="application/json",
                            expected_tenant_id=principal.tenant_id,
                        )
                        object_manifests_repaired.append(meta.uri)
                        if primary_manifest_uri is None or _scheme_label in {"gs", "s3"}:
                            primary_manifest_uri = meta.uri
                    except Exception as exc:
                        warnings.append(
                            f"Failed writing repaired sync_manifest on {_scheme_label}: {exc}"
                        )

        if needs_pg_manifest_repair or object_manifests_repaired:
            repaired_manifest_rec = DomainSyncManifestRecord(
                tenant_id=principal.tenant_id,
                domain_id=clean_domain,
                kind=kind_val,
                source_dir=source_dir_val,
                repo_commit_shas=dict(sorted(pg_repo_shas.items())),
                empty_doc_shas=empty_docs_val,
                code_files_indexed=len(code_files),
                ast_symbols_indexed=ast_symbols_count,
                ast_edges_indexed=ast_edges_count,
                doc_chunks_indexed=doc_chunks_count,
                s3_manifest_uri=primary_manifest_uri,
                manifest_sha256=canonical_sha,
                synced_at=now_utc,
            )
            if is_pg:
                assert self.connection_manager is not None
                assert isinstance(self.repository, PostgresRLSRepository)
                with self.connection_manager.scoped_transaction(
                    principal, enable_hnsw_iterative_scan=False
                ) as conn:
                    self.repository.upsert_domain_sync_manifest(conn, repaired_manifest_rec)
            else:
                assert isinstance(self.repository, InMemoryRLSRepository)
                self.repository.upsert_domain_sync_manifest(principal, repaired_manifest_rec)

        # ------------------------------------------------------------------
        # Plane 1c: Reconcile Zoekt trigram shards (Local & Remote gRPC-JSON)
        # ------------------------------------------------------------------
        zoekt_purged_repos = 0
        zoekt_purged_files = 0
        zoekt_reindexed_files = 0
        for zoekt in self.zoekt_clients:
            pr, pf, rf = self._reconcile_zoekt_client(
                zoekt,
                principal=principal,
                domain_id=clean_domain,
                active_repos=active_repos,
                pg_files_by_key=pg_files_by_key,
                warnings=warnings,
            )
            zoekt_purged_repos += pr
            zoekt_purged_files += pf
            zoekt_reindexed_files += rf

        # ------------------------------------------------------------------
        # Plane 2: Reconcile Embedding Watermarks (`embedding_spaces`)
        # ------------------------------------------------------------------
        if is_pg:
            assert self.connection_manager is not None
            with self.connection_manager.scoped_transaction(
                principal, enable_hnsw_iterative_scan=False
            ) as conn:
                embedding_drift = self._reconcile_embeddings_pg(
                    conn,
                    principal=principal,
                    domain_id=clean_domain,
                    warnings=warnings,
                )
        else:
            embedding_drift = self._reconcile_embeddings_inmemory(
                principal=principal,
                domain_id=clean_domain,
                warnings=warnings,
            )

        # ------------------------------------------------------------------
        # Plane 3: Reconcile Model Artifacts (`model_artifacts` vs GCS/S3 & vLLM)
        # ------------------------------------------------------------------
        adapter_drift = self._reconcile_adapters(
            principal=principal,
            domain_id=clean_domain,
            is_pg=is_pg,
            warnings=warnings,
        )

        # ------------------------------------------------------------------
        # Plane 4: Stale queue reclamation, permission cache invalidation, audit
        # ------------------------------------------------------------------
        stale_tasks_reclaimed = 0
        if self.task_queue is not None:
            stale_tasks_reclaimed = int(
                self.task_queue.reclaim_stale_tasks(
                    stale_threshold_seconds=self.stale_task_seconds,
                    tenant_id=principal.tenant_id,
                )
            )

        perm_invalidated = False
        if self.permission_cache is not None:
            if hasattr(self.permission_cache, "invalidate_tenant"):
                self.permission_cache.invalidate_tenant(principal.tenant_id)
                perm_invalidated = True
            elif hasattr(self.permission_cache, "invalidate_principal"):
                self.permission_cache.invalidate_principal(principal.tenant_id, principal.user_id)
                perm_invalidated = True
            elif hasattr(self.permission_cache, "clear"):
                self.permission_cache.clear()
                perm_invalidated = True

        drift_detected = bool(
            commit_drifts
            or zoekt_purged_repos > 0
            or zoekt_purged_files > 0
            or zoekt_reindexed_files > 0
            or watermarks_rewound > 0
            or object_manifests_repaired
            or embedding_drift.orphaned_shadow_rows_cleared > 0
            or embedding_drift.drifted_symbol_rows_found > 0
            or embedding_drift.drifted_chunk_rows_found > 0
            or adapter_drift.retired_broken_artifacts
            or adapter_drift.orphaned_object_keys
            or stale_tasks_reclaimed > 0
        )

        unhealed_markers = (
            "remote zoekt",
            "failed writing repaired sync_manifest",
            "failed pruning orphan",
            "failed listing adapter keys",
            "adaptersyncsidecar failed",
            "adaptersyncsidecar reconciliation failed",
            "fail-closed embedding reconciliation deferred",
        )
        healed = embedding_drift.unhealed_rows == 0 and not any(
            any(marker in w.lower() for marker in unhealed_markers) for w in warnings
        )

        elapsed_seconds = round(time.perf_counter() - t0, 4)

        get_audit_chain_service().record_event(
            principal=principal,
            domain_id=clean_domain,
            event_type="ops.state_reconciler.reconcile",
            severity="NOTICE" if drift_detected else "INFO",
            resource_uri=f"ops://state-reconciler/{clean_domain}",
            outcome="success" if healed else "failure",
            payload={
                "drift_detected": drift_detected,
                "healed": healed,
                "watermarks_rewound": watermarks_rewound,
                "zoekt_purged_repos": zoekt_purged_repos,
                "zoekt_purged_files": zoekt_purged_files,
                "zoekt_reindexed_files": zoekt_reindexed_files,
                "symbol_rows_reembedded": embedding_drift.symbol_rows_reembedded,
                "chunk_rows_reembedded": embedding_drift.chunk_rows_reembedded,
                "retired_broken_artifacts": adapter_drift.retired_broken_artifacts,
                "promoted_artifact_id": adapter_drift.promoted_artifact_id,
                "stale_tasks_reclaimed": stale_tasks_reclaimed,
            },
        )

        return StateReconciliationReport(
            domain_id=clean_domain,
            tenant_id=str(principal.tenant_id),
            drift_detected=drift_detected,
            healed=healed,
            commit_drifts=commit_drifts,
            zoekt_purged_repos=zoekt_purged_repos,
            zoekt_purged_files=zoekt_purged_files,
            zoekt_reindexed_files=zoekt_reindexed_files,
            watermarks_rewound=watermarks_rewound,
            reconcile_tasks_enqueued=reconcile_tasks_enqueued,
            stale_tasks_reclaimed=stale_tasks_reclaimed,
            object_manifests_repaired=object_manifests_repaired,
            embedding_drift=embedding_drift,
            adapter_drift=adapter_drift,
            permission_cache_invalidated=perm_invalidated,
            elapsed_seconds=elapsed_seconds,
            warnings=warnings,
        )


__all__ = [
    "AdapterWatermarkDrift",
    "CommitWatermarkDrift",
    "EmbeddingWatermarkDrift",
    "StateReconciler",
    "StateReconcilerError",
    "StateReconciliationReport",
]
