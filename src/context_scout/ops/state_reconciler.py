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
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from context_scout.agents.llm_backend import AdapterRouter
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    InMemoryRLSRepository,
    PostgresRLSRepository,
)
from context_scout.db.schema import (
    CodeFileRecord,
    ConnectorWatermarkRecord,
    DomainSyncManifestRecord,
    RepositoryRecord,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.ops.reconciler_pkg.embedding_reconciler import _EmbeddingReconcilerMixin
from context_scout.ops.reconciler_pkg.models import (
    _GIT_SOURCE_IDS as _GIT_SOURCE_IDS,
)
from context_scout.ops.reconciler_pkg.models import (
    AdapterWatermarkDrift as AdapterWatermarkDrift,
)
from context_scout.ops.reconciler_pkg.models import (
    CommitWatermarkDrift as CommitWatermarkDrift,
)
from context_scout.ops.reconciler_pkg.models import (
    EmbeddingWatermarkDrift as EmbeddingWatermarkDrift,
)
from context_scout.ops.reconciler_pkg.models import (
    StateReconcilerError as StateReconcilerError,
)
from context_scout.ops.reconciler_pkg.models import (
    StateReconciliationReport as StateReconciliationReport,
)
from context_scout.ops.reconciler_pkg.pg_drift_scanner import _PgDriftScannerMixin
from context_scout.ops.reconciler_pkg.zoekt_and_adapters import _ZoektAndAdaptersMixin
from context_scout.queue.task_queue import PostgresTaskQueue, TaskQueueProtocol
from context_scout.retrieval.embedder import EmbeddingProviderSPI
from context_scout.schemas.contracts import DomainPackManifest, PrincipalContext
from context_scout.security.audit_chain import get_audit_chain_service
from context_scout.serving.adapter_sync_sidecar import AdapterSyncSidecar
from context_scout.storage.object_store import (
    ArtifactObjectStore,
    build_tenant_domain_key,
)

logger = logging.getLogger(__name__)


class StateReconciler(
    _PgDriftScannerMixin,
    _EmbeddingReconcilerMixin,
    _ZoektAndAdaptersMixin,
):
    """Post-PITR multi-store reconciler for PG17/Cloud SQL, Zoekt, GCS/S3, and vLLM pods."""

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
        reconcile_batch_size: int = 64,
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
        self.reconcile_batch_size = max(1, int(reconcile_batch_size))

        if isinstance(self.task_queue, PostgresTaskQueue):
            self._validate_non_admin_rls_manager(self.task_queue._conn_mgr, self._settings)

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
