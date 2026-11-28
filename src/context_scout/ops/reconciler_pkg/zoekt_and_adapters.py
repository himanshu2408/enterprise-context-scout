"""Zoekt shard reconciliation, model artifact verification, and vLLM pod adapter sweep."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any

import httpx

from context_scout.db.repositories import (
    InMemoryRLSRepository,
    PostgresRLSRepository,
)
from context_scout.db.schema import (
    CodeFileRecord,
    ModelArtifactRecord,
    RepositoryRecord,
)
from context_scout.ops.reconciler_pkg.base import _StateReconcilerBase
from context_scout.ops.reconciler_pkg.models import (
    AdapterWatermarkDrift,
)
from context_scout.schemas.contracts import PrincipalContext
from context_scout.serving.adapter_sync_sidecar import AdapterSyncSidecar
from context_scout.storage.object_store import (
    validate_object_key,
    validate_object_prefix,
)

logger = logging.getLogger(__name__)


class _ZoektAndAdaptersMixin(_StateReconcilerBase):
    """Zoekt shard reconciliation, model artifact verification, and vLLM pod adapter sweep."""

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
            from context_scout.retrieval.zoekt_client import (
                format_zoekt_shard_repo_name,
                strip_zoekt_shard_repo_name,
            )

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
                        raw_r = str(r).strip()
                        if raw_r:
                            bare_r, matched = strip_zoekt_shard_repo_name(
                                principal.tenant_id, raw_r
                            )
                            if matched and bare_r:
                                remote_repos.add(bare_r)
                    for item in payload.get("files") or []:
                        if not isinstance(item, Mapping):
                            continue
                        raw_rn = str(item.get("repo_name") or item.get("Repository") or "").strip()
                        fpath = str(item.get("file_path") or item.get("FileName") or "").strip()
                        csum = str(item.get("checksum") or item.get("Checksum") or "").strip()
                        rname, matched = strip_zoekt_shard_repo_name(principal.tenant_id, raw_rn)
                        if matched and rname and fpath:
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
                            "active_repos": [
                                format_zoekt_shard_repo_name(principal.tenant_id, r)
                                for r in sorted(active_repos)
                            ],
                            "delete_files": [
                                {
                                    "repo_name": format_zoekt_shard_repo_name(
                                        principal.tenant_id, rn
                                    ),
                                    "file_path": fp,
                                }
                                for rn, fp in extra_remote_files
                            ],
                            "upsert_files": [
                                {
                                    "repo_name": format_zoekt_shard_repo_name(
                                        principal.tenant_id, cf.repo_name
                                    ),
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
