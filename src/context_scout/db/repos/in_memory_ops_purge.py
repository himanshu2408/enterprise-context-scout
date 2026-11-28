"""In-memory RLS repository mixin: telemetry, artifact purge, sessions, model artifacts."""

from __future__ import annotations

import dataclasses
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from context_scout.db.repos.in_memory_base import _InMemoryStateBase
from context_scout.db.repos.postgres_repo import PostgresRLSRepository
from context_scout.db.repos.types_and_helpers import (
    _scopes_intersect,
    normalize_purge_selectors,
)
from context_scout.db.schema import (
    ModelArtifactKind,
    ModelArtifactRecord,
    ModelArtifactStatus,
    ScoutSessionRecord,
)
from context_scout.schemas.contracts import (
    ExecutionTelemetryRecord,
    PrincipalContext,
)

logger = logging.getLogger(__name__)


class _InMemoryOpsAndPurgeMixin(_InMemoryStateBase):
    def record_execution_telemetry(
        self,
        conn_or_record: Any,
        maybe_record: ExecutionTelemetryRecord | None = None,
    ) -> None:
        if isinstance(conn_or_record, ExecutionTelemetryRecord):
            record = conn_or_record
        elif maybe_record is not None:
            record = maybe_record
        else:
            raise ValueError("record_execution_telemetry requires an ExecutionTelemetryRecord")
        self._telemetry[record.trace_id] = record

    def get_execution_telemetry(
        self,
        trace_id_or_conn: Any,
        maybe_trace_id: UUID | None = None,
    ) -> ExecutionTelemetryRecord | None:
        if isinstance(trace_id_or_conn, UUID):
            return self._telemetry.get(trace_id_or_conn)
        if maybe_trace_id is not None:
            return self._telemetry.get(maybe_trace_id)
        if isinstance(trace_id_or_conn, str):
            return self._telemetry.get(UUID(trace_id_or_conn))
        return None

    def get_principal_paid_tokens_consumed(
        self,
        principal_or_conn: Any,
        maybe_principal_id: str | None = None,
        since_hours: int = 24,
        *,
        tenant_id: UUID | None = None,
    ) -> int:
        resolved_tid: UUID | None = tenant_id
        if isinstance(principal_or_conn, PrincipalContext):
            principal_id = principal_or_conn.user_id
            resolved_tid = resolved_tid or principal_or_conn.tenant_id
        elif isinstance(principal_or_conn, str):
            principal_id = principal_or_conn
        elif maybe_principal_id is not None:
            principal_id = maybe_principal_id
        else:
            return 0
        return sum(
            rec.context_pack_tokens_sent
            for rec in self._telemetry.values()
            if rec.principal_id == principal_id
            and str(rec.execution_mode) == "gemini_handoff"
            and (resolved_tid is None or rec.tenant_id == resolved_tid)
        )

    def purge_domain_artifacts(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        doc_id: str | None = None,
        confluence_page_id: str | None = None,
        jira_issue_key: str | None = None,
        repo_name: str | None = None,
        file_path: str | None = None,
        commit_sha: str | None = None,
        blob_sha: str | None = None,
        include_derived_wiki: bool = True,
    ) -> dict[str, int]:
        norm = normalize_purge_selectors(
            domain_id,
            doc_id=doc_id,
            confluence_page_id=confluence_page_id,
            jira_issue_key=jira_issue_key,
            repo_name=repo_name,
            file_path=file_path,
            commit_sha=commit_sha,
            blob_sha=blob_sha,
        )
        domain_id = norm.domain_id
        doc_id = norm.doc_id
        confluence_page_id = norm.confluence_page_id
        jira_issue_key = norm.jira_issue_key
        repo_name = norm.repo_name
        file_path = norm.file_path
        commit_sha = norm.commit_sha
        blob_sha = norm.blob_sha
        counts: dict[str, int] = {
            "side_effects": 0,
            "overlay_file_deltas": 0,
            "ref_snapshots": 0,
            "ast_edges": 0,
            "ast_symbols": 0,
            "code_files": 0,
            "repo_commits": 0,
            "repositories": 0,
            "doc_chunks": 0,
            "confluence_pages": 0,
            "jira_issues": 0,
        }
        with self._lock:
            target_doc_id = doc_id
            if target_doc_id is None and confluence_page_id:
                target_doc_id = f"confluence:{confluence_page_id}"
            elif target_doc_id is None and jira_issue_key:
                target_doc_id = f"jira:{jira_issue_key}"

            if target_doc_id:
                del_doc_ids = [
                    cid
                    for cid, d in self._docs.items()
                    if d.tenant_id == principal.tenant_id
                    and d.domain_id == domain_id
                    and d.doc_id == target_doc_id
                    and _scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
                ]
                for cid in del_doc_ids:
                    self._docs.pop(cid, None)
                counts["doc_chunks"] += len(del_doc_ids)

            if confluence_page_id:
                del_cf_ids = [
                    pid
                    for pid, p in self._confluence.items()
                    if p.tenant_id == principal.tenant_id
                    and p.domain_id == domain_id
                    and p.confluence_page_id == confluence_page_id
                    and _scopes_intersect(p.allowed_scopes, principal.allowed_scopes)
                ]
                for pid in del_cf_ids:
                    self._confluence.pop(pid, None)
                counts["confluence_pages"] += len(del_cf_ids)

            if jira_issue_key:
                del_jira_ids = [
                    iid
                    for iid, j in self._jira.items()
                    if j.tenant_id == principal.tenant_id
                    and j.domain_id == domain_id
                    and j.issue_key == jira_issue_key
                    and _scopes_intersect(j.allowed_scopes, principal.allowed_scopes)
                ]
                for iid in del_jira_ids:
                    self._jira.pop(iid, None)
                counts["jira_issues"] += len(del_jira_ids)

            repo_variants = (repo_name, repo_name.replace("/", "_")) if repo_name else ()
            target_file_ids: list[UUID] = []
            if repo_name and file_path:
                target_file_ids = [
                    fid
                    for fid, f in self._files.items()
                    if f.tenant_id == principal.tenant_id
                    and f.domain_id == domain_id
                    and f.repo_name in repo_variants
                    and f.file_path == file_path
                    and _scopes_intersect(f.allowed_scopes, principal.allowed_scopes)
                ]
                del_ov_keys = [
                    k
                    for k, d in self._overlay_deltas.items()
                    if d.tenant_id == principal.tenant_id
                    and d.domain_id == domain_id
                    and d.repo_name in repo_variants
                    and (d.file_path == file_path or d.previous_file_path == file_path)
                    and _scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
                ]
                for k in del_ov_keys:
                    self._overlay_deltas.pop(k, None)
                counts["overlay_file_deltas"] += len(del_ov_keys)
                del_se_ids = [
                    se_id
                    for se_id, se in self._side_effects.items()
                    if se.tenant_id == principal.tenant_id
                    and se.domain_id == domain_id
                    and se.repo_name in repo_variants
                    and se.file_path == file_path
                    and _scopes_intersect(se.allowed_scopes, principal.allowed_scopes)
                ]
                for se_id in del_se_ids:
                    self._side_effects.pop(se_id, None)
                counts["side_effects"] += len(del_se_ids)
            elif blob_sha:
                target_file_ids = [
                    fid
                    for fid, f in self._files.items()
                    if f.tenant_id == principal.tenant_id
                    and f.domain_id == domain_id
                    and f.git_blob_sha == blob_sha
                    and _scopes_intersect(f.allowed_scopes, principal.allowed_scopes)
                ]
                del_ov_keys = [
                    k
                    for k, d in self._overlay_deltas.items()
                    if d.tenant_id == principal.tenant_id
                    and d.domain_id == domain_id
                    and (d.overlay_blob_sha == blob_sha or d.base_blob_sha == blob_sha)
                    and _scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
                ]
                for k in del_ov_keys:
                    self._overlay_deltas.pop(k, None)
                counts["overlay_file_deltas"] += len(del_ov_keys)
            elif commit_sha:
                target_file_ids = [
                    fid
                    for fid, f in self._files.items()
                    if f.tenant_id == principal.tenant_id
                    and f.domain_id == domain_id
                    and f.git_commit_sha == commit_sha
                    and _scopes_intersect(f.allowed_scopes, principal.allowed_scopes)
                ]
                domain_repo_ids = {
                    r.repo_id
                    for r in self._repos.values()
                    if r.tenant_id == principal.tenant_id
                    and r.domain_id == domain_id
                    and _scopes_intersect(r.allowed_scopes, principal.allowed_scopes)
                }
                del_commit_ids = [
                    cid
                    for cid, c in self._commits.items()
                    if c.tenant_id == principal.tenant_id
                    and c.commit_sha == commit_sha
                    and (not domain_repo_ids or c.repo_id in domain_repo_ids)
                ]
                for cid in del_commit_ids:
                    self._commits.pop(cid, None)
                counts["repo_commits"] += len(del_commit_ids)
                del_snap_ids = [
                    sid
                    for sid, s in self._ref_snapshots.items()
                    if s.tenant_id == principal.tenant_id
                    and s.domain_id == domain_id
                    and (s.head_commit_sha == commit_sha or s.base_commit_sha == commit_sha)
                    and _scopes_intersect(s.allowed_scopes, principal.allowed_scopes)
                ]
                for sid in del_snap_ids:
                    self._ref_snapshots.pop(sid, None)
                    del_ov = [k for k in self._overlay_deltas if k[1] == sid]
                    for k in del_ov:
                        self._overlay_deltas.pop(k, None)
                    counts["overlay_file_deltas"] += len(del_ov)
                counts["ref_snapshots"] += len(del_snap_ids)
            elif repo_name:
                target_file_ids = [
                    fid
                    for fid, f in self._files.items()
                    if f.tenant_id == principal.tenant_id
                    and f.domain_id == domain_id
                    and f.repo_name in repo_variants
                    and _scopes_intersect(f.allowed_scopes, principal.allowed_scopes)
                ]
                del_se_ids = [
                    se_id
                    for se_id, se in self._side_effects.items()
                    if se.tenant_id == principal.tenant_id
                    and se.domain_id == domain_id
                    and se.repo_name in repo_variants
                    and _scopes_intersect(se.allowed_scopes, principal.allowed_scopes)
                ]
                for se_id in del_se_ids:
                    self._side_effects.pop(se_id, None)
                counts["side_effects"] += len(del_se_ids)
                del_ov_keys = [
                    k
                    for k, d in self._overlay_deltas.items()
                    if d.tenant_id == principal.tenant_id
                    and d.domain_id == domain_id
                    and d.repo_name in repo_variants
                    and _scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
                ]
                for k in del_ov_keys:
                    self._overlay_deltas.pop(k, None)
                counts["overlay_file_deltas"] += len(del_ov_keys)
                del_snap_ids = [
                    sid
                    for sid, s in self._ref_snapshots.items()
                    if s.tenant_id == principal.tenant_id
                    and s.domain_id == domain_id
                    and s.repo_name in repo_variants
                    and _scopes_intersect(s.allowed_scopes, principal.allowed_scopes)
                ]
                for sid in del_snap_ids:
                    self._ref_snapshots.pop(sid, None)
                counts["ref_snapshots"] += len(del_snap_ids)
                del_repo_ids = [
                    rid
                    for rid, r in self._repos.items()
                    if r.tenant_id == principal.tenant_id
                    and r.domain_id == domain_id
                    and r.repo_name in repo_variants
                    and _scopes_intersect(r.allowed_scopes, principal.allowed_scopes)
                ]
                del_repo_set = set(del_repo_ids)
                del_commit_ids = [
                    cid
                    for cid, c in self._commits.items()
                    if c.tenant_id == principal.tenant_id and c.repo_id in del_repo_set
                ]
                for cid in del_commit_ids:
                    self._commits.pop(cid, None)
                counts["repo_commits"] += len(del_commit_ids)
                for rid in del_repo_ids:
                    self._repos.pop(rid, None)
                counts["repositories"] += len(del_repo_ids)

            if target_file_ids:
                file_set = set(target_file_ids)
                del_se_f = [
                    se_id for se_id, se in self._side_effects.items() if se.file_id in file_set
                ]
                for se_id in del_se_f:
                    self._side_effects.pop(se_id, None)
                counts["side_effects"] += len(del_se_f)

                del_sym_ids = [sid for sid, sym in self._symbols.items() if sym.file_id in file_set]
                sym_set = set(del_sym_ids)
                if sym_set:
                    del_edge_ids = [
                        eid
                        for eid, e in self._edges.items()
                        if e.domain_id == domain_id
                        and (e.source_symbol_id in sym_set or e.target_symbol_id in sym_set)
                    ]
                    for eid in del_edge_ids:
                        self._edges.pop(eid, None)
                    counts["ast_edges"] += len(del_edge_ids)
                for sid in del_sym_ids:
                    self._symbols.pop(sid, None)
                counts["ast_symbols"] += len(del_sym_ids)
                for fid in target_file_ids:
                    self._files.pop(fid, None)
                counts["code_files"] += len(target_file_ids)

            if include_derived_wiki and repo_name:
                wiki_prefixes = (
                    f"repo_wiki:{repo_name}",
                    f"repo_wiki:{repo_name.replace('/', '_')}",
                )
                del_wiki_ids = [
                    cid
                    for cid, d in self._docs.items()
                    if d.tenant_id == principal.tenant_id
                    and d.domain_id == domain_id
                    and d.source_type == "repo_wiki"
                    and (
                        d.doc_id in wiki_prefixes
                        or any(d.doc_id.startswith(f"{pfx}:") for pfx in wiki_prefixes)
                    )
                    and _scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
                ]
                for cid in del_wiki_ids:
                    self._docs.pop(cid, None)
                counts["doc_chunks"] += len(del_wiki_ids)

            return counts

    def get_scout_session(
        self,
        principal: PrincipalContext,
        session_id: str,
    ) -> ScoutSessionRecord | None:
        with self._lock:
            rec = self._sessions.get((principal.tenant_id, session_id))
            if rec is None:
                return None
            if rec.principal_id != principal.user_id:
                return None
            if not _scopes_intersect(rec.allowed_scopes, principal.allowed_scopes):
                return None
            return rec

    def upsert_scout_session(
        self,
        principal: PrincipalContext,
        record: ScoutSessionRecord,
    ) -> ScoutSessionRecord:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on upsert_scout_session")
        if record.principal_id != principal.user_id:
            raise PermissionError("Principal user_id mismatch on upsert_scout_session")
        if not _scopes_intersect(record.allowed_scopes, principal.allowed_scopes):
            raise PermissionError("Principal lacks scope to upsert scout_session row")
        with self._lock:
            key = (record.tenant_id, record.session_id)
            prev = self._sessions.get(key)
            if prev is not None and prev.principal_id != principal.user_id:
                raise PermissionError(
                    f"Session '{record.session_id}' belongs to principal '{prev.principal_id}', "
                    f"not '{principal.user_id}'."
                )
            now = datetime.now(UTC)
            stored = ScoutSessionRecord(
                tenant_id=record.tenant_id,
                session_id=record.session_id,
                domain_id=record.domain_id,
                principal_id=record.principal_id,
                allowed_scopes=list(record.allowed_scopes),
                turn_count=record.turn_count,
                compacted_summary=record.compacted_summary,
                recent_turns_json=[dict(t) for t in record.recent_turns_json],
                pinned_file_blobs_json={
                    str(k): dict(v) for k, v in record.pinned_file_blobs_json.items()
                },
                pinned_symbols_json=list(record.pinned_symbols_json),
                total_tokens_compacted=record.total_tokens_compacted,
                created_at=(
                    prev.created_at if prev and prev.created_at else (record.created_at or now)
                ),
                updated_at=now,
            )
            self._sessions[key] = stored
            return stored

    def delete_scout_session(
        self,
        principal: PrincipalContext,
        session_id: str,
    ) -> int:
        with self._lock:
            key = (principal.tenant_id, session_id)
            rec = self._sessions.get(key)
            if rec is None or not _scopes_intersect(rec.allowed_scopes, principal.allowed_scopes):
                return 0
            self._sessions.pop(key, None)
            return 1

    def register_model_artifact(
        self,
        principal: PrincipalContext,
        record: ModelArtifactRecord,
    ) -> ModelArtifactRecord:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on register_model_artifact")
        PostgresRLSRepository._validate_model_artifact_record(record)
        with self._lock:
            prev = self._model_artifacts.get(
                (principal.tenant_id, record.domain_id, record.artifact_id)
            )
            if prev is not None and prev.status == "revoked":
                raise PermissionError(
                    f"Revoked model artifact '{record.artifact_id}' cannot be "
                    "modified or re-activated."
                )
            now = datetime.now(UTC)
            if record.is_active:
                for key, existing in list(self._model_artifacts.items()):
                    if (
                        key[0] == principal.tenant_id
                        and key[1] == record.domain_id
                        and key[2] != record.artifact_id
                        and existing.artifact_kind == record.artifact_kind
                        and existing.is_active
                    ):
                        self._model_artifacts[key] = dataclasses.replace(
                            existing,
                            is_active=False,
                            status="retired" if existing.status == "active" else existing.status,
                            updated_at=now,
                        )
            effective_status: ModelArtifactStatus = "active" if record.is_active else record.status
            stored = dataclasses.replace(
                record,
                status=effective_status,
                created_at=(
                    prev.created_at if prev and prev.created_at else (record.created_at or now)
                ),
                activated_at=(
                    (
                        prev.activated_at
                        if prev and prev.activated_at
                        else (record.activated_at or now)
                    )
                    if record.is_active
                    else (prev.activated_at if prev else record.activated_at)
                ),
                updated_at=now,
            )
            self._model_artifacts[(principal.tenant_id, record.domain_id, record.artifact_id)] = (
                stored
            )
            return stored

    def activate_model_artifact(
        self,
        principal: PrincipalContext,
        domain_id: str,
        artifact_id: str,
        *,
        artifact_kind: ModelArtifactKind = "lora_adapter",
    ) -> ModelArtifactRecord:
        with self._lock:
            target_key = (principal.tenant_id, domain_id, artifact_id)
            target = self._model_artifacts.get(target_key)
            if target is None or target.artifact_kind != artifact_kind:
                raise KeyError(
                    f"Model artifact '{artifact_id}' ({artifact_kind}) not registered "
                    f"for '{domain_id}'"
                )
            if target.status == "revoked":
                raise PermissionError(
                    f"Revoked model artifact '{artifact_id}' cannot be re-activated."
                )
            now = datetime.now(UTC)
            for key, existing in list(self._model_artifacts.items()):
                if (
                    key[0] == principal.tenant_id
                    and key[1] == domain_id
                    and key[2] != artifact_id
                    and existing.artifact_kind == artifact_kind
                    and existing.is_active
                ):
                    self._model_artifacts[key] = dataclasses.replace(
                        existing,
                        is_active=False,
                        status="retired" if existing.status == "active" else existing.status,
                        updated_at=now,
                    )
            activated = dataclasses.replace(
                target,
                is_active=True,
                status="active",
                activated_at=target.activated_at or now,
                updated_at=now,
            )
            self._model_artifacts[target_key] = activated
            return activated

    def retire_model_artifact(
        self,
        principal: PrincipalContext,
        domain_id: str,
        artifact_id: str,
        *,
        revoke: bool = False,
    ) -> bool:
        with self._lock:
            key = (principal.tenant_id, domain_id, artifact_id)
            existing = self._model_artifacts.get(key)
            if existing is None:
                return False
            self._model_artifacts[key] = dataclasses.replace(
                existing,
                is_active=False,
                status="revoked" if revoke else "retired",
                updated_at=datetime.now(UTC),
            )
            return True

    def get_active_model_artifact(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        artifact_kind: ModelArtifactKind = "lora_adapter",
    ) -> ModelArtifactRecord | None:
        with self._lock:
            for (tid, did, _aid), rec in self._model_artifacts.items():
                if (
                    tid == principal.tenant_id
                    and did == domain_id
                    and rec.artifact_kind == artifact_kind
                    and rec.is_active
                ):
                    return rec
            return None

    def list_active_model_artifacts(
        self,
        principal: PrincipalContext,
        *,
        artifact_kind: ModelArtifactKind = "lora_adapter",
    ) -> list[ModelArtifactRecord]:
        with self._lock:
            matched = [
                rec
                for (tid, _did, _aid), rec in self._model_artifacts.items()
                if tid == principal.tenant_id
                and rec.artifact_kind == artifact_kind
                and rec.is_active
            ]
            return sorted(matched, key=lambda r: (r.domain_id, r.artifact_id))

    def list_model_artifacts(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        artifact_kind: ModelArtifactKind | None = None,
    ) -> list[ModelArtifactRecord]:
        with self._lock:
            matched = [
                rec
                for (tid, did, _aid), rec in self._model_artifacts.items()
                if tid == principal.tenant_id
                and did == domain_id
                and (artifact_kind is None or rec.artifact_kind == artifact_kind)
            ]
        epoch = datetime.min.replace(tzinfo=UTC)
        matched.sort(key=lambda r: r.artifact_id)
        matched.sort(key=lambda r: r.created_at or epoch, reverse=True)
        return matched
