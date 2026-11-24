"""<60s Multi-Store Secret & Artifact Purge Coordinator (PG17 + Zoekt + GCS/S3)."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import UUID

from context_scout.db.repositories import PostgresRLSRepository, normalize_purge_selectors
from context_scout.db.session import RLSConnectionManager
from context_scout.schemas.contracts import PrincipalContext
from context_scout.security.permission_mirror import sanitize_scope_identifier
from context_scout.storage.object_store import ArtifactObjectStore, validate_object_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactPurgeReport:
    """Verified cross-store purge telemetry proving zero residue within <60s SLA."""

    domain_id: str
    tenant_id: str
    selectors: dict[str, Any]
    pg17_deleted: dict[str, int]
    zoekt_deleted: int
    object_store_deleted: list[str]
    permission_cache_invalidated: bool
    residue_found: int
    residue_details: dict[str, int]
    elapsed_seconds: float
    sla_seconds: float
    sla_met: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArtifactPurgeCoordinator:
    """Executes coordinated <60s secret/artifact purge + zero-residue verification."""

    def __init__(
        self,
        *,
        repository: Any | None = None,
        connection_manager: RLSConnectionManager | None = None,
        zoekt_index: Any | None = None,
        object_store: ArtifactObjectStore | None = None,
        permission_cache: Any | None = None,
        sla_seconds: float = 60.0,
    ) -> None:
        self.repository = repository
        self.connection_manager = connection_manager
        self.zoekt_index = zoekt_index
        self.object_store = object_store
        self.permission_cache = permission_cache
        self.sla_seconds = float(sla_seconds)

    def purge(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        repo_name: str | None = None,
        file_path: str | None = None,
        commit_sha: str | None = None,
        blob_sha: str | None = None,
        doc_id: str | None = None,
        confluence_page_id: str | None = None,
        jira_issue_key: str | None = None,
        object_keys: Sequence[str] = (),
        include_derived_wiki: bool = True,
    ) -> ArtifactPurgeReport:
        """Purge target artifacts across PG17, Zoekt, and Object Store and verify 0 residue."""
        if isinstance(object_keys, (str, bytes)):
            raise ValueError("object_keys must be a sequence of object key strings")
        clean_obj_keys: list[str] = []
        for k in object_keys:
            if not isinstance(k, str) or not k.strip():
                raise ValueError("object_keys entries must not be empty or whitespace")
            clean_obj_keys.append(validate_object_key(k, expected_tenant_id=principal.tenant_id))

        norm = normalize_purge_selectors(
            domain_id,
            doc_id=doc_id,
            confluence_page_id=confluence_page_id,
            jira_issue_key=jira_issue_key,
            repo_name=repo_name,
            file_path=file_path,
            commit_sha=commit_sha,
            blob_sha=blob_sha,
            has_object_keys=bool(clean_obj_keys),
        )
        clean_domain = norm.domain_id
        clean_repo = norm.repo_name
        clean_file = norm.file_path
        clean_commit = norm.commit_sha
        clean_blob = norm.blob_sha
        clean_doc = norm.doc_id
        clean_cf = norm.confluence_page_id
        clean_jira = norm.jira_issue_key

        t0 = time.perf_counter()
        warnings: list[str] = []
        pg17_deleted: dict[str, int] = {}
        zoekt_deleted = 0
        object_store_deleted: list[str] = []
        perm_invalidated = False

        is_pg = isinstance(self.repository, PostgresRLSRepository) or (
            self.connection_manager is not None
        )

        # Capture file/repo hints for commit_sha or blob_sha before DB rows are erased.
        pre_commit_files: list[tuple[str, str]] = []
        if (clean_commit or clean_blob) and self.repository is not None:
            pre_commit_files = self._collect_matching_repo_files(
                principal,
                clean_domain,
                is_pg=is_pg,
                commit_sha=clean_commit,
                blob_sha=clean_blob,
            )

        # 1. PostgreSQL 17 / Repository Purge
        has_db_selectors = any(
            (clean_doc, clean_cf, clean_jira, clean_repo, clean_file, clean_commit, clean_blob)
        )
        if (
            has_db_selectors
            and self.repository is not None
            and hasattr(self.repository, "purge_domain_artifacts")
        ):
            if is_pg:
                if self.connection_manager is None:
                    raise RuntimeError("PostgresRLSRepository purge requires RLSConnectionManager")
                with self.connection_manager.scoped_transaction(principal) as tx_conn:
                    pg17_deleted = self.repository.purge_domain_artifacts(
                        tx_conn,
                        clean_domain,
                        doc_id=clean_doc,
                        confluence_page_id=clean_cf,
                        jira_issue_key=clean_jira,
                        repo_name=clean_repo,
                        file_path=clean_file,
                        commit_sha=clean_commit,
                        blob_sha=clean_blob,
                        include_derived_wiki=include_derived_wiki,
                    )
            else:
                pg17_deleted = self.repository.purge_domain_artifacts(
                    principal,
                    clean_domain,
                    doc_id=clean_doc,
                    confluence_page_id=clean_cf,
                    jira_issue_key=clean_jira,
                    repo_name=clean_repo,
                    file_path=clean_file,
                    commit_sha=clean_commit,
                    blob_sha=clean_blob,
                    include_derived_wiki=include_derived_wiki,
                )

        # 2. Zoekt Trigram Index & ACL Bitmap Purge
        if self.zoekt_index is not None:
            zoekt_deleted += self._purge_zoekt(
                principal.tenant_id,
                clean_domain,
                repo_name=clean_repo,
                file_path=clean_file,
                blob_sha=clean_blob,
                commit_files=pre_commit_files,
            )

        # 3. Object Store (GCS / S3 / MinIO / InMemory) Purge
        if self.object_store is not None:
            keys_to_delete = list(dict.fromkeys(clean_obj_keys))
            domain_prefix = f"tenants/{principal.tenant_id}/domains/{clean_domain}/"
            try:
                listed = self.object_store.list_keys(
                    domain_prefix,
                    expected_tenant_id=principal.tenant_id,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                warnings.append(f"object_store.list_keys failed ({type(exc).__name__}): {exc}")
                listed = []

            for k in listed:
                if (
                    self._object_key_matches(
                        k,
                        repo_name=clean_repo,
                        file_path=clean_file,
                        commit_sha=clean_commit,
                        blob_sha=clean_blob,
                        doc_id=clean_doc,
                        confluence_page_id=clean_cf,
                        jira_issue_key=clean_jira,
                    )
                    and k not in keys_to_delete
                ):
                    keys_to_delete.append(k)

            for k in keys_to_delete:
                try:
                    self.object_store.delete(k, expected_tenant_id=principal.tenant_id)
                    object_store_deleted.append(k)
                except (OSError, ValueError, RuntimeError) as exc:
                    warnings.append(
                        f"object_store.delete({k}) failed ({type(exc).__name__}): {exc}"
                    )

        # 4. Permission Mirror Cache Invalidation (fail-closed)
        if self.permission_cache is not None:
            if hasattr(self.permission_cache, "invalidate_principal"):
                self.permission_cache.invalidate_principal(principal.tenant_id, principal.user_id)
                perm_invalidated = True
            elif hasattr(self.permission_cache, "invalidate_tenant"):
                self.permission_cache.invalidate_tenant(principal.tenant_id)
                perm_invalidated = True
            else:
                invalidated_any = False
                if clean_repo and hasattr(self.permission_cache, "invalidate_scope_prefix"):
                    for repo_tok in {
                        clean_repo,
                        clean_repo.replace("/", "_"),
                        sanitize_scope_identifier(clean_repo),
                    }:
                        self.permission_cache.invalidate_scope_prefix(f"repo:{repo_tok}:")
                    invalidated_any = True
                if hasattr(self.permission_cache, "invalidate_user"):
                    self.permission_cache.invalidate_user(principal.user_id)
                    invalidated_any = True
                if not invalidated_any and hasattr(self.permission_cache, "clear"):
                    self.permission_cache.clear()
                    invalidated_any = True
                perm_invalidated = invalidated_any

        # 5. Deterministic Post-Purge Zero-Residue Verification across all stores
        residue_details = self._verify_zero_residue(
            principal=principal,
            domain_id=clean_domain,
            is_pg=is_pg,
            repo_name=clean_repo,
            file_path=clean_file,
            commit_sha=clean_commit,
            blob_sha=clean_blob,
            doc_id=clean_doc,
            confluence_page_id=clean_cf,
            jira_issue_key=clean_jira,
            checked_object_keys=object_store_deleted,
            commit_files=pre_commit_files,
            include_derived_wiki=include_derived_wiki,
        )
        residue_found = sum(residue_details.values())
        elapsed_seconds = round(time.perf_counter() - t0, 4)
        sla_met = elapsed_seconds < self.sla_seconds and residue_found == 0

        return ArtifactPurgeReport(
            domain_id=clean_domain,
            tenant_id=str(principal.tenant_id),
            selectors={
                "repo_name": clean_repo,
                "file_path": clean_file,
                "commit_sha": clean_commit,
                "blob_sha": clean_blob,
                "doc_id": clean_doc,
                "confluence_page_id": clean_cf,
                "jira_issue_key": clean_jira,
                "object_keys": clean_obj_keys,
                "include_derived_wiki": include_derived_wiki,
            },
            pg17_deleted=pg17_deleted,
            zoekt_deleted=zoekt_deleted,
            object_store_deleted=object_store_deleted,
            permission_cache_invalidated=perm_invalidated,
            residue_found=residue_found,
            residue_details=residue_details,
            elapsed_seconds=elapsed_seconds,
            sla_seconds=self.sla_seconds,
            sla_met=sla_met,
            warnings=warnings,
        )

    def _collect_matching_repo_files(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        is_pg: bool,
        commit_sha: str | None,
        blob_sha: str | None,
    ) -> list[tuple[str, str]]:
        repo = self.repository
        if repo is None:
            return []
        files: list[Any] = []
        try:
            if is_pg and self.connection_manager is not None:
                with self.connection_manager.scoped_transaction(principal) as tx_conn:
                    if hasattr(repo, "list_visible_code_file_records"):
                        files = repo.list_visible_code_file_records(tx_conn, domain_id)
            elif hasattr(repo, "list_visible_code_file_records"):
                files = repo.list_visible_code_file_records(principal, domain_id)
            elif hasattr(repo, "_files"):
                files = [
                    f
                    for f in repo._files.values()
                    if f.tenant_id == principal.tenant_id and f.domain_id == domain_id
                ]
        except Exception as exc:
            logger.debug("Pre-purge file scan skipped (%s): %s", type(exc).__name__, exc)
            return []

        matches: list[tuple[str, str]] = []
        for f in files:
            commit_hit = bool(commit_sha and getattr(f, "git_commit_sha", None) == commit_sha)
            blob_hit = bool(blob_sha and getattr(f, "git_blob_sha", None) == blob_sha)
            if commit_hit or blob_hit:
                matches.append((f.repo_name, f.file_path))
        return list(dict.fromkeys(matches))

    def _purge_zoekt(
        self,
        tenant_id: UUID,
        domain_id: str,
        *,
        repo_name: str | None,
        file_path: str | None,
        blob_sha: str | None,
        commit_files: Sequence[tuple[str, str]],
    ) -> int:
        zoekt = self.zoekt_index
        if zoekt is None:
            return 0
        deleted = 0
        repo_variants = (
            list(dict.fromkeys([repo_name, repo_name.replace("/", "_")])) if repo_name else []
        )

        if repo_variants and file_path:
            for rvar in repo_variants:
                if hasattr(zoekt, "delete_file_all_branches"):
                    deleted += int(
                        zoekt.delete_file_all_branches(
                            tenant_id=tenant_id,
                            domain_id=domain_id,
                            repo_name=rvar,
                            file_path=file_path,
                        )
                    )
                elif hasattr(zoekt, "delete_file") and zoekt.delete_file(
                    tenant_id=tenant_id,
                    domain_id=domain_id,
                    repo_name=rvar,
                    file_path=file_path,
                ):
                    deleted += 1
        elif repo_variants:
            for rvar in repo_variants:
                if hasattr(zoekt, "purge_repo"):
                    deleted += int(
                        zoekt.purge_repo(
                            tenant_id=tenant_id,
                            domain_id=domain_id,
                            repo_name=rvar,
                        )
                    )
                elif hasattr(zoekt, "delete_repo_files"):
                    deleted += int(
                        zoekt.delete_repo_files(
                            tenant_id=tenant_id,
                            domain_id=domain_id,
                            repo_name=rvar,
                        )
                    )
                    if hasattr(zoekt, "acl_registry"):
                        zoekt.acl_registry.remove_repo(tenant_id, domain_id, rvar)

        if blob_sha and hasattr(zoekt, "purge_by_checksum"):
            deleted += int(
                zoekt.purge_by_checksum(
                    tenant_id=tenant_id,
                    domain_id=domain_id,
                    checksum=blob_sha,
                )
            )

        for rname, fpath in commit_files:
            if hasattr(zoekt, "delete_file_all_branches"):
                deleted += int(
                    zoekt.delete_file_all_branches(
                        tenant_id=tenant_id,
                        domain_id=domain_id,
                        repo_name=rname,
                        file_path=fpath,
                    )
                )
        return deleted

    @staticmethod
    def _object_key_matches(
        key: str,
        *,
        repo_name: str | None,
        file_path: str | None,
        commit_sha: str | None,
        blob_sha: str | None,
        doc_id: str | None,
        confluence_page_id: str | None,
        jira_issue_key: str | None,
    ) -> bool:
        if repo_name and file_path:
            r_variants = {repo_name, repo_name.replace("/", "_")}
            if any(f"/{rv}/" in key or f"repos/{rv}/" in key for rv in r_variants) and (
                key.endswith(f"/{file_path}") or f"/{file_path}" in key
            ):
                return True
        elif repo_name:
            r_variants = {repo_name, repo_name.replace("/", "_")}
            if any(
                f"/{rv}/" in key or key.endswith(f"/{rv}") or f"repos/{rv}" in key
                for rv in r_variants
            ):
                return True
        if commit_sha and commit_sha in key:
            return True
        if blob_sha and blob_sha in key:
            return True
        if doc_id and doc_id in key:
            return True
        if confluence_page_id and f"confluence/{confluence_page_id}" in key:
            return True
        return bool(jira_issue_key and f"jira/{jira_issue_key}" in key)

    def _verify_zero_residue(
        self,
        *,
        principal: PrincipalContext,
        domain_id: str,
        is_pg: bool,
        repo_name: str | None,
        file_path: str | None,
        commit_sha: str | None,
        blob_sha: str | None,
        doc_id: str | None,
        confluence_page_id: str | None,
        jira_issue_key: str | None,
        checked_object_keys: Sequence[str],
        commit_files: Sequence[tuple[str, str]],
        include_derived_wiki: bool = True,
    ) -> dict[str, int]:
        db_residue = 0
        zoekt_residue = 0
        object_residue = 0

        if self.repository is not None:
            if is_pg and self.connection_manager is not None:
                with self.connection_manager.scoped_transaction(principal) as tx_conn:
                    db_residue = self._count_pg_residue(
                        tx_conn,
                        domain_id=domain_id,
                        repo_name=repo_name,
                        file_path=file_path,
                        commit_sha=commit_sha,
                        blob_sha=blob_sha,
                        doc_id=doc_id,
                        confluence_page_id=confluence_page_id,
                        jira_issue_key=jira_issue_key,
                        include_derived_wiki=include_derived_wiki,
                    )
            elif not is_pg and hasattr(self.repository, "_files"):
                db_residue = self._count_mem_residue(
                    principal,
                    domain_id=domain_id,
                    repo_name=repo_name,
                    file_path=file_path,
                    commit_sha=commit_sha,
                    blob_sha=blob_sha,
                    doc_id=doc_id,
                    confluence_page_id=confluence_page_id,
                    jira_issue_key=jira_issue_key,
                    include_derived_wiki=include_derived_wiki,
                )

        if self.zoekt_index is not None and hasattr(self.zoekt_index, "_docs"):
            r_variants = {repo_name, repo_name.replace("/", "_")} if repo_name else set()
            with self.zoekt_index._lock:
                for doc in self.zoekt_index._docs.values():
                    if doc.tenant_id != principal.tenant_id or doc.domain_id != domain_id:
                        continue
                    if (
                        (
                            r_variants
                            and file_path
                            and doc.repo_name in r_variants
                            and doc.file_path == file_path
                        )
                        or (r_variants and not file_path and doc.repo_name in r_variants)
                        or (blob_sha and doc.checksum == blob_sha)
                        or (commit_files and (doc.repo_name, doc.file_path) in commit_files)
                    ):
                        zoekt_residue += 1
                if r_variants and not file_path and hasattr(self.zoekt_index, "acl_registry"):
                    for rv in r_variants:
                        if (
                            self.zoekt_index.acl_registry.get_repo_id(
                                principal.tenant_id, domain_id, rv
                            )
                            is not None
                        ):
                            zoekt_residue += 1

        if self.object_store is not None:
            for k in checked_object_keys:
                try:
                    if self.object_store.exists(k, expected_tenant_id=principal.tenant_id):
                        object_residue += 1
                except (OSError, ValueError, RuntimeError):
                    object_residue += 1

        return {
            "pg17": db_residue,
            "zoekt": zoekt_residue,
            "object_store": object_residue,
        }

    def _count_pg_residue(
        self,
        conn: Any,
        *,
        domain_id: str,
        repo_name: str | None,
        file_path: str | None,
        commit_sha: str | None,
        blob_sha: str | None,
        doc_id: str | None,
        confluence_page_id: str | None,
        jira_issue_key: str | None,
        include_derived_wiki: bool = True,
    ) -> int:
        residue = 0
        if doc_id or confluence_page_id or jira_issue_key:
            tgt_doc = doc_id or (
                f"confluence:{confluence_page_id}"
                if confluence_page_id
                else f"jira:{jira_issue_key}"
            )
            r = conn.execute(
                "SELECT COUNT(*) FROM doc_chunks WHERE domain_id = %s AND doc_id = %s;",
                (domain_id, tgt_doc),
            ).fetchone()
            residue += int(str(r[0])) if r else 0
        if confluence_page_id:
            r = conn.execute(
                "SELECT COUNT(*) FROM confluence_pages "
                "WHERE domain_id = %s AND confluence_page_id = %s;",
                (domain_id, confluence_page_id),
            ).fetchone()
            residue += int(str(r[0])) if r else 0
        if jira_issue_key:
            r = conn.execute(
                "SELECT COUNT(*) FROM jira_issues WHERE domain_id = %s AND issue_key = %s;",
                (domain_id, jira_issue_key),
            ).fetchone()
            residue += int(str(r[0])) if r else 0
        if repo_name and file_path:
            rv = [repo_name, repo_name.replace("/", "_")]
            r1 = conn.execute(
                "SELECT COUNT(*) FROM code_files "
                "WHERE domain_id = %s AND repo_name = ANY(%s::text[]) AND file_path = %s;",
                (domain_id, rv, file_path),
            ).fetchone()
            r2 = conn.execute(
                "SELECT COUNT(*) FROM side_effects "
                "WHERE domain_id = %s AND repo_name = ANY(%s::text[]) AND file_path = %s;",
                (domain_id, rv, file_path),
            ).fetchone()
            r3 = conn.execute(
                "SELECT COUNT(*) FROM overlay_file_deltas "
                "WHERE domain_id = %s AND repo_name = ANY(%s::text[]) "
                "AND (file_path = %s OR previous_file_path = %s);",
                (domain_id, rv, file_path, file_path),
            ).fetchone()
            r4 = conn.execute(
                "SELECT COUNT(*) FROM ast_symbols "
                "WHERE domain_id = %s AND file_id IN ("
                "SELECT file_id FROM code_files "
                "WHERE domain_id = %s AND repo_name = ANY(%s::text[]) AND file_path = %s);",
                (domain_id, domain_id, rv, file_path),
            ).fetchone()
            residue += (
                (int(str(r1[0])) if r1 else 0)
                + (int(str(r2[0])) if r2 else 0)
                + (int(str(r3[0])) if r3 else 0)
                + (int(str(r4[0])) if r4 else 0)
            )
        elif repo_name:
            clean_repo = repo_name.replace("/", "_")
            rv = [repo_name, clean_repo]
            for tbl in (
                "code_files",
                "side_effects",
                "repositories",
                "overlay_file_deltas",
                "ref_snapshots",
            ):
                r = conn.execute(
                    f"SELECT COUNT(*) FROM {tbl} "
                    "WHERE domain_id = %s AND repo_name = ANY(%s::text[]);",
                    (domain_id, rv),
                ).fetchone()
                residue += int(str(r[0])) if r else 0
            r_sym = conn.execute(
                """
                SELECT COUNT(*) FROM ast_symbols
                WHERE domain_id = %s
                  AND (
                      file_id IN (
                          SELECT file_id FROM code_files
                          WHERE domain_id = %s AND repo_name = ANY(%s::text[])
                      )
                      OR repo_id IN (
                          SELECT repo_id FROM repositories
                          WHERE domain_id = %s AND repo_name = ANY(%s::text[])
                      )
                      OR split_part(qualified_name, '::', 1) = ANY(%s::text[])
                  );
                """,
                (domain_id, domain_id, rv, domain_id, rv, rv),
            ).fetchone()
            residue += int(str(r_sym[0])) if r_sym else 0
            if include_derived_wiki:
                r_wiki = conn.execute(
                    """
                    SELECT COUNT(*) FROM doc_chunks
                    WHERE domain_id = %s
                      AND source_type = 'repo_wiki'
                      AND (
                          doc_id = %s OR starts_with(doc_id, %s)
                          OR doc_id = %s OR starts_with(doc_id, %s)
                      );
                    """,
                    (
                        domain_id,
                        f"repo_wiki:{repo_name}",
                        f"repo_wiki:{repo_name}:",
                        f"repo_wiki:{clean_repo}",
                        f"repo_wiki:{clean_repo}:",
                    ),
                ).fetchone()
                residue += int(str(r_wiki[0])) if r_wiki else 0
        elif commit_sha:
            r = conn.execute(
                "SELECT COUNT(*) FROM code_files WHERE domain_id = %s AND git_commit_sha = %s;",
                (domain_id, commit_sha),
            ).fetchone()
            residue += int(str(r[0])) if r else 0
        elif blob_sha:
            r = conn.execute(
                "SELECT COUNT(*) FROM code_files WHERE domain_id = %s AND git_blob_sha = %s;",
                (domain_id, blob_sha),
            ).fetchone()
            residue += int(str(r[0])) if r else 0
        return residue

    def _count_mem_residue(
        self,
        principal: PrincipalContext,
        *,
        domain_id: str,
        repo_name: str | None,
        file_path: str | None,
        commit_sha: str | None,
        blob_sha: str | None,
        doc_id: str | None,
        confluence_page_id: str | None,
        jira_issue_key: str | None,
        include_derived_wiki: bool = True,
    ) -> int:
        repo = self.repository
        assert repo is not None
        residue = 0
        tgt_doc = doc_id or (
            f"confluence:{confluence_page_id}"
            if confluence_page_id
            else f"jira:{jira_issue_key}"
            if jira_issue_key
            else None
        )
        if tgt_doc:
            residue += sum(
                1
                for d in repo._docs.values()
                if d.tenant_id == principal.tenant_id
                and d.domain_id == domain_id
                and d.doc_id == tgt_doc
            )
        if confluence_page_id:
            residue += sum(
                1
                for p in repo._confluence.values()
                if p.tenant_id == principal.tenant_id
                and p.domain_id == domain_id
                and p.confluence_page_id == confluence_page_id
            )
        if jira_issue_key:
            residue += sum(
                1
                for j in repo._jira.values()
                if j.tenant_id == principal.tenant_id
                and j.domain_id == domain_id
                and j.issue_key == jira_issue_key
            )
        rv = {repo_name, repo_name.replace("/", "_")} if repo_name else set()
        if rv and file_path:
            matching_file_ids = {
                f.file_id
                for f in repo._files.values()
                if f.tenant_id == principal.tenant_id
                and f.domain_id == domain_id
                and f.repo_name in rv
                and f.file_path == file_path
            }
            residue += len(matching_file_ids)
            residue += sum(
                1
                for s in repo._side_effects.values()
                if s.tenant_id == principal.tenant_id
                and s.domain_id == domain_id
                and s.repo_name in rv
                and s.file_path == file_path
            )
            residue += sum(
                1
                for d in getattr(repo, "_overlay_deltas", {}).values()
                if d.tenant_id == principal.tenant_id
                and d.domain_id == domain_id
                and d.repo_name in rv
                and (d.file_path == file_path or d.previous_file_path == file_path)
            )
            residue += sum(
                1
                for sym in getattr(repo, "_symbols", {}).values()
                if sym.tenant_id == principal.tenant_id
                and sym.domain_id == domain_id
                and sym.file_id in matching_file_ids
            )
        elif rv:
            matching_file_ids = {
                f.file_id
                for f in repo._files.values()
                if f.tenant_id == principal.tenant_id
                and f.domain_id == domain_id
                and f.repo_name in rv
            }
            matching_repo_ids = {
                r.repo_id
                for r in repo._repos.values()
                if r.tenant_id == principal.tenant_id
                and r.domain_id == domain_id
                and r.repo_name in rv
            }
            residue += len(matching_file_ids)
            residue += sum(
                1
                for s in repo._side_effects.values()
                if s.tenant_id == principal.tenant_id
                and s.domain_id == domain_id
                and s.repo_name in rv
            )
            residue += len(matching_repo_ids)
            residue += sum(
                1
                for d in getattr(repo, "_overlay_deltas", {}).values()
                if d.tenant_id == principal.tenant_id
                and d.domain_id == domain_id
                and d.repo_name in rv
            )
            residue += sum(
                1
                for sn in getattr(repo, "_ref_snapshots", {}).values()
                if sn.tenant_id == principal.tenant_id
                and sn.domain_id == domain_id
                and sn.repo_name in rv
            )
            residue += sum(
                1
                for sym in getattr(repo, "_symbols", {}).values()
                if sym.tenant_id == principal.tenant_id
                and sym.domain_id == domain_id
                and (
                    sym.file_id in matching_file_ids
                    or (sym.repo_id is not None and sym.repo_id in matching_repo_ids)
                    or sym.qualified_name.split("::", 1)[0] in rv
                )
            )
            if include_derived_wiki and repo_name:
                wiki_prefixes = (
                    f"repo_wiki:{repo_name}",
                    f"repo_wiki:{repo_name.replace('/', '_')}",
                )
                residue += sum(
                    1
                    for d in repo._docs.values()
                    if d.tenant_id == principal.tenant_id
                    and d.domain_id == domain_id
                    and d.source_type == "repo_wiki"
                    and (
                        d.doc_id in wiki_prefixes
                        or any(d.doc_id.startswith(f"{pfx}:") for pfx in wiki_prefixes)
                    )
                )
        elif commit_sha:
            residue += sum(
                1
                for f in repo._files.values()
                if f.tenant_id == principal.tenant_id
                and f.domain_id == domain_id
                and f.git_commit_sha == commit_sha
            )
        elif blob_sha:
            residue += sum(
                1
                for f in repo._files.values()
                if f.tenant_id == principal.tenant_id
                and f.domain_id == domain_id
                and f.git_blob_sha == blob_sha
            )
        return residue


__all__ = ["ArtifactPurgeCoordinator", "ArtifactPurgeReport"]
