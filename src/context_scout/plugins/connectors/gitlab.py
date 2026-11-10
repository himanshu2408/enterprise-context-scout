"""GitLab REST API Connector, Merkle Diff Ingestion, and Webhook Handler (F17)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel

from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.db.repositories import PostgresRLSRepository
from context_scout.db.schema import (
    ASTSymbolRecord,
    CodeFileRecord,
    RepoCommitRecord,
    RepositoryRecord,
)
from context_scout.ingestion.git_merkle_indexer import compute_git_blob_sha
from context_scout.plugins.base import SourceConnectorPlugin, SyncResult
from context_scout.plugins.registry import PluginRegistry
from context_scout.queue.task_queue import QueueTask
from context_scout.security.permission_mirror import (
    GitLabRole,
    PermissionMirror,
    sanitize_scope_identifier,
)

logger = logging.getLogger(__name__)


class GitLabCommitDiff(BaseModel):
    """File-level diff entry from GitLab Repository Compare API."""

    old_path: str
    new_path: str
    new_file: bool = False
    renamed_file: bool = False
    deleted_file: bool = False
    diff: str = ""


class GitLabConnectorPlugin(SourceConnectorPlugin):
    """Production remote GitLab v4 REST client and webhook processor."""

    def __init__(
        self,
        base_url: str = "https://gitlab.example.com",
        private_token: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if private_token:
            headers["PRIVATE-TOKEN"] = private_token

        self.client = client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0,
        )

    @property
    def source_id(self) -> str:
        return "gitlab"

    @property
    def source_type(self) -> str:
        return "gitlab"

    def mirror_permissions(self, container_key: str) -> list[str]:
        """Maps GitLab project path to RLS allowed_scopes."""
        clean_path = sanitize_scope_identifier(container_key)
        return PermissionMirror.map_gitlab_role_to_scopes(clean_path, GitLabRole.REPORTER)

    def fetch_item(self, item_id: str, version: str | None = None) -> dict[str, Any] | None:
        """Fetches project metadata by project ID or URL-encoded path."""
        encoded_id = httpx.URL(item_id).raw_path.decode() if "/" in item_id else item_id
        resp = self.client.get(f"/api/v4/projects/{encoded_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def compare_commits(
        self,
        project_id_or_path: str | int,
        from_sha: str,
        to_sha: str,
    ) -> list[GitLabCommitDiff]:
        """Calls GitLab Compare API to retrieve modified file paths."""
        endpoint = f"/api/v4/projects/{project_id_or_path}/repository/compare"
        params = {"from": from_sha, "to": to_sha}
        resp = self.client.get(endpoint, params=params)
        resp.raise_for_status()
        data = resp.json()

        diffs: list[GitLabCommitDiff] = []
        for d in data.get("diffs", []):
            diffs.append(
                GitLabCommitDiff(
                    old_path=d.get("old_path", ""),
                    new_path=d.get("new_path", ""),
                    new_file=bool(d.get("new_file", False)),
                    renamed_file=bool(d.get("renamed_file", False)),
                    deleted_file=bool(d.get("deleted_file", False)),
                    diff=d.get("diff", ""),
                )
            )
        return diffs

    def fetch_raw_blob(
        self, project_id_or_path: str | int, file_path: str, ref_sha: str
    ) -> bytes:
        """Fetches raw blob bytes directly from GitLab."""
        import urllib.parse

        url_path = urllib.parse.quote(file_path, safe="")
        endpoint = f"/api/v4/projects/{project_id_or_path}/repository/files/{url_path}/raw"
        params = {"ref": ref_sha}
        resp = self.client.get(endpoint, params=params)
        resp.raise_for_status()
        return resp.content

    def process_push_webhook(
        self,
        task: QueueTask,
        repository: PostgresRLSRepository | Any,
        conn: Any = None,
        principal: Any = None,
    ) -> SyncResult:
        """Processes gitlab_push queue task using incremental Merkle diffing."""
        start_time = datetime.now(UTC)
        payload = task.payload
        domain_id = task.domain_id

        # 1. Extract push metadata
        project_info = payload.get("project", {})
        project_id = project_info.get("id") or payload.get("project_id")
        project_path = project_info.get("path_with_namespace") or payload.get("repo_name", "")
        clean_path = sanitize_scope_identifier(project_path)
        default_branch = project_info.get("default_branch", "main")
        ref = payload.get("ref", "")

        before_sha = payload.get("before") or payload.get("before_sha")
        after_sha = payload.get("after") or payload.get("after_sha")

        if not before_sha or not after_sha or not project_id:
            logger.warning("Incomplete GitLab push webhook payload for task %s", task.task_id)
            return SyncResult(
                source_type="gitlab",
                items_scanned=0,
                items_indexed=0,
                items_updated=0,
                items_deleted=0,
                stale_items_flagged=0,
                cursor_or_checkpoint=after_sha or "",
                duration_ms=0.0,
                errors=["Incomplete push webhook payload: missing before/after SHA or project ID"],
            )

        allowed_scopes = self.mirror_permissions(clean_path)
        is_postgres = hasattr(repository, "upsert_repository") and conn is not None

        # 2. Upsert Repository Record
        repo_rec = RepositoryRecord(
            repo_id=uuid4(),
            domain_id=domain_id,
            repo_name=clean_path,
            source_type="gitlab",
            remote_url=project_info.get("git_http_url", f"{self.base_url}/{clean_path}.git"),
            default_branch=default_branch,
            allowed_scopes=allowed_scopes,
            head_commit_sha=after_sha,
            metadata_json={"project_id": project_id, "last_push_ref": ref},
            is_active=True,
            updated_at=datetime.now(UTC),
        )
        if is_postgres:
            repository.upsert_repository(conn, repo_rec)
        elif hasattr(repository, "_repos"):
            # Purge existing record to prevent duplicate repository records
            for old_id, r in list(repository._repos.items()):
                if r.domain_id == domain_id and r.repo_name in (clean_path, project_path):
                    repository._repos.pop(old_id, None)
            repository._repos[repo_rec.repo_id] = repo_rec

        # 3. Insert Commit Records
        for commit in payload.get("commits", []):
            commit_sha = commit.get("id")
            if commit_sha:
                c_rec = RepoCommitRecord(
                    commit_id=uuid4(),
                    repo_id=repo_rec.repo_id,
                    commit_sha=commit_sha,
                    parent_shas=[],
                    author_name=commit.get("author", {}).get("name", "Unknown"),
                    author_email=commit.get("author", {}).get("email", ""),
                    commit_message=commit.get("message", ""),
                    committed_at=datetime.fromisoformat(
                        commit.get("timestamp", datetime.now(UTC).isoformat()).replace(
                            "Z", "+00:00"
                        )
                    ),
                    indexed_at=datetime.now(UTC),
                )
                if is_postgres:
                    repository.insert_repo_commit(conn, c_rec)
                elif hasattr(repository, "_commits"):
                    repository._commits[c_rec.commit_id] = c_rec

        # 4. Compare diffs and sync modified blobs
        items_indexed = 0
        items_deleted = 0
        is_initial_push = before_sha.strip("0") == ""

        # Determine target files
        files_to_index: list[tuple[str, bool]] = []  # (file_path, is_deleted)
        if is_initial_push:
            try:
                tree_resp = self.client.get(
                    f"/api/v4/projects/{project_id}/repository/tree",
                    params={"ref": after_sha, "recursive": True, "per_page": 100},
                )
                if tree_resp.status_code == 200:
                    for item in tree_resp.json():
                        if item.get("type") == "blob":
                            files_to_index.append((item["path"], False))
            except Exception as e:
                logger.warning("Failed to fetch repository tree for initial push: %s", e)
        else:
            diffs = self.compare_commits(project_id, before_sha, after_sha)
            for diff in diffs:
                if diff.deleted_file:
                    files_to_index.append((diff.old_path, True))
                elif diff.renamed_file:
                    # In Git diff semantics, rename = delete old_path + index new_path
                    files_to_index.append((diff.old_path, True))
                    files_to_index.append((diff.new_path, False))
                else:
                    files_to_index.append((diff.new_path, False))

        extractor = ASTExtractor()
        for file_path, is_deleted in files_to_index:
            if is_deleted:
                if is_postgres:
                    if hasattr(repository, "delete_code_file_and_symbols"):
                        repository.delete_code_file_and_symbols(
                            conn, domain_id, project_path, file_path
                        )
                    elif hasattr(repository, "get_code_files_by_paths") and hasattr(
                        repository, "delete_code_file"
                    ):
                        del_files = repository.get_code_files_by_paths(
                            conn, domain_id, [file_path]
                        )
                        deleted_any = False
                        for f in del_files:
                            if f.repo_name in (project_path, clean_path):
                                repository.delete_code_file(conn, f.file_id)
                                deleted_any = True
                        if not deleted_any:
                            conn.execute(
                                """
                                DELETE FROM code_files
                                WHERE domain_id = %s
                                  AND (repo_name = %s OR repo_name = %s)
                                  AND file_path = %s;
                                """,
                                (domain_id, project_path, clean_path, file_path),
                            )
                    else:
                        conn.execute(
                            """
                            DELETE FROM code_files
                            WHERE domain_id = %s
                              AND (repo_name = %s OR repo_name = %s)
                              AND file_path = %s;
                            """,
                            (domain_id, project_path, clean_path, file_path),
                        )
                elif hasattr(repository, "delete_code_file_and_symbols"):
                    repository.delete_code_file_and_symbols(domain_id, project_path, file_path)
                elif hasattr(repository, "_files"):
                    del_ids = [
                        fid
                        for fid, f in repository._files.items()
                        if f.domain_id == domain_id
                        and f.repo_name in (project_path, clean_path)
                        and f.file_path == file_path
                    ]
                    for fid in del_ids:
                        repository._files.pop(fid, None)
                        if hasattr(repository, "_symbols"):
                            sym_del_ids = [
                                sid
                                for sid, sym in repository._symbols.items()
                                if sym.file_id == fid
                            ]
                            for sid in sym_del_ids:
                                repository._symbols.pop(sid, None)
                items_deleted += 1
            else:
                raw_bytes = self.fetch_raw_blob(project_id, file_path, after_sha)
                blob_sha = compute_git_blob_sha(raw_bytes)
                raw_content = raw_bytes.decode("utf-8", errors="replace")

                lang_plugin = PluginRegistry.get_plugin_for_file(file_path)
                lang = (
                    lang_plugin.language_id
                    if lang_plugin
                    else (
                        "python"
                        if file_path.endswith(".py")
                        else "typescript"
                        if file_path.endswith((".ts", ".tsx"))
                        else "text"
                    )
                )

                file_rec = CodeFileRecord(
                    file_id=uuid4(),
                    domain_id=domain_id,
                    repo_name=project_path,
                    file_path=file_path,
                    git_commit_sha=after_sha,
                    git_blob_sha=blob_sha,
                    language=lang,
                    allowed_scopes=allowed_scopes,
                    raw_content=raw_content,
                    line_count=len(raw_content.splitlines()),
                    repo_id=repo_rec.repo_id,
                )

                # Clean up existing record & symbols before re-inserting modified file
                if is_postgres:
                    if hasattr(repository, "delete_code_file_and_symbols"):
                        repository.delete_code_file_and_symbols(
                            conn, domain_id, project_path, file_path
                        )
                    else:
                        conn.execute(
                            """
                            DELETE FROM code_files
                            WHERE domain_id = %s
                              AND (repo_name = %s OR repo_name = %s)
                              AND file_path = %s;
                            """,
                            (domain_id, project_path, clean_path, file_path),
                        )
                elif hasattr(repository, "delete_code_file_and_symbols"):
                    repository.delete_code_file_and_symbols(domain_id, project_path, file_path)
                elif hasattr(repository, "_files"):
                    existing_del_ids = [
                        fid
                        for fid, f in repository._files.items()
                        if f.domain_id == domain_id
                        and f.repo_name in (project_path, clean_path)
                        and f.file_path == file_path
                    ]
                    for fid in existing_del_ids:
                        repository._files.pop(fid, None)
                        if hasattr(repository, "_symbols"):
                            sym_del_ids = [
                                sid
                                for sid, sym in repository._symbols.items()
                                if sym.file_id == fid
                            ]
                            for sid in sym_del_ids:
                                repository._symbols.pop(sid, None)

                if is_postgres:
                    repository.insert_code_file(conn, file_rec)
                else:
                    if hasattr(repository, "insert_code_file") and principal is not None:
                        repository.insert_code_file(principal, file_rec)
                    elif hasattr(repository, "_files"):
                        repository._files[file_rec.file_id] = file_rec

                if lang != "text":
                    try:
                        symbols, _edges = extractor.parse_source(
                            raw_content, lang, file_path=file_path
                        )
                        for sym in symbols:
                            sym_rec = ASTSymbolRecord(
                                symbol_id=uuid4(),
                                file_id=file_rec.file_id,
                                domain_id=domain_id,
                                qualified_name=f"{project_path}::{sym.name}",
                                symbol_kind=sym.symbol_kind,
                                start_line=sym.start_line,
                                end_line=sym.end_line,
                                signature_skeleton=sym.signature or sym.name,
                                contextual_summary="",
                                allowed_scopes=allowed_scopes,
                                embedding=[0.0] * 768,
                            )
                            if is_postgres and hasattr(repository, "insert_ast_symbol"):
                                repository.insert_ast_symbol(conn, sym_rec)
                            elif hasattr(repository, "insert_ast_symbol") and principal is not None:
                                repository.insert_ast_symbol(principal, sym_rec)
                            elif hasattr(repository, "_symbols"):
                                repository._symbols[sym_rec.symbol_id] = sym_rec
                    except Exception as err:
                        logger.debug("AST extraction skipped for %s: %s", file_path, err)

                items_indexed += 1

        duration = (datetime.now(UTC) - start_time).total_seconds() * 1000.0

        return SyncResult(
            source_type="gitlab",
            items_scanned=len(payload.get("commits", [])),
            items_indexed=items_indexed,
            items_updated=1,
            items_deleted=items_deleted,
            stale_items_flagged=0,
            cursor_or_checkpoint=after_sha,
            duration_ms=round(duration, 2),
        )

    def process_merge_request_webhook(
        self,
        task: QueueTask,
        repository: PostgresRLSRepository | Any,
        conn: Any = None,
        principal: Any = None,
    ) -> SyncResult:
        """Processes Merge Request webhook: indexes merged state on default branch."""
        payload = task.payload
        attrs = payload.get("object_attributes", {})
        state = attrs.get("state")
        action = attrs.get("action")
        target_branch = attrs.get("target_branch")

        if state == "merged" or action == "merge":
            merge_commit = attrs.get("merge_commit_sha") or attrs.get("last_commit", {}).get("id")
            logger.info(
                "GitLab MR merged to %s at commit %s. Enqueuing push sync.",
                target_branch,
                merge_commit,
            )
            fake_push_task = QueueTask(
                task_id=task.task_id,
                domain_id=task.domain_id,
                event_type="gitlab_push",
                payload={
                    "project": payload.get("project", {}),
                    "repo_name": payload.get("project", {}).get("path_with_namespace"),
                    "ref": f"refs/heads/{target_branch}",
                    "before": (
                        attrs.get("target", {}).get("default_branch")
                        or "0000000000000000000000000000000000000000"
                    ),
                    "after": merge_commit,
                    "commits": [attrs.get("last_commit", {})],
                },
                priority=20,
                status="PROCESSING",
                worker_id=task.worker_id,
                attempts=task.attempts,
                max_attempts=task.max_attempts,
                error_message=None,
                scheduled_at=task.scheduled_at,
                created_at=task.created_at,
            )
            return self.process_push_webhook(fake_push_task, repository, conn, principal)

        return SyncResult(
            source_type="gitlab",
            items_scanned=0,
            items_indexed=0,
            items_updated=0,
            items_deleted=0,
            stale_items_flagged=0,
            cursor_or_checkpoint="",
            duration_ms=0.0,
        )
