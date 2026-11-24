"""GitHub & GitHub Enterprise REST API Connector, Merkle Diff Ingestion, and App ACLs (M1-05)."""

from __future__ import annotations

import base64
import dataclasses
import logging
import urllib.parse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, Field

from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.ast_engine.side_effect_extractor import SideEffectExtractor
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
from context_scout.schemas.contracts import DEFAULT_TENANT_ID
from context_scout.security.permission_mirror import sanitize_scope_identifier
from context_scout.security.pre_index_scrubber import PreIndexScrubber

logger = logging.getLogger(__name__)

_READ_PERMS: frozenset[str] = frozenset(
    {"pull", "triage", "push", "maintain", "admin", "read", "write"}
)
_WRITE_PERMS: frozenset[str] = frozenset({"push", "maintain", "admin", "write"})
_ADMIN_PERMS: frozenset[str] = frozenset({"maintain", "admin"})


class GitHubCommitDiff(BaseModel):
    """File-level diff entry from GitHub Repository Compare API."""

    filename: str
    previous_filename: str | None = None
    status: str = "modified"
    patch: str = ""
    sha: str = ""


class GitHubAppInstallationACL(BaseModel):
    """GitHub App installation permissions and repository selection scope."""

    installation_id: int | str
    repository_selection: Literal["all", "selected"] = "selected"
    repositories: list[str] = Field(default_factory=list)
    permissions: dict[str, str] = Field(default_factory=dict)
    suspended: bool = False


class GitHubEnterpriseConnectorPlugin(SourceConnectorPlugin):
    """Production GitHub.com & GitHub Enterprise Server REST client and webhook processor."""

    def __init__(
        self,
        base_url: str = "https://api.github.com",
        installation_token: str | None = None,
        bearer_token: str | None = None,
        installation_id: int | str | None = None,
        client: httpx.Client | None = None,
        scrubber: PreIndexScrubber | None = None,
        side_effect_extractor: SideEffectExtractor | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.installation_id = installation_id
        self.scrubber = scrubber or PreIndexScrubber()
        self.side_effect_extractor = side_effect_extractor or SideEffectExtractor()
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        token = installation_token or bearer_token
        if token:
            headers["Authorization"] = f"Bearer {token}"

        self.client = client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0,
        )

    @property
    def source_id(self) -> str:
        return "github"

    @property
    def source_type(self) -> str:
        return "github"

    @staticmethod
    def _normalize_repo_path(repo_full_name: str) -> tuple[str, str]:
        """Splits and validates 'owner/repo' into (owner, repo)."""
        clean_repo = sanitize_scope_identifier(repo_full_name)
        parts = clean_repo.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid GitHub repository identifier: '{repo_full_name}'")
        return parts[0], parts[1]

    @staticmethod
    def _extract_permission_level(entry: Mapping[str, Any]) -> str:
        """Extracts normalized permission string from GitHub team/collaborator payloads."""
        if entry.get("read_contents") is False:
            return "none"
        raw_perm = str(entry.get("permission", "")).strip().lower()
        if raw_perm:
            return raw_perm
        perms_map = entry.get("permissions")
        if isinstance(perms_map, Mapping):
            for candidate in ("admin", "maintain", "push", "triage", "pull"):
                if perms_map.get(candidate) is True:
                    return candidate
        return "none"

    def fetch_app_installation_repos(self) -> list[str]:
        """Fetches repositories accessible to the GitHub App installation."""
        try:
            resp = self.client.get("/installation/repositories")
        except httpx.HTTPError:
            return []
        if resp.status_code != 200:
            return []
        try:
            data: Any = resp.json()
        except ValueError:
            return []
        if not isinstance(data, Mapping):
            return []

        repos: list[str] = []
        for item in data.get("repositories", []):
            if isinstance(item, Mapping):
                full_name = item.get("full_name")
                if isinstance(full_name, str) and full_name.strip():
                    repos.append(sanitize_scope_identifier(full_name))
        return repos

    def fetch_repo_teams(self, repo_full_name: str) -> list[dict[str, Any]]:
        """Fetches teams with access to a repository; fails closed ([]) on 403/404."""
        owner, repo = self._normalize_repo_path(repo_full_name)
        try:
            resp = self.client.get(f"/repos/{owner}/{repo}/teams")
        except httpx.HTTPError:
            return []
        if resp.status_code != 200:
            return []
        try:
            data: Any = resp.json()
        except ValueError:
            return []
        if not isinstance(data, list):
            return []
        return [dict(item) for item in data if isinstance(item, Mapping)]

    def fetch_collaborator_permission(self, repo_full_name: str, username: str) -> dict[str, Any]:
        """Fetches a collaborator's permission for a repository; fails closed ({}) on 403/404."""
        owner, repo = self._normalize_repo_path(repo_full_name)
        encoded_user = urllib.parse.quote(username, safe="")
        try:
            resp = self.client.get(f"/repos/{owner}/{repo}/collaborators/{encoded_user}/permission")
        except httpx.HTTPError:
            return {}
        if resp.status_code != 200:
            return {}
        try:
            data: Any = resp.json()
        except ValueError:
            return {}
        return dict(data) if isinstance(data, Mapping) else {}

    def resolve_repo_allowed_scopes(
        self,
        repo_full_name: str,
        *,
        installation_acl: GitHubAppInstallationACL | None = None,
        teams: Sequence[Mapping[str, Any]] | None = None,
        collaborators: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[str]:
        """Resolves fail-closed RLS allowed_scopes for a GitHub repository."""
        clean_repo = sanitize_scope_identifier(repo_full_name)

        if installation_acl is not None:
            if installation_acl.suspended:
                return []
            contents_perm = installation_acl.permissions.get("contents", "").strip().lower()
            if contents_perm not in {"read", "write", "admin"}:
                return []
            if installation_acl.repository_selection == "selected":
                allowed_repos = {
                    sanitize_scope_identifier(r) for r in installation_acl.repositories
                }
                if clean_repo not in allowed_repos:
                    return []

        if teams is None and collaborators is None:
            return [f"repo:{clean_repo}:read"]

        default_org = clean_repo.split("/", 1)[0] if "/" in clean_repo else "org"
        scopes: list[str] = []

        for team in teams or ():
            perm = self._extract_permission_level(team)
            if perm not in _READ_PERMS:
                continue
            raw_org = str(team.get("org") or default_org)
            raw_slug = str(team.get("slug") or team.get("name") or "")
            if not raw_slug.strip():
                continue
            clean_org = sanitize_scope_identifier(raw_org)
            clean_slug = sanitize_scope_identifier(raw_slug)
            scopes.append(f"github:team:{clean_org}/{clean_slug}:read")
            if perm in _WRITE_PERMS:
                scopes.append(f"github:team:{clean_org}/{clean_slug}:write")
            if perm in _ADMIN_PERMS:
                scopes.append(f"github:team:{clean_org}/{clean_slug}:admin")

        for collab in collaborators or ():
            perm = self._extract_permission_level(collab)
            if perm not in _READ_PERMS:
                continue
            raw_user = str(
                collab.get("username") or collab.get("login") or collab.get("user") or ""
            )
            if not raw_user.strip():
                continue
            clean_user = sanitize_scope_identifier(raw_user)
            scopes.append(f"github:user:{clean_user}:read")
            if perm in _WRITE_PERMS:
                scopes.append(f"github:user:{clean_user}:write")

        return list(dict.fromkeys(scopes))

    def mirror_permissions(
        self,
        container_key: str,
        *,
        installation_acl: GitHubAppInstallationACL | None = None,
        teams: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[str]:
        """Maps GitHub repository and optional App/Team ACLs to RLS allowed_scopes."""
        return self.resolve_repo_allowed_scopes(
            container_key,
            installation_acl=installation_acl,
            teams=teams,
        )

    def fetch_item(self, item_id: str, version: str | None = None) -> dict[str, Any] | None:
        """Fetches repository metadata by 'owner/repo'."""
        owner, repo = self._normalize_repo_path(item_id)
        resp = self.client.get(f"/repos/{owner}/{repo}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data: Any = resp.json()
        return dict(data) if isinstance(data, Mapping) else None

    def compare_commits(
        self,
        repo_full_name: str,
        base_sha: str,
        head_sha: str,
    ) -> list[GitHubCommitDiff]:
        """Calls GitHub Compare API to retrieve modified file paths."""
        owner, repo = self._normalize_repo_path(repo_full_name)
        resp = self.client.get(f"/repos/{owner}/{repo}/compare/{base_sha}...{head_sha}")
        resp.raise_for_status()
        data: Any = resp.json()
        if not isinstance(data, Mapping):
            return []

        diffs: list[GitHubCommitDiff] = []
        for entry in data.get("files", []):
            if not isinstance(entry, Mapping):
                continue
            diffs.append(
                GitHubCommitDiff(
                    filename=str(entry.get("filename", "")),
                    previous_filename=(
                        str(entry["previous_filename"]) if entry.get("previous_filename") else None
                    ),
                    status=str(entry.get("status", "modified")),
                    patch=str(entry.get("patch", "")),
                    sha=str(entry.get("sha", "")),
                )
            )
        return diffs

    def fetch_raw_blob(self, repo_full_name: str, file_path: str, ref_sha: str) -> bytes:
        """Fetches raw blob bytes from GitHub Contents API."""
        owner, repo = self._normalize_repo_path(repo_full_name)
        quoted_path = urllib.parse.quote(file_path.lstrip("/"), safe="/")
        resp = self.client.get(
            f"/repos/{owner}/{repo}/contents/{quoted_path}",
            params={"ref": ref_sha},
            headers={"Accept": "application/vnd.github.raw+json"},
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type.lower():
            try:
                payload = resp.json()
                if isinstance(payload, Mapping) and isinstance(payload.get("content"), str):
                    return base64.b64decode(payload["content"])
            except ValueError:
                pass
        return resp.content

    def process_push_webhook(
        self,
        task: QueueTask,
        repository: PostgresRLSRepository | Any,
        conn: Any = None,
        principal: Any = None,
        *,
        installation_acl: GitHubAppInstallationACL | None = None,
    ) -> SyncResult:
        """Processes github_push queue task using incremental Merkle diffing and ACL checks."""
        start_time = datetime.now(UTC)
        payload = task.payload
        domain_id = task.domain_id

        repo_info = payload.get("repository", {})
        repo_full_name = str(repo_info.get("full_name") or payload.get("repo_name") or "").strip()
        before_sha = payload.get("before") or payload.get("before_sha")
        after_sha = payload.get("after") or payload.get("after_sha")

        if not repo_full_name or not before_sha or not after_sha:
            return SyncResult(
                source_type="github",
                items_scanned=0,
                items_indexed=0,
                items_updated=0,
                items_deleted=0,
                stale_items_flagged=0,
                cursor_or_checkpoint=str(after_sha or ""),
                duration_ms=0.0,
                errors=["Incomplete GitHub push webhook payload: missing repo or SHAs"],
            )

        clean_repo = sanitize_scope_identifier(repo_full_name)
        raw_acl = payload.get("installation_acl")
        effective_acl = installation_acl
        if effective_acl is None and isinstance(raw_acl, Mapping):
            effective_acl = GitHubAppInstallationACL.model_validate(raw_acl)

        teams_payload = payload.get("teams")
        collaborators_payload = payload.get("collaborators")
        allowed_scopes = self.resolve_repo_allowed_scopes(
            clean_repo,
            installation_acl=effective_acl,
            teams=teams_payload if isinstance(teams_payload, Sequence) else None,
            collaborators=(
                collaborators_payload if isinstance(collaborators_payload, Sequence) else None
            ),
        )
        if not allowed_scopes:
            logger.warning("GitHub push webhook failed closed for %s", clean_repo)
            return SyncResult(
                source_type="github",
                items_scanned=0,
                items_indexed=0,
                items_updated=0,
                items_deleted=0,
                stale_items_flagged=0,
                cursor_or_checkpoint=str(after_sha),
                duration_ms=0.0,
                errors=[f"Repository '{clean_repo}' failed closed: ACL denied access"],
            )

        default_branch = str(repo_info.get("default_branch", "main"))
        ref = str(payload.get("ref", ""))
        raw_principal_tid = getattr(principal, "tenant_id", None) if principal is not None else None
        raw_task_tid = getattr(task, "tenant_id", DEFAULT_TENANT_ID)
        effective_tenant_id: UUID = (
            raw_principal_tid
            if isinstance(raw_principal_tid, UUID)
            else (raw_task_tid if isinstance(raw_task_tid, UUID) else DEFAULT_TENANT_ID)
        )
        is_postgres = hasattr(repository, "upsert_repository") and conn is not None

        repo_rec = RepositoryRecord(
            repo_id=uuid4(),
            domain_id=domain_id,
            repo_name=clean_repo,
            source_type="github",
            remote_url=str(repo_info.get("clone_url") or f"{self.base_url}/{clean_repo}.git"),
            default_branch=default_branch,
            allowed_scopes=allowed_scopes,
            head_commit_sha=str(after_sha),
            metadata_json={
                "repo_id": repo_info.get("id"),
                "installation_id": self.installation_id,
                "last_push_ref": ref,
            },
            is_active=True,
            updated_at=datetime.now(UTC),
            tenant_id=effective_tenant_id,
        )
        if is_postgres:
            upsert_fn = getattr(repository, "upsert_repository", None)
            if callable(upsert_fn):
                upsert_fn(conn, repo_rec)
            if hasattr(repository, "get_repository"):
                existing_repo = repository.get_repository(conn, domain_id, clean_repo)
                if existing_repo is not None:
                    repo_rec = dataclasses.replace(repo_rec, repo_id=existing_repo.repo_id)
        elif hasattr(repository, "_repos"):
            for old_id, r in list(repository._repos.items()):
                if (
                    r.tenant_id == effective_tenant_id
                    and r.domain_id == domain_id
                    and r.repo_name in (clean_repo, repo_full_name)
                ):
                    repository._repos.pop(old_id, None)
            repository._repos[repo_rec.repo_id] = repo_rec

        for commit in payload.get("commits", []):
            if not isinstance(commit, Mapping):
                continue
            commit_sha = commit.get("id")
            if commit_sha:
                author = commit.get("author", {})
                c_rec = RepoCommitRecord(
                    commit_id=uuid4(),
                    repo_id=repo_rec.repo_id,
                    commit_sha=str(commit_sha),
                    parent_shas=[],
                    author_name=str(author.get("name", "Unknown"))
                    if isinstance(author, Mapping)
                    else "Unknown",
                    author_email=str(author.get("email", ""))
                    if isinstance(author, Mapping)
                    else "",
                    commit_message=str(commit.get("message", "")),
                    committed_at=datetime.fromisoformat(
                        str(commit.get("timestamp", datetime.now(UTC).isoformat())).replace(
                            "Z", "+00:00"
                        )
                    ),
                    indexed_at=datetime.now(UTC),
                    tenant_id=effective_tenant_id,
                )
                if is_postgres:
                    repository.insert_repo_commit(conn, c_rec)
                elif hasattr(repository, "_commits"):
                    repository._commits[c_rec.commit_id] = c_rec

        items_indexed = 0
        items_deleted = 0
        is_initial_push = str(before_sha).strip("0") == ""
        files_to_index: list[tuple[str, bool]] = []

        if is_initial_push:
            owner, repo = self._normalize_repo_path(clean_repo)
            try:
                tree_resp = self.client.get(
                    f"/repos/{owner}/{repo}/git/trees/{after_sha}",
                    params={"recursive": "1"},
                )
                if tree_resp.status_code == 200:
                    tree_data = tree_resp.json()
                    for item in tree_data.get("tree", []):
                        if isinstance(item, Mapping) and item.get("type") == "blob":
                            path = item.get("path")
                            if isinstance(path, str):
                                files_to_index.append((path, False))
            except Exception as exc:
                logger.warning("Failed to fetch GitHub tree for initial push: %s", exc)
        else:
            diffs = self.compare_commits(clean_repo, str(before_sha), str(after_sha))
            for diff in diffs:
                if diff.status == "removed":
                    files_to_index.append((diff.filename, True))
                elif diff.status == "renamed" and diff.previous_filename:
                    files_to_index.append((diff.previous_filename, True))
                    files_to_index.append((diff.filename, False))
                else:
                    files_to_index.append((diff.filename, False))

        extractor = ASTExtractor()
        for file_path, is_deleted in files_to_index:
            if is_deleted:
                self._delete_existing_file(
                    repository, conn, domain_id, clean_repo, repo_full_name, file_path, is_postgres
                )
                items_deleted += 1
                continue

            raw_bytes = self.fetch_raw_blob(clean_repo, file_path, str(after_sha))
            blob_sha = compute_git_blob_sha(raw_bytes)
            raw_content = raw_bytes.decode("utf-8", errors="replace")
            scrubbed_content = self.scrubber.scrub_text(
                raw_content, source_name=f"{clean_repo}:{file_path}"
            ).scrubbed_text

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
                repo_name=clean_repo,
                file_path=file_path,
                git_commit_sha=str(after_sha),
                git_blob_sha=blob_sha,
                language=lang,
                allowed_scopes=allowed_scopes,
                raw_content=scrubbed_content,
                line_count=len(scrubbed_content.splitlines()),
                repo_id=repo_rec.repo_id,
                tenant_id=effective_tenant_id,
            )

            self._delete_existing_file(
                repository, conn, domain_id, clean_repo, repo_full_name, file_path, is_postgres
            )

            if is_postgres:
                repository.insert_code_file(conn, file_rec)
            elif hasattr(repository, "insert_code_file") and principal is not None:
                repository.insert_code_file(principal, file_rec)
            elif hasattr(repository, "_files"):
                repository._files[file_rec.file_id] = file_rec

            created_sym_records: list[ASTSymbolRecord] = []
            if lang != "text":
                from context_scout.retrieval.embedder import get_embedding_provider

                symbols, _edges = extractor.parse_source(
                    scrubbed_content, lang, file_path=file_path
                )
                if symbols:
                    embedder = get_embedding_provider()
                    sym_texts = [
                        f"{clean_repo}::{sym.name} {sym.signature or sym.name}".strip()
                        for sym in symbols
                    ]
                    sym_embeddings = embedder.embed_batch(sym_texts, is_query=False)
                    for sym, sym_vec in zip(symbols, sym_embeddings, strict=True):
                        sym_rec = ASTSymbolRecord(
                            symbol_id=uuid4(),
                            file_id=file_rec.file_id,
                            domain_id=domain_id,
                            qualified_name=f"{clean_repo}::{sym.name}",
                            symbol_kind=sym.symbol_kind,
                            start_line=sym.start_line,
                            end_line=sym.end_line,
                            signature_skeleton=sym.signature or sym.name,
                            contextual_summary="",
                            allowed_scopes=allowed_scopes,
                            embedding=sym_vec,
                            repo_id=repo_rec.repo_id,
                            tenant_id=effective_tenant_id,
                        )
                        created_sym_records.append(sym_rec)
                        if is_postgres and hasattr(repository, "insert_ast_symbol"):
                            repository.insert_ast_symbol(conn, sym_rec)
                        elif hasattr(repository, "insert_ast_symbol") and principal is not None:
                            repository.insert_ast_symbol(principal, sym_rec)
                        elif hasattr(repository, "_symbols"):
                            repository._symbols[sym_rec.symbol_id] = sym_rec

            side_effects = self.side_effect_extractor.extract_from_code_file(
                file_rec, created_sym_records
            )
            if side_effects:
                if is_postgres and hasattr(repository, "upsert_side_effects_batch"):
                    repository.delete_side_effects_for_file(conn, domain_id, clean_repo, file_path)
                    repository.upsert_side_effects_batch(conn, side_effects)
                elif hasattr(repository, "upsert_side_effects_batch") and principal is not None:
                    repository.delete_side_effects_for_file(
                        principal, domain_id, clean_repo, file_path
                    )
                    repository.upsert_side_effects_batch(principal, side_effects)
                elif hasattr(repository, "_side_effects"):
                    for old_se_id, old_se in list(repository._side_effects.items()):
                        if (
                            old_se.domain_id == domain_id
                            and old_se.repo_name in (repo_full_name, clean_repo)
                            and old_se.file_path == file_path
                        ):
                            repository._side_effects.pop(old_se_id, None)
                    for se in side_effects:
                        repository._side_effects[se.side_effect_id] = se

            items_indexed += 1

        duration = (datetime.now(UTC) - start_time).total_seconds() * 1000.0
        return SyncResult(
            source_type="github",
            items_scanned=len(payload.get("commits", [])),
            items_indexed=items_indexed,
            items_updated=1,
            items_deleted=items_deleted,
            stale_items_flagged=0,
            cursor_or_checkpoint=str(after_sha),
            duration_ms=round(duration, 2),
        )

    @staticmethod
    def _delete_existing_file(
        repository: Any,
        conn: Any,
        domain_id: str,
        clean_repo: str,
        raw_repo: str,
        file_path: str,
        is_postgres: bool,
    ) -> None:
        """Removes stale code file and symbol records prior to re-indexing or deletion."""
        if is_postgres:
            if hasattr(repository, "delete_code_file_and_symbols"):
                repository.delete_code_file_and_symbols(conn, domain_id, clean_repo, file_path)
            else:
                conn.execute(
                    """
                    DELETE FROM code_files
                    WHERE domain_id = %s
                      AND (repo_name = %s OR repo_name = %s)
                      AND file_path = %s;
                    """,
                    (domain_id, raw_repo, clean_repo, file_path),
                )
        elif (
            del_fn := getattr(repository, "delete_code_file_and_symbols", None)
        ) is not None and callable(del_fn):
            del_fn(domain_id, clean_repo, file_path)
        elif hasattr(repository, "_files"):
            del_ids = [
                fid
                for fid, f in repository._files.items()
                if f.domain_id == domain_id
                and f.repo_name in (raw_repo, clean_repo)
                and f.file_path == file_path
            ]
            for fid in del_ids:
                repository._files.pop(fid, None)
                if hasattr(repository, "_symbols"):
                    sym_del_ids = [
                        sid for sid, sym in repository._symbols.items() if sym.file_id == fid
                    ]
                    for sid in sym_del_ids:
                        repository._symbols.pop(sid, None)
                if hasattr(repository, "_side_effects"):
                    se_del_ids = [
                        se_id for se_id, se in repository._side_effects.items() if se.file_id == fid
                    ]
                    for se_id in se_del_ids:
                        repository._side_effects.pop(se_id, None)

    def process_pull_request_webhook(
        self,
        task: QueueTask,
        repository: PostgresRLSRepository | Any,
        conn: Any = None,
        principal: Any = None,
        *,
        installation_acl: GitHubAppInstallationACL | None = None,
    ) -> SyncResult:
        """Processes Pull Request webhook: indexes merged state into target branch."""
        payload = task.payload
        action = str(payload.get("action", "")).strip().lower()
        pr = payload.get("pull_request", {})
        is_merged = isinstance(pr, Mapping) and bool(pr.get("merged", False))

        if action == "closed" and is_merged:
            base_sha = (
                pr.get("base", {}).get("sha") if isinstance(pr.get("base"), Mapping) else None
            ) or "0000000000000000000000000000000000000000"
            merge_commit = pr.get("merge_commit_sha") or (
                pr.get("head", {}).get("sha") if isinstance(pr.get("head"), Mapping) else ""
            )
            target_ref = (
                pr.get("base", {}).get("ref", "main")
                if isinstance(pr.get("base"), Mapping)
                else "main"
            )
            push_task = QueueTask(
                task_id=task.task_id,
                domain_id=task.domain_id,
                event_type="github_push",
                payload={
                    "repository": payload.get("repository", {}),
                    "repo_name": (
                        payload.get("repository", {}).get("full_name")
                        if isinstance(payload.get("repository"), Mapping)
                        else payload.get("repo_name")
                    ),
                    "ref": f"refs/heads/{target_ref}",
                    "before": base_sha,
                    "after": merge_commit,
                    "commits": [{"id": merge_commit}] if merge_commit else [],
                    "installation_acl": payload.get("installation_acl"),
                    "teams": payload.get("teams"),
                    "collaborators": payload.get("collaborators"),
                },
                priority=20,
                status="PROCESSING",
                worker_id=task.worker_id,
                attempts=task.attempts,
                max_attempts=task.max_attempts,
                error_message=None,
                scheduled_at=task.scheduled_at,
                created_at=task.created_at,
                tenant_id=task.tenant_id,
            )
            return self.process_push_webhook(
                push_task,
                repository,
                conn,
                principal,
                installation_acl=installation_acl,
            )

        return SyncResult(
            source_type="github",
            items_scanned=0,
            items_indexed=0,
            items_updated=0,
            items_deleted=0,
            stale_items_flagged=0,
            cursor_or_checkpoint="",
            duration_ms=0.0,
        )
