"""Background indexing worker with signal handling and event dispatching (F07)."""

from __future__ import annotations

import logging
import signal
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from context_scout.db.repositories import PostgresRLSRepository
from context_scout.db.schema import JiraIssueRecord
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.repo_wiki_builder import RepoWikiBuilder
from context_scout.plugins.connector_sdk import (
    ConnectorRateLimitError,
    ConnectorSyncCoordinator,
    RedisRateLimitCoordinator,
)
from context_scout.queue.task_queue import QueueTask, TaskQueueProtocol
from context_scout.schemas.contracts import PrincipalContext, RoleType

logger = logging.getLogger(__name__)


class IndexingWorker:
    """Production background worker claiming tasks via SKIP LOCKED and dispatching events."""

    def __init__(
        self,
        task_queue: TaskQueueProtocol,
        worker_id: str | None = None,
        poll_interval_seconds: float = 1.0,
        max_idle_backoff_seconds: float = 5.0,
        stale_reclaim_interval_seconds: float = 60.0,
        *,
        repository: Any | None = None,
        connection_manager: RLSConnectionManager | None = None,
        gitlab_connector: Any | None = None,
        github_connector: Any | None = None,
        confluence_connector: Any | None = None,
        jira_connector: Any | None = None,
        sync_coordinator: ConnectorSyncCoordinator | None = None,
        rate_limiter: RedisRateLimitCoordinator | None = None,
        artifact_purger: Any | None = None,
        zoekt_index: Any | None = None,
        object_store: Any | None = None,
        build_repo_wiki: bool = True,
        allowed_scopes: Sequence[str] = ("public",),
    ) -> None:
        self.task_queue = task_queue
        self.worker_id = worker_id or f"scout-worker-{uuid4().hex[:8]}"
        self.poll_interval = poll_interval_seconds
        self.max_idle_backoff = max_idle_backoff_seconds
        self.stale_reclaim_interval = stale_reclaim_interval_seconds

        self.repository = repository
        self.connection_manager = connection_manager
        self.gitlab_connector = gitlab_connector
        self.github_connector = github_connector
        self.confluence_connector = confluence_connector
        self.jira_connector = jira_connector
        self.rate_limiter = rate_limiter or (
            sync_coordinator.rate_limiter
            if sync_coordinator is not None
            else RedisRateLimitCoordinator()
        )
        self.artifact_purger = artifact_purger
        self.zoekt_index = zoekt_index
        self.object_store = object_store
        self.build_repo_wiki = build_repo_wiki
        self._default_allowed_scopes = [s.strip() for s in allowed_scopes if s and s.strip()] or [
            "public"
        ]

        if sync_coordinator is not None:
            self.sync_coordinator: ConnectorSyncCoordinator | None = sync_coordinator
        elif repository is not None or connection_manager is not None:
            repo_obj = repository or PostgresRLSRepository()
            self.repository = repo_obj
            self.sync_coordinator = ConnectorSyncCoordinator(
                repository=repo_obj,
                rate_limiter=self.rate_limiter,
                connection_manager=connection_manager,
                gitlab_connector=gitlab_connector,
                github_connector=github_connector,
                confluence_connector=confluence_connector,
                jira_connector=jira_connector,
            )
        else:
            self.sync_coordinator = None

        self._stop_event = threading.Event()
        self._current_task_id: UUID | None = None
        self._handlers: dict[str, Callable[[QueueTask], None]] = {}
        self._custom_event_types: set[str] = set()
        self._register_default_handlers()

    def register_handler(self, event_type: str, handler: Callable[[QueueTask], None]) -> None:
        """Registers or overrides an event dispatcher handler."""
        self._handlers[event_type] = handler
        self._custom_event_types.add(event_type)

    def stop(self) -> None:
        """Signals the worker loop to stop gracefully after completing current in-flight task."""
        logger.info("Worker %s received stop signal. Stopping gracefully...", self.worker_id)
        self._stop_event.set()

    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def install_signal_handlers(self) -> None:
        """Installs SIGINT and SIGTERM handlers to trigger graceful stop."""

        def _handle_signal(signum: int, _frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            logger.info("Signal %s received by worker %s", sig_name, self.worker_id)
            self.stop()

        try:
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
        except (ValueError, AttributeError) as exc:
            # Happens in secondary threads during testing
            logger.debug(
                "Skipping signal handler registration outside main thread (%s): %s",
                type(exc).__name__,
                exc,
            )

    def _assert_configured(self, event_type: str | None = None) -> None:
        if event_type is not None and event_type in self._custom_event_types:
            return
        if event_type == "purge_artifact" and self.artifact_purger is not None:
            return
        if (
            self.sync_coordinator is None
            and self.repository is None
            and self.connection_manager is None
        ):
            raise RuntimeError(
                "IndexingWorker requires a configured repository, connection_manager, "
                "or sync_coordinator before run_once() can process tasks."
            )

    def run_once(self) -> bool:
        """Attempts to claim and execute a single task. Returns True if task processed."""
        if not self._custom_event_types and self.artifact_purger is None:
            self._assert_configured()
        task: QueueTask | None = None
        try:
            task = self.task_queue.fetch_next_task(self.worker_id)
            if not task:
                return False

            self._assert_configured(task.event_type)
            self._current_task_id = task.task_id
            logger.info(
                "Worker %s claimed task %s [%s] (attempt %d/%d)",
                self.worker_id,
                task.task_id,
                task.event_type,
                task.attempts,
                task.max_attempts,
            )

            handler = self._handlers.get(task.event_type)
            if not handler:
                raise ValueError(f"No registered handler for event_type '{task.event_type}'")

            handler(task)
            self.task_queue.complete_task(task.task_id)
            logger.info("Worker %s successfully completed task %s", self.worker_id, task.task_id)
            return True

        except ConnectorRateLimitError as rl_exc:
            if task is not None:
                delay = self.rate_limiter.record_rate_limit(
                    task.tenant_id,
                    rl_exc.source_id,
                    rl_exc.retry_after_seconds,
                    container_key=rl_exc.container_key,
                )
                logger.warning(
                    "Worker %s deferred rate-limited task %s (%s) by %.2fs",
                    self.worker_id,
                    task.task_id,
                    rl_exc.source_id,
                    delay,
                )
                self.task_queue.defer_task(
                    task.task_id,
                    delay,
                    reason=str(rl_exc),
                    tenant_id=task.tenant_id,
                )
            return True

        except Exception as exc:
            if task is not None:
                err_msg = str(exc)
                tb = traceback.format_exc()
                logger.error(
                    "Worker %s failed executing task %s: %s\n%s",
                    self.worker_id,
                    task.task_id,
                    err_msg,
                    tb,
                )
                self.task_queue.fail_task(task.task_id, error_message=err_msg, stack_trace=tb)
            return True
        finally:
            self._current_task_id = None

    def run(self) -> None:
        """Main worker execution loop."""
        logger.info("Starting IndexingWorker %s", self.worker_id)
        idle_cycles = 0
        last_stale_reclaim = time.monotonic()

        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - last_stale_reclaim >= self.stale_reclaim_interval:
                try:
                    reclaimed = self.task_queue.reclaim_stale_tasks(stale_threshold_seconds=300)
                    if reclaimed > 0:
                        logger.warning(
                            "Worker %s reclaimed %d stale tasks", self.worker_id, reclaimed
                        )
                except Exception as exc:
                    logger.error("Error during stale task reclamation: %s", exc)
                last_stale_reclaim = now

            processed = self.run_once()
            if processed:
                idle_cycles = 0
            else:
                idle_cycles += 1
                backoff = min(
                    self.max_idle_backoff, self.poll_interval * (1.2 ** min(idle_cycles, 10))
                )
                self._stop_event.wait(timeout=backoff)

        logger.info("IndexingWorker %s stopped cleanly", self.worker_id)

    def _register_default_handlers(self) -> None:
        self._handlers["gitlab_push"] = self._handle_gitlab_push
        self._handlers["github_push"] = self._handle_github_push
        self._handlers["gitlab_mr"] = self._handle_gitlab_mr
        self._handlers["github_pr"] = self._handle_github_pr
        self._handlers["confluence_update"] = self._handle_confluence_update
        self._handlers["jira_update"] = self._handle_jira_update
        self._handlers["cross_repo_relink"] = self._handle_cross_repo_relink
        self._handlers["watermark_sync"] = self._handle_watermark_sync
        self._handlers["reconcile_sweep"] = self._handle_reconcile_sweep
        self._handlers["purge_artifact"] = self._handle_purge_artifact

    def _principal_for_task(self, task: QueueTask) -> PrincipalContext:
        raw_scopes = task.payload.get("allowed_scopes")
        scopes = (
            [str(s).strip() for s in raw_scopes if str(s).strip()]
            if isinstance(raw_scopes, Sequence) and not isinstance(raw_scopes, (str, bytes))
            else list(self._default_allowed_scopes)
        )
        principal = PrincipalContext(
            user_id=f"worker:{self.worker_id}",
            role=RoleType.ENGINEER,
            allowed_scopes=scopes or ["public"],
            tenant_id=task.tenant_id,
        )
        if principal.tenant_id != task.tenant_id:
            raise ValueError(
                f"Worker principal tenant_id {principal.tenant_id} != "
                f"task tenant_id {task.tenant_id}"
            )
        return principal

    def _is_pg_repo(self) -> bool:
        return self.connection_manager is not None or (
            self.repository is not None
            and self.repository.__class__.__name__ == "PostgresRLSRepository"
        )

    def _run_with_tx(
        self,
        principal: PrincipalContext,
        fn: Callable[[Any | None], Any],
    ) -> Any:
        if self.connection_manager is not None and self._is_pg_repo():
            with self.connection_manager.scoped_transaction(principal) as conn:
                return fn(conn)
        return fn(None)

    def _relink_and_rebuild_wiki(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        repo_names: Sequence[str] = (),
        commit_sha: str = "HEAD",
        conn: Any | None = None,
    ) -> None:
        if self.sync_coordinator is None:
            return
        self.sync_coordinator._run_contract_relink(principal, domain_id, conn=conn)
        if not self.build_repo_wiki:
            return
        targets = [r.strip() for r in repo_names if r and r.strip()]
        if not targets and self.repository is not None:
            if conn is not None and hasattr(self.repository, "list_repositories"):
                targets = [r.repo_name for r in self.repository.list_repositories(conn, domain_id)]
            elif hasattr(self.repository, "list_repositories"):
                targets = [
                    r.repo_name for r in self.repository.list_repositories(principal, domain_id)
                ]
        builder = RepoWikiBuilder()
        for rname in dict.fromkeys(targets):
            builder.sync_repo_wiki_to_repository(
                domain_id=domain_id,
                repo_name=rname,
                repository=self.repository,
                conn=conn,
                principal=principal,
                commit_sha=commit_sha or "HEAD",
                tenant_id=principal.tenant_id,
            )

    def _handle_gitlab_push(self, task: QueueTask) -> None:
        project_obj = task.payload.get("project")
        repo_name = (
            str(project_obj.get("path_with_namespace") or project_obj.get("name") or "")
            if isinstance(project_obj, Mapping)
            else str(task.payload.get("repo_name") or "")
        )
        self.rate_limiter.check_or_raise(
            task.tenant_id,
            "gitlab",
            container_key=repo_name or "*",
        )
        if self.sync_coordinator is None:
            raise RuntimeError("gitlab_push task requires configured repository/sync_coordinator")
        if self.sync_coordinator.get_connector("gitlab") is None:
            raise RuntimeError("gitlab_push task requires registered 'gitlab' connector")

        principal = self._principal_for_task(task)
        commit_sha = str(task.payload.get("after") or task.payload.get("after_sha") or "HEAD")

        def _exec(conn: Any | None) -> None:
            assert self.sync_coordinator is not None
            res = self.sync_coordinator.run_tier1_webhook_event(
                task,
                principal=principal,
                conn=conn,
            )
            if res.items_indexed > 0 and repo_name:
                self._relink_and_rebuild_wiki(
                    principal,
                    task.domain_id,
                    repo_names=[repo_name],
                    commit_sha=commit_sha,
                    conn=conn,
                )

        self._run_with_tx(principal, _exec)

    def _handle_gitlab_mr(self, task: QueueTask) -> None:
        project_obj = task.payload.get("project")
        repo_name = (
            str(project_obj.get("path_with_namespace") or "")
            if isinstance(project_obj, Mapping)
            else str(task.payload.get("repo_name") or "")
        )
        self.rate_limiter.check_or_raise(
            task.tenant_id,
            "gitlab",
            container_key=repo_name or "*",
        )
        if self.sync_coordinator is None:
            raise RuntimeError("gitlab_mr task requires configured repository/sync_coordinator")
        if self.sync_coordinator.get_connector("gitlab") is None:
            raise RuntimeError("gitlab_mr task requires registered 'gitlab' connector")
        principal = self._principal_for_task(task)
        self._run_with_tx(
            principal,
            lambda conn: self.sync_coordinator.run_tier1_webhook_event(  # type: ignore[union-attr]
                task, principal=principal, conn=conn
            ),
        )

    def _handle_github_push(self, task: QueueTask) -> None:
        repo_obj = task.payload.get("repository")
        repo_name = (
            str(repo_obj.get("full_name") or repo_obj.get("name") or "")
            if isinstance(repo_obj, Mapping)
            else str(task.payload.get("repo_name") or "")
        )
        self.rate_limiter.check_or_raise(
            task.tenant_id,
            "github",
            container_key=repo_name or "*",
        )
        if self.sync_coordinator is None:
            raise RuntimeError("github_push task requires configured repository/sync_coordinator")
        if self.sync_coordinator.get_connector("github") is None:
            raise RuntimeError("github_push task requires registered 'github' connector")

        principal = self._principal_for_task(task)
        commit_sha = str(task.payload.get("after") or "HEAD")

        def _exec(conn: Any | None) -> None:
            assert self.sync_coordinator is not None
            res = self.sync_coordinator.run_tier1_webhook_event(
                task,
                principal=principal,
                conn=conn,
            )
            if res.items_indexed > 0 and repo_name:
                self._relink_and_rebuild_wiki(
                    principal,
                    task.domain_id,
                    repo_names=[repo_name],
                    commit_sha=commit_sha,
                    conn=conn,
                )

        self._run_with_tx(principal, _exec)

    def _handle_github_pr(self, task: QueueTask) -> None:
        repo_obj = task.payload.get("repository")
        repo_name = (
            str(repo_obj.get("full_name") or "")
            if isinstance(repo_obj, Mapping)
            else str(task.payload.get("repo_name") or "")
        )
        self.rate_limiter.check_or_raise(
            task.tenant_id,
            "github",
            container_key=repo_name or "*",
        )
        if self.sync_coordinator is None:
            raise RuntimeError("github_pr task requires configured repository/sync_coordinator")
        if self.sync_coordinator.get_connector("github") is None:
            raise RuntimeError("github_pr task requires registered 'github' connector")
        principal = self._principal_for_task(task)
        self._run_with_tx(
            principal,
            lambda conn: self.sync_coordinator.run_tier1_webhook_event(  # type: ignore[union-attr]
                task, principal=principal, conn=conn
            ),
        )

    def _handle_confluence_update(self, task: QueueTask) -> None:
        space_key = str(task.payload.get("space_key") or "*")
        page_id = str(task.payload.get("page_id") or "")
        self.rate_limiter.check_or_raise(
            task.tenant_id,
            "confluence",
            container_key=space_key,
        )
        if self.sync_coordinator is None:
            raise RuntimeError(
                "confluence_update task requires configured repository/sync_coordinator"
            )

        principal = self._principal_for_task(task)
        connector = self.sync_coordinator.get_connector("confluence")
        is_delete = bool(task.payload.get("deleted")) or (
            str(task.payload.get("event") or "").lower() == "page_removed"
        )
        if is_delete and page_id:

            def _tombstone(conn: Any | None) -> None:
                assert self.sync_coordinator is not None
                if connector is not None:
                    connector.tombstone_deleted_or_restricted_pages(
                        task.domain_id,
                        self.sync_coordinator.repository,
                        [page_id],
                        conn=conn,
                    )
                elif hasattr(self.sync_coordinator.repository, "purge_domain_artifacts"):
                    if conn is not None:
                        self.sync_coordinator.repository.purge_domain_artifacts(
                            conn, task.domain_id, confluence_page_id=page_id
                        )
                    else:
                        self.sync_coordinator.repository.purge_domain_artifacts(
                            principal, task.domain_id, confluence_page_id=page_id
                        )
                self.sync_coordinator.record_watermark(
                    principal,
                    domain_id=task.domain_id,
                    source_id="confluence",
                    container_key=space_key,
                    cursor=datetime.now(UTC).isoformat(),
                    sync_tier="webhook",
                    items_synced=0,
                    metadata={"deleted_page_id": page_id},
                    conn=conn,
                )

            self._run_with_tx(principal, _tombstone)
            return

        if connector is None:
            logger.info(
                "No Confluence connector attached for task %s (space=%s, page=%s)",
                task.task_id,
                space_key,
                page_id,
            )
            return

        self._run_with_tx(
            principal,
            lambda conn: self.sync_coordinator.run_tier1_webhook_event(  # type: ignore[union-attr]
                task, principal=principal, conn=conn
            ),
        )

    def _handle_jira_update(self, task: QueueTask) -> None:
        issue_obj = task.payload.get("issue")
        issue_key = str(
            task.payload.get("issue_key")
            or (issue_obj.get("key") if isinstance(issue_obj, Mapping) else "")
            or ""
        )
        project_key = str(
            task.payload.get("project_key")
            or (issue_key.split("-", 1)[0] if "-" in issue_key else "*")
        )
        self.rate_limiter.check_or_raise(
            task.tenant_id,
            "jira",
            container_key=project_key,
        )
        if self.sync_coordinator is None:
            raise RuntimeError("jira_update task requires configured repository/sync_coordinator")

        principal = self._principal_for_task(task)
        connector = self.sync_coordinator.get_connector("jira")
        is_delete = bool(task.payload.get("deleted")) or (
            str(task.payload.get("webhookEvent") or "").lower() == "jira:issue_deleted"
        )
        if is_delete and issue_key:

            def _delete_issue(conn: Any | None) -> None:
                assert self.sync_coordinator is not None
                self.sync_coordinator._call_purge(
                    principal,
                    task.domain_id,
                    conn=conn,
                    jira_issue_key=issue_key,
                )
                self.sync_coordinator.record_watermark(
                    principal,
                    domain_id=task.domain_id,
                    source_id="jira",
                    container_key=project_key,
                    cursor=datetime.now(UTC).isoformat(),
                    sync_tier="webhook",
                    items_synced=0,
                    metadata={"deleted_issue_key": issue_key},
                    conn=conn,
                )

            self._run_with_tx(principal, _delete_issue)
            return

        if connector is None:
            logger.info(
                "No Jira connector attached for task %s (project=%s, issue=%s)",
                task.task_id,
                project_key,
                issue_key,
            )
            return

        if isinstance(issue_obj, Mapping) and not task.payload.get("force_project_sync"):

            def _upsert_single(conn: Any | None) -> None:
                assert self.sync_coordinator is not None
                dto = connector._parse_jira_issue(dict(issue_obj), project_key)
                scopes = connector.resolve_issue_allowed_scopes(dto)
                summary_clean = connector.scrubber.scrub_text(
                    dto.summary, source_name=f"jira:{dto.issue_key}:summary"
                ).scrubbed_text
                rec = JiraIssueRecord(
                    issue_id=uuid4(),
                    tenant_id=principal.tenant_id,
                    domain_id=task.domain_id,
                    issue_key=dto.issue_key,
                    project_key=dto.project_key,
                    summary=summary_clean,
                    issue_type=dto.issue_type,
                    status=dto.status,
                    resolution=dto.resolution,
                    assignee_username=dto.assignee,
                    source_url=dto.source_url,
                    allowed_scopes=scopes,
                    updated_at=dto.updated_at,
                    indexed_at=datetime.now(UTC),
                    epic_key=dto.epic_key,
                    parent_issue_key=dto.parent_issue_key,
                    linked_issues=[lk.model_dump() for lk in dto.linked_issues],
                    linked_mrs=list(dto.linked_mrs),
                )
                repo = self.sync_coordinator.repository
                chunks = connector.chunk_issue_verbatim(dto, task.domain_id)
                if conn is not None:
                    repo.purge_domain_artifacts(conn, task.domain_id, jira_issue_key=dto.issue_key)
                    repo.insert_jira_issue(conn, rec)
                    if chunks:
                        repo.insert_doc_chunks_batch(conn, chunks)
                else:
                    if hasattr(repo, "purge_domain_artifacts"):
                        repo.purge_domain_artifacts(
                            principal, task.domain_id, jira_issue_key=dto.issue_key
                        )
                    repo.insert_jira_issue(principal, rec)
                    if chunks:
                        repo.insert_doc_chunks_batch(principal, chunks)
                self.sync_coordinator.record_watermark(
                    principal,
                    domain_id=task.domain_id,
                    source_id="jira",
                    container_key=dto.project_key,
                    cursor=dto.updated_at.isoformat(),
                    sync_tier="webhook",
                    items_synced=len(chunks),
                    metadata={"issue_key": dto.issue_key},
                    conn=conn,
                )

            self._run_with_tx(principal, _upsert_single)
            return

        self._run_with_tx(
            principal,
            lambda conn: self.sync_coordinator.run_tier1_webhook_event(  # type: ignore[union-attr]
                task, principal=principal, conn=conn
            ),
        )

    def _handle_cross_repo_relink(self, task: QueueTask) -> None:
        if self.sync_coordinator is None:
            raise RuntimeError(
                "cross_repo_relink task requires configured repository/sync_coordinator"
            )
        principal = self._principal_for_task(task)
        repo_name = str(task.payload.get("repo_name") or "")
        commit_sha = str(task.payload.get("commit_sha") or "HEAD")
        self._run_with_tx(
            principal,
            lambda conn: self._relink_and_rebuild_wiki(
                principal,
                task.domain_id,
                repo_names=[repo_name] if repo_name else (),
                commit_sha=commit_sha,
                conn=conn,
            ),
        )

    def _handle_watermark_sync(self, task: QueueTask) -> None:
        if self.sync_coordinator is None:
            raise RuntimeError(
                "watermark_sync task requires configured repository/sync_coordinator"
            )
        principal = self._principal_for_task(task)
        source_id = str(task.payload.get("source_id") or "")
        container_key = str(task.payload.get("container_key") or "*")
        since_cursor = task.payload.get("since_cursor")
        self._run_with_tx(
            principal,
            lambda conn: self.sync_coordinator.run_tier2_watermark_sync(  # type: ignore[union-attr]
                principal,
                task.domain_id,
                source_id,
                container_key,
                since_cursor=str(since_cursor) if since_cursor else None,
                conn=conn,
            ),
        )

    def _handle_reconcile_sweep(self, task: QueueTask) -> None:
        if self.sync_coordinator is None:
            raise RuntimeError(
                "reconcile_sweep task requires configured repository/sync_coordinator"
            )
        principal = self._principal_for_task(task)
        source_id = str(task.payload.get("source_id") or "")
        container_key = str(task.payload.get("container_key") or "*")
        raw_deleted = task.payload.get("deleted_ids") or ()
        deleted_ids = [str(x) for x in raw_deleted] if isinstance(raw_deleted, Sequence) else []
        self._run_with_tx(
            principal,
            lambda conn: self.sync_coordinator.run_tier3_reconciliation_sweep(  # type: ignore[union-attr]
                principal,
                task.domain_id,
                source_id,
                container_key,
                deleted_ids=deleted_ids,
                reconcile_contracts=bool(task.payload.get("reconcile_contracts", True)),
                rebuild_wiki=bool(task.payload.get("rebuild_wiki", self.build_repo_wiki)),
                conn=conn,
            ),
        )

    def _resolve_artifact_purger(self) -> Any:
        if self.artifact_purger is not None:
            return self.artifact_purger
        from context_scout.config import ScoutSettings
        from context_scout.ops.artifact_purger import ArtifactPurgeCoordinator

        resolved_zoekt = self.zoekt_index
        if resolved_zoekt is None:
            settings = ScoutSettings()
            zoekt_ep = (settings.zoekt_endpoint or "").strip()
            if zoekt_ep:
                from context_scout.retrieval.zoekt_client import ZoektGrpcJsonClient

                resolved_zoekt = ZoektGrpcJsonClient(
                    zoekt_ep,
                    protocol=settings.zoekt_protocol,
                    timeout_seconds=settings.zoekt_timeout_seconds,
                    repository=self.repository,
                    connection_manager=self.connection_manager,
                )

        return ArtifactPurgeCoordinator(
            repository=self.repository,
            connection_manager=self.connection_manager,
            zoekt_index=resolved_zoekt,
            object_store=self.object_store,
        )

    def _handle_purge_artifact(self, task: QueueTask) -> None:
        purger = self._resolve_artifact_purger()
        principal = self._principal_for_task(task)
        p = task.payload
        raw_obj_keys = p.get("object_keys")
        if raw_obj_keys is None:
            object_keys: list[str] = []
        elif isinstance(raw_obj_keys, Sequence) and not isinstance(raw_obj_keys, (str, bytes)):
            object_keys = ["" if k is None else str(k) for k in raw_obj_keys]
        else:
            raise ValueError("object_keys must be a sequence of object key strings")

        def _opt_str(key: str) -> str | None:
            val = p.get(key)
            return None if val is None else str(val)

        purger.purge(
            principal,
            task.domain_id,
            repo_name=_opt_str("repo_name"),
            file_path=_opt_str("file_path"),
            commit_sha=_opt_str("commit_sha"),
            blob_sha=_opt_str("blob_sha"),
            doc_id=_opt_str("doc_id"),
            confluence_page_id=_opt_str("confluence_page_id"),
            jira_issue_key=_opt_str("jira_issue_key"),
            object_keys=object_keys,
            include_derived_wiki=bool(p.get("include_derived_wiki", True)),
        )
