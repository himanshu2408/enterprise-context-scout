"""ConnectorSDK v2: 3-tier sync coordinator and Redis 429 Retry-After backoff."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Literal
from uuid import UUID, uuid4

import httpx

from context_scout.ast_engine.contract_linker import CrossBoundaryContractLinker
from context_scout.db.schema import ASTEdgeRecord, ConnectorWatermarkRecord
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.repo_wiki_builder import RepoWikiBuilder
from context_scout.plugins.base import SyncResult
from context_scout.queue.task_queue import QueueTask
from context_scout.schemas.contracts import PrincipalContext, RoleType
from context_scout.security.permission_mirror import _StdlibRESP2Client

logger = logging.getLogger(__name__)


class ConnectorRateLimitError(RuntimeError):
    """Raised when an upstream SaaS/SCM connector enforces HTTP 429 / Retry-After cooldown."""

    def __init__(
        self,
        source_id: str,
        retry_after_seconds: float,
        *,
        status_code: int = 429,
        container_key: str = "",
        message: str | None = None,
    ) -> None:
        self.source_id = source_id
        self.retry_after_seconds = max(0.5, float(retry_after_seconds))
        self.status_code = status_code
        self.container_key = container_key
        msg = message or (
            f"Connector '{source_id}' rate limited (HTTP {status_code}, "
            f"retry_after={self.retry_after_seconds:.2f}s, container={container_key or '*'})"
        )
        super().__init__(msg)


def parse_retry_after_seconds(
    header_value: str | int | float | None,
    *,
    now: datetime | None = None,
    default_seconds: float = 30.0,
    max_seconds: float = 3600.0,
) -> float:
    """Parse HTTP ``Retry-After`` delta-seconds or HTTP-date header cleanly."""
    if header_value is None:
        return max(0.5, min(float(default_seconds), float(max_seconds)))

    if isinstance(header_value, (int, float)):
        return max(0.5, min(float(header_value), float(max_seconds)))

    raw = str(header_value).strip()
    if not raw:
        return max(0.5, min(float(default_seconds), float(max_seconds)))

    try:
        numeric = float(raw)
        if not math.isfinite(numeric):
            return max(0.5, min(float(default_seconds), float(max_seconds)))
        return max(0.5, min(numeric, float(max_seconds)))
    except ValueError:
        pass

    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        ref = now or datetime.now(UTC)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=UTC)
        delta = (dt - ref).total_seconds()
        return max(0.5, min(delta, float(max_seconds)))
    except (TypeError, ValueError):
        return max(0.5, min(float(default_seconds), float(max_seconds)))


def raise_if_rate_limited(
    response: httpx.Response,
    *,
    source_id: str,
    container_key: str = "",
    default_seconds: float = 30.0,
) -> None:
    """Raise ``ConnectorRateLimitError`` on HTTP 429 or 503 with Retry-After header."""
    if response.status_code == 429 or (
        response.status_code == 503 and "Retry-After" in response.headers
    ):
        delay = parse_retry_after_seconds(
            response.headers.get("Retry-After"),
            default_seconds=default_seconds,
        )
        raise ConnectorRateLimitError(
            source_id=source_id,
            retry_after_seconds=delay,
            status_code=response.status_code,
            container_key=container_key,
        )


def _is_rate_limit_http_error(exc: httpx.HTTPStatusError) -> tuple[bool, float]:
    resp = exc.response
    if resp.status_code == 429 or (resp.status_code == 503 and "Retry-After" in resp.headers):
        return True, parse_retry_after_seconds(resp.headers.get("Retry-After"))
    return False, 0.0


def _parse_iso_datetime(cursor: str | None) -> datetime | None:
    if not cursor or not cursor.strip():
        return None
    raw = cursor.strip()
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


class RedisRateLimitCoordinator:
    """Shared Redis + in-memory 429 Retry-After cooldown coordinator across worker pods."""

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        redis_client: Any | None = None,
        time_fn: Callable[[], float] | None = None,
        key_prefix: str = "scout:rl",
    ) -> None:
        self._time_fn: Callable[[], float] = time_fn or time.time
        self._key_prefix = key_prefix.strip(":") or "scout:rl"
        self._lock = threading.RLock()
        self._memory_unlocks: dict[str, float] = {}
        self._redis: Any | None = redis_client
        self._owns_redis = False

        if self._redis is None and redis_url and redis_url.strip():
            try:
                self._redis = _StdlibRESP2Client(redis_url.strip())
                self._owns_redis = True
            except (OSError, ValueError) as exc:
                logger.warning(
                    "RedisRateLimitCoordinator failed to init Redis client (%s); "
                    "using local memory cooldown store",
                    type(exc).__name__,
                )
                self._redis = None

    def close(self) -> None:
        if self._owns_redis and self._redis is not None and hasattr(self._redis, "close"):
            try:
                self._redis.close()
            except OSError as exc:
                logger.debug("Ignoring Redis close error: %s", exc)

    def _build_key(self, tenant_id: UUID, source_id: str, container_key: str = "*") -> str:
        clean_src = source_id.strip().lower() or "unknown"
        clean_cont = container_key.strip() or "*"
        return f"{self._key_prefix}:{tenant_id}:{clean_src}:{clean_cont}"

    def _redis_get_unlock(self, key: str) -> float | None:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(key)
            if raw is None:
                return None
            val_str = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            return float(val_str)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Redis GET failed for rate-limit key '%s' (%s)", key, type(exc).__name__)
            return None

    def _redis_set_unlock(self, key: str, unlock_epoch: float, ttl_seconds: float) -> bool:
        if self._redis is None:
            return False
        ttl_int = max(1, math.ceil(ttl_seconds))
        payload = f"{unlock_epoch:.6f}"
        try:
            if hasattr(self._redis, "setex"):
                self._redis.setex(key, ttl_int, payload)
                return True
            if hasattr(self._redis, "set"):
                self._redis.set(key, payload, ex=ttl_int)
                return True
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(
                "Redis SETEX failed for rate-limit key '%s' (%s)", key, type(exc).__name__
            )
        return False

    def record_rate_limit(
        self,
        tenant_id: UUID,
        source_id: str,
        retry_after_seconds: float,
        *,
        container_key: str = "*",
    ) -> float:
        """Record an upstream 429 cooldown window and return the effective remaining delay."""
        now = float(self._time_fn())
        delay = max(0.5, min(float(retry_after_seconds), 3600.0))
        target_unlock = now + delay
        key = self._build_key(tenant_id, source_id, container_key)

        with self._lock:
            existing = self._redis_get_unlock(key)
            mem_existing = self._memory_unlocks.get(key)
            if mem_existing is not None and (existing is None or mem_existing > existing):
                existing = mem_existing
            if existing is not None and existing > target_unlock:
                target_unlock = existing
                delay = max(0.5, existing - now)

            self._memory_unlocks[key] = target_unlock
            self._redis_set_unlock(key, target_unlock, delay)
            return delay

    def remaining_backoff_seconds(
        self,
        tenant_id: UUID,
        source_id: str,
        *,
        container_key: str = "*",
    ) -> float:
        """Return remaining cooldown seconds (>0.0 if currently rate-limited)."""
        now = float(self._time_fn())
        keys = [self._build_key(tenant_id, source_id, "*")]
        clean_cont = container_key.strip() or "*"
        if clean_cont != "*":
            keys.append(self._build_key(tenant_id, source_id, clean_cont))

        max_remaining = 0.0
        with self._lock:
            for key in keys:
                unlock_at = self._redis_get_unlock(key)
                mem_val = self._memory_unlocks.get(key)
                if unlock_at is None or (mem_val is not None and mem_val > unlock_at):
                    unlock_at = mem_val
                if unlock_at is None:
                    continue
                rem = unlock_at - now
                if rem <= 0.0:
                    self._memory_unlocks.pop(key, None)
                elif rem > max_remaining:
                    max_remaining = rem
        return max_remaining

    def check_or_raise(
        self,
        tenant_id: UUID,
        source_id: str,
        *,
        container_key: str = "*",
    ) -> None:
        """Fail fast with ``ConnectorRateLimitError`` if cooldown window is active."""
        remaining = self.remaining_backoff_seconds(
            tenant_id,
            source_id,
            container_key=container_key,
        )
        if remaining > 0.0:
            raise ConnectorRateLimitError(
                source_id=source_id,
                retry_after_seconds=remaining,
                container_key=container_key,
                message=(
                    f"Shared cooldown active for '{source_id}' ({container_key or '*'}); "
                    f"defer {remaining:.2f}s"
                ),
            )

    def clear_backoff(
        self,
        tenant_id: UUID,
        source_id: str,
        *,
        container_key: str = "*",
    ) -> None:
        """Remove recorded cooldown for a source/container."""
        key = self._build_key(tenant_id, source_id, container_key)
        with self._lock:
            self._memory_unlocks.pop(key, None)
            if self._redis is not None and hasattr(self._redis, "delete"):
                try:
                    self._redis.delete(key)
                except (OSError, ValueError, TypeError) as exc:
                    logger.debug("Redis DELETE ignored for '%s': %s", key, exc)


class ConnectorSyncCoordinator:
    """ConnectorSDK v2 3-tier orchestrator (Tier 1 webhook, Tier 2 watermark, Tier 3 reconcile)."""

    def __init__(
        self,
        *,
        repository: Any,
        rate_limiter: RedisRateLimitCoordinator | None = None,
        connection_manager: RLSConnectionManager | None = None,
        gitlab_connector: Any | None = None,
        github_connector: Any | None = None,
        confluence_connector: Any | None = None,
        jira_connector: Any | None = None,
        extra_connectors: Mapping[str, Any] | None = None,
    ) -> None:
        self.repository = repository
        self.rate_limiter = rate_limiter or RedisRateLimitCoordinator()
        self.connection_manager = connection_manager
        self._connectors: dict[str, Any] = {}
        if gitlab_connector is not None:
            self._connectors["gitlab"] = gitlab_connector
        if github_connector is not None:
            self._connectors["github"] = github_connector
        if confluence_connector is not None:
            self._connectors["confluence"] = confluence_connector
        if jira_connector is not None:
            self._connectors["jira"] = jira_connector
        if extra_connectors:
            for k, v in extra_connectors.items():
                self._connectors[k.strip().lower()] = v

    def get_connector(self, source_id: str) -> Any | None:
        return self._connectors.get(source_id.strip().lower())

    def register_connector(self, source_id: str, connector: Any) -> None:
        self._connectors[source_id.strip().lower()] = connector

    @staticmethod
    def _default_principal(
        tenant_id: UUID, allowed_scopes: Sequence[str] = ("public",)
    ) -> PrincipalContext:
        clean_scopes = [s.strip() for s in allowed_scopes if s and s.strip()] or ["public"]
        return PrincipalContext(
            user_id="connector_sync_coordinator",
            role=RoleType.ENGINEER,
            allowed_scopes=clean_scopes,
            tenant_id=tenant_id,
        )

    def _is_postgres_repo(self) -> bool:
        return self.connection_manager is not None or (
            hasattr(self.repository, "upsert_connector_watermark")
            and self.repository.__class__.__name__ == "PostgresRLSRepository"
        )

    def get_watermark(
        self,
        principal: PrincipalContext,
        domain_id: str,
        source_id: str,
        container_key: str,
        *,
        conn: Any | None = None,
    ) -> ConnectorWatermarkRecord | None:
        if not hasattr(self.repository, "get_connector_watermark"):
            return None
        if conn is not None:
            return self.repository.get_connector_watermark(  # type: ignore[no-any-return]
                conn, domain_id, source_id, container_key
            )
        if self.connection_manager is not None and self._is_postgres_repo():
            with self.connection_manager.scoped_transaction(principal) as tx_conn:
                return self.repository.get_connector_watermark(  # type: ignore[no-any-return]
                    tx_conn, domain_id, source_id, container_key
                )
        return self.repository.get_connector_watermark(  # type: ignore[no-any-return]
            principal, domain_id, source_id, container_key
        )

    def record_watermark(
        self,
        principal: PrincipalContext,
        *,
        domain_id: str,
        source_id: str,
        container_key: str,
        cursor: str,
        sync_tier: Literal["webhook", "watermark", "reconcile", "bundle"],
        items_synced: int,
        metadata: Mapping[str, Any] | None = None,
        conn: Any | None = None,
    ) -> ConnectorWatermarkRecord:
        now = datetime.now(UTC)
        rec = ConnectorWatermarkRecord(
            tenant_id=principal.tenant_id,
            domain_id=domain_id,
            source_id=source_id,
            container_key=container_key,
            watermark_cursor=cursor,
            sync_tier=sync_tier,
            items_synced=items_synced,
            last_webhook_at=now if sync_tier == "webhook" else None,
            last_watermark_sync_at=now if sync_tier == "watermark" else None,
            last_reconciliation_at=now if sync_tier == "reconcile" else None,
            metadata_json=dict(metadata or {}),
            updated_at=now,
        )
        if hasattr(self.repository, "upsert_connector_watermark"):
            if conn is not None:
                self.repository.upsert_connector_watermark(conn, rec)
            elif self.connection_manager is not None and self._is_postgres_repo():
                with self.connection_manager.scoped_transaction(principal) as tx_conn:
                    self.repository.upsert_connector_watermark(tx_conn, rec)
            else:
                self.repository.upsert_connector_watermark(principal, rec)
        return rec

    def run_tier1_webhook_event(
        self,
        task: QueueTask,
        *,
        principal: PrincipalContext | None = None,
        conn: Any | None = None,
    ) -> SyncResult:
        """Tier 1 real-time webhook execution with shared 429 rate-limit enforcement."""
        eff_principal = principal or self._default_principal(task.tenant_id)
        payload = task.payload
        event_type = task.event_type

        if event_type in ("gitlab_push", "gitlab_mr"):
            source_id = "gitlab"
            container_key = str(
                payload.get("project", {}).get("path_with_namespace")
                if isinstance(payload.get("project"), Mapping)
                else payload.get("repo_name") or payload.get("project_id") or "*"
            )
        elif event_type in ("github_push", "github_pr"):
            source_id = "github"
            container_key = str(
                payload.get("repository", {}).get("full_name")
                if isinstance(payload.get("repository"), Mapping)
                else payload.get("repo_name") or "*"
            )
        elif event_type == "confluence_update":
            source_id = "confluence"
            container_key = str(payload.get("space_key") or "*")
        elif event_type == "jira_update":
            source_id = "jira"
            container_key = str(payload.get("project_key") or "*")
        else:
            raise ValueError(f"Unsupported Tier-1 webhook event_type '{event_type}'")

        self.rate_limiter.check_or_raise(
            eff_principal.tenant_id,
            source_id,
            container_key=container_key,
        )

        connector = self.get_connector(source_id)
        if connector is None:
            raise RuntimeError(f"No connector configured for Tier-1 event '{event_type}'")

        try:
            if event_type == "gitlab_push":
                result: SyncResult = connector.process_push_webhook(
                    task, self.repository, conn=conn, principal=eff_principal
                )
            elif event_type == "gitlab_mr":
                result = connector.process_merge_request_webhook(
                    task, self.repository, conn=conn, principal=eff_principal
                )
            elif event_type == "github_push":
                result = connector.process_push_webhook(
                    task, self.repository, conn=conn, principal=eff_principal
                )
            elif event_type == "github_pr":
                result = connector.process_pull_request_webhook(
                    task, self.repository, conn=conn, principal=eff_principal
                )
            elif event_type == "confluence_update":
                result = connector.sync_space_to_repository(
                    task.domain_id,
                    container_key,
                    self.repository,
                    conn=conn,
                    principal=eff_principal,
                )
            else:
                since_dt = _parse_iso_datetime(str(payload.get("since_cursor") or ""))
                result = connector.sync_project_to_repository(
                    task.domain_id,
                    container_key,
                    self.repository,
                    conn=conn,
                    principal=eff_principal,
                    since_datetime=since_dt,
                )
        except ConnectorRateLimitError as rl_exc:
            self.rate_limiter.record_rate_limit(
                eff_principal.tenant_id,
                rl_exc.source_id,
                rl_exc.retry_after_seconds,
                container_key=rl_exc.container_key or container_key,
            )
            raise
        except httpx.HTTPStatusError as http_exc:
            is_rl, delay = _is_rate_limit_http_error(http_exc)
            if is_rl:
                self.rate_limiter.record_rate_limit(
                    eff_principal.tenant_id,
                    source_id,
                    delay,
                    container_key=container_key,
                )
                raise ConnectorRateLimitError(
                    source_id=source_id,
                    retry_after_seconds=delay,
                    status_code=http_exc.response.status_code,
                    container_key=container_key,
                ) from http_exc
            raise

        self.record_watermark(
            eff_principal,
            domain_id=task.domain_id,
            source_id=source_id,
            container_key=container_key,
            cursor=result.cursor_or_checkpoint or datetime.now(UTC).isoformat(),
            sync_tier="webhook",
            items_synced=result.items_indexed,
            metadata={"event_type": event_type, "task_id": str(task.task_id)},
            conn=conn,
        )
        return result

    def run_tier2_watermark_sync(
        self,
        principal: PrincipalContext,
        domain_id: str,
        source_id: str,
        container_key: str,
        *,
        since_cursor: str | None = None,
        conn: Any | None = None,
    ) -> SyncResult:
        """Tier 2 watermark cursor incremental polling for missed/batched deltas."""
        clean_src = source_id.strip().lower()
        self.rate_limiter.check_or_raise(
            principal.tenant_id,
            clean_src,
            container_key=container_key,
        )
        existing_wm = self.get_watermark(
            principal,
            domain_id,
            clean_src,
            container_key,
            conn=conn,
        )
        effective_cursor = since_cursor or (
            existing_wm.watermark_cursor if existing_wm is not None else ""
        )
        since_dt = _parse_iso_datetime(effective_cursor)

        connector = self.get_connector(clean_src)
        if connector is None:
            raise RuntimeError(f"No connector configured for Tier-2 watermark sync '{clean_src}'")

        try:
            if clean_src == "confluence":
                result: SyncResult = connector.sync_space_to_repository(
                    domain_id,
                    container_key,
                    self.repository,
                    conn=conn,
                    principal=principal,
                    since_updated_at=since_dt,
                )
            elif clean_src == "jira":
                result = connector.sync_project_to_repository(
                    domain_id,
                    container_key,
                    self.repository,
                    conn=conn,
                    principal=principal,
                    since_datetime=since_dt,
                )
            elif hasattr(connector, "sync_incremental"):
                result = connector.sync_incremental(
                    domain_id=domain_id,
                    container_key=container_key,
                    cursor=effective_cursor or None,
                    repository=self.repository,
                    conn=conn,
                    principal=principal,
                )
            else:
                raise RuntimeError(
                    f"Connector '{clean_src}' does not implement incremental watermark sync"
                )
        except ConnectorRateLimitError as rl_exc:
            self.rate_limiter.record_rate_limit(
                principal.tenant_id,
                rl_exc.source_id,
                rl_exc.retry_after_seconds,
                container_key=rl_exc.container_key or container_key,
            )
            raise
        except httpx.HTTPStatusError as http_exc:
            is_rl, delay = _is_rate_limit_http_error(http_exc)
            if is_rl:
                self.rate_limiter.record_rate_limit(
                    principal.tenant_id,
                    clean_src,
                    delay,
                    container_key=container_key,
                )
                raise ConnectorRateLimitError(
                    source_id=clean_src,
                    retry_after_seconds=delay,
                    status_code=http_exc.response.status_code,
                    container_key=container_key,
                ) from http_exc
            raise

        next_cursor = result.cursor_or_checkpoint or datetime.now(UTC).isoformat()
        self.record_watermark(
            principal,
            domain_id=domain_id,
            source_id=clean_src,
            container_key=container_key,
            cursor=next_cursor,
            sync_tier="watermark",
            items_synced=result.items_indexed,
            metadata={"previous_cursor": effective_cursor},
            conn=conn,
        )
        return result

    def run_tier3_reconciliation_sweep(
        self,
        principal: PrincipalContext,
        domain_id: str,
        source_id: str,
        container_key: str,
        *,
        deleted_ids: Sequence[str] = (),
        reconcile_contracts: bool = False,
        rebuild_wiki: bool = False,
        conn: Any | None = None,
    ) -> SyncResult:
        """Tier 3 periodic full reconciliation, deletion sweep, and graph/wiki relinking."""
        t0 = time.perf_counter()
        clean_src = source_id.strip().lower()
        self.rate_limiter.check_or_raise(
            principal.tenant_id,
            clean_src,
            container_key=container_key,
        )
        connector = self.get_connector(clean_src)

        items_deleted = 0
        items_indexed = 0
        items_scanned = 0
        cursor = datetime.now(UTC).isoformat()

        try:
            if clean_src == "confluence" and connector is not None:
                res = connector.sync_space_to_repository(
                    domain_id,
                    container_key,
                    self.repository,
                    conn=conn,
                    principal=principal,
                    reconcile_deletes=True,
                    deleted_page_ids=list(deleted_ids) if deleted_ids else None,
                )
                items_scanned = res.items_scanned
                items_indexed = res.items_indexed
                items_deleted = res.items_deleted
                cursor = res.cursor_or_checkpoint or cursor
            elif clean_src == "jira":
                if deleted_ids and hasattr(self.repository, "purge_domain_artifacts"):
                    for issue_key in deleted_ids:
                        counts = self._call_purge(
                            principal,
                            domain_id,
                            jira_issue_key=issue_key,
                            conn=conn,
                        )
                        items_deleted += sum(counts.values())
                if connector is not None and not deleted_ids:
                    res = connector.sync_project_to_repository(
                        domain_id,
                        container_key,
                        self.repository,
                        conn=conn,
                        principal=principal,
                        since_datetime=None,
                    )
                    items_scanned = res.items_scanned
                    items_indexed = res.items_indexed
                    cursor = res.cursor_or_checkpoint or cursor
            elif clean_src in {"gitlab", "github", "local_git"}:
                for rel_path in deleted_ids:
                    counts = self._call_purge(
                        principal,
                        domain_id,
                        repo_name=container_key,
                        file_path=rel_path,
                        conn=conn,
                    )
                    items_deleted += sum(counts.values())
            elif deleted_ids:
                for did in deleted_ids:
                    counts = self._call_purge(principal, domain_id, doc_id=did, conn=conn)
                    items_deleted += sum(counts.values())
        except ConnectorRateLimitError as rl_exc:
            self.rate_limiter.record_rate_limit(
                principal.tenant_id,
                rl_exc.source_id,
                rl_exc.retry_after_seconds,
                container_key=rl_exc.container_key or container_key,
            )
            raise
        except httpx.HTTPStatusError as http_exc:
            is_rl, delay = _is_rate_limit_http_error(http_exc)
            if is_rl:
                self.rate_limiter.record_rate_limit(
                    principal.tenant_id,
                    clean_src,
                    delay,
                    container_key=container_key,
                )
                raise ConnectorRateLimitError(
                    source_id=clean_src,
                    retry_after_seconds=delay,
                    status_code=http_exc.response.status_code,
                    container_key=container_key,
                ) from http_exc
            raise

        edges_added = 0
        if reconcile_contracts:
            edges_added = self._run_contract_relink(principal, domain_id, conn=conn)
        if rebuild_wiki and container_key and container_key != "*":
            if conn is None and self.connection_manager is not None and self._is_postgres_repo():
                with self.connection_manager.scoped_transaction(principal) as tx_conn:
                    wiki_chunks = RepoWikiBuilder().sync_repo_wiki_to_repository(
                        domain_id=domain_id,
                        repo_name=container_key,
                        repository=self.repository,
                        conn=tx_conn,
                        principal=principal,
                        tenant_id=principal.tenant_id,
                    )
            else:
                wiki_chunks = RepoWikiBuilder().sync_repo_wiki_to_repository(
                    domain_id=domain_id,
                    repo_name=container_key,
                    repository=self.repository,
                    conn=conn,
                    principal=principal,
                    tenant_id=principal.tenant_id,
                )
            items_indexed += len(wiki_chunks)

        self.record_watermark(
            principal,
            domain_id=domain_id,
            source_id=clean_src,
            container_key=container_key,
            cursor=cursor,
            sync_tier="reconcile",
            items_synced=items_indexed,
            metadata={
                "items_deleted": items_deleted,
                "contract_edges_added": edges_added,
                "deleted_ids": list(deleted_ids),
            },
            conn=conn,
        )
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return SyncResult(
            source_type=clean_src,
            items_scanned=items_scanned or len(deleted_ids),
            items_indexed=items_indexed,
            items_updated=1,
            items_deleted=items_deleted,
            stale_items_flagged=0,
            cursor_or_checkpoint=cursor,
            duration_ms=duration_ms,
        )

    def _call_purge(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        conn: Any | None = None,
        **selectors: Any,
    ) -> dict[str, int]:
        if conn is not None:
            return self.repository.purge_domain_artifacts(  # type: ignore[no-any-return]
                conn, domain_id, **selectors
            )
        if self.connection_manager is not None and self._is_postgres_repo():
            with self.connection_manager.scoped_transaction(principal) as tx_conn:
                return self.repository.purge_domain_artifacts(  # type: ignore[no-any-return]
                    tx_conn, domain_id, **selectors
                )
        return self.repository.purge_domain_artifacts(  # type: ignore[no-any-return]
            principal, domain_id, **selectors
        )

    def _run_contract_relink(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        conn: Any | None = None,
    ) -> int:
        def _execute_relink(target_arg: Any) -> int:
            files = (
                self.repository.list_visible_code_file_records(target_arg, domain_id)
                if hasattr(self.repository, "list_visible_code_file_records")
                else self.repository.list_visible_code_files(target_arg, domain_id)
            )
            syms = (
                self.repository.list_visible_ast_symbol_records(target_arg, domain_id)
                if hasattr(self.repository, "list_visible_ast_symbol_records")
                else self.repository.list_visible_ast_symbols(target_arg, domain_id)
            )
            docs = self.repository.list_visible_doc_chunks(target_arg, domain_id)
            existing_edges = (
                self.repository.list_visible_ast_edge_records(target_arg, domain_id)
                if hasattr(self.repository, "list_visible_ast_edge_records")
                else self.repository.list_visible_ast_edges(target_arg, domain_id)
                if hasattr(self.repository, "list_visible_ast_edges")
                else []
            )
            existing_keys = {
                (e.source_symbol_id, e.target_qualified_name, e.edge_kind)
                for e in existing_edges
                if hasattr(e, "source_symbol_id")
            }
            candidate_edges: list[ASTEdgeRecord] = CrossBoundaryContractLinker().link_contracts(
                domain_id,
                syms,
                files,
                docs,
                tenant_id=principal.tenant_id,
            )
            new_edges = [
                edge
                for edge in candidate_edges
                if (edge.source_symbol_id, edge.target_qualified_name, edge.edge_kind)
                not in existing_keys
            ]
            if not new_edges:
                return 0
            if hasattr(self.repository, "insert_ast_edge"):
                for edge in new_edges:
                    self.repository.insert_ast_edge(target_arg, edge)
            elif hasattr(self.repository, "_edges"):
                for edge in new_edges:
                    eid = edge.edge_id or uuid4()
                    self.repository._edges[eid] = edge
            return len(new_edges)

        if conn is not None:
            return _execute_relink(conn)
        if self.connection_manager is not None and self._is_postgres_repo():
            with self.connection_manager.scoped_transaction(principal) as tx_conn:
                return _execute_relink(tx_conn)
        return _execute_relink(principal)


__all__ = [
    "ConnectorRateLimitError",
    "ConnectorSyncCoordinator",
    "RedisRateLimitCoordinator",
    "parse_retry_after_seconds",
    "raise_if_rate_limited",
]
