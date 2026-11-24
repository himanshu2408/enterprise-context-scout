"""Transactional task queue with PostgreSQL 17 SKIP LOCKED, DLQ, and jittered backoff (F07)."""

from __future__ import annotations

import json
import logging
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Literal
from uuid import UUID, uuid4

from psycopg.rows import dict_row

from context_scout.db.session import PoolRole, RLSConnectionManager
from context_scout.schemas.contracts import DEFAULT_TENANT_ID, PrincipalContext, RoleType

logger = logging.getLogger(__name__)

EventType = Literal[
    "gitlab_push",
    "github_push",
    "gitlab_mr",
    "github_pr",
    "confluence_update",
    "jira_update",
    "cross_repo_relink",
    "watermark_sync",
    "reconcile_sweep",
    "purge_artifact",
]

TaskStatus = Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"]

VALID_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "gitlab_push",
        "github_push",
        "gitlab_mr",
        "github_pr",
        "confluence_update",
        "jira_update",
        "cross_repo_relink",
        "watermark_sync",
        "reconcile_sweep",
        "purge_artifact",
    }
)


@dataclass(frozen=True)
class QueueTask:
    task_id: UUID
    domain_id: str
    event_type: str
    payload: dict[str, Any]
    priority: int
    status: TaskStatus
    worker_id: str | None
    attempts: int
    max_attempts: int
    error_message: str | None
    scheduled_at: datetime
    created_at: datetime
    started_at: datetime | None = None
    processed_at: datetime | None = None
    tenant_id: UUID = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class DeadLetterRecord:
    dlq_id: UUID
    task_id: UUID
    domain_id: str
    event_type: str
    payload: dict[str, Any]
    failure_reason: str
    stack_trace: str
    attempts: int
    failed_at: datetime
    tenant_id: UUID = DEFAULT_TENANT_ID


def compute_backoff_delay(
    attempts: int,
    base_delay_seconds: float = 2.0,
    max_delay_seconds: float = 300.0,
) -> float:
    """Computes exponential backoff with Equal Jitter to eliminate retry storms."""
    if attempts <= 0:
        return 0.0
    delay: float = min(max_delay_seconds, base_delay_seconds * (2.0 ** (attempts - 1)))
    half_delay = delay / 2.0
    return float(half_delay + random.uniform(0.0, half_delay))


class TaskQueueProtocol(ABC):
    """Abstract interface defining transactional queue operations."""

    @abstractmethod
    def enqueue_task(
        self,
        domain_id: str,
        event_type: str,
        payload: dict[str, Any],
        priority: int = 10,
        max_attempts: int = 3,
        scheduled_at: datetime | None = None,
        tenant_id: UUID | None = None,
    ) -> UUID: ...

    @abstractmethod
    def fetch_next_task(
        self, worker_id: str, tenant_id: UUID | None = None
    ) -> QueueTask | None: ...

    @abstractmethod
    def complete_task(self, task_id: UUID, tenant_id: UUID | None = None) -> None: ...

    @abstractmethod
    def fail_task(
        self,
        task_id: UUID,
        error_message: str,
        stack_trace: str = "",
        base_delay_seconds: float = 2.0,
        max_delay_seconds: float = 300.0,
        tenant_id: UUID | None = None,
    ) -> bool: ...

    @abstractmethod
    def reclaim_stale_tasks(
        self, stale_threshold_seconds: int = 300, tenant_id: UUID | None = None
    ) -> int: ...

    @abstractmethod
    def get_queue_stats(self, tenant_id: UUID | None = None) -> dict[str, int]: ...

    @abstractmethod
    def list_dead_letter_records(
        self, limit: int = 50, tenant_id: UUID | None = None
    ) -> list[DeadLetterRecord]: ...

    @abstractmethod
    def retry_dead_letter_task(self, task_id: UUID, tenant_id: UUID | None = None) -> bool: ...

    @abstractmethod
    def defer_task(
        self,
        task_id: UUID,
        delay_seconds: float,
        reason: str = "Rate limited",
        tenant_id: UUID | None = None,
    ) -> bool: ...


class PostgresTaskQueue(TaskQueueProtocol):
    """PostgreSQL 17 SKIP LOCKED task queue running via ingestion_pool under FORCE RLS."""

    def __init__(
        self,
        db_url: str | None = None,
        *,
        connection_manager: RLSConnectionManager | None = None,
        default_tenant_id: UUID = DEFAULT_TENANT_ID,
    ) -> None:
        if connection_manager is not None:
            self._conn_mgr = connection_manager
            self.db_url = connection_manager._app_db_url
        else:
            if not db_url:
                from context_scout.config import ScoutSettings

                db_url = ScoutSettings().db_app_url
            self.db_url = db_url
            self._conn_mgr = RLSConnectionManager(
                db_url,
                use_pool=True,
                pool_role=PoolRole.INGESTION,
            )
        self.default_tenant_id = default_tenant_id

    def _queue_principal(self, tenant_id: UUID | None = None) -> PrincipalContext:
        return PrincipalContext(
            user_id="indexing_queue_worker",
            role=RoleType.ENGINEER,
            allowed_scopes=["public"],
            tenant_id=tenant_id or self.default_tenant_id,
        )

    def enqueue_task(
        self,
        domain_id: str,
        event_type: str,
        payload: dict[str, Any],
        priority: int = 10,
        max_attempts: int = 3,
        scheduled_at: datetime | None = None,
        tenant_id: UUID | None = None,
    ) -> UUID:
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid event_type '{event_type}'. Must be one of {sorted(VALID_EVENT_TYPES)}"
            )

        effective_tid = tenant_id or self.default_tenant_id
        task_id = uuid4()
        payload_bytes = json.dumps(payload)
        principal = self._queue_principal(effective_tid)

        with (
            self._conn_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn,
            conn.cursor() as cur,
        ):
            cur.execute(
                """
                INSERT INTO indexing_queue (
                    task_id, tenant_id, domain_id, event_type, payload_json,
                    priority, max_attempts, scheduled_at, status
                )
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, COALESCE(%s, NOW()), 'PENDING')
                RETURNING task_id;
                """,
                (
                    task_id,
                    effective_tid,
                    domain_id,
                    event_type,
                    payload_bytes,
                    priority,
                    max_attempts,
                    scheduled_at,
                ),
            )
        return task_id

    def fetch_next_task(self, worker_id: str, tenant_id: UUID | None = None) -> QueueTask | None:
        """Atomically claims next pending task using PostgreSQL 17 SKIP LOCKED."""
        effective_tid = tenant_id or self.default_tenant_id
        principal = self._queue_principal(effective_tid)
        with (
            self._conn_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            cur.execute(
                """
                WITH next_candidate AS (
                    SELECT task_id
                    FROM indexing_queue
                    WHERE status = 'PENDING' AND scheduled_at <= NOW()
                    ORDER BY priority DESC, created_at ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE indexing_queue q
                SET status = 'PROCESSING',
                    worker_id = %s,
                    attempts = q.attempts + 1,
                    started_at = NOW()
                FROM next_candidate nc
                WHERE q.task_id = nc.task_id
                RETURNING q.task_id, q.tenant_id, q.domain_id, q.event_type, q.payload_json,
                          q.priority, q.status, q.worker_id, q.attempts,
                          q.max_attempts, q.error_message, q.scheduled_at,
                          q.created_at, q.started_at, q.processed_at;
                """,
                (worker_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            return QueueTask(
                task_id=row["task_id"],
                tenant_id=row["tenant_id"],
                domain_id=row["domain_id"],
                event_type=row["event_type"],
                payload=row["payload_json"]
                if isinstance(row["payload_json"], dict)
                else json.loads(row["payload_json"]),
                priority=row["priority"],
                status=row["status"],
                worker_id=row["worker_id"],
                attempts=row["attempts"],
                max_attempts=row["max_attempts"],
                error_message=row["error_message"],
                scheduled_at=row["scheduled_at"],
                created_at=row["created_at"],
                started_at=row["started_at"],
                processed_at=row["processed_at"],
            )

    def complete_task(self, task_id: UUID, tenant_id: UUID | None = None) -> None:
        principal = self._queue_principal(tenant_id)
        with (
            self._conn_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn,
            conn.cursor() as cur,
        ):
            cur.execute(
                """
                UPDATE indexing_queue
                SET status = 'COMPLETED',
                    processed_at = NOW(),
                    error_message = NULL
                WHERE task_id = %s;
                """,
                (task_id,),
            )

    def fail_task(
        self,
        task_id: UUID,
        error_message: str,
        stack_trace: str = "",
        base_delay_seconds: float = 2.0,
        max_delay_seconds: float = 300.0,
        tenant_id: UUID | None = None,
    ) -> bool:
        """Handles task failure: retries with jittered backoff or moves to DLQ."""
        effective_tid = tenant_id or self.default_tenant_id
        principal = self._queue_principal(effective_tid)
        with (
            self._conn_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            cur.execute(
                """
                SELECT task_id, tenant_id, domain_id, event_type,
                       payload_json, attempts, max_attempts
                FROM indexing_queue
                WHERE task_id = %s;
                """,
                (task_id,),
            )
            task = cur.fetchone()
            if not task:
                return False

            attempts = task["attempts"]
            max_attempts = task["max_attempts"]

            if attempts >= max_attempts:
                cur.execute(
                    """
                    INSERT INTO indexing_dead_letter (
                        task_id, tenant_id, domain_id, event_type, payload_json,
                        failure_reason, stack_trace, attempts, failed_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW());
                    """,
                    (
                        task_id,
                        task["tenant_id"],
                        task["domain_id"],
                        task["event_type"],
                        json.dumps(task["payload_json"])
                        if isinstance(task["payload_json"], dict)
                        else task["payload_json"],
                        error_message,
                        stack_trace,
                        attempts,
                    ),
                )
                cur.execute(
                    """
                    UPDATE indexing_queue
                    SET status = 'FAILED',
                        error_message = %s,
                        processed_at = NOW()
                    WHERE task_id = %s;
                    """,
                    (error_message, task_id),
                )
                return False

            delay_sec = compute_backoff_delay(attempts, base_delay_seconds, max_delay_seconds)
            cur.execute(
                """
                UPDATE indexing_queue
                SET status = 'PENDING',
                    worker_id = NULL,
                    scheduled_at = NOW() + (%s * INTERVAL '1 second'),
                    error_message = %s
                WHERE task_id = %s;
                """,
                (delay_sec, error_message, task_id),
            )
            return True

    def reclaim_stale_tasks(
        self, stale_threshold_seconds: int = 300, tenant_id: UUID | None = None
    ) -> int:
        """Reclaims tasks orphaned by crashed or OOM-killed workers."""
        principal = self._queue_principal(tenant_id)
        with (
            self._conn_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn,
            conn.cursor() as cur,
        ):
            cur.execute(
                """
                UPDATE indexing_queue
                SET status = 'PENDING',
                    worker_id = NULL,
                    error_message = 'Worker heart-beat timeout: reclaimed'
                WHERE status = 'PROCESSING'
                  AND started_at < NOW() - (%s * INTERVAL '1 second')
                RETURNING task_id;
                """,
                (stale_threshold_seconds,),
            )
            return len(cur.fetchall())

    def get_queue_stats(self, tenant_id: UUID | None = None) -> dict[str, int]:
        stats = {"PENDING": 0, "PROCESSING": 0, "COMPLETED": 0, "FAILED": 0, "DLQ": 0}
        principal = self._queue_principal(tenant_id)
        with (
            self._conn_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT status, count(*) FROM indexing_queue GROUP BY status;")
            for status_val, count in cur.fetchall():
                stats[str(status_val)] = int(str(count))
            cur.execute("SELECT count(*) FROM indexing_dead_letter;")
            dlq_row = cur.fetchone()
            stats["DLQ"] = int(str(dlq_row[0])) if dlq_row is not None else 0
        return stats

    def list_dead_letter_records(
        self, limit: int = 50, tenant_id: UUID | None = None
    ) -> list[DeadLetterRecord]:
        principal = self._queue_principal(tenant_id)
        with (
            self._conn_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            cur.execute(
                """
                SELECT dlq_id, task_id, tenant_id, domain_id, event_type, payload_json,
                       failure_reason, stack_trace, attempts, failed_at
                FROM indexing_dead_letter
                ORDER BY failed_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            records = []
            for row in cur.fetchall():
                records.append(
                    DeadLetterRecord(
                        dlq_id=row["dlq_id"],
                        task_id=row["task_id"],
                        tenant_id=row["tenant_id"],
                        domain_id=row["domain_id"],
                        event_type=row["event_type"],
                        payload=row["payload_json"]
                        if isinstance(row["payload_json"], dict)
                        else json.loads(row["payload_json"]),
                        failure_reason=row["failure_reason"],
                        stack_trace=row["stack_trace"],
                        attempts=row["attempts"],
                        failed_at=row["failed_at"],
                    )
                )
            return records

    def retry_dead_letter_task(self, task_id: UUID, tenant_id: UUID | None = None) -> bool:
        principal = self._queue_principal(tenant_id)
        with (
            self._conn_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn,
            conn.cursor() as cur,
        ):
            cur.execute(
                """
                UPDATE indexing_queue
                SET status = 'PENDING',
                    attempts = 0,
                    worker_id = NULL,
                    error_message = NULL,
                    scheduled_at = NOW()
                WHERE task_id = %s AND status = 'FAILED'
                RETURNING task_id;
                """,
                (task_id,),
            )
            updated = cur.fetchone()
            if updated:
                cur.execute(
                    "DELETE FROM indexing_dead_letter WHERE task_id = %s;",
                    (task_id,),
                )
                return True
            return False

    def defer_task(
        self,
        task_id: UUID,
        delay_seconds: float,
        reason: str = "Rate limited",
        tenant_id: UUID | None = None,
    ) -> bool:
        """Re-schedules a rate-limited task without consuming retry budget."""
        safe_delay = max(0.5, float(delay_seconds))
        principal = self._queue_principal(tenant_id)
        with (
            self._conn_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn,
            conn.cursor() as cur,
        ):
            cur.execute(
                """
                UPDATE indexing_queue
                SET status = 'PENDING',
                    worker_id = NULL,
                    attempts = GREATEST(0, attempts - 1),
                    scheduled_at = NOW() + (%s * INTERVAL '1 second'),
                    error_message = %s
                WHERE task_id = %s
                RETURNING task_id;
                """,
                (safe_delay, reason, task_id),
            )
            return cur.fetchone() is not None


class InMemoryTaskQueue(TaskQueueProtocol):
    """Thread-safe in-memory task queue for high-speed unit tests."""

    def __init__(self, default_tenant_id: UUID = DEFAULT_TENANT_ID) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[UUID, dict[str, Any]] = {}
        self._dlq: dict[UUID, DeadLetterRecord] = {}
        self.default_tenant_id = default_tenant_id

    def enqueue_task(
        self,
        domain_id: str,
        event_type: str,
        payload: dict[str, Any],
        priority: int = 10,
        max_attempts: int = 3,
        scheduled_at: datetime | None = None,
        tenant_id: UUID | None = None,
    ) -> UUID:
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"Invalid event_type '{event_type}'")

        task_id = uuid4()
        now = datetime.now(UTC)
        effective_tid = tenant_id or self.default_tenant_id
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "tenant_id": effective_tid,
                "domain_id": domain_id,
                "event_type": event_type,
                "payload": payload,
                "priority": priority,
                "status": "PENDING",
                "worker_id": None,
                "attempts": 0,
                "max_attempts": max_attempts,
                "error_message": None,
                "scheduled_at": scheduled_at or now,
                "created_at": now,
                "started_at": None,
                "processed_at": None,
            }
        return task_id

    def fetch_next_task(self, worker_id: str, tenant_id: UUID | None = None) -> QueueTask | None:
        now = datetime.now(UTC)
        with self._lock:
            candidates = [
                t
                for t in self._tasks.values()
                if t["status"] == "PENDING"
                and t["scheduled_at"] <= now
                and (tenant_id is None or t.get("tenant_id", self.default_tenant_id) == tenant_id)
            ]
            if not candidates:
                return None

            candidates.sort(key=lambda x: (-x["priority"], x["created_at"]))
            claimed = candidates[0]
            claimed["status"] = "PROCESSING"
            claimed["worker_id"] = worker_id
            claimed["attempts"] += 1
            claimed["started_at"] = now
            if "tenant_id" not in claimed:
                claimed["tenant_id"] = self.default_tenant_id

            return QueueTask(**claimed)

    def complete_task(self, task_id: UUID, tenant_id: UUID | None = None) -> None:
        with self._lock:
            if task_id in self._tasks and (
                tenant_id is None
                or self._tasks[task_id].get("tenant_id", self.default_tenant_id) == tenant_id
            ):
                self._tasks[task_id]["status"] = "COMPLETED"
                self._tasks[task_id]["processed_at"] = datetime.now(UTC)
                self._tasks[task_id]["error_message"] = None

    def fail_task(
        self,
        task_id: UUID,
        error_message: str,
        stack_trace: str = "",
        base_delay_seconds: float = 2.0,
        max_delay_seconds: float = 300.0,
        tenant_id: UUID | None = None,
    ) -> bool:
        now = datetime.now(UTC)
        with self._lock:
            if task_id not in self._tasks:
                return False
            task = self._tasks[task_id]
            task_tid = task.get("tenant_id", self.default_tenant_id)
            if tenant_id is not None and task_tid != tenant_id:
                return False
            attempts = task["attempts"]
            max_attempts = task["max_attempts"]

            if attempts >= max_attempts:
                task["status"] = "FAILED"
                task["error_message"] = error_message
                task["processed_at"] = now
                dlq_id = uuid4()
                self._dlq[dlq_id] = DeadLetterRecord(
                    dlq_id=dlq_id,
                    task_id=task_id,
                    tenant_id=task_tid,
                    domain_id=task["domain_id"],
                    event_type=task["event_type"],
                    payload=task["payload"],
                    failure_reason=error_message,
                    stack_trace=stack_trace,
                    attempts=attempts,
                    failed_at=now,
                )
                return False

            delay = compute_backoff_delay(attempts, base_delay_seconds, max_delay_seconds)
            task["status"] = "PENDING"
            task["worker_id"] = None
            task["scheduled_at"] = datetime.fromtimestamp(now.timestamp() + delay, tz=UTC)
            task["error_message"] = error_message
            return True

    def reclaim_stale_tasks(
        self, stale_threshold_seconds: int = 300, tenant_id: UUID | None = None
    ) -> int:
        now = datetime.now(UTC)
        reclaimed = 0
        with self._lock:
            for task in self._tasks.values():
                if (
                    (
                        tenant_id is None
                        or task.get("tenant_id", self.default_tenant_id) == tenant_id
                    )
                    and task["status"] == "PROCESSING"
                    and task["started_at"] is not None
                    and (now - task["started_at"]).total_seconds() > stale_threshold_seconds
                ):
                    task["status"] = "PENDING"
                    task["worker_id"] = None
                    task["error_message"] = "Worker heart-beat timeout: reclaimed"
                    reclaimed += 1
        return reclaimed

    def get_queue_stats(self, tenant_id: UUID | None = None) -> dict[str, int]:
        stats = {"PENDING": 0, "PROCESSING": 0, "COMPLETED": 0, "FAILED": 0, "DLQ": 0}
        with self._lock:
            for task in self._tasks.values():
                if tenant_id is None or task.get("tenant_id", self.default_tenant_id) == tenant_id:
                    stats[task["status"]] += 1
            stats["DLQ"] = sum(
                1 for rec in self._dlq.values() if tenant_id is None or rec.tenant_id == tenant_id
            )
        return stats

    def list_dead_letter_records(
        self, limit: int = 50, tenant_id: UUID | None = None
    ) -> list[DeadLetterRecord]:
        with self._lock:
            records = [
                rec for rec in self._dlq.values() if tenant_id is None or rec.tenant_id == tenant_id
            ]
            records.sort(key=lambda r: r.failed_at, reverse=True)
            return records[:limit]

    def retry_dead_letter_task(self, task_id: UUID, tenant_id: UUID | None = None) -> bool:
        with self._lock:
            if (
                task_id in self._tasks
                and self._tasks[task_id]["status"] == "FAILED"
                and (
                    tenant_id is None
                    or self._tasks[task_id].get("tenant_id", self.default_tenant_id) == tenant_id
                )
            ):
                self._tasks[task_id]["status"] = "PENDING"
                self._tasks[task_id]["attempts"] = 0
                self._tasks[task_id]["error_message"] = None
                self._tasks[task_id]["scheduled_at"] = datetime.now(UTC)
                self._dlq = {k: v for k, v in self._dlq.items() if v.task_id != task_id}
                return True
            return False

    def defer_task(
        self,
        task_id: UUID,
        delay_seconds: float,
        reason: str = "Rate limited",
        tenant_id: UUID | None = None,
    ) -> bool:
        now = datetime.now(UTC)
        safe_delay = max(0.5, float(delay_seconds))
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            task_tid = task.get("tenant_id", self.default_tenant_id)
            if tenant_id is not None and task_tid != tenant_id:
                return False
            task["status"] = "PENDING"
            task["worker_id"] = None
            task["attempts"] = max(0, int(task.get("attempts", 1)) - 1)
            task["scheduled_at"] = datetime.fromtimestamp(now.timestamp() + safe_delay, tz=UTC)
            task["error_message"] = reason
            return True
