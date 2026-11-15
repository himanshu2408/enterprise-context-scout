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

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

EventType = Literal[
    "gitlab_push",
    "github_push",
    "confluence_update",
    "jira_update",
    "cross_repo_relink",
]

TaskStatus = Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"]

VALID_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {"gitlab_push", "github_push", "confluence_update", "jira_update", "cross_repo_relink"}
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


def compute_backoff_delay(
    attempts: int,
    base_delay_seconds: float = 2.0,
    max_delay_seconds: float = 300.0,
) -> float:
    """Computes exponential backoff with Equal Jitter to eliminate retry storms.

    Formula:
        delay = min(max_delay, base_delay * 2^(attempts - 1))
        jittered = delay / 2 + uniform(0, delay / 2)
    """
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
    ) -> UUID: ...

    @abstractmethod
    def fetch_next_task(self, worker_id: str) -> QueueTask | None: ...

    @abstractmethod
    def complete_task(self, task_id: UUID) -> None: ...

    @abstractmethod
    def fail_task(
        self,
        task_id: UUID,
        error_message: str,
        stack_trace: str = "",
        base_delay_seconds: float = 2.0,
        max_delay_seconds: float = 300.0,
    ) -> bool: ...

    @abstractmethod
    def reclaim_stale_tasks(self, stale_threshold_seconds: int = 300) -> int: ...

    @abstractmethod
    def get_queue_stats(self) -> dict[str, int]: ...

    @abstractmethod
    def list_dead_letter_records(self, limit: int = 50) -> list[DeadLetterRecord]: ...

    @abstractmethod
    def retry_dead_letter_task(self, task_id: UUID) -> bool: ...


class PostgresTaskQueue(TaskQueueProtocol):
    """PostgreSQL 17 SKIP LOCKED task queue using atomic two-phase claim transactions."""

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def enqueue_task(
        self,
        domain_id: str,
        event_type: str,
        payload: dict[str, Any],
        priority: int = 10,
        max_attempts: int = 3,
        scheduled_at: datetime | None = None,
    ) -> UUID:
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid event_type '{event_type}'. Must be one of {sorted(VALID_EVENT_TYPES)}"
            )

        task_id = uuid4()
        payload_bytes = json.dumps(payload)

        with psycopg.connect(self.db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO indexing_queue (
                        task_id, domain_id, event_type, payload_json,
                        priority, max_attempts, scheduled_at, status
                    )
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s, COALESCE(%s, NOW()), 'PENDING')
                    RETURNING task_id;
                    """,
                (
                    task_id,
                    domain_id,
                    event_type,
                    payload_bytes,
                    priority,
                    max_attempts,
                    scheduled_at,
                ),
            )
            conn.commit()
        return task_id

    def fetch_next_task(self, worker_id: str) -> QueueTask | None:
        """Atomically claims next pending task using PostgreSQL 17 SKIP LOCKED."""
        with psycopg.connect(self.db_url) as conn, conn.cursor(row_factory=dict_row) as cur:
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
                    RETURNING q.task_id, q.domain_id, q.event_type, q.payload_json,
                              q.priority, q.status, q.worker_id, q.attempts,
                              q.max_attempts, q.error_message, q.scheduled_at,
                              q.created_at, q.started_at, q.processed_at;
                    """,
                (worker_id,),
            )
            row = cur.fetchone()
            conn.commit()

            if not row:
                return None

            return QueueTask(
                task_id=row["task_id"],
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

    def complete_task(self, task_id: UUID) -> None:
        with psycopg.connect(self.db_url) as conn, conn.cursor() as cur:
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
            conn.commit()

    def fail_task(
        self,
        task_id: UUID,
        error_message: str,
        stack_trace: str = "",
        base_delay_seconds: float = 2.0,
        max_delay_seconds: float = 300.0,
    ) -> bool:
        """Handles task failure: retries with jittered backoff or moves to DLQ."""
        with psycopg.connect(self.db_url) as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                    SELECT task_id, domain_id, event_type, payload_json, attempts, max_attempts
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
                            task_id, domain_id, event_type, payload_json,
                            failure_reason, stack_trace, attempts, failed_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW());
                        """,
                    (
                        task_id,
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
                conn.commit()
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
            conn.commit()
            return True

    def reclaim_stale_tasks(self, stale_threshold_seconds: int = 300) -> int:
        """Reclaims tasks orphaned by crashed or OOM-killed workers."""
        with psycopg.connect(self.db_url) as conn, conn.cursor() as cur:
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
            reclaimed = len(cur.fetchall())
            conn.commit()
            return reclaimed

    def get_queue_stats(self) -> dict[str, int]:
        stats = {"PENDING": 0, "PROCESSING": 0, "COMPLETED": 0, "FAILED": 0, "DLQ": 0}
        with psycopg.connect(self.db_url) as conn, conn.cursor() as cur:
            cur.execute("SELECT status, count(*) FROM indexing_queue GROUP BY status;")
            for status, count in cur.fetchall():
                stats[status] = count
            cur.execute("SELECT count(*) FROM indexing_dead_letter;")
            dlq_row = cur.fetchone()
            stats["DLQ"] = int(dlq_row[0]) if dlq_row is not None else 0
        return stats

    def list_dead_letter_records(self, limit: int = 50) -> list[DeadLetterRecord]:
        with psycopg.connect(self.db_url) as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                    SELECT dlq_id, task_id, domain_id, event_type, payload_json,
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

    def retry_dead_letter_task(self, task_id: UUID) -> bool:
        with psycopg.connect(self.db_url) as conn, conn.cursor() as cur:
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
                conn.commit()
                return True
            conn.commit()
            return False


class InMemoryTaskQueue(TaskQueueProtocol):
    """Thread-safe in-memory task queue for high-speed unit tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[UUID, dict[str, Any]] = {}
        self._dlq: dict[UUID, DeadLetterRecord] = {}

    def enqueue_task(
        self,
        domain_id: str,
        event_type: str,
        payload: dict[str, Any],
        priority: int = 10,
        max_attempts: int = 3,
        scheduled_at: datetime | None = None,
    ) -> UUID:
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"Invalid event_type '{event_type}'")

        task_id = uuid4()
        now = datetime.now(UTC)
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
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

    def fetch_next_task(self, worker_id: str) -> QueueTask | None:
        now = datetime.now(UTC)
        with self._lock:
            candidates = [
                t
                for t in self._tasks.values()
                if t["status"] == "PENDING" and t["scheduled_at"] <= now
            ]
            if not candidates:
                return None

            candidates.sort(key=lambda x: (-x["priority"], x["created_at"]))
            claimed = candidates[0]
            claimed["status"] = "PROCESSING"
            claimed["worker_id"] = worker_id
            claimed["attempts"] += 1
            claimed["started_at"] = now

            return QueueTask(**claimed)

    def complete_task(self, task_id: UUID) -> None:
        with self._lock:
            if task_id in self._tasks:
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
    ) -> bool:
        now = datetime.now(UTC)
        with self._lock:
            if task_id not in self._tasks:
                return False
            task = self._tasks[task_id]
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

    def reclaim_stale_tasks(self, stale_threshold_seconds: int = 300) -> int:
        now = datetime.now(UTC)
        reclaimed = 0
        with self._lock:
            for task in self._tasks.values():
                if (
                    task["status"] == "PROCESSING"
                    and task["started_at"] is not None
                    and (now - task["started_at"]).total_seconds() > stale_threshold_seconds
                ):
                    task["status"] = "PENDING"
                    task["worker_id"] = None
                    task["error_message"] = "Worker heart-beat timeout: reclaimed"
                    reclaimed += 1
        return reclaimed

    def get_queue_stats(self) -> dict[str, int]:
        stats = {"PENDING": 0, "PROCESSING": 0, "COMPLETED": 0, "FAILED": 0, "DLQ": 0}
        with self._lock:
            for task in self._tasks.values():
                stats[task["status"]] += 1
            stats["DLQ"] = len(self._dlq)
        return stats

    def list_dead_letter_records(self, limit: int = 50) -> list[DeadLetterRecord]:
        with self._lock:
            records = list(self._dlq.values())
            records.sort(key=lambda r: r.failed_at, reverse=True)
            return records[:limit]

    def retry_dead_letter_task(self, task_id: UUID) -> bool:
        with self._lock:
            if task_id in self._tasks and self._tasks[task_id]["status"] == "FAILED":
                self._tasks[task_id]["status"] = "PENDING"
                self._tasks[task_id]["attempts"] = 0
                self._tasks[task_id]["error_message"] = None
                self._tasks[task_id]["scheduled_at"] = datetime.now(UTC)
                self._dlq = {k: v for k, v in self._dlq.items() if v.task_id != task_id}
                return True
            return False
