"""FastAPI webhook routes for event-driven repository and documentation synchronization (F07)."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

from context_scout.config import ScoutSettings
from context_scout.queue.task_queue import (
    InMemoryTaskQueue,
    PostgresTaskQueue,
    TaskQueueProtocol,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

_GLOBAL_QUEUE: TaskQueueProtocol | None = None


def get_task_queue() -> TaskQueueProtocol:
    """Dependency provider returning the active TaskQueueProtocol instance."""
    global _GLOBAL_QUEUE
    if _GLOBAL_QUEUE is not None:
        return _GLOBAL_QUEUE

    settings = ScoutSettings()
    try:
        queue = PostgresTaskQueue(settings.db_app_url)
        queue.get_queue_stats()
        _GLOBAL_QUEUE = queue
        return _GLOBAL_QUEUE
    except Exception as exc:
        logger.warning("PostgresTaskQueue initialization failed, using InMemoryTaskQueue: %s", exc)
        _GLOBAL_QUEUE = InMemoryTaskQueue()
        return _GLOBAL_QUEUE


def set_task_queue(queue: TaskQueueProtocol | None) -> None:
    """Sets or resets the global task queue instance (used for testing)."""
    global _GLOBAL_QUEUE
    _GLOBAL_QUEUE = queue


def _verify_secret_token(
    header_token: str | None,
    expected_secret_env: str,
    vendor_name: str,
) -> None:
    expected_secret = os.environ.get(expected_secret_env)
    if not expected_secret:
        return

    if not header_token or not hmac.compare_digest(header_token, expected_secret):
        logger.warning("Unauthorized %s webhook attempt: invalid token", vendor_name)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid {vendor_name} authentication token",
        )


@router.post("/gitlab")
async def gitlab_webhook(
    request: Request,
    queue: Annotated[TaskQueueProtocol, Depends(get_task_queue)],
    x_gitlab_token: Annotated[str | None, Header(alias="X-Gitlab-Token")] = None,
) -> dict[str, Any]:
    """Ingests GitLab push webhooks after HMAC / token validation."""
    _verify_secret_token(x_gitlab_token, "SCOUT_WEBHOOK_GITLAB_SECRET", "GitLab")

    payload: dict[str, Any] = await request.json()
    domain_id = str(payload.get("domain_id") or payload.get("project", {}).get("name") or "default")

    task_id = queue.enqueue_task(
        domain_id=domain_id,
        event_type="gitlab_push",
        payload=payload,
        priority=payload.get("priority", 10),
    )
    return {
        "status": "queued",
        "task_id": str(task_id),
        "event_type": "gitlab_push",
    }


@router.post("/github")
async def github_webhook(
    request: Request,
    queue: Annotated[TaskQueueProtocol, Depends(get_task_queue)],
    x_hub_signature_256: Annotated[str | None, Header(alias="X-Hub-Signature-256")] = None,
) -> dict[str, Any]:
    """Ingests GitHub push webhooks after HMAC-SHA256 signature verification."""
    expected_secret = os.environ.get("SCOUT_WEBHOOK_GITHUB_SECRET")
    body_bytes = await request.body()

    if expected_secret:
        if not x_hub_signature_256:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-Hub-Signature-256 header",
            )
        expected_sig = (
            "sha256="
            + hmac.new(expected_secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
        )
        if not hmac.compare_digest(x_hub_signature_256, expected_sig):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid GitHub webhook signature",
            )

    payload: dict[str, Any] = await request.json()
    domain_id = str(
        payload.get("domain_id") or payload.get("repository", {}).get("name") or "default"
    )

    task_id = queue.enqueue_task(
        domain_id=domain_id,
        event_type="github_push",
        payload=payload,
        priority=payload.get("priority", 10),
    )
    return {
        "status": "queued",
        "task_id": str(task_id),
        "event_type": "github_push",
    }


@router.post("/confluence")
async def confluence_webhook(
    request: Request,
    queue: Annotated[TaskQueueProtocol, Depends(get_task_queue)],
    x_confluence_secret: Annotated[str | None, Header(alias="X-Confluence-Secret")] = None,
) -> dict[str, Any]:
    """Ingests Confluence page creation/update events."""
    _verify_secret_token(x_confluence_secret, "SCOUT_WEBHOOK_CONFLUENCE_SECRET", "Confluence")

    payload: dict[str, Any] = await request.json()
    domain_id = str(payload.get("domain_id") or payload.get("space_key") or "default")

    task_id = queue.enqueue_task(
        domain_id=domain_id,
        event_type="confluence_update",
        payload=payload,
        priority=payload.get("priority", 10),
    )
    return {
        "status": "queued",
        "task_id": str(task_id),
        "event_type": "confluence_update",
    }


@router.post("/jira")
async def jira_webhook(
    request: Request,
    queue: Annotated[TaskQueueProtocol, Depends(get_task_queue)],
    x_jira_secret: Annotated[str | None, Header(alias="X-Jira-Secret")] = None,
) -> dict[str, Any]:
    """Ingests Jira issue lifecycle change events."""
    _verify_secret_token(x_jira_secret, "SCOUT_WEBHOOK_JIRA_SECRET", "Jira")

    payload: dict[str, Any] = await request.json()
    domain_id = str(payload.get("domain_id") or payload.get("project_key") or "default")

    task_id = queue.enqueue_task(
        domain_id=domain_id,
        event_type="jira_update",
        payload=payload,
        priority=payload.get("priority", 10),
    )
    return {
        "status": "queued",
        "task_id": str(task_id),
        "event_type": "jira_update",
    }


@router.get("/stats")
def get_queue_stats(
    queue: Annotated[TaskQueueProtocol, Depends(get_task_queue)],
) -> dict[str, Any]:
    """Returns real-time indexing queue and DLQ metrics."""
    return {
        "status": "ok",
        "stats": queue.get_queue_stats(),
    }


@router.get("/dlq")
def list_dlq(
    queue: Annotated[TaskQueueProtocol, Depends(get_task_queue)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    """Inspects failed indexing tasks residing in the Dead-Letter Queue."""
    records = queue.list_dead_letter_records(limit=limit)
    return {
        "status": "ok",
        "total": len(records),
        "records": [
            {
                "dlq_id": str(r.dlq_id),
                "task_id": str(r.task_id),
                "domain_id": r.domain_id,
                "event_type": r.event_type,
                "payload": r.payload,
                "failure_reason": r.failure_reason,
                "stack_trace": r.stack_trace,
                "attempts": r.attempts,
                "failed_at": r.failed_at.isoformat(),
            }
            for r in records
        ],
    }


@router.post("/dlq/{task_id}/retry")
def retry_dlq_task(
    task_id: UUID,
    queue: Annotated[TaskQueueProtocol, Depends(get_task_queue)],
) -> dict[str, Any]:
    """Re-enqueues a dead-lettered indexing task for reprocessing."""
    success = queue.retry_dead_letter_task(task_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found in dead-letter queue or is not failed",
        )
    return {
        "status": "retried",
        "task_id": str(task_id),
    }
