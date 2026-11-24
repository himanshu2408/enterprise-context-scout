"""Fail-closed Webhook Ingress & RBAC/RLS-Scoped Queue Management Routes (F07, M5-01)."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import threading
import time
from collections import deque
from typing import Annotated, Any
from uuid import UUID

import psycopg.errors
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

from context_scout.api.dependencies import get_principal_context, is_development_environment
from context_scout.config import ScoutSettings
from context_scout.queue.task_queue import (
    InMemoryTaskQueue,
    PostgresTaskQueue,
    TaskQueueProtocol,
)
from context_scout.schemas.contracts import DEFAULT_TENANT_ID, PrincipalContext, RoleType
from context_scout.security.audit_chain import get_audit_chain_service

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


def _allow_unsigned_dev_webhooks() -> bool:
    if not is_development_environment():
        return False
    raw = os.environ.get("SCOUT_ALLOW_UNSIGNED_DEV_WEBHOOKS", "false").strip().lower()
    return raw in {"1", "true", "yes"}


def _verify_webhook_credentials(
    *,
    expected_secret_env: str,
    vendor_name: str,
    body_bytes: bytes,
    token_header: str | None = None,
    signature_header: str | None = None,
    allow_token_header: bool = True,
) -> None:
    """Enforces fail-closed HMAC-SHA256 / token verification on webhook ingress."""
    expected_secret = (os.environ.get(expected_secret_env) or "").strip()
    if not expected_secret:
        if _allow_unsigned_dev_webhooks():
            logger.warning(
                "Accepting unsigned %s webhook in development mode "
                "(SCOUT_ALLOW_UNSIGNED_DEV_WEBHOOKS=true).",
                vendor_name,
            )
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"{vendor_name} webhook verification secret ({expected_secret_env}) is unset; "
                "failing closed."
            ),
        )

    if signature_header and signature_header.strip():
        expected_sig = (
            "sha256="
            + hmac.new(expected_secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
        )
        if hmac.compare_digest(signature_header.strip(), expected_sig):
            return
        logger.warning("Unauthorized %s webhook attempt: invalid HMAC signature", vendor_name)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid {vendor_name} webhook signature",
        )

    if allow_token_header and token_header is not None and token_header.strip():
        if hmac.compare_digest(token_header.strip(), expected_secret):
            return
        logger.warning("Unauthorized %s webhook attempt: invalid token", vendor_name)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid {vendor_name} authentication token",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=f"Missing {vendor_name} webhook authentication signature or token",
    )


def _resolve_webhook_tenant(raw_tid: str | None) -> UUID:
    if not raw_tid or not raw_tid.strip():
        return DEFAULT_TENANT_ID
    try:
        return UUID(raw_tid.strip())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid X-Scout-Tenant-Id UUID: {raw_tid!r}.",
        ) from None


def _has_nul_or_surrogate(value: str) -> bool:
    return "\x00" in value or any(0xD800 <= ord(ch) <= 0xDFFF for ch in value)


def _contains_nul_or_surrogate(value: Any, *, depth: int = 0) -> bool:
    if depth > 32:
        return True
    if isinstance(value, str):
        return _has_nul_or_surrogate(value)
    if isinstance(value, dict):
        return any(
            (isinstance(k, str) and _has_nul_or_surrogate(k))
            or _contains_nul_or_surrogate(v, depth=depth + 1)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_nul_or_surrogate(item, depth=depth + 1) for item in value)
    return False


def _parse_json_payload(body_bytes: bytes) -> dict[str, Any]:
    if not body_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty JSON body.")
    try:
        parsed = json.loads(body_bytes.decode("utf-8"))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON webhook payload."
        ) from None
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook payload must be a JSON object.",
        )
    if _contains_nul_or_surrogate(parsed):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Webhook payload must not contain NUL bytes, lone surrogates, or exceed max depth."
            ),
        )
    return parsed


def _parse_webhook_priority(payload: dict[str, Any]) -> int:
    if "priority" not in payload:
        return 10
    raw = payload["priority"]
    if isinstance(raw, bool):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook 'priority' must be an integer, not boolean.",
        )
    if isinstance(raw, int):
        val = raw
    elif isinstance(raw, str):
        cleaned = raw.strip()
        if not re.fullmatch(r"[+-]?[0-9]+", cleaned):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Webhook 'priority' must be a valid integer.",
            )
        try:
            val = int(cleaned)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Webhook 'priority' must be a valid integer.",
            ) from None
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook 'priority' must be an integer.",
        )
    if val < -2_147_483_648 or val > 2_147_483_647:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook 'priority' must fit within 32-bit signed integer range.",
        )
    return val


def _extract_webhook_domain_id(
    payload: dict[str, Any],
    *,
    nested_key: str | None = None,
    flat_key: str | None = None,
) -> str:
    candidate: Any = payload.get("domain_id")
    if candidate is not None and not isinstance(candidate, (str, int)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook 'domain_id' must be a string.",
        )
    resolved = str(candidate).strip() if candidate is not None else ""
    if not resolved and nested_key is not None and nested_key in payload:
        nested = payload[nested_key]
        if nested is not None and not isinstance(nested, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Webhook '{nested_key}' must be a JSON object.",
            )
        if isinstance(nested, dict):
            name_val = nested.get("name")
            if name_val is not None and not isinstance(name_val, (str, int)):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Webhook '{nested_key}.name' must be a string.",
                )
            if name_val is not None:
                resolved = str(name_val).strip()
    if not resolved and flat_key is not None and flat_key in payload:
        flat_val = payload[flat_key]
        if flat_val is not None and not isinstance(flat_val, (str, int)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Webhook '{flat_key}' must be a string.",
            )
        if flat_val is not None:
            resolved = str(flat_val).strip()
    final_domain = resolved or "default"
    if _has_nul_or_surrogate(final_domain) or len(final_domain) > 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook domain_id.",
        )
    return final_domain


_WEBHOOK_REPLAY_LOCK = threading.Lock()
_SEEN_WEBHOOK_DELIVERIES: deque[tuple[str, float]] = deque(maxlen=2048)
_SEEN_WEBHOOK_IDS: set[str] = set()
_WEBHOOK_MAX_SKEW_SECONDS = 300.0
_ASCII_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _validate_optional_replay_headers(request: Request, vendor: str) -> None:
    """Enforces optional timestamp freshness & delivery-ID replay check when headers are sent."""
    ts_candidates = [
        v.strip()
        for v in (
            *request.headers.getlist("x-scout-webhook-timestamp"),
            *request.headers.getlist("x-webhook-timestamp"),
        )
        if v.strip()
    ]
    if len(ts_candidates) > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conflicting or duplicate webhook timestamp headers are not permitted.",
        )
    if ts_candidates:
        ts_str = ts_candidates[0]
        if not _ASCII_FLOAT_RE.fullmatch(ts_str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid webhook timestamp header.",
            )
        try:
            ts_val = float(ts_str)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid webhook timestamp header.",
            ) from None
        if not math.isfinite(ts_val):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid webhook timestamp header.",
            )
        if abs(time.time() - ts_val) > _WEBHOOK_MAX_SKEW_SECONDS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Stale webhook timestamp exceeds replay tolerance (ASI07).",
            )

    delivery_id = (
        request.headers.get("X-Scout-Delivery-Id")
        or request.headers.get("X-GitHub-Delivery")
        or request.headers.get("X-Gitlab-Event-UUID")
    )
    if delivery_id is not None and delivery_id.strip():
        clean_id = delivery_id.strip()
        if _has_nul_or_surrogate(clean_id) or len(clean_id) > 128:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid webhook delivery ID header.",
            )
        scoped_id = f"{vendor}:{clean_id}"
        now = time.time()
        with _WEBHOOK_REPLAY_LOCK:
            while (
                _SEEN_WEBHOOK_DELIVERIES
                and (now - _SEEN_WEBHOOK_DELIVERIES[0][1]) > _WEBHOOK_MAX_SKEW_SECONDS
            ):
                expired_id, _ = _SEEN_WEBHOOK_DELIVERIES.popleft()
                _SEEN_WEBHOOK_IDS.discard(expired_id)
            if scoped_id in _SEEN_WEBHOOK_IDS:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Duplicate webhook delivery ID rejected (ASI07 replay guard).",
                )
            if len(_SEEN_WEBHOOK_DELIVERIES) == _SEEN_WEBHOOK_DELIVERIES.maxlen:
                evicted_id, _ = _SEEN_WEBHOOK_DELIVERIES.popleft()
                _SEEN_WEBHOOK_IDS.discard(evicted_id)
            _SEEN_WEBHOOK_DELIVERIES.append((scoped_id, now))
            _SEEN_WEBHOOK_IDS.add(scoped_id)


def _enqueue_webhook_task_safely(
    queue: TaskQueueProtocol,
    *,
    domain_id: str,
    event_type: str,
    payload: dict[str, Any],
    priority: int,
    tenant_id: UUID,
) -> UUID:
    try:
        return queue.enqueue_task(
            domain_id=domain_id,
            event_type=event_type,
            payload=payload,
            priority=priority,
            tenant_id=tenant_id,
        )
    except (
        psycopg.errors.ForeignKeyViolation,
        psycopg.errors.IntegrityError,
        psycopg.errors.DataError,
        ValueError,
    ) as exc:
        safe_domain = domain_id.encode("utf-8", errors="replace").decode("utf-8")
        safe_err = str(exc).encode("utf-8", errors="replace").decode("utf-8")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid webhook task parameters for domain '{safe_domain}': {safe_err}",
        ) from None


def _require_webhook_admin_role(principal: PrincipalContext) -> None:
    if principal.role not in {RoleType.ENGINEER, RoleType.COMPLIANCE_OFFICER}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Webhook queue inspection and retry require Engineer or "
                "Compliance Officer privileges."
            ),
        )


@router.post("/gitlab")
async def gitlab_webhook(
    request: Request,
    queue: Annotated[TaskQueueProtocol, Depends(get_task_queue)],
    x_gitlab_token: Annotated[str | None, Header(alias="X-Gitlab-Token")] = None,
    x_gitlab_signature_256: Annotated[str | None, Header(alias="X-Gitlab-Signature-256")] = None,
    x_scout_tenant_id: Annotated[str | None, Header(alias="X-Scout-Tenant-Id")] = None,
) -> dict[str, Any]:
    """Ingests GitLab push webhooks after fail-closed token or HMAC-SHA256 validation."""
    body_bytes = await request.body()
    _verify_webhook_credentials(
        expected_secret_env="SCOUT_WEBHOOK_GITLAB_SECRET",
        vendor_name="GitLab",
        body_bytes=body_bytes,
        token_header=x_gitlab_token,
        signature_header=x_gitlab_signature_256,
        allow_token_header=True,
    )
    _validate_optional_replay_headers(request, "gitlab")
    tenant_id = _resolve_webhook_tenant(x_scout_tenant_id)
    payload = _parse_json_payload(body_bytes)
    domain_id = _extract_webhook_domain_id(payload, nested_key="project")
    priority = _parse_webhook_priority(payload)

    task_id = _enqueue_webhook_task_safely(
        queue,
        domain_id=domain_id,
        event_type="gitlab_push",
        payload=payload,
        priority=priority,
        tenant_id=tenant_id,
    )
    get_audit_chain_service().record_event(
        tenant_id=tenant_id,
        domain_id=domain_id,
        event_type="webhook.ingress",
        severity="INFO",
        actor_id="webhook:gitlab",
        actor_role="system",
        resource_uri="/api/v1/webhooks/gitlab",
        outcome="allow",
        payload={"task_id": str(task_id)},
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
    x_scout_tenant_id: Annotated[str | None, Header(alias="X-Scout-Tenant-Id")] = None,
) -> dict[str, Any]:
    """Ingests GitHub push webhooks after fail-closed HMAC-SHA256 signature verification."""
    body_bytes = await request.body()
    _verify_webhook_credentials(
        expected_secret_env="SCOUT_WEBHOOK_GITHUB_SECRET",
        vendor_name="GitHub",
        body_bytes=body_bytes,
        token_header=None,
        signature_header=x_hub_signature_256,
        allow_token_header=False,
    )
    _validate_optional_replay_headers(request, "github")
    tenant_id = _resolve_webhook_tenant(x_scout_tenant_id)
    payload = _parse_json_payload(body_bytes)
    domain_id = _extract_webhook_domain_id(payload, nested_key="repository")
    priority = _parse_webhook_priority(payload)

    task_id = _enqueue_webhook_task_safely(
        queue,
        domain_id=domain_id,
        event_type="github_push",
        payload=payload,
        priority=priority,
        tenant_id=tenant_id,
    )
    get_audit_chain_service().record_event(
        tenant_id=tenant_id,
        domain_id=domain_id,
        event_type="webhook.ingress",
        severity="INFO",
        actor_id="webhook:github",
        actor_role="system",
        resource_uri="/api/v1/webhooks/github",
        outcome="allow",
        payload={"task_id": str(task_id)},
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
    x_confluence_signature_256: Annotated[
        str | None, Header(alias="X-Confluence-Signature-256")
    ] = None,
    x_scout_tenant_id: Annotated[str | None, Header(alias="X-Scout-Tenant-Id")] = None,
) -> dict[str, Any]:
    """Ingests Confluence page creation/update events after fail-closed HMAC/token validation."""
    body_bytes = await request.body()
    _verify_webhook_credentials(
        expected_secret_env="SCOUT_WEBHOOK_CONFLUENCE_SECRET",
        vendor_name="Confluence",
        body_bytes=body_bytes,
        token_header=x_confluence_secret,
        signature_header=x_confluence_signature_256,
        allow_token_header=True,
    )
    _validate_optional_replay_headers(request, "confluence")
    tenant_id = _resolve_webhook_tenant(x_scout_tenant_id)
    payload = _parse_json_payload(body_bytes)
    domain_id = _extract_webhook_domain_id(payload, flat_key="space_key")
    priority = _parse_webhook_priority(payload)

    task_id = _enqueue_webhook_task_safely(
        queue,
        domain_id=domain_id,
        event_type="confluence_update",
        payload=payload,
        priority=priority,
        tenant_id=tenant_id,
    )
    get_audit_chain_service().record_event(
        tenant_id=tenant_id,
        domain_id=domain_id,
        event_type="webhook.ingress",
        severity="INFO",
        actor_id="webhook:confluence",
        actor_role="system",
        resource_uri="/api/v1/webhooks/confluence",
        outcome="allow",
        payload={"task_id": str(task_id)},
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
    x_jira_signature_256: Annotated[str | None, Header(alias="X-Jira-Signature-256")] = None,
    x_scout_tenant_id: Annotated[str | None, Header(alias="X-Scout-Tenant-Id")] = None,
) -> dict[str, Any]:
    """Ingests Jira issue lifecycle events after fail-closed HMAC/token validation."""
    body_bytes = await request.body()
    _verify_webhook_credentials(
        expected_secret_env="SCOUT_WEBHOOK_JIRA_SECRET",
        vendor_name="Jira",
        body_bytes=body_bytes,
        token_header=x_jira_secret,
        signature_header=x_jira_signature_256,
        allow_token_header=True,
    )
    _validate_optional_replay_headers(request, "jira")
    tenant_id = _resolve_webhook_tenant(x_scout_tenant_id)
    payload = _parse_json_payload(body_bytes)
    domain_id = _extract_webhook_domain_id(payload, flat_key="project_key")
    priority = _parse_webhook_priority(payload)

    task_id = _enqueue_webhook_task_safely(
        queue,
        domain_id=domain_id,
        event_type="jira_update",
        payload=payload,
        priority=priority,
        tenant_id=tenant_id,
    )
    get_audit_chain_service().record_event(
        tenant_id=tenant_id,
        domain_id=domain_id,
        event_type="webhook.ingress",
        severity="INFO",
        actor_id="webhook:jira",
        actor_role="system",
        resource_uri="/api/v1/webhooks/jira",
        outcome="allow",
        payload={"task_id": str(task_id)},
    )
    return {
        "status": "queued",
        "task_id": str(task_id),
        "event_type": "jira_update",
    }


@router.get("/stats")
def get_queue_stats(
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
    queue: Annotated[TaskQueueProtocol, Depends(get_task_queue)],
) -> dict[str, Any]:
    """Returns RBAC & RLS-scoped indexing queue and DLQ metrics."""
    _require_webhook_admin_role(principal)
    return {
        "status": "ok",
        "tenant_id": str(principal.tenant_id),
        "stats": queue.get_queue_stats(tenant_id=principal.tenant_id),
    }


@router.get("/dlq")
def list_dlq(
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
    queue: Annotated[TaskQueueProtocol, Depends(get_task_queue)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    """Inspects tenant-isolated Dead-Letter Queue records under RBAC + Forced RLS."""
    _require_webhook_admin_role(principal)
    records = queue.list_dead_letter_records(limit=limit, tenant_id=principal.tenant_id)
    return {
        "status": "ok",
        "tenant_id": str(principal.tenant_id),
        "total": len(records),
        "records": [
            {
                "dlq_id": str(r.dlq_id),
                "task_id": str(r.task_id),
                "tenant_id": str(r.tenant_id),
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
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
    queue: Annotated[TaskQueueProtocol, Depends(get_task_queue)],
) -> dict[str, Any]:
    """Re-enqueues a tenant-scoped dead-lettered task under RBAC + Forced RLS."""
    _require_webhook_admin_role(principal)
    success = queue.retry_dead_letter_task(task_id, tenant_id=principal.tenant_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found in dead-letter queue or is not failed",
        )
    get_audit_chain_service().record_event(
        principal=principal,
        event_type="webhook.dlq_retry",
        severity="NOTICE",
        resource_uri=f"/api/v1/webhooks/dlq/{task_id}/retry",
        outcome="success",
        payload={"task_id": str(task_id)},
    )
    return {
        "status": "retried",
        "task_id": str(task_id),
    }
