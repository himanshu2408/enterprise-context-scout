"""Tests for Milestone 2 Feature F07: Event-Driven Task Queue, Indexing Worker, and Webhooks."""

import hashlib
import hmac
from datetime import UTC, datetime

import pytest
from starlette.testclient import TestClient

from context_scout.api.main import create_app
from context_scout.api.routes.webhooks import set_task_queue
from context_scout.config import ScoutSettings
from context_scout.queue.indexing_worker import IndexingWorker
from context_scout.queue.task_queue import (
    InMemoryTaskQueue,
    PostgresTaskQueue,
    compute_backoff_delay,
)


@pytest.fixture
def in_memory_queue() -> InMemoryTaskQueue:
    queue = InMemoryTaskQueue()
    set_task_queue(queue)
    return queue


@pytest.fixture
def app_client(in_memory_queue: InMemoryTaskQueue) -> TestClient:
    app = create_app()
    return TestClient(app)


def test_f07_equal_jitter_backoff() -> None:
    """Verifies that exponential backoff computes Equal Jitter within valid bounds."""
    for attempt in range(1, 6):
        delay = compute_backoff_delay(attempt, base_delay_seconds=2.0, max_delay_seconds=300.0)
        max_possible = min(300.0, 2.0 * (2 ** (attempt - 1)))
        min_possible = max_possible / 2.0
        assert min_possible <= delay <= max_possible


def test_f07_in_memory_queue_claim_and_dlq(in_memory_queue: InMemoryTaskQueue) -> None:
    """Verifies atomic claim, concurrent skip, and DLQ routing on max attempts."""
    domain_id = "trading_code_scout"
    payload = {"repo_name": "order-service", "before_sha": "0" * 40, "after_sha": "a" * 40}

    # 1. Enqueue task
    task_id = in_memory_queue.enqueue_task(
        domain_id=domain_id,
        event_type="gitlab_push",
        payload=payload,
        priority=10,
        max_attempts=2,
    )
    assert task_id is not None

    # 2. Worker 1 claims task
    task = in_memory_queue.fetch_next_task(worker_id="worker_1")
    assert task is not None
    assert task.task_id == task_id
    assert task.status == "PROCESSING"
    assert task.worker_id == "worker_1"

    # 3. Worker 2 attempts concurrent claim -> row is locked/claimed, returns None
    second_claim = in_memory_queue.fetch_next_task(worker_id="worker_2")
    assert second_claim is None

    # 4. Fail task (attempt 1) -> backoff delay rescheduled
    in_memory_queue.fail_task(
        task_id=task_id,
        error_message="Simulated git timeout",
        stack_trace="Traceback: timeout in git fetch",
    )

    # Task is now pending again with attempts == 1.
    # Manually reset scheduled_at to past to test attempt 2
    in_memory_queue._tasks[task_id] = in_memory_queue._tasks[task_id].__class__(
        task_id=task.task_id,
        domain_id=task.domain_id,
        event_type=task.event_type,
        payload=task.payload,
        priority=task.priority,
        status="PENDING",
        worker_id=None,
        attempts=1,
        max_attempts=2,
        error_message="Simulated git timeout",
        scheduled_at=datetime.now(UTC),
        created_at=task.created_at,
    )

    # 5. Worker 1 claims for attempt 2
    task_retry = in_memory_queue.fetch_next_task(worker_id="worker_1")
    assert task_retry is not None
    assert task_retry.attempts == 2

    # 6. Fail task (attempt 2 >= max_attempts 2) -> routes to DLQ
    in_memory_queue.fail_task(
        task_id=task_id,
        error_message="Permanent poison pill payload",
        stack_trace="Traceback: cannot parse corrupted git blob",
    )

    dlqs = in_memory_queue.list_dead_letter_records()
    assert len(dlqs) == 1
    assert dlqs[0].task_id == task_id
    assert dlqs[0].failure_reason == "Permanent poison pill payload"

    # 7. Retry from DLQ
    re_enqueued = in_memory_queue.retry_dead_letter_task(dlqs[0].task_id)
    assert re_enqueued is True
    assert len(in_memory_queue.list_dead_letter_records()) == 0

    reclaimed = in_memory_queue.fetch_next_task(worker_id="worker_1")
    assert reclaimed is not None
    assert reclaimed.task_id == task_id
    assert reclaimed.attempts == 1

    in_memory_queue.complete_task(task_id)
    assert in_memory_queue.fetch_next_task(worker_id="worker_1") is None


def test_f07_indexing_worker_run_once(in_memory_queue: InMemoryTaskQueue) -> None:
    """Verifies that IndexingWorker fetches and dispatches tasks cleanly."""
    worker = IndexingWorker(
        task_queue=in_memory_queue,
        worker_id="test_worker_unit",
    )

    # Enqueue a gitlab_push task
    in_memory_queue.enqueue_task(
        domain_id="trading_code_scout",
        event_type="gitlab_push",
        payload={
            "repo_name": "pricing-service",
            "before_sha": "0" * 40,
            "after_sha": "b" * 40,
        },
        priority=10,
    )

    processed = worker.run_once()
    assert processed is True
    # Queue is now empty
    assert worker.run_once() is False


def test_f07_webhook_endpoints_hmac_and_ingress(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies webhook ingress routes with constant-time HMAC validation."""
    gitlab_secret = "test_gitlab_secret_key"
    github_secret = "test_github_secret_key"
    monkeypatch.setenv("SCOUT_WEBHOOK_GITLAB_SECRET", gitlab_secret)
    monkeypatch.setenv("SCOUT_WEBHOOK_GITHUB_SECRET", github_secret)

    payload = {
        "event_type": "push",
        "domain_id": "trading_code_scout",
        "repository": {"name": "routing-engine"},
        "before": "0" * 40,
        "after": "c" * 40,
    }

    # 1. GitLab Webhook - Missing/Bad Secret Token -> 401
    resp_bad = app_client.post(
        "/api/v1/webhooks/gitlab",
        json=payload,
        headers={"X-Gitlab-Token": "invalid_token"},
    )
    assert resp_bad.status_code == 401

    # 2. GitLab Webhook - Valid Token -> 200/202 Queued
    resp_ok = app_client.post(
        "/api/v1/webhooks/gitlab",
        json=payload,
        headers={"X-Gitlab-Token": gitlab_secret},
    )
    assert resp_ok.status_code == 200
    data = resp_ok.json()
    assert data["status"] == "queued"
    assert "task_id" in data

    # 3. GitHub Webhook - HMAC-SHA256 signature verification
    raw_body = b'{"ref": "refs/heads/main", "domain_id": "trading_code_scout"}'
    valid_sig = "sha256=" + hmac.new(github_secret.encode(), raw_body, hashlib.sha256).hexdigest()

    resp_gh = app_client.post(
        "/api/v1/webhooks/github",
        content=raw_body,
        headers={
            "X-Hub-Signature-256": valid_sig,
            "Content-Type": "application/json",
        },
    )
    assert resp_gh.status_code == 200
    data_gh = resp_gh.json()
    assert data_gh["status"] == "queued"

    # GitHub Webhook - Bad signature -> 401
    resp_gh_bad = app_client.post(
        "/api/v1/webhooks/github",
        content=raw_body,
        headers={
            "X-Hub-Signature-256": "sha256=" + "0" * 64,
            "Content-Type": "application/json",
        },
    )
    assert resp_gh_bad.status_code == 401


def test_f07_live_postgres_task_queue() -> None:
    """Verifies PostgreSQL 17 SKIP LOCKED atomic claims on live database."""
    settings = ScoutSettings()
    pg_queue = PostgresTaskQueue(db_url=settings.db_app_url)
    domain_id = "trading_code_scout"

    # Enqueue task
    t_id = pg_queue.enqueue_task(
        domain_id=domain_id,
        event_type="cross_repo_relink",
        payload={"domain_id": domain_id},
        priority=20,
    )
    assert t_id is not None

    # Atomic claim with SKIP LOCKED
    task = pg_queue.fetch_next_task(worker_id="pg_worker_1")
    assert task is not None
    assert task.task_id == t_id
    assert task.status == "PROCESSING"

    # Concurrent fetch from worker 2 skips the locked row
    task_2 = pg_queue.fetch_next_task(worker_id="pg_worker_2")
    if task_2 is not None:
        assert task_2.task_id != t_id

    # Complete task
    pg_queue.complete_task(t_id)
