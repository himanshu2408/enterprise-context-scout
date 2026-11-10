"""Empirical Adversarial Challenge Suite for Milestone 2 (F05, F06, F07).

Authored by teamwork_challenger_m2_1.
Tests:
1. Concurrently racing 10 workers on task queue claiming (zero double-claims, zero skipped tasks).
2. Equal Jitter backoff mathematical bounds and automatic DLQ routing on retry exhaustion.
3. Poison pill corrupted payload isolation and worker crash immunity.
4. Webhook HMAC forgery, payload tampering, and timing attack resistance.
5. Recursive CTE cycle safety (cyclic graph A->B->C->A termination in <10ms).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import hmac
import statistics
import threading
import time
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import psycopg
import pytest
from starlette.testclient import TestClient

from context_scout.api.main import create_app
from context_scout.api.routes.webhooks import set_task_queue
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    InMemoryRLSRepository,
    PostgresRLSRepository,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.git_merkle_indexer import generate_deterministic_embedding
from context_scout.queue.indexing_worker import IndexingWorker
from context_scout.queue.task_queue import (
    InMemoryTaskQueue,
    PostgresTaskQueue,
    QueueTask,
    compute_backoff_delay,
)
from context_scout.schemas.contracts import PrincipalContext, RoleType

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def settings() -> ScoutSettings:
    return ScoutSettings()


@pytest.fixture
def in_memory_queue() -> InMemoryTaskQueue:
    q = InMemoryTaskQueue()
    set_task_queue(q)
    return q


@pytest.fixture
def app_client(in_memory_queue: InMemoryTaskQueue) -> TestClient:
    app = create_app()
    return TestClient(app)


@pytest.fixture
def principal_tester() -> PrincipalContext:
    return PrincipalContext(
        user_id="challenger_m2_runner",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "core_dev"],
    )


# =============================================================================
# Challenge 1: Queue Concurrency & Race Conditions (10 Workers, 100 Tasks)
# =============================================================================


def test_empirical_m2_challenge_01_in_memory_queue_10_worker_race_100_tasks() -> None:
    """Stress test: 10 worker threads claiming 100 queued tasks in InMemoryTaskQueue."""
    queue = InMemoryTaskQueue()
    domain_id = "trading_code_scout"
    num_tasks = 100
    num_workers = 10

    # 1. Enqueue 100 distinct tasks with diverse priorities
    enqueued_ids: list[UUID] = []
    for i in range(num_tasks):
        tid = queue.enqueue_task(
            domain_id=domain_id,
            event_type="gitlab_push",
            payload={"task_index": i, "commit": f"sha_{i:04d}"},
            priority=(i % 10) + 1,
        )
        enqueued_ids.append(tid)

    assert len(enqueued_ids) == num_tasks

    claimed_by_worker: dict[str, list[UUID]] = {f"worker_{w}": [] for w in range(num_workers)}
    barrier = threading.Barrier(num_workers)
    all_claimed_lock = threading.Lock()
    all_claimed: list[UUID] = []

    def worker_loop(worker_name: str) -> None:
        barrier.wait()
        while True:
            task = queue.fetch_next_task(worker_id=worker_name)
            if not task:
                break
            with all_claimed_lock:
                all_claimed.append(task.task_id)
                claimed_by_worker[worker_name].append(task.task_id)
            time.sleep(0.0001)
            queue.complete_task(task.task_id)

    # 2. Concurrently execute 10 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_loop, f"worker_{w}") for w in range(num_workers)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    # 3. Assertions: 0 double claims, 0 skipped tasks
    assert len(all_claimed) == num_tasks, f"Expected {num_tasks} claimed, got {len(all_claimed)}"
    unique_claimed = set(all_claimed)
    assert len(unique_claimed) == num_tasks, (
        f"Double-claim detected! {num_tasks - len(unique_claimed)} duplicate claims occurred."
    )
    assert unique_claimed == set(enqueued_ids), "Skipped tasks detected! Claimed != enqueued."

    # Verify queue statistics
    stats = queue.get_queue_stats()
    assert stats["PENDING"] == 0
    assert stats["PROCESSING"] == 0
    assert stats["COMPLETED"] == num_tasks
    assert stats["FAILED"] == 0
    assert stats["DLQ"] == 0

    active_workers = [w for w, tasks in claimed_by_worker.items() if len(tasks) > 0]
    assert len(active_workers) >= 5, f"Expected >=5 active workers, got {len(active_workers)}"


def test_empirical_m2_challenge_01_postgres_queue_10_worker_race_100_tasks(
    settings: ScoutSettings,
) -> None:
    """Stress test: 10 worker threads claiming 100 queued tasks in live PostgreSQL 17."""
    pg_queue = PostgresTaskQueue(db_url=settings.db_app_url)
    domain_id = "trading_code_scout"
    num_tasks = 100
    num_workers = 10

    # 1. Enqueue 100 tasks
    enqueued_ids: list[UUID] = []
    for i in range(num_tasks):
        tid = pg_queue.enqueue_task(
            domain_id=domain_id,
            event_type="cross_repo_relink",
            payload={"race_index": i, "batch": "challenger_m2_batch"},
            priority=(i % 5) + 1,
        )
        enqueued_ids.append(tid)

    assert len(enqueued_ids) == num_tasks

    claimed_lock = threading.Lock()
    all_claimed: list[UUID] = []
    barrier = threading.Barrier(num_workers)

    def pg_worker_loop(worker_name: str) -> None:
        worker_queue = PostgresTaskQueue(db_url=settings.db_app_url)
        barrier.wait()
        while True:
            task = worker_queue.fetch_next_task(worker_id=worker_name)
            if not task:
                break
            with claimed_lock:
                all_claimed.append(task.task_id)
            time.sleep(0.0001)
            worker_queue.complete_task(task.task_id)

    # 2. Race 10 threads against PostgreSQL
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(pg_worker_loop, f"pg_worker_{w}") for w in range(num_workers)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    our_claimed = [tid for tid in all_claimed if tid in set(enqueued_ids)]

    # 3. Assertions
    assert len(our_claimed) == num_tasks, f"Expected {num_tasks} claimed, got {len(our_claimed)}"
    assert len(set(our_claimed)) == num_tasks, "PostgreSQL SKIP LOCKED double-claim detected!"
    assert set(our_claimed) == set(enqueued_ids), "Skipped tasks detected in PostgreSQL queue!"


# =============================================================================
# Challenge 2: Exponential Jitter Bounds & Automatic DLQ Exhaustion
# =============================================================================


def test_empirical_m2_challenge_02_jitter_mathematical_bounds_empirical_distribution() -> None:
    """Stress test: 5,000 trials verifying Equal Jitter mathematical bounds and distribution."""
    base_delays = [0.5, 1.0, 2.0, 5.0]
    max_delays = [30.0, 60.0, 300.0]

    # Edge cases
    assert compute_backoff_delay(0) == 0.0
    assert compute_backoff_delay(-3) == 0.0

    # Clamping test
    clamped_delay = compute_backoff_delay(100, base_delay_seconds=2.0, max_delay_seconds=60.0)
    assert 30.0 <= clamped_delay <= 60.0

    # 5,000 statistical trials
    trials_per_config = 500
    for base in base_delays:
        for max_d in max_delays:
            for attempt in range(1, 6):
                delays = [
                    compute_backoff_delay(
                        attempt, base_delay_seconds=base, max_delay_seconds=max_d
                    )
                    for _ in range(trials_per_config)
                ]
                delay_cap = min(max_d, base * (2 ** (attempt - 1)))
                min_expected = delay_cap / 2.0
                max_expected = delay_cap

                for d in delays:
                    assert min_expected <= d <= max_expected, (
                        f"Jitter violation: delay {d} outside bounds [{min_expected}, "
                        f"{max_expected}] (attempt={attempt}, base={base}, max={max_d})"
                    )

                variance = statistics.pvariance(delays)
                assert variance > 0.0, "Jitter is deterministic, expected non-zero variance!"

                sample_mean = statistics.mean(delays)
                expected_mean = 0.75 * delay_cap
                assert abs(sample_mean - expected_mean) / expected_mean < 0.08, (
                    f"Mean {sample_mean} diverged from Equal Jitter center {expected_mean}"
                )


def test_empirical_m2_challenge_02_dlq_exhaustion_in_memory(
    in_memory_queue: InMemoryTaskQueue,
) -> None:
    """Stress test: Verify automatic DLQ routing upon exhausting retry attempts in InMemory."""
    domain_id = "trading_code_scout"
    max_attempts = 3

    task_id = in_memory_queue.enqueue_task(
        domain_id=domain_id,
        event_type="github_push",
        payload={"repo": "adversarial-repo", "head": "bad_commit"},
        max_attempts=max_attempts,
    )

    # Attempt 1
    t1 = in_memory_queue.fetch_next_task(worker_id="w1")
    assert t1 is not None and t1.attempts == 1
    retried1 = in_memory_queue.fail_task(task_id, error_message="Failure attempt 1")
    assert retried1 is True
    assert in_memory_queue.get_queue_stats()["DLQ"] == 0

    # Fast-forward scheduled time
    in_memory_queue._tasks[task_id]["scheduled_at"] = datetime.now(UTC) - timedelta(seconds=1)

    # Attempt 2
    t2 = in_memory_queue.fetch_next_task(worker_id="w1")
    assert t2 is not None and t2.attempts == 2
    retried2 = in_memory_queue.fail_task(task_id, error_message="Failure attempt 2")
    assert retried2 is True
    assert in_memory_queue.get_queue_stats()["DLQ"] == 0

    in_memory_queue._tasks[task_id]["scheduled_at"] = datetime.now(UTC) - timedelta(seconds=1)

    # Attempt 3 (Retry Exhaustion)
    t3 = in_memory_queue.fetch_next_task(worker_id="w1")
    assert t3 is not None and t3.attempts == 3
    retried3 = in_memory_queue.fail_task(
        task_id,
        error_message="Permanent schema failure",
        stack_trace="Traceback: SchemaValidationError: corrupt AST",
    )
    assert retried3 is False

    # Verify DLQ
    stats = in_memory_queue.get_queue_stats()
    assert stats["DLQ"] == 1
    assert stats["FAILED"] == 1
    assert stats["PENDING"] == 0

    dlqs = in_memory_queue.list_dead_letter_records()
    assert len(dlqs) == 1
    record = dlqs[0]
    assert record.task_id == task_id
    assert record.attempts == max_attempts
    assert record.failure_reason == "Permanent schema failure"
    assert "SchemaValidationError" in record.stack_trace

    # Verify recovery via retry_dead_letter_task
    recovered = in_memory_queue.retry_dead_letter_task(task_id)
    assert recovered is True
    assert in_memory_queue.get_queue_stats()["DLQ"] == 0
    assert in_memory_queue.get_queue_stats()["PENDING"] == 1

    t_reclaimed = in_memory_queue.fetch_next_task(worker_id="w1")
    assert t_reclaimed is not None
    assert t_reclaimed.task_id == task_id
    assert t_reclaimed.attempts == 1


def test_empirical_m2_challenge_02_dlq_exhaustion_live_postgres(
    settings: ScoutSettings,
) -> None:
    """Stress test: Verify automatic DLQ routing on retry exhaustion in live PostgreSQL 17."""
    pg_queue = PostgresTaskQueue(db_url=settings.db_app_url)
    domain_id = "trading_code_scout"
    max_attempts = 2

    task_id = pg_queue.enqueue_task(
        domain_id=domain_id,
        event_type="confluence_update",
        payload={"page_id": "adversarial-page-404"},
        max_attempts=max_attempts,
    )

    # Attempt 1: Fetch and fail
    t1 = pg_queue.fetch_next_task(worker_id="dlq_tester")
    assert t1 is not None and t1.task_id == task_id
    assert t1.attempts == 1

    requeued = pg_queue.fail_task(task_id, error_message="HTTP 503 Confluence unavailable")
    assert requeued is True

    # Advance scheduled_at in DB
    with psycopg.connect(settings.db_app_url) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE indexing_queue SET scheduled_at = NOW() - INTERVAL '1 second' "
            "WHERE task_id = %s;",
            (task_id,),
        )
        conn.commit()

    # Attempt 2: Fetch and fail (exhaustion)
    t2 = pg_queue.fetch_next_task(worker_id="dlq_tester")
    assert t2 is not None and t2.task_id == task_id
    assert t2.attempts == 2

    exhausted = pg_queue.fail_task(
        task_id,
        error_message="Permanent 404 Page Deleted",
        stack_trace="Traceback: PageNotFoundError: page does not exist",
    )
    assert exhausted is False

    # Verify live PostgreSQL DLQ table
    dlqs = pg_queue.list_dead_letter_records(limit=10)
    matching = [r for r in dlqs if r.task_id == task_id]
    assert len(matching) == 1
    assert matching[0].attempts == 2
    assert matching[0].failure_reason == "Permanent 404 Page Deleted"

    # Retry from DLQ
    retried = pg_queue.retry_dead_letter_task(task_id)
    assert retried is True

    # Complete task to leave clean state
    t_clean = pg_queue.fetch_next_task(worker_id="dlq_tester")
    assert t_clean is not None and t_clean.task_id == task_id
    pg_queue.complete_task(task_id)


# =============================================================================
# Challenge 3: Poison Pill Isolation & Worker Crash Immunity
# =============================================================================


def test_empirical_m2_challenge_03_poison_pill_handler_crash_isolation() -> None:
    """Stress test: Submit catastrophic poison pill payloads; assert worker loop never crashes."""
    queue = InMemoryTaskQueue()
    worker = IndexingWorker(task_queue=queue, worker_id="resilient_worker_1")

    def corrupt_gitlab_handler(task: QueueTask) -> None:
        if task.payload.get("poison") == "zero_div":
            raise ZeroDivisionError("Catastrophic divide by zero in git diff")
        if task.payload.get("poison") == "missing_key":
            _ = task.payload["non_existent_key"]
        if task.payload.get("poison") == "memory_sim":
            raise MemoryError("Out of memory in tree-sitter AST parser")
        task.payload["processed"] = True

    worker.register_handler("gitlab_push", corrupt_gitlab_handler)

    # 1. Enqueue 5 poison pills (max_attempts=1 for immediate DLQ)
    poison_types = ["zero_div", "missing_key", "memory_sim", "zero_div", "memory_sim"]
    poison_ids: list[UUID] = []
    for p in poison_types:
        tid = queue.enqueue_task(
            domain_id="trading_code_scout",
            event_type="gitlab_push",
            payload={"poison": p, "data": "corrupted"},
            max_attempts=1,
        )
        poison_ids.append(tid)

    # 2. Enqueue 5 healthy tasks
    healthy_ids: list[UUID] = []
    for i in range(5):
        tid = queue.enqueue_task(
            domain_id="trading_code_scout",
            event_type="gitlab_push",
            payload={"healthy_index": i},
            max_attempts=3,
        )
        healthy_ids.append(tid)

    # 3. Worker processes all 10 tasks sequentially via run_once()
    for _ in range(10):
        processed = worker.run_once()
        assert processed is True, "Worker failed to claim available task!"

    # No more pending tasks
    assert worker.run_once() is False

    # 4. Assert worker health and task isolation
    assert worker.is_running() is True
    stats = queue.get_queue_stats()
    assert stats["DLQ"] == 5, f"Expected 5 DLQ records, got {stats['DLQ']}"
    assert stats["COMPLETED"] == 5, f"Expected 5 completed healthy tasks, got {stats['COMPLETED']}"
    assert stats["FAILED"] == 5

    dlqs = queue.list_dead_letter_records(limit=10)
    reasons = [d.failure_reason for d in dlqs]
    assert any("Catastrophic divide by zero" in r for r in reasons)
    assert any("non_existent_key" in r for r in reasons)
    assert any("Out of memory" in r for r in reasons)


def test_empirical_m2_challenge_03_background_worker_thread_liveness_under_fire() -> None:
    """Stress test: Run IndexingWorker in a background thread while pumping poison pills."""
    queue = InMemoryTaskQueue()
    worker = IndexingWorker(
        task_queue=queue,
        worker_id="thread_liveness_worker",
        poll_interval_seconds=0.01,
        max_idle_backoff_seconds=0.05,
    )

    def failing_jira_handler(task: QueueTask) -> None:
        if task.payload.get("fatal"):
            raise RuntimeError("Fatal explosion in Jira issue parsing")

    worker.register_handler("jira_update", failing_jira_handler)

    worker_thread = threading.Thread(target=worker.run, daemon=True)
    worker_thread.start()

    try:
        # Enqueue 15 poison pills and 15 healthy tasks
        for i in range(15):
            queue.enqueue_task(
                domain_id="trading_code_scout",
                event_type="jira_update",
                payload={"fatal": True, "idx": i},
                max_attempts=1,
            )
            queue.enqueue_task(
                domain_id="trading_code_scout",
                event_type="jira_update",
                payload={"fatal": False, "idx": i},
                max_attempts=2,
            )

        deadline = time.time() + 5.0
        while time.time() < deadline:
            stats = queue.get_queue_stats()
            if stats["COMPLETED"] == 15 and stats["DLQ"] == 15:
                break
            time.sleep(0.05)

        stats = queue.get_queue_stats()
        assert stats["COMPLETED"] == 15
        assert stats["DLQ"] == 15
        assert worker_thread.is_alive() is True, "Worker thread died unexpectedly!"

    finally:
        worker.stop()
        worker_thread.join(timeout=2.0)
        assert not worker_thread.is_alive()


# =============================================================================
# Challenge 4: Webhook Security, HMAC Forgery & Timing Attack Resistance
# =============================================================================


def test_empirical_m2_challenge_04_github_hmac_tampering_forgery_suite(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stress test: Exhaustive forgery, payload tampering, and header permutations."""
    secret = "adversarial_gh_secret_2026_argon_key"
    monkeypatch.setenv("SCOUT_WEBHOOK_GITHUB_SECRET", secret)

    valid_payload = b'{"ref": "refs/heads/main", "repository": {"name": "core-engine"}}'
    valid_sig = "sha256=" + hmac.new(secret.encode(), valid_payload, hashlib.sha256).hexdigest()

    # 1. Valid request -> 200 Queued
    res_valid = app_client.post(
        "/api/v1/webhooks/github",
        content=valid_payload,
        headers={"X-Hub-Signature-256": valid_sig, "Content-Type": "application/json"},
    )
    assert res_valid.status_code == 200
    assert res_valid.json()["status"] == "queued"

    wrong_sig_hash = hmac.new(b"wrong_secret", valid_payload, hashlib.sha256).hexdigest()
    raw_hash = hmac.new(secret.encode(), valid_payload, hashlib.sha256).hexdigest()

    # 2. Forged & Malformed Cases -> Must all return 401
    adversarial_cases: list[tuple[str, bytes, dict[str, str]]] = [
        ("missing_header", valid_payload, {"Content-Type": "application/json"}),
        (
            "empty_sig",
            valid_payload,
            {"X-Hub-Signature-256": "", "Content-Type": "application/json"},
        ),
        (
            "random_sig",
            valid_payload,
            {"X-Hub-Signature-256": "sha256=" + "a" * 64, "Content-Type": "application/json"},
        ),
        (
            "missing_prefix",
            valid_payload,
            {"X-Hub-Signature-256": raw_hash, "Content-Type": "application/json"},
        ),
        (
            "sha1_prefix",
            valid_payload,
            {"X-Hub-Signature-256": "sha1=" + "a" * 40, "Content-Type": "application/json"},
        ),
        (
            "truncated_sig",
            valid_payload,
            {"X-Hub-Signature-256": valid_sig[:30], "Content-Type": "application/json"},
        ),
        (
            "extended_sig",
            valid_payload,
            {"X-Hub-Signature-256": valid_sig + "extra", "Content-Type": "application/json"},
        ),
        (
            "wrong_secret",
            valid_payload,
            {
                "X-Hub-Signature-256": "sha256=" + wrong_sig_hash,
                "Content-Type": "application/json",
            },
        ),
        (
            "tampered_payload",
            b'{"ref": "refs/heads/pain", "repository": {"name": "core-engine"}}',
            {"X-Hub-Signature-256": valid_sig, "Content-Type": "application/json"},
        ),
        (
            "trailing_space",
            valid_payload,
            {"X-Hub-Signature-256": valid_sig + " ", "Content-Type": "application/json"},
        ),
        (
            "null_byte",
            valid_payload,
            {"X-Hub-Signature-256": valid_sig + "\x00", "Content-Type": "application/json"},
        ),
    ]

    for case_name, body, headers in adversarial_cases:
        res = app_client.post("/api/v1/webhooks/github", content=body, headers=headers)
        assert res.status_code == 401, (
            f"Security flaw! Case '{case_name}' returned {res.status_code} instead of 401"
        )


def test_empirical_m2_challenge_04_gitlab_token_tampering_suite(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stress test: Exhaustive forgery and header permutations on GitLab webhook token."""
    secret = "gitlab_secure_corp_token_m2_challenge"
    monkeypatch.setenv("SCOUT_WEBHOOK_GITLAB_SECRET", secret)

    payload = {"project": {"name": "payment-gateway"}, "domain_id": "trading_code_scout"}

    # 1. Valid token -> 200 Queued
    res_ok = app_client.post(
        "/api/v1/webhooks/gitlab",
        json=payload,
        headers={"X-Gitlab-Token": secret},
    )
    assert res_ok.status_code == 200

    # 2. Forged cases -> 401
    bad_tokens = [
        "",
        "wrong_token",
        secret[:10],
        secret[10:],
        secret + "_extra",
        secret[:-1] + "X",
        secret.upper(),
        f"{secret}\x00extra",
        f"Bearer {secret}",
    ]

    for tok in bad_tokens:
        res = app_client.post(
            "/api/v1/webhooks/gitlab",
            json=payload,
            headers={"X-Gitlab-Token": tok},
        )
        assert res.status_code == 401, f"Security breach: Token '{tok}' returned {res.status_code}"

    res_none = app_client.post("/api/v1/webhooks/gitlab", json=payload)
    assert res_none.status_code == 401


def test_empirical_m2_challenge_04_webhook_timing_attack_resistance(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empirically measures timing variance between prefix-matched and wrong tokens."""
    secret = "A" * 64
    monkeypatch.setenv("SCOUT_WEBHOOK_GITLAB_SECRET", secret)
    payload = {"domain_id": "trading_code_scout"}

    wrong_token = "B" * 64
    almost_matching = "A" * 63 + "B"

    trials = 200
    times_wrong: list[float] = []
    times_almost: list[float] = []

    for _ in range(trials):
        t0 = time.perf_counter_ns()
        res1 = app_client.post(
            "/api/v1/webhooks/gitlab",
            json=payload,
            headers={"X-Gitlab-Token": wrong_token},
        )
        t1 = time.perf_counter_ns()
        assert res1.status_code == 401
        times_wrong.append(t1 - t0)

        t2 = time.perf_counter_ns()
        res2 = app_client.post(
            "/api/v1/webhooks/gitlab",
            json=payload,
            headers={"X-Gitlab-Token": almost_matching},
        )
        t3 = time.perf_counter_ns()
        assert res2.status_code == 401
        times_almost.append(t3 - t2)

    mean_wrong = statistics.mean(times_wrong)
    mean_almost = statistics.mean(times_almost)

    ratio = max(mean_wrong, mean_almost) / min(mean_wrong, mean_almost)
    assert ratio < 1.30, (
        f"Possible timing leak! Ratio: {ratio:.3f} (wrong={mean_wrong}ns, almost={mean_almost}ns)"
    )


# =============================================================================
# Challenge 5: Recursive CTE Cycles & Graph Traversal Safety (<10ms Termination)
# =============================================================================


def test_empirical_m2_challenge_05_cyclic_dependency_graph_in_memory(
    principal_tester: PrincipalContext,
) -> None:
    """Stress test: Traverse circular dependency graph (A -> B -> C -> A) in InMemory."""
    repo = InMemoryRLSRepository()
    domain_id = "trading_code_scout"
    scopes = ["core_dev"]

    file_a = uuid4()
    file_b = uuid4()
    file_c = uuid4()

    for fid, path in [(file_a, "src/a.py"), (file_b, "src/b.py"), (file_c, "src/c.py")]:
        repo.insert_code_file(
            principal_tester,
            CodeFileRecord(
                file_id=fid,
                domain_id=domain_id,
                repo_name="core-mono",
                file_path=path,
                git_commit_sha="c_head",
                git_blob_sha=f"blob_{fid}",
                language="python",
                line_count=100,
                raw_content="# source",
                allowed_scopes=scopes,
            ),
        )

    sym_a = uuid4()
    sym_b = uuid4()
    sym_c = uuid4()

    symbols = [
        (sym_a, file_a, "ModuleA::ClassA"),
        (sym_b, file_b, "ModuleB::ClassB"),
        (sym_c, file_c, "ModuleC::ClassC"),
    ]
    for sid, fid, qname in symbols:
        repo.insert_ast_symbol(
            principal_tester,
            ASTSymbolRecord(
                symbol_id=sid,
                file_id=fid,
                domain_id=domain_id,
                qualified_name=qname,
                symbol_kind="class",
                start_line=1,
                end_line=20,
                signature_skeleton=f"class {qname.split('::')[-1]}",
                contextual_summary="Cyclic test symbol",
                allowed_scopes=scopes,
            ),
        )

    # 1. Circular USES_TYPE cycle: A -> B -> C -> A
    # 2. Circular CALLS cycle: A -> B -> C -> A
    # 3. Self-loops: A -> A (USES_TYPE and CALLS)
    edges = [
        (sym_a, sym_b, "USES_TYPE"),
        (sym_b, sym_c, "USES_TYPE"),
        (sym_c, sym_a, "USES_TYPE"),
        (sym_a, sym_b, "CALLS"),
        (sym_b, sym_c, "CALLS"),
        (sym_c, sym_a, "CALLS"),
        (sym_a, sym_a, "USES_TYPE"),
        (sym_a, sym_a, "CALLS"),
    ]
    for s_id, t_id, kind in edges:
        repo.insert_ast_edge(
            principal_tester,
            ASTEdgeRecord(
                edge_id=uuid4(),
                domain_id=domain_id,
                source_symbol_id=s_id,
                target_symbol_id=t_id,
                edge_kind=kind,
                target_qualified_name="target",
                allowed_scopes=scopes,
            ),
        )

    # Traversal on Focus = A
    t_start = time.perf_counter()
    neighborhood = repo.traverse_1hop_neighborhood(
        principal_or_conn=principal_tester,
        domain_id=domain_id,
        focus_symbol_id=sym_a,
    )
    duration_ms = (time.perf_counter() - t_start) * 1000.0

    assert duration_ms < 10.0, f"InMemory traversal exceeded 10ms SLA: {duration_ms:.2f}ms"
    assert len(neighborhood) > 0

    result_sym_ids = [rec.symbol.symbol_id for rec in neighborhood]
    assert len(result_sym_ids) == len(set(result_sym_ids)), "Neighborhood has duplicate symbols!"
    assert sym_a not in result_sym_ids

    for rec in neighborhood:
        assert rec.depth in (1, 2)


def test_empirical_m2_challenge_05_cyclic_dependency_graph_live_postgres_cte(
    settings: ScoutSettings, principal_tester: PrincipalContext
) -> None:
    """Stress test: Circular graph cycle safety in PostgreSQL 17 Recursive CTE."""
    mgr = RLSConnectionManager(settings.db_app_url)
    repo = PostgresRLSRepository()
    domain_id = "trading_code_scout"
    scopes = ["public_eng"]

    file_a = uuid4()
    file_b = uuid4()
    file_c = uuid4()
    sym_a = uuid4()
    sym_b = uuid4()
    sym_c = uuid4()

    files = [
        (file_a, "src/pg_cycle_a.py"),
        (file_b, "src/pg_cycle_b.py"),
        (file_c, "src/pg_cycle_c.py"),
    ]

    with mgr.scoped_transaction(principal_tester, enable_hnsw_iterative_scan=False) as conn:
        for fid, path in files:
            repo.insert_code_file(
                conn,
                CodeFileRecord(
                    file_id=fid,
                    domain_id=domain_id,
                    repo_name="live-cycle-repo",
                    file_path=path,
                    git_commit_sha="live_sha",
                    git_blob_sha=f"blob_{fid}",
                    language="python",
                    line_count=50,
                    raw_content="# live cycle content",
                    allowed_scopes=scopes,
                ),
            )

        sym_defs = [
            (sym_a, file_a, "PG::CycleA"),
            (sym_b, file_b, "PG::CycleB"),
            (sym_c, file_c, "PG::CycleC"),
        ]
        for sid, fid, qname in sym_defs:
            repo.insert_ast_symbol(
                conn,
                ASTSymbolRecord(
                    symbol_id=sid,
                    file_id=fid,
                    domain_id=domain_id,
                    qualified_name=qname,
                    symbol_kind="class",
                    start_line=1,
                    end_line=20,
                    signature_skeleton=f"class {qname}",
                    contextual_summary="PG cycle test",
                    allowed_scopes=scopes,
                    embedding=generate_deterministic_embedding(qname),
                ),
            )

        edge_data = [
            (sym_a, sym_b, "USES_TYPE"),
            (sym_b, sym_c, "USES_TYPE"),
            (sym_c, sym_a, "USES_TYPE"),
            (sym_a, sym_b, "CALLS"),
            (sym_b, sym_c, "CALLS"),
            (sym_c, sym_a, "CALLS"),
            (sym_a, sym_a, "USES_TYPE"),
        ]
        for s_id, t_id, kind in edge_data:
            repo.insert_ast_edge(
                conn,
                ASTEdgeRecord(
                    edge_id=uuid4(),
                    domain_id=domain_id,
                    source_symbol_id=s_id,
                    target_symbol_id=t_id,
                    edge_kind=kind,
                    target_qualified_name="PG::Target",
                    allowed_scopes=scopes,
                ),
            )

        durations: list[float] = []
        results = []
        for _ in range(3):
            t_start = time.perf_counter()
            results = repo.traverse_1hop_neighborhood(
                conn=conn,
                domain_id=domain_id,
                focus_symbol_id=sym_a,
                max_direct_callers=3,
            )
            durations.append((time.perf_counter() - t_start) * 1000.0)

        min_duration_ms = min(durations)
        assert min_duration_ms < 10.0, (
            f"PostgreSQL Recursive CTE exceeded 10ms SLA: min={min_duration_ms:.2f}ms "
            f"(all runs: {[f'{d:.2f}ms' for d in durations]})"
        )
        assert len(results) > 0

        sym_role_pairs = [(r.symbol.symbol_id, r.intent_role) for r in results]
        assert len(sym_role_pairs) == len(set(sym_role_pairs)), (
            "PostgreSQL CTE produced duplicate (symbol_id, intent_role) pairs!"
        )

        for r in results:
            assert r.depth in (1, 2)

        conn.execute(
            "DELETE FROM ast_edges WHERE domain_id = %s "
            "AND source_symbol_id IN (%s, %s, %s);",
            (domain_id, sym_a, sym_b, sym_c),
        )
        conn.execute(
            "DELETE FROM ast_symbols WHERE domain_id = %s "
            "AND symbol_id IN (%s, %s, %s);",
            (domain_id, sym_a, sym_b, sym_c),
        )
        conn.execute(
            "DELETE FROM code_files WHERE domain_id = %s "
            "AND file_id IN (%s, %s, %s);",
            (domain_id, file_a, file_b, file_c),
        )


def test_empirical_m2_challenge_05_dense_clique_graph_scale(
    principal_tester: PrincipalContext,
) -> None:
    """Stress test: 6-node dense complete graph (clique)."""
    repo = InMemoryRLSRepository()
    domain_id = "trading_code_scout"
    scopes = ["core_dev"]
    num_nodes = 6

    node_files: list[UUID] = [uuid4() for _ in range(num_nodes)]
    node_syms: list[UUID] = [uuid4() for _ in range(num_nodes)]

    for i in range(num_nodes):
        repo.insert_code_file(
            principal_tester,
            CodeFileRecord(
                file_id=node_files[i],
                domain_id=domain_id,
                repo_name="clique_repo",
                file_path=f"src/node_{i}.py",
                git_commit_sha="clique_sha",
                git_blob_sha=f"blob_{i}",
                language="python",
                line_count=20,
                raw_content="",
                allowed_scopes=scopes,
            ),
        )
        repo.insert_ast_symbol(
            principal_tester,
            ASTSymbolRecord(
                symbol_id=node_syms[i],
                file_id=node_files[i],
                domain_id=domain_id,
                qualified_name=f"Clique::Node_{i}",
                symbol_kind="class",
                start_line=1,
                end_line=10,
                signature_skeleton=f"class Node_{i}",
                contextual_summary="",
                allowed_scopes=scopes,
            ),
        )

    for src in node_syms:
        for tgt in node_syms:
            if src != tgt:
                for kind in ("USES_TYPE", "CALLS"):
                    repo.insert_ast_edge(
                        principal_tester,
                        ASTEdgeRecord(
                            edge_id=uuid4(),
                            domain_id=domain_id,
                            source_symbol_id=src,
                            target_symbol_id=tgt,
                            edge_kind=kind,
                            target_qualified_name="target",
                            allowed_scopes=scopes,
                        ),
                    )

    t0 = time.perf_counter()
    neighborhood = repo.traverse_1hop_neighborhood(
        principal_or_conn=principal_tester,
        domain_id=domain_id,
        focus_symbol_id=node_syms[0],
        max_direct_callers=3,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 10.0, f"Dense clique traversal took too long: {elapsed_ms:.2f}ms"
    assert len(neighborhood) <= num_nodes
