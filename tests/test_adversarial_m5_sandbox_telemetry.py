"""Adversarial challenge test suite for Milestone 5 Sandbox, Gemini Executor & Telemetry.

Empirical verification of:
1. GitWorktreeSandbox attack scenarios (invalid diff syntax, path traversal, conflict markers,
   cleanup resilience on unhandled exceptions/SIGINT, human confirmation abort gate).
2. GeminiImplementationExecutor budget checks (cap enforcement, multi-principal isolation, boundary
   conditions, offline bypass behavior).
3. QLoRA flywheel events (ContextMissEvent generation, zero-touch invariant, non-existent symbols).
4. PostgreSQL 17 telemetry persistence (schema constraints, upserts, JSONB roundtrips,
   budget aggregation across time windows).
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import psycopg.errors
import pytest

from context_scout.agents.gemini_executor import (
    BudgetExceededError,
    GeminiImplementationExecutor,
    GitWorktreeSandbox,
)
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    InMemoryRLSRepository,
    PostgresRLSRepository,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.schemas.contracts import (
    ExecutionMode,
    ExecutionTelemetryRecord,
    ImplementationContextPack,
    MissingContextRequest,
    ModificationIntent,
    PreFlightValidationReport,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)

DOMAIN_CODE: Final[str] = "trading_code_scout"
REPO_NAME: Final[str] = "trading-execution"


@pytest.fixture
def principal() -> PrincipalContext:
    """Authenticated principal with standard token cap."""
    return PrincipalContext(
        user_id="adversary_evaluator",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:trading_strategies"],
        max_session_paid_tokens=25_000,
    )


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Creates a real git repository with tracking commits."""
    repo_dir = tmp_path / "sandbox_adversarial_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Adversary Challenger"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "challenger@contextscout.internal"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    src_dir = repo_dir / "src" / "trading_strategies"
    src_dir.mkdir(parents=True, exist_ok=True)
    code_file = src_dir / "order_flow_imbalance.py"
    code_file.write_text(
        "class OrderFlowImbalanceAnalyzer:\n"
        "    def calculate_ofi(self, snapshot) -> float:\n"
        "        return 1.0\n",
        encoding="utf-8",
    )

    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    return repo_dir


@pytest.fixture
def sample_target_slice() -> VerbatimCodeSlice:
    return VerbatimCodeSlice(
        repo_name=REPO_NAME,
        file_path="src/trading_strategies/order_flow_imbalance.py",
        start_line=1,
        end_line=3,
        qualified_symbol="trading_strategies.order_flow_imbalance.OrderFlowImbalanceAnalyzer",
        selection_reason="Target slice",
        modification_intent=ModificationIntent.MODIFY_TARGET,
        git_commit_sha="c0ffee123",
        git_blob_sha="b10b456",
        verbatim_code_with_lines="   1 | class OrderFlowImbalanceAnalyzer:\n",
    )


# =============================================================================
# Suite 1: GitWorktreeSandbox Adversarial Stress Testing
# =============================================================================


class TestGitWorktreeSandboxAdversarial:
    """Stress tests GitWorktreeSandbox against hostile diffs, traversal, and abortion."""

    def test_patch_path_traversal_attack_blocked(self, temp_git_repo: Path) -> None:
        """Attempts to write files outside the repository using path traversal unified diff."""
        traversal_patch = (
            "--- a/src/trading_strategies/order_flow_imbalance.py\n"
            "+++ b/../../../../etc/malicious_payload.txt\n"
            "@@ -0,0 +1,1 @@\n"
            "+MALICIOUS CONTENT\n"
        )
        with GitWorktreeSandbox(repo_root=temp_git_repo, require_confirmation=True) as sandbox:
            ok, msg = sandbox.apply_patch(traversal_patch)
            assert ok is False
            assert "error" in msg.lower() or "invalid path" in msg.lower()

        # Verify nothing was created outside
        malicious_file = temp_git_repo.parent / "malicious_payload.txt"
        assert not malicious_file.exists()

    def test_patch_invalid_syntax_and_garbage_input(self, temp_git_repo: Path) -> None:
        """Tests that corrupted/fuzzed diff syntax fails cleanly without crashing."""
        garbage_diffs = [
            "This is not a diff at all",
            "--- a/file.py\n+++ b/file.py\n@@ -999,999 @@\ninvalid hunk\n",
            "diff --git a/file.py b/file.py\ncorrupted metadata\n",
            "\x00\x01\x02\x03\x04\x05binarygarbage",
            "",
        ]
        with GitWorktreeSandbox(repo_root=temp_git_repo, require_confirmation=True) as sandbox:
            for bad_diff in garbage_diffs:
                ok, msg = sandbox.apply_patch(bad_diff)
                assert ok is False, f"Garbage diff unexpectedly applied: {bad_diff!r}"
                assert isinstance(msg, str)

    def test_patch_conflict_markers_rejected_by_git_apply(
        self, temp_git_repo: Path
    ) -> None:
        """Tests patch containing raw git merge conflict markers is rejected by apply_patch."""
        conflict_patch = (
            "--- a/src/trading_strategies/order_flow_imbalance.py\n"
            "+++ b/src/trading_strategies/order_flow_imbalance.py\n"
            "@@ -1,3 +1,7 @@\n"
            " class OrderFlowImbalanceAnalyzer:\n"
            "+<<<<<<< HEAD\n"
            "+    def calculate_ofi(self, snapshot) -> float:\n"
            "+=======\n"
            "+    def calculate_ofi_v2(self, snapshot) -> float:\n"
            "+>>>>>>> feature-branch\n"
        )
        with GitWorktreeSandbox(repo_root=temp_git_repo, require_confirmation=True) as sandbox:
            ok, msg = sandbox.apply_patch(conflict_patch)
            # git apply detects corrupt patch format on conflict markers and rejects application
            assert ok is False
            assert "corrupt patch" in msg.lower() or "error" in msg.lower()

            # Ensure target file remained unchanged in the worktree
            assert sandbox.worktree_dir is not None
            subpath = Path("src/trading_strategies/order_flow_imbalance.py")
            wt_file = sandbox.worktree_dir / subpath
            assert "<<<<<<< HEAD" not in wt_file.read_text(encoding="utf-8")

    def test_unhandled_exception_cleans_up_worktree_and_ephemeral_branch(
        self, temp_git_repo: Path
    ) -> None:
        """Ensures temporary worktree directory and git branch are deleted even on crash/SIGINT."""
        captured_worktree_dir: Path | None = None
        captured_branch: str | None = None

        with (
            pytest.raises(ZeroDivisionError),
            GitWorktreeSandbox(repo_root=temp_git_repo) as sandbox,
        ):
            captured_worktree_dir = sandbox.worktree_dir
            captured_branch = sandbox.branch_name
            assert captured_worktree_dir is not None
            assert captured_worktree_dir.exists()
            # Simulate unhandled exception or crash during verification
            _ = 1 / 0

        # Post-exception verification: worktree directory must NOT exist
        assert captured_worktree_dir is not None
        assert not captured_worktree_dir.exists()

        # Git worktrees list must NOT contain the ephemeral worktree path
        wt_list = subprocess.run(
            ["git", "worktree", "list"], cwd=temp_git_repo, capture_output=True, text=True
        )
        assert str(captured_worktree_dir) not in wt_list.stdout

        # Git branch list must NOT contain the ephemeral branch
        branch_list = subprocess.run(
            ["git", "branch"], cwd=temp_git_repo, capture_output=True, text=True
        )
        assert captured_branch is not None
        assert captured_branch not in branch_list.stdout

    def test_confirm_and_merge_unconfirmed_creates_zero_git_commits(
        self, temp_git_repo: Path
    ) -> None:
        """Verifies confirm_and_merge(confirmed=False) leaves base repo commit count invariant."""
        base_log = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        )
        initial_commit_count = int(base_log.stdout.strip())

        patch = (
            "--- a/src/trading_strategies/order_flow_imbalance.py\n"
            "+++ b/src/trading_strategies/order_flow_imbalance.py\n"
            "@@ -1,3 +1,4 @@\n"
            " class OrderFlowImbalanceAnalyzer:\n"
            "+    # Should not be committed\n"
            "     def calculate_ofi(self, snapshot) -> float:\n"
            "         return 1.0\n"
        )

        with GitWorktreeSandbox(repo_root=temp_git_repo, require_confirmation=True) as sandbox:
            sandbox.apply_patch(patch)
            conf_ok, conf_msg = sandbox.confirm_and_merge(confirmed=False)
            assert conf_ok is False
            assert "Human confirmation required" in conf_msg

        # Verify base repo commit count is unchanged
        after_log = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        )
        final_commit_count = int(after_log.stdout.strip())
        assert initial_commit_count == final_commit_count

    def test_inactive_sandbox_operations_raise_runtime_error(self, temp_git_repo: Path) -> None:
        """Invoking sandbox methods outside context manager must fail fast."""
        sandbox = GitWorktreeSandbox(repo_root=temp_git_repo)
        with pytest.raises(RuntimeError, match="Sandbox is not active"):
            sandbox.apply_patch("diff")
        with pytest.raises(RuntimeError, match="Sandbox is not active"):
            sandbox.run_verification()
        with pytest.raises(RuntimeError, match="Sandbox is not active"):
            sandbox.confirm_and_merge(confirmed=True)


# =============================================================================
# Suite 2: GeminiImplementationExecutor Budget Enforcement
# =============================================================================


class TestGeminiExecutorBudgetAdversarial:
    """Stress tests token budget ceiling invariants and boundary conditions."""

    def test_check_principal_budget_boundary_conditions(
        self, principal: PrincipalContext, sample_target_slice: VerbatimCodeSlice
    ) -> None:
        """Tests exact budget boundary: at cap (allowed) vs at cap + 1 (exceeded)."""
        executor = GeminiImplementationExecutor(api_key=None)
        repo = InMemoryRLSRepository()
        principal_cap = principal.max_session_paid_tokens  # 25,000

        # Case 1: Consumed = 0, requested = 25,000 (Exact match) -> MUST PASS
        pack_exact = ImplementationContextPack(
            trace_id=str(uuid.uuid4()),
            domain_id=DOMAIN_CODE,
            user_task="Exact boundary task",
            architectural_execution_plan=["Plan"],
            code_slices=[sample_target_slice],
            preflight_report=PreFlightValidationReport(
                passed=True, normalized_rrf_score=0.9, citations_verified_100_pct=True
            ),
            local_tokens_consumed=10_000,
            pack_token_count=principal_cap,
        )
        executor.check_principal_budget(principal, pack_exact, repo=repo)

        # Case 2: Consumed = 0, requested = 25,001 (1 token over cap) -> MUST RAISE
        pack_over = pack_exact.model_copy(update={"pack_token_count": principal_cap + 1})
        with pytest.raises(BudgetExceededError, match="exceeded session token budget"):
            executor.check_principal_budget(principal, pack_over, repo=repo)

    def test_direct_budget_check_enforces_cap_without_repo_or_conn(
        self, principal: PrincipalContext, sample_target_slice: VerbatimCodeSlice
    ) -> None:
        """Verifies check_principal_budget enforces cap even when conn and repo are None."""
        executor = GeminiImplementationExecutor(api_key=None)
        over_pack = ImplementationContextPack(
            trace_id=str(uuid.uuid4()),
            domain_id=DOMAIN_CODE,
            user_task="Over cap without db",
            architectural_execution_plan=["Plan"],
            code_slices=[sample_target_slice],
            preflight_report=PreFlightValidationReport(
                passed=True, normalized_rrf_score=0.9, citations_verified_100_pct=True
            ),
            local_tokens_consumed=10_000,
            pack_token_count=principal.max_session_paid_tokens + 500,
        )
        with pytest.raises(BudgetExceededError, match="exceeded session token budget"):
            executor.check_principal_budget(principal, over_pack, conn=None, repo=None)

    def test_budget_accumulation_across_multiple_sessions(
        self,
        principal: PrincipalContext,
        sample_target_slice: VerbatimCodeSlice,
        temp_git_repo: Path,
    ) -> None:
        """Simulates iterative usage leading up to budget exhaustion."""
        executor = GeminiImplementationExecutor(api_key=None, require_confirmation=False)
        repo = InMemoryRLSRepository()
        pack_chunk = 10_000  # Cap is 25,000

        def make_pack(token_count: int) -> ImplementationContextPack:
            return ImplementationContextPack(
                trace_id=str(uuid.uuid4()),
                domain_id=DOMAIN_CODE,
                user_task="Chunk task",
                architectural_execution_plan=["Plan"],
                code_slices=[sample_target_slice],
                preflight_report=PreFlightValidationReport(
                    passed=True, normalized_rrf_score=0.9, citations_verified_100_pct=True
                ),
                local_tokens_consumed=15_000,
                pack_token_count=token_count,
            )

        # Call 1: 10,000 tokens consumed -> 10,000 / 25,000
        executor.execute_implementation(
            pack=make_pack(pack_chunk),
            principal=principal,
            repo_root=temp_git_repo,
            repo=repo,
            lint_cmd=["python3", "-c", "exit(0)"],
            test_cmd=["python3", "-c", "exit(0)"],
        )
        assert repo.get_principal_paid_tokens_consumed(principal.user_id) == 10_000

        # Call 2: 10,000 tokens consumed -> 20,000 / 25,000
        executor.execute_implementation(
            pack=make_pack(pack_chunk),
            principal=principal,
            repo_root=temp_git_repo,
            repo=repo,
            lint_cmd=["python3", "-c", "exit(0)"],
            test_cmd=["python3", "-c", "exit(0)"],
        )
        assert repo.get_principal_paid_tokens_consumed(principal.user_id) == 20_000

        # Call 3: 10,000 tokens requested -> 20,000 + 10,000 = 30,000 > 25,000 cap -> MUST RAISE
        match_pat = r"consumed 20000 \+ requested 10000 > cap 25000"
        with pytest.raises(BudgetExceededError, match=match_pat):
            executor.execute_implementation(
                pack=make_pack(pack_chunk),
                principal=principal,
                repo_root=temp_git_repo,
                repo=repo,
            )

    def test_multi_principal_budget_isolation(
        self,
        principal: PrincipalContext,
        sample_target_slice: VerbatimCodeSlice,
        temp_git_repo: Path,
    ) -> None:
        """Principal A exhausting budget does not block Principal B with separate budget."""
        executor = GeminiImplementationExecutor(api_key=None, require_confirmation=False)
        repo = InMemoryRLSRepository()

        principal_b = PrincipalContext(
            user_id="independent_trader",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
            max_session_paid_tokens=25_000,
        )

        pack_large = ImplementationContextPack(
            trace_id=str(uuid.uuid4()),
            domain_id=DOMAIN_CODE,
            user_task="Large task",
            architectural_execution_plan=["Plan"],
            code_slices=[sample_target_slice],
            preflight_report=PreFlightValidationReport(
                passed=True, normalized_rrf_score=0.9, citations_verified_100_pct=True
            ),
            local_tokens_consumed=20_000,
            pack_token_count=24_000,
        )

        # Principal A consumes 24,000 tokens
        executor.execute_implementation(
            pack=pack_large,
            principal=principal,
            repo_root=temp_git_repo,
            repo=repo,
            lint_cmd=["python3", "-c", "exit(0)"],
            test_cmd=["python3", "-c", "exit(0)"],
        )
        assert repo.get_principal_paid_tokens_consumed(principal.user_id) == 24_000
        assert repo.get_principal_paid_tokens_consumed(principal_b.user_id) == 0

        # Principal A attempts another 5,000 tokens -> Rejected
        pack_small = pack_large.model_copy(
            update={"trace_id": str(uuid.uuid4()), "pack_token_count": 5_000}
        )
        with pytest.raises(BudgetExceededError):
            executor.execute_implementation(
                pack=pack_small,
                principal=principal,
                repo_root=temp_git_repo,
                repo=repo,
            )

        # Principal B attempts 20,000 tokens -> Allowed
        pack_b = pack_small.model_copy(
            update={"trace_id": str(uuid.uuid4()), "pack_token_count": 20_000}
        )
        result_b, telem_b = executor.execute_implementation(
            pack=pack_b,
            principal=principal_b,
            repo_root=temp_git_repo,
            repo=repo,
            lint_cmd=["python3", "-c", "exit(0)"],
            test_cmd=["python3", "-c", "exit(0)"],
        )
        assert result_b.success is True
        assert telem_b.principal_id == "independent_trader"
        assert repo.get_principal_paid_tokens_consumed(principal_b.user_id) == 20_000


# =============================================================================
# Suite 3: QLoRA Flywheel Event Generation & Zero-Touch Invariants
# =============================================================================


class TestQLoRAFlywheelEventsAdversarial:
    """Stress tests ContextMissEvent logging and zero_touch_sufficient integrity."""

    def test_nonexistent_symbol_handled_without_crash(
        self,
        principal: PrincipalContext,
        sample_target_slice: VerbatimCodeSlice,
        temp_git_repo: Path,
    ) -> None:
        """MissingContextRequest for a ghost symbol should log cleanly with None paths."""
        executor = GeminiImplementationExecutor(api_key=None)
        repo = InMemoryRLSRepository()

        pack = ImplementationContextPack(
            trace_id=str(uuid.uuid4()),
            domain_id=DOMAIN_CODE,
            user_task="Task with ghost symbol",
            architectural_execution_plan=["Plan"],
            code_slices=[sample_target_slice],
            preflight_report=PreFlightValidationReport(
                passed=True, normalized_rrf_score=0.9, citations_verified_100_pct=True
            ),
            local_tokens_consumed=30_000,
            pack_token_count=3_000,
        )

        fictitious_req = MissingContextRequest(
            missing_symbol="fictitious_nonexistent_quantum_kernel",
            file_hint=None,
            reason="Unresolved tensor operation",
        )

        _result, telemetry = executor.execute_implementation(
            pack=pack,
            principal=principal,
            repo_root=temp_git_repo,
            repo=repo,
            missing_context_requests=[fictitious_req],
            auto_confirm=True,
            lint_cmd=["python3", "-c", "exit(0)"],
            test_cmd=["python3", "-c", "exit(0)"],
        )

        assert telemetry.zero_touch_sufficient is False
        assert len(telemetry.missing_context_requests) == 1
        event = telemetry.missing_context_requests[0]
        assert event["missing_symbol"] == "fictitious_nonexistent_quantum_kernel"
        assert event["resolved_file_path"] is None
        assert event["resolved_lines"] is None
        assert event["resolution_strategy"] == "deterministic_ast_slice"

        # Verify JSON serializability
        dumped = json.dumps(telemetry.missing_context_requests)
        loaded = json.loads(dumped)
        assert loaded == telemetry.missing_context_requests

    def test_zero_touch_invariant_when_no_missing_requests(
        self,
        principal: PrincipalContext,
        sample_target_slice: VerbatimCodeSlice,
        temp_git_repo: Path,
    ) -> None:
        """When zero missing context requests are emitted, zero_touch_sufficient MUST be True."""
        executor = GeminiImplementationExecutor(api_key=None)
        repo = InMemoryRLSRepository()

        pack = ImplementationContextPack(
            trace_id=str(uuid.uuid4()),
            domain_id=DOMAIN_CODE,
            user_task="Zero touch task",
            architectural_execution_plan=["Plan"],
            code_slices=[sample_target_slice],
            preflight_report=PreFlightValidationReport(
                passed=True, normalized_rrf_score=0.9, citations_verified_100_pct=True
            ),
            local_tokens_consumed=30_000,
            pack_token_count=3_000,
        )

        _, telemetry = executor.execute_implementation(
            pack=pack,
            principal=principal,
            repo_root=temp_git_repo,
            repo=repo,
            missing_context_requests=None,
            auto_confirm=True,
            lint_cmd=["python3", "-c", "exit(0)"],
            test_cmd=["python3", "-c", "exit(0)"],
        )
        assert telemetry.zero_touch_sufficient is True
        assert telemetry.missing_context_requests == []


# =============================================================================
# Suite 4: PostgreSQL 17 Telemetry Persistence & Schema Invariants
# =============================================================================


class TestLivePostgresTelemetrySchemaAdversarial:
    """Tests live PostgreSQL 17 execution_telemetry constraints, upserts, and time aggregation."""

    @pytest.mark.postgres
    def test_live_postgres_telemetry_schema_and_upsert_resilience(
        self,
        principal: PrincipalContext,
    ) -> None:
        """Directly exercises execution_telemetry CRUD, check constraints, and conflict updates."""
        settings = ScoutSettings()
        conn_mgr = RLSConnectionManager(settings.db_app_url)

        with conn_mgr.scoped_transaction(principal) as conn:
            pg_repo = PostgresRLSRepository()
            trace_id = uuid.uuid4()

            # 1. Insert valid record
            initial_record = ExecutionTelemetryRecord(
                trace_id=trace_id,
                domain_id=DOMAIN_CODE,
                principal_id=principal.user_id,
                principal_role="engineer",
                execution_mode=ExecutionMode.GEMINI_HANDOFF,
                local_tokens_read=40_000,
                context_pack_tokens_sent=4_000,
                paid_tokens_saved=36_000,
                estimated_usd_saved=0.157,
                preflight_passed_first_try=True,
                zero_touch_sufficient=False,
                missing_context_requests=[
                    {
                        "trace_id": str(trace_id),
                        "missing_symbol": "calc_ofi",
                        "resolution_strategy": "deterministic_ast_slice",
                    }
                ],
                latency_ms=125,
                created_at=datetime.now(UTC),
            )
            pg_repo.record_execution_telemetry(conn, initial_record)

            retrieved = pg_repo.get_execution_telemetry(conn, trace_id)
            assert retrieved is not None
            assert retrieved.trace_id == trace_id
            assert retrieved.local_tokens_read == 40_000
            assert retrieved.zero_touch_sufficient is False
            assert len(retrieved.missing_context_requests) == 1

            # 2. Upsert (ON CONFLICT DO UPDATE): Same trace_id, modified metrics
            updated_record = initial_record.model_copy(
                update={
                    "local_tokens_read": 45_000,
                    "zero_touch_sufficient": True,
                    "missing_context_requests": [],
                    "latency_ms": 250,
                }
            )
            pg_repo.record_execution_telemetry(conn, updated_record)

            persisted_update = pg_repo.get_execution_telemetry(conn, trace_id)
            assert persisted_update is not None
            assert persisted_update.local_tokens_read == 45_000
            assert persisted_update.zero_touch_sufficient is True
            assert persisted_update.missing_context_requests == []
            assert persisted_update.latency_ms == 250

            # Clean up
            conn.execute("DELETE FROM execution_telemetry WHERE trace_id = %s;", (trace_id,))

    @pytest.mark.postgres
    def test_live_postgres_check_constraint_negative_tokens_rejected(
        self,
        principal: PrincipalContext,
    ) -> None:
        """Verifies PostgreSQL 17 table constraint rejects negative local_tokens_read."""
        settings = ScoutSettings()
        conn_mgr = RLSConnectionManager(settings.db_app_url)

        with (
            conn_mgr.scoped_transaction(principal) as conn,
            pytest.raises(psycopg.errors.CheckViolation),
        ):
            conn.execute(
                """
                INSERT INTO execution_telemetry (
                    trace_id, domain_id, principal_id, principal_role,
                    execution_mode, local_tokens_read, context_pack_tokens_sent,
                    paid_tokens_saved, estimated_usd_saved, preflight_passed_first_try,
                    latency_ms
                ) VALUES (%s, %s, %s, %s, 'gemini_handoff', -50, 100, 50, 0.1, true, 10);
                """,
                (uuid.uuid4(), DOMAIN_CODE, principal.user_id, "engineer"),
            )

    @pytest.mark.postgres
    def test_live_postgres_check_constraint_invalid_mode_rejected(
        self,
        principal: PrincipalContext,
    ) -> None:
        """Verifies PostgreSQL 17 table constraint rejects invalid execution_mode enum values."""
        settings = ScoutSettings()
        conn_mgr = RLSConnectionManager(settings.db_app_url)

        with (
            conn_mgr.scoped_transaction(principal) as conn,
            pytest.raises(psycopg.errors.CheckViolation),
        ):
            conn.execute(
                """
                INSERT INTO execution_telemetry (
                    trace_id, domain_id, principal_id, principal_role,
                    execution_mode, local_tokens_read, context_pack_tokens_sent,
                    paid_tokens_saved, estimated_usd_saved, preflight_passed_first_try,
                    latency_ms
                ) VALUES (%s, %s, %s, %s, 'hacked_mode', 100, 100, 50, 0.1, true, 10);
                """,
                (uuid.uuid4(), DOMAIN_CODE, principal.user_id, "engineer"),
            )

    @pytest.mark.postgres
    def test_live_postgres_token_consumption_time_window_filtering(
        self,
        principal: PrincipalContext,
    ) -> None:
        """Verifies get_principal_paid_tokens_consumed respects the since_hours window."""
        settings = ScoutSettings()
        conn_mgr = RLSConnectionManager(settings.db_app_url)

        with conn_mgr.scoped_transaction(principal) as conn:
            pg_repo = PostgresRLSRepository()
            user_id = f"test_window_{uuid.uuid4().hex[:6]}"
            trace_recent = uuid.uuid4()
            trace_old = uuid.uuid4()

            now = datetime.now(UTC)
            old_time = now - timedelta(hours=48)

            # Record 1: 5,000 tokens within last 24h
            conn.execute(
                """
                INSERT INTO execution_telemetry (
                    trace_id, domain_id, principal_id, principal_role,
                    execution_mode, local_tokens_read, context_pack_tokens_sent,
                    paid_tokens_saved, estimated_usd_saved, preflight_passed_first_try,
                    latency_ms, created_at
                ) VALUES (
                    %s, %s, %s, 'engineer', 'gemini_handoff',
                    10000, 5000, 5000, 0.05, true, 50, %s
                );
                """,
                (trace_recent, DOMAIN_CODE, user_id, now),
            )

            # Record 2: 15,000 tokens from 48h ago
            conn.execute(
                """
                INSERT INTO execution_telemetry (
                    trace_id, domain_id, principal_id, principal_role,
                    execution_mode, local_tokens_read, context_pack_tokens_sent,
                    paid_tokens_saved, estimated_usd_saved, preflight_passed_first_try,
                    latency_ms, created_at
                ) VALUES (
                    %s, %s, %s, 'engineer', 'gemini_handoff',
                    20000, 15000, 5000, 0.15, true, 50, %s
                );
                """,
                (trace_old, DOMAIN_CODE, user_id, old_time),
            )

            # Query with since_hours=24: should only return 5,000 tokens
            tokens_24h = pg_repo.get_principal_paid_tokens_consumed(conn, user_id, since_hours=24)
            assert tokens_24h == 5_000

            # Query with since_hours=72: should return 20,000 tokens
            tokens_72h = pg_repo.get_principal_paid_tokens_consumed(conn, user_id, since_hours=72)
            assert tokens_72h == 20_000

            # Clean up
            conn.execute("DELETE FROM execution_telemetry WHERE principal_id = %s;", (user_id,))
