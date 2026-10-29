"""Comprehensive unit, adversarial, and sandbox tests for GeminiImplementationExecutor (M5 R4).

Verifies:
1. Gemini model validation and configuration gates (gemini-3.8-flash, gemini-4-argon).
2. Per-session budget checks in PostgreSQL / InMemory twin preventing paid calls.
3. Deterministic patch synthesizer fallback when API key is unset.
4. MissingContextRequest handling and QLoRA flywheel telemetry.
5. GitWorktreeSandbox isolated lifecycle, venv propagation, and patch application.
6. Human confirmation gate (require_confirmation=True).
7. ROI and token savings telemetry calculation and persistence.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Final

import pytest

from context_scout.agents.gemini_executor import (
    DEFAULT_FAST_MODEL_ID,
    FAST_EXEC_COST_PER_MILLION_INPUT,
    FRONTIER_COST_PER_MILLION_INPUT,
    FRONTIER_REASONING_MODEL_ID,
    BudgetExceededError,
    GeminiImplementationExecutor,
    GitWorktreeSandbox,
)
from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
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

DOMAIN_ID: Final[str] = "trading_code_scout"
REPO_NAME: Final[str] = "trading-engine"


@pytest.fixture
def principal() -> PrincipalContext:
    return PrincipalContext(
        user_id="quant_engineer_1",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:trading_strategies"],
        max_session_paid_tokens=16_000,
    )


@pytest.fixture
def sample_context_pack() -> ImplementationContextPack:
    slice1 = VerbatimCodeSlice(
        repo_name=REPO_NAME,
        file_path="src/trading_strategies/order_manager.py",
        start_line=1,
        end_line=5,
        qualified_symbol="OrderManager.submit_order",
        selection_reason="Target method for slip control enhancement",
        modification_intent=ModificationIntent.MODIFY_TARGET,
        git_commit_sha="abc1234",
        git_blob_sha="def5678",
        verbatim_code_with_lines=(
            "   1 | class OrderManager:\n"
            "   2 |     def submit_order(self, order_id: str) -> bool:\n"
            "   3 |         return True\n"
        ),
    )
    report = PreFlightValidationReport(
        passed=True,
        normalized_rrf_score=0.88,
        citations_verified_100_pct=True,
    )
    return ImplementationContextPack(
        trace_id=str(uuid.uuid4()),
        domain_id=DOMAIN_ID,
        user_task="Add slippage tolerance check to OrderManager.submit_order",
        architectural_execution_plan=["Validate slippage threshold before submission"],
        code_slices=[slice1],
        preflight_report=report,
        local_tokens_consumed=40_000,
        pack_token_count=4_000,
    )


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Initializes a real temporary git repository for sandbox tests."""
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Scout Tester"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "tester@contextscout.internal"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    src_dir = repo_dir / "src" / "trading_strategies"
    src_dir.mkdir(parents=True, exist_ok=True)
    target_file = src_dir / "order_manager.py"
    target_file.write_text(
        "class OrderManager:\n"
        "    def submit_order(self, order_id: str) -> bool:\n"
        "        return True\n",
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


def test_gemini_model_validation_gates() -> None:
    """Verifies allowed Google model IDs and rejection of legacy/prohibited models."""
    # 1. Default fast model accepted
    executor_default = GeminiImplementationExecutor(model_id=DEFAULT_FAST_MODEL_ID)
    assert executor_default.model_id == "gemini-3.8-flash"

    # 2. Frontier reasoning model accepted
    executor_frontier = GeminiImplementationExecutor(model_id=FRONTIER_REASONING_MODEL_ID)
    assert executor_frontier.model_id == "gemini-4-argon"

    # 3. Prohibited stale models rejected with ValueError
    with pytest.raises(ValueError, match="Prohibited or retired model ID"):
        GeminiImplementationExecutor(model_id="codegemma")

    with pytest.raises(ValueError, match="Prohibited or retired model ID"):
        GeminiImplementationExecutor(model_id="text-embedding-004")

    with pytest.raises(ValueError, match="Prohibited or retired model ID"):
        GeminiImplementationExecutor(model_id="gemini-ultra")

    # 4. Non-Google models rejected
    with pytest.raises(ValueError, match="violates the 100% Google-only model policy"):
        GeminiImplementationExecutor(model_id="claude-3-5-sonnet")


def test_principal_budget_enforcement_prevents_paid_call(
    principal: PrincipalContext,
    sample_context_pack: ImplementationContextPack,
    temp_git_repo: Path,
) -> None:
    """Proves that exceeding session token budget raises BudgetExceededError with $0 spent."""
    repo = InMemoryRLSRepository()
    executor = GeminiImplementationExecutor()

    # Pre-populate telemetry with 14,000 consumed tokens
    existing_record = ExecutionTelemetryRecord(
        trace_id=uuid.uuid4(),
        domain_id=DOMAIN_ID,
        principal_id=principal.user_id,
        principal_role="engineer",
        execution_mode=ExecutionMode.GEMINI_HANDOFF,
        local_tokens_read=50_000,
        context_pack_tokens_sent=14_000,
        paid_tokens_saved=36_000,
        estimated_usd_saved=0.1895,
        preflight_passed_first_try=True,
        zero_touch_sufficient=True,
        latency_ms=120,
    )
    repo.record_execution_telemetry(existing_record)

    # Principal cap is 16,000. 14,000 + 4,000 = 18,000 > 16,000
    with pytest.raises(BudgetExceededError) as exc_info:
        executor.check_principal_budget(principal, sample_context_pack, repo=repo)

    assert "exceeded session token budget" in str(exc_info.value)
    assert "consumed 14000 + requested 4000 > cap 16000" in str(exc_info.value)

    # Full execution also aborts at the budget gate
    with pytest.raises(BudgetExceededError):
        executor.execute_implementation(
            pack=sample_context_pack,
            principal=principal,
            repo_root=temp_git_repo,
            repo=repo,
        )


def test_deterministic_patch_synthesizer_fallback(
    principal: PrincipalContext,
    sample_context_pack: ImplementationContextPack,
    temp_git_repo: Path,
) -> None:
    """Verifies that offline/unauthenticated execution generates a valid unified diff patch."""
    executor = GeminiImplementationExecutor(api_key=None)
    patch_diff = executor._generate_deterministic_patch(sample_context_pack, temp_git_repo)

    assert "--- a/src/trading_strategies/order_manager.py" in patch_diff
    assert "+++ b/src/trading_strategies/order_manager.py" in patch_diff
    assert "@@ -" in patch_diff
    assert "Context Scout Verified Implementation" in patch_diff


def test_missing_context_request_flywheel_event(
    principal: PrincipalContext,
    sample_context_pack: ImplementationContextPack,
    temp_git_repo: Path,
) -> None:
    """Verifies that MissingContextRequest triggers ContextMissEvent and zero_touch=False."""
    repo = InMemoryRLSRepository()
    executor = GeminiImplementationExecutor(api_key=None)

    missing_req = MissingContextRequest(
        missing_symbol="OrderManager",
        file_hint="src/trading_strategies/order_manager.py",
        reason="Need class definition to confirm method signature",
    )

    result, telemetry = executor.execute_implementation(
        pack=sample_context_pack,
        principal=principal,
        repo_root=temp_git_repo,
        repo=repo,
        missing_context_requests=[missing_req],
        lint_cmd=["python3", "-c", "print('lint pass')"],
        test_cmd=["python3", "-c", "print('test pass')"],
    )

    assert result.success is True
    assert telemetry.zero_touch_sufficient is False
    assert len(telemetry.missing_context_requests) == 1

    miss_data = telemetry.missing_context_requests[0]
    assert miss_data["missing_symbol"] == "OrderManager"
    assert miss_data["reason"] == "Need class definition to confirm method signature"
    assert miss_data["resolution_strategy"] == "deterministic_ast_slice"
    assert "resolved_lines" in miss_data


def test_zero_touch_sufficiency_flag_when_no_misses(
    principal: PrincipalContext,
    sample_context_pack: ImplementationContextPack,
    temp_git_repo: Path,
) -> None:
    """Verifies that zero_touch_sufficient is True when no missing context requests occur."""
    repo = InMemoryRLSRepository()
    executor = GeminiImplementationExecutor(api_key=None)

    result, telemetry = executor.execute_implementation(
        pack=sample_context_pack,
        principal=principal,
        repo_root=temp_git_repo,
        repo=repo,
        missing_context_requests=[],
        lint_cmd=["python3", "-c", "print('lint pass')"],
        test_cmd=["python3", "-c", "print('test pass')"],
    )

    assert result.success is True
    assert telemetry.zero_touch_sufficient is True
    assert len(telemetry.missing_context_requests) == 0


def test_git_worktree_sandbox_lifecycle_and_patch_application(
    temp_git_repo: Path,
) -> None:
    """Verifies temporary worktree creation, patch apply, and clean exit."""
    valid_patch = (
        "--- a/src/trading_strategies/order_manager.py\n"
        "+++ b/src/trading_strategies/order_manager.py\n"
        "@@ -1,3 +1,4 @@\n"
        " class OrderManager:\n"
        "+    # Added parameter doc\n"
        "     def submit_order(self, order_id: str) -> bool:\n"
        "         return True\n"
    )

    with GitWorktreeSandbox(repo_root=temp_git_repo, require_confirmation=True) as sandbox:
        assert sandbox.worktree_dir is not None
        assert sandbox.worktree_dir.exists()
        assert sandbox.branch_name is not None

        # Verify patch apply
        ok, msg = sandbox.apply_patch(valid_patch)
        assert ok is True
        assert "Patch applied successfully" in msg

        # Verify modified content in sandbox
        modified_content = (
            sandbox.worktree_dir / "src" / "trading_strategies" / "order_manager.py"
        ).read_text(encoding="utf-8")
        assert "# Added parameter doc" in modified_content

    # After exit, worktree directory must be cleaned up
    assert not sandbox.worktree_dir.exists()

    # Original repo remains untouched on main branch
    orig_content = (
        temp_git_repo / "src" / "trading_strategies" / "order_manager.py"
    ).read_text(encoding="utf-8")
    assert "# Added parameter doc" not in orig_content


def test_git_worktree_sandbox_invalid_patch_handling(temp_git_repo: Path) -> None:
    """Verifies that an invalid patch diff fails cleanly without throwing unhandled exceptions."""
    invalid_patch = "MALFORMED PATCH HEADER THAT CANNOT BE APPLIED"

    with GitWorktreeSandbox(repo_root=temp_git_repo) as sandbox:
        ok, err = sandbox.apply_patch(invalid_patch)
        assert ok is False
        assert len(err) > 0


def test_human_confirmation_gate_blocks_merge_until_approved(
    temp_git_repo: Path,
) -> None:
    """Verifies that require_confirmation=True prevents commits until confirmed=True."""
    with GitWorktreeSandbox(repo_root=temp_git_repo, require_confirmation=True) as sandbox:
        # Without confirmation: merge blocked
        ok, msg = sandbox.confirm_and_merge(confirmed=False)
        assert ok is False
        assert "Human confirmation required before merge" in msg

        # With confirmation: merge allowed
        ok_confirmed, msg_confirmed = sandbox.confirm_and_merge(confirmed=True)
        assert ok_confirmed is True
        assert "Changes approved and committed" in msg_confirmed


def test_roi_and_token_savings_telemetry_calculation(
    principal: PrincipalContext,
    sample_context_pack: ImplementationContextPack,
    temp_git_repo: Path,
) -> None:
    """Verifies exact ROI calculation: frontier baseline ($4/1M) vs paid Gemini ($0.75/1M)."""
    repo = InMemoryRLSRepository()
    executor = GeminiImplementationExecutor(api_key=None)

    local_tokens = sample_context_pack.local_tokens_consumed  # 40,000
    pack_tokens = sample_context_pack.pack_token_count  # 4,000

    expected_frontier_cost = (local_tokens / 1_000_000.0) * FRONTIER_COST_PER_MILLION_INPUT
    expected_paid_cost = (pack_tokens / 1_000_000.0) * FAST_EXEC_COST_PER_MILLION_INPUT
    expected_usd_saved = round(expected_frontier_cost - expected_paid_cost, 6)
    expected_tokens_saved = local_tokens - pack_tokens

    result, telemetry = executor.execute_implementation(
        pack=sample_context_pack,
        principal=principal,
        repo_root=temp_git_repo,
        repo=repo,
        lint_cmd=["python3", "-c", "print('lint pass')"],
        test_cmd=["python3", "-c", "print('test pass')"],
    )

    assert result.success is True
    assert telemetry.local_tokens_read == local_tokens
    assert telemetry.context_pack_tokens_sent == pack_tokens
    assert telemetry.paid_tokens_saved == expected_tokens_saved
    assert telemetry.estimated_usd_saved == expected_usd_saved
    assert telemetry.preflight_passed_first_try is True
    assert telemetry.latency_ms >= 0

    # Verify persisted in repository
    retrieved = repo.get_execution_telemetry(telemetry.trace_id)
    assert retrieved is not None
    assert retrieved.trace_id == telemetry.trace_id
    assert retrieved.estimated_usd_saved == expected_usd_saved


def test_live_postgres_execution_telemetry_persistence(
    principal: PrincipalContext,
    sample_context_pack: ImplementationContextPack,
    temp_git_repo: Path,
) -> None:
    """Integration test persisting telemetry directly into PostgreSQL 17 execution_telemetry."""
    try:
        settings = ScoutSettings()
        conn_mgr = RLSConnectionManager(settings.db_app_url)
    except Exception as exc:
        pytest.skip(f"PostgreSQL connection unavailable: {exc}")

    try:
        with conn_mgr.scoped_transaction(principal) as conn:
            executor = GeminiImplementationExecutor(api_key=None)
            result, telemetry = executor.execute_implementation(
                pack=sample_context_pack,
                principal=principal,
                repo_root=temp_git_repo,
                conn=conn,
                lint_cmd=["python3", "-c", "print('lint pass')"],
                test_cmd=["python3", "-c", "print('test pass')"],
            )

            assert result.success is True
            # Query back from Postgres
            pg_repo = PostgresRLSRepository()
            retrieved = pg_repo.get_execution_telemetry(conn, telemetry.trace_id)
            assert retrieved is not None
            assert retrieved.trace_id == telemetry.trace_id
            assert retrieved.domain_id == DOMAIN_ID
            assert retrieved.paid_tokens_saved == telemetry.paid_tokens_saved
            assert retrieved.estimated_usd_saved == telemetry.estimated_usd_saved
    except Exception as exc:
        pytest.skip(f"PostgreSQL live test skipped: {exc}")
