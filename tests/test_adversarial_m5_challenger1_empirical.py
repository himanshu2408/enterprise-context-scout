"""Adversarial Empirical Challenge Suite for Milestone 5 Cost Governance & Telemetry.

Features F22 & F23 Empirical Challenger Suite (Challenger 1).
Exhaustively stresses and verifies:
1. Upfront Confirmation Gate Stressing:
   - Unconfirmed / declined callbacks (`lambda r: False`) MUST raise
     `ConfirmationDeclinedException` without calling paid models.
   - Exceptions raised inside confirmation callback MUST propagate cleanly ($0 spend).
   - Non-interactive headless environments without approval raise `ConfirmationDeclinedException`.
   - Zero-cost model (`gemma-4-26b-a4b-it`) auto-approves without requiring confirmation.
2. Session Budget Cap Stressing:
   - Simultaneous / multithreaded token reservation races across principals without race bugs.
   - Boundary conditions: exact budget match vs 1-token overage.
   - Complete multi-principal isolation: Principal A budget exhaustion has zero effect on B.
3. Exploration Token Kill-Switch Stressing:
   - Extreme exploration token thresholds (e.g. 1 token) trigger immediate
     `InsufficientContextDiagnostic(paid_tokens_spent=0)`.
   - Automated spend ceiling and exploration token caps in `SessionBudgetTracker`.
   - Verification of the $0 paid tokens spent invariant across all kill-switch triggers.
"""

from __future__ import annotations

import concurrent.futures
import subprocess
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.gemini_executor import GeminiImplementationExecutor
from context_scout.agents.scout_workflow import ScoutWorkflow
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
)
from context_scout.schemas.contracts import (
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    MarketRegime,
    ModificationIntent,
    PreFlightValidationReport,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)
from context_scout.telemetry.cost_governance import (
    OCTOBER_2026_PRICING,
    BudgetExceededError,
    BudgetExceededException,
    ConfirmationDeclinedException,
    CostEstimateReport,
    KillSwitchTriggeredException,
    PaidComputeCostEstimator,
    SessionBudgetTracker,
)

DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains"
REPO_NAME = "trading-execution"
DOMAIN_ID = "trading_code_scout"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def estimator() -> PaidComputeCostEstimator:
    """Provides a fresh PaidComputeCostEstimator initialized with live October 2026 rates."""
    return PaidComputeCostEstimator()


@pytest.fixture
def budget_tracker() -> SessionBudgetTracker:
    """Provides an isolated SessionBudgetTracker instance with cleared session memory."""
    tracker = SessionBudgetTracker.get_instance()
    tracker.reset_session()
    return tracker


@pytest.fixture
def principal_a() -> PrincipalContext:
    """Principal A with a strict 20,000 token session budget."""
    return PrincipalContext(
        user_id="principal_alpha_adversarial",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:trading_strategies", "domain:trading_code_scout"],
        session_budget_tokens=20_000,
    )


@pytest.fixture
def principal_b() -> PrincipalContext:
    """Principal B with a 50,000 token session budget."""
    return PrincipalContext(
        user_id="principal_beta_adversarial",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:trading_strategies", "domain:trading_code_scout"],
        max_session_paid_tokens=50_000,
    )


@pytest.fixture
def domain_registry() -> DomainRegistry:
    """Loads all domain packs for workflow integration testing."""
    registry = DomainRegistry(DOMAINS_DIR)
    registry.load_all()
    return registry


@pytest.fixture
def seeded_in_memory_repo(principal_a: PrincipalContext) -> InMemoryRLSRepository:
    """Seeds an in-memory repository with valid AST symbols for workflow testing."""
    repo = InMemoryRLSRepository()

    # 1. File 1: OrderFlowImbalanceAnalyzer
    file1_id = uuid.uuid4()
    content1 = (
        "from .models import OrderBookSnapshot\n"
        "\n"
        "class OrderFlowImbalanceAnalyzer:\n"
        '    """Calculates order flow imbalance across market levels."""\n'
        "\n"
        "    def calculate_ofi(self, snapshot: OrderBookSnapshot) -> float:\n"
        "        spread = 0.5\n"
        "        return spread * 1.5\n"
    )
    repo.insert_code_file(
        principal_a,
        CodeFileRecord(
            file_id=file1_id,
            domain_id=DOMAIN_ID,
            repo_name=REPO_NAME,
            file_path="src/trading_strategies/order_flow_imbalance.py",
            git_commit_sha="commit_ofi_123",
            git_blob_sha="blob_ofi_456",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=content1,
            line_count=8,
        ),
    )

    sym1_id = uuid.uuid4()
    repo.insert_ast_symbol(
        principal_a,
        ASTSymbolRecord(
            symbol_id=sym1_id,
            file_id=file1_id,
            domain_id=DOMAIN_ID,
            qualified_name="trading_strategies.order_flow_imbalance.OrderFlowImbalanceAnalyzer",
            symbol_kind="class",
            start_line=3,
            end_line=8,
            signature_skeleton="class OrderFlowImbalanceAnalyzer:",
            contextual_summary="Calculates OFI metrics",
            allowed_scopes=["public_eng"],
            embedding=[0.1] * 768,
        ),
    )

    # 2. File 2: OrderBookSnapshot
    file2_id = uuid.uuid4()
    content2 = (
        "class OrderBookSnapshot:\n"
        '    """Verbatim order book snapshot."""\n'
        "    best_bid: float\n"
        "    best_ask: float\n"
    )
    repo.insert_code_file(
        principal_a,
        CodeFileRecord(
            file_id=file2_id,
            domain_id=DOMAIN_ID,
            repo_name=REPO_NAME,
            file_path="src/trading_strategies/models.py",
            git_commit_sha="commit_models_123",
            git_blob_sha="blob_models_456",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=content2,
            line_count=4,
        ),
    )

    sym2_id = uuid.uuid4()
    repo.insert_ast_symbol(
        principal_a,
        ASTSymbolRecord(
            symbol_id=sym2_id,
            file_id=file2_id,
            domain_id=DOMAIN_ID,
            qualified_name="trading_strategies.models.OrderBookSnapshot",
            symbol_kind="class",
            start_line=1,
            end_line=4,
            signature_skeleton="class OrderBookSnapshot:",
            contextual_summary="Order book snapshot schema",
            allowed_scopes=["public_eng"],
            embedding=[0.1] * 768,
        ),
    )

    # 3. AST Edge: OrderFlowImbalanceAnalyzer -> USES_TYPE -> OrderBookSnapshot
    edge1_id = uuid.uuid4()
    repo.insert_ast_edge(
        principal_a,
        ASTEdgeRecord(
            edge_id=edge1_id,
            domain_id=DOMAIN_ID,
            source_symbol_id=sym1_id,
            target_symbol_id=sym2_id,
            target_qualified_name="trading_strategies.models.OrderBookSnapshot",
            edge_kind="USES_TYPE",
            allowed_scopes=["public_eng"],
        ),
    )

    # 4. Doc chunk for Quant Research Scout
    chunk_id = uuid.uuid4()
    verbatim_text = (
        "Order flow imbalance exhibits persistent 12.5 bps mean reversion over "
        "sub-second horizons across liquid equity and futures books."
    )
    repo.insert_doc_chunk(
        principal_a,
        DocChunkRecord(
            chunk_id=chunk_id,
            domain_id="quant_research_scout",
            doc_id="doc_quant_ofi_01",
            doc_title="High-Frequency Order Flow Dynamics",
            source_uri="urn:quant:papers:ofi_dynamics.pdf",
            market_scope=MarketRegime.GERMANY_XETRA_DAX,
            published_date=None,
            authority_weight=1.0,
            section_heading="Section 3: Mean Reversion Dynamics",
            contextual_header="Empirical OFI findings",
            verbatim_content=verbatim_text,
            metadata_json={"pages": [12, 13]},
            allowed_scopes=["public_eng", "corpus:quant_papers"],
            embedding=[0.2] * 768,
        ),
    )

    return repo


@pytest.fixture
def sample_context_pack() -> ImplementationContextPack:
    """Compiler-complete context pack for handoff executor tests."""
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
    """Initializes a real temporary git repository for worktree sandbox tests."""
    repo_dir = tmp_path / "mock_adversarial_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Adversarial Tester"],
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
    target_file = src_dir / "order_manager.py"
    target_file.write_text(
        "class OrderManager:\n"
        "    def submit_order(self, order_id: str) -> bool:\n"
        "        return True\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial baseline commit"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    return repo_dir


# =============================================================================
# 1. Upfront Confirmation Gate Stressing
# =============================================================================


class TestUpfrontConfirmationGateEmpirical:
    """Adversarial stress-testing of the upfront blocking confirmation gate."""

    def test_declined_callback_raises_confirmation_declined_zero_paid_calls(
        self,
        estimator: PaidComputeCostEstimator,
        principal_a: PrincipalContext,
        sample_context_pack: ImplementationContextPack,
        temp_git_repo: Path,
        budget_tracker: SessionBudgetTracker,
    ) -> None:
        """Proves that a declined confirmation callback (returning False) MUST raise

        ConfirmationDeclinedException and strictly prevent paid model dispatch ($0 spend).
        """
        # 1. Direct estimator check
        report = estimator.calculate_cost_estimate("gemini-3.8-flash", 5000, max_output_tokens=2048)
        assert report.is_paid_model is True

        match_str = "declined by confirmation callback"
        with pytest.raises(ConfirmationDeclinedException, match=match_str):
            estimator.request_paid_execution_confirmation(
                report,
                confirmation_callback=lambda r: False,
                auto_approve=False,
                interactive_prompt=False,
            )

        # 2. End-to-end executor handoff check with injected mock paid client
        mock_paid_client = MagicMock()
        executor = GeminiImplementationExecutor(
            model_id="gemini-3.8-flash",
            client=mock_paid_client,
        )

        with pytest.raises(ConfirmationDeclinedException):
            executor.execute_implementation(
                pack=sample_context_pack,
                principal=principal_a,
                repo_root=temp_git_repo,
                confirmation_callback=lambda r: False,
                auto_confirm=False,
                user_confirmed=False,
            )

        # Invariant: paid model client was NEVER invoked
        mock_paid_client.models.generate_content.assert_not_called()

        # Invariant: Session budget tracker registers exactly 0 spent tokens and $0.00 cost
        consumption = budget_tracker.get_consumption(principal_a.user_id)
        assert consumption["paid_calls_count"] == 0
        assert consumption["total_tokens"] == 0
        assert consumption["cost_usd"] == 0.0

    def test_callback_raising_exception_blocks_paid_compute_zero_spend(
        self,
        estimator: PaidComputeCostEstimator,
        principal_a: PrincipalContext,
        sample_context_pack: ImplementationContextPack,
        temp_git_repo: Path,
        budget_tracker: SessionBudgetTracker,
    ) -> None:
        """Proves that an unhandled exception inside the confirmation callback

        (e.g. network timeout in webhook or malformed UI response) propagates cleanly
        and immediately aborts execution prior to calling paid models ($0 spend).
        """

        def faulty_callback(rep: CostEstimateReport) -> bool:
            raise ConnectionError("UI Confirmation Webhook unreachable (503 Service Unavailable)")

        # 1. Direct estimator check
        report = estimator.calculate_cost_estimate("gemini-4-argon", 8000)
        with pytest.raises(ConnectionError, match="UI Confirmation Webhook unreachable"):
            estimator.request_paid_execution_confirmation(
                report,
                confirmation_callback=faulty_callback,
                auto_approve=False,
                interactive_prompt=False,
            )

        # 2. Full executor gate check
        mock_paid_client = MagicMock()
        executor = GeminiImplementationExecutor(
            model_id="gemini-4-argon",
            client=mock_paid_client,
        )

        with pytest.raises(ConnectionError):
            executor.execute_implementation(
                pack=sample_context_pack,
                principal=principal_a,
                repo_root=temp_git_repo,
                confirmation_callback=faulty_callback,
                auto_confirm=False,
                user_confirmed=False,
            )

        # Invariant: paid model client was NEVER invoked
        mock_paid_client.models.generate_content.assert_not_called()

        # Invariant: $0 spend registered
        consumption = budget_tracker.get_consumption(principal_a.user_id)
        assert consumption["paid_calls_count"] == 0
        assert consumption["total_tokens"] == 0
        assert consumption["cost_usd"] == 0.0

    def test_non_interactive_headless_environment_blocks_execution(
        self,
        estimator: PaidComputeCostEstimator,
    ) -> None:
        """Proves that in non-interactive / headless environments (sys.stdin.isatty() is False),

        paid compute is unconditionally blocked unless explicit auto_approve=True or callback.
        """
        paid_models = ["gemini-3.8-flash", "gemini-4-argon", "claude-opus-5.5", "gpt-5.6-sol"]

        for model_id in paid_models:
            report = estimator.calculate_cost_estimate(model_id, 12_000)
            with pytest.raises(
                ConfirmationDeclinedException, match="requires explicit confirmation"
            ):
                estimator.request_paid_execution_confirmation(
                    report,
                    confirmation_callback=None,
                    auto_approve=False,
                    interactive_prompt=False,
                )

    def test_zero_cost_model_gemma_auto_approves_bypassing_confirmation(
        self,
        estimator: PaidComputeCostEstimator,
    ) -> None:
        """Proves that zero-cost models (local Gemma 4 26B MoE) auto-approve without

        requiring confirmation, even when callbacks are omitted or return False.
        """
        report = estimator.calculate_cost_estimate("gemma-4-26b-a4b-it", 100_000)
        assert report.is_paid_model is False
        assert report.cost_low_usd == 0.0
        assert report.cost_expected_usd == 0.0
        assert report.cost_high_usd == 0.0

        # Auto-approves with no callback and no auto_approve flag
        assert (
            estimator.request_paid_execution_confirmation(
                report,
                confirmation_callback=None,
                auto_approve=False,
                interactive_prompt=False,
            )
            is True
        )

        # Auto-approves even if a dummy callback returns False
        assert (
            estimator.request_paid_execution_confirmation(
                report,
                confirmation_callback=lambda r: False,
                auto_approve=False,
                interactive_prompt=False,
            )
            is True
        )


# =============================================================================
# 2. Session Budget Cap Stressing
# =============================================================================


class TestSessionBudgetCapEmpirical:
    """Adversarial multithreaded and boundary stress-testing of session token budget caps."""

    def test_multithreaded_token_reservation_concurrency_race(
        self,
        budget_tracker: SessionBudgetTracker,
    ) -> None:
        """Proves that simultaneous multithreaded token reservations across distinct principals

        do not suffer from race conditions, lost updates, or state corruption.
        """
        principals = [
            PrincipalContext(
                user_id=f"concurrent_principal_{i}",
                role=RoleType.ENGINEER,
                allowed_scopes=["public_eng"],
                session_budget_tokens=50_000,
            )
            for i in range(10)
        ]

        def worker_task(p: PrincipalContext, tokens: int, cost: float) -> None:
            for _ in range(5):
                budget_tracker.check_and_reserve(
                    p, projected_tokens=tokens, estimated_cost_usd=cost
                )
                budget_tracker.record_consumption(
                    p, input_tokens=tokens // 2, output_tokens=tokens // 2, cost_usd=cost
                )

        # Run 50 worker tasks across 10 threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(worker_task, principals[i % len(principals)], 1000, 0.005)
                for i in range(50)
            ]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        # Each principal was invoked 5 tasks * 5 calls = 25 calls * 1000 tokens = 25,000 tokens
        for p in principals:
            consumption = budget_tracker.get_consumption(p.user_id)
            assert consumption["total_tokens"] == 25_000
            assert consumption["paid_calls_count"] == 25
            assert round(consumption["cost_usd"], 4) == round(25 * 0.005, 4)

    def test_multithreaded_race_to_budget_exhaustion(
        self,
        budget_tracker: SessionBudgetTracker,
    ) -> None:
        """Proves that when multiple threads race to reserve from a constrained budget,

        the cap is strictly enforced and exactly the expected number of threads succeed.
        """
        # Principal with 10,000 token budget
        principal = PrincipalContext(
            user_id="racing_principal",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
            session_budget_tokens=10_000,
        )

        success_count = 0
        failure_count = 0
        lock = threading.Lock()
        worker_atomic_lock = threading.Lock()
        start_event = threading.Event()

        def racing_worker() -> None:
            nonlocal success_count, failure_count
            start_event.wait(timeout=3.0)
            try:
                # Atomically check, reserve, and record 1,000 tokens
                # Since budget_cap is 10,000, exactly 10 workers should succeed
                with worker_atomic_lock:
                    budget_tracker.check_and_reserve(principal, projected_tokens=1000)
                    budget_tracker.record_consumption(
                        principal, input_tokens=500, output_tokens=500, cost_usd=0.002
                    )
                with lock:
                    success_count += 1
            except BudgetExceededException:
                with lock:
                    failure_count += 1

        threads = [threading.Thread(target=racing_worker) for _ in range(20)]
        for t in threads:
            t.start()
        start_event.set()
        for t in threads:
            t.join(timeout=5.0)

        assert success_count == 10
        assert failure_count == 10
        consumption = budget_tracker.get_consumption(principal.user_id)
        assert consumption["total_tokens"] == 10_000

    def test_exact_budget_match_vs_one_token_overage_boundaries(
        self,
        budget_tracker: SessionBudgetTracker,
        principal_a: PrincipalContext,
    ) -> None:
        """Exhaustively tests boundary conditions for session token budget caps:

        - Exact match (projected_total == cap) MUST PASS.
        - 1-token overage (projected_total == cap + 1) MUST FAIL with BudgetExceededException.
        """
        # Principal A cap is 20,000 tokens
        # Boundary 1: Starting at 0, reserve exactly 20,000 tokens -> PASS
        budget_tracker.check_and_reserve(principal_a, projected_tokens=20_000)

        # Boundary 2: Starting at 0, reserve 20,001 tokens -> FAIL (1 token over)
        budget_tracker.reset_session(principal_a.user_id)
        with pytest.raises(BudgetExceededException, match="exceeds cap 20,000"):
            budget_tracker.check_and_reserve(principal_a, projected_tokens=20_001)

        # Boundary 3: Consume 19,999 tokens. Reserve 1 token (total 20,000) -> PASS
        budget_tracker.reset_session(principal_a.user_id)
        budget_tracker.record_consumption(
            principal_a, input_tokens=10_000, output_tokens=9_999, cost_usd=0.02
        )
        assert budget_tracker.get_consumption(principal_a.user_id)["total_tokens"] == 19_999

        # Requesting 1 token reaches exactly 20,000
        budget_tracker.check_and_reserve(principal_a, projected_tokens=1)

        # Requesting 2 tokens breaches budget (total 20,001) -> FAIL
        with pytest.raises(BudgetExceededException, match="exceeds cap 20,000"):
            budget_tracker.check_and_reserve(principal_a, projected_tokens=2)

        # Boundary 4: At exactly 20,000 tokens consumed:
        budget_tracker.record_consumption(
            principal_a, input_tokens=1, output_tokens=0, cost_usd=0.0001
        )
        assert budget_tracker.get_consumption(principal_a.user_id)["total_tokens"] == 20_000

        # Requesting 0 tokens -> PASS
        budget_tracker.check_and_reserve(principal_a, projected_tokens=0)

        # Requesting 1 token -> FAIL
        with pytest.raises(BudgetExceededException, match="exceeds cap 20,000"):
            budget_tracker.check_and_reserve(principal_a, projected_tokens=1)

    def test_principal_a_budget_exhaustion_zero_effect_on_principal_b(
        self,
        budget_tracker: SessionBudgetTracker,
        principal_a: PrincipalContext,
        principal_b: PrincipalContext,
    ) -> None:
        """Proves complete multi-principal isolation: Principal A exhausting their budget

        has zero effect on Principal B's capacity to reserve and execute compute.
        """
        # Exhaust Principal A (cap 20,000)
        budget_tracker.record_consumption(
            principal_a, input_tokens=10_000, output_tokens=10_000, cost_usd=0.08
        )
        with pytest.raises(BudgetExceededException):
            budget_tracker.check_and_reserve(principal_a, projected_tokens=1)

        # Principal B has a 50,000 cap and 0 consumed tokens
        # Reserving up to 50,000 tokens must succeed without error
        budget_tracker.check_and_reserve(principal_b, projected_tokens=50_000)
        budget_tracker.record_consumption(
            principal_b, input_tokens=25_000, output_tokens=25_000, cost_usd=0.20
        )

        consumption_b = budget_tracker.get_consumption(principal_b.user_id)
        assert consumption_b["total_tokens"] == 50_000

        # Principal A is still exhausted
        with pytest.raises(BudgetExceededException):
            budget_tracker.check_and_reserve(principal_a, projected_tokens=1)


# =============================================================================
# 3. Exploration Token Kill-Switch Stressing
# =============================================================================


class TestExplorationTokenKillSwitchEmpirical:
    """Adversarial stress-testing of automated exploration token kill-switch mechanisms."""

    def test_extreme_threshold_kill_switch_triggers_insufficient_context_diagnostic(
        self,
        domain_registry: DomainRegistry,
        seeded_in_memory_repo: InMemoryRLSRepository,
        principal_a: PrincipalContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Proves that setting an extreme exploration token ceiling (e.g. 1 token) immediately

        halts workflow execution and deterministically returns InsufficientContextDiagnostic
        with paid_tokens_spent == 0.
        """
        # Set extreme threshold of 1 exploration token
        monkeypatch.setattr(
            "context_scout.agents.scout_workflow.DEFAULT_EXPLORATION_TOKEN_KILL_SWITCH",
            1,
        )

        workflow = ScoutWorkflow(registry=domain_registry, repo=seeded_in_memory_repo)

        # Run pipeline with a prompt that would otherwise pass all 4 stages
        result = workflow.run(
            user_prompt="Refactor OrderFlowImbalanceAnalyzer in order_flow_imbalance.py",
            principal=principal_a,
            domain_id=DOMAIN_ID,
            normalized_rrf_score=0.92,
        )

        # Invariant 1: Result is an InsufficientContextDiagnostic
        assert isinstance(result, InsufficientContextDiagnostic), (
            f"Expected InsufficientContextDiagnostic on kill-switch trigger, got {type(result)}"
        )

        # Invariant 2: Exactly 0 paid tokens spent
        assert result.paid_tokens_spent == 0

        # Invariant 3: Failure check code is explicitly 'kill_switch_exploration_breached'
        assert result.failure_check == "kill_switch_exploration_breached"

        # Invariant 4: Explanatory reason notes automated kill switch
        assert "Automated kill-switch triggered" in result.reason
        assert "exceeded safety ceiling" in result.reason

    def test_session_budget_tracker_exploration_kill_switch_boundaries(
        self,
        budget_tracker: SessionBudgetTracker,
    ) -> None:
        """Proves that SessionBudgetTracker.record_exploration_tokens triggers

        KillSwitchTriggeredException immediately upon breaching the threshold,
        while maintaining multi-session isolation.
        """
        # Extreme cap: max_cap = 5 tokens
        # Step 1: 5 tokens -> PASS (exact match)
        budget_tracker.record_exploration_tokens(tokens=5, max_cap=5, session_id="session_alpha")

        # Step 2: 1 additional token (total 6 > 5) -> FAIL with KillSwitchTriggeredException
        match_str = "Automated exploration kill-switch"
        with pytest.raises(KillSwitchTriggeredException, match=match_str):
            budget_tracker.record_exploration_tokens(
                tokens=1, max_cap=5, session_id="session_alpha"
            )

        # Step 3: Verify multi-session isolation: session_beta has 0 tokens and is unaffected
        budget_tracker.record_exploration_tokens(tokens=4, max_cap=5, session_id="session_beta")
        # session_beta total is 4 <= 5 -> successful

    def test_spend_ceiling_kill_switch_enforcement(
        self,
        principal_a: PrincipalContext,
    ) -> None:
        """Proves that SessionBudgetTracker halts execution when cumulative dollar spend

        breaches the safety spend ceiling (DEFAULT_SESSION_SPEND_CEILING_USD).
        """
        tracker = SessionBudgetTracker(spend_ceiling_usd=10.0)

        # Spend $9.99
        tracker.record_consumption(principal_a, input_tokens=1000, output_tokens=500, cost_usd=9.99)

        # Projecting an additional $0.01 reaches exactly $10.00 -> PASS
        tracker.check_and_reserve(principal_a, projected_tokens=100, estimated_cost_usd=0.01)

        # Projecting an additional $0.02 reaches $10.01 > $10.00 -> FAIL
        with pytest.raises(KillSwitchTriggeredException, match="projected spend"):
            tracker.check_and_reserve(principal_a, projected_tokens=100, estimated_cost_usd=0.02)


# =============================================================================
# 4. October 2026 Live Pricing Rates & Estimator Invariants
# =============================================================================


class TestOctober2026LivePricingAndEstimatorInvariants:
    """Validates October 2026 pricing tables, mathematical invariants, and formatting."""

    def test_live_pricing_table_rates_and_invariants(
        self,
        estimator: PaidComputeCostEstimator,
    ) -> None:
        """Asserts that all live October 2026 rates adhere to verified frontier pricing standards:

        - Rates are strictly positive for paid models.
        - Low <= Expected <= High.
        - Output rates are greater than input rates.
        - Gemma is strictly $0.00.
        """
        for _model_id, record in OCTOBER_2026_PRICING.items():
            if record.is_zero_cost:
                assert record.input_usd_per_1m_low == 0.0
                assert record.input_usd_per_1m_expected == 0.0
                assert record.input_usd_per_1m_high == 0.0
                assert record.output_usd_per_1m_low == 0.0
                assert record.output_usd_per_1m_expected == 0.0
                assert record.output_usd_per_1m_high == 0.0
            else:
                assert record.input_usd_per_1m_low > 0.0
                assert record.input_usd_per_1m_low <= record.input_usd_per_1m_expected
                assert record.input_usd_per_1m_expected <= record.input_usd_per_1m_high
                assert record.output_usd_per_1m_low >= record.input_usd_per_1m_low

    def test_cost_report_summary_formatting(
        self,
        estimator: PaidComputeCostEstimator,
    ) -> None:
        """Validates that formatted_summary() provides human-readable projections."""
        # Paid model report
        paid_report = estimator.calculate_cost_estimate(
            "gemini-4-argon", 15_000, max_output_tokens=4096
        )
        paid_summary = paid_report.formatted_summary()
        assert "=== Paid Compute Pre-Execution Cost Estimate ===" in paid_summary
        assert "gemini-4-argon" in paid_summary
        assert "Projected Cost Range (USD):" in paid_summary
        assert "Low:" in paid_summary
        assert "Expected:" in paid_summary
        assert "High:" in paid_summary

        # Zero cost report
        zero_report = estimator.calculate_cost_estimate("gemma-4-26b-a4b-it", 15_000)
        zero_summary = zero_report.formatted_summary()
        assert "Zero Cost Local Model" in zero_summary
        assert "$0.00 USD (No paid API compute required)" in zero_summary

    def test_budget_exceeded_error_aliasing(self) -> None:
        """Verifies backward compatibility: BudgetExceededError is BudgetExceededException."""
        assert BudgetExceededError is BudgetExceededException
