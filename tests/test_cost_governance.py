"""Unit, boundary, and regression tests for Feature F22 & F23 Cost Governance.

Verifies:
1. Exact live October 2026 model pricing rates.
2. Low/Expected/High cost range calculations and formatting.
3. Heuristic token estimation logic.
4. Upfront blocking confirmation gate ($0 spend on rejection).
5. Upfront confirmation passing on explicit auto_approve or callback approval.
6. Session budget limit enforcement on principal token budgets.
7. Automated exploration kill-switch on runaway token traversal.
8. Automated spend ceiling kill-switch.
9. Multi-principal budget tracker isolation.
10. PrincipalContext dual-field compatibility (session_budget_tokens and max_session_paid_tokens).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from context_scout.schemas.contracts import (
    PrincipalContext,
    RoleType,
)
from context_scout.telemetry.cost_governance import (
    BudgetExceededError,
    BudgetExceededException,
    ConfirmationDeclinedException,
    KillSwitchTriggeredException,
    PaidComputeCostEstimator,
    SessionBudgetTracker,
)


@pytest.fixture
def estimator() -> PaidComputeCostEstimator:
    return PaidComputeCostEstimator()


@pytest.fixture
def budget_tracker() -> SessionBudgetTracker:
    tracker = SessionBudgetTracker.get_instance()
    tracker.reset_session()
    return tracker


@pytest.fixture
def principal_a() -> PrincipalContext:
    return PrincipalContext(
        user_id="principal_alpha",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:code_architecture_scout"],
        session_budget_tokens=20_000,
    )


@pytest.fixture
def principal_b() -> PrincipalContext:
    return PrincipalContext(
        user_id="principal_beta",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:code_architecture_scout"],
        max_session_paid_tokens=50_000,
    )


def test_october_2026_pricing_table(estimator: PaidComputeCostEstimator) -> None:
    """Verifies live October 2026 rates for all 5 frontier/local models."""
    # Gemini 4 Argon
    p_argon = estimator.get_pricing_record("gemini-4-argon")
    assert p_argon.input_usd_per_1m_low == 2.00
    assert p_argon.input_usd_per_1m_expected == 3.00
    assert p_argon.input_usd_per_1m_high == 4.00
    assert p_argon.output_usd_per_1m_low == 10.00
    assert p_argon.output_usd_per_1m_expected == 15.00
    assert p_argon.output_usd_per_1m_high == 20.00
    assert p_argon.is_zero_cost is False

    # Claude Opus 5.5
    p_opus = estimator.get_pricing_record("claude-opus-5.5")
    assert p_opus.input_usd_per_1m_low == 10.00
    assert p_opus.input_usd_per_1m_expected == 15.00
    assert p_opus.input_usd_per_1m_high == 20.00
    assert p_opus.output_usd_per_1m_low == 50.00
    assert p_opus.output_usd_per_1m_expected == 75.00
    assert p_opus.output_usd_per_1m_high == 90.00

    # GPT-5.6 Sol
    p_sol = estimator.get_pricing_record("gpt-5.6-sol")
    assert p_sol.input_usd_per_1m_low == 2.00
    assert p_sol.input_usd_per_1m_expected == 2.50
    assert p_sol.output_usd_per_1m_expected == 10.00

    # Gemini 3.8 Flash
    p_flash = estimator.get_pricing_record("gemini-3.8-flash")
    assert p_flash.input_usd_per_1m_low == 0.75
    assert p_flash.input_usd_per_1m_expected == 1.00
    assert p_flash.input_usd_per_1m_high == 1.50
    assert p_flash.output_usd_per_1m_low == 3.75
    assert p_flash.output_usd_per_1m_expected == 5.00
    assert p_flash.output_usd_per_1m_high == 7.50

    # Gemma 4 26B MoE (Zero Cost)
    p_gemma = estimator.get_pricing_record("gemma-4-26b-a4b-it")
    assert p_gemma.input_usd_per_1m_expected == 0.0
    assert p_gemma.output_usd_per_1m_expected == 0.0
    assert p_gemma.is_zero_cost is True


def test_cost_estimate_calculation_ranges(estimator: PaidComputeCostEstimator) -> None:
    """Asserts Cost_low <= Cost_expected <= Cost_high for paid models, and 0 for Gemma."""
    # Paid model
    report = estimator.calculate_cost_estimate(
        model_id="gemini-4-argon",
        estimated_input_tokens=10_000,
        max_output_tokens=4096,
    )
    assert report.is_paid_model is True
    assert report.cost_low_usd > 0.0
    assert report.cost_low_usd <= report.cost_expected_usd
    assert report.cost_expected_usd <= report.cost_high_usd
    summary = report.formatted_summary()
    assert "Paid Compute Pre-Execution Cost Estimate" in summary
    assert "gemini-4-argon" in summary

    # Zero cost model
    gemma_report = estimator.calculate_cost_estimate(
        model_id="gemma-4-26b-a4b-it",
        estimated_input_tokens=10_000,
        max_output_tokens=4096,
    )
    assert gemma_report.is_paid_model is False
    assert gemma_report.cost_low_usd == 0.0
    assert gemma_report.cost_expected_usd == 0.0
    assert gemma_report.cost_high_usd == 0.0
    gemma_summary = gemma_report.formatted_summary()
    assert "Zero Cost Local Model" in gemma_summary


def test_heuristic_token_counting(estimator: PaidComputeCostEstimator) -> None:
    """Verifies heuristic fallback token estimation logic."""
    text = "a" * 380
    tokens = estimator.estimate_prompt_tokens(text, client=None)
    assert tokens == 100

    # With empty string
    assert estimator.estimate_prompt_tokens("", client=None) == 0

    # Mock live client
    mock_client = MagicMock()
    mock_client.models.count_tokens.return_value = MagicMock(total_tokens=123)
    tokens_live = estimator.estimate_prompt_tokens("test prompt", client=mock_client)
    assert tokens_live == 123


def test_upfront_confirmation_gate_blocks_when_declined(
    estimator: PaidComputeCostEstimator,
) -> None:
    """Proves ConfirmationDeclinedException is raised when compute is unconfirmed."""
    report = estimator.calculate_cost_estimate(
        model_id="gemini-3.8-flash",
        estimated_input_tokens=5000,
        max_output_tokens=2048,
    )

    # Declined by callback returning False
    with pytest.raises(ConfirmationDeclinedException, match="declined by confirmation callback"):
        estimator.request_paid_execution_confirmation(
            report,
            confirmation_callback=lambda r: False,
            auto_approve=False,
            interactive_prompt=False,
        )

    # Declined in non-interactive environment without auto_approve
    with pytest.raises(ConfirmationDeclinedException, match="requires explicit confirmation"):
        estimator.request_paid_execution_confirmation(
            report,
            confirmation_callback=None,
            auto_approve=False,
            interactive_prompt=False,
        )


def test_upfront_confirmation_gate_passes_when_approved(
    estimator: PaidComputeCostEstimator,
) -> None:
    """Verifies approval bypasses gate without exception."""
    report = estimator.calculate_cost_estimate(
        model_id="gemini-3.8-flash",
        estimated_input_tokens=5000,
    )

    # Approved via auto_approve=True
    assert (
        estimator.request_paid_execution_confirmation(
            report,
            auto_approve=True,
            interactive_prompt=False,
        )
        is True
    )

    # Approved via callback
    assert (
        estimator.request_paid_execution_confirmation(
            report,
            confirmation_callback=lambda r: True,
            auto_approve=False,
            interactive_prompt=False,
        )
        is True
    )

    # Zero-cost models auto-pass regardless
    zero_report = estimator.calculate_cost_estimate("gemma-4-26b-a4b-it", 10_000)
    assert (
        estimator.request_paid_execution_confirmation(
            zero_report,
            auto_approve=False,
            interactive_prompt=False,
        )
        is True
    )


def test_hard_session_budget_limit(
    budget_tracker: SessionBudgetTracker, principal_a: PrincipalContext
) -> None:
    """Proves BudgetExceededException is raised when principal budget is breached."""
    # Principal A cap is 20,000 tokens
    # Initial request 15,000 -> fits
    budget_tracker.check_and_reserve(principal_a, projected_tokens=15_000)
    budget_tracker.record_consumption(
        principal_a, input_tokens=10_000, output_tokens=5_000, cost_usd=0.05
    )

    # Second request of 10,000 -> 15,000 + 10,000 = 25,000 > 20,000 -> MUST RAISE
    with pytest.raises(BudgetExceededException, match="session budget exceeded"):
        budget_tracker.check_and_reserve(principal_a, projected_tokens=10_000)


def test_automated_kill_switch_on_exploration_runaway(
    budget_tracker: SessionBudgetTracker,
) -> None:
    """Proves KillSwitchTriggeredException is raised when exploration tokens breach ceiling."""
    # Safe recording
    budget_tracker.record_exploration_tokens(tokens=100_000, max_cap=250_000)
    budget_tracker.record_exploration_tokens(tokens=100_000, max_cap=250_000)

    # 100k + 100k + 60k = 260k > 250k -> MUST RAISE
    with pytest.raises(KillSwitchTriggeredException, match="Automated exploration kill-switch"):
        budget_tracker.record_exploration_tokens(tokens=60_000, max_cap=250_000)


def test_automated_kill_switch_on_session_spend(
    budget_tracker: SessionBudgetTracker, principal_a: PrincipalContext
) -> None:
    """Proves KillSwitchTriggeredException halts execution when spend ceiling is breached."""
    # Spend ceiling is $25.00
    budget_tracker.record_consumption(
        principal_a, input_tokens=1000, output_tokens=500, cost_usd=20.00
    )

    # Projected cost adds $6.00 -> $26.00 > $25.00 -> MUST RAISE
    with pytest.raises(KillSwitchTriggeredException, match="projected spend"):
        budget_tracker.check_and_reserve(principal_a, projected_tokens=100, estimated_cost_usd=6.00)


def test_multi_principal_budget_and_tracker_isolation(
    budget_tracker: SessionBudgetTracker,
    principal_a: PrincipalContext,
    principal_b: PrincipalContext,
) -> None:
    """Proves Principal A exhausting their budget does not penalize Principal B."""
    # Exhaust Principal A (cap 20,000)
    budget_tracker.record_consumption(
        principal_a, input_tokens=15_000, output_tokens=5_000, cost_usd=0.10
    )
    with pytest.raises(BudgetExceededException):
        budget_tracker.check_and_reserve(principal_a, projected_tokens=1_000)

    # Principal B has 50,000 tokens and 0 consumed -> should comfortably pass
    budget_tracker.check_and_reserve(principal_b, projected_tokens=25_000)
    consumption_b = budget_tracker.get_consumption(principal_b.user_id)
    assert consumption_b["total_tokens"] == 0


def test_principal_context_dual_field_compatibility() -> None:
    """Proves session_budget_tokens and max_session_paid_tokens are mutually synchronized."""
    # 1. Specifying session_budget_tokens synchronizes max_session_paid_tokens
    p1 = PrincipalContext(
        user_id="user_1",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
        session_budget_tokens=42_000,
    )
    assert p1.session_budget_tokens == 42_000
    assert p1.max_session_paid_tokens == 42_000

    # 2. Specifying max_session_paid_tokens synchronizes session_budget_tokens
    p2 = PrincipalContext(
        user_id="user_2",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
        max_session_paid_tokens=85_000,
    )
    assert p2.session_budget_tokens == 85_000
    assert p2.max_session_paid_tokens == 85_000

    # 3. Aliasing check for BudgetExceededError and BudgetExceededException
    assert BudgetExceededError is BudgetExceededException


def test_cost_estimator_boundary_condition_prevents_output_token_inversion(
    estimator: PaidComputeCostEstimator,
) -> None:
    """Verifies that estimated_output_tokens never exceeds max_output_tokens even for tight caps."""
    # When max_output_tokens is small (< 64, e.g. 32)
    report_32 = estimator.calculate_cost_estimate(
        model_id="gemini-3.8-flash",
        estimated_input_tokens=1000,
        max_output_tokens=32,
    )
    assert report_32.estimated_output_tokens <= 32
    assert report_32.estimated_output_tokens <= report_32.max_output_tokens
    assert report_32.cost_low_usd <= report_32.cost_expected_usd <= report_32.cost_high_usd

    # When max_output_tokens = 1
    report_1 = estimator.calculate_cost_estimate(
        model_id="gemini-3.8-flash",
        estimated_input_tokens=500,
        max_output_tokens=1,
    )
    assert report_1.estimated_output_tokens == 1
    assert report_1.max_output_tokens == 1


def test_single_source_yaml_pricing_includes_gemini4_hydrogen_neon_carbon(
    estimator: PaidComputeCostEstimator,
) -> None:
    """PR-4 (ECS-046/ECS-086/ECS-109): YAML pricing must include active Gemini 4 tiers."""
    from pathlib import Path

    import yaml

    from context_scout.config import ModelGovernancePolicy
    from context_scout.telemetry.cost_governance import OCTOBER_2026_PRICING

    raw_pricing = yaml.safe_load(Path("config/model_pricing.yaml").read_text(encoding="utf-8"))
    yaml_models = raw_pricing["models"]
    for model_id in (
        "gemini-4-hydrogen",
        "gemini-4-neon",
        "gemini-4-carbon",
        "gemini-3.8-flash",
        "gemini-4-argon",
        "claude-sonnet-5.5",
        "claude-opus-5.5",
        "gpt-5.6-sol",
    ):
        assert model_id in yaml_models, f"Missing {model_id} in config/model_pricing.yaml"
        rec = estimator.get_pricing_record(model_id)
        entry = yaml_models[model_id]
        assert rec.input_usd_per_1m_expected == float(entry["input_expected"])
        assert rec.output_usd_per_1m_expected == float(entry["output_expected"])
        assert rec.input_usd_per_1m_low == float(entry["input_low"])
        assert rec.input_usd_per_1m_high == float(entry["input_high"])
        assert rec.output_usd_per_1m_low == float(entry["output_low"])
        assert rec.output_usd_per_1m_high == float(entry["output_high"])
        assert rec.is_zero_cost is False
        assert OCTOBER_2026_PRICING[model_id] == rec

    # Registry also permits active Gemini 4 models only when external egress is allowed
    policy_off = ModelGovernancePolicy.from_registry_file("config/model_registry.yaml")
    for model_id in ("gemini-4-hydrogen", "gemini-4-neon", "gemini-4-carbon"):
        with pytest.raises(ValueError, match="allow_external_cloud_egress=False"):
            policy_off.validate_cloud_egress(model_id)

    policy_on = ModelGovernancePolicy.from_registry_file(
        "config/model_registry.yaml",
        allow_external_cloud_egress=True,
    )
    for model_id in (
        "gemini-4-hydrogen",
        "gemini-4-neon",
        "gemini-4-carbon",
        "gemini-3.8-flash",
        "gemini-4-argon",
    ):
        assert policy_on.validate_cloud_egress(model_id) == model_id

    for retired in (
        "gemini-2.5-flash",
        "codegemma",
        "gemma-3-27b-it",
        "text-embedding-004",
        "gemini-ultra",
    ):
        with pytest.raises(ValueError, match="Prohibited or retired model ID"):
            policy_on.validate_cloud_egress(retired)


def test_budget_tracker_atomic_reservation_under_concurrency() -> None:
    """PR-4 (ECS-012): Concurrent check_and_reserve calls must be atomic under lock."""
    import concurrent.futures

    tracker = SessionBudgetTracker()
    principal = PrincipalContext(
        user_id="concurrent_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
        session_budget_tokens=1000,
    )

    def _try_reserve() -> bool:
        try:
            tracker.check_and_reserve(principal, projected_tokens=400, estimated_cost_usd=0.01)
            return True
        except BudgetExceededError:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _try_reserve(), range(8)))

    assert sum(1 for ok in results if ok) == 2
    assert tracker.get_consumption("concurrent_user")["total_tokens"] == 0
    assert tracker.get_consumption("concurrent_user")["reserved_tokens"] == 800
    tracker.release_reservation(principal, reserved_tokens=400, reserved_cost_usd=0.01)
    assert tracker.get_consumption("concurrent_user")["reserved_tokens"] == 400
