"""Permanent regression test suite for Critical A: Fail-closed deterministic model abstention.

Verifies:
1. When LocalGemmaClient is invoked with invalid/non-existent model path or when MLX raises:
   - NEVER fabricates model output.
   - Deterministically abstains with explicit diagnostic "model unavailable".
   - Enforces paid_tokens_spent=0 and cost_usd=0.0.
2. When offline_mode=True without an explicitly injected test fixture:
   - Deterministically abstains with "model unavailable".
   - When completion_fixture IS injected, permits test-fixture execution.
3. Production path verification:
   - Static AST check proving scout_workflow.py contains 0 calls, imports, or definitions
     of simulation generators.
4. Mutation check:
   - Proves introducing a simulated completion generator back into complete() fails tests.
"""

from __future__ import annotations

import ast
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest

from context_scout.agents.scout_workflow import (
    LocalGemmaClient,
    ModelUnavailableError,
    ScoutWorkflow,
)
from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.schemas.contracts import (
    InsufficientContextDiagnostic,
    PrincipalContext,
    RoleType,
)
from context_scout.telemetry.otel import (
    ATTR_SCOUT_COST_USD,
    ATTR_SCOUT_IS_ABSTENTION,
    ATTR_SCOUT_PAID_TOKENS_SPENT,
    GEN_AI_USAGE_INPUT_TOKENS,
    ScoutTelemetryTracer,
)

# =============================================================================
# 1. Invalid Model Path / MLX Exception Abstention (Part A)
# =============================================================================


@contextmanager
def _override_mlx_lm(replacement: types.ModuleType | None) -> Iterator[None]:
    """Overrides only sys.modules['mlx_lm'] without clearing unrelated C++ modules like torch."""
    had_prev = "mlx_lm" in sys.modules
    prev_mod = sys.modules.get("mlx_lm")
    sys.modules["mlx_lm"] = replacement  # type: ignore[assignment]
    try:
        yield
    finally:
        if had_prev and prev_mod is not None:
            sys.modules["mlx_lm"] = prev_mod
        else:
            sys.modules.pop("mlx_lm", None)


class TestModelUnavailableAbstentionOnModelPathFailure:
    """Verifies LocalGemmaClient and ScoutWorkflow fail closed on runtime MLX errors."""

    def test_client_complete_invalid_model_path_raises_model_unavailable(self) -> None:
        """When model path does not exist, complete() must raise ModelUnavailableError."""
        client = LocalGemmaClient(
            model_id="google/gemma-4-26b-a4b-it",
            use_live_mlx=True,
            offline_mode=False,
        )

        fake_mlx = types.ModuleType("mlx_lm")
        fake_mlx.load = MagicMock(  # type: ignore[attr-defined]
            side_effect=FileNotFoundError("No model weights at invalid/path")
        )
        fake_mlx.generate = MagicMock()  # type: ignore[attr-defined]

        with (
            _override_mlx_lm(fake_mlx),
            pytest.raises(ModelUnavailableError) as exc_info,
        ):
            client.complete(
                prompt="Synthesize quant strategy brief",
                domain_id="quant_research_scout",
            )

        err = exc_info.value
        assert "model unavailable" in str(err).lower()
        assert err.diagnostic is not None
        assert err.diagnostic.is_abstention is True
        assert err.diagnostic.paid_tokens_spent == 0

    def test_client_complete_mlx_runtime_exception_never_fabricates_output(self) -> None:
        """When MLX raises runtime error, client must fail closed without fabrication."""
        client = LocalGemmaClient(use_live_mlx=True, offline_mode=False)

        fake_mlx = types.ModuleType("mlx_lm")
        fake_mlx.load = MagicMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError("Metal GPU OOM on Apple M5 Pro")
        )
        fake_mlx.generate = MagicMock()  # type: ignore[attr-defined]

        with (
            _override_mlx_lm(fake_mlx),
            pytest.raises(ModelUnavailableError) as exc_info,
        ):
            output = client.complete(
                prompt="Extract AST pointers from order_flow_imbalance.py",
                domain_id="trading_code_scout",
            )
            pytest.fail(f"Expected ModelUnavailableError, got fabricated output: {output}")

        assert "model unavailable" in str(exc_info.value).lower()

    def test_client_complete_when_mlx_lm_not_installed_abstains_cleanly(self) -> None:
        """On Linux x86_64 where mlx_lm is absent, complete() fails closed cleanly."""
        client = LocalGemmaClient(use_live_mlx=True, offline_mode=False)

        with (
            _override_mlx_lm(None),
            pytest.raises(ModelUnavailableError) as exc_info,
        ):
            client.complete(
                prompt="Extract AST pointers from order_flow_imbalance.py",
                domain_id="trading_code_scout",
            )

        err = exc_info.value
        assert "model unavailable" in str(err).lower()
        assert err.diagnostic is not None
        assert err.diagnostic.is_abstention is True
        assert err.diagnostic.paid_tokens_spent == 0

    def test_scout_workflow_abstains_with_zero_tokens_and_zero_cost_on_model_failure(self) -> None:
        """When local model fails, ScoutWorkflow must return diagnostic with $0 spend."""
        tracer = ScoutTelemetryTracer.reset_instance()
        tracer.exporter.clear()

        client = LocalGemmaClient(use_live_mlx=True, offline_mode=False)
        repo = InMemoryRLSRepository()
        principal = PrincipalContext(
            user_id="test_user",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", "repo:trading_strategies"],
        )
        workflow = ScoutWorkflow(repo=repo, local_client=client)

        with patch.object(
            client,
            "complete",
            side_effect=ModelUnavailableError("model unavailable: MLX crashed"),
        ):
            result = workflow.execute(
                user_prompt="Refactor OrderFlowImbalanceAnalyzer",
                principal=principal,
                domain_id="trading_code_scout",
                trace_id="trace_test_mlx_fail_closed",
            )

        assert isinstance(result, InsufficientContextDiagnostic)
        assert result.is_abstention is True
        assert "model unavailable" in result.reason.lower()
        assert result.paid_tokens_spent == 0
        assert "model:local_gemma:unavailable" in result.missing_items

        # Verify OpenTelemetry trace span enforces 0 tokens and $0.00 spend
        spans = tracer.exporter.get_finished_spans()
        assert len(spans) >= 1
        wf_span = spans[-1]
        assert wf_span.attributes is not None
        assert wf_span.attributes.get(ATTR_SCOUT_IS_ABSTENTION) is True
        assert wf_span.attributes.get(ATTR_SCOUT_PAID_TOKENS_SPENT) == 0
        assert wf_span.attributes.get(ATTR_SCOUT_COST_USD) == 0.0
        assert wf_span.attributes.get(GEN_AI_USAGE_INPUT_TOKENS) == 0


# =============================================================================
# 2. Offline Mode Without Explicit Test Fixture (Part B)
# =============================================================================


class TestOfflineModeDeterministicAbstention:
    """Verifies offline_mode=True fails closed unless a test fixture is explicitly injected."""

    def test_offline_mode_without_fixture_raises_model_unavailable(self) -> None:
        """Calling complete() with offline_mode=True and no fixture must raise error."""
        client = LocalGemmaClient(offline_mode=True)
        assert client.completion_fixture is None

        with pytest.raises(ModelUnavailableError) as exc_info:
            client.complete("Synthesize quant brief", domain_id="quant_research_scout")

        assert "model unavailable" in str(exc_info.value).lower()
        assert exc_info.value.diagnostic is not None
        assert exc_info.value.diagnostic.paid_tokens_spent == 0
        assert exc_info.value.diagnostic.is_abstention is True

    def test_offline_mode_scout_workflow_emits_insufficient_context_diagnostic(self) -> None:
        """ScoutWorkflow with default offline client returns InsufficientContextDiagnostic."""
        client = LocalGemmaClient(offline_mode=True)
        repo = InMemoryRLSRepository()
        principal = PrincipalContext(
            user_id="test_user",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", "corpus:quant_papers"],
        )
        workflow = ScoutWorkflow(repo=repo, local_client=client)

        result = workflow.execute(
            user_prompt="Synthesize DAX Momentum strategy",
            principal=principal,
            domain_id="quant_research_scout",
        )

        assert isinstance(result, InsufficientContextDiagnostic)
        assert result.is_abstention is True
        assert "model unavailable" in result.reason.lower()
        assert result.paid_tokens_spent == 0

    def test_offline_mode_with_explicitly_injected_fixture_permits_testing(self) -> None:
        """When a test injects completion_fixture, client executes it without fallback."""

        def mock_fixture(prompt: str, **kwargs: Any) -> str:
            return '{"test_fixture_acknowledged": true}'

        client = LocalGemmaClient(offline_mode=True, completion_fixture=mock_fixture)
        output = client.complete("Test prompt", domain_id="trading_code_scout")
        assert output == '{"test_fixture_acknowledged": true}'


# =============================================================================
# 3. Production Path Static AST Verification (Part C)
# =============================================================================


class TestProductionPathStaticASTVerification:
    """Static AST verification proving scout_workflow.py contains zero simulation code."""

    FORBIDDEN_SIMULATION_SYMBOLS: ClassVar[set[str]] = {
        "_deterministic_completion_generator",
        "mock_gemma_generator",
        "simulate_untuned_base_model_completion",
    }

    def test_scout_workflow_ast_has_zero_forbidden_simulation_symbols(self) -> None:
        """Scout workflow must contain 0 calls, 0 imports, and 0 definitions of simulators."""
        workflow_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "context_scout"
            / "agents"
            / "scout_workflow.py"
        )
        assert workflow_path.exists(), f"scout_workflow.py not found at {workflow_path}"

        tree = ast.parse(workflow_path.read_text(encoding="utf-8"))
        violations: list[str] = []

        for node in ast.walk(tree):
            # 1. Check imports: from ... import ...
            if isinstance(node, ast.ImportFrom):
                if node.module and any(f in node.module for f in self.FORBIDDEN_SIMULATION_SYMBOLS):
                    violations.append(f"ImportFrom module '{node.module}' at line {node.lineno}")
                for alias in node.names:
                    if alias.name in self.FORBIDDEN_SIMULATION_SYMBOLS:
                        violations.append(f"ImportFrom name '{alias.name}' at line {node.lineno}")
            # 2. Check imports: import ...
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(f in alias.name for f in self.FORBIDDEN_SIMULATION_SYMBOLS):
                        violations.append(f"Import module '{alias.name}' at line {node.lineno}")
            # 3. Check function calls: func(...) or obj.func(...)
            elif isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in self.FORBIDDEN_SIMULATION_SYMBOLS
                ):
                    violations.append(f"Call Name '{node.func.id}' at line {node.lineno}")
                elif (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in self.FORBIDDEN_SIMULATION_SYMBOLS
                ):
                    violations.append(f"Call Attr '{node.func.attr}' at line {node.lineno}")
            # 4. Check function/method definitions: def func(...)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in self.FORBIDDEN_SIMULATION_SYMBOLS:
                    violations.append(f"FunctionDef '{node.name}' at line {node.lineno}")

        msg = (
            f"Production path 'scout_workflow.py' contains {len(violations)} forbidden "
            f"simulation symbols:\n" + "\n".join(f" - {v}" for v in violations)
        )
        assert len(violations) == 0, msg


# =============================================================================
# 4. Mutation Check (Part D)
# =============================================================================


class TestModelAbstentionMutationProof:
    """Proves that introducing a simulated completion generator back into complete() fails."""

    def test_mutation_simulated_completion_fails_regression_assertion(self) -> None:
        """Mutates complete() to return simulated JSON; proves fail-closed check detects it."""
        client = LocalGemmaClient(offline_mode=True)

        # Mutated function simulating the regression: fabricating output when offline
        def mutated_complete(prompt: str, **kwargs: Any) -> str:
            return '{"strategy_name": "Fabricated Untuned Strategy", "verified_citations": []}'

        def regression_check(c: LocalGemmaClient) -> None:
            # The regression invariant: complete() must raise ModelUnavailableError when offline
            try:
                out = c.complete("Test prompt")
                raise AssertionError(f"Regression detected: model output was fabricated: {out}")
            except ModelUnavailableError:
                pass  # Correct fail-closed behavior

        # Clean check passes
        regression_check(client)

        # Mutate complete() to simulate fallback regression
        with (
            patch.object(
                client,
                "complete",
                side_effect=lambda *a, **kw: mutated_complete(*a, **kw),
            ),
            pytest.raises(AssertionError, match="Regression detected: model output was fabricated"),
        ):
            regression_check(client)
