"""Unit and integration tests for OpenTelemetry GenAI Semantic Instrumentation.

Tests:
1. ScoutTelemetryTracer lifecycle (singleton, reset, in-memory exporter).
2. Standard GenAI semantic conventions and Context Scout custom attributes.
3. Nested span hierarchy and context propagation (workflow -> preflight -> executor).
4. Query utilities (get_finished_spans, clear_spans, find_spans_by_name, find_spans_by_attribute).
5. Error handling and exception recording on spans.
6. Live end-to-end pipeline span capture.
"""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Final

import opentelemetry.trace as trace
import pytest

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.gemini_executor import GeminiImplementationExecutor
from context_scout.agents.scout_workflow import ScoutWorkflow
from context_scout.db.repositories import (
    ASTSymbolRecord,
    CodeFileRecord,
    InMemoryRLSRepository,
)
from context_scout.schemas.contracts import (
    ImplementationContextPack,
    ModificationIntent,
    PreFlightValidationReport,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)
from context_scout.telemetry import (
    ATTR_AUTO_ATTACHED_AST_COUNT,
    ATTR_CITATIONS_VERIFIED,
    ATTR_DOMAIN_ID,
    ATTR_ESTIMATED_USD_SAVED,
    ATTR_NORMALIZED_RRF_SCORE,
    ATTR_PREFLIGHT_PASSED,
    ATTR_PRINCIPAL_ID,
    ATTR_PRINCIPAL_ROLE,
    ATTR_TOKENS_SAVED,
    ATTR_TRACE_ID,
    ATTR_ZERO_TOUCH_SUFFICIENT,
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    ScoutTelemetryTracer,
    get_tracer,
    reset_tracer,
)

DOMAIN_ID: Final[str] = "trading_code_scout"
REPO_NAME: Final[str] = "trading-engine"


@pytest.fixture(autouse=True)
def clean_tracer() -> ScoutTelemetryTracer:
    """Resets global tracer before each test for total test isolation."""
    t = reset_tracer()
    t.clear_spans()
    return t


@pytest.fixture
def principal() -> PrincipalContext:
    return PrincipalContext(
        user_id="test_trader_1",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:trading_strategies"],
        max_session_paid_tokens=16_000,
    )


@pytest.fixture
def sample_pack() -> ImplementationContextPack:
    slice_item = VerbatimCodeSlice(
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
        unresolved_symbols=[],
        missing_reference_identifiers=[],
        novel_target_identifiers=[],
        normalized_rrf_score=0.92,
        citations_verified_100_pct=True,
        auto_attached_ast_slices_count=1,
    )
    return ImplementationContextPack(
        trace_id="trace_test_12345",
        domain_id=DOMAIN_ID,
        user_task="Enhance slip control in OrderManager",
        architectural_execution_plan=["Step 1", "Step 2"],
        code_slices=[slice_item],
        supporting_doc_citations=[],
        quant_strategy_spec=None,
        preflight_report=report,
        local_tokens_consumed=450,
        pack_token_count=180,
    )


def test_tracer_lifecycle_and_singleton() -> None:
    """Verifies get_instance, get_tracer, and reset_tracer singleton mechanics."""
    t1 = get_tracer()
    t2 = ScoutTelemetryTracer.get_instance()
    assert t1 is t2

    t3 = reset_tracer()
    assert t3 is not t1
    assert t3 is get_tracer()


def test_trace_workflow_attributes(sample_pack: ImplementationContextPack) -> None:
    """Verifies that trace_workflow records standard GenAI and Context Scout attributes."""
    tracer = get_tracer()

    with tracer.trace_workflow(
        span_name="scout.workflow",
        domain_id=DOMAIN_ID,
        trace_id="test_wf_trace",
        model_id="gemma-4-moe",
        system="gemma",
    ) as wf_span:
        wf_span.record_pack(sample_pack)

    spans = tracer.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "scout.workflow"

    attrs = span.attributes
    assert attrs is not None
    assert attrs[GEN_AI_SYSTEM] == "gemma"
    assert attrs[GEN_AI_REQUEST_MODEL] == "gemma-4-moe"
    assert attrs[GEN_AI_OPERATION_NAME] == "workflow"
    assert attrs[ATTR_DOMAIN_ID] == DOMAIN_ID
    assert attrs[ATTR_TRACE_ID] == sample_pack.trace_id
    assert attrs[ATTR_PREFLIGHT_PASSED] is True
    assert attrs[ATTR_NORMALIZED_RRF_SCORE] == 0.92
    assert attrs[ATTR_CITATIONS_VERIFIED] is True
    assert attrs[ATTR_AUTO_ATTACHED_AST_COUNT] == 1
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 450
    assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 180
    assert span.status.status_code == trace.StatusCode.OK


def test_trace_preflight_attributes() -> None:
    """Verifies that trace_preflight records preflight report metrics."""
    tracer = get_tracer()

    report = PreFlightValidationReport(
        passed=False,
        unresolved_symbols=["missing_func"],
        missing_reference_identifiers=[],
        novel_target_identifiers=[],
        normalized_rrf_score=0.42,
        citations_verified_100_pct=False,
        auto_attached_ast_slices_count=0,
        bounce_back_feedback="Check 1 failed: missing_func",
    )

    with tracer.trace_preflight(
        span_name="scout.preflight_gate",
        domain_id=DOMAIN_ID,
    ) as pf_span:
        pf_span.record_report(report)

    spans = tracer.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "scout.preflight_gate"

    attrs = span.attributes
    assert attrs is not None
    assert attrs[ATTR_DOMAIN_ID] == DOMAIN_ID
    assert attrs[GEN_AI_OPERATION_NAME] == "preflight_gate"
    assert attrs[ATTR_PREFLIGHT_PASSED] is False
    assert attrs[ATTR_NORMALIZED_RRF_SCORE] == 0.42
    assert attrs[ATTR_CITATIONS_VERIFIED] is False
    assert attrs[ATTR_AUTO_ATTACHED_AST_COUNT] == 0
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.status.description == "Check 1 failed: missing_func"


def test_trace_context_propagation_hierarchy(sample_pack: ImplementationContextPack) -> None:
    """Verifies context propagation: child spans share the trace ID and reference parent span ID."""
    tracer = get_tracer()

    with (
        tracer.trace_workflow(
            span_name="scout.workflow",
            domain_id=DOMAIN_ID,
            trace_id="hierarchical_trace",
        ) as wf_span,
        tracer.trace_preflight(
            span_name="scout.preflight_gate",
            domain_id=DOMAIN_ID,
        ) as pf_span,
    ):
        pf_span.record_report(sample_pack.preflight_report)
        wf_span.record_pack(sample_pack)

    spans = tracer.get_finished_spans()
    assert len(spans) == 2

    child_span = tracer.find_spans_by_name("scout.preflight_gate")[0]
    parent_span = tracer.find_spans_by_name("scout.workflow")[0]

    assert child_span.context.trace_id == parent_span.context.trace_id
    assert child_span.parent is not None
    assert child_span.parent.span_id == parent_span.context.span_id


def test_query_utilities() -> None:
    """Verifies find_spans_by_name, find_spans_by_attribute, and clear_spans."""
    tracer = get_tracer()

    with tracer.trace_workflow(span_name="wf_1", domain_id="domain_a"):
        pass

    with tracer.trace_workflow(span_name="wf_2", domain_id="domain_b"):
        pass

    assert len(tracer.get_finished_spans()) == 2
    assert len(tracer.find_spans_by_name("wf_1")) == 1
    assert len(tracer.find_spans_by_name("wf_2")) == 1

    matches_a = tracer.find_spans_by_attribute(ATTR_DOMAIN_ID, "domain_a")
    assert len(matches_a) == 1
    assert matches_a[0].name == "wf_1"

    matches_b = tracer.find_spans_by_attribute(ATTR_DOMAIN_ID, "domain_b")
    assert len(matches_b) == 1
    assert matches_b[0].name == "wf_2"

    tracer.clear_spans()
    assert len(tracer.get_finished_spans()) == 0


def test_exception_recording_on_span() -> None:
    """Verifies that an unhandled exception inside a traced block records the exception."""
    tracer = get_tracer()

    with pytest.raises(RuntimeError, match="Simulated crash"), tracer.trace_workflow(
        span_name="scout.workflow"
    ):
        raise RuntimeError("Simulated crash")

    spans = tracer.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == trace.StatusCode.ERROR
    assert "Simulated crash" in (span.status.description or "")

    assert len(span.events) >= 1
    assert span.events[0].name == "exception"


def test_end_to_end_scout_workflow_span_capture(principal: PrincipalContext) -> None:
    """End-to-end integration: ScoutWorkflow.run records root and child spans."""
    tracer = get_tracer()
    domains_dir = Path(__file__).parent.parent / "domains"
    registry = DomainRegistry(domains_dir=domains_dir)

    repo = InMemoryRLSRepository()
    file_id = uuid.uuid4()
    content = "class OrderManager:\n    def submit_order(self) -> bool:\n        return True\n"
    repo.insert_code_file(
        principal,
        CodeFileRecord(
            file_id=file_id,
            domain_id=DOMAIN_ID,
            repo_name=REPO_NAME,
            file_path="src/trading_strategies/order_manager.py",
            git_commit_sha="commit_12345",
            git_blob_sha="blob_1234567",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=content,
            line_count=3,
        ),
    )
    repo.insert_ast_symbol(
        principal,
        ASTSymbolRecord(
            symbol_id=uuid.uuid4(),
            file_id=file_id,
            domain_id=DOMAIN_ID,
            qualified_name="OrderManager.submit_order",
            symbol_kind="method",
            start_line=2,
            end_line=3,
            signature_skeleton="def submit_order(self) -> bool:",
            contextual_summary="Submits order",
            allowed_scopes=["public_eng"],
            embedding=[0.1] * 768,
        ),
    )

    workflow = ScoutWorkflow(registry=registry, repo=repo)
    pack = workflow.run(
        user_prompt="Refactor OrderManager.submit_order",
        principal=principal,
        domain_id=DOMAIN_ID,
        normalized_rrf_score=0.90,
    )
    assert isinstance(pack, ImplementationContextPack)

    finished = tracer.get_finished_spans()
    assert len(finished) >= 2

    wf_spans = tracer.find_spans_by_name("scout.workflow")
    assert len(wf_spans) == 1
    pf_spans = tracer.find_spans_by_name("scout.preflight_gate")
    assert len(pf_spans) >= 1

    root_attrs = wf_spans[0].attributes
    assert root_attrs is not None
    assert root_attrs[GEN_AI_SYSTEM] == "gemma"
    assert root_attrs[ATTR_DOMAIN_ID] == DOMAIN_ID
    assert root_attrs[ATTR_TRACE_ID] == pack.trace_id
    assert root_attrs[ATTR_PREFLIGHT_PASSED] is True
    assert wf_spans[0].status.status_code == trace.StatusCode.OK


def test_end_to_end_gemini_executor_span_capture(
    principal: PrincipalContext, sample_pack: ImplementationContextPack
) -> None:
    """End-to-end integration: GeminiImplementationExecutor records executor span."""
    tracer = get_tracer()

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)
        target = repo_root / "src/trading_strategies/order_manager.py"
        target.parent.mkdir(parents=True)
        target.write_text(
            "class OrderManager:\n"
            "    def submit_order(self, order_id: str) -> bool:\n"
            "        return True\n"
        )

        subprocess.run(
            ["git", "init", "-b", "main"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_root,
            check=True,
        )
        subprocess.run(["git", "add", "-A"], cwd=repo_root, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=repo_root, check=True)

        repo = InMemoryRLSRepository()
        executor = GeminiImplementationExecutor(require_confirmation=False)
        _result, telemetry = executor.execute_implementation(
            pack=sample_pack,
            principal=principal,
            repo_root=repo_root,
            repo=repo,
            custom_patch="",
        )

    exec_spans = tracer.find_spans_by_name("scout.executor.handoff")
    assert len(exec_spans) == 1
    attrs = exec_spans[0].attributes
    assert attrs is not None
    assert attrs[GEN_AI_SYSTEM] == "gemini"
    assert attrs[GEN_AI_REQUEST_MODEL] == "gemini-3.8-flash"
    assert attrs[ATTR_DOMAIN_ID] == DOMAIN_ID
    assert attrs[ATTR_TOKENS_SAVED] == telemetry.paid_tokens_saved
    assert attrs[ATTR_ESTIMATED_USD_SAVED] == telemetry.estimated_usd_saved
    assert attrs[ATTR_ZERO_TOUCH_SUFFICIENT] is True
    assert attrs[ATTR_PRINCIPAL_ID] == principal.user_id
    assert attrs[ATTR_PRINCIPAL_ROLE] == principal.role.value
