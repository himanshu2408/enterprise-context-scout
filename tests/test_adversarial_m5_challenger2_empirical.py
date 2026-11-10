"""Empirical adversarial stress test suite for Feature F24 (OpenTelemetry Semantic Instrumentation).

Challenger 2 suite verifying:
1. 4-Phase Span Hierarchy & Nesting (scout.workflow -> retrieval, slicing, gating, execution),
   including child exception recording, event emission, and error status propagation.
2. Zero-Spend Abstention Invariant on Traces (is_abstention=True, paid_tokens_spent=0, cost_usd=0.0)
   across all deterministic code gates, guardrail rejections, and tampered diagnostic payloads.
3. StructuredJsonSpanExporter single-line JSONL compliance, schema completeness, and resilience
   against broken streams, multiline strings, JSON injection, and complex attributes.
4. W3C TraceContext propagation boundaries: valid injection/extraction roundtrips, multithreaded
   isolation, and graceful fallback to new root traces on corrupted/malformed traceparents.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import opentelemetry.trace as trace
import pytest

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.scout_workflow import ScoutWorkflow
from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.schemas.contracts import (
    InsufficientContextDiagnostic,
    MissingItemRecord,
    PrincipalContext,
    RoleType,
)
from context_scout.telemetry import (
    ATTR_DOMAIN_ID,
    ATTR_PREFLIGHT_PASSED,
    ATTR_SCOUT_COST_USD,
    ATTR_SCOUT_DOMAIN_ID,
    ATTR_SCOUT_FAILURE_CHECK,
    ATTR_SCOUT_IS_ABSTENTION,
    ATTR_SCOUT_MISSING_ITEMS_COUNT,
    ATTR_SCOUT_PAID_TOKENS_SPENT,
    ATTR_SCOUT_PHASE,
    ATTR_SCOUT_PREFLIGHT_PASSED,
    ATTR_SCOUT_REJECTION_REASON,
    ATTR_SCOUT_TRACE_ID,
    ATTR_TRACE_ID,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    PHASE_EXECUTION,
    PHASE_GATING,
    PHASE_RETRIEVAL,
    PHASE_SLICING,
    PHASE_WORKFLOW,
    ScoutTelemetryTracer,
    StructuredJsonSpanExporter,
    extract_trace_context,
    inject_trace_context,
    reset_tracer,
)


class SimulatedPhaseError(RuntimeError):
    """Custom exception simulating an unhandled failure inside a pipeline phase."""


@pytest.fixture(autouse=True)
def isolated_tracer() -> ScoutTelemetryTracer:
    """Provides a pristine, isolated ScoutTelemetryTracer before each test."""
    t = reset_tracer()
    t.clear_spans()
    return t


# =============================================================================
# 1. 4-Phase Span Hierarchy & Nesting
# =============================================================================


class TestFourPhaseSpanHierarchyAndNesting:
    """Stress tests 4-phase span hierarchy, parent-child linkages, and exception recording."""

    def test_root_workflow_with_all_four_active_phases_in_sequence(
        self, isolated_tracer: ScoutTelemetryTracer
    ) -> None:
        tracer = isolated_tracer
        domain_id = "trading_code_scout"
        root_trace_id = "trace_adversarial_hier_001"

        with tracer.trace_workflow(
            span_name="scout.workflow",
            domain_id=domain_id,
            trace_id=root_trace_id,
        ) as wf_ctx:
            # Phase 1: Retrieval
            with tracer.trace_retrieval(domain_id=domain_id, query="OrderBook") as r_ctx:
                r_ctx.record_retrieval(
                    doc_chunks_count=4,
                    ast_symbols_count=18,
                    query="OrderBook",
                )

            # Phase 2: Slicing
            with tracer.trace_slicing(domain_id=domain_id, file_path="src/order_book.py") as s_ctx:
                s_ctx.record_slicing(
                    file_path="src/order_book.py",
                    lines_sliced=82,
                    syntax_valid=True,
                )

            # Phase 3: Gating
            with tracer.trace_gating(domain_id=domain_id, prompt="Refactor book") as g_ctx:
                report = MagicMock(
                    passed=True,
                    normalized_rrf_score=0.91,
                    citations_verified_100_pct=True,
                    auto_attached_ast_slices_count=2,
                )
                g_ctx.record_report(report)

            # Phase 4: Execution
            with tracer.trace_execution(domain_id=domain_id, model_id="gemini-3.8-flash") as e_ctx:
                result = MagicMock(
                    success=True,
                    token_count=350,
                    verification_passed=True,
                    patch_applied=True,
                )
                telemetry = MagicMock(
                    paid_tokens_saved=1200,
                    estimated_usd_saved=0.009,
                    zero_touch_sufficient=True,
                    context_pack_tokens_sent=800,
                    preflight_passed_first_try=True,
                )
                e_ctx.record_result(result, telemetry)

            mock_pack = MagicMock(
                preflight_report=report,
                local_tokens_consumed=150,
                pack_token_count=800,
                trace_id=root_trace_id,
                domain_id=domain_id,
            )
            wf_ctx.record_pack(mock_pack)

        spans = tracer.get_finished_spans()
        assert len(spans) == 5

        root_spans = tracer.find_spans_by_name("scout.workflow")
        assert len(root_spans) == 1
        root_span = root_spans[0]

        retrieval_spans = tracer.find_spans_by_name("scout.retrieval")
        slicing_spans = tracer.find_spans_by_name("scout.slicing")
        gating_spans = tracer.find_spans_by_name("scout.preflight_gate")
        exec_spans = tracer.find_spans_by_name("scout.executor.handoff")

        assert len(retrieval_spans) == 1
        assert len(slicing_spans) == 1
        assert len(gating_spans) == 1
        assert len(exec_spans) == 1

        child_spans = [
            retrieval_spans[0],
            slicing_spans[0],
            gating_spans[0],
            exec_spans[0],
        ]

        expected_phases = [
            PHASE_RETRIEVAL,
            PHASE_SLICING,
            PHASE_GATING,
            PHASE_EXECUTION,
        ]

        root_trace_id_hex = trace.format_trace_id(root_span.context.trace_id)
        root_span_id_hex = trace.format_span_id(root_span.context.span_id)

        for child, expected_phase in zip(child_spans, expected_phases, strict=True):
            assert trace.format_trace_id(child.context.trace_id) == root_trace_id_hex
            assert child.parent is not None
            assert trace.format_span_id(child.parent.span_id) == root_span_id_hex
            assert child.attributes is not None
            assert child.attributes[ATTR_SCOUT_PHASE] == expected_phase
            assert child.status.status_code == trace.StatusCode.OK

        assert root_span.attributes is not None
        assert root_span.attributes[ATTR_SCOUT_PHASE] == PHASE_WORKFLOW
        assert root_span.status.status_code == trace.StatusCode.OK

    def test_multi_tier_dag_subspan_hierarchy(
        self, isolated_tracer: ScoutTelemetryTracer
    ) -> None:
        tracer = isolated_tracer

        with (
            tracer.trace_workflow(span_name="scout.workflow") as _wf,
            tracer.trace_gating(span_name="scout.preflight_gate") as _gating,
            tracer.tracer.start_as_current_span("scout.preflight_gate.check_1_syntax") as sub,
        ):
            sub.set_attribute("check.id", "check_1")

        spans = tracer.get_finished_spans()
        assert len(spans) == 3

        root = tracer.find_spans_by_name("scout.workflow")[0]
        gate = tracer.find_spans_by_name("scout.preflight_gate")[0]
        sub_span = tracer.find_spans_by_name("scout.preflight_gate.check_1_syntax")[0]

        assert gate.parent is not None
        gate_parent_hex = trace.format_span_id(gate.parent.span_id)
        assert gate_parent_hex == trace.format_span_id(root.context.span_id)

        assert sub_span.parent is not None
        sub_parent_hex = trace.format_span_id(sub_span.parent.span_id)
        assert sub_parent_hex == trace.format_span_id(gate.context.span_id)

    @pytest.mark.parametrize(
        ("phase_type", "span_name"),
        [
            ("retrieval", "scout.retrieval"),
            ("slicing", "scout.slicing"),
            ("gating", "scout.preflight_gate"),
            ("execution", "scout.executor.handoff"),
            ("workflow", "scout.workflow"),
        ],
    )
    def test_exception_status_and_events_in_each_phase_wrapper(
        self,
        isolated_tracer: ScoutTelemetryTracer,
        phase_type: str,
        span_name: str,
    ) -> None:
        tracer = isolated_tracer
        fault_msg = f"Fault injected into phase: {phase_type}"

        def _invoke_phase(pt: str) -> Any:
            if pt == "retrieval":
                return tracer.trace_retrieval(domain_id="test_dom")
            if pt == "slicing":
                return tracer.trace_slicing(domain_id="test_dom")
            if pt == "gating":
                return tracer.trace_gating(domain_id="test_dom")
            if pt == "execution":
                return tracer.trace_execution(domain_id="test_dom")
            if pt == "workflow":
                return tracer.trace_workflow(domain_id="test_dom")
            raise ValueError(f"Unknown phase {pt}")

        with (
            pytest.raises(SimulatedPhaseError, match=fault_msg),
            _invoke_phase(phase_type),
        ):
            raise SimulatedPhaseError(fault_msg)

        spans = tracer.find_spans_by_name(span_name)
        assert len(spans) == 1
        span = spans[0]

        assert span.status.status_code == trace.StatusCode.ERROR
        assert fault_msg in (span.status.description or "")

        exception_events = [e for e in span.events if e.name == "exception"]
        assert len(exception_events) >= 1
        event_attrs = exception_events[0].attributes
        assert event_attrs is not None
        assert str(event_attrs.get("exception.type", "")).endswith("SimulatedPhaseError")
        assert fault_msg in str(event_attrs.get("exception.message"))

    def test_child_exception_handled_by_workflow_does_not_corrupt_tracer(
        self, isolated_tracer: ScoutTelemetryTracer
    ) -> None:
        tracer = isolated_tracer

        with tracer.trace_workflow(span_name="scout.workflow") as wf:
            # Phase 1: Fails but handled by caller
            try:
                with tracer.trace_retrieval(domain_id="test_dom"):
                    raise SimulatedPhaseError("Corrupted vector index")
            except SimulatedPhaseError:
                pass

            # Phase 2: Secondary fallback succeeds cleanly
            with tracer.trace_slicing(domain_id="test_dom") as s_ctx:
                s_ctx.record_slicing("fallback.py", 10, True)

            wf.set_status(trace.StatusCode.OK)

        spans = tracer.get_finished_spans()
        assert len(spans) == 3

        retrieval_span = tracer.find_spans_by_name("scout.retrieval")[0]
        slicing_span = tracer.find_spans_by_name("scout.slicing")[0]
        workflow_span = tracer.find_spans_by_name("scout.workflow")[0]

        assert retrieval_span.status.status_code == trace.StatusCode.ERROR
        assert "Corrupted vector index" in (retrieval_span.status.description or "")
        assert slicing_span.status.status_code == trace.StatusCode.OK
        assert workflow_span.status.status_code == trace.StatusCode.OK


# =============================================================================
# 2. Zero-Spend Abstention Invariant on Traces
# =============================================================================


class TestZeroSpendAbstentionInvariantOnTraces:
    """Stress tests the strict enforcement of the zero-spend abstention invariant on spans."""

    @pytest.mark.parametrize(
        ("check_name", "reason"),
        [
            ("check_1_symbols", "Check 1 failed: missing AST symbol 'OrderBook'"),
            ("check_2_closure", "Check 2 failed: missing 1-hop caller closure"),
            ("check_3_rrf", "Check 3 failed: normalized RRF score 0.51 < 0.65 threshold"),
            ("check_4_staleness", "Check 4 failed: Confluence version superseded / closed Jira"),
            ("check_5_citations", "Check 5 failed: claim token overlap < 0.98 threshold"),
            ("guardrail_rejected", "Security guardrail rejected prompt injection"),
            ("budget_breach", "Session budget ceiling breached before dispatch"),
            ("kill_switch_triggered", "Exploration token kill switch triggered (>250k tokens)"),
        ],
    )
    def test_abstention_invariant_across_all_deterministic_failure_codes(
        self,
        isolated_tracer: ScoutTelemetryTracer,
        check_name: str,
        reason: str,
    ) -> None:
        tracer = isolated_tracer
        trace_id = f"trace_abstain_{check_name}"
        domain_id = "trading_code_scout"

        diagnostic = InsufficientContextDiagnostic(
            trace_id=trace_id,
            domain_id=domain_id,
            reason=reason,
            failure_check=check_name,
            missing_symbols=["OrderBook"] if "symbols" in check_name else [],
            paid_tokens_spent=0,
        )

        with tracer.trace_workflow(
            span_name="scout.workflow.abstention",
            domain_id=domain_id,
            trace_id=trace_id,
        ) as wf_ctx:
            wf_ctx.record_pack(diagnostic)

        spans = tracer.find_spans_by_name("scout.workflow.abstention")
        assert len(spans) == 1
        span = spans[0]
        attrs = span.attributes
        assert attrs is not None

        # Mandatory Zero-Spend Invariant Assertions
        assert attrs[ATTR_SCOUT_IS_ABSTENTION] is True
        assert attrs[ATTR_SCOUT_PAID_TOKENS_SPENT] == 0
        assert attrs[ATTR_SCOUT_COST_USD] == 0.0
        assert attrs[ATTR_SCOUT_PREFLIGHT_PASSED] is False
        assert attrs[ATTR_SCOUT_FAILURE_CHECK] == check_name
        assert attrs[ATTR_SCOUT_REJECTION_REASON] == reason
        assert attrs[ATTR_SCOUT_TRACE_ID] == trace_id
        assert attrs[ATTR_SCOUT_DOMAIN_ID] == domain_id

        # GenAI Semantic conventions: 0 paid tokens consumed
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 0
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 0

        # Legacy backward-compatibility attributes
        assert attrs[ATTR_PREFLIGHT_PASSED] is False
        assert attrs[ATTR_TRACE_ID] == trace_id
        assert attrs[ATTR_DOMAIN_ID] == domain_id
        assert attrs["context_scout.rejection_reason"] == reason

        # Span status MUST reflect ERROR with reason description
        assert span.status.status_code == trace.StatusCode.ERROR
        assert span.status.description == reason

    def test_abstention_with_structured_missing_items(
        self, isolated_tracer: ScoutTelemetryTracer
    ) -> None:
        tracer = isolated_tracer
        missing_items = [
            MissingItemRecord(
                kind="code_symbol",
                identifier="OrderBook.cancel",
                reason="Symbol absent from repository AST index",
            ),
            MissingItemRecord(
                kind="document",
                identifier="CONF-DOC-42-v3",
                reason="Superseded by CONF-DOC-42-v4",
            ),
            MissingItemRecord(
                kind="file",
                identifier="src/book.py",
                reason="File not found on current branch",
            ),
        ]

        diagnostic = InsufficientContextDiagnostic(
            trace_id="trace_missing_items_001",
            domain_id="trading_code_scout",
            reason="Missing multiple required context items",
            failure_check="check_1_symbols",
            missing_items=missing_items,
            paid_tokens_spent=0,
        )

        with tracer.trace_workflow(span_name="scout.workflow.missing_items") as wf_ctx:
            wf_ctx.record_pack(diagnostic)

        span = tracer.find_spans_by_name("scout.workflow.missing_items")[0]
        attrs = span.attributes
        assert attrs is not None
        assert attrs[ATTR_SCOUT_IS_ABSTENTION] is True
        assert attrs[ATTR_SCOUT_PAID_TOKENS_SPENT] == 0
        assert attrs[ATTR_SCOUT_COST_USD] == 0.0
        assert attrs[ATTR_SCOUT_MISSING_ITEMS_COUNT] == 3

    def test_adversarial_tampered_diagnostic_forces_zero_spend(
        self, isolated_tracer: ScoutTelemetryTracer
    ) -> None:
        tracer = isolated_tracer

        class TamperedDiagnosticPayload:
            trace_id = "trace_malicious_forged_spend"
            domain_id = "trading_code_scout"
            reason = "Preflight gate failed but adversary forged spend metrics"
            failure_check = "check_2_closure"
            paid_tokens_spent = 999_999
            cost_usd = 450.0
            is_abstention = False
            preflight_passed = True

        with tracer.trace_workflow(span_name="scout.workflow.adversarial") as wf_ctx:
            wf_ctx.record_abstention(TamperedDiagnosticPayload())

        span = tracer.find_spans_by_name("scout.workflow.adversarial")[0]
        attrs = span.attributes
        assert attrs is not None

        # Even with forged non-zero attributes on payload, span strictly overrides to 0
        assert attrs[ATTR_SCOUT_IS_ABSTENTION] is True
        assert attrs[ATTR_SCOUT_PAID_TOKENS_SPENT] == 0
        assert attrs[ATTR_SCOUT_COST_USD] == 0.0
        assert attrs[ATTR_SCOUT_PREFLIGHT_PASSED] is False
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 0
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 0

    def test_live_scout_workflow_abstention_trace_integration(
        self, isolated_tracer: ScoutTelemetryTracer
    ) -> None:
        tracer = isolated_tracer
        domains_dir = Path(__file__).parent.parent / "domains"
        registry = DomainRegistry(domains_dir=domains_dir)
        repo = InMemoryRLSRepository()
        workflow = ScoutWorkflow(registry=registry, repo=repo)

        principal = PrincipalContext(
            user_id="adversarial_trader_01",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )

        result = workflow.run(
            user_prompt="Refactor NonExistentQuantumHedgingAlgorithm9999 in order_manager.py",
            principal=principal,
            domain_id="trading_code_scout",
        )

        assert isinstance(result, InsufficientContextDiagnostic)
        assert result.is_abstention is True
        assert result.paid_tokens_spent == 0

        wf_spans = tracer.find_spans_by_name("scout.workflow")
        assert len(wf_spans) == 1
        attrs = wf_spans[0].attributes
        assert attrs is not None
        assert attrs[ATTR_SCOUT_IS_ABSTENTION] is True
        assert attrs[ATTR_SCOUT_PAID_TOKENS_SPENT] == 0
        assert attrs[ATTR_SCOUT_COST_USD] == 0.0


# =============================================================================
# 3. Structured JSON Log Exporter Stressing
# =============================================================================


class TestStructuredJsonSpanExporterStressing:
    """Stress tests StructuredJsonSpanExporter for JSONL compliance, schema, and resilience."""

    def test_single_line_jsonl_invariance_under_adversarial_characters(
        self, isolated_tracer: ScoutTelemetryTracer
    ) -> None:
        stream = io.StringIO()
        exporter = StructuredJsonSpanExporter(stream=stream)

        processor_module = pytest.importorskip("opentelemetry.sdk.trace.export")
        isolated_tracer.provider.add_span_processor(
            processor_module.SimpleSpanProcessor(exporter)
        )

        test_payloads = [
            'multiline\ntext\r\nwith\nbreaks\tand\ttabs',
            '{"injected_json": true, "payload": ["item1", 123]}',
            'quotes: "double" and \'single\' and `backtick`',
            'unicode: \u2603 \u2705 \u03c0 \u222b, homoglyphs: \u0430 (cyrillic) vs a (latin)',
            'special_chars: & < > / \\ \b \f \0 (null byte)',
        ]

        with isolated_tracer.trace_workflow(span_name="scout.workflow.json_stress") as wf:
            for idx, payload in enumerate(test_payloads):
                with isolated_tracer.trace_retrieval(domain_id="stress_domain") as r:
                    r.set_attribute(f"stress.payload_{idx}", payload)
            wf.set_attribute("stress.root_summary", "Completed stress batch")

        raw_output = stream.getvalue()
        lines = [line for line in raw_output.split("\n") if line.strip()]

        # Must have exactly 1 JSON line per finished span (5 retrieval + 1 workflow = 6 spans)
        assert len(lines) == 6

        # Every single line must parse cleanly into valid JSON
        for line_num, line in enumerate(lines, start=1):
            assert "\n" not in line
            assert "\r" not in line
            parsed = json.loads(line)
            assert isinstance(parsed, dict), f"Line {line_num} failed to parse into dict"
            assert "trace_id" in parsed
            assert "span_id" in parsed
            assert "name" in parsed

    def test_structured_json_schema_completeness(
        self, isolated_tracer: ScoutTelemetryTracer
    ) -> None:
        stream = io.StringIO()
        exporter = StructuredJsonSpanExporter(stream=stream)

        processor_module = pytest.importorskip("opentelemetry.sdk.trace.export")
        isolated_tracer.provider.add_span_processor(
            processor_module.SimpleSpanProcessor(exporter)
        )

        with (
            isolated_tracer.trace_workflow(span_name="scout.workflow") as wf,
            isolated_tracer.trace_retrieval(domain_id="trading_code_scout") as r,
        ):
            r.record_retrieval(doc_chunks_count=5, ast_symbols_count=10)
            wf.set_status(trace.StatusCode.OK)

        logs = exporter.get_json_logs()
        assert len(logs) == 2

        child_log = next(entry for entry in logs if entry["name"] == "scout.retrieval")
        root_log = next(entry for entry in logs if entry["name"] == "scout.workflow")

        # Field validation
        for entry in (child_log, root_log):
            assert len(entry["trace_id"]) == 32
            int(entry["trace_id"], 16)  # Valid hex
            assert len(entry["span_id"]) == 16
            int(entry["span_id"], 16)  # Valid hex
            assert entry["status"] == "OK"
            assert entry["status_description"] is None
            assert isinstance(entry["duration_ms"], float)
            assert entry["duration_ms"] >= 0.0
            assert isinstance(entry["attributes"], dict)

        assert root_log["parent_span_id"] is None
        assert child_log["parent_span_id"] == root_log["span_id"]

    def test_exporter_resilience_to_broken_stream_io_error(self) -> None:
        class FailingStream:
            def write(self, _content: str) -> int:
                raise OSError("Simulated disk write failure: No space left on device")

            def flush(self) -> None:
                pass

        failing_stream = FailingStream()
        exporter = StructuredJsonSpanExporter(stream=failing_stream)

        mock_span = MagicMock()
        mock_span.context.trace_id = 0x4BF92F3577B34DA6A3CE929D0E0E4736
        mock_span.context.span_id = 0x00F067AA0BA902B7
        mock_span.parent = None
        mock_span.start_time = 1_000_000
        mock_span.end_time = 2_000_000
        mock_span.name = "test.failing_stream_span"
        mock_span.status.status_code.name = "OK"
        mock_span.status.description = None
        mock_span.attributes = {"test.key": "test.value"}

        # Export must swallow the stream exception, record to buffer, and return SUCCESS
        result = exporter.export([mock_span])
        assert result.name == "SUCCESS"
        assert len(exporter.get_json_logs()) == 1
        assert exporter.get_json_logs()[0]["name"] == "test.failing_stream_span"

    def test_exporter_buffer_lifecycle_and_clear(
        self, isolated_tracer: ScoutTelemetryTracer
    ) -> None:
        stream = io.StringIO()
        exporter = StructuredJsonSpanExporter(stream=stream)

        processor_module = pytest.importorskip("opentelemetry.sdk.trace.export")
        isolated_tracer.provider.add_span_processor(
            processor_module.SimpleSpanProcessor(exporter)
        )

        with isolated_tracer.trace_workflow(span_name="span_batch_1"):
            pass

        assert len(exporter.get_json_logs()) == 1
        exporter.clear()
        assert len(exporter.get_json_logs()) == 0

        with isolated_tracer.trace_workflow(span_name="span_batch_2"):
            pass

        assert len(exporter.get_json_logs()) == 1
        assert exporter.get_json_logs()[0]["name"] == "span_batch_2"
        exporter.shutdown()
        assert len(exporter.get_json_logs()) == 0


# =============================================================================
# 4. W3C Context Propagation Boundary Stressing
# =============================================================================


class TestW3CContextPropagationBoundaryStressing:
    """Stress tests W3C TraceContext extraction, injection, and corruption fallbacks."""

    def test_valid_traceparent_injection_and_extraction_roundtrip(
        self, isolated_tracer: ScoutTelemetryTracer
    ) -> None:
        tracer = isolated_tracer

        with tracer.tracer.start_as_current_span("root_producer") as root:
            carrier = inject_trace_context()
            assert "traceparent" in carrier
            traceparent = carrier["traceparent"]
            assert traceparent.startswith("00-")

            expected_trace_hex = trace.format_trace_id(root.context.trace_id)
            expected_span_hex = trace.format_span_id(root.context.span_id)
            assert f"00-{expected_trace_hex}-{expected_span_hex}-" in traceparent

        # Extract context from carrier in independent consumer context
        extracted_ctx = extract_trace_context(carrier)
        with tracer.tracer.start_as_current_span("child_consumer", context=extracted_ctx) as child:
            child.set_attribute("consumer.status", "received")

        child_spans = tracer.find_spans_by_name("child_consumer")
        assert len(child_spans) == 1
        assert child_spans[0].parent is not None
        assert trace.format_trace_id(child_spans[0].context.trace_id) == expected_trace_hex
        assert trace.format_span_id(child_spans[0].parent.span_id) == expected_span_hex

    def test_carrier_injection_edge_cases(
        self, isolated_tracer: ScoutTelemetryTracer
    ) -> None:
        tracer = isolated_tracer

        with tracer.tracer.start_as_current_span("test_carrier_span") as span:
            # 1. Injection into None creates new carrier
            c1 = inject_trace_context(None)
            assert isinstance(c1, dict)
            assert "traceparent" in c1

            # 2. Injection preserves pre-existing headers
            existing_carrier = {
                "Authorization": "Bearer signed_jwt_token",
                "X-Principal-User-Id": "trader_99",
            }
            c2 = inject_trace_context(existing_carrier)
            assert c2["Authorization"] == "Bearer signed_jwt_token"
            assert c2["X-Principal-User-Id"] == "trader_99"
            assert "traceparent" in c2
            assert trace.format_span_id(span.context.span_id) in c2["traceparent"]

    @pytest.mark.parametrize(
        "corrupted_header",
        [
            "",
            "not-a-traceparent",
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",  # all zeros trace_id
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",  # all zeros span_id
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",  # forbidden version ff
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",  # truncated, missing flags
            "00-12345678-00f067aa0ba902b7-01",  # short trace_id
            "00-4bf92f3577b34da6a3ce929d0e0e47361234-00f067aa0ba902b7-01",  # oversized trace_id
            "00-4bf92f3577b34da6a3ce929d0e0e47zz-00f067aa0ba902b7-01",  # non-hex chars
            "00_4bf92f3577b34da6a3ce929d0e0e4736_00f067aa0ba902b7_01",  # wrong delimiter
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-trailing",  # extra segment
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01\x00",  # null byte
        ],
    )
    def test_corrupted_traceparent_graceful_fallback_to_new_root(
        self,
        isolated_tracer: ScoutTelemetryTracer,
        corrupted_header: str,
    ) -> None:
        tracer = isolated_tracer
        carrier = {"traceparent": corrupted_header}

        # Must not raise an unhandled exception during extraction
        extracted_ctx = extract_trace_context(carrier)

        # Must gracefully fall back to a fresh root span
        with tracer.tracer.start_as_current_span(
            "fallback_span", context=extracted_ctx
        ) as s:
            assert s.parent is None, f"Expected None parent for '{corrupted_header}'"
            assert s.context.trace_id != 0, f"Expected non-zero trace ID for '{corrupted_header}'"
            trace_id_hex = trace.format_trace_id(s.context.trace_id)
            assert len(trace_id_hex) == 32
            int(trace_id_hex, 16)

    def test_empty_carrier_graceful_fallback(
        self, isolated_tracer: ScoutTelemetryTracer
    ) -> None:
        tracer = isolated_tracer
        carrier: dict[str, str] = {}
        extracted_ctx = extract_trace_context(carrier)

        with tracer.tracer.start_as_current_span(
            "fallback_empty_carrier", context=extracted_ctx
        ) as s:
            assert s.parent is None
            assert s.context.trace_id != 0

    def test_multithreaded_context_propagation_isolation(
        self, isolated_tracer: ScoutTelemetryTracer
    ) -> None:
        tracer = isolated_tracer

        def _worker_task(idx: int) -> bool:
            trace_id_hex = f"{idx:032x}"
            span_id_hex = f"{idx:016x}"
            carrier = {"traceparent": f"00-{trace_id_hex}-{span_id_hex}-01"}
            extracted = extract_trace_context(carrier)

            with tracer.tracer.start_as_current_span(f"worker_{idx}", context=extracted) as s:
                extracted_parent = trace.format_span_id(s.parent.span_id)  # type: ignore[union-attr]
                extracted_trace = trace.format_trace_id(s.context.trace_id)
                assert extracted_parent == span_id_hex
                assert extracted_trace == trace_id_hex
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_worker_task, i) for i in range(1, 17)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 16
        assert all(results)
