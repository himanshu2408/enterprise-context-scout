"""OpenTelemetry GenAI Semantic Instrumentation and Distributed Tracer (Feature F24).

Provides ScoutTelemetryTracer with:
1. Standardized scout.* attribute namespace conforming to production requirements,
   with dual-emission backward compatibility for context_scout.* attributes.
2. 4-Phase span context managers: trace_retrieval, trace_slicing, trace_gating, trace_execution.
3. Deterministic abstention zero-spend invariant enforcement:
   scout.is_abstention=True, scout.paid_tokens_spent=0, scout.cost_usd=0.0.
4. Exporters: InMemorySpanExporter (deterministic offline assertions),
   StructuredJsonSpanExporter (single-line JSON logs), and configurable OTLP support.
5. W3C TraceContext (traceparent) carrier injection and extraction helpers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, ClassVar, Final
from uuid import UUID

import opentelemetry.context as otel_context
import opentelemetry.trace as trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

try:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
except ImportError:
    try:
        from opentelemetry.sdk.trace.export import (  # type: ignore[attr-defined,no-redef]
            InMemorySpanExporter,
        )
    except ImportError:  # pragma: no cover

        class InMemorySpanExporter(SpanExporter):  # type: ignore[no-redef]
            def __init__(self) -> None:
                self._spans: list[ReadableSpan] = []

            def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
                self._spans.extend(spans)
                return SpanExportResult.SUCCESS

            def get_finished_spans(self) -> tuple[ReadableSpan, ...]:
                return tuple(self._spans)

            def clear(self) -> None:
                self._spans.clear()

            def shutdown(self) -> None:
                self.clear()


logger = logging.getLogger(__name__)

# =============================================================================
# Standardized Scout OpenTelemetry Attributes (M5 F24 Specification)
# =============================================================================
ATTR_SCOUT_TRACE_ID: Final[str] = "scout.trace_id"
ATTR_SCOUT_DOMAIN_ID: Final[str] = "scout.domain_id"
ATTR_SCOUT_PRINCIPAL_ID: Final[str] = "scout.principal_id"
ATTR_SCOUT_PHASE: Final[str] = "scout.phase"
ATTR_SCOUT_MODEL_ID: Final[str] = "scout.model_id"
ATTR_SCOUT_PAID_TOKENS_SPENT: Final[str] = "scout.paid_tokens_spent"
ATTR_SCOUT_IS_ABSTENTION: Final[str] = "scout.is_abstention"
ATTR_SCOUT_LATENCY_MS: Final[str] = "scout.latency_ms"
ATTR_SCOUT_COST_USD: Final[str] = "scout.cost_usd"

# Phase-Specific Extended Attributes
ATTR_SCOUT_PREFLIGHT_PASSED: Final[str] = "scout.preflight_passed"
ATTR_SCOUT_NORMALIZED_RRF_SCORE: Final[str] = "scout.normalized_rrf_score"
ATTR_SCOUT_CITATIONS_VERIFIED: Final[str] = "scout.citations_verified"
ATTR_SCOUT_FAILURE_CHECK: Final[str] = "scout.failure_check"
ATTR_SCOUT_MISSING_ITEMS_COUNT: Final[str] = "scout.missing_items_count"
ATTR_SCOUT_REJECTION_REASON: Final[str] = "scout.rejection_reason"
ATTR_SCOUT_TOKENS_SAVED: Final[str] = "scout.tokens_saved"
ATTR_SCOUT_ESTIMATED_USD_SAVED: Final[str] = "scout.estimated_usd_saved"
ATTR_SCOUT_ZERO_TOUCH_SUFFICIENT: Final[str] = "scout.zero_touch_sufficient"
ATTR_SCOUT_VERIFICATION_PASSED: Final[str] = "scout.verification_passed"
ATTR_SCOUT_PATCH_APPLIED: Final[str] = "scout.patch_applied"
ATTR_SCOUT_AUTO_ATTACHED_AST_COUNT: Final[str] = "scout.auto_attached_ast_count"
ATTR_SCOUT_PRINCIPAL_ROLE: Final[str] = "scout.principal_role"

# Pipeline Phase Constants
PHASE_RETRIEVAL: Final[str] = "retrieval"
PHASE_SLICING: Final[str] = "slicing"
PHASE_GATING: Final[str] = "gating"
PHASE_EXECUTION: Final[str] = "execution"
PHASE_WORKFLOW: Final[str] = "workflow"

# OpenTelemetry Standard GenAI Semantic Conventions
GEN_AI_SYSTEM: Final[str] = "gen_ai.system"
GEN_AI_REQUEST_MODEL: Final[str] = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS: Final[str] = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS: Final[str] = "gen_ai.usage.output_tokens"
GEN_AI_OPERATION_NAME: Final[str] = "gen_ai.operation.name"

# Legacy Compatibility Aliases (context_scout.*)
ATTR_TOKENS_SAVED: Final[str] = "context_scout.tokens_saved"
ATTR_ESTIMATED_USD_SAVED: Final[str] = "context_scout.estimated_usd_saved"
ATTR_DOMAIN_ID: Final[str] = "context_scout.domain_id"
ATTR_TRACE_ID: Final[str] = "context_scout.trace_id"
ATTR_PREFLIGHT_PASSED: Final[str] = "context_scout.preflight_passed"
ATTR_ZERO_TOUCH_SUFFICIENT: Final[str] = "context_scout.zero_touch_sufficient"
ATTR_NORMALIZED_RRF_SCORE: Final[str] = "context_scout.normalized_rrf_score"
ATTR_CITATIONS_VERIFIED: Final[str] = "context_scout.citations_verified"
ATTR_AUTO_ATTACHED_AST_COUNT: Final[str] = "context_scout.auto_attached_ast_count"
ATTR_PRINCIPAL_ID: Final[str] = "context_scout.principal_id"
ATTR_PRINCIPAL_ROLE: Final[str] = "context_scout.principal_role"


# =============================================================================
# W3C Trace Context Propagation Helpers
# =============================================================================

_W3C_PROPAGATOR = TraceContextTextMapPropagator()


def inject_trace_context(
    carrier: dict[str, str] | None = None,
    ctx: otel_context.Context | None = None,
) -> dict[str, str]:
    """Injects W3C TraceContext ('traceparent' header) into a carrier dictionary."""
    carrier_dict = {} if carrier is None else carrier
    active_ctx = ctx or otel_context.get_current()
    _W3C_PROPAGATOR.inject(carrier_dict, context=active_ctx)
    return carrier_dict


def extract_trace_context(carrier: dict[str, str]) -> otel_context.Context:
    """Extracts active OpenTelemetry Context from a W3C TraceContext carrier dictionary."""
    return _W3C_PROPAGATOR.extract(carrier)


# =============================================================================
# Structured JSON Span Exporter (Single-Line JSONL Log Output)
# =============================================================================


class StructuredJsonSpanExporter(SpanExporter):
    """Formats OpenTelemetry spans as single-line structured JSON logs for audit logging."""

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream
        self._exported_logs: list[dict[str, Any]] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            trace_id_hex = trace.format_trace_id(span.context.trace_id)
            span_id_hex = trace.format_span_id(span.context.span_id)
            parent_id_hex = trace.format_span_id(span.parent.span_id) if span.parent else None

            # Convert nanoseconds to milliseconds
            start_ms = (span.start_time or 0) / 1_000_000.0
            end_ms = (span.end_time or 0) / 1_000_000.0
            duration_ms = max(0.0, round(end_ms - start_ms, 3))

            from context_scout.security.betrvg_privacy import BetrVGPrivacyGuard

            raw_attrs = dict(span.attributes or {})
            raw_tid = raw_attrs.get("scout.tenant_id")
            parsed_tid: UUID | None = None
            if raw_tid is not None:
                try:
                    parsed_tid = UUID(str(raw_tid))
                except ValueError:
                    parsed_tid = None
            guard = BetrVGPrivacyGuard()
            sanitized_attrs = guard.sanitize_telemetry_attributes(raw_attrs, tenant_id=parsed_tid)
            entry: dict[str, Any] = {
                "trace_id": trace_id_hex,
                "span_id": span_id_hex,
                "parent_span_id": parent_id_hex,
                "name": span.name,
                "status": span.status.status_code.name,
                "status_description": span.status.description,
                "duration_ms": duration_ms,
                "attributes": sanitized_attrs,
            }
            self._exported_logs.append(entry)

            if self.stream is not None:
                try:
                    self.stream.write(json.dumps(entry) + "\n")
                    if hasattr(self.stream, "flush"):
                        self.stream.flush()
                except Exception as e:
                    logger.debug("Failed writing structured span JSON: %s", e)

        return SpanExportResult.SUCCESS

    def get_json_logs(self) -> list[dict[str, Any]]:
        """Returns the in-memory buffer of exported structured JSON logs."""
        return list(self._exported_logs)

    def clear(self) -> None:
        """Clears the in-memory buffer of exported structured JSON logs."""
        self._exported_logs.clear()

    def shutdown(self) -> None:
        self.clear()


# =============================================================================
# Base and Specialized Span Context Wrappers
# =============================================================================


class BaseScoutSpanContext:
    """Base span wrapper providing unified attribute setting and abstention invariants."""

    def __init__(self, span: trace.Span, phase: str = PHASE_WORKFLOW) -> None:
        self.span = span
        self.phase = phase
        self.span.set_attribute(ATTR_SCOUT_PHASE, phase)

    def set_attribute(self, key: str, value: Any) -> None:
        self.span.set_attribute(key, value)

    def set_status(self, status_code: trace.StatusCode, description: str | None = None) -> None:
        self.span.set_status(status_code, description=description)

    def record_abstention(self, diagnostic: Any) -> None:
        """Enforces the strict abstention zero-spend invariant across span context.

        Invariants:
        1. scout.is_abstention = True
        2. scout.paid_tokens_spent = 0
        3. scout.cost_usd = 0.0
        """
        reason = getattr(diagnostic, "reason", "Context insufficient; abstaining")
        failure_check = getattr(diagnostic, "failure_check", None) or "abstention"

        # Standard scout attributes
        self.span.set_attribute(ATTR_SCOUT_IS_ABSTENTION, True)
        self.span.set_attribute(ATTR_SCOUT_PAID_TOKENS_SPENT, 0)
        self.span.set_attribute(ATTR_SCOUT_COST_USD, 0.0)
        self.span.set_attribute(ATTR_SCOUT_FAILURE_CHECK, str(failure_check))
        self.span.set_attribute(ATTR_SCOUT_REJECTION_REASON, str(reason))
        self.span.set_attribute(ATTR_SCOUT_PREFLIGHT_PASSED, False)

        # Legacy backward compatibility attributes
        self.span.set_attribute(ATTR_PREFLIGHT_PASSED, False)
        self.span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, 0)
        self.span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, 0)
        self.span.set_attribute("context_scout.rejection_reason", str(reason))

        if hasattr(diagnostic, "trace_id"):
            self.span.set_attribute(ATTR_SCOUT_TRACE_ID, str(diagnostic.trace_id))
            self.span.set_attribute(ATTR_TRACE_ID, str(diagnostic.trace_id))
        if hasattr(diagnostic, "domain_id"):
            self.span.set_attribute(ATTR_SCOUT_DOMAIN_ID, str(diagnostic.domain_id))
            self.span.set_attribute(ATTR_DOMAIN_ID, str(diagnostic.domain_id))
        if hasattr(diagnostic, "missing_items"):
            count = len(diagnostic.missing_items)
            self.span.set_attribute(ATTR_SCOUT_MISSING_ITEMS_COUNT, count)

        self.span.set_status(trace.StatusCode.ERROR, description=str(reason))


class RetrievalSpanContext(BaseScoutSpanContext):
    """Span context wrapper for Phase 1: Context Retrieval (AST + Docs)."""

    def __init__(self, span: trace.Span) -> None:
        super().__init__(span, phase=PHASE_RETRIEVAL)

    def record_retrieval(
        self,
        doc_chunks_count: int,
        ast_symbols_count: int = 0,
        query: str | None = None,
    ) -> None:
        self.span.set_attribute("scout.retrieval.doc_chunks_count", doc_chunks_count)
        self.span.set_attribute("scout.retrieval.ast_symbols_count", ast_symbols_count)
        if query:
            from context_scout.security.betrvg_privacy import is_betrvg_enabled

            self.span.set_attribute(
                "scout.retrieval.query_snippet",
                "[REDACTED_BETRVG]" if is_betrvg_enabled() else query[:100],
            )
        self.span.set_status(trace.StatusCode.OK)


class SlicingSpanContext(BaseScoutSpanContext):
    """Span context wrapper for Phase 2: Deterministic Slicing."""

    def __init__(self, span: trace.Span) -> None:
        super().__init__(span, phase=PHASE_SLICING)

    def record_slicing(
        self,
        file_path: str,
        lines_sliced: int,
        syntax_valid: bool = True,
    ) -> None:
        self.span.set_attribute("scout.slicing.file_path", file_path)
        self.span.set_attribute("scout.slicing.lines_count", lines_sliced)
        self.span.set_attribute("scout.slicing.syntax_valid", syntax_valid)
        self.span.set_status(trace.StatusCode.OK if syntax_valid else trace.StatusCode.ERROR)


class GatingSpanContext(BaseScoutSpanContext):
    """Span context wrapper for Phase 3: Deterministic Preflight & Verification Gating."""

    def __init__(self, span: trace.Span) -> None:
        super().__init__(span, phase=PHASE_GATING)

    def record_report(self, report: Any) -> None:
        passed = bool(getattr(report, "passed", False))
        rrf = float(getattr(report, "normalized_rrf_score", 0.0))
        cites_ok = bool(getattr(report, "citations_verified_100_pct", False))
        auto_ast = int(getattr(report, "auto_attached_ast_slices_count", 0))

        # Primary scout.* attributes
        self.span.set_attribute(ATTR_SCOUT_PREFLIGHT_PASSED, passed)
        self.span.set_attribute(ATTR_SCOUT_NORMALIZED_RRF_SCORE, rrf)
        self.span.set_attribute(ATTR_SCOUT_CITATIONS_VERIFIED, cites_ok)
        self.span.set_attribute(ATTR_SCOUT_AUTO_ATTACHED_AST_COUNT, auto_ast)

        # Legacy context_scout.* attributes
        self.span.set_attribute(ATTR_PREFLIGHT_PASSED, passed)
        self.span.set_attribute(ATTR_NORMALIZED_RRF_SCORE, rrf)
        self.span.set_attribute(ATTR_CITATIONS_VERIFIED, cites_ok)
        self.span.set_attribute(ATTR_AUTO_ATTACHED_AST_COUNT, auto_ast)

        if passed:
            self.span.set_status(trace.StatusCode.OK)
        else:
            fb = getattr(report, "bounce_back_feedback", "Preflight gate checks failed")
            self.span.set_status(trace.StatusCode.ERROR, description=str(fb))


# Backward compatibility alias
PreflightSpanContext = GatingSpanContext


class ExecutionSpanContext(BaseScoutSpanContext):
    """Span context wrapper for Phase 4: Paid Frontier Execution & Sandbox."""

    def __init__(self, span: trace.Span) -> None:
        super().__init__(span, phase=PHASE_EXECUTION)

    def record_result(self, result: Any, telemetry: Any) -> None:
        tokens_saved = int(getattr(telemetry, "paid_tokens_saved", 0))
        usd_saved = float(getattr(telemetry, "estimated_usd_saved", 0.0))
        zero_touch = bool(getattr(telemetry, "zero_touch_sufficient", False))
        pack_sent = int(getattr(telemetry, "context_pack_tokens_sent", 0))
        out_tokens = int(getattr(result, "token_count", 0))
        preflight_ok = bool(getattr(telemetry, "preflight_passed_first_try", True))

        # Primary scout.* attributes
        self.span.set_attribute(ATTR_SCOUT_TOKENS_SAVED, tokens_saved)
        self.span.set_attribute(ATTR_SCOUT_ESTIMATED_USD_SAVED, usd_saved)
        self.span.set_attribute(ATTR_SCOUT_ZERO_TOUCH_SUFFICIENT, zero_touch)
        self.span.set_attribute(ATTR_SCOUT_PAID_TOKENS_SPENT, pack_sent + out_tokens)
        self.span.set_attribute(ATTR_SCOUT_PREFLIGHT_PASSED, preflight_ok)
        self.span.set_attribute(ATTR_SCOUT_IS_ABSTENTION, False)

        # Legacy context_scout.* attributes
        self.span.set_attribute(ATTR_TOKENS_SAVED, tokens_saved)
        self.span.set_attribute(ATTR_ESTIMATED_USD_SAVED, usd_saved)
        self.span.set_attribute(ATTR_ZERO_TOUCH_SUFFICIENT, zero_touch)
        self.span.set_attribute(ATTR_PREFLIGHT_PASSED, preflight_ok)

        # GenAI semantic conventions
        self.span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, pack_sent)
        self.span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, out_tokens)

        verif_passed = getattr(result, "verification_passed", getattr(result, "success", True))
        self.span.set_attribute(ATTR_SCOUT_VERIFICATION_PASSED, bool(verif_passed))
        self.span.set_attribute(
            ATTR_SCOUT_PATCH_APPLIED, bool(getattr(result, "patch_applied", True))
        )

        if getattr(result, "success", True):
            self.span.set_status(trace.StatusCode.OK)
        else:
            err = getattr(result, "error_message", "Execution failed")
            self.span.set_status(trace.StatusCode.ERROR, description=str(err))


# Backward compatibility alias
ExecutorSpanContext = ExecutionSpanContext


class WorkflowSpanContext(BaseScoutSpanContext):
    """Wrapper around root workflow span providing end-to-end telemetry recording."""

    def __init__(self, span: trace.Span) -> None:
        super().__init__(span, phase=PHASE_WORKFLOW)

    def record_pack(self, pack: Any) -> None:
        """Records attributes from ImplementationContextPack or InsufficientContextDiagnostic."""
        if hasattr(pack, "preflight_report"):
            report = pack.preflight_report
            passed = bool(report.passed)
            rrf = float(report.normalized_rrf_score)
            cites = bool(report.citations_verified_100_pct)
            ast_count = int(report.auto_attached_ast_slices_count)
            in_tokens = int(getattr(pack, "local_tokens_consumed", 0))
            out_tokens = int(getattr(pack, "pack_token_count", 0))

            # Primary scout.* attributes
            self.span.set_attribute(ATTR_SCOUT_PREFLIGHT_PASSED, passed)
            self.span.set_attribute(ATTR_SCOUT_NORMALIZED_RRF_SCORE, rrf)
            self.span.set_attribute(ATTR_SCOUT_CITATIONS_VERIFIED, cites)
            self.span.set_attribute(ATTR_SCOUT_AUTO_ATTACHED_AST_COUNT, ast_count)
            self.span.set_attribute(ATTR_SCOUT_IS_ABSTENTION, False)

            # Legacy context_scout.* attributes
            self.span.set_attribute(ATTR_PREFLIGHT_PASSED, passed)
            self.span.set_attribute(ATTR_NORMALIZED_RRF_SCORE, rrf)
            self.span.set_attribute(ATTR_CITATIONS_VERIFIED, cites)
            self.span.set_attribute(ATTR_AUTO_ATTACHED_AST_COUNT, ast_count)
            self.span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, in_tokens)
            self.span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, out_tokens)

            if hasattr(pack, "trace_id"):
                self.span.set_attribute(ATTR_SCOUT_TRACE_ID, str(pack.trace_id))
                self.span.set_attribute(ATTR_TRACE_ID, str(pack.trace_id))
            if hasattr(pack, "domain_id"):
                self.span.set_attribute(ATTR_SCOUT_DOMAIN_ID, str(pack.domain_id))
                self.span.set_attribute(ATTR_DOMAIN_ID, str(pack.domain_id))

            self.span.set_status(trace.StatusCode.OK)
        elif hasattr(pack, "reason"):
            self.record_abstention(pack)


# =============================================================================
# Scout OpenTelemetry Tracer
# =============================================================================


class ScoutTelemetryTracer:
    """Enterprise OpenTelemetry tracer managing multiple exporters and 4-phase spans."""

    _instance: ClassVar[ScoutTelemetryTracer | None] = None

    def __init__(
        self,
        service_name: str = "enterprise-context-scout",
        enable_console: bool = False,
        enable_json_logger: bool = True,
    ) -> None:
        self.service_name = service_name
        self.exporter = InMemorySpanExporter()
        self.json_exporter = StructuredJsonSpanExporter()
        self.provider = TracerProvider()

        # In-memory exporter for fast, deterministic CI assertions
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))

        # Structured JSON exporter for single-line JSONL audit logs
        if enable_json_logger:
            self.provider.add_span_processor(SimpleSpanProcessor(self.json_exporter))

        # Optional console exporter
        if enable_console:
            self.provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        self.tracer: trace.Tracer = self.provider.get_tracer(service_name)

    @classmethod
    def get_instance(cls, service_name: str = "enterprise-context-scout") -> ScoutTelemetryTracer:
        """Retrieves or initializes the global singleton tracer."""
        if cls._instance is None:
            cls._instance = cls(service_name=service_name)
        return cls._instance

    @classmethod
    def reset_instance(cls, service_name: str = "enterprise-context-scout") -> ScoutTelemetryTracer:
        """Resets the singleton tracer instance for isolated test execution."""
        cls._instance = cls(service_name=service_name)
        return cls._instance

    @contextmanager
    def trace_workflow(
        self,
        span_name: str = "scout.workflow",
        domain_id: str | None = None,
        trace_id: str | None = None,
        model_id: str = "gemma-4-moe",
        system: str = "gemma",
    ) -> Iterator[WorkflowSpanContext]:
        """Context manager tracing a root scout workflow DAG invocation."""
        with self.tracer.start_as_current_span(span_name) as span:
            ctx = WorkflowSpanContext(span)
            span.set_attribute(ATTR_SCOUT_PHASE, PHASE_WORKFLOW)
            if system:
                span.set_attribute(GEN_AI_SYSTEM, system)
            if model_id:
                span.set_attribute(GEN_AI_REQUEST_MODEL, model_id)
                span.set_attribute(ATTR_SCOUT_MODEL_ID, model_id)
            if domain_id:
                span.set_attribute(ATTR_SCOUT_DOMAIN_ID, domain_id)
                span.set_attribute(ATTR_DOMAIN_ID, domain_id)
            if trace_id:
                span.set_attribute(ATTR_SCOUT_TRACE_ID, str(trace_id))
                span.set_attribute(ATTR_TRACE_ID, str(trace_id))
            span.set_attribute(GEN_AI_OPERATION_NAME, "workflow")
            try:
                yield ctx
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, description=str(exc))
                raise

    @contextmanager
    def trace_retrieval(
        self,
        span_name: str = "scout.retrieval",
        domain_id: str | None = None,
        trace_id: str | None = None,
        query: str | None = None,
    ) -> Iterator[RetrievalSpanContext]:
        """Context manager tracing Phase 1: Context Retrieval."""
        with self.tracer.start_as_current_span(span_name) as span:
            ctx = RetrievalSpanContext(span)
            if domain_id:
                span.set_attribute(ATTR_SCOUT_DOMAIN_ID, domain_id)
                span.set_attribute(ATTR_DOMAIN_ID, domain_id)
            if trace_id:
                span.set_attribute(ATTR_SCOUT_TRACE_ID, str(trace_id))
                span.set_attribute(ATTR_TRACE_ID, str(trace_id))
            if query:
                from context_scout.security.betrvg_privacy import is_betrvg_enabled

                span.set_attribute(
                    "scout.retrieval.query",
                    "[REDACTED_BETRVG]" if is_betrvg_enabled() else query[:200],
                )
            span.set_attribute(GEN_AI_OPERATION_NAME, "retrieval")
            try:
                yield ctx
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, description=str(exc))
                raise

    @contextmanager
    def trace_slicing(
        self,
        span_name: str = "scout.slicing",
        domain_id: str | None = None,
        file_path: str | None = None,
    ) -> Iterator[SlicingSpanContext]:
        """Context manager tracing Phase 2: Code Slicing."""
        with self.tracer.start_as_current_span(span_name) as span:
            ctx = SlicingSpanContext(span)
            if domain_id:
                span.set_attribute(ATTR_SCOUT_DOMAIN_ID, domain_id)
                span.set_attribute(ATTR_DOMAIN_ID, domain_id)
            if file_path:
                span.set_attribute("scout.slicing.file_path", file_path)
            span.set_attribute(GEN_AI_OPERATION_NAME, "slicing")
            try:
                yield ctx
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, description=str(exc))
                raise

    @contextmanager
    def trace_gating(
        self,
        span_name: str = "scout.preflight_gate",
        domain_id: str | None = None,
        prompt: str | None = None,
    ) -> Iterator[GatingSpanContext]:
        """Context manager tracing Phase 3: Preflight & Verification Gating."""
        with self.tracer.start_as_current_span(span_name) as span:
            ctx = GatingSpanContext(span)
            if domain_id:
                span.set_attribute(ATTR_SCOUT_DOMAIN_ID, domain_id)
                span.set_attribute(ATTR_DOMAIN_ID, domain_id)
            if prompt:
                from context_scout.security.betrvg_privacy import is_betrvg_enabled

                span.set_attribute(
                    "scout.gating.prompt_snippet",
                    "[REDACTED_BETRVG]" if is_betrvg_enabled() else prompt[:100],
                )
            span.set_attribute(GEN_AI_OPERATION_NAME, "preflight_gate")
            try:
                yield ctx
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, description=str(exc))
                raise

    # Backwards compatible alias for trace_gating
    trace_preflight = trace_gating

    @contextmanager
    def trace_execution(
        self,
        span_name: str = "scout.executor.handoff",
        pack: Any | None = None,
        principal: Any | None = None,
        model_id: str = "gemini-3.8-flash",
        system: str = "gemini",
        domain_id: str | None = None,
    ) -> Iterator[ExecutionSpanContext]:
        """Context manager tracing Phase 4: Paid Frontier Execution."""
        with self.tracer.start_as_current_span(span_name) as span:
            ctx = ExecutionSpanContext(span)
            if system:
                span.set_attribute(GEN_AI_SYSTEM, system)
            if model_id:
                span.set_attribute(GEN_AI_REQUEST_MODEL, model_id)
                span.set_attribute(ATTR_SCOUT_MODEL_ID, model_id)
            if domain_id:
                span.set_attribute(ATTR_SCOUT_DOMAIN_ID, domain_id)
                span.set_attribute(ATTR_DOMAIN_ID, domain_id)
            if pack is not None:
                if hasattr(pack, "domain_id"):
                    span.set_attribute(ATTR_SCOUT_DOMAIN_ID, str(pack.domain_id))
                    span.set_attribute(ATTR_DOMAIN_ID, str(pack.domain_id))
                if hasattr(pack, "trace_id"):
                    span.set_attribute(ATTR_SCOUT_TRACE_ID, str(pack.trace_id))
                    span.set_attribute(ATTR_TRACE_ID, str(pack.trace_id))
            if principal is not None:
                if hasattr(principal, "user_id"):
                    from context_scout.schemas.contracts import DEFAULT_TENANT_ID
                    from context_scout.security.betrvg_privacy import pseudonymize_actor_if_enabled

                    p_tid = getattr(principal, "tenant_id", DEFAULT_TENANT_ID)
                    safe_uid = pseudonymize_actor_if_enabled(str(principal.user_id), p_tid)
                    span.set_attribute(ATTR_SCOUT_PRINCIPAL_ID, safe_uid)
                    span.set_attribute(ATTR_PRINCIPAL_ID, safe_uid)
                if hasattr(principal, "role"):
                    role_val = (
                        principal.role.value
                        if hasattr(principal.role, "value")
                        else str(principal.role)
                    )
                    span.set_attribute(ATTR_SCOUT_PRINCIPAL_ROLE, str(role_val))
                    span.set_attribute(ATTR_PRINCIPAL_ROLE, str(role_val))
            span.set_attribute(GEN_AI_OPERATION_NAME, "implementation_executor")
            try:
                yield ctx
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, description=str(exc))
                raise

    # Backwards compatible alias for trace_execution
    trace_executor = trace_execution

    def get_finished_spans(self) -> tuple[ReadableSpan, ...]:
        """Returns all completed spans captured by the in-memory exporter."""
        return tuple(self.exporter.get_finished_spans())

    def get_json_logs(self) -> list[dict[str, Any]]:
        """Returns all completed structured JSON log records."""
        return self.json_exporter.get_json_logs()

    def clear_spans(self) -> None:
        """Clears all accumulated spans across in-memory and JSON exporters."""
        self.exporter.clear()
        self.json_exporter.clear()

    def find_spans_by_name(self, name: str) -> list[ReadableSpan]:
        """Finds all finished spans matching the given name."""
        return [s for s in self.get_finished_spans() if s.name == name]

    def find_spans_by_attribute(self, key: str, value: Any) -> list[ReadableSpan]:
        """Finds all finished spans containing key=value in attributes."""
        matches: list[ReadableSpan] = []
        for s in self.get_finished_spans():
            if s.attributes and s.attributes.get(key) == value:
                matches.append(s)
        return matches

    def reset(self) -> None:
        """Clears spans and resets in-memory exporter."""
        self.clear_spans()


def get_tracer(service_name: str = "enterprise-context-scout") -> ScoutTelemetryTracer:
    """Convenience getter for global ScoutTelemetryTracer singleton."""
    return ScoutTelemetryTracer.get_instance(service_name)


def reset_tracer(service_name: str = "enterprise-context-scout") -> ScoutTelemetryTracer:
    """Convenience reset for global ScoutTelemetryTracer singleton."""
    return ScoutTelemetryTracer.reset_instance(service_name)


__all__ = [
    "ATTR_AUTO_ATTACHED_AST_COUNT",
    "ATTR_CITATIONS_VERIFIED",
    "ATTR_DOMAIN_ID",
    "ATTR_ESTIMATED_USD_SAVED",
    "ATTR_NORMALIZED_RRF_SCORE",
    "ATTR_PREFLIGHT_PASSED",
    "ATTR_PRINCIPAL_ID",
    "ATTR_PRINCIPAL_ROLE",
    "ATTR_SCOUT_AUTO_ATTACHED_AST_COUNT",
    "ATTR_SCOUT_CITATIONS_VERIFIED",
    "ATTR_SCOUT_COST_USD",
    "ATTR_SCOUT_DOMAIN_ID",
    "ATTR_SCOUT_ESTIMATED_USD_SAVED",
    "ATTR_SCOUT_FAILURE_CHECK",
    "ATTR_SCOUT_IS_ABSTENTION",
    "ATTR_SCOUT_LATENCY_MS",
    "ATTR_SCOUT_MISSING_ITEMS_COUNT",
    "ATTR_SCOUT_MODEL_ID",
    "ATTR_SCOUT_NORMALIZED_RRF_SCORE",
    "ATTR_SCOUT_PAID_TOKENS_SPENT",
    "ATTR_SCOUT_PATCH_APPLIED",
    "ATTR_SCOUT_PHASE",
    "ATTR_SCOUT_PREFLIGHT_PASSED",
    "ATTR_SCOUT_PRINCIPAL_ID",
    "ATTR_SCOUT_PRINCIPAL_ROLE",
    "ATTR_SCOUT_REJECTION_REASON",
    "ATTR_SCOUT_TOKENS_SAVED",
    "ATTR_SCOUT_TRACE_ID",
    "ATTR_SCOUT_VERIFICATION_PASSED",
    "ATTR_SCOUT_ZERO_TOUCH_SUFFICIENT",
    "ATTR_TOKENS_SAVED",
    "ATTR_TRACE_ID",
    "ATTR_ZERO_TOUCH_SUFFICIENT",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_SYSTEM",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    "PHASE_EXECUTION",
    "PHASE_GATING",
    "PHASE_RETRIEVAL",
    "PHASE_SLICING",
    "PHASE_WORKFLOW",
    "BaseScoutSpanContext",
    "ExecutionSpanContext",
    "ExecutorSpanContext",
    "GatingSpanContext",
    "InMemorySpanExporter",
    "PreflightSpanContext",
    "RetrievalSpanContext",
    "ScoutTelemetryTracer",
    "SlicingSpanContext",
    "StructuredJsonSpanExporter",
    "WorkflowSpanContext",
    "extract_trace_context",
    "get_tracer",
    "inject_trace_context",
    "reset_tracer",
]
