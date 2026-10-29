"""OpenTelemetry GenAI Semantic Instrumentation and Distributed Tracer.

Provides ScoutTelemetryTracer with in-memory span collection for deterministic
CI/CD assertions and production OTLP/Console export capabilities.
Adheres strictly to OpenTelemetry GenAI Semantic Conventions (gen_ai.*) and
enterprise Context Scout operational metrics (context_scout.*).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, ClassVar, Final

import opentelemetry.trace as trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

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
        # Defensive fallback for minimal environments
        from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

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
# OpenTelemetry Standard GenAI Semantic Conventions
# =============================================================================
GEN_AI_SYSTEM: Final[str] = "gen_ai.system"
GEN_AI_REQUEST_MODEL: Final[str] = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS: Final[str] = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS: Final[str] = "gen_ai.usage.output_tokens"
GEN_AI_OPERATION_NAME: Final[str] = "gen_ai.operation.name"

# =============================================================================
# Context Scout Operational Domain Attributes
# =============================================================================
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


class WorkflowSpanContext:
    """Wrapper around active workflow span providing type-safe metric recording."""

    def __init__(self, span: trace.Span) -> None:
        self.span = span

    def set_attribute(self, key: str, value: Any) -> None:
        self.span.set_attribute(key, value)

    def set_status(
        self, status_code: trace.StatusCode, description: str | None = None
    ) -> None:
        self.span.set_status(status_code, description=description)

    def record_pack(self, pack: Any) -> None:
        """Records attributes from ImplementationContextPack or InsufficientContextDiagnostic."""
        if hasattr(pack, "preflight_report"):
            report = pack.preflight_report
            self.span.set_attribute(ATTR_PREFLIGHT_PASSED, bool(report.passed))
            self.span.set_attribute(
                ATTR_NORMALIZED_RRF_SCORE, float(report.normalized_rrf_score)
            )
            self.span.set_attribute(
                ATTR_CITATIONS_VERIFIED, bool(report.citations_verified_100_pct)
            )
            self.span.set_attribute(
                ATTR_AUTO_ATTACHED_AST_COUNT,
                int(report.auto_attached_ast_slices_count),
            )
            if hasattr(pack, "local_tokens_consumed"):
                self.span.set_attribute(
                    GEN_AI_USAGE_INPUT_TOKENS, int(pack.local_tokens_consumed)
                )
            if hasattr(pack, "pack_token_count"):
                self.span.set_attribute(
                    GEN_AI_USAGE_OUTPUT_TOKENS, int(pack.pack_token_count)
                )
            if hasattr(pack, "trace_id"):
                self.span.set_attribute(ATTR_TRACE_ID, str(pack.trace_id))
            if hasattr(pack, "domain_id"):
                self.span.set_attribute(ATTR_DOMAIN_ID, str(pack.domain_id))
            self.span.set_status(trace.StatusCode.OK)
        elif hasattr(pack, "reason"):
            self.span.set_attribute(ATTR_PREFLIGHT_PASSED, False)
            self.span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, 0)
            self.span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, 0)
            self.span.set_attribute("context_scout.rejection_reason", str(pack.reason))
            if hasattr(pack, "trace_id"):
                self.span.set_attribute(ATTR_TRACE_ID, str(pack.trace_id))
            if hasattr(pack, "domain_id"):
                self.span.set_attribute(ATTR_DOMAIN_ID, str(pack.domain_id))
            self.span.set_status(trace.StatusCode.ERROR, description=str(pack.reason))


class PreflightSpanContext:
    """Wrapper around active pre-flight gate span providing metric recording."""

    def __init__(self, span: trace.Span) -> None:
        self.span = span

    def set_attribute(self, key: str, value: Any) -> None:
        self.span.set_attribute(key, value)

    def set_status(
        self, status_code: trace.StatusCode, description: str | None = None
    ) -> None:
        self.span.set_status(status_code, description=description)

    def record_report(self, report: Any) -> None:
        """Records attributes from PreFlightValidationReport."""
        self.span.set_attribute(ATTR_PREFLIGHT_PASSED, bool(report.passed))
        self.span.set_attribute(
            ATTR_NORMALIZED_RRF_SCORE, float(report.normalized_rrf_score)
        )
        self.span.set_attribute(
            ATTR_CITATIONS_VERIFIED, bool(report.citations_verified_100_pct)
        )
        self.span.set_attribute(
            ATTR_AUTO_ATTACHED_AST_COUNT,
            int(report.auto_attached_ast_slices_count),
        )
        if report.passed:
            self.span.set_status(trace.StatusCode.OK)
        else:
            feedback = report.bounce_back_feedback or "PreFlight gate checks failed"
            self.span.set_status(trace.StatusCode.ERROR, description=str(feedback))


class ExecutorSpanContext:
    """Wrapper around active implementation executor span providing metric recording."""

    def __init__(self, span: trace.Span) -> None:
        self.span = span

    def set_attribute(self, key: str, value: Any) -> None:
        self.span.set_attribute(key, value)

    def set_status(
        self, status_code: trace.StatusCode, description: str | None = None
    ) -> None:
        self.span.set_status(status_code, description=description)

    def record_result(self, result: Any, telemetry: Any) -> None:
        """Records attributes from PatchExecutionResult and ExecutionTelemetryRecord."""
        self.span.set_attribute(ATTR_TOKENS_SAVED, int(telemetry.paid_tokens_saved))
        self.span.set_attribute(
            ATTR_ESTIMATED_USD_SAVED, float(telemetry.estimated_usd_saved)
        )
        self.span.set_attribute(
            ATTR_ZERO_TOUCH_SUFFICIENT, bool(telemetry.zero_touch_sufficient)
        )
        self.span.set_attribute(
            GEN_AI_USAGE_INPUT_TOKENS, int(telemetry.context_pack_tokens_sent)
        )
        output_tokens = getattr(result, "token_count", 0)
        self.span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, int(output_tokens))
        self.span.set_attribute(
            ATTR_PREFLIGHT_PASSED, bool(telemetry.preflight_passed_first_try)
        )

        if getattr(result, "success", True):
            self.span.set_status(trace.StatusCode.OK)
        else:
            err = getattr(result, "error_message", "Execution failed")
            self.span.set_status(trace.StatusCode.ERROR, description=str(err))


class ScoutTelemetryTracer:
    """Singleton/Instance OpenTelemetry tracer for Context Scout."""

    _instance: ClassVar[ScoutTelemetryTracer | None] = None

    def __init__(
        self,
        service_name: str = "enterprise-context-scout",
        enable_console: bool = False,
    ) -> None:
        self.service_name = service_name
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        if enable_console:
            self.provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        self.tracer: trace.Tracer = self.provider.get_tracer(service_name)

    @classmethod
    def get_instance(
        cls, service_name: str = "enterprise-context-scout"
    ) -> ScoutTelemetryTracer:
        """Retrieves or initializes the global singleton tracer."""
        if cls._instance is None:
            cls._instance = cls(service_name=service_name)
        return cls._instance

    @classmethod
    def reset_instance(
        cls, service_name: str = "enterprise-context-scout"
    ) -> ScoutTelemetryTracer:
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
        """Context manager tracing a scout workflow invocation."""
        with self.tracer.start_as_current_span(span_name) as span:
            ctx = WorkflowSpanContext(span)
            if system:
                span.set_attribute(GEN_AI_SYSTEM, system)
            if model_id:
                span.set_attribute(GEN_AI_REQUEST_MODEL, model_id)
            if domain_id:
                span.set_attribute(ATTR_DOMAIN_ID, domain_id)
            if trace_id:
                span.set_attribute(ATTR_TRACE_ID, str(trace_id))
            span.set_attribute(GEN_AI_OPERATION_NAME, "workflow")
            try:
                yield ctx
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, description=str(exc))
                raise

    @contextmanager
    def trace_preflight(
        self,
        span_name: str = "scout.preflight_gate",
        domain_id: str | None = None,
        prompt: str | None = None,
    ) -> Iterator[PreflightSpanContext]:
        """Context manager tracing a deterministic pre-flight gate check."""
        with self.tracer.start_as_current_span(span_name) as span:
            ctx = PreflightSpanContext(span)
            if domain_id:
                span.set_attribute(ATTR_DOMAIN_ID, domain_id)
            span.set_attribute(GEN_AI_OPERATION_NAME, "preflight_gate")
            try:
                yield ctx
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, description=str(exc))
                raise

    @contextmanager
    def trace_executor(
        self,
        span_name: str = "scout.executor.handoff",
        pack: Any | None = None,
        principal: Any | None = None,
        model_id: str = "gemini-3.8-flash",
        system: str = "gemini",
    ) -> Iterator[ExecutorSpanContext]:
        """Context manager tracing a paid Gemini executor handoff and execution."""
        with self.tracer.start_as_current_span(span_name) as span:
            ctx = ExecutorSpanContext(span)
            if system:
                span.set_attribute(GEN_AI_SYSTEM, system)
            if model_id:
                span.set_attribute(GEN_AI_REQUEST_MODEL, model_id)
            if pack is not None:
                if hasattr(pack, "domain_id"):
                    span.set_attribute(ATTR_DOMAIN_ID, str(pack.domain_id))
                if hasattr(pack, "trace_id"):
                    span.set_attribute(ATTR_TRACE_ID, str(pack.trace_id))
            if principal is not None:
                if hasattr(principal, "user_id"):
                    span.set_attribute(ATTR_PRINCIPAL_ID, str(principal.user_id))
                if hasattr(principal, "role"):
                    role_val = (
                        principal.role.value
                        if hasattr(principal.role, "value")
                        else str(principal.role)
                    )
                    span.set_attribute(ATTR_PRINCIPAL_ROLE, str(role_val))
            span.set_attribute(GEN_AI_OPERATION_NAME, "implementation_executor")
            try:
                yield ctx
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, description=str(exc))
                raise

    def get_finished_spans(self) -> tuple[ReadableSpan, ...]:
        """Returns all completed spans captured by the in-memory exporter."""
        return tuple(self.exporter.get_finished_spans())

    def clear_spans(self) -> None:
        """Clears all accumulated spans in the in-memory exporter."""
        self.exporter.clear()

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
