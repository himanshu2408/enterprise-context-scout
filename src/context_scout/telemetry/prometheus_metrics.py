"""Prometheus & Google Managed Service for Prometheus (GMP) Retrieval-Drift Metrics (M5-01)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Final
from uuid import UUID

_RRF_BUCKETS: Final[tuple[float, ...]] = (0.1, 0.2, 0.34, 0.5, 0.65, 0.8, 0.9, 0.95)
_RERANKER_BUCKETS: Final[tuple[float, ...]] = (0.1, 0.3, 0.5, 0.6, 0.75, 0.9)

_FORBIDDEN_HIGH_CARDINALITY_LABELS: Final[frozenset[str]] = frozenset(
    {"user_id", "principal_id", "actor_id", "trace_id", "query", "email"}
)


def _escape_prom_label(val: str) -> str:
    return val.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    for k, _ in labels:
        if k in _FORBIDDEN_HIGH_CARDINALITY_LABELS:
            raise ValueError(f"Prohibited high-cardinality/PII Prometheus label: {k!r}")
    parts = [f'{k}="{_escape_prom_label(v)}"' for k, v in labels]
    return "{" + ",".join(parts) + "}"


@dataclass
class _HistogramState:
    buckets: tuple[float, ...]
    counts: list[int] = field(default_factory=list)
    total_count: int = 0
    total_sum: float = 0.0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        v = max(0.0, float(value))
        self.total_count += 1
        self.total_sum += v
        for idx, upper in enumerate(self.buckets):
            if v <= upper:
                self.counts[idx] += 1


@dataclass
class _DriftState:
    ewma_confidence: float
    baseline_confidence: float
    observations: int = 0


class ScoutPrometheusRegistry:
    """Thread-safe Prometheus 0.0.4 / GMP metrics registry for retrieval drift & audit volume."""

    def __init__(
        self,
        *,
        drift_ewma_alpha: float = 0.2,
        drift_alert_threshold: float = 0.15,
        max_tracked_tenants: int = 500,
    ) -> None:
        self._lock = threading.RLock()
        self.drift_ewma_alpha = float(drift_ewma_alpha)
        self.drift_alert_threshold = float(drift_alert_threshold)
        self.max_tracked_tenants = max(1, int(max_tracked_tenants))
        self._known_tenants: set[str] = set()
        self._tenant_overflow_total: float = 0.0

        self._queries_total: dict[tuple[tuple[str, str], ...], float] = {}
        self._abstention_total: dict[tuple[tuple[str, str], ...], float] = {}
        self._zero_hit_total: dict[tuple[tuple[str, str], ...], float] = {}
        self._citation_failures_total: dict[tuple[tuple[str, str], ...], float] = {}
        self._commit_drift_total: dict[tuple[tuple[str, str], ...], float] = {}
        self._audit_events_total: dict[tuple[tuple[str, str], ...], float] = {}

        self._rrf_hist: dict[tuple[tuple[str, str], ...], _HistogramState] = {}
        self._reranker_hist: dict[tuple[tuple[str, str], ...], _HistogramState] = {}
        self._drift_states: dict[tuple[str, str], _DriftState] = {}
        self._explicit_baselines: dict[tuple[str, str], float] = {}

    def _normalize_tenant_label(self, tenant_id: UUID | str) -> str:
        tid = str(tenant_id)
        if tid in self._known_tenants:
            return tid
        if len(self._known_tenants) < self.max_tracked_tenants:
            self._known_tenants.add(tid)
            return tid
        self._tenant_overflow_total += 1.0
        return "_overflow"

    def set_baseline_confidence(
        self, tenant_id: UUID | str, domain_id: str, baseline: float
    ) -> None:
        """Sets an explicit confidence baseline for (tenant_id, domain_id) drift detection."""
        did = domain_id.strip() or "default"
        clamped = min(max(float(baseline), 0.0), 1.0)
        with self._lock:
            tid = self._normalize_tenant_label(tenant_id)
            self._explicit_baselines[(tid, did)] = clamped
            state = self._drift_states.get((tid, did))
            if state is not None:
                state.baseline_confidence = clamped

    def record_query_outcome(
        self,
        *,
        tenant_id: UUID | str,
        domain_id: str,
        outcome: str,
        rrf_score: float | None = None,
        reranker_score: float | None = None,
        zero_hit: bool = False,
        reason_code: str | None = None,
        citation_failure: bool = False,
        commit_drift: bool = False,
    ) -> None:
        """Records low-cardinality query outcome and updates EWMA retrieval-drift gauges."""
        did = domain_id.strip() or "default"
        out_label = outcome.strip() or "unknown"

        with self._lock:
            tid = self._normalize_tenant_label(tenant_id)
            base_labels = (("domain_id", did), ("tenant_id", tid))
            q_labels = (("domain_id", did), ("outcome", out_label), ("tenant_id", tid))
            self._queries_total[q_labels] = self._queries_total.get(q_labels, 0.0) + 1.0

            if out_label in {"abstain", "blocked", "error"} or reason_code:
                rc = (reason_code or out_label).strip() or "unknown"
                ab_labels = (("domain_id", did), ("reason_code", rc), ("tenant_id", tid))
                self._abstention_total[ab_labels] = self._abstention_total.get(ab_labels, 0.0) + 1.0

            if zero_hit:
                self._zero_hit_total[base_labels] = self._zero_hit_total.get(base_labels, 0.0) + 1.0

            if citation_failure:
                self._citation_failures_total[base_labels] = (
                    self._citation_failures_total.get(base_labels, 0.0) + 1.0
                )

            if commit_drift:
                self._commit_drift_total[base_labels] = (
                    self._commit_drift_total.get(base_labels, 0.0) + 1.0
                )

            if rrf_score is not None:
                clamped_rrf = min(max(float(rrf_score), 0.0), 1.0)
                hist = self._rrf_hist.get(base_labels)
                if hist is None:
                    hist = _HistogramState(buckets=_RRF_BUCKETS)
                    self._rrf_hist[base_labels] = hist
                hist.observe(clamped_rrf)

                key = (tid, did)
                state = self._drift_states.get(key)
                if state is None:
                    init_base = self._explicit_baselines.get(key, clamped_rrf)
                    state = _DriftState(
                        ewma_confidence=clamped_rrf,
                        baseline_confidence=init_base,
                        observations=1,
                    )
                    self._drift_states[key] = state
                else:
                    alpha = self.drift_ewma_alpha
                    state.ewma_confidence = (alpha * clamped_rrf) + (
                        (1.0 - alpha) * state.ewma_confidence
                    )
                    state.observations += 1
                    if key not in self._explicit_baselines and state.observations <= 5:
                        # Establish stable initial warmup baseline across early nominal queries
                        state.baseline_confidence = (
                            state.baseline_confidence * (state.observations - 1) + clamped_rrf
                        ) / state.observations

            if reranker_score is not None:
                clamped_rerank = min(max(float(reranker_score), 0.0), 1.0)
                r_hist = self._reranker_hist.get(base_labels)
                if r_hist is None:
                    r_hist = _HistogramState(buckets=_RERANKER_BUCKETS)
                    self._reranker_hist[base_labels] = r_hist
                r_hist.observe(clamped_rerank)

    def record_audit_event(
        self,
        *,
        tenant_id: UUID | str,
        event_type: str,
        severity: str,
    ) -> None:
        """Records low-cardinality audit event counter."""
        with self._lock:
            tid = self._normalize_tenant_label(tenant_id)
            labels = (
                ("event_type", event_type.strip() or "unknown"),
                ("severity", severity.strip().upper() or "INFO"),
                ("tenant_id", tid),
            )
            self._audit_events_total[labels] = self._audit_events_total.get(labels, 0.0) + 1.0

    def get_drift_snapshot(
        self, tenant_id: UUID | str, domain_id: str
    ) -> dict[str, float | bool | int]:
        """Returns current EWMA confidence, baseline, drift score, and alert status."""
        key = (str(tenant_id), domain_id.strip() or "default")
        with self._lock:
            state = self._drift_states.get(key)
            if state is None:
                return {
                    "observations": 0,
                    "ewma_confidence": 0.0,
                    "baseline_confidence": 0.0,
                    "drift_score": 0.0,
                    "drift_alert": False,
                }
            drift_score = round(max(0.0, state.baseline_confidence - state.ewma_confidence), 6)
            return {
                "observations": state.observations,
                "ewma_confidence": round(state.ewma_confidence, 6),
                "baseline_confidence": round(state.baseline_confidence, 6),
                "drift_score": drift_score,
                "drift_alert": drift_score >= self.drift_alert_threshold,
            }

    def render_prometheus_text(self) -> str:
        """Renders all metrics in Prometheus 0.0.4 / GMP exposition text format."""
        lines: list[str] = []
        with self._lock:
            self._emit_counter(
                lines,
                "scout_queries_total",
                "Total Context Scout queries processed by tenant, domain, and outcome.",
                self._queries_total,
            )
            self._emit_counter(
                lines,
                "scout_abstention_total",
                "Total $0 abstentions and blocked queries by tenant, domain, and reason.",
                self._abstention_total,
            )
            self._emit_counter(
                lines,
                "scout_zero_hit_queries_total",
                "Total queries returning zero grounded context slices or citations.",
                self._zero_hit_total,
            )
            self._emit_counter(
                lines,
                "scout_citation_verification_failures_total",
                "Total failed verbatim citation verification checks.",
                self._citation_failures_total,
            )
            self._emit_counter(
                lines,
                "scout_commit_drift_alerts_total",
                "Total session commit-drift alerts detected on stale git blobs.",
                self._commit_drift_total,
            )
            self._emit_counter(
                lines,
                "scout_audit_events_total",
                "Total hash-chained audit events recorded by tenant, type, and severity.",
                self._audit_events_total,
            )
            self._emit_histogram(
                lines,
                "scout_retrieval_rrf_score",
                "Distribution of normalized 3-way RRF retrieval scores.",
                self._rrf_hist,
            )
            self._emit_histogram(
                lines,
                "scout_retrieval_reranker_score",
                "Distribution of cross-encoder reranker relevance scores.",
                self._reranker_hist,
            )

            ewma_series: dict[tuple[tuple[str, str], ...], float] = {}
            drift_series: dict[tuple[tuple[str, str], ...], float] = {}
            alert_series: dict[tuple[tuple[str, str], ...], float] = {}
            for (tid, did), st in sorted(self._drift_states.items()):
                lbl = (("domain_id", did), ("tenant_id", tid))
                drift_val = max(0.0, st.baseline_confidence - st.ewma_confidence)
                ewma_series[lbl] = round(st.ewma_confidence, 6)
                drift_series[lbl] = round(drift_val, 6)
                alert_series[lbl] = 1.0 if drift_val >= self.drift_alert_threshold else 0.0

            self._emit_gauge(
                lines,
                "scout_retrieval_confidence_ewma",
                "Exponentially weighted moving average of retrieval confidence.",
                ewma_series,
            )
            self._emit_gauge(
                lines,
                "scout_retrieval_drift_score",
                "Drop in EWMA retrieval confidence relative to tenant/domain baseline.",
                drift_series,
            )
            self._emit_gauge(
                lines,
                "scout_retrieval_drift_alert",
                "Binary gauge (1.0 when drift_score >= threshold, else 0.0).",
                alert_series,
            )
            lines.append(
                "# HELP scout_prometheus_tenant_overflow_total Total events bucketed "
                "into _overflow due to max_tracked_tenants cap."
            )
            lines.append("# TYPE scout_prometheus_tenant_overflow_total counter")
            lines.append(f"scout_prometheus_tenant_overflow_total {self._tenant_overflow_total:g}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _emit_counter(
        lines: list[str],
        name: str,
        help_text: str,
        series: dict[tuple[tuple[str, str], ...], float],
    ) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} counter")
        for labels, val in sorted(series.items()):
            lines.append(f"{name}{_format_labels(labels)} {val:g}")

    @staticmethod
    def _emit_gauge(
        lines: list[str],
        name: str,
        help_text: str,
        series: dict[tuple[tuple[str, str], ...], float],
    ) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        for labels, val in sorted(series.items()):
            lines.append(f"{name}{_format_labels(labels)} {val:g}")

    @staticmethod
    def _emit_histogram(
        lines: list[str],
        name: str,
        help_text: str,
        series: dict[tuple[tuple[str, str], ...], _HistogramState],
    ) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} histogram")
        for labels, hist in sorted(series.items()):
            for upper, count in zip(hist.buckets, hist.counts, strict=True):
                b_labels = (*labels, ("le", f"{upper:g}"))
                lines.append(f"{name}_bucket{_format_labels(b_labels)} {count}")
            inf_labels = (*labels, ("le", "+Inf"))
            lines.append(f"{name}_bucket{_format_labels(inf_labels)} {hist.total_count}")
            lines.append(f"{name}_sum{_format_labels(labels)} {hist.total_sum:.6f}")
            lines.append(f"{name}_count{_format_labels(labels)} {hist.total_count}")

    def reset(self) -> None:
        with self._lock:
            self._queries_total.clear()
            self._abstention_total.clear()
            self._zero_hit_total.clear()
            self._citation_failures_total.clear()
            self._commit_drift_total.clear()
            self._audit_events_total.clear()
            self._rrf_hist.clear()
            self._reranker_hist.clear()
            self._drift_states.clear()
            self._explicit_baselines.clear()
            self._known_tenants.clear()
            self._tenant_overflow_total = 0.0


_PROM_LOCK = threading.Lock()
_GLOBAL_PROM_REGISTRY: ScoutPrometheusRegistry | None = None


def get_prometheus_registry() -> ScoutPrometheusRegistry:
    """Returns shared singleton ScoutPrometheusRegistry."""
    global _GLOBAL_PROM_REGISTRY
    if _GLOBAL_PROM_REGISTRY is not None:
        return _GLOBAL_PROM_REGISTRY
    with _PROM_LOCK:
        if _GLOBAL_PROM_REGISTRY is None:
            from context_scout.config import ScoutSettings

            cfg = ScoutSettings()
            _GLOBAL_PROM_REGISTRY = ScoutPrometheusRegistry(
                drift_ewma_alpha=cfg.drift_ewma_alpha,
                drift_alert_threshold=cfg.drift_alert_threshold,
            )
        return _GLOBAL_PROM_REGISTRY


def reset_prometheus_registry() -> ScoutPrometheusRegistry:
    """Resets shared singleton ScoutPrometheusRegistry (used in tests)."""
    global _GLOBAL_PROM_REGISTRY
    with _PROM_LOCK:
        _GLOBAL_PROM_REGISTRY = ScoutPrometheusRegistry()
        return _GLOBAL_PROM_REGISTRY


__all__ = [
    "ScoutPrometheusRegistry",
    "get_prometheus_registry",
    "reset_prometheus_registry",
]
