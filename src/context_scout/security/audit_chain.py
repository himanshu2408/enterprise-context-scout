"""Tamper-Evident Hash-Chained Audit Log, GCP Cloud Logging & SIEM Exporters (M5-01)."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import os
import socket
import threading
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import urlparse
from uuid import UUID

import httpx
import opentelemetry.trace as otel_trace
from pydantic import BaseModel, Field

from context_scout.config import ScoutSettings
from context_scout.db.session import RLSConnectionManager, get_query_pool
from context_scout.schemas.contracts import DEFAULT_TENANT_ID, PrincipalContext, RoleType
from context_scout.security.betrvg_privacy import BetrVGPrivacyGuard
from context_scout.telemetry.prometheus_metrics import get_prometheus_registry

logger = logging.getLogger(__name__)

GENESIS_PREV_HASH: Final[str] = "0" * 64

AuditSeverity = Literal[
    "DEFAULT",
    "DEBUG",
    "INFO",
    "NOTICE",
    "WARNING",
    "ERROR",
    "CRITICAL",
    "ALERT",
    "EMERGENCY",
]
AuditOutcome = Literal["allow", "deny", "success", "failure", "abstain"]


class AuditEventRecord(BaseModel):
    """Canonical immutable hash-chained audit event record."""

    tenant_id: UUID = Field(default=DEFAULT_TENANT_ID)
    seq_no: int = Field(ge=1)
    event_id: UUID
    domain_id: str = Field(default="default")
    event_type: str = Field(min_length=1)
    severity: AuditSeverity = Field(default="INFO")
    actor_id: str = Field(min_length=1)
    actor_role: str = Field(min_length=1)
    resource_uri: str = Field(default="")
    outcome: AuditOutcome
    trace_id: str = Field(default="")
    span_id: str = Field(default="")
    prev_hash_sha256: str = Field(min_length=64, max_length=64)
    event_hash_sha256: str = Field(min_length=64, max_length=64)
    payload_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AuditChainVerificationReport(BaseModel):
    """Result of full cryptographic SHA-256 chain integrity verification."""

    tenant_id: str
    valid: bool
    checked_count: int
    first_corrupted_seq_no: int | None = None
    head_hash_sha256: str = GENESIS_PREV_HASH
    reason: str = "ok"


def _normalize_payload_for_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)
    parsed: dict[str, Any] = json.loads(serialized)
    return parsed


def compute_audit_event_hash(
    *,
    tenant_id: UUID,
    seq_no: int,
    prev_hash_sha256: str,
    event_id: UUID,
    domain_id: str,
    event_type: str,
    severity: str,
    actor_id: str,
    actor_role: str,
    resource_uri: str,
    outcome: str,
    trace_id: str,
    span_id: str,
    payload_json: Mapping[str, Any],
    created_at: datetime,
) -> str:
    """Computes deterministic SHA-256 digest binding current event to prev_hash_sha256."""
    canonical_body = {
        "tenant_id": str(tenant_id),
        "seq_no": int(seq_no),
        "prev_hash_sha256": prev_hash_sha256,
        "event_id": str(event_id),
        "domain_id": domain_id,
        "event_type": event_type,
        "severity": severity,
        "actor_id": actor_id,
        "actor_role": actor_role,
        "resource_uri": resource_uri,
        "outcome": outcome,
        "trace_id": trace_id,
        "span_id": span_id,
        "payload_json": _normalize_payload_for_hash(payload_json),
        "created_at": created_at.astimezone(UTC).isoformat(timespec="microseconds"),
    }
    encoded = json.dumps(canonical_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def format_gcp_cloud_logging_entry(
    record: AuditEventRecord,
    *,
    gcp_project: str | None = None,
    trace_sampled: bool = True,
) -> dict[str, Any]:
    """Formats an AuditEventRecord into GCP Cloud Logging structured JSON."""
    project = (gcp_project or os.environ.get("SCOUT_GCP_PROJECT") or "").strip()
    trace_field = ""
    if record.trace_id:
        trace_field = f"projects/{project}/traces/{record.trace_id}" if project else record.trace_id
    target = record.resource_uri or record.domain_id
    return {
        "severity": record.severity,
        "message": f"{record.event_type} ({record.outcome}) on {target}",
        "time": record.created_at.astimezone(UTC).isoformat(timespec="microseconds"),
        "logging.googleapis.com/trace": trace_field,
        "logging.googleapis.com/spanId": record.span_id,
        "logging.googleapis.com/trace_sampled": bool(trace_sampled and bool(record.trace_id)),
        "logging.googleapis.com/labels": {
            "tenant_id": str(record.tenant_id),
            "domain_id": record.domain_id,
            "event_type": record.event_type,
            "outcome": record.outcome,
        },
        "audit": {
            "seq_no": record.seq_no,
            "event_id": str(record.event_id),
            "actor_id": record.actor_id,
            "actor_role": record.actor_role,
            "resource_uri": record.resource_uri,
            "prev_hash_sha256": record.prev_hash_sha256,
            "event_hash_sha256": record.event_hash_sha256,
            "payload": record.payload_json,
        },
    }


class NDJSONSIEMExporter:
    """Exports audit records as NDJSON lines (GCP Cloud Logging structured JSON or flat)."""

    def __init__(
        self,
        *,
        file_path: Path | None = None,
        gcp_project: str | None = None,
        max_buffer_size: int = 2000,
    ) -> None:
        self.file_path = file_path
        self.gcp_project = gcp_project
        self._lock = threading.Lock()
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_buffer_size)

    def export_record(
        self,
        record: AuditEventRecord,
        *,
        fmt: Literal["gcp_logging", "ndjson"] = "gcp_logging",
    ) -> dict[str, Any]:
        entry = (
            format_gcp_cloud_logging_entry(record, gcp_project=self.gcp_project)
            if fmt == "gcp_logging"
            else record.model_dump(mode="json")
        )
        line = json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            self._buffer.append(entry)
            if self.file_path is not None:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                with self.file_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        return entry

    def get_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._buffer)


_BLOCKED_SIEM_HOSTS: Final[frozenset[str]] = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
        "metadata",
        "instance-data",
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
    }
)
_BLOCKED_SIEM_NETWORKS: Final[tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]] = (
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT & Alibaba Cloud metadata (100.100.100.200)
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local & AWS/GCP/Azure IMDS
    ipaddress.ip_network("fd00:ec2::/32"),  # AWS EC2 IPv6 metadata
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
)


def _assert_safe_siem_ip(
    ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    raw_host: str,
) -> None:
    candidates: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [ip_obj]
    if isinstance(ip_obj, ipaddress.IPv6Address):
        if ip_obj.ipv4_mapped is not None:
            candidates.append(ip_obj.ipv4_mapped)
        if ip_obj.sixtofour is not None:
            candidates.append(ip_obj.sixtofour)
        if (int(ip_obj) >> 32) == 0 and int(ip_obj) != 0:
            candidates.append(ipaddress.IPv4Address(int(ip_obj)))

    for ip in candidates:
        if (
            ip.is_link_local
            or ip.is_unspecified
            or ip.is_multicast
            or ip.is_loopback
            or ip.is_private
            or ip.is_reserved
            or any(ip in net for net in _BLOCKED_SIEM_NETWORKS)
        ):
            raise ValueError(
                f"SIEM webhook_url blocks non-public/metadata IP {str(ip)!r} "
                f"for host {raw_host!r} (SSRF)."
            )


def _validate_siem_webhook_url(raw_url: str) -> str:
    cleaned = raw_url.strip()
    if not cleaned:
        return ""
    if any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in cleaned):
        raise ValueError("SIEM webhook_url must not contain control characters or whitespace.")
    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError(f"SIEM webhook_url must use http or https scheme (got {parsed.scheme!r}).")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ValueError("SIEM webhook_url must not contain embedded userinfo credentials.")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"SIEM webhook_url contains invalid port: {exc}") from exc

    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise ValueError("SIEM webhook_url must specify a valid hostname.")
    if "%" in host:
        raise ValueError("SIEM webhook_url must not contain IPv6 zone IDs.")
    if (
        host in _BLOCKED_SIEM_HOSTS
        or host.endswith(".localhost")
        or host.endswith(".metadata.google.internal")
    ):
        raise ValueError(f"SIEM webhook_url blocks restricted/metadata host {host!r} (SSRF).")

    ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        ip_obj = ipaddress.ip_address(host)
    except ValueError:
        ip_obj = None

    if ip_obj is not None:
        _assert_safe_siem_ip(ip_obj, raw_host=host)
        return cleaned

    try:
        packed = socket.inet_aton(host)
    except OSError:
        packed = None
    if packed is not None:
        canonical = ipaddress.IPv4Address(packed)
        _assert_safe_siem_ip(canonical, raw_host=host)
        raise ValueError(f"SIEM webhook_url blocks non-canonical IPv4 literal {host!r} (SSRF).")

    try:
        addr_infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        addr_infos = []
    for info in addr_infos:
        raw_addr = str(info[4][0]).split("%", 1)[0]
        try:
            resolved_ip = ipaddress.ip_address(raw_addr)
        except ValueError:
            continue
        _assert_safe_siem_ip(resolved_ip, raw_host=host)

    return cleaned


class WebhookSIEMExporter:
    """Exports structured audit events to external SIEM webhook with HMAC-SHA256 signature."""

    def __init__(
        self,
        webhook_url: str,
        *,
        webhook_secret: str | None = None,
        gcp_project: str | None = None,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.webhook_url = _validate_siem_webhook_url(webhook_url)
        self.webhook_secret = (webhook_secret or "").strip()
        self.gcp_project = gcp_project
        self._http_client = http_client
        self._timeout = timeout_seconds

    def build_signed_payload(self, record: AuditEventRecord) -> tuple[bytes, dict[str, str]]:
        body_dict = format_gcp_cloud_logging_entry(record, gcp_project=self.gcp_project)
        body_bytes = json.dumps(body_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.webhook_secret:
            sig = hmac.new(
                self.webhook_secret.encode("utf-8"), body_bytes, hashlib.sha256
            ).hexdigest()
            headers["X-Scout-SIEM-Signature-256"] = f"sha256={sig}"
        return (body_bytes, headers)

    def export_record(self, record: AuditEventRecord) -> bool:
        if not self.webhook_url:
            return False
        body_bytes, headers = self.build_signed_payload(record)
        try:
            if self._http_client is not None:
                resp = self._http_client.post(
                    self.webhook_url, content=body_bytes, headers=headers, timeout=self._timeout
                )
            else:
                parsed = urlparse(self.webhook_url)
                host = (parsed.hostname or "").strip().lower().rstrip(".")
                port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
                try:
                    ip_literal: ipaddress.IPv4Address | ipaddress.IPv6Address | None = (
                        ipaddress.ip_address(host)
                    )
                except ValueError:
                    ip_literal = None
                if ip_literal is not None:
                    _assert_safe_siem_ip(ip_literal, raw_host=host)
                else:
                    addr_infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
                    if not addr_infos:
                        raise ValueError(f"SIEM webhook host {host!r} failed DNS resolution.")
                    for info in addr_infos:
                        raw_addr = str(info[4][0]).split("%", 1)[0]
                        _assert_safe_siem_ip(ipaddress.ip_address(raw_addr), raw_host=host)
                with httpx.Client(timeout=self._timeout, follow_redirects=False) as client:
                    resp = client.post(self.webhook_url, content=body_bytes, headers=headers)
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning(
                "SIEM webhook export to %s failed (%s): %s",
                self.webhook_url,
                type(exc).__name__,
                exc,
            )
            return False


def _active_trace_ids() -> tuple[str, str]:
    span = otel_trace.get_current_span()
    if span is None:
        return ("", "")
    ctx = span.get_span_context()
    if ctx is None or not ctx.is_valid:
        return ("", "")
    return (otel_trace.format_trace_id(ctx.trace_id), otel_trace.format_span_id(ctx.span_id))


class AuditEventChainService:
    """Append-only hash-chained audit event store backed by PG17 FORCE RLS or memory."""

    def __init__(
        self,
        *,
        connection_manager: RLSConnectionManager | None = None,
        betrvg_guard: BetrVGPrivacyGuard | None = None,
        ndjson_exporter: NDJSONSIEMExporter | None = None,
        webhook_exporter: WebhookSIEMExporter | None = None,
        gcp_project: str | None = None,
    ) -> None:
        self._conn_mgr = connection_manager
        self._gcp_project = gcp_project
        self._betrvg = betrvg_guard or BetrVGPrivacyGuard()
        self._ndjson_exporter = ndjson_exporter or NDJSONSIEMExporter(
            file_path=(
                Path(os.environ["SCOUT_SIEM_NDJSON_PATH"])
                if os.environ.get("SCOUT_SIEM_NDJSON_PATH", "").strip()
                else None
            ),
            gcp_project=gcp_project,
        )
        self._webhook_exporter = webhook_exporter
        if self._webhook_exporter is None and os.environ.get("SCOUT_SIEM_WEBHOOK_URL", "").strip():
            self._webhook_exporter = WebhookSIEMExporter(
                webhook_url=os.environ["SCOUT_SIEM_WEBHOOK_URL"],
                webhook_secret=os.environ.get("SCOUT_SIEM_WEBHOOK_SECRET"),
                gcp_project=gcp_project,
            )
        self._lock = threading.RLock()
        self._events_by_tenant: dict[UUID, list[AuditEventRecord]] = {}

    def record_event(
        self,
        *,
        principal: PrincipalContext | None = None,
        tenant_id: UUID | None = None,
        domain_id: str = "default",
        event_type: str,
        severity: AuditSeverity = "INFO",
        actor_id: str | None = None,
        actor_role: str | None = None,
        resource_uri: str = "",
        outcome: AuditOutcome = "allow",
        trace_id: str | None = None,
        span_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> AuditEventRecord:
        """Appends a tamper-evident hash-chained audit event and dispatches SIEM exporters."""
        resolved_tid = tenant_id or (principal.tenant_id if principal else DEFAULT_TENANT_ID)
        raw_actor = (
            actor_id
            if actor_id is not None
            else (principal.user_id if principal is not None else "system")
        ).strip() or "anonymous"
        resolved_actor = (
            self._betrvg.pseudonymize_user_id(raw_actor, resolved_tid)
            if self._betrvg.enabled and raw_actor.lower() not in {"anonymous", "system"}
            else raw_actor
        )
        resolved_role = (
            actor_role
            if actor_role is not None
            else (principal.role.value if principal is not None else "system")
        ).strip() or "system"

        active_trace, active_span = _active_trace_ids()
        resolved_trace = (trace_id if trace_id is not None else active_trace).strip()
        resolved_span = (span_id if span_id is not None else active_span).strip()

        clean_payload = _normalize_payload_for_hash(payload or {})
        if self._betrvg.enabled:
            clean_payload = self._betrvg.sanitize_telemetry_attributes(
                clean_payload, tenant_id=resolved_tid
            )

        event_id = uuid.uuid4()
        created_at = datetime.now(UTC)

        if self._conn_mgr is not None:
            tx_principal = principal or PrincipalContext(
                user_id=resolved_actor,
                role=RoleType.COMPLIANCE_OFFICER,
                allowed_scopes=["audit:compliance"],
                tenant_id=resolved_tid,
            )
            with self._conn_mgr.scoped_transaction(
                tx_principal, enable_hnsw_iterative_scan=False
            ) as conn:
                conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended('audit_chain:' || %s, 0));",
                    (str(resolved_tid),),
                )
                prev_row = conn.execute(
                    """
                    SELECT seq_no, event_hash_sha256
                    FROM audit_events
                    ORDER BY seq_no DESC
                    LIMIT 1;
                    """
                ).fetchone()
                next_seq = (int(str(prev_row[0])) + 1) if prev_row is not None else 1
                prev_hash = str(prev_row[1]) if prev_row is not None else GENESIS_PREV_HASH
                event_hash = compute_audit_event_hash(
                    tenant_id=resolved_tid,
                    seq_no=next_seq,
                    prev_hash_sha256=prev_hash,
                    event_id=event_id,
                    domain_id=domain_id,
                    event_type=event_type,
                    severity=severity,
                    actor_id=resolved_actor,
                    actor_role=resolved_role,
                    resource_uri=resource_uri,
                    outcome=outcome,
                    trace_id=resolved_trace,
                    span_id=resolved_span,
                    payload_json=clean_payload,
                    created_at=created_at,
                )
                conn.execute(
                    """
                    INSERT INTO audit_events (
                        tenant_id, seq_no, event_id, domain_id, event_type, severity,
                        actor_id, actor_role, resource_uri, outcome, trace_id, span_id,
                        prev_hash_sha256, event_hash_sha256, payload_json, created_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s::jsonb, %s
                    );
                    """,
                    (
                        resolved_tid,
                        next_seq,
                        event_id,
                        domain_id,
                        event_type,
                        severity,
                        resolved_actor,
                        resolved_role,
                        resource_uri,
                        outcome,
                        resolved_trace,
                        resolved_span,
                        prev_hash,
                        event_hash,
                        json.dumps(clean_payload, sort_keys=True, separators=(",", ":")),
                        created_at,
                    ),
                )
                record = AuditEventRecord(
                    tenant_id=resolved_tid,
                    seq_no=next_seq,
                    event_id=event_id,
                    domain_id=domain_id,
                    event_type=event_type,
                    severity=severity,
                    actor_id=resolved_actor,
                    actor_role=resolved_role,
                    resource_uri=resource_uri,
                    outcome=outcome,
                    trace_id=resolved_trace,
                    span_id=resolved_span,
                    prev_hash_sha256=prev_hash,
                    event_hash_sha256=event_hash,
                    payload_json=clean_payload,
                    created_at=created_at,
                )
        else:
            with self._lock:
                chain = self._events_by_tenant.setdefault(resolved_tid, [])
                next_seq = (chain[-1].seq_no + 1) if chain else 1
                prev_hash = chain[-1].event_hash_sha256 if chain else GENESIS_PREV_HASH
                event_hash = compute_audit_event_hash(
                    tenant_id=resolved_tid,
                    seq_no=next_seq,
                    prev_hash_sha256=prev_hash,
                    event_id=event_id,
                    domain_id=domain_id,
                    event_type=event_type,
                    severity=severity,
                    actor_id=resolved_actor,
                    actor_role=resolved_role,
                    resource_uri=resource_uri,
                    outcome=outcome,
                    trace_id=resolved_trace,
                    span_id=resolved_span,
                    payload_json=clean_payload,
                    created_at=created_at,
                )
                record = AuditEventRecord(
                    tenant_id=resolved_tid,
                    seq_no=next_seq,
                    event_id=event_id,
                    domain_id=domain_id,
                    event_type=event_type,
                    severity=severity,
                    actor_id=resolved_actor,
                    actor_role=resolved_role,
                    resource_uri=resource_uri,
                    outcome=outcome,
                    trace_id=resolved_trace,
                    span_id=resolved_span,
                    prev_hash_sha256=prev_hash,
                    event_hash_sha256=event_hash,
                    payload_json=clean_payload,
                    created_at=created_at,
                )
                chain.append(record)

        self._ndjson_exporter.export_record(record, fmt="gcp_logging")
        if self._webhook_exporter is not None:
            self._webhook_exporter.export_record(record)
        get_prometheus_registry().record_audit_event(
            tenant_id=resolved_tid,
            event_type=event_type,
            severity=severity,
        )
        return record

    def list_events(
        self,
        principal: PrincipalContext,
        *,
        domain_id: str | None = None,
        limit: int = 500,
    ) -> list[AuditEventRecord]:
        """Lists tenant-visible audit events ordered by seq_no ASC."""
        safe_limit = max(1, min(int(limit), 10_000))
        if self._conn_mgr is not None:
            with self._conn_mgr.scoped_transaction(
                principal, enable_hnsw_iterative_scan=False
            ) as conn:
                if domain_id is not None:
                    rows = conn.execute(
                        """
                        SELECT tenant_id, seq_no, event_id, domain_id, event_type, severity,
                               actor_id, actor_role, resource_uri, outcome, trace_id, span_id,
                               prev_hash_sha256, event_hash_sha256, payload_json, created_at
                        FROM audit_events
                        WHERE domain_id = %s
                        ORDER BY seq_no ASC
                        LIMIT %s;
                        """,
                        (domain_id, safe_limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT tenant_id, seq_no, event_id, domain_id, event_type, severity,
                               actor_id, actor_role, resource_uri, outcome, trace_id, span_id,
                               prev_hash_sha256, event_hash_sha256, payload_json, created_at
                        FROM audit_events
                        ORDER BY seq_no ASC
                        LIMIT %s;
                        """,
                        (safe_limit,),
                    ).fetchall()
                return [self._row_to_record(r) for r in rows]

        with self._lock:
            items = list(self._events_by_tenant.get(principal.tenant_id, ()))
            if domain_id is not None:
                items = [e for e in items if e.domain_id == domain_id]
            return items[:safe_limit]

    def verify_chain_integrity(self, principal: PrincipalContext) -> AuditChainVerificationReport:
        """Verifies contiguous seq_no, prev_hash linkage, and SHA-256 integrity for tenant."""
        events = self.list_events(principal, domain_id=None, limit=10_000)
        expected_prev = GENESIS_PREV_HASH
        for expected_seq, rec in enumerate(events, start=1):
            if rec.seq_no != expected_seq:
                return AuditChainVerificationReport(
                    tenant_id=str(principal.tenant_id),
                    valid=False,
                    checked_count=expected_seq - 1,
                    first_corrupted_seq_no=expected_seq,
                    head_hash_sha256=expected_prev,
                    reason=(
                        f"Sequence gap detected: expected seq_no={expected_seq}, got {rec.seq_no}."
                    ),
                )
            if rec.prev_hash_sha256 != expected_prev:
                return AuditChainVerificationReport(
                    tenant_id=str(principal.tenant_id),
                    valid=False,
                    checked_count=expected_seq,
                    first_corrupted_seq_no=rec.seq_no,
                    head_hash_sha256=expected_prev,
                    reason=f"prev_hash_sha256 mismatch at seq_no={rec.seq_no}.",
                )
            recomputed = compute_audit_event_hash(
                tenant_id=rec.tenant_id,
                seq_no=rec.seq_no,
                prev_hash_sha256=rec.prev_hash_sha256,
                event_id=rec.event_id,
                domain_id=rec.domain_id,
                event_type=rec.event_type,
                severity=rec.severity,
                actor_id=rec.actor_id,
                actor_role=rec.actor_role,
                resource_uri=rec.resource_uri,
                outcome=rec.outcome,
                trace_id=rec.trace_id,
                span_id=rec.span_id,
                payload_json=rec.payload_json,
                created_at=rec.created_at,
            )
            if not hmac.compare_digest(recomputed, rec.event_hash_sha256):
                return AuditChainVerificationReport(
                    tenant_id=str(principal.tenant_id),
                    valid=False,
                    checked_count=expected_seq,
                    first_corrupted_seq_no=rec.seq_no,
                    head_hash_sha256=expected_prev,
                    reason=f"event_hash_sha256 tamper detected at seq_no={rec.seq_no}.",
                )
            expected_prev = rec.event_hash_sha256

        return AuditChainVerificationReport(
            tenant_id=str(principal.tenant_id),
            valid=True,
            checked_count=len(events),
            first_corrupted_seq_no=None,
            head_hash_sha256=expected_prev,
            reason="ok",
        )

    def export_ndjson_lines(
        self,
        principal: PrincipalContext,
        *,
        fmt: Literal["gcp_logging", "ndjson"] = "gcp_logging",
        domain_id: str | None = None,
        limit: int = 500,
    ) -> Sequence[str]:
        """Exports tenant-scoped audit events serialized as NDJSON lines."""
        events = self.list_events(principal, domain_id=domain_id, limit=limit)
        lines: list[str] = []
        for rec in events:
            obj = (
                format_gcp_cloud_logging_entry(rec, gcp_project=self._gcp_project)
                if fmt == "gcp_logging"
                else rec.model_dump(mode="json")
            )
            lines.append(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str))
        return lines

    @staticmethod
    def _row_to_record(row: tuple[object, ...]) -> AuditEventRecord:
        raw_payload = row[14]
        payload = (
            json.loads(raw_payload) if isinstance(raw_payload, str) else dict(raw_payload or {})  # type: ignore[call-overload]
        )
        return AuditEventRecord(
            tenant_id=row[0] if isinstance(row[0], UUID) else UUID(str(row[0])),
            seq_no=int(str(row[1])),
            event_id=row[2] if isinstance(row[2], UUID) else UUID(str(row[2])),
            domain_id=str(row[3]),
            event_type=str(row[4]),
            severity=str(row[5]),  # type: ignore[arg-type]
            actor_id=str(row[6]),
            actor_role=str(row[7]),
            resource_uri=str(row[8] or ""),
            outcome=str(row[9]),  # type: ignore[arg-type]
            trace_id=str(row[10] or ""),
            span_id=str(row[11] or ""),
            prev_hash_sha256=str(row[12]),
            event_hash_sha256=str(row[13]),
            payload_json=payload,
            created_at=row[15] if isinstance(row[15], datetime) else datetime.now(UTC),
        )


_AUDIT_LOCK = threading.Lock()
_GLOBAL_AUDIT_SERVICE: AuditEventChainService | None = None


def get_audit_chain_service() -> AuditEventChainService:
    """Returns shared AuditEventChainService backed by PG17 RLS when reachable, else memory."""
    global _GLOBAL_AUDIT_SERVICE
    if _GLOBAL_AUDIT_SERVICE is not None:
        return _GLOBAL_AUDIT_SERVICE
    with _AUDIT_LOCK:
        if _GLOBAL_AUDIT_SERVICE is not None:
            return _GLOBAL_AUDIT_SERVICE
        settings = ScoutSettings()
        try:
            conn_mgr = get_query_pool(settings)
            probe = PrincipalContext(
                user_id="audit_probe",
                role=RoleType.COMPLIANCE_OFFICER,
                allowed_scopes=["audit:compliance"],
            )
            with conn_mgr.scoped_transaction(probe, enable_hnsw_iterative_scan=False) as conn:
                conn.execute("SELECT 1 FROM audit_events LIMIT 1;")
            _GLOBAL_AUDIT_SERVICE = AuditEventChainService(connection_manager=conn_mgr)
        except Exception as exc:
            logger.debug(
                "PG17 audit_events unavailable (%s); using in-memory AuditEventChainService.",
                exc,
            )
            _GLOBAL_AUDIT_SERVICE = AuditEventChainService()
        return _GLOBAL_AUDIT_SERVICE


def set_audit_chain_service(service: AuditEventChainService | None) -> None:
    """Overrides or resets the global AuditEventChainService instance (used in tests)."""
    global _GLOBAL_AUDIT_SERVICE
    with _AUDIT_LOCK:
        _GLOBAL_AUDIT_SERVICE = service


__all__ = [
    "GENESIS_PREV_HASH",
    "AuditChainVerificationReport",
    "AuditEventChainService",
    "AuditEventRecord",
    "AuditOutcome",
    "AuditSeverity",
    "NDJSONSIEMExporter",
    "WebhookSIEMExporter",
    "compute_audit_event_hash",
    "format_gcp_cloud_logging_entry",
    "get_audit_chain_service",
    "set_audit_chain_service",
]
