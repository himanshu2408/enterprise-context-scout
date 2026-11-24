"""RBAC & RLS-scoped Telemetry, ROI Metrics, and Prometheus/GMP Export Endpoints."""

from __future__ import annotations

import hmac
import json
import logging
import os
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import PlainTextResponse

from context_scout.api.dependencies import (
    authenticate_headers,
    get_principal_context,
    get_repo_root,
    get_repository,
)
from context_scout.api.schemas import (
    AggregateROISummary,
    ROIMetricsResponse,
)
from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.db.session import get_query_pool
from context_scout.schemas.contracts import (
    ExecutionTelemetryRecord,
    PrincipalContext,
    RoleType,
)
from context_scout.telemetry.prometheus_metrics import get_prometheus_registry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Telemetry"])


def _require_non_unprivileged_principal(principal: PrincipalContext) -> None:
    if all(s.startswith("unprivileged:") for s in principal.allowed_scopes):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unprivileged principals lack permission to query tenant ROI telemetry.",
        )


@router.get("/telemetry/roi", response_model=ROIMetricsResponse)
def get_roi_metrics(
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
    repo: Annotated[Any, Depends(get_repository)],
) -> ROIMetricsResponse:
    """Returns RBAC + RLS-scoped ROI metrics and Milestone 6/7 evaluation reports."""
    _require_non_unprivileged_principal(principal)
    repo_root = get_repo_root()

    m6_report: dict[str, Any] | None = None
    m6_path = repo_root / "reports" / "m6_eval_report.json"
    if m6_path.exists():
        try:
            with open(m6_path, encoding="utf-8") as f:
                m6_report = json.load(f)
        except Exception as exc:
            logger.warning(
                "Failed loading M6 eval report from %s (%s): %s",
                m6_path,
                type(exc).__name__,
                exc,
            )
            m6_report = None

    m7_report: dict[str, Any] | None = None
    m7_path = repo_root / "reports" / "m7_qlora_benchmark_report.json"
    if m7_path.exists():
        try:
            with open(m7_path, encoding="utf-8") as f:
                m7_report = json.load(f)
        except Exception as exc:
            logger.warning(
                "Failed loading M7 benchmark report from %s (%s): %s",
                m7_path,
                type(exc).__name__,
                exc,
            )
            m7_report = None

    records: list[ExecutionTelemetryRecord] = []
    total_queries = 0
    total_local = 0
    total_pack = 0
    total_saved = 0
    total_usd = 0.0
    telemetry_source: str

    if isinstance(repo, InMemoryRLSRepository):
        records = [r for r in repo._telemetry.values() if r.tenant_id == principal.tenant_id]
        total_queries = len(records)
        total_local = sum(r.local_tokens_read for r in records)
        total_pack = sum(r.context_pack_tokens_sent for r in records)
        total_saved = sum(r.paid_tokens_saved for r in records)
        total_usd = sum(r.estimated_usd_saved for r in records)
        telemetry_source = "in_memory_session"
    else:
        settings = getattr(repo, "settings", None) or ScoutSettings()
        query_mgr = getattr(repo, "connection_manager", None) or get_query_pool(settings)
        try:
            with query_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*),
                           COALESCE(SUM(local_tokens_read), 0),
                           COALESCE(SUM(context_pack_tokens_sent), 0),
                           COALESCE(SUM(paid_tokens_saved), 0),
                           COALESCE(SUM(estimated_usd_saved), 0.0)
                    FROM execution_telemetry;
                    """
                ).fetchone()
                if row is not None:
                    total_queries = int(str(row[0] or 0))
                    total_local = int(str(row[1] or 0))
                    total_pack = int(str(row[2] or 0))
                    total_saved = int(str(row[3] or 0))
                    total_usd = float(str(row[4] or 0.0))
            telemetry_source = "postgresql_live"
        except Exception as exc:
            logger.warning(
                "PostgreSQL telemetry ROI aggregation failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PostgreSQL RLS telemetry aggregation is currently unavailable.",
            ) from None

    ratio_pct = round((total_saved / max(1, total_local)) * 100.0, 2) if total_local > 0 else 0.0
    qlora_eliminated_total = float(total_queries * 1634.0)

    summary = AggregateROISummary(
        total_queries=total_queries,
        total_local_tokens_explored=total_local,
        total_pack_tokens_sent=total_pack,
        total_paid_tokens_saved=total_saved,
        total_estimated_usd_saved=round(total_usd, 4),
        frontier_savings_ratio_pct=ratio_pct,
        qlora_overhead_eliminated_per_query=1634.0,
        total_qlora_overhead_eliminated=qlora_eliminated_total,
        telemetry_source=telemetry_source,  # type: ignore[arg-type]
    )

    return ROIMetricsResponse(
        summary=summary,
        m6_evaluation_report=m6_report,
        m7_qlora_benchmark_report=m7_report,
        timestamp=datetime.now(UTC).isoformat(),
    )


@router.get(
    "/telemetry/metrics",
    response_class=PlainTextResponse,
    responses={
        200: {
            "description": "Prometheus 0.0.4 text exposition metrics",
            "content": {"text/plain": {"schema": {"type": "string"}}},
        }
    },
)
def get_prometheus_metrics(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> PlainTextResponse:
    """Exports Prometheus 0.0.4 / Google Managed Prometheus (GMP) retrieval-drift metrics."""
    metrics_token = (os.environ.get("SCOUT_METRICS_BEARER_TOKEN") or "").strip()
    if metrics_token and authorization and authorization.strip().lower().startswith("bearer "):
        provided = authorization.strip()[7:].strip()
        if provided and hmac.compare_digest(provided, metrics_token):
            return PlainTextResponse(
                content=get_prometheus_registry().render_prometheus_text(),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

    principal = authenticate_headers(dict(request.headers))
    if principal.role not in {RoleType.ENGINEER, RoleType.COMPLIANCE_OFFICER}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Metrics scrape restricted to Engineer, Compliance Officer, or GMP scraper.",
        )

    return PlainTextResponse(
        content=get_prometheus_registry().render_prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
