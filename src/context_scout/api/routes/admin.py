"""RBAC & RLS-Scoped Admin Purge & Audit Chain Verification/Export Routes (M5-01)."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from context_scout.api.dependencies import get_principal_context, get_repository
from context_scout.api.schemas import (
    _reject_nul_or_surrogate_str,
    reject_nul_and_surrogates_recursive,
)
from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
from context_scout.db.session import get_ingestion_pool
from context_scout.ops.artifact_purger import ArtifactPurgeCoordinator
from context_scout.schemas.contracts import PrincipalContext, RoleType
from context_scout.security.audit_chain import (
    AuditChainVerificationReport,
    get_audit_chain_service,
)

router = APIRouter(prefix="/admin", tags=["Admin"])


class AdminPurgeRequest(BaseModel):
    """Request payload for <60s multi-store secret & artifact purge."""

    domain_id: str = Field(min_length=1, max_length=128)
    tenant_id: str | None = None
    repo_name: str | None = None
    file_path: str | None = None
    commit_sha: str | None = None
    blob_sha: str | None = None
    doc_id: str | None = None
    confluence_page_id: str | None = None
    jira_issue_key: str | None = None
    object_keys: list[str] = Field(default_factory=list)
    include_derived_wiki: bool = True

    @field_validator(
        "domain_id",
        "tenant_id",
        "repo_name",
        "file_path",
        "commit_sha",
        "blob_sha",
        "doc_id",
        "confluence_page_id",
        "jira_issue_key",
    )
    @classmethod
    def _validate_no_nul_str(cls, v: str | None, info: Any) -> str | None:
        return _reject_nul_or_surrogate_str(v, info.field_name or "selector")

    @field_validator("object_keys")
    @classmethod
    def _validate_no_nul_object_keys(cls, v: list[str]) -> list[str]:
        for key in v:
            _reject_nul_or_surrogate_str(key, "object_keys")
        return v

    @model_validator(mode="after")
    def _validate_no_nul_or_surrogates_recursive(self) -> Self:
        reject_nul_and_surrogates_recursive(self, "AdminPurgeRequest")
        return self


class AdminPurgeResponse(BaseModel):
    """Verified cross-store purge telemetry response."""

    domain_id: str
    tenant_id: str
    selectors: dict[str, Any]
    pg17_deleted: dict[str, int]
    zoekt_deleted: int
    object_store_deleted: list[str]
    permission_cache_invalidated: bool
    residue_found: int
    residue_details: dict[str, int]
    elapsed_seconds: float
    sla_seconds: float
    sla_met: bool
    warnings: list[str] = Field(default_factory=list)


def _require_compliance_or_purge_rbac(
    principal: PrincipalContext, *, require_purge_perm: bool = False
) -> None:
    allowed = (
        principal.role == RoleType.COMPLIANCE_OFFICER
        or "audit:compliance" in principal.allowed_scopes
        or (require_purge_perm and "admin:purge" in principal.allowed_scopes)
    )
    if not allowed:
        get_audit_chain_service().record_event(
            principal=principal,
            event_type="admin.rbac.deny",
            severity="WARNING",
            resource_uri="/api/v1/admin/purge" if require_purge_perm else "/api/v1/admin/audit",
            outcome="deny",
            payload={"role": principal.role.value},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Administrative compliance endpoint restricted to Compliance Officer "
                "or audit:compliance / admin:purge principals."
            ),
        )


def get_admin_purge_coordinator(
    repo: Annotated[Any, Depends(get_repository)],
) -> ArtifactPurgeCoordinator:
    """Builds an RLS-scoped ArtifactPurgeCoordinator (never uses db_admin_url)."""
    if isinstance(repo, InMemoryRLSRepository):
        return ArtifactPurgeCoordinator(repository=repo)
    settings = getattr(repo, "settings", None) or ScoutSettings()
    ingestion_mgr = get_ingestion_pool(settings)
    zoekt_client: Any = None
    zoekt_ep = (settings.zoekt_endpoint or "").strip()
    if zoekt_ep:
        from context_scout.retrieval.zoekt_client import ZoektGrpcJsonClient

        zoekt_client = ZoektGrpcJsonClient(
            zoekt_ep,
            protocol=settings.zoekt_protocol,
            timeout_seconds=settings.zoekt_timeout_seconds,
            connection_manager=ingestion_mgr,
        )
    return ArtifactPurgeCoordinator(
        repository=PostgresRLSRepository(),
        connection_manager=ingestion_mgr,
        zoekt_index=zoekt_client,
    )


@router.post("/purge", response_model=AdminPurgeResponse)
def purge_artifacts(
    request: AdminPurgeRequest,
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
    coordinator: Annotated[ArtifactPurgeCoordinator, Depends(get_admin_purge_coordinator)],
) -> AdminPurgeResponse:
    """Executes RBAC + RLS-scoped <60s multi-store purge with zero-residue verification."""
    _require_compliance_or_purge_rbac(principal, require_purge_perm=True)
    if request.tenant_id is not None and request.tenant_id.strip():
        try:
            requested_tid = UUID(request.tenant_id.strip())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid tenant_id UUID format.",
            ) from None
        if requested_tid != principal.tenant_id:
            get_audit_chain_service().record_event(
                principal=principal,
                domain_id=request.domain_id,
                event_type="admin.purge.cross_tenant_deny",
                severity="WARNING",
                resource_uri="/api/v1/admin/purge",
                outcome="deny",
                payload={"requested_tenant_id": str(requested_tid)},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Cross-tenant purge denied: request tenant_id does not match "
                    "authenticated principal tenant."
                ),
            )
    try:
        report = coordinator.purge(
            principal=principal,
            domain_id=request.domain_id,
            repo_name=request.repo_name,
            file_path=request.file_path,
            commit_sha=request.commit_sha,
            blob_sha=request.blob_sha,
            doc_id=request.doc_id,
            confluence_page_id=request.confluence_page_id,
            jira_issue_key=request.jira_issue_key,
            object_keys=request.object_keys,
            include_derived_wiki=request.include_derived_wiki,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from None
    except ValueError as exc:
        msg = str(exc)
        code = (
            status.HTTP_403_FORBIDDEN
            if ("tenant" in msg.lower() or "cross-tenant" in msg.lower())
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail=msg) from None

    get_audit_chain_service().record_event(
        principal=principal,
        domain_id=report.domain_id,
        event_type="admin.purge",
        severity="NOTICE",
        resource_uri="/api/v1/admin/purge",
        outcome="success" if report.residue_found == 0 else "failure",
        payload={
            "pg17_deleted": report.pg17_deleted,
            "zoekt_deleted": report.zoekt_deleted,
            "residue_found": report.residue_found,
            "elapsed_seconds": report.elapsed_seconds,
        },
    )
    return AdminPurgeResponse(**report.to_dict())


@router.get("/audit/verify", response_model=AuditChainVerificationReport)
def verify_audit_chain(
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
) -> AuditChainVerificationReport:
    """Verifies SHA-256 hash-chain integrity across tenant-scoped audit_events under RLS."""
    _require_compliance_or_purge_rbac(principal, require_purge_perm=False)
    return get_audit_chain_service().verify_chain_integrity(principal)


@router.get(
    "/audit/export",
    response_class=PlainTextResponse,
    responses={
        200: {
            "description": "Audit events exported as NDJSON",
            "content": {"application/x-ndjson": {"schema": {"type": "string"}}},
        }
    },
)
def export_audit_events(
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
    fmt: Annotated[Literal["gcp_logging", "ndjson"], Query(alias="format")] = "gcp_logging",
    domain_id: Annotated[str | None, Query(max_length=128)] = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> PlainTextResponse:
    """Exports tenant-scoped audit_events as GCP Cloud Logging structured JSON or NDJSON."""
    _require_compliance_or_purge_rbac(principal, require_purge_perm=False)
    if domain_id is not None and "\x00" in domain_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="domain_id must not contain NUL bytes.",
        )
    lines = get_audit_chain_service().export_ndjson_lines(
        principal,
        fmt=fmt,
        domain_id=domain_id,
        limit=limit,
    )
    body = ("\n".join(lines) + "\n") if lines else ""
    return PlainTextResponse(content=body, media_type="application/x-ndjson")


__all__ = [
    "AdminPurgeRequest",
    "AdminPurgeResponse",
    "get_admin_purge_coordinator",
    "router",
]
