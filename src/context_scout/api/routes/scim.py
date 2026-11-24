"""SCIM 2.0 Directory Provisioning Routes (RFC 7643 / RFC 7644, M5-01)."""

from __future__ import annotations

import hmac
import os
from typing import Annotated, Any
from uuid import UUID

import psycopg.errors
from fastapi import APIRouter, Depends, Header, HTTPException, Response, status

from context_scout.schemas.contracts import DEFAULT_TENANT_ID
from context_scout.security.audit_chain import get_audit_chain_service
from context_scout.security.scim import SCIMDirectoryService, get_scim_directory_service

router = APIRouter(prefix="/scim/v2", tags=["SCIM"])


def require_scim_context(
    authorization: Annotated[str | None, Header()] = None,
    x_scout_tenant_id: Annotated[str | None, Header(alias="X-Scout-Tenant-Id")] = None,
) -> UUID:
    """Authenticates fail-closed SCIM 2.0 Bearer token and resolves tenant UUID."""
    expected = (os.environ.get("SCOUT_SCIM_BEARER_TOKEN") or "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SCIM 2.0 provisioning is disabled (SCOUT_SCIM_BEARER_TOKEN is unset).",
        )
    if not authorization or not authorization.strip().lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="SCIM 2.0 endpoint requires Authorization: Bearer <token>.",
        )
    provided = authorization.strip()[7:].strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid SCIM 2.0 Bearer token.",
        )

    if x_scout_tenant_id and x_scout_tenant_id.strip():
        try:
            return UUID(x_scout_tenant_id.strip())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid X-Scout-Tenant-Id UUID: {x_scout_tenant_id!r}.",
            ) from None
    return DEFAULT_TENANT_ID


@router.get("/Users")
def list_scim_users(
    tenant_id: Annotated[UUID, Depends(require_scim_context)],
    scim: Annotated[SCIMDirectoryService, Depends(get_scim_directory_service)],
) -> dict[str, Any]:
    users = scim.list_users(tenant_id)
    groups = scim.list_groups(tenant_id)
    resources = []
    for u in users:
        u_keys = {k.lower() for k in (u.scim_user_id, u.user_name, u.external_id, u.email) if k}
        member_of = [
            g
            for g in groups
            if u_keys & {m.strip().lower() for m in g.member_user_ids if m.strip()}
        ]
        resources.append(u.to_scim_resource(member_of))
    return {
        "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
        "totalResults": len(resources),
        "itemsPerPage": len(resources),
        "startIndex": 1,
        "Resources": resources,
    }


def _safe_error_detail(exc: Exception) -> str:
    return str(exc).encode("utf-8", errors="replace").decode("utf-8")


@router.post("/Users", status_code=status.HTTP_201_CREATED)
def create_scim_user(
    payload: dict[str, Any],
    tenant_id: Annotated[UUID, Depends(require_scim_context)],
    scim: Annotated[SCIMDirectoryService, Depends(get_scim_directory_service)],
) -> dict[str, Any]:
    try:
        rec = scim.upsert_user(tenant_id, payload)
    except (ValueError, psycopg.errors.DataError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_safe_error_detail(exc)
        ) from None
    get_audit_chain_service().record_event(
        tenant_id=tenant_id,
        event_type="scim.user.upsert",
        severity="INFO",
        actor_id="scim_provisioner",
        actor_role="compliance_officer",
        resource_uri=f"/api/v1/scim/v2/Users/{rec.scim_user_id}",
        outcome="success",
        payload={"user_name": rec.user_name, "active": rec.active},
    )
    return rec.to_scim_resource()


@router.get("/Users/{user_id}")
def get_scim_user(
    user_id: str,
    tenant_id: Annotated[UUID, Depends(require_scim_context)],
    scim: Annotated[SCIMDirectoryService, Depends(get_scim_directory_service)],
) -> dict[str, Any]:
    try:
        rec = scim.get_user(tenant_id, user_id)
    except (ValueError, psycopg.errors.DataError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_safe_error_detail(exc)
        ) from None
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SCIM User not found.")
    return rec.to_scim_resource()


@router.patch("/Users/{user_id}")
def patch_scim_user(
    user_id: str,
    payload: dict[str, Any],
    tenant_id: Annotated[UUID, Depends(require_scim_context)],
    scim: Annotated[SCIMDirectoryService, Depends(get_scim_directory_service)],
) -> dict[str, Any]:
    ops = payload["Operations"] if "Operations" in payload else payload.get("operations")
    if not isinstance(ops, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SCIM PatchOp payload must include an 'Operations' array.",
        )
    try:
        rec = scim.patch_user(tenant_id, user_id, ops)
    except (ValueError, psycopg.errors.DataError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_safe_error_detail(exc)
        ) from None
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SCIM User not found.")
    get_audit_chain_service().record_event(
        tenant_id=tenant_id,
        event_type="scim.user.patch",
        severity="NOTICE" if not rec.active else "INFO",
        actor_id="scim_provisioner",
        actor_role="compliance_officer",
        resource_uri=f"/api/v1/scim/v2/Users/{rec.scim_user_id}",
        outcome="success",
        payload={"active": rec.active, "ops_count": len(ops)},
    )
    return rec.to_scim_resource()


@router.delete("/Users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scim_user(
    user_id: str,
    tenant_id: Annotated[UUID, Depends(require_scim_context)],
    scim: Annotated[SCIMDirectoryService, Depends(get_scim_directory_service)],
) -> Response:
    try:
        deleted = scim.delete_user(tenant_id, user_id)
    except (ValueError, psycopg.errors.DataError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_safe_error_detail(exc)
        ) from None
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SCIM User not found.")
    get_audit_chain_service().record_event(
        tenant_id=tenant_id,
        event_type="scim.user.delete",
        severity="NOTICE",
        actor_id="scim_provisioner",
        actor_role="compliance_officer",
        resource_uri=f"/api/v1/scim/v2/Users/{user_id}",
        outcome="success",
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/Groups")
def list_scim_groups(
    tenant_id: Annotated[UUID, Depends(require_scim_context)],
    scim: Annotated[SCIMDirectoryService, Depends(get_scim_directory_service)],
) -> dict[str, Any]:
    groups = scim.list_groups(tenant_id)
    resources = [g.to_scim_resource() for g in groups]
    return {
        "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
        "totalResults": len(resources),
        "itemsPerPage": len(resources),
        "startIndex": 1,
        "Resources": resources,
    }


@router.post("/Groups", status_code=status.HTTP_201_CREATED)
def create_scim_group(
    payload: dict[str, Any],
    tenant_id: Annotated[UUID, Depends(require_scim_context)],
    scim: Annotated[SCIMDirectoryService, Depends(get_scim_directory_service)],
) -> dict[str, Any]:
    try:
        rec = scim.upsert_group(tenant_id, payload)
    except (ValueError, psycopg.errors.DataError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_safe_error_detail(exc)
        ) from None
    get_audit_chain_service().record_event(
        tenant_id=tenant_id,
        event_type="scim.group.upsert",
        severity="INFO",
        actor_id="scim_provisioner",
        actor_role="compliance_officer",
        resource_uri=f"/api/v1/scim/v2/Groups/{rec.scim_group_id}",
        outcome="success",
        payload={"display_name": rec.display_name, "members_count": len(rec.member_user_ids)},
    )
    return rec.to_scim_resource()


@router.get("/Groups/{group_id}")
def get_scim_group(
    group_id: str,
    tenant_id: Annotated[UUID, Depends(require_scim_context)],
    scim: Annotated[SCIMDirectoryService, Depends(get_scim_directory_service)],
) -> dict[str, Any]:
    try:
        rec = scim.get_group(tenant_id, group_id)
    except (ValueError, psycopg.errors.DataError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_safe_error_detail(exc)
        ) from None
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SCIM Group not found.")
    return rec.to_scim_resource()


@router.patch("/Groups/{group_id}")
def patch_scim_group(
    group_id: str,
    payload: dict[str, Any],
    tenant_id: Annotated[UUID, Depends(require_scim_context)],
    scim: Annotated[SCIMDirectoryService, Depends(get_scim_directory_service)],
) -> dict[str, Any]:
    ops = payload["Operations"] if "Operations" in payload else payload.get("operations")
    if not isinstance(ops, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SCIM PatchOp payload must include an 'Operations' array.",
        )
    try:
        rec = scim.patch_group(tenant_id, group_id, ops)
    except (ValueError, psycopg.errors.DataError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_safe_error_detail(exc)
        ) from None
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SCIM Group not found.")
    get_audit_chain_service().record_event(
        tenant_id=tenant_id,
        event_type="scim.group.patch",
        severity="INFO",
        actor_id="scim_provisioner",
        actor_role="compliance_officer",
        resource_uri=f"/api/v1/scim/v2/Groups/{rec.scim_group_id}",
        outcome="success",
        payload={"members_count": len(rec.member_user_ids), "ops_count": len(ops)},
    )
    return rec.to_scim_resource()


@router.delete("/Groups/{group_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scim_group(
    group_id: str,
    tenant_id: Annotated[UUID, Depends(require_scim_context)],
    scim: Annotated[SCIMDirectoryService, Depends(get_scim_directory_service)],
) -> Response:
    try:
        deleted = scim.delete_group(tenant_id, group_id)
    except (ValueError, psycopg.errors.DataError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_safe_error_detail(exc)
        ) from None
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SCIM Group not found.")
    get_audit_chain_service().record_event(
        tenant_id=tenant_id,
        event_type="scim.group.delete",
        severity="NOTICE",
        actor_id="scim_provisioner",
        actor_role="compliance_officer",
        resource_uri=f"/api/v1/scim/v2/Groups/{group_id}",
        outcome="success",
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["require_scim_context", "router"]
