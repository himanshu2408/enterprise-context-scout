"""Authentication session exchange and principal verification endpoints (SEC-04)."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from context_scout.api.dependencies import (
    DEFAULT_ROLE_SCOPES,
    create_signed_principal_token,
    get_principal_context,
    verify_signed_principal_token,
)
from context_scout.api.schemas import AuthSessionRequest, AuthSessionResponse
from context_scout.config import is_dev_auth_session_allowed
from context_scout.schemas.contracts import PrincipalContext
from context_scout.security.audit_chain import get_audit_chain_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Auth"])

SESSION_TTL_SECONDS = 3600


@router.post("/auth/session", response_model=AuthSessionResponse)
def create_auth_session(request: AuthSessionRequest) -> AuthSessionResponse:
    """Mints an HMAC-SHA256 signed Bearer token strictly in allowlisted dev environments."""
    if not is_dev_auth_session_allowed():
        safe_actor = request.user_id.replace("\x00", "").strip()[:128] or "anonymous"
        try:
            get_audit_chain_service().record_event(
                tenant_id=request.tenant_id,
                event_type="auth.session.mint",
                severity="WARNING",
                actor_id=safe_actor,
                actor_role=request.role.value,
                resource_uri="/api/v1/auth/session",
                outcome="deny",
                payload={"reason": "blocked_outside_development"},
            )
        except Exception as exc:
            logger.warning("Failed recording denied auth.session.mint audit event: %s", exc)
        raise HTTPException(
            status_code=403,
            detail=(
                "Interactive persona token minting is restricted to development environments and "
                "disabled in production; supply an external Cloud IAP or OIDC Bearer token "
                "(SEC-04)."
            ),
        )

    clean_user_id = request.user_id.strip()
    if not clean_user_id:
        raise HTTPException(status_code=400, detail="user_id must not be blank.")

    if request.allowed_scopes is None:
        scopes = list(DEFAULT_ROLE_SCOPES[request.role])
    else:
        scopes = [s.strip() for s in request.allowed_scopes if s.strip()]
        if not scopes:
            raise HTTPException(
                status_code=400,
                detail="allowed_scopes must contain at least one non-empty scope token.",
            )
        for scope in scopes:
            if scope.lower().startswith("system:"):
                raise HTTPException(
                    status_code=403,
                    detail=f"Reserved internal scope '{scope}' cannot be requested externally.",
                )
            if "," in scope or ";" in scope or any(ch.isspace() for ch in scope):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid RBAC scope token '{scope}' "
                        "(no commas, semicolons, or whitespace)."
                    ),
                )

    try:
        token = create_signed_principal_token(
            user_id=clean_user_id,
            role=request.role,
            allowed_scopes=scopes,
            ttl_seconds=SESSION_TTL_SECONDS,
            max_session_paid_tokens=request.max_session_paid_tokens,
            tenant_id=request.tenant_id,
        )
        principal = verify_signed_principal_token(token)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Failed minting signed principal token (%s): %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to mint signed principal token: {exc}",
        ) from exc

    get_audit_chain_service().record_event(
        principal=principal,
        event_type="auth.session.mint",
        severity="INFO",
        resource_uri="/api/v1/auth/session",
        outcome="allow",
        payload={"scopes_count": len(principal.allowed_scopes)},
    )

    return AuthSessionResponse(
        access_token=token,
        token_type="Bearer",
        expires_in_seconds=SESSION_TTL_SECONDS,
        principal=principal,
    )


@router.get("/auth/me", response_model=PrincipalContext)
def get_authenticated_principal(
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
) -> PrincipalContext:
    """Returns the verified PrincipalContext extracted from the signed Bearer token."""
    return principal
