"""API routes aggregator module."""

from fastapi import APIRouter

from context_scout.api.routes.admin import router as admin_router
from context_scout.api.routes.auth import router as auth_router
from context_scout.api.routes.domains import router as domains_router
from context_scout.api.routes.handoff import router as handoff_router
from context_scout.api.routes.health import router as health_router
from context_scout.api.routes.scim import router as scim_router
from context_scout.api.routes.scout import router as scout_router
from context_scout.api.routes.telemetry import router as telemetry_router
from context_scout.api.routes.webhooks import router as webhooks_router
from context_scout.api.routes.workspace import router as workspace_router

_ERROR_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"detail": {}},
    "required": ["detail"],
}

api_router = APIRouter(
    responses={
        400: {
            "description": "Bad Request",
            "content": {"application/json": {"schema": _ERROR_SCHEMA}},
        },
        401: {
            "description": "Unauthorized",
            "content": {"application/json": {"schema": _ERROR_SCHEMA}},
        },
        403: {
            "description": "Forbidden",
            "content": {"application/json": {"schema": _ERROR_SCHEMA}},
        },
        404: {
            "description": "Not Found",
            "content": {"application/json": {"schema": _ERROR_SCHEMA}},
        },
    }
)

api_router.include_router(auth_router)
api_router.include_router(scim_router)
api_router.include_router(admin_router)
api_router.include_router(health_router)
api_router.include_router(domains_router)
api_router.include_router(workspace_router)
api_router.include_router(scout_router)
api_router.include_router(handoff_router)
api_router.include_router(telemetry_router)
api_router.include_router(webhooks_router)

__all__ = ["api_router"]
