"""API routes aggregator module."""

from fastapi import APIRouter

from context_scout.api.routes.domains import router as domains_router
from context_scout.api.routes.handoff import router as handoff_router
from context_scout.api.routes.health import router as health_router
from context_scout.api.routes.scout import router as scout_router
from context_scout.api.routes.telemetry import router as telemetry_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(domains_router)
api_router.include_router(scout_router)
api_router.include_router(handoff_router)
api_router.include_router(telemetry_router)

__all__ = ["api_router"]
