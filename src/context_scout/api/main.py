"""FastAPI application factory, CORS middleware, and OpenTelemetry instrumentation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from context_scout.api.routes import api_router
from context_scout.telemetry import ScoutTelemetryTracer

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Creates and configures the production Enterprise Context Scout FastAPI application."""
    app = FastAPI(
        title="Enterprise Context Scout API",
        description=(
            "FastAPI SSE Streaming Backend & OpenTelemetry "
            "instrumentation for Enterprise Context Scout"
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. CORS Middleware for enterprise web UI
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. OpenTelemetry HTTP request tracing middleware
    @app.middleware("http")
    async def otel_request_tracing_middleware(
        request: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        tracer = ScoutTelemetryTracer.get_instance()
        route_path = request.url.path
        method = request.method
        with tracer.tracer.start_as_current_span(f"http.{method.lower()}.{route_path}") as span:
            span.set_attribute("http.method", method)
            span.set_attribute("http.url", str(request.url))
            principal_user = request.headers.get("X-Principal-User-Id", "anonymous")
            span.set_attribute("context_scout.principal_id", principal_user)

            response: Response = await call_next(request)
            span.set_attribute("http.status_code", response.status_code)
            return response

    # 3. Mount versioned API routes under /api/v1
    app.include_router(api_router, prefix="/api/v1")

    # 4. Top-level convenience route
    @app.get("/health", include_in_schema=False)
    def root_health_redirect() -> RedirectResponse:
        return RedirectResponse(url="/api/v1/health")

    return app


# Module-level application instance for ASGI servers (uvicorn context_scout.api.main:app)
app = create_app()
