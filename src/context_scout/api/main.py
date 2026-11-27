"""FastAPI application factory, CORS middleware, OTel tracing, and stateless MCP mounts."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from context_scout.api.dependencies import (
    _REQUEST_PRINCIPAL_SLOT,
    authenticate_headers,
    validate_auth_secret,
)
from context_scout.api.routes import api_router
from context_scout.config import ScoutSettings
from context_scout.mcp_servers.code_repo_mcp import (
    create_code_repo_mcp_http_app,
    create_code_repo_mcp_server,
)
from context_scout.mcp_servers.common import (
    build_mcp_well_known_metadata,
    build_oauth_protected_resource_metadata,
    parse_cloud_trace_to_traceparent,
)
from context_scout.mcp_servers.doc_corpus_mcp import (
    create_doc_corpus_mcp_http_app,
    create_doc_corpus_mcp_server,
)
from context_scout.security.betrvg_privacy import pseudonymize_actor_if_enabled
from context_scout.security.rbac import _CURRENT_PRINCIPAL
from context_scout.telemetry import (
    ATTR_SCOUT_PRINCIPAL_ID,
    ScoutTelemetryTracer,
    extract_trace_context,
    inject_trace_context,
)

logger = logging.getLogger(__name__)


def _sanitize_surrogates_for_json(value: Any) -> Any:
    """Replaces lone UTF-16 surrogates so JSONResponse UTF-8 encoding never raises 500."""
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, Mapping):
        return {
            _sanitize_surrogates_for_json(k): _sanitize_surrogates_for_json(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_surrogates_for_json(item) for item in value]
    return value


def create_app() -> FastAPI:
    """Creates and configures the production Enterprise Context Scout FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Upfront fail-closed validation: abort server startup if secret is insecure or unconfigured
        validate_auth_secret()
        yield

    app = FastAPI(
        title="Enterprise Context Scout API",
        description=(
            "FastAPI SSE Streaming Backend & OpenTelemetry "
            "instrumentation for Enterprise Context Scout"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # 1. CORS Middleware for enterprise web UI (restricted to explicit frontend origins)
    raw_origins = os.environ.get(
        "SCOUT_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    )
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. OpenTelemetry HTTP request tracing middleware (W3C traceparent + Cloud Trace)
    @app.middleware("http")
    async def otel_request_tracing_middleware(
        request: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        tracer = ScoutTelemetryTracer.get_instance()
        route_path = request.url.path
        method = request.method
        carrier = {k.lower(): v for k, v in request.headers.items()}
        if "traceparent" not in carrier and "x-cloud-trace-context" in carrier:
            converted_tp = parse_cloud_trace_to_traceparent(carrier["x-cloud-trace-context"])
            if converted_tp:
                carrier["traceparent"] = converted_tp
        extracted_ctx = extract_trace_context(carrier)
        with tracer.tracer.start_as_current_span(
            f"http.{method.lower()}.{route_path}",
            context=extracted_ctx,
        ) as span:
            span.set_attribute("http.method", method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute(ATTR_SCOUT_PRINCIPAL_ID, "anonymous")
            span.set_attribute("context_scout.principal_id", "anonymous")

            principal_slot: list[Any] = []
            slot_token = _REQUEST_PRINCIPAL_SLOT.set(principal_slot)
            token = _CURRENT_PRINCIPAL.set(None)
            try:
                response: Response = await call_next(request)
                span.set_attribute("http.status_code", response.status_code)
                bound_principal = principal_slot[-1] if principal_slot else _CURRENT_PRINCIPAL.get()
                if bound_principal is not None:
                    safe_pid = pseudonymize_actor_if_enabled(
                        bound_principal.user_id,
                        bound_principal.tenant_id,
                    )
                    span.set_attribute(ATTR_SCOUT_PRINCIPAL_ID, safe_pid)
                    span.set_attribute("context_scout.principal_id", safe_pid)
                out_carrier = inject_trace_context({})
                if out_carrier.get("traceparent") and "traceparent" not in response.headers:
                    response.headers["traceparent"] = out_carrier["traceparent"]
                return response
            finally:
                _CURRENT_PRINCIPAL.reset(token)
                _REQUEST_PRINCIPAL_SLOT.reset(slot_token)

    # 3. Mount versioned API routes under /api/v1
    app.include_router(api_router, prefix="/api/v1")

    mcp_settings = ScoutSettings()
    code_mcp_http_app = create_code_repo_mcp_http_app(
        settings=mcp_settings, streamable_http_path="/mcp"
    )
    doc_mcp_http_app = create_doc_corpus_mcp_http_app(
        settings=mcp_settings, streamable_http_path="/mcp"
    )
    app.mount("/mcp/code", code_mcp_http_app)
    app.mount("/mcp/docs", doc_mcp_http_app)

    # 4. SEP-2243 MCP discovery, RFC 9728 metadata & stateless MCP 2026-07-28 mounts
    @app.get("/.well-known/oauth-protected-resource", include_in_schema=False)
    @app.get("/.well-known/oauth-protected-resource/{subpath:path}", include_in_schema=False)
    def mcp_oauth_protected_resource(request: Request, subpath: str = "") -> JSONResponse:
        active_settings = ScoutSettings()
        origin = str(request.base_url).rstrip("/")
        clean_sub = f"/{subpath.lstrip('/')}" if subpath else "/mcp"
        return JSONResponse(
            build_oauth_protected_resource_metadata(
                origin, settings=active_settings, resource_path=clean_sub
            ),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/.well-known/mcp", include_in_schema=False)
    @app.get("/api/v1/.well-known/mcp", include_in_schema=False)
    def mcp_well_known_discovery(request: Request) -> JSONResponse:
        active_settings = ScoutSettings()
        include_schemas = bool(active_settings.mcp_public_tool_schemas)
        if not include_schemas:
            hdrs = {k.lower(): v for k, v in request.headers.items()}
            if "authorization" in hdrs or "x-goog-iap-jwt-assertion" in hdrs:
                try:
                    authenticate_headers(hdrs)
                    include_schemas = True
                except StarletteHTTPException:
                    include_schemas = False
        code_srv = getattr(code_mcp_http_app, "server", None) or create_code_repo_mcp_server(
            settings=active_settings
        )
        doc_srv = getattr(doc_mcp_http_app, "server", None) or create_doc_corpus_mcp_server(
            settings=active_settings
        )
        code_meta = build_mcp_well_known_metadata(
            code_srv,
            endpoint_path="/mcp/code/mcp",
            include_tool_schemas=include_schemas,
        )
        doc_meta = build_mcp_well_known_metadata(
            doc_srv,
            endpoint_path="/mcp/docs/mcp",
            include_tool_schemas=include_schemas,
        )
        return JSONResponse(
            {
                **code_meta,
                "server": "enterprise-context-scout",
                "servers": [code_meta, doc_meta],
            },
            headers={"Cache-Control": "no-store"},
        )

    # 5. Top-level convenience route
    @app.get("/health", include_in_schema=False)
    def root_health_redirect() -> RedirectResponse:
        return RedirectResponse(url="/api/v1/health")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del request
        safe_errors = _sanitize_surrogates_for_json(jsonable_encoder(exc.errors()))
        return JSONResponse(status_code=422, content={"detail": safe_errors})

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
        del request
        if exc.status_code in {204, 304}:
            return Response(status_code=exc.status_code, headers=exc.headers)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": _sanitize_surrogates_for_json(exc.detail)},
            headers=exc.headers,
        )

    return app


# Module-level application instance for ASGI servers (uvicorn context_scout.api.main:app)
app = create_app()
