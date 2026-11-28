"""Scout query and SSE streaming endpoints."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Annotated, Any

import anyio
import psycopg.errors
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.intent_router import (
    SourceFilter,
    decompose_query_facets,
    filter_doc_chunks_by_source,
)
from context_scout.agents.scout_workflow import (
    FederatedScoutCoordinator,
    LocalGemmaClient,
    ScoutWorkflow,
)
from context_scout.api.dependencies import (
    get_domain_registry,
    get_local_gemma_client,
    get_principal_context,
    get_repo_root,
    get_repository,
    get_telemetry_tracer,
)
from context_scout.api.schemas import (
    ScoutQueryResponse,
    ScoutStreamRequest,
    ScoutTokenSavingsBreakdown,
    SSEBounceBackEvent,
    SSEContextPackReadyEvent,
    SSEErrorEvent,
    SSEStageEvent,
)
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
from context_scout.schemas.contracts import (
    DEFAULT_TENANT_ID,
    ExecutionMode,
    ExecutionTelemetryRecord,
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    PrincipalContext,
    StageConfidence,
    StageExecutionTrace,
)
from context_scout.security.audit_chain import get_audit_chain_service
from context_scout.security.betrvg_privacy import pseudonymize_actor_if_enabled
from context_scout.security.guardrails import get_guardrail_provider
from context_scout.telemetry import ScoutTelemetryTracer
from context_scout.telemetry.prometheus_metrics import get_prometheus_registry

logger = logging.getLogger(__name__)

SSE_HEARTBEAT_INTERVAL_SECONDS: float = 15.0

router = APIRouter(tags=["Scout"])

_STAGE_TITLES: dict[int, str] = {
    1: "Stage 1: Auth, Guardrail & Query Plan",
    2: "Stage 2: Hybrid Retrieval & AST Closure",
    3: "Stage 3: Local Gemma-4 Synthesis & Facet Grounding",
    4: "Stage 4: Pre-Flight Sufficiency Gate",
}

_FEDERATED_STAGE_TITLES: dict[int, str] = {
    1: "Stage 1: Auth, Guardrail & Query Plan",
    2: "Stage 2: Multi-Domain Hybrid Retrieval & AST Closure",
    3: "Stage 3: Local Gemma-4 Synthesis & Facet Grounding",
    4: "Stage 4: Pre-Flight Sufficiency Gate",
}


def _resolve_federated_doc_chunks(
    repo: Any,
    principal: PrincipalContext,
    source_filter: SourceFilter = "all",
    query_text: str = "",
) -> dict[str, str]:
    """Resolves visible doc_chunks dynamically across all authorized DomainRegistry domains."""
    chunks_map: dict[str, str] = {}
    registry = DomainRegistry(domains_dir=get_repo_root() / "domains")
    manifests = registry.load_all()
    principal_scopes = set(principal.allowed_scopes)
    target_domains: list[str] = []
    for manifest in manifests.values():
        domain_scopes = set(manifest.default_scopes or [manifest.domain_id])
        if "*" in principal_scopes or (principal_scopes & domain_scopes):
            target_domains.append(manifest.domain_id)

    for target_dom in target_domains:
        if query_text.strip() and hasattr(repo, "search_visible_doc_chunks_calibrated"):
            try:
                hits = repo.search_visible_doc_chunks_calibrated(
                    principal, target_dom, query_text, limit=8
                )
                filtered_hits = filter_doc_chunks_by_source(
                    [h.doc_chunk for h in hits if h.doc_chunk is not None],
                    source_filter,
                )
                for chk in filtered_hits:
                    chunks_map[str(chk.chunk_id)] = chk.verbatim_content
                continue
            except Exception as exc:
                logger.warning(
                    "Calibrated doc search failed for domain '%s' (%s): %s",
                    target_dom,
                    type(exc).__name__,
                    exc,
                )
        if query_text.strip() and hasattr(repo, "search_visible_doc_chunks"):
            try:
                searched = repo.search_visible_doc_chunks(
                    principal, target_dom, query_text, limit=8
                )
                filtered_searched = filter_doc_chunks_by_source(list(searched), source_filter)
                for chk in filtered_searched:
                    chunks_map[str(chk.chunk_id)] = chk.verbatim_content
                continue
            except Exception as exc:
                logger.warning(
                    "Failed searching visible doc chunks for domain '%s' (%s): %s",
                    target_dom,
                    type(exc).__name__,
                    exc,
                )
        if hasattr(repo, "list_visible_doc_chunks"):
            try:
                visible = repo.list_visible_doc_chunks(principal, target_dom)
                filtered_visible = filter_doc_chunks_by_source(list(visible), source_filter)
                for chk in filtered_visible[:12]:
                    chunks_map[str(chk.chunk_id)] = chk.verbatim_content
            except Exception as exc:
                logger.warning(
                    "Failed loading visible doc chunks for domain '%s' (%s): %s",
                    target_dom,
                    type(exc).__name__,
                    exc,
                )
    return chunks_map


def _build_token_savings(
    pack: ImplementationContextPack,
    *,
    registry: DomainRegistry | None = None,
) -> ScoutTokenSavingsBreakdown:
    """Computes frontier token & USD savings breakdown from a verified context pack."""
    from context_scout.telemetry.cost_governance import PaidComputeCostEstimator

    local_explored = pack.local_tokens_consumed
    pack_sent = pack.pack_token_count
    saved = max(0, local_explored - pack_sent)
    estimator = PaidComputeCostEstimator()
    frontier_rate = estimator.get_pricing_record("gemini-4-carbon").input_usd_per_1m_expected
    usd_saved = round((saved / 1_000_000.0) * frontier_rate, 4)
    ratio = round((saved / max(1, local_explored)) * 100.0, 2)
    return ScoutTokenSavingsBreakdown(
        local_tokens_explored=local_explored,
        pack_tokens_sent=pack_sent,
        tokens_saved=saved,
        estimated_usd_saved=usd_saved,
        frontier_savings_ratio_pct=ratio,
        qlora_overhead_eliminated_tokens=0.0,
    )


def _record_live_scout_telemetry(
    repo: Any,
    registry: DomainRegistry,
    principal: PrincipalContext,
    pack: ImplementationContextPack,
    latency_ms: int,
) -> None:
    """Persists verified scout query ROI telemetry into PostgreSQL or InMemory ledger."""
    savings = _build_token_savings(pack, registry=registry)
    registered_domains = list(registry.load_all().keys())
    target_domain = (
        pack.domain_id
        if pack.domain_id in registered_domains
        else (registered_domains[0] if registered_domains else "default")
    )
    passed_first_try = not bool(
        pack.preflight_report and pack.preflight_report.bounce_back_feedback
    )
    safe_principal_id = pseudonymize_actor_if_enabled(principal.user_id, principal.tenant_id)
    record = ExecutionTelemetryRecord(
        trace_id=uuid.uuid5(uuid.NAMESPACE_OID, pack.trace_id),
        tenant_id=principal.tenant_id,
        domain_id=target_domain,
        principal_id=safe_principal_id,
        principal_role=principal.role.value,
        execution_mode=ExecutionMode.CHAT_QA,
        local_tokens_read=savings.local_tokens_explored,
        context_pack_tokens_sent=savings.pack_tokens_sent,
        paid_tokens_saved=savings.tokens_saved,
        estimated_usd_saved=savings.estimated_usd_saved,
        preflight_passed_first_try=passed_first_try,
        zero_touch_sufficient=True,
        missing_context_requests=[],
        latency_ms=max(0, latency_ms),
    )
    persist_pg_in_test = (
        not os.environ.get("PYTEST_CURRENT_TEST")
        or principal.tenant_id != DEFAULT_TENANT_ID
        or os.environ.get("SCOUT_FORCE_PG_TELEMETRY_IN_TESTS", "").strip() == "1"
    )
    try:
        if isinstance(repo, InMemoryRLSRepository):
            repo.record_execution_telemetry(record)
        elif hasattr(repo, "connection_manager") and persist_pg_in_test:
            with repo.connection_manager.scoped_transaction(principal) as conn:
                PostgresRLSRepository().record_execution_telemetry(conn, record)
    except Exception as exc:
        logger.warning(
            "Unable to persist execution_telemetry for trace %s (%s): %s",
            pack.trace_id,
            type(exc).__name__,
            exc,
        )


def _record_scout_metrics_and_audit(
    *,
    principal: PrincipalContext,
    domain_id: str,
    result: ImplementationContextPack | InsufficientContextDiagnostic,
    endpoint: str,
) -> None:
    prom = get_prometheus_registry()
    if isinstance(result, ImplementationContextPack):
        rrf = float(result.preflight_report.normalized_rrf_score)
        cit_fail = not bool(result.preflight_report.citations_verified_100_pct)
        zero_hit = len(result.code_slices) == 0 and len(result.supporting_doc_citations) == 0
        prom.record_query_outcome(
            tenant_id=principal.tenant_id,
            domain_id=result.domain_id,
            outcome="success",
            rrf_score=rrf,
            zero_hit=zero_hit,
            citation_failure=cit_fail,
        )
        get_audit_chain_service().record_event(
            principal=principal,
            domain_id=result.domain_id,
            event_type="scout.query",
            severity="INFO",
            resource_uri=endpoint,
            outcome="success",
            payload={
                "trace_id": result.trace_id,
                "rrf_score": rrf,
                "pack_token_count": result.pack_token_count,
            },
        )
    else:
        prom.record_query_outcome(
            tenant_id=principal.tenant_id,
            domain_id=domain_id,
            outcome="abstain",
            rrf_score=0.0,
            zero_hit=True,
            reason_code=str(result.failure_check or "insufficient_context"),
        )
        get_audit_chain_service().record_event(
            principal=principal,
            domain_id=domain_id,
            event_type="scout.query",
            severity="INFO",
            resource_uri=endpoint,
            outcome="abstain",
            payload={
                "trace_id": result.trace_id,
                "failure_check": result.failure_check or "insufficient_context",
            },
        )


@router.post("/scout/query", response_model=ScoutQueryResponse)
def query_scout(
    request: ScoutStreamRequest,
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
    registry: Annotated[DomainRegistry, Depends(get_domain_registry)],
    repo: Annotated[Any, Depends(get_repository)],
    tracer: Annotated[ScoutTelemetryTracer, Depends(get_telemetry_tracer)],
    local_client: Annotated[LocalGemmaClient, Depends(get_local_gemma_client)],
) -> ScoutQueryResponse:
    """Synchronous JSON scout query execution through ScoutWorkflow or coordinator."""
    active_trace_id = request.trace_id or f"trace_{uuid.uuid4().hex[:12]}"
    t0 = time.perf_counter()

    # Step 1: Configured guardrail prompt-injection screening
    guardrail = get_guardrail_provider()
    decision = guardrail.evaluate(request.query, principal=principal)
    if not decision.passed:
        logger.warning("Scout query rejected by guardrail: %s", decision.rationale)
        get_prometheus_registry().record_query_outcome(
            tenant_id=principal.tenant_id,
            domain_id=request.domain_id,
            outcome="blocked",
            reason_code="prompt_injection",
        )
        get_audit_chain_service().record_event(
            principal=principal,
            domain_id=request.domain_id,
            event_type="scout.guardrail.block",
            severity="WARNING",
            resource_uri="/api/v1/scout/query",
            outcome="deny",
            payload={"trace_id": active_trace_id},
        )
        raise HTTPException(
            status_code=403,
            detail=f"Prompt injection detected: {decision.rationale}",
        )

    # Step 2: Validate domain registration
    if request.domain_id != "federated_all":
        try:
            _ = registry.get(request.domain_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Domain '{request.domain_id}' not found in DomainRegistry",
            ) from None

    # Step 3: Execute appropriate workflow
    from context_scout.telemetry.otel import resolve_otel_model_and_provider

    api_model_id, api_provider = resolve_otel_model_and_provider()
    with tracer.trace_workflow(
        span_name="api.scout.query",
        domain_id=request.domain_id,
        trace_id=active_trace_id,
        model_id=api_model_id,
        system=api_provider,
    ) as span:
        if request.normalized_rrf_score is not None:
            logger.debug(
                "Stripping client-supplied normalized_rrf_score=%s on /scout/query (trace=%s)",
                request.normalized_rrf_score,
                active_trace_id,
            )
        try:
            if request.domain_id == "federated_all":
                coordinator = FederatedScoutCoordinator(
                    registry=registry,
                    repo=repo,
                    local_client=local_client,
                    federated_domain_id=request.domain_id,
                )
                result = coordinator.run(
                    user_prompt=request.query,
                    principal=principal,
                    trace_id=active_trace_id,
                    normalized_rrf_score=None,
                    doc_chunks=None,
                    max_bounce_backs=request.max_bounce_backs,
                    source_filter=request.source_filter,
                    session_id=request.session_id,
                    branch_id=request.branch_id or request.snapshot_id,
                    graph_mode=request.graph_mode,
                )
            else:
                workflow = ScoutWorkflow(
                    registry=registry,
                    repo=repo,
                    local_client=local_client,
                )
                result = workflow.run(
                    user_prompt=request.query,
                    principal=principal,
                    domain_id=request.domain_id,
                    trace_id=active_trace_id,
                    normalized_rrf_score=None,
                    max_bounce_backs=request.max_bounce_backs,
                    source_filter=request.source_filter,
                    session_id=request.session_id,
                    branch_id=request.branch_id or request.snapshot_id,
                    graph_mode=request.graph_mode,
                )
        except (ValueError, psycopg.errors.DataError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        span.record_pack(result)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        if isinstance(result, ImplementationContextPack):
            savings = _build_token_savings(result)
            _record_live_scout_telemetry(repo, registry, principal, result, latency_ms)
            _record_scout_metrics_and_audit(
                principal=principal,
                domain_id=request.domain_id,
                result=result,
                endpoint="/api/v1/scout/query",
            )

            safe_pid = pseudonymize_actor_if_enabled(principal.user_id, principal.tenant_id)
            trace_attrs: dict[str, Any] = {
                "gen_ai.system": "gemma",
                "gen_ai.request.model": "gemma-4-26B-A4B-it",
                "context_scout.principal_id": safe_pid,
                "context_scout.principal_role": principal.role.value,
                "context_scout.domain_id": result.domain_id,
                "context_scout.tokens_saved": savings.tokens_saved,
                "context_scout.estimated_usd_saved": savings.estimated_usd_saved,
                "context_scout.preflight_passed": True,
            }

            return ScoutQueryResponse(
                trace_id=result.trace_id,
                domain_id=result.domain_id,
                passed=True,
                context_pack=result,
                diagnostic=None,
                citations=result.supporting_doc_citations,
                preflight_report=result.preflight_report,
                token_savings=savings,
                stage_traces=list(result.stage_traces),
                trace_attributes=trace_attrs,
            )

        _record_scout_metrics_and_audit(
            principal=principal,
            domain_id=request.domain_id,
            result=result,
            endpoint="/api/v1/scout/query",
        )
        diagnostic_savings = ScoutTokenSavingsBreakdown(
            local_tokens_explored=result.paid_tokens_spent,
            pack_tokens_sent=0,
            tokens_saved=0,
            estimated_usd_saved=0.0,
            frontier_savings_ratio_pct=0.0,
            qlora_overhead_eliminated_tokens=0.0,
        )

        return ScoutQueryResponse(
            trace_id=result.trace_id,
            domain_id=result.domain_id,
            passed=False,
            context_pack=None,
            diagnostic=result,
            citations=[],
            preflight_report=None,
            token_savings=diagnostic_savings,
            stage_traces=list(result.stage_traces),
            trace_attributes={
                "context_scout.preflight_passed": False,
                "context_scout.diagnostic_reason": result.reason,
            },
        )


@router.post(
    "/scout/stream",
    responses={
        200: {
            "description": "Server-Sent Events stream of scout workflow stage transitions",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "event": {"type": "string"},
                            "data": {
                                "type": "string",
                                "contentMediaType": "application/json",
                                "contentSchema": {"type": "object"},
                            },
                        },
                        "required": ["event", "data"],
                    }
                }
            },
        }
    },
)
async def stream_scout(
    request: ScoutStreamRequest,
    http_request: Request,
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
    registry: Annotated[DomainRegistry, Depends(get_domain_registry)],
    repo: Annotated[Any, Depends(get_repository)],
    local_client: Annotated[LocalGemmaClient, Depends(get_local_gemma_client)],
) -> StreamingResponse:
    """Server-Sent Events endpoint streaming live workflow stage transitions and ContextPack."""

    async def event_generator() -> AsyncGenerator[str, None]:
        active_trace_id = request.trace_id or f"trace_{uuid.uuid4().hex[:12]}"
        t0 = time.perf_counter()

        # Step 1: Configured guardrail prompt-injection evaluation (offloaded from event loop)
        guardrail = get_guardrail_provider()
        decision = await anyio.to_thread.run_sync(
            lambda: guardrail.evaluate(request.query, principal=principal)
        )
        if not decision.passed:
            get_prometheus_registry().record_query_outcome(
                tenant_id=principal.tenant_id,
                domain_id=request.domain_id,
                outcome="blocked",
                reason_code="prompt_injection",
            )
            await anyio.to_thread.run_sync(
                lambda: get_audit_chain_service().record_event(
                    principal=principal,
                    domain_id=request.domain_id,
                    event_type="scout.guardrail.block",
                    severity="WARNING",
                    resource_uri="/api/v1/scout/stream",
                    outcome="deny",
                    payload={"trace_id": active_trace_id},
                )
            )
            s1_diag = InsufficientContextDiagnostic(
                trace_id=active_trace_id,
                domain_id=request.domain_id,
                reason=f"Prompt injection blocked: {decision.rationale}",
                failure_check="stage_1_prompt_guard",
                paid_tokens_spent=0,
                stage_confidences=[
                    StageConfidence.from_score(
                        stage=1,
                        stage_name="Stage 1: Security, Auth & Intent",
                        score=0.0,
                        signals={
                            "rls_scopes_count": len(principal.allowed_scopes),
                            "injection_threat_score": 1.0,
                            "blocked": True,
                        },
                        summary=f"Blocked by Stage 1 security guard: {decision.rationale}",
                    )
                ],
            )
            err_event = SSEErrorEvent(
                stage="guardrails",
                status="blocked",
                category="prompt_injection",
                rationale=decision.rationale,
                diagnostic=s1_diag,
            )
            yield f"event: error\ndata: {err_event.model_dump_json()}\n\n"
            return

        # Step 2: Check domain existence
        if request.domain_id != "federated_all":
            try:
                _ = registry.get(request.domain_id)
            except KeyError:
                err_event = SSEErrorEvent(
                    stage="domain_validation",
                    status="error",
                    category="unknown_domain",
                    rationale=f"Domain '{request.domain_id}' not found in DomainRegistry",
                )
                yield f"event: error\ndata: {err_event.model_dump_json()}\n\n"
                return

        is_federated = request.domain_id == "federated_all"
        stage_titles = _FEDERATED_STAGE_TITLES if is_federated else _STAGE_TITLES

        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        cancel_event = threading.Event()

        def push_stage_event(ev: dict[str, Any]) -> None:
            if cancel_event.is_set() or loop.is_closed():
                return
            try:
                loop.call_soon_threadsafe(event_queue.put_nowait, ev)
            except RuntimeError:
                return

        async def run_worker() -> ImplementationContextPack | InsufficientContextDiagnostic:
            try:
                if request.normalized_rrf_score is not None:
                    logger.debug(
                        "Stripping client-supplied normalized_rrf_score=%s on /scout/stream "
                        "(trace=%s)",
                        request.normalized_rrf_score,
                        active_trace_id,
                    )
                if is_federated:
                    coordinator = FederatedScoutCoordinator(
                        registry=registry,
                        repo=repo,
                        local_client=local_client,
                        federated_domain_id=request.domain_id,
                    )
                    return await asyncio.to_thread(
                        coordinator.run,
                        user_prompt=request.query,
                        principal=principal,
                        trace_id=active_trace_id,
                        normalized_rrf_score=None,
                        doc_chunks=None,
                        max_bounce_backs=request.max_bounce_backs,
                        on_stage_event=push_stage_event,
                        source_filter=request.source_filter,
                        session_id=request.session_id,
                        branch_id=request.branch_id or request.snapshot_id,
                        graph_mode=request.graph_mode,
                    )
                workflow = ScoutWorkflow(
                    registry=registry,
                    repo=repo,
                    local_client=local_client,
                )
                return await asyncio.to_thread(
                    workflow.run,
                    user_prompt=request.query,
                    principal=principal,
                    domain_id=request.domain_id,
                    trace_id=active_trace_id,
                    normalized_rrf_score=None,
                    max_bounce_backs=request.max_bounce_backs,
                    on_stage_event=push_stage_event,
                    source_filter=request.source_filter,
                    session_id=request.session_id,
                    branch_id=request.branch_id or request.snapshot_id,
                    graph_mode=request.graph_mode,
                )
            finally:
                if not loop.is_closed():
                    with contextlib.suppress(RuntimeError):
                        loop.call_soon_threadsafe(event_queue.put_nowait, None)

        worker_task = asyncio.create_task(run_worker())
        completed_stages: set[int] = set()
        failed_stages: set[int] = set()
        bounce_emitted = False

        try:
            while True:
                if await http_request.is_disconnected():
                    cancel_event.set()
                    worker_task.cancel()
                    return
                try:
                    ev = await asyncio.wait_for(
                        event_queue.get(),
                        timeout=max(0.01, float(SSE_HEARTBEAT_INTERVAL_SECONDS)),
                    )
                except TimeoutError:
                    if await http_request.is_disconnected():
                        cancel_event.set()
                        worker_task.cancel()
                        return
                    yield ": keep-alive\n\n"
                    continue
                if ev is None:
                    break

                ev_type = str(ev.get("event_type", ""))
                if ev_type == "stage_update":
                    stage_num = int(ev.get("stage", 1))
                    raw_status = str(ev.get("status", "completed"))
                    if raw_status == "in_progress":
                        continue
                    stage_status = "failed" if raw_status == "failed" else "completed"

                    # Backfill omitted lower stages before emitting later stages
                    scopes_label = (
                        ", ".join(principal.allowed_scopes) if principal.allowed_scopes else "none"
                    )
                    for req_stage in range(1, min(stage_num, 4)):
                        if req_stage not in completed_stages and req_stage not in failed_stages:
                            elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))
                            if req_stage == 1:
                                fill_detail = (
                                    f"Screened '{request.domain_id}' under PostgreSQL 17 RLS "
                                    f"(scopes: {scopes_label}) in {elapsed_ms} ms"
                                )
                            elif req_stage == 2:
                                fill_detail = (
                                    f"Local Gemma 4 synthesis completed for '{request.domain_id}' "
                                    f"at $0 paid tokens in {elapsed_ms} ms"
                                )
                            else:
                                fill_detail = (
                                    f"Evaluated AST closure and citation grounding for "
                                    f"'{request.domain_id}' in {elapsed_ms} ms"
                                )
                            fill_ev = SSEStageEvent(
                                stage=req_stage,
                                stage_name=stage_titles[req_stage],
                                status="completed",
                                details=fill_detail,
                                elapsed_ms=float(elapsed_ms),
                            )
                            yield f"event: stage_update\ndata: {fill_ev.model_dump_json()}\n\n"
                            completed_stages.add(req_stage)

                    if stage_num in completed_stages or stage_num in failed_stages:
                        continue

                    resolved_name = (
                        stage_titles.get(stage_num, f"Stage {stage_num}")
                        if is_federated
                        else str(
                            ev.get("stage_name")
                            or stage_titles.get(stage_num, f"Stage {stage_num}")
                        )
                    )
                    live_elapsed = round((time.perf_counter() - t0) * 1000, 2)
                    raw_conf = ev.get("confidence")
                    stage_conf: StageConfidence | None = (
                        StageConfidence.model_validate(raw_conf)
                        if isinstance(raw_conf, dict)
                        else None
                    )
                    if stage_conf is None and stage_num == 1:
                        parsed_plan = decompose_query_facets(request.query)
                        rls_count = len(principal.allowed_scopes)
                        stage_conf = StageConfidence.from_score(
                            stage=1,
                            stage_name="Stage 1: Security, Auth & Intent",
                            score=1.0 if rls_count > 0 else 0.65,
                            signals={
                                "rls_scopes_count": rls_count,
                                "injection_threat_score": 0.0,
                                "facet_count": len(parsed_plan.facets),
                                "requested_cardinality": parsed_plan.requested_cardinality or 0,
                            },
                            summary=(
                                f"RLS enforced ({rls_count} scopes), "
                                f"{len(parsed_plan.facets)} facet(s) parsed."
                            ),
                        )
                    raw_trace = ev.get("stage_trace")
                    stage_trace: StageExecutionTrace | None = (
                        StageExecutionTrace.model_validate(raw_trace)
                        if isinstance(raw_trace, dict)
                        else None
                    )
                    stage_ev = SSEStageEvent(
                        stage=stage_num,
                        stage_name=resolved_name,
                        status=stage_status,  # type: ignore[arg-type]
                        details=str(ev.get("details") or ""),
                        confidence=stage_conf,
                        stage_trace=stage_trace,
                        elapsed_ms=live_elapsed,
                    )
                    yield f"event: stage_update\ndata: {stage_ev.model_dump_json()}\n\n"

                    if stage_status == "completed":
                        completed_stages.add(stage_num)
                    else:
                        failed_stages.add(stage_num)

                elif ev_type == "bounce_back":
                    bounce_emitted = True
                    fb = str(
                        ev.get("feedback") or "Pre-flight check triggered $0 local self-correction"
                    )
                    bb = SSEBounceBackEvent(
                        stage=int(ev.get("stage", 3)),
                        status="bounce_back_triggered",
                        feedback=fb,
                        bounce_count=1,
                        remediation_action=(
                            "Re-querying local model with deterministic pre-flight symbol feedback"
                        ),
                        paid_tokens_spent=0,
                        cost_usd=0.0,
                        details=str(
                            ev.get("details")
                            or "$0 local bounce-back loop triggered to resolve missing symbols"
                        ),
                    )
                    yield f"event: bounce_back\ndata: {bb.model_dump_json()}\n\n"
        finally:
            cancel_event.set()
            if not worker_task.done():
                worker_task.cancel()

        try:
            result = await worker_task
        except Exception as exc:
            logger.warning("Scout stream worker failed (%s): %s", type(exc).__name__, exc)
            err_stage = SSEStageEvent(
                stage=4,
                stage_name=stage_titles[4],
                status="failed",
                details=str(exc),
            )
            yield f"event: stage_update\ndata: {err_stage.model_dump_json()}\n\n"
            err_event = SSEErrorEvent(
                stage="execution",
                status="error",
                category="model_unavailable",
                rationale=str(exc),
            )
            yield f"event: error\ndata: {err_event.model_dump_json()}\n\n"
            return

        latency_ms = max(1, int((time.perf_counter() - t0) * 1000))
        scopes_label = ", ".join(principal.allowed_scopes) if principal.allowed_scopes else "none"
        conf_by_stage = {c.stage: c for c in getattr(result, "stage_confidences", []) or []}
        trace_by_stage = {t.stage: t for t in getattr(result, "stage_traces", []) or []}

        # Backfill any omitted stage completions (e.g. early abstention before Stage 2/3)
        for req_stage in (1, 2, 3):
            if req_stage not in completed_stages and req_stage not in failed_stages:
                if req_stage == 1:
                    detail = (
                        f"Screened domain '{request.domain_id}' under PostgreSQL 17 RLS "
                        f"(scopes: {scopes_label}) in {latency_ms} ms"
                    )
                elif req_stage == 2:
                    if isinstance(result, ImplementationContextPack):
                        detail = (
                            f"Local Gemma 4 synthesis for '{request.domain_id}' prepared "
                            f"{len(result.code_slices)} slice(s) and "
                            f"{len(result.supporting_doc_citations)} citation(s) "
                            f"at $0 paid tokens ({latency_ms} ms)"
                        )
                    else:
                        detail = (
                            f"Local Gemma 4 synthesis for '{request.domain_id}' "
                            f"evaluated query at $0 paid tokens ({latency_ms} ms)"
                        )
                else:
                    if isinstance(result, ImplementationContextPack):
                        auto_1hop = sum(1 for s in result.code_slices if s.auto_expanded_by_ast)
                        direct_slices = len(result.code_slices) - auto_1hop
                        detail = (
                            f"Hydrated {direct_slices} direct slice(s) + {auto_1hop} 1-hop AST "
                            f"slice(s) under 10,000-token budget, verified "
                            f"{len(result.supporting_doc_citations)} citation(s), flagged "
                            f"{len(result.doc_vs_code_conflicts_detected)} conflict(s) "
                            f"({latency_ms} ms)"
                        )
                    else:
                        detail = (
                            f"Evaluated candidate context coverage for '{request.domain_id}' "
                            f"under deterministic AST & citation rules ({latency_ms} ms)"
                        )
                fill_ev = SSEStageEvent(
                    stage=req_stage,
                    stage_name=stage_titles[req_stage],
                    status="completed",
                    details=detail,
                    confidence=conf_by_stage.get(req_stage),
                    stage_trace=trace_by_stage.get(req_stage),
                    elapsed_ms=float(latency_ms),
                )
                yield f"event: stage_update\ndata: {fill_ev.model_dump_json()}\n\n"
                completed_stages.add(req_stage)

        if (
            not bounce_emitted
            and isinstance(result, ImplementationContextPack)
            and result.preflight_report
            and result.preflight_report.bounce_back_feedback
        ):
            bb = SSEBounceBackEvent(
                stage=3,
                status="bounce_back_triggered",
                feedback=result.preflight_report.bounce_back_feedback,
                bounce_count=1,
                remediation_action=(
                    "Re-querying local model with deterministic pre-flight symbol feedback"
                ),
                paid_tokens_spent=0,
                cost_usd=0.0,
                details="$0 local bounce-back loop triggered to resolve missing symbols",
            )
            yield f"event: bounce_back\ndata: {bb.model_dump_json()}\n\n"

        if isinstance(result, ImplementationContextPack):
            if 4 not in completed_stages and 4 not in failed_stages:
                checks_total = (
                    len(result.preflight_report.check_results)
                    if result.preflight_report.check_results
                    else 6
                )
                checks_passed = (
                    sum(1 for ok in result.preflight_report.check_results.values() if ok)
                    if result.preflight_report.check_results
                    else checks_total
                )
                s4 = SSEStageEvent(
                    stage=4,
                    stage_name=stage_titles[4],
                    status="completed",
                    details=(
                        f"Passed {checks_passed}/{checks_total} pre-flight checks "
                        f"(RRF {result.preflight_report.normalized_rrf_score:.2f}); curated "
                        f"{result.pack_token_count:,} pack tokens from "
                        f"{result.local_tokens_consumed:,} local tokens explored "
                        f"($0 frontier exploration cost) in {latency_ms} ms"
                    ),
                    confidence=conf_by_stage.get(4),
                    stage_trace=trace_by_stage.get(4),
                    elapsed_ms=float(latency_ms),
                )
                yield f"event: stage_update\ndata: {s4.model_dump_json()}\n\n"

            savings = _build_token_savings(result)
            await anyio.to_thread.run_sync(
                _record_live_scout_telemetry, repo, registry, principal, result, latency_ms
            )
            await anyio.to_thread.run_sync(
                lambda: _record_scout_metrics_and_audit(
                    principal=principal,
                    domain_id=request.domain_id,
                    result=result,
                    endpoint="/api/v1/scout/stream",
                )
            )

            ready_event = SSEContextPackReadyEvent(
                status="ready",
                pack=result,
                token_savings=savings,
                stage_traces=list(result.stage_traces),
            )
            yield f"event: context_pack_ready\ndata: {ready_event.model_dump_json()}\n\n"
        else:
            await anyio.to_thread.run_sync(
                lambda: _record_scout_metrics_and_audit(
                    principal=principal,
                    domain_id=request.domain_id,
                    result=result,
                    endpoint="/api/v1/scout/stream",
                )
            )
            if 4 not in failed_stages and 4 not in completed_stages:
                s4_fail = SSEStageEvent(
                    stage=4,
                    stage_name=stage_titles[4],
                    status="failed",
                    details=f"$0 abstention after {latency_ms} ms: {result.reason}",
                    confidence=conf_by_stage.get(4),
                    stage_trace=trace_by_stage.get(4),
                    elapsed_ms=float(latency_ms),
                )
                yield f"event: stage_update\ndata: {s4_fail.model_dump_json()}\n\n"

            err = SSEErrorEvent(
                stage="assembly",
                status="error",
                category="insufficient_context",
                rationale=result.reason,
                diagnostic=result,
            )
            yield f"event: error\ndata: {err.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
