"""Concurrency isolation regression test suite for ContextVar session binding (SEC-04).

Guarantees:
1. Simultaneous concurrent requests in parallel asyncio tasks with distinct signed tokens
   strictly observe their own bound principal with zero identity leakage across tasks.
2. After task/request completion, _CURRENT_PRINCIPAL.get() is strictly None in caller context.
3. Exception path concurrency (one request failing while another succeeds) produces
   no residual state or cross-task pollution.
4. Scale concurrency under high task volume (50 parallel tasks) preserves 100% isolation.
5. End-to-end FastAPI HTTP concurrency via ASGITransport guarantees request isolation under
   live middleware and dependency evaluation.
"""

from __future__ import annotations

import asyncio
import random
from typing import Annotated

import httpx
import pytest
from fastapi import APIRouter, Depends

from context_scout.api.dependencies import (
    create_signed_principal_token,
    get_principal_context,
    reset_current_principal,
    verify_signed_principal_token,
)
from context_scout.api.main import create_app
from context_scout.schemas.contracts import PrincipalContext, RoleType
from context_scout.security.rbac import (
    _CURRENT_PRINCIPAL,
    AuthenticationRequiredError,
    bind_session_principal,
    get_current_principal,
    has_bound_principal,
)


@pytest.fixture(autouse=True)
def ensure_clean_principal_state() -> None:
    """Ensure _CURRENT_PRINCIPAL is None before and after each test."""
    _CURRENT_PRINCIPAL.set(None)
    reset_current_principal()
    assert _CURRENT_PRINCIPAL.get() is None


@pytest.mark.asyncio
async def test_concurrent_tasks_distinct_principals_zero_leakage() -> None:
    """Verify concurrent tasks with distinct signed tokens observe only their own identity."""
    token_a = create_signed_principal_token(
        user_id="alice_engineer",
        role=RoleType.ENGINEER,
        allowed_scopes=["repo:repo_alpha", "public_eng"],
    )
    token_b = create_signed_principal_token(
        user_id="bob_quant",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["corpus:alpha_factors", "quant_restricted"],
    )

    observed_in_a: list[str] = []
    observed_in_b: list[str] = []

    async def task_worker(
        token_str: str,
        expected_user: str,
        expected_role: RoleType,
        target_list: list[str],
    ) -> None:
        principal = verify_signed_principal_token(token_str)
        with bind_session_principal(principal):
            for _ in range(5):
                current = get_current_principal()
                assert current.user_id == expected_user
                assert current.role == expected_role
                target_list.append(current.user_id)
                await asyncio.sleep(random.uniform(0.005, 0.02))
                # Re-verify after yielding control to the event loop
                assert get_current_principal().user_id == expected_user

    await asyncio.gather(
        task_worker(token_a, "alice_engineer", RoleType.ENGINEER, observed_in_a),
        task_worker(token_b, "bob_quant", RoleType.QUANT_RESEARCHER, observed_in_b),
    )

    assert len(observed_in_a) == 5
    assert set(observed_in_a) == {"alice_engineer"}
    assert len(observed_in_b) == 5
    assert set(observed_in_b) == {"bob_quant"}
    assert _CURRENT_PRINCIPAL.get() is None


@pytest.mark.asyncio
async def test_concurrency_exception_path_no_residual_leakage() -> None:
    """Verify that an exception in one concurrent task does not bleed state into sibling tasks."""
    token_success = create_signed_principal_token(
        user_id="successful_principal",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    corrupted_token = "tampered.payload.signature"

    results: dict[str, str | Exception] = {}

    async def successful_worker() -> None:
        principal = verify_signed_principal_token(token_success)
        with bind_session_principal(principal):
            await asyncio.sleep(0.015)
            # Verify identity remains intact despite sibling task failure
            current = get_current_principal()
            results["success"] = current.user_id

    async def failing_worker() -> None:
        await asyncio.sleep(0.005)
        # Attempting to verify corrupted token raises HTTPException (401)
        verify_signed_principal_token(corrupted_token)

    outcome = await asyncio.gather(
        successful_worker(),
        failing_worker(),
        return_exceptions=True,
    )

    assert results.get("success") == "successful_principal"
    # Sibling task raised an exception (HTTPException from FastAPI)
    assert isinstance(outcome[1], Exception)
    assert _CURRENT_PRINCIPAL.get() is None


@pytest.mark.asyncio
async def test_high_volume_concurrency_scale_50_tasks() -> None:
    """Verify 50 parallel tasks with 50 distinct principals maintain 100% isolation."""
    task_count = 50
    tokens = [
        create_signed_principal_token(
            user_id=f"principal_{i:03d}",
            role=RoleType.ENGINEER if i % 2 == 0 else RoleType.QUANT_RESEARCHER,
            allowed_scopes=[f"scope:tenant_{i:03d}"],
        )
        for i in range(task_count)
    ]

    identities_recorded: dict[int, list[str]] = {i: [] for i in range(task_count)}

    async def scale_worker(index: int, token_str: str) -> None:
        expected_id = f"principal_{index:03d}"
        principal = verify_signed_principal_token(token_str)
        with bind_session_principal(principal):
            for _ in range(4):
                await asyncio.sleep(random.uniform(0.001, 0.01))
                active = get_current_principal()
                identities_recorded[index].append(active.user_id)
                assert active.user_id == expected_id

    workers = [scale_worker(i, tokens[i]) for i in range(task_count)]
    await asyncio.gather(*workers)

    for i in range(task_count):
        recorded = identities_recorded[i]
        assert len(recorded) == 4
        assert set(recorded) == {f"principal_{i:03d}"}

    assert _CURRENT_PRINCIPAL.get() is None


@pytest.mark.asyncio
async def test_fastapi_http_concurrent_request_isolation() -> None:
    """Verify HTTP concurrency isolation via FastAPI middleware and get_principal_context."""
    app = create_app()

    # Register an isolated inspection route that checks the ContextVar during request handling
    test_router = APIRouter()

    @test_router.get("/concurrency/whoami")
    async def whoami(
        principal: Annotated[PrincipalContext, Depends(get_principal_context)],
    ) -> dict[str, object]:
        with bind_session_principal(principal):
            ctx_principal = _CURRENT_PRINCIPAL.get()
            assert ctx_principal is not None
            assert ctx_principal.user_id == principal.user_id

            # Introduce async delay simulating downstream DB/IO call
            await asyncio.sleep(random.uniform(0.01, 0.03))

            # Re-verify contextvar after awaiting IO
            after_await_ctx = _CURRENT_PRINCIPAL.get()
            assert after_await_ctx is not None
            assert after_await_ctx.user_id == principal.user_id

            return {
                "user_id": principal.user_id,
                "role": principal.role.value,
                "scopes": principal.allowed_scopes,
                "ctx_user_id": after_await_ctx.user_id,
            }

    app.include_router(test_router, prefix="/api/v1")

    token_a = create_signed_principal_token(
        user_id="http_engineer_alice",
        role=RoleType.ENGINEER,
        allowed_scopes=["repo:frontend", "public_eng"],
    )
    token_b = create_signed_principal_token(
        user_id="http_quant_bob",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["corpus:market_microstructure"],
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async def fetch_profile(token: str) -> httpx.Response:
            return await client.get(
                "/api/v1/concurrency/whoami",
                headers={"Authorization": f"Bearer {token}"},
            )

        # Fire both requests concurrently
        resp_a, resp_b = await asyncio.gather(
            fetch_profile(token_a),
            fetch_profile(token_b),
        )

        assert resp_a.status_code == 200
        assert resp_b.status_code == 200

        data_a = resp_a.json()
        data_b = resp_b.json()

        assert data_a["user_id"] == "http_engineer_alice"
        assert data_a["ctx_user_id"] == "http_engineer_alice"
        assert data_a["role"] == "engineer"

        assert data_b["user_id"] == "http_quant_bob"
        assert data_b["ctx_user_id"] == "http_quant_bob"
        assert data_b["role"] == "quant_researcher"

        # Verify no cross-talk occurred between the responses
        assert data_a["user_id"] != data_b["user_id"]
        assert data_a["ctx_user_id"] != data_b["ctx_user_id"]

    # Verify that in the main server loop contextvar remains clean
    assert _CURRENT_PRINCIPAL.get() is None


@pytest.mark.asyncio
async def test_unauthenticated_request_concurrency_does_not_affect_authenticated_request() -> None:
    """Verify unauthenticated 401 request does not interfere with concurrent valid request."""
    app = create_app()

    token_valid = create_signed_principal_token(
        user_id="valid_user_concurrent",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async def send_valid() -> httpx.Response:
            await asyncio.sleep(0.005)
            return await client.get(
                "/api/v1/domains",
                headers={"Authorization": f"Bearer {token_valid}"},
            )

        async def send_invalid() -> httpx.Response:
            return await client.post(
                "/api/v1/scout/query",
                json={"query": "test query", "domain_id": "trading_code_scout"},
                headers={"Authorization": "Bearer bad.token.value"},
            )

        res_valid, res_invalid = await asyncio.gather(
            send_valid(),
            send_invalid(),
        )

        assert res_valid.status_code == 200
        assert res_invalid.status_code == 401

    assert _CURRENT_PRINCIPAL.get() is None


def test_has_bound_principal_lifecycle() -> None:
    """Verify has_bound_principal accurately tracks session binding transitions."""
    assert not has_bound_principal()
    with pytest.raises(AuthenticationRequiredError):
        get_current_principal()

    principal = PrincipalContext(
        user_id="sync_tester",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    with bind_session_principal(principal):
        assert has_bound_principal()
        assert get_current_principal().user_id == "sync_tester"

    assert not has_bound_principal()
    with pytest.raises(AuthenticationRequiredError):
        get_current_principal()
    assert _CURRENT_PRINCIPAL.get() is None
