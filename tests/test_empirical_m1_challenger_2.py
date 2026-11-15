"""Empirical Challenger 2 Test Suite: Concurrency, ContextVar, and Edge Cases (Milestone 1).

Adversarial stress-testing of Commit 1724d5ac5e9ee467a1f7926a0f1a0205463166de (Critical Item B).

Vectors tested:
1. Concurrency Stress (Asyncio): 50 interleaved parallel tasks under distinct identities with
   delays, asserting 0 cross-task principal bleeding in _CURRENT_PRINCIPAL.
2. Concurrency Stress (OS Threads): Native OS threads via ThreadPoolExecutor asserting ContextVar
   isolation across threads.
3. Concurrency Under Failure & Exceptions: Interleaved valid requests, 401 auth failures, and 500
   server crashes across concurrent tasks, proving failed requests never poison valid requests.
4. Post-Request Reset Guarantees: _CURRENT_PRINCIPAL strictly resets to None after every request
   (including unhandled exceptions, 401s, 403s, and 500s).
5. Whitespace Edge Cases: Trailing/leading whitespace, tabs, and newlines in secrets and headers.
6. Bearer Scheme Case Sensitivity: "Bearer", "bearer", "BEARER", "bEaReR", invalid schemes, etc.
7. Huge Token Payloads & DoS Resistance: Megabyte payloads, excessive delimiter dots, oversized
   fields, and timing verification.
8. Documented White-Box Boundary Conditions: Unhandled exceptions on non-integer exp/max_tokens,
   non-ascii bytes, and whitespace-padded explicit secret keys.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import hmac
import json
import random
import time
from typing import Annotated

import httpx
import pytest
from fastapi import APIRouter, Depends, HTTPException, Response

from context_scout.api.dependencies import (
    _resolve_auth_secret,
    create_signed_principal_token,
    get_principal_context,
    reset_current_principal,
    verify_signed_principal_token,
)
from context_scout.api.main import create_app
from context_scout.schemas.contracts import PrincipalContext, RoleType
from context_scout.security.rbac import (
    _CURRENT_PRINCIPAL,
    bind_session_principal,
    get_current_principal,
    has_bound_principal,
)

STRONG_CHALLENGER_SECRET = "production-cryptographic-secret-key-32-chars-long-ok!"


@pytest.fixture(autouse=True)
def configure_challenger_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configures clean test environment with valid secret and pristine ContextVar."""
    monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", STRONG_CHALLENGER_SECRET)
    monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)
    monkeypatch.setenv("SCOUT_REQUIRE_SIGNED_AUTH", "true")
    _CURRENT_PRINCIPAL.set(None)
    reset_current_principal()
    assert _CURRENT_PRINCIPAL.get() is None


# ==============================================================================
# Vector 1: Asyncio Concurrency Stress (50 Interleaved Tasks)
# ==============================================================================
class TestAsyncioConcurrencyStress:
    """Empirically stress-tests ContextVar isolation across 50 concurrent asyncio tasks."""

    @pytest.mark.asyncio
    async def test_50_concurrent_asyncio_tasks_distinct_principals(self) -> None:
        """50 parallel asyncio tasks with 50 distinct identities interleaved with delays."""
        task_count = 50
        roles = [
            RoleType.ENGINEER,
            RoleType.QUANT_RESEARCHER,
            RoleType.PRODUCT_MANAGER,
            RoleType.COMPLIANCE_OFFICER,
        ]

        tokens: list[tuple[str, str, RoleType, list[str]]] = []
        for i in range(task_count):
            uid = f"challenger_async_user_{i:03d}"
            role = roles[i % len(roles)]
            scopes = [f"scope:tenant_{i:03d}", f"role:{role.value}"]
            token = create_signed_principal_token(
                user_id=uid,
                role=role,
                allowed_scopes=scopes,
                secret_key=STRONG_CHALLENGER_SECRET,
            )
            tokens.append((token, uid, role, scopes))

        observed_identities: dict[int, list[str]] = {i: [] for i in range(task_count)}

        async def worker(
            index: int, token_str: str, expected_uid: str, expected_role: RoleType
        ) -> None:
            principal = verify_signed_principal_token(token_str)
            with bind_session_principal(principal):
                for _ in range(5):
                    # Check identity before yielding to event loop
                    active = get_current_principal()
                    assert active.user_id == expected_uid
                    assert active.role == expected_role
                    observed_identities[index].append(active.user_id)

                    # Artificial async delay yielding control back to event loop
                    await asyncio.sleep(random.uniform(0.005, 0.025))

                    # Re-verify identity after awakening
                    active_post = get_current_principal()
                    assert active_post.user_id == expected_uid
                    assert active_post.role == expected_role

        workers = [worker(i, tokens[i][0], tokens[i][1], tokens[i][2]) for i in range(task_count)]
        await asyncio.gather(*workers)

        # Assert every single task observed ONLY its own identity, exactly 5 times
        for i in range(task_count):
            records = observed_identities[i]
            expected_uid = tokens[i][1]
            assert len(records) == 5
            assert set(records) == {expected_uid}

        # Assert post-concurrency ContextVar is clean
        assert _CURRENT_PRINCIPAL.get() is None


# ==============================================================================
# Vector 2: Multi-Thread Concurrency Stress (Native OS Threads)
# ==============================================================================
class TestMultiThreadConcurrencyStress:
    """Empirically stress-tests ContextVar isolation across native OS threads."""

    def test_30_concurrent_threads_distinct_principals(self) -> None:
        """30 concurrent OS threads via ThreadPoolExecutor maintain ContextVar isolation."""
        thread_count = 30
        results: dict[int, list[str]] = {i: [] for i in range(thread_count)}

        def thread_worker(index: int) -> list[str]:
            uid = f"challenger_thread_user_{index:03d}"
            role = RoleType.ENGINEER if index % 2 == 0 else RoleType.QUANT_RESEARCHER
            scopes = [f"scope:thread_{index:03d}"]
            token = create_signed_principal_token(
                user_id=uid,
                role=role,
                allowed_scopes=scopes,
                secret_key=STRONG_CHALLENGER_SECRET,
            )
            principal = verify_signed_principal_token(token)

            with bind_session_principal(principal):
                local_obs: list[str] = []
                for _ in range(4):
                    curr = get_current_principal()
                    assert curr.user_id == uid
                    local_obs.append(curr.user_id)
                    time.sleep(random.uniform(0.005, 0.02))
                    curr_after = get_current_principal()
                    assert curr_after.user_id == uid
                return local_obs

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = {executor.submit(thread_worker, i): i for i in range(thread_count)}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        for i in range(thread_count):
            assert len(results[i]) == 4
            assert set(results[i]) == {f"challenger_thread_user_{i:03d}"}

        # Main thread context remains clean
        assert _CURRENT_PRINCIPAL.get() is None


# ==============================================================================
# Vector 3: FastAPI HTTP Concurrency Under Failures and Exceptions
# ==============================================================================
class TestFastApiHttpConcurrencyUnderFailureAndExceptions:
    """Stress-tests HTTP concurrency when valid, failing, and crashing requests interleave."""

    @pytest.mark.asyncio
    async def test_interleaved_valid_and_failing_and_crashing_http_requests(self) -> None:
        """60 concurrent requests (20 Alice, 20 Bob, 10 Auth Failures, 10 Server Crashes)."""
        app = create_app()
        test_router = APIRouter()

        @test_router.get("/probe/identity")
        async def probe_identity(
            principal: Annotated[PrincipalContext, Depends(get_principal_context)],
        ) -> dict[str, str]:
            with bind_session_principal(principal):
                active = get_current_principal()
                assert active.user_id == principal.user_id

                # Simulated I/O delay yielding execution back to event loop
                await asyncio.sleep(random.uniform(0.01, 0.03))

                active_post = get_current_principal()
                assert active_post.user_id == principal.user_id

                return {
                    "user_id": principal.user_id,
                    "ctx_user_id": active_post.user_id,
                    "role": principal.role.value,
                }

        @test_router.get("/probe/crash")
        async def probe_crash(
            principal: Annotated[PrincipalContext, Depends(get_principal_context)],
        ) -> dict[str, str]:
            with bind_session_principal(principal):
                await asyncio.sleep(0.01)
                raise RuntimeError("Simulated internal backend crash during active session")

        app.include_router(test_router, prefix="/api/v1")

        token_alice = create_signed_principal_token(
            "alice_http",
            RoleType.ENGINEER,
            ["public_eng"],
            secret_key=STRONG_CHALLENGER_SECRET,
        )
        token_bob = create_signed_principal_token(
            "bob_http",
            RoleType.QUANT_RESEARCHER,
            ["quant_restricted"],
            secret_key=STRONG_CHALLENGER_SECRET,
        )
        token_crashed_user = create_signed_principal_token(
            "crash_user",
            RoleType.PRODUCT_MANAGER,
            ["public_eng"],
            secret_key=STRONG_CHALLENGER_SECRET,
        )
        invalid_token = "invalid.tampered.token"

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

            async def req_alice() -> tuple[str, int, str]:
                res = await client.get(
                    "/api/v1/probe/identity",
                    headers={"Authorization": f"Bearer {token_alice}"},
                )
                uid = res.json().get("ctx_user_id", "") if res.status_code == 200 else ""
                return ("alice", res.status_code, uid)

            async def req_bob() -> tuple[str, int, str]:
                res = await client.get(
                    "/api/v1/probe/identity",
                    headers={"Authorization": f"Bearer {token_bob}"},
                )
                uid = res.json().get("ctx_user_id", "") if res.status_code == 200 else ""
                return ("bob", res.status_code, uid)

            async def req_invalid() -> tuple[str, int, str]:
                res = await client.get(
                    "/api/v1/probe/identity",
                    headers={"Authorization": f"Bearer {invalid_token}"},
                )
                return ("invalid", res.status_code, "")

            async def req_crash() -> tuple[str, int, str]:
                try:
                    res = await client.get(
                        "/api/v1/probe/crash",
                        headers={"Authorization": f"Bearer {token_crashed_user}"},
                    )
                    return ("crash", res.status_code, "")
                except Exception:
                    return ("crash", 500, "")

            tasks = []
            for _ in range(20):
                tasks.append(req_alice())
                tasks.append(req_bob())
            for _ in range(10):
                tasks.append(req_invalid())
                tasks.append(req_crash())

            random.shuffle(tasks)
            results = await asyncio.gather(*tasks, return_exceptions=True)

            alice_success = 0
            bob_success = 0
            invalid_blocked = 0
            crash_count = 0

            for r in results:
                if isinstance(r, Exception):
                    crash_count += 1
                    continue
                if isinstance(r, BaseException):
                    raise r
                req_type, status, observed_uid = r
                if req_type == "alice":
                    assert status == 200
                    assert observed_uid == "alice_http"
                    alice_success += 1
                elif req_type == "bob":
                    assert status == 200
                    assert observed_uid == "bob_http"
                    bob_success += 1
                elif req_type == "invalid":
                    assert status == 401
                    invalid_blocked += 1
                elif req_type == "crash":
                    assert status == 500
                    crash_count += 1

            assert alice_success == 20
            assert bob_success == 20
            assert invalid_blocked == 10
            assert crash_count == 10

        # Invariant: Server thread root ContextVar MUST reset to None after all requests
        assert _CURRENT_PRINCIPAL.get() is None


# ==============================================================================
# Vector 4: ContextVar Post-Request Reset Guarantees
# ==============================================================================
class TestContextVarResetGuarantees:
    """Verifies ContextVar lifecycle guarantees and reset invariants."""

    def test_bind_session_principal_resets_on_normal_exit(self) -> None:
        """bind_session_principal resets ContextVar to prior state on normal exit."""
        assert _CURRENT_PRINCIPAL.get() is None
        p1 = PrincipalContext(
            user_id="user_1", role=RoleType.ENGINEER, allowed_scopes=["public_eng"]
        )

        with bind_session_principal(p1):
            assert _CURRENT_PRINCIPAL.get() == p1
            assert get_current_principal().user_id == "user_1"

        assert _CURRENT_PRINCIPAL.get() is None
        assert not has_bound_principal()

    def test_bind_session_principal_resets_on_exception(self) -> None:
        """bind_session_principal resets ContextVar to prior state on exception."""
        assert _CURRENT_PRINCIPAL.get() is None
        p1 = PrincipalContext(
            user_id="user_1", role=RoleType.ENGINEER, allowed_scopes=["public_eng"]
        )

        with pytest.raises(ZeroDivisionError), bind_session_principal(p1):
            assert _CURRENT_PRINCIPAL.get() == p1
            _ = 1 / 0

        assert _CURRENT_PRINCIPAL.get() is None
        assert not has_bound_principal()

    def test_nested_bind_session_principal_restores_outer_principal(self) -> None:
        """Nested bind_session_principal restores outer principal correctly."""
        p_outer = PrincipalContext(
            user_id="outer", role=RoleType.ENGINEER, allowed_scopes=["public_eng"]
        )
        p_inner = PrincipalContext(
            user_id="inner",
            role=RoleType.COMPLIANCE_OFFICER,
            allowed_scopes=["audit:compliance"],
        )

        with bind_session_principal(p_outer):
            assert get_current_principal().user_id == "outer"
            with bind_session_principal(p_inner):
                assert get_current_principal().user_id == "inner"
            assert get_current_principal().user_id == "outer"

        assert _CURRENT_PRINCIPAL.get() is None

    @pytest.mark.asyncio
    async def test_middleware_finally_block_resets_contextvar_on_401_403_500(self) -> None:
        """HTTP requests hitting 401, 403, and 500 guarantee clean ContextVar after return."""
        app = create_app()
        test_router = APIRouter()

        @test_router.get("/status/{code}")
        async def status_route(
            code: int,
            principal: Annotated[PrincipalContext, Depends(get_principal_context)],
        ) -> Response:
            with bind_session_principal(principal):
                if code == 403:
                    raise HTTPException(status_code=403, detail="Forbidden")
                if code == 500:
                    raise RuntimeError("Server crash")
                return Response(content="ok", status_code=200)

        app.include_router(test_router, prefix="/api/v1")

        valid_token = create_signed_principal_token(
            "test_user",
            RoleType.ENGINEER,
            ["public_eng"],
            secret_key=STRONG_CHALLENGER_SECRET,
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Test 401
            res_401 = await client.get(
                "/api/v1/status/200",
                headers={"Authorization": "Bearer bad_token"},
            )
            assert res_401.status_code == 401
            assert _CURRENT_PRINCIPAL.get() is None

            # 2. Test 403
            res_403 = await client.get(
                "/api/v1/status/403",
                headers={"Authorization": f"Bearer {valid_token}"},
            )
            assert res_403.status_code == 403
            assert _CURRENT_PRINCIPAL.get() is None

            # 3. Test 500
            try:
                res_500 = await client.get(
                    "/api/v1/status/500",
                    headers={"Authorization": f"Bearer {valid_token}"},
                )
                assert res_500.status_code == 500
            except RuntimeError:
                pass
            assert _CURRENT_PRINCIPAL.get() is None


# ==============================================================================
# Vector 5: Whitespace Edge Cases in Secrets and Headers
# ==============================================================================
class TestWhitespaceEdgeCases:
    """Verifies secret and header handling under leading/trailing/internal whitespace."""

    def test_environment_secret_with_surrounding_whitespace_is_trimmed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SCOUT_AUTH_HMAC_SECRET with whitespace, tabs, and newlines is trimmed."""
        padded_secret = f"  \t  \n {STRONG_CHALLENGER_SECRET}  \n  "
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", padded_secret)

        resolved = _resolve_auth_secret()
        assert resolved == STRONG_CHALLENGER_SECRET.encode("utf-8")

        # Create token with trimmed secret and verify it succeeds
        token = create_signed_principal_token("alice", RoleType.ENGINEER, ["public_eng"])
        principal = verify_signed_principal_token(token)
        assert principal.user_id == "alice"

    def test_whitespace_only_environment_secret_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment secret consisting solely of whitespace characters raises RuntimeError."""
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", "    \t\n   ")
        with pytest.raises(RuntimeError, match="SCOUT_AUTH_HMAC_SECRET is unset or empty"):
            _resolve_auth_secret()

    def test_whitespace_padded_weak_secret_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Weak secret padded with whitespace to reach 32 characters is stripped and rejected."""
        padded_weak = "       short_secret       "
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", padded_weak)
        with pytest.raises(RuntimeError, match="minimum 32 characters"):
            _resolve_auth_secret()

    @pytest.mark.parametrize(
        "auth_header_val",
        [
            f"   Bearer {STRONG_CHALLENGER_SECRET}",
            f"Bearer {STRONG_CHALLENGER_SECRET}   ",
            f"   Bearer {STRONG_CHALLENGER_SECRET}   ",
            f"Bearer     {STRONG_CHALLENGER_SECRET}",
        ],
    )
    def test_authorization_header_surrounding_whitespace_handled(
        self, auth_header_val: str
    ) -> None:
        """Authorization header with various whitespace permutations resolves token correctly."""
        token = create_signed_principal_token(
            "whitespace_user",
            RoleType.ENGINEER,
            ["public_eng"],
            secret_key=STRONG_CHALLENGER_SECRET,
        )
        real_header = auth_header_val.replace(STRONG_CHALLENGER_SECRET, token)

        principal = get_principal_context(authorization=real_header)
        assert principal.user_id == "whitespace_user"
        assert _CURRENT_PRINCIPAL.get() == principal
        _CURRENT_PRINCIPAL.set(None)

    @pytest.mark.parametrize(
        "invalid_header",
        [
            "   ",
            "Bearer",
            "Bearer ",
            "Bearer\t",
            "Bearer   ",
            f"Bearer\t{STRONG_CHALLENGER_SECRET}",
        ],
    )
    def test_invalid_authorization_header_whitespace_rejected_with_401(
        self, invalid_header: str
    ) -> None:
        """Invalid authorization whitespace formats raise HTTP 401."""
        with pytest.raises(HTTPException) as exc_info:
            get_principal_context(authorization=invalid_header)
        assert exc_info.value.status_code == 401

    def test_token_surrounding_whitespace_in_verify(self) -> None:
        """verify_signed_principal_token strips surrounding whitespace around token."""
        token = create_signed_principal_token(
            "strip_user",
            RoleType.ENGINEER,
            ["public_eng"],
            secret_key=STRONG_CHALLENGER_SECRET,
        )
        padded_token = f"   \t\n {token} \n   "
        principal = verify_signed_principal_token(padded_token)
        assert principal.user_id == "strip_user"

    def test_token_internal_whitespace_around_dot_rejected(self) -> None:
        """Whitespace around the dot delimiter causes signature verification failure (401)."""
        token = create_signed_principal_token(
            "dot_user",
            RoleType.ENGINEER,
            ["public_eng"],
            secret_key=STRONG_CHALLENGER_SECRET,
        )
        b64, sig = token.split(".")
        corrupted_token = f"{b64} . {sig}"
        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(corrupted_token)
        assert exc_info.value.status_code == 401


# ==============================================================================
# Vector 6: Bearer Scheme Case Sensitivity and Variations
# ==============================================================================
class TestBearerSchemeCaseSensitivity:
    """Verifies scheme casing resilience and strict rejection of unauthorized schemes."""

    @pytest.mark.parametrize(
        "scheme_prefix",
        [
            "Bearer ",
            "bearer ",
            "BEARER ",
            "bEaReR ",
            "BeArEr ",
        ],
    )
    def test_bearer_scheme_case_insensitivity(self, scheme_prefix: str) -> None:
        """Authorization header supports case-insensitive 'Bearer ' scheme."""
        token = create_signed_principal_token(
            "case_user",
            RoleType.ENGINEER,
            ["public_eng"],
            secret_key=STRONG_CHALLENGER_SECRET,
        )
        header_val = f"{scheme_prefix}{token}"
        principal = get_principal_context(authorization=header_val)
        assert principal.user_id == "case_user"
        _CURRENT_PRINCIPAL.set(None)

    @pytest.mark.parametrize(
        "invalid_scheme",
        [
            "Token ",
            "Basic ",
            "Digest ",
            "OAuth ",
            "MAC ",
            "Bearer:",
            "Bearer-",
        ],
    )
    def test_unsupported_auth_schemes_rejected_with_401(self, invalid_scheme: str) -> None:
        """Non-Bearer schemes raise HTTP 401."""
        token = create_signed_principal_token(
            "scheme_user",
            RoleType.ENGINEER,
            ["public_eng"],
            secret_key=STRONG_CHALLENGER_SECRET,
        )
        header_val = f"{invalid_scheme}{token}"
        with pytest.raises(HTTPException) as exc_info:
            get_principal_context(authorization=header_val)
        assert exc_info.value.status_code == 401
        expected_msg = "Authorization header must use 'Bearer <signed_token>' scheme"
        assert expected_msg in exc_info.value.detail


# ==============================================================================
# Vector 7: Huge Token Payloads and DoS Resistance
# ==============================================================================
class TestHugeTokenPayloadsAndDosResistance:
    """Stress-tests token verification with oversized payloads and DoS attack vectors."""

    def test_huge_1mb_token_fast_rejection(self) -> None:
        """1MB token string is rejected with HTTP 401 in sub-100ms without CPU lockup."""
        huge_token = "A" * 1_000_000 + "." + "sig"
        t0 = time.perf_counter()
        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(huge_token)
        elapsed = time.perf_counter() - t0

        assert exc_info.value.status_code == 401
        assert elapsed < 0.1

    def test_excessive_dot_delimiters_dos_rejection(self) -> None:
        """Token with 10,000 dots is rejected immediately as malformed (401)."""
        dots_token = ".".join(["x"] * 10_000)
        t0 = time.perf_counter()
        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(dots_token)
        elapsed = time.perf_counter() - t0

        assert exc_info.value.status_code == 401
        assert "Malformed signed principal token" in exc_info.value.detail
        assert elapsed < 0.05

    def test_huge_scopes_list_in_signed_token(self) -> None:
        """Validly signed token containing 5,000 scopes is parsed accurately."""
        huge_scopes = [f"scope:repo_tenant_{i:04d}" for i in range(5_000)]
        token = create_signed_principal_token(
            user_id="huge_scope_user",
            role=RoleType.ENGINEER,
            allowed_scopes=huge_scopes,
            secret_key=STRONG_CHALLENGER_SECRET,
        )
        principal = verify_signed_principal_token(token)
        assert principal.user_id == "huge_scope_user"
        assert len(principal.allowed_scopes) == 5_000

    def test_huge_user_id_string_in_signed_token(self) -> None:
        """Validly signed token with a 10,000-character user_id parses without error."""
        huge_uid = "u" * 10_000
        token = create_signed_principal_token(
            user_id=huge_uid,
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
            secret_key=STRONG_CHALLENGER_SECRET,
        )
        principal = verify_signed_principal_token(token)
        assert principal.user_id == huge_uid


# ==============================================================================
# Vector 8: White-Box Edge Conditions & Security Boundaries
# ==============================================================================
class TestWhiteBoxSecurityBoundaries:
    """Probes edge conditions identified during white-box code audit."""

    def test_whitebox_non_integer_exp_raises_value_error(self) -> None:
        """Signed token with non-integer 'exp' raises ValueError in verify_signed_principal_token.

        Documented finding: exp = int(data.get('exp', 0)) is outside the try/except block.
        If a token payload has a non-integer exp (e.g. string or None), int() raises ValueError.
        """
        payload = {"sub": "alice", "role": "engineer", "exp": "not_an_int"}
        b64 = (
            base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8"))
            .decode("ascii")
            .rstrip("=")
        )
        sig = hmac.new(
            STRONG_CHALLENGER_SECRET.encode("utf-8"),
            b64.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        tampered_token = f"{b64}.{sig}"

        with pytest.raises(ValueError):
            verify_signed_principal_token(tampered_token)

    def test_whitebox_non_integer_max_tokens_raises_value_error(self) -> None:
        """Signed token with non-integer 'max_tokens' raises ValueError in verify.

        Documented finding: max_session_paid_tokens = int(data.get('max_tokens', 50_000))
        is outside the try/except block.
        """
        payload = {
            "sub": "alice",
            "role": "engineer",
            "exp": int(time.time()) + 3600,
            "max_tokens": "invalid_int",
        }
        b64 = (
            base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8"))
            .decode("ascii")
            .rstrip("=")
        )
        sig = hmac.new(
            STRONG_CHALLENGER_SECRET.encode("utf-8"),
            b64.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        tampered_token = f"{b64}.{sig}"

        with pytest.raises(ValueError):
            verify_signed_principal_token(tampered_token)

    def test_whitebox_explicit_secret_key_whitespace_padding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit secret_key argument does not strip whitespace before length check.

        Documented finding: _resolve_auth_secret(secret_key=...) checks len(raw_key) < 32
        without stripping raw_key. A string of 32 spaces ' ' * 32 satisfies len >= 32.
        In contrast, SCOUT_AUTH_HMAC_SECRET checks env_secret.strip().
        """
        monkeypatch.setenv("SCOUT_ENV", "production")
        spaces_key = " " * 32
        resolved = _resolve_auth_secret(secret_key=spaces_key)
        assert resolved == spaces_key.encode("utf-8")
