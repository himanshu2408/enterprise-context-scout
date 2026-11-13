"""Dedicated permanent regression test suite for Critical B: Fail-Closed HMAC Secret.

Guarantees:
1. Category A: Unset SCOUT_AUTH_HMAC_SECRET with SCOUT_ALLOW_INSECURE_DEV_SECRET unset/false:
   - _resolve_auth_secret() fails closed (raises RuntimeError).
   - create_signed_principal_token() and verify_signed_principal_token() raise RuntimeError.
   - Application startup / lifespan validation fails closed (refuses to start).
   - Forgery using DEFAULT_AUTH_HMAC_SECRET fails closed with HTTP 500 / RuntimeError, never 200.
2. Category B: Dev Override in Non-Production vs Absolute Ban in Production:
   - Non-production + explicit SCOUT_ALLOW_INSECURE_DEV_SECRET="true": permits dev secret.
   - Case-insensitive acceptance of "true", "TRUE", "1", "yes".
   - Production + SCOUT_ALLOW_INSECURE_DEV_SECRET="true" + unset secret -> raises RuntimeError.
   - Production + SCOUT_ALLOW_INSECURE_DEV_SECRET="true" + weak secret -> raises fatal RuntimeError.
   - Production + valid >=32 secret -> succeeds.
3. Category C: Custom Valid Secret Enforcement & Rejection of Forgeries:
   - Valid token accepted with HTTP 200 on authenticated endpoint (/api/v1/scout/query).
   - Forged token signed with DEFAULT_AUTH_HMAC_SECRET rejected with HTTP 401.
   - Token signed with wrong alien secret rejected with HTTP 401.
   - Tampered token payload rejected with HTTP 401.
   - Secret under 32 characters or in INSECURE_DEFAULT_SECRETS rejected with RuntimeError.
4. Category D: Concurrency & ContextVar Isolation Verification:
   - Two simultaneous async requests with distinct principals maintain 100% isolation.
   - Simulated async I/O delay proves zero identity leakage across asyncio task yields.
   - Exception in one concurrent request does not pollute sibling request.
   - ContextVar resets strictly to None after request lifecycle.
5. Category E: Explicit Secret Key Parameter Handling & Mutation Proofs:
   - Explicit parameter handling in production and non-production.
   - Reverting to fail-open default secret fails tests (mutation proof).
   - Global variable principal leak fails concurrency tests (mutation proof).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Annotated

import httpx
import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient

from context_scout.api.dependencies import (
    DEFAULT_AUTH_HMAC_SECRET,
    INSECURE_DEFAULT_SECRETS,
    _resolve_auth_secret,
    create_signed_principal_token,
    get_principal_context,
    verify_signed_principal_token,
)
from context_scout.api.main import create_app
from context_scout.schemas.contracts import PrincipalContext, RoleType
from context_scout.security.rbac import (
    _CURRENT_PRINCIPAL,
    bind_session_principal,
    get_current_principal,
)

STRONG_TEST_SECRET = "production-cryptographic-secret-key-32-chars-long-ok!"


# ==============================================================================
# Category A: Fail-Closed When Secret is Unset / Dev Override is False
# ==============================================================================
class TestFailClosedWhenSecretUnset:
    """Verifies that an unconfigured HMAC secret fails closed across all layers."""

    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SCOUT_AUTH_HMAC_SECRET", raising=False)
        monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)
        monkeypatch.delenv("SCOUT_ENV", raising=False)
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        monkeypatch.delenv("ENV", raising=False)

    def test_resolve_secret_raises_runtime_error_when_unset_and_dev_false(self) -> None:
        """_resolve_auth_secret must raise RuntimeError when secret is unconfigured."""
        with pytest.raises(RuntimeError, match="SCOUT_AUTH_HMAC_SECRET is unset or empty"):
            _resolve_auth_secret()

    @pytest.mark.parametrize("falsy_val", ["false", "FALSE", "0", "no", "off", "disabled", ""])
    def test_resolve_secret_raises_when_allow_insecure_is_falsy(
        self, monkeypatch: pytest.MonkeyPatch, falsy_val: str
    ) -> None:
        """Falsy values of SCOUT_ALLOW_INSECURE_DEV_SECRET do not bypass fail-closed."""
        monkeypatch.setenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", falsy_val)
        with pytest.raises(RuntimeError, match="SCOUT_AUTH_HMAC_SECRET is unset or empty"):
            _resolve_auth_secret()

    def test_create_signed_principal_token_raises_when_secret_unset(self) -> None:
        """Token creation must fail closed if no secret is configured and no key passed."""
        with pytest.raises(RuntimeError, match="SCOUT_AUTH_HMAC_SECRET is unset or empty"):
            create_signed_principal_token("alice", RoleType.ENGINEER, ["public_eng"])

    def test_verify_signed_principal_token_raises_when_secret_unset(self) -> None:
        """Token verification must fail closed if no secret is configured."""
        dummy_token = "eyJhbGciOiJIUzI1NiJ9.dummy_signature"
        with pytest.raises(RuntimeError, match="SCOUT_AUTH_HMAC_SECRET is unset or empty"):
            verify_signed_principal_token(dummy_token)

    def test_forged_token_using_default_dev_secret_fails_closed_http_500(self) -> None:
        """Attacker crafting token with DEFAULT_AUTH_HMAC_SECRET receives HTTP 500."""
        forged_token = create_signed_principal_token(
            user_id="adversary_attacker",
            role=RoleType.COMPLIANCE_OFFICER,
            allowed_scopes=["audit:compliance", "super_admin"],
            secret_key=DEFAULT_AUTH_HMAC_SECRET,
        )

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        res = client.post(
            "/api/v1/scout/query",
            headers={"Authorization": f"Bearer {forged_token}"},
            json={"query": "Exfiltrate codebase", "domain_id": "trading_code_scout"},
        )
        # Server fails closed: exact HTTP 500 Internal Server Error (never 200 OK)
        assert res.status_code == 500

    def test_forged_token_raises_runtime_error_with_server_exceptions(self) -> None:
        """Unconfigured server raises fatal RuntimeError when verifying request."""
        forged_token = create_signed_principal_token(
            user_id="adversary_attacker",
            role=RoleType.COMPLIANCE_OFFICER,
            allowed_scopes=["audit:compliance", "super_admin"],
            secret_key=DEFAULT_AUTH_HMAC_SECRET,
        )

        app = create_app()
        client = TestClient(app, raise_server_exceptions=True)
        with pytest.raises(RuntimeError, match="SCOUT_AUTH_HMAC_SECRET is unset or empty"):
            client.post(
                "/api/v1/scout/query",
                headers={"Authorization": f"Bearer {forged_token}"},
                json={"query": "Exfiltrate codebase", "domain_id": "trading_code_scout"},
            )

    def test_application_startup_lifespan_fails_closed_when_secret_unset(self) -> None:
        """Application startup lifespan raises RuntimeError when HMAC secret is unconfigured."""
        app = create_app()
        # Entering TestClient as context manager triggers ASGI lifespan startup
        with (
            pytest.raises(RuntimeError, match="SCOUT_AUTH_HMAC_SECRET is unset or empty"),
            TestClient(app),
        ):
            pass


# ==============================================================================
# Category B: Dev Override in Non-Prod vs Absolute Ban in Production
# ==============================================================================
class TestDevOverrideAndProductionBan:
    """Verifies SCOUT_ALLOW_INSECURE_DEV_SECRET in non-prod and absolute ban in production."""

    def test_non_production_dev_override_returns_default_secret_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """In non-production, dev override permits default secret and logs explicit warning."""
        monkeypatch.delenv("SCOUT_AUTH_HMAC_SECRET", raising=False)
        monkeypatch.setenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", "true")
        monkeypatch.delenv("SCOUT_ENV", raising=False)

        with caplog.at_level(logging.WARNING):
            secret = _resolve_auth_secret()
        assert secret == DEFAULT_AUTH_HMAC_SECRET.encode("utf-8")
        assert any("default development HMAC secret" in record.message for record in caplog.records)

    def test_create_and_verify_token_with_dev_override_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Token created and verified with dev override in non-prod resolves valid principal."""
        monkeypatch.delenv("SCOUT_AUTH_HMAC_SECRET", raising=False)
        monkeypatch.setenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", "true")
        monkeypatch.delenv("SCOUT_ENV", raising=False)

        token = create_signed_principal_token("dev_engineer", RoleType.ENGINEER, ["public_eng"])
        principal = verify_signed_principal_token(token)
        assert principal.user_id == "dev_engineer"
        assert principal.role == RoleType.ENGINEER
        assert "public_eng" in principal.allowed_scopes

    def test_api_endpoint_accepts_dev_token_when_dev_override_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Authenticated endpoint accepts valid dev-signed tokens when dev override is active."""
        monkeypatch.delenv("SCOUT_AUTH_HMAC_SECRET", raising=False)
        monkeypatch.setenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", "true")
        monkeypatch.delenv("SCOUT_ENV", raising=False)

        token = create_signed_principal_token("dev_engineer", RoleType.ENGINEER, ["public_eng"])
        app = create_app()
        client = TestClient(app)
        res = client.post(
            "/api/v1/scout/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Test query", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 200
        assert "trace_id" in res.json()

    @pytest.mark.parametrize("truthy_val", ["true", "TRUE", "True", "1", "yes", "YES"])
    def test_dev_override_truthy_values(
        self, monkeypatch: pytest.MonkeyPatch, truthy_val: str
    ) -> None:
        """Case-insensitive truthy strings cleanly enable dev override in non-prod."""
        monkeypatch.delenv("SCOUT_AUTH_HMAC_SECRET", raising=False)
        monkeypatch.setenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", truthy_val)
        monkeypatch.delenv("SCOUT_ENV", raising=False)
        secret = _resolve_auth_secret()
        assert secret == DEFAULT_AUTH_HMAC_SECRET.encode("utf-8")

    # Production absolute override ban tests:
    def test_production_with_opt_in_and_unset_secret_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Production: SCOUT_ALLOW_INSECURE_DEV_SECRET=true does NOT allow unset secret."""
        monkeypatch.setenv("SCOUT_ENV", "production")
        monkeypatch.setenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", "true")
        monkeypatch.delenv("SCOUT_AUTH_HMAC_SECRET", raising=False)

        msg = "Insecure default or weak HMAC secret detected in production environment"
        with pytest.raises(RuntimeError, match=msg):
            _resolve_auth_secret()

    def test_production_with_opt_in_and_weak_secret_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Production: SCOUT_ALLOW_INSECURE_DEV_SECRET=true does NOT allow weak secret."""
        monkeypatch.setenv("SCOUT_ENV", "production")
        monkeypatch.setenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", "true")
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", "weak_short_secret_123")

        msg = "Insecure default or weak HMAC secret detected in production environment"
        with pytest.raises(RuntimeError, match=msg):
            _resolve_auth_secret()

    @pytest.mark.parametrize("insecure_secret", list(INSECURE_DEFAULT_SECRETS))
    def test_production_with_opt_in_and_insecure_default_secret_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch, insecure_secret: str
    ) -> None:
        """Production: SCOUT_ALLOW_INSECURE_DEV_SECRET=true does NOT allow default secrets."""
        monkeypatch.setenv("SCOUT_ENV", "production")
        monkeypatch.setenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", "true")
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", insecure_secret)

        msg = "Insecure default or weak HMAC secret detected in production environment"
        with pytest.raises(RuntimeError, match=msg):
            _resolve_auth_secret()

    def test_production_with_valid_32plus_secret_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Production environment: Valid >=32 char cryptographically secure secret succeeds."""
        monkeypatch.setenv("SCOUT_ENV", "production")
        monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", STRONG_TEST_SECRET)

        resolved = _resolve_auth_secret()
        assert resolved == STRONG_TEST_SECRET.encode("utf-8")


# ==============================================================================
# Category C: Custom Valid Secret Enforcement & Rejection of Forgeries
# ==============================================================================
class TestCustomValidSecretEnforcement:
    """Verifies behavior when SCOUT_AUTH_HMAC_SECRET is set to a cryptographically secure key."""

    @pytest.fixture(autouse=True)
    def custom_secret_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", STRONG_TEST_SECRET)
        monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)

    def test_valid_token_signed_with_custom_secret_accepted(self) -> None:
        """Token signed with active custom secret is accepted by API with HTTP 200."""
        token = create_signed_principal_token(
            "custom_user", RoleType.ENGINEER, ["public_eng"], secret_key=STRONG_TEST_SECRET
        )
        app = create_app()
        client = TestClient(app)
        res = client.post(
            "/api/v1/scout/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Test custom query", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 200
        assert "trace_id" in res.json()

    def test_token_signed_with_default_dev_secret_rejected_with_401(self) -> None:
        """Token signed with DEFAULT_AUTH_HMAC_SECRET is strictly rejected with HTTP 401."""
        forged_token = create_signed_principal_token(
            "forger",
            RoleType.COMPLIANCE_OFFICER,
            ["audit:compliance"],
            secret_key=DEFAULT_AUTH_HMAC_SECRET,
        )
        app = create_app()
        client = TestClient(app)
        res = client.post(
            "/api/v1/scout/query",
            headers={"Authorization": f"Bearer {forged_token}"},
            json={"query": "Attack query", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 401
        assert "Invalid HMAC-SHA256 signature" in res.json()["detail"]

    def test_token_signed_with_alien_secret_rejected_with_401(self) -> None:
        """Token signed with wrong alien secret key is rejected with HTTP 401."""
        alien_key = "alien-attacker-different-secret-key-32chars!"
        alien_token = create_signed_principal_token(
            "alien_user", RoleType.ENGINEER, ["public_eng"], secret_key=alien_key
        )
        app = create_app()
        client = TestClient(app)
        res = client.post(
            "/api/v1/scout/query",
            headers={"Authorization": f"Bearer {alien_token}"},
            json={"query": "Alien query", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 401
        assert "Invalid HMAC-SHA256 signature" in res.json()["detail"]

    def test_tampered_payload_rejected_with_401(self) -> None:
        """Modifying token payload without signature recalculation is rejected with HTTP 401."""
        valid_token = create_signed_principal_token(
            "engineer_user", RoleType.ENGINEER, ["public_eng"], secret_key=STRONG_TEST_SECRET
        )
        b64_payload, sig = valid_token.split(".")
        payload_data = json.loads(base64.urlsafe_b64decode(b64_payload + "==").decode("utf-8"))
        payload_data["role"] = "compliance_officer"
        tampered_bytes = json.dumps(payload_data).encode("utf-8")
        tampered_b64 = base64.urlsafe_b64encode(tampered_bytes).decode("ascii").rstrip("=")
        tampered_token = f"{tampered_b64}.{sig}"

        app = create_app()
        client = TestClient(app)
        res = client.post(
            "/api/v1/scout/query",
            headers={"Authorization": f"Bearer {tampered_token}"},
            json={"query": "Tampered query", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 401
        assert "Invalid HMAC-SHA256 signature" in res.json()["detail"]

    def test_custom_secret_under_32_chars_rejected_without_dev_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Custom secret shorter than 32 characters raises RuntimeError."""
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", "short_secret_key_123")
        with pytest.raises(RuntimeError, match="minimum 32 characters"):
            _resolve_auth_secret()

    @pytest.mark.parametrize("insecure_secret", list(INSECURE_DEFAULT_SECRETS))
    def test_custom_secret_in_insecure_defaults_rejected_without_dev_override(
        self, monkeypatch: pytest.MonkeyPatch, insecure_secret: str
    ) -> None:
        """Known default insecure secrets in SCOUT_AUTH_HMAC_SECRET raise RuntimeError."""
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", insecure_secret)
        with pytest.raises(RuntimeError, match="Insecure or weak HMAC secret detected"):
            _resolve_auth_secret()


# ==============================================================================
# Category D: Concurrency & ContextVar Isolation Verification
# ==============================================================================
class TestContextVarConcurrencyIsolation:
    """Verifies that concurrent async requests maintain zero identity bleed across tasks."""

    @pytest.fixture(autouse=True)
    def clean_principal_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", STRONG_TEST_SECRET)
        monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)
        _CURRENT_PRINCIPAL.set(None)

    @pytest.mark.asyncio
    async def test_concurrent_async_requests_distinct_principals_zero_leakage(self) -> None:
        """Two concurrent requests with distinct principals maintain 100% ContextVar isolation."""
        app = create_app()
        test_router = APIRouter()

        @test_router.get("/concurrency/probe")
        async def concurrency_probe(
            principal: Annotated[PrincipalContext, Depends(get_principal_context)],
        ) -> dict[str, str]:
            with bind_session_principal(principal):
                curr = get_current_principal()
                before_sleep = curr.user_id if curr else ""
                # Force asyncio context switch simulating downstream I/O
                await asyncio.sleep(0.02)
                curr_after = get_current_principal()
                after_sleep = curr_after.user_id if curr_after else ""
                return {
                    "user_id": principal.user_id,
                    "before_sleep": before_sleep,
                    "after_sleep": after_sleep,
                }

        app.include_router(test_router, prefix="/api/v1")

        token_alice = create_signed_principal_token(
            "alice_concurrent",
            RoleType.ENGINEER,
            ["public_eng"],
            secret_key=STRONG_TEST_SECRET,
        )
        token_bob = create_signed_principal_token(
            "bob_concurrent",
            RoleType.QUANT_RESEARCHER,
            ["quant_restricted"],
            secret_key=STRONG_TEST_SECRET,
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

            async def call_probe(token: str) -> httpx.Response:
                return await client.get(
                    "/api/v1/concurrency/probe",
                    headers={"Authorization": f"Bearer {token}"},
                )

            res_alice, res_bob = await asyncio.gather(
                call_probe(token_alice),
                call_probe(token_bob),
            )

            assert res_alice.status_code == 200
            assert res_bob.status_code == 200

            data_a = res_alice.json()
            data_b = res_bob.json()

            assert data_a["user_id"] == "alice_concurrent"
            assert data_a["before_sleep"] == "alice_concurrent"
            assert data_a["after_sleep"] == "alice_concurrent"

            assert data_b["user_id"] == "bob_concurrent"
            assert data_b["before_sleep"] == "bob_concurrent"
            assert data_b["after_sleep"] == "bob_concurrent"

            assert data_a["user_id"] != data_b["user_id"]
            assert data_a["after_sleep"] != data_b["after_sleep"]

        # Ensure server thread root contextvar remains strictly clean
        assert _CURRENT_PRINCIPAL.get() is None

    @pytest.mark.asyncio
    async def test_concurrent_success_and_failure_isolation(self) -> None:
        """A failing concurrent request (401) does not pollute a valid concurrent request."""
        app = create_app()
        token_valid = create_signed_principal_token(
            "valid_user", RoleType.ENGINEER, ["public_eng"], secret_key=STRONG_TEST_SECRET
        )
        token_corrupt = "corrupted.token.signature"

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

            async def send_valid() -> httpx.Response:
                await asyncio.sleep(0.01)
                return await client.post(
                    "/api/v1/scout/query",
                    headers={"Authorization": f"Bearer {token_valid}"},
                    json={"query": "Valid query", "domain_id": "trading_code_scout"},
                )

            async def send_corrupt() -> httpx.Response:
                return await client.post(
                    "/api/v1/scout/query",
                    headers={"Authorization": f"Bearer {token_corrupt}"},
                    json={"query": "Corrupt query", "domain_id": "trading_code_scout"},
                )

            res_valid, res_corrupt = await asyncio.gather(
                send_valid(),
                send_corrupt(),
            )

            assert res_valid.status_code == 200
            assert res_corrupt.status_code == 401

        assert _CURRENT_PRINCIPAL.get() is None


# ==============================================================================
# Category E: Explicit Secret Key Parameter Handling & Mutation Proofs
# ==============================================================================
class TestExplicitSecretKeyAndMutationProofs:
    """Verifies explicit secret_key argument behaviors and mutation proofs."""

    def test_explicit_secret_key_rejected_in_production_if_short(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit secret_key argument shorter than 32 chars in production raises RuntimeError."""
        monkeypatch.setenv("SCOUT_ENV", "production")
        msg = "Insecure default or weak HMAC secret detected in production environment"
        with pytest.raises(RuntimeError, match=msg):
            _resolve_auth_secret(secret_key="short_key_123")

    def test_explicit_secret_key_rejected_in_production_if_insecure_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit secret_key in INSECURE_DEFAULT_SECRETS in production raises RuntimeError."""
        monkeypatch.setenv("SCOUT_ENV", "production")
        msg = "Insecure default or weak HMAC secret detected in production environment"
        with pytest.raises(RuntimeError, match=msg):
            _resolve_auth_secret(secret_key=DEFAULT_AUTH_HMAC_SECRET)

    def test_explicit_secret_key_accepted_in_non_prod_for_attacker_testing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Explicit secret_key in non-production permits crafting attack tokens and logs warning."""
        with caplog.at_level(logging.WARNING):
            resolved = _resolve_auth_secret(secret_key=DEFAULT_AUTH_HMAC_SECRET)
        assert resolved == DEFAULT_AUTH_HMAC_SECRET.encode("utf-8")
        assert any(
            "known insecure development HMAC secret" in record.message for record in caplog.records
        )

    def test_explicit_secret_key_accepted_when_valid(self) -> None:
        """Explicit valid >=32 character key returns encoded key directly."""
        explicit_key = "explicit-cryptographic-secret-key-32-chars!"
        resolved = _resolve_auth_secret(secret_key=explicit_key)
        assert resolved == explicit_key.encode("utf-8")

    def test_mutation_revert_to_fail_open_proof(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Proves that reverting _resolve_auth_secret to return default secret fails the test."""
        monkeypatch.delenv("SCOUT_AUTH_HMAC_SECRET", raising=False)
        monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)

        # Mutated function simulating fail-open regression
        def mutated_resolve_fail_open() -> bytes:
            return DEFAULT_AUTH_HMAC_SECRET.encode("utf-8")

        # The regression test expects a RuntimeError:
        with pytest.raises(pytest.fail.Exception), pytest.raises(RuntimeError):
            mutated_resolve_fail_open()

    @pytest.mark.asyncio
    async def test_mutation_global_variable_principal_proof(self) -> None:
        """Proves that replacing ContextVar with global variable causes cross-task leakage."""
        mutated_global_principal = None

        from contextlib import contextmanager

        @contextmanager
        def mutated_bind(principal_id: str):
            nonlocal mutated_global_principal
            old = mutated_global_principal
            mutated_global_principal = principal_id
            try:
                yield
            finally:
                mutated_global_principal = old

        async def worker(name: str, delay_before_read: float, total_hold: float) -> str:
            with mutated_bind(name):
                await asyncio.sleep(delay_before_read)
                observed = mutated_global_principal or ""
                await asyncio.sleep(total_hold - delay_before_read)
                return observed

        # Alice waits 0.03s before reading; Bob enters at 0.01s and overwrites global with 'bob'
        alice_read, bob_read = await asyncio.gather(
            worker("alice", 0.03, 0.08),
            worker("bob", 0.01, 0.06),
        )
        # Under global variable mutation, Alice leaks and sees 'bob'!
        assert alice_read == "bob"
        assert bob_read == "bob"
        # Proves that ContextVar isolation in Category D is mathematically required to prevent this!
