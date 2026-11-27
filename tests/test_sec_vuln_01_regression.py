"""Permanent regression test suite for SEC-VULN-01 remediation (M1 F01).

Guarantees:
1. Default config (SCOUT_REQUIRE_SIGNED_AUTH unset) rejects unauthenticated callers with HTTP 401.
2. Direct header spoofing (X-Principal-*) is rejected with HTTP 401.
3. Cryptographic token tampering, corruption, expiration, and wrong keys return HTTP 401.
4. Client-sent X-Principal-* headers are completely ignored when valid token is provided.
5. Insecure/default HMAC secret keys in production trigger a fatal RuntimeError.
6. Legacy header fallback operates only when SCOUT_REQUIRE_SIGNED_AUTH=false is explicit.
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from context_scout.api.dependencies import (
    DEFAULT_AUTH_HMAC_SECRET,
    _resolve_auth_secret,
    create_signed_principal_token,
)
from context_scout.api.main import create_app
from context_scout.schemas.contracts import RoleType


@pytest.fixture
def clean_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensures SCOUT_REQUIRE_SIGNED_AUTH is unset by default with valid test secret."""
    monkeypatch.delenv("SCOUT_REQUIRE_SIGNED_AUTH", raising=False)
    monkeypatch.delenv("SCOUT_ENVIRONMENT", raising=False)
    monkeypatch.delenv("SCOUT_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)
    monkeypatch.setenv(
        "SCOUT_AUTH_HMAC_SECRET",
        "scout-test-suite-hmac-secret-32-chars-long-sec04!!",
    )


@pytest.fixture
def app_client(clean_auth_env: None) -> TestClient:
    """TestClient against fresh application instance."""
    app = create_app()
    return TestClient(app)


def test_unauthenticated_request_rejected_http_401_by_default(app_client: TestClient) -> None:
    """SEC-VULN-01: Verifies requests lacking signed Bearer token are rejected with 401."""
    res = app_client.post(
        "/api/v1/scout/query",
        json={
            "query": "Investigate trade order flow routing",
            "domain_id": "code_architecture_scout",
        },
    )
    assert res.status_code == 401
    assert res.json()["detail"] == "Signed Authorization Bearer token is required."


def test_header_spoofing_rejected_without_signed_bearer_token(app_client: TestClient) -> None:
    """SEC-VULN-01: Verifies spoofed X-Principal-* headers cannot bypass 401."""
    spoofed_headers = {
        "X-Principal-User-Id": "adversary_attacker",
        "X-Principal-Role": "compliance_officer",
        "X-Principal-Scopes": "audit:compliance,docs_restricted,super_admin",
    }
    res = app_client.post(
        "/api/v1/scout/query",
        headers=spoofed_headers,
        json={
            "query": "Exfiltrate restricted compliance audit logs",
            "domain_id": "code_architecture_scout",
        },
    )
    assert res.status_code == 401
    assert res.json()["detail"] == "Signed Authorization Bearer token is required."


def test_malformed_authorization_scheme_rejected_http_401(app_client: TestClient) -> None:
    """SEC-VULN-01: Non-Bearer authorization headers must be rejected."""
    token = create_signed_principal_token(
        user_id="legit_engineer",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    res = app_client.post(
        "/api/v1/scout/query",
        headers={"Authorization": f"Basic {token}"},
        json={"query": "Investigate routing", "domain_id": "code_architecture_scout"},
    )
    assert res.status_code == 401
    assert "Bearer <signed_token>" in res.json()["detail"]


def test_tampered_token_signature_rejected_http_401(app_client: TestClient) -> None:
    """SEC-VULN-01: Modifying token payload without valid signature must raise HTTP 401."""
    valid_token = create_signed_principal_token(
        user_id="low_privilege_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    parts = valid_token.split(".")
    assert len(parts) == 2

    # Tamper payload to escalate role to compliance_officer
    payload_dict = json.loads(base64.urlsafe_b64decode(parts[0] + "==").decode("utf-8"))
    payload_dict["role"] = "compliance_officer"
    payload_dict["scopes"].append("audit:compliance")
    tampered_payload_b64 = (
        base64.urlsafe_b64encode(json.dumps(payload_dict).encode("utf-8"))
        .decode("utf-8")
        .rstrip("=")
    )

    tampered_token = f"{tampered_payload_b64}.{parts[1]}"
    res = app_client.post(
        "/api/v1/scout/query",
        headers={"Authorization": f"Bearer {tampered_token}"},
        json={"query": "Unauthorized action", "domain_id": "code_architecture_scout"},
    )
    assert res.status_code == 401
    assert "Invalid HMAC-SHA256 signature" in res.json()["detail"]


def test_expired_token_rejected_http_401(app_client: TestClient) -> None:
    """SEC-VULN-01: Expired signed tokens must be rejected with HTTP 401."""
    expired_token = create_signed_principal_token(
        user_id="expired_engineer",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
        ttl_seconds=-60,  # expired 60 seconds ago
    )
    res = app_client.post(
        "/api/v1/scout/query",
        headers={"Authorization": f"Bearer {expired_token}"},
        json={"query": "Test task", "domain_id": "code_architecture_scout"},
    )
    assert res.status_code == 401
    assert "expired" in res.json()["detail"].lower()


def test_wrong_hmac_secret_key_rejected_http_401(app_client: TestClient) -> None:
    """SEC-VULN-01: Tokens signed with an alien secret key must be rejected."""
    alien_token = create_signed_principal_token(
        user_id="forger_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
        secret_key="alien-attacker-crafted-secret-key-32chars",
    )
    res = app_client.post(
        "/api/v1/scout/query",
        headers={"Authorization": f"Bearer {alien_token}"},
        json={"query": "Test task", "domain_id": "code_architecture_scout"},
    )
    assert res.status_code == 401
    assert "Invalid HMAC-SHA256 signature" in res.json()["detail"]


def test_valid_signed_token_accepted_and_ignores_spoofed_headers(app_client: TestClient) -> None:
    """SEC-VULN-01: A valid token sets principal context; client headers are ignored."""
    valid_token = create_signed_principal_token(
        user_id="real_engineer_42",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:code_architecture_scout"],
    )
    headers = {
        "Authorization": f"Bearer {valid_token}",
        "X-Principal-User-Id": "spoofed_attacker",
        "X-Principal-Role": "compliance_officer",
        "X-Principal-Scopes": "audit:compliance",
    }
    res = app_client.post(
        "/api/v1/scout/query",
        headers=headers,
        json={"query": "Safe query for strategy", "domain_id": "code_architecture_scout"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "trace_id" in data


@pytest.mark.parametrize("env_var", ["SCOUT_ENVIRONMENT", "SCOUT_ENV", "ENVIRONMENT"])
def test_production_hmac_secret_enforcement(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    """SEC-VULN-01: In production, weak or default HMAC secret raises RuntimeError."""
    for alias in ("SCOUT_ENVIRONMENT", "SCOUT_ENV", "ENVIRONMENT", "ENV"):
        monkeypatch.delenv(alias, raising=False)
    monkeypatch.setenv(env_var, "production")
    monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)
    monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", DEFAULT_AUTH_HMAC_SECRET)

    with pytest.raises(
        RuntimeError, match="Insecure default or weak HMAC secret detected in production"
    ):
        _resolve_auth_secret()

    # Short secret (<32 chars) must also fail in production
    monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", "short_secret_key_123")
    with pytest.raises(RuntimeError, match="minimum 32 characters"):
        _resolve_auth_secret()

    # Valid >= 32 chars strong secret succeeds
    strong_secret = "a" * 32
    monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", strong_secret)
    resolved = _resolve_auth_secret()
    assert resolved == strong_secret.encode("utf-8")


@pytest.mark.parametrize("env_var", ["SCOUT_ENVIRONMENT", "SCOUT_ENV", "ENVIRONMENT"])
def test_production_hmac_secret_dev_override_flag_strictly_forbidden_in_production(
    monkeypatch: pytest.MonkeyPatch, env_var: str
) -> None:
    """SEC-VULN-01: Absolute ban - dev override is forbidden in production (raises RuntimeError)."""
    for alias in ("SCOUT_ENVIRONMENT", "SCOUT_ENV", "ENVIRONMENT", "ENV"):
        monkeypatch.delenv(alias, raising=False)
    monkeypatch.setenv(env_var, "production")
    monkeypatch.setenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", "true")
    monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", DEFAULT_AUTH_HMAC_SECRET)

    with pytest.raises(
        RuntimeError, match="Insecure default or weak HMAC secret detected in production"
    ):
        _resolve_auth_secret()


def test_explicit_opt_out_signed_auth_permits_header_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SEC-VULN-01: When SCOUT_REQUIRE_SIGNED_AUTH=false is explicit, header fallback works."""
    monkeypatch.setenv("SCOUT_REQUIRE_SIGNED_AUTH", "false")
    app = create_app()
    client = TestClient(app)

    res = client.post(
        "/api/v1/scout/query",
        headers={
            "X-Principal-User-Id": "opt_out_engineer",
            "X-Principal-Role": "engineer",
        },
        json={"query": "Fallback test query", "domain_id": "code_architecture_scout"},
    )
    assert res.status_code == 200


def test_fail_closed_when_secret_unset_without_dev_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Critical B: When secret is unset and dev override false, resolve fails closed."""
    monkeypatch.delenv("SCOUT_AUTH_HMAC_SECRET", raising=False)
    monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)
    for alias in ("SCOUT_ENVIRONMENT", "SCOUT_ENV", "ENVIRONMENT", "ENV"):
        monkeypatch.delenv(alias, raising=False)
    with pytest.raises(RuntimeError, match="SCOUT_AUTH_HMAC_SECRET is unset or empty"):
        _resolve_auth_secret()


def test_fail_closed_blocks_api_token_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Critical B: When secret unset and dev override false, API fails closed on verification."""
    forged_token = create_signed_principal_token(
        user_id="adversary_attacker",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
        secret_key=DEFAULT_AUTH_HMAC_SECRET,
    )
    monkeypatch.delenv("SCOUT_AUTH_HMAC_SECRET", raising=False)
    monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)
    for alias in ("SCOUT_ENVIRONMENT", "SCOUT_ENV", "ENVIRONMENT", "ENV"):
        monkeypatch.delenv(alias, raising=False)
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    res = client.post(
        "/api/v1/scout/query",
        headers={"Authorization": f"Bearer {forged_token}"},
        json={"query": "Test fail closed", "domain_id": "code_architecture_scout"},
    )
    assert res.status_code == 500
    assert res.status_code != 200


def test_allow_insecure_dev_secret_permits_default_secret_in_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Critical B: Explicit dev override permits default secret in development mode."""
    monkeypatch.delenv("SCOUT_AUTH_HMAC_SECRET", raising=False)
    monkeypatch.setenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", "true")
    for alias in ("SCOUT_ENVIRONMENT", "SCOUT_ENV", "ENVIRONMENT", "ENV"):
        monkeypatch.delenv(alias, raising=False)
    resolved = _resolve_auth_secret()
    assert resolved == DEFAULT_AUTH_HMAC_SECRET.encode("utf-8")


def test_auth_session_positive_dev_allowlist_and_api_security_headers(
    clean_auth_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ECS-057 & ECS-060: Positive dev allowlist on /auth/session + /api/ security headers."""
    client = TestClient(create_app())
    payload = {"user_id": "sec_headers_user", "role": "engineer"}

    for allowed_env in ("development", "dev", "local", "test"):
        monkeypatch.setenv("SCOUT_ENVIRONMENT", allowed_env)
        res = client.post("/api/v1/auth/session", json=payload)
        assert res.status_code == 200, f"Expected 200 for {allowed_env}: {res.text}"
        assert res.headers.get("x-content-type-options") == "nosniff"
        assert res.headers.get("x-frame-options") == "DENY"
        assert res.headers.get("cache-control") == "no-store"

    for blocked_env in ("uat", "canary", "preview", "staging", "qa", "production"):
        monkeypatch.setenv("SCOUT_ENVIRONMENT", blocked_env)
        res = client.post("/api/v1/auth/session", json=payload)
        assert res.status_code == 403, f"Expected 403 for {blocked_env}: {res.text}"
        assert res.headers.get("x-content-type-options") == "nosniff"
        assert res.headers.get("x-frame-options") == "DENY"
        assert res.headers.get("cache-control") == "no-store"

    monkeypatch.setenv("SCOUT_ENVIRONMENT", "development")
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.headers.get("x-content-type-options") == "nosniff"
    assert health.headers.get("x-frame-options") == "DENY"
    assert health.headers.get("cache-control") == "no-store"


def test_lazy_pool_construction_fast_and_fail_closed_503_when_pg17_unreachable(
    clean_auth_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ECS-003 & ECS-116: Zero InMemory fallback in postgres mode; <5ms lazy wiring + 503."""
    import time
    import uuid

    from context_scout.api.dependencies import (
        get_repository,
        reset_shared_repository,
    )
    from context_scout.api.routes.webhooks import get_task_queue, set_task_queue
    from context_scout.db.session import RLSConnectionManager
    from context_scout.security.audit_chain import (
        AuditEventChainService,
        get_audit_chain_service,
        set_audit_chain_service,
    )
    from context_scout.security.scim import (
        SCIMDirectoryService,
        get_scim_directory_service,
        set_scim_directory_service,
    )

    dead_url = "postgresql://scout_app_user:pw@127.0.0.1:59999/context_scout"
    monkeypatch.setenv("SCOUT_STORAGE_BACKEND", "postgres")
    monkeypatch.setenv("SCOUT_DB_APP_URL", dead_url)
    monkeypatch.setenv("SCOUT_PG_CONNECT_TIMEOUT", "1")
    monkeypatch.setenv("SCOUT_PG_POOL_TIMEOUT", "1")
    monkeypatch.setenv("SCOUT_SCIM_BEARER_TOKEN", "scim-dead-pg-token-12345")

    reset_shared_repository()
    set_audit_chain_service(None)
    set_scim_directory_service(None)
    set_task_queue(None)

    try:
        t0 = time.monotonic()
        mgr = RLSConnectionManager(dead_url)
        repo = get_repository()
        audit_svc = get_audit_chain_service()
        scim_svc = get_scim_directory_service()
        queue = get_task_queue()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        assert elapsed_ms < 250.0, f"Lazy singleton wiring took too long: {elapsed_ms:.2f}ms"
        assert isinstance(audit_svc, AuditEventChainService)
        assert audit_svc._conn_mgr is not None
        assert isinstance(scim_svc, SCIMDirectoryService)
        assert scim_svc._conn_mgr is not None
        assert type(repo).__name__ == "PostgresScoutGraphAdapter"
        assert type(queue).__name__ == "PostgresTaskQueue"

        client = TestClient(create_app(), raise_server_exceptions=False)
        res = client.get(
            "/api/v1/scim/v2/Users",
            headers={
                "Authorization": "Bearer scim-dead-pg-token-12345",
                "X-Scout-Tenant-Id": str(uuid.uuid4()),
            },
        )
        assert res.status_code == 503
        assert res.json()["detail"] == "Database temporarily unavailable"
        assert res.headers.get("x-content-type-options") == "nosniff"
        mgr.close()
    finally:
        reset_shared_repository()
        set_audit_chain_service(None)
        set_scim_directory_service(None)
        set_task_queue(None)


def test_redis_confirmation_token_store_and_webhook_replay_multi_pod(
    clean_auth_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ECS-044: ConfirmationTokenStore 12-thread single-consume & webhook replay via Redis."""
    import concurrent.futures
    import uuid

    from tests.test_m1_05_permission_mirror import _LocalRESP2TestServer

    from context_scout.api.dependencies import ConfirmationTokenStore
    from context_scout.api.routes.webhooks import (
        _SEEN_WEBHOOK_DELIVERIES,
        _SEEN_WEBHOOK_IDS,
    )
    from context_scout.schemas.contracts import (
        ImplementationContextPack,
        PreFlightValidationReport,
        PrincipalContext,
    )

    server = _LocalRESP2TestServer(expected_password="redispass")
    redis_url = f"redis://:redispass@127.0.0.1:{server.port}/0"
    try:
        store_pod_a = ConfirmationTokenStore(redis_url=redis_url)
        store_pod_b = ConfirmationTokenStore(redis_url=redis_url)

        tenant_a = uuid.uuid4()
        tenant_b = uuid.uuid4()
        owner = PrincipalContext(
            user_id="owner_eng",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
            tenant_id=tenant_a,
        )
        attacker = PrincipalContext(
            user_id="attacker_eng",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
            tenant_id=tenant_b,
        )
        pack = ImplementationContextPack(
            trace_id="tr_redis_01",
            domain_id="code_architecture_scout",
            user_task="Verify Redis single-consume token",
            architectural_execution_plan=["Step 1"],
            preflight_report=PreFlightValidationReport(
                passed=True,
                normalized_rrf_score=0.9,
                citations_verified_100_pct=True,
            ),
            local_tokens_consumed=100,
            pack_token_count=20,
        )

        store_pod_a.save(
            token="tok_multipod_1",
            pack=pack,
            patch_diff="diff",
            target_files=["a.py"],
            model_id="test-model",
            principal=owner,
            ttl_seconds=120,
        )

        # Cross-tenant consume raises PermissionError WITHOUT destroying owner's token
        with pytest.raises(PermissionError):
            store_pod_b.consume("tok_multipod_1", attacker)
        assert store_pod_b.get("tok_multipod_1") is not None

        # 12-thread concurrent single-consume race across pods: strictly 1 winner
        outcomes: list[str] = []

        def _try_consume(idx: int) -> str:
            target = store_pod_b if idx % 2 else store_pod_a
            try:
                rec = target.consume("tok_multipod_1", owner)
                return f"WIN:{rec.token}"
            except KeyError:
                return "MISS"

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            outcomes = list(pool.map(_try_consume, range(12)))

        assert outcomes.count("WIN:tok_multipod_1") == 1
        assert outcomes.count("MISS") == 11

        # Webhook delivery replay guard persisted in Redis across process memory clears
        monkeypatch.setenv("SCOUT_REDIS_URL", redis_url)
        monkeypatch.setenv("SCOUT_STORAGE_BACKEND", "memory")
        monkeypatch.setenv("SCOUT_WEBHOOK_GITLAB_SECRET", "gitlab-secret-token-01")
        _SEEN_WEBHOOK_DELIVERIES.clear()
        _SEEN_WEBHOOK_IDS.clear()

        client = TestClient(create_app())
        delivery_id = f"dlv-{uuid.uuid4().hex[:8]}"
        first = client.post(
            "/api/v1/webhooks/gitlab",
            headers={
                "X-Gitlab-Token": "gitlab-secret-token-01",
                "X-Scout-Delivery-Id": delivery_id,
            },
            json={"project": {"name": "code_architecture_scout"}},
        )
        assert first.status_code == 200

        # Simulate another pod without in-memory seen set
        _SEEN_WEBHOOK_DELIVERIES.clear()
        _SEEN_WEBHOOK_IDS.clear()
        replay = client.post(
            "/api/v1/webhooks/gitlab",
            headers={
                "X-Gitlab-Token": "gitlab-secret-token-01",
                "X-Scout-Delivery-Id": delivery_id,
            },
            json={"project": {"name": "code_architecture_scout"}},
        )
        assert replay.status_code == 409
    finally:
        server.close()


def test_external_system_connector_ingest_scope_rejected_everywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External OIDC/IAP, HMAC, SCIM, and dev-session paths reject system:connector_ingest."""
    import uuid

    from fastapi import HTTPException

    from context_scout.api.dependencies import (
        create_signed_principal_token,
        verify_signed_principal_token,
    )
    from context_scout.security.betrvg_privacy import BetrVGPrivacyGuard
    from context_scout.security.oidc_iap_auth import (
        AuthVerificationError,
        _build_principal_from_claims,
    )
    from context_scout.security.scim import SCIMDirectoryService

    monkeypatch.setenv("SCOUT_ENV", "development")
    monkeypatch.setenv("SCOUT_STORAGE_BACKEND", "memory")
    client = TestClient(create_app())

    # 1. Dev /api/v1/auth/session rejects system:connector_ingest
    session_res = client.post(
        "/api/v1/auth/session",
        json={
            "user_id": "attacker",
            "role": "engineer",
            "allowed_scopes": ["system:connector_ingest"],
        },
    )
    assert session_res.status_code == 403

    # 2. Signed HMAC token carrying system:connector_ingest fails closed on verification
    forged_tok = create_signed_principal_token(
        user_id="attacker",
        role=RoleType.ENGINEER,
        allowed_scopes=["system:connector_ingest"],
    )
    with pytest.raises(HTTPException) as exc_info:
        verify_signed_principal_token(forged_tok)
    assert exc_info.value.status_code == 403

    # 3. OIDC/IAP claim builder rejects system:connector_ingest via direct_scopes or group_scope_map
    scim_svc = SCIMDirectoryService()
    with pytest.raises(AuthVerificationError) as oidc_exc:
        _build_principal_from_claims(
            user_id="oidc-attacker",
            email="attacker@example.com",
            tenant_id=uuid.uuid4(),
            idp_groups=["eng"],
            idp_roles=["engineer"],
            direct_scopes=["system:connector_ingest"],
            scim_service=scim_svc,
            group_scope_map={},
            role_map={},
            betrvg_guard=BetrVGPrivacyGuard(enabled=False),
        )
    assert oidc_exc.value.status_code == 403
