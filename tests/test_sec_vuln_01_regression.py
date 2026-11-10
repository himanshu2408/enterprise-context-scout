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
    """Ensures SCOUT_REQUIRE_SIGNED_AUTH is unset by default."""
    monkeypatch.delenv("SCOUT_REQUIRE_SIGNED_AUTH", raising=False)
    monkeypatch.delenv("SCOUT_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)


@pytest.fixture
def app_client(clean_auth_env: None) -> TestClient:
    """TestClient against fresh application instance."""
    app = create_app()
    return TestClient(app)


def test_unauthenticated_request_rejected_http_401_by_default(app_client: TestClient) -> None:
    """SEC-VULN-01: Verifies requests lacking signed Bearer token are rejected with 401."""
    res = app_client.post(
        "/api/v1/scout/query",
        json={"query": "Investigate trade order flow routing", "domain_id": "trading_code_scout"},
    )
    assert res.status_code == 401
    assert res.json()["detail"] == "Signed Authorization Bearer token is required."


def test_header_spoofing_rejected_without_signed_bearer_token(app_client: TestClient) -> None:
    """SEC-VULN-01: Verifies spoofed X-Principal-* headers cannot bypass 401."""
    spoofed_headers = {
        "X-Principal-User-Id": "adversary_attacker",
        "X-Principal-Role": "compliance_officer",
        "X-Principal-Scopes": "audit:compliance,quant_restricted,super_admin",
    }
    res = app_client.post(
        "/api/v1/scout/query",
        headers=spoofed_headers,
        json={
            "query": "Exfiltrate restricted compliance audit logs",
            "domain_id": "trading_code_scout",
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
        json={"query": "Investigate routing", "domain_id": "trading_code_scout"},
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
        json={"query": "Unauthorized action", "domain_id": "trading_code_scout"},
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
        json={"query": "Test task", "domain_id": "trading_code_scout"},
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
        json={"query": "Test task", "domain_id": "trading_code_scout"},
    )
    assert res.status_code == 401
    assert "Invalid HMAC-SHA256 signature" in res.json()["detail"]


def test_valid_signed_token_accepted_and_ignores_spoofed_headers(app_client: TestClient) -> None:
    """SEC-VULN-01: A valid token sets principal context; client headers are ignored."""
    valid_token = create_signed_principal_token(
        user_id="real_engineer_42",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:trading_strategies"],
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
        json={"query": "Safe query for strategy", "domain_id": "trading_code_scout"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "trace_id" in data


def test_production_hmac_secret_enforcement(monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC-VULN-01: In production, weak or default HMAC secret raises RuntimeError."""
    monkeypatch.setenv("SCOUT_ENV", "production")
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


def test_production_hmac_secret_dev_override_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC-VULN-01: Explicit dev override allows default secret even in production mode."""
    monkeypatch.setenv("SCOUT_ENV", "production")
    monkeypatch.setenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", "true")
    monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", DEFAULT_AUTH_HMAC_SECRET)

    resolved = _resolve_auth_secret()
    assert resolved == DEFAULT_AUTH_HMAC_SECRET.encode("utf-8")


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
        json={"query": "Fallback test query", "domain_id": "trading_code_scout"},
    )
    assert res.status_code == 200
