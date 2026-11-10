"""Adversarial empirical challenge suite for authentication and cryptographic boundaries.

Challenger Test Suite:
1. Header Forging Attacks Without Bearer Token (SEC-VULN-01 direct exploitation)
2. Header Forging Alongside Valid Bearer Token (Privilege Escalation & Override Attacks)
3. Cryptographic HMAC Payload & Signature Tampering
4. Expired, Zero, and Time-Manipulated Tokens
5. Alien, Unknown, and Insecure Secret Key Attacks
6. Production HMAC Secret Key Security Enforcement
7. Permission Mirroring, Scope Sanitization, and Injection Resistance
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from context_scout.api.dependencies import (
    DEFAULT_AUTH_HMAC_SECRET,
    INSECURE_DEFAULT_SECRETS,
    _resolve_auth_secret,
    create_signed_principal_token,
    get_principal_context,
    is_production_environment,
    verify_signed_principal_token,
)
from context_scout.api.main import create_app
from context_scout.schemas.contracts import PrincipalContext, RoleType
from context_scout.security.permission_mirror import (
    EnterpriseIdentityProfile,
    GitLabRole,
    PermissionMirror,
    sanitize_scope_identifier,
)


@pytest.fixture
def clean_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensures a clean testing environment for auth testing."""
    monkeypatch.delenv("SCOUT_REQUIRE_SIGNED_AUTH", raising=False)
    monkeypatch.delenv("SCOUT_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)
    monkeypatch.delenv("SCOUT_AUTH_HMAC_SECRET", raising=False)


@pytest.fixture
def client(clean_auth_env: None) -> TestClient:
    """TestClient against fresh context-scout API."""
    return TestClient(create_app(), raise_server_exceptions=False)


# ==============================================================================
# Category 1: Header Forging Without Bearer Token
# ==============================================================================
class TestHeaderForgingAttacksWithoutToken:
    """Adversarial tests attempting to forge out-of-band X-Principal-* headers."""

    def test_attack_01_forged_user_id_rejected_on_query(self, client: TestClient) -> None:
        """Attacker sends X-Principal-User-Id: admin to scout query endpoint."""
        res = client.post(
            "/api/v1/scout/query",
            headers={"X-Principal-User-Id": "admin"},
            json={"query": "Exfiltrate codebase", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 401
        assert res.json()["detail"] == "Signed Authorization Bearer token is required."

    def test_attack_02_forged_role_escalation_rejected_on_query(self, client: TestClient) -> None:
        """Attacker attempts privilege escalation by forging compliance_officer role."""
        res = client.post(
            "/api/v1/scout/query",
            headers={"X-Principal-Role": "compliance_officer"},
            json={"query": "Read restricted audit logs", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 401
        assert res.json()["detail"] == "Signed Authorization Bearer token is required."

    def test_attack_03_forged_scopes_rejected_on_query(self, client: TestClient) -> None:
        """Attacker attempts scope expansion by supplying raw scopes header."""
        res = client.post(
            "/api/v1/scout/query",
            headers={"X-Principal-Scopes": "audit:compliance,quant_restricted,super_admin"},
            json={"query": "Access all domains", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 401
        assert res.json()["detail"] == "Signed Authorization Bearer token is required."

    def test_attack_04_forged_full_identity_suite_rejected_on_query(
        self, client: TestClient
    ) -> None:
        """Attacker supplies full forged identity triple (User, Role, Scopes)."""
        res = client.post(
            "/api/v1/scout/query",
            headers={
                "X-Principal-User-Id": "root_attacker",
                "X-Principal-Role": "compliance_officer",
                "X-Principal-Scopes": "audit:compliance,public_quant,quant_restricted",
            },
            json={"query": "Full access test", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 401
        assert res.json()["detail"] == "Signed Authorization Bearer token is required."

    def test_attack_05_forged_headers_on_stream_endpoint(self, client: TestClient) -> None:
        """Attacker attempts to forge headers on the SSE streaming endpoint."""
        res = client.post(
            "/api/v1/scout/stream",
            headers={
                "X-Principal-User-Id": "stream_attacker",
                "X-Principal-Role": "engineer",
            },
            json={"query": "Stream test", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 401
        assert res.json()["detail"] == "Signed Authorization Bearer token is required."

    def test_attack_06_forged_headers_on_handoff_preview(self, client: TestClient) -> None:
        """Attacker attempts to access handoff preview with forged headers."""
        res = client.post(
            "/api/v1/handoff/preview",
            headers={"X-Principal-User-Id": "handoff_attacker"},
            json={"model_id": "gemini-3.8-flash", "pack": {}},
        )
        assert res.status_code == 401
        assert res.json()["detail"] == "Signed Authorization Bearer token is required."

    def test_attack_07_forged_headers_on_handoff_apply(self, client: TestClient) -> None:
        """Attacker attempts to execute code modification via apply with forged headers."""
        res = client.post(
            "/api/v1/handoff/apply",
            headers={"X-Principal-User-Id": "confirm_attacker"},
            json={"confirmation_token": "fake-token-1234"},
        )
        assert res.status_code == 401
        assert res.json()["detail"] == "Signed Authorization Bearer token is required."

    @pytest.mark.parametrize(
        "header_case",
        [
            "x-principal-user-id",
            "X-PRINCIPAL-USER-ID",
            "X-pRiNcIpAl-UsEr-Id",
            "x-principal-role",
            "X-PRINCIPAL-SCOPES",
        ],
    )
    def test_attack_08_case_insensitive_header_spoofing_rejected(
        self, client: TestClient, header_case: str
    ) -> None:
        """Attacker attempts casing variations to bypass FastAPI header matching."""
        res = client.post(
            "/api/v1/scout/query",
            headers={header_case: "stealth_attacker"},
            json={"query": "Test casing", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 401
        assert res.json()["detail"] == "Signed Authorization Bearer token is required."

    @pytest.mark.parametrize(
        "invalid_auth",
        [
            "",
            "   ",
            "Bearer",
            "Bearer ",
            "Bearer   ",
            "Basic dXNlcjpwYXNz",
            "Token some_api_token",
            'Digest username="Mufasa"',
            "CustomScheme token123",
        ],
    )
    def test_attack_09_invalid_or_non_bearer_auth_header_rejected(
        self, client: TestClient, invalid_auth: str
    ) -> None:
        """Attacker supplies empty, malformed, or non-Bearer authorization schemes."""
        res = client.post(
            "/api/v1/scout/query",
            headers={"Authorization": invalid_auth},
            json={"query": "Test auth schemes", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 401

    def test_attack_10_direct_dependency_call_without_token_raises_401(self) -> None:
        """Direct programmatic invocation of get_principal_context without token raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            get_principal_context(
                x_principal_user_id="direct_attacker",
                x_principal_role="compliance_officer",
                x_principal_scopes="audit:compliance",
                authorization=None,
            )
        assert exc_info.value.status_code == 401
        assert "Signed Authorization Bearer token is required." in exc_info.value.detail


# ==============================================================================
# Category 2: Header Forging Alongside Valid Bearer Token (Override Attacks)
# ==============================================================================
class TestHeaderForgingAlongsideValidToken:
    """Attacks where a user provides a valid token while injecting malicious headers."""

    def test_attack_11_user_id_header_override_ignored(self) -> None:
        """Attacker provides valid token for user_dev but header X-Principal-User-Id: admin."""
        valid_token = create_signed_principal_token(
            user_id="user_dev_01",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        principal = get_principal_context(
            x_principal_user_id="admin_override_attempt",
            authorization=f"Bearer {valid_token}",
        )
        assert principal.user_id == "user_dev_01"
        assert principal.user_id != "admin_override_attempt"

    def test_attack_12_role_header_override_ignored(self) -> None:
        """Attacker provides valid token for engineer but role: compliance_officer."""
        valid_token = create_signed_principal_token(
            user_id="user_dev_02",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        principal = get_principal_context(
            x_principal_role="compliance_officer",
            authorization=f"Bearer {valid_token}",
        )
        assert principal.role == RoleType.ENGINEER
        assert principal.role != RoleType.COMPLIANCE_OFFICER

    def test_attack_13_scopes_header_override_ignored(self) -> None:
        """Attacker provides valid token for public_eng but scopes: audit:compliance."""
        valid_token = create_signed_principal_token(
            user_id="user_dev_03",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        principal = get_principal_context(
            x_principal_scopes="audit:compliance,quant_restricted",
            authorization=f"Bearer {valid_token}",
        )
        assert principal.allowed_scopes == ["public_eng"]
        assert "audit:compliance" not in principal.allowed_scopes

    def test_attack_14_combined_privilege_escalation_attempt_in_api_request(
        self, client: TestClient
    ) -> None:
        """End-to-end API call: valid token + forged escalation headers."""
        valid_token = create_signed_principal_token(
            user_id="legit_engineer",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        spoofed_headers = {
            "Authorization": f"Bearer {valid_token}",
            "X-Principal-User-Id": "attacker_ceo",
            "X-Principal-Role": "compliance_officer",
            "X-Principal-Scopes": "audit:compliance,quant_restricted",
        }
        res = client.post(
            "/api/v1/scout/query",
            headers=spoofed_headers,
            json={"query": "Test query execution", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 200


# ==============================================================================
# Category 3: Cryptographic HMAC Payload & Signature Tampering
# ==============================================================================
class TestHMACPayloadAndSignatureTampering:
    """Attacks targeting the HMAC-SHA256 signature verification and payload decoding."""

    def test_attack_15_single_bit_flip_in_payload_detected(self) -> None:
        """Attacker modifies 1 bit in base64 payload while keeping signature intact."""
        token = create_signed_principal_token("alice", RoleType.ENGINEER, ["public_eng"])
        payload_b64, sig = token.split(".")
        modified_char = "B" if payload_b64[5] == "A" else "A"
        tampered_payload = payload_b64[:5] + modified_char + payload_b64[6:]
        tampered_token = f"{tampered_payload}.{sig}"

        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(tampered_token)
        assert exc_info.value.status_code == 401
        assert "Invalid HMAC-SHA256 signature" in exc_info.value.detail

    def test_attack_16_single_hex_char_flip_in_signature_detected(self) -> None:
        """Attacker modifies 1 character in the 64-char HMAC signature."""
        token = create_signed_principal_token("alice", RoleType.ENGINEER, ["public_eng"])
        payload_b64, sig = token.split(".")
        flipped_sig = ("0" if sig[0] != "0" else "1") + sig[1:]
        tampered_token = f"{payload_b64}.{flipped_sig}"

        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(tampered_token)
        assert exc_info.value.status_code == 401
        assert "Invalid HMAC-SHA256 signature" in exc_info.value.detail

    def test_attack_17_role_escalation_in_payload_rejected(self) -> None:
        """Attacker decodes payload, escalates role, re-encodes with old sig."""
        token = create_signed_principal_token("bob", RoleType.ENGINEER, ["public_eng"])
        payload_b64, sig = token.split(".")
        pad = "=" * (-len(payload_b64) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload_b64 + pad))
        data["role"] = "compliance_officer"
        tampered_b64 = (
            base64.urlsafe_b64encode(json.dumps(data).encode("utf-8")).decode("ascii").rstrip("=")
        )
        tampered_token = f"{tampered_b64}.{sig}"

        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(tampered_token)
        assert exc_info.value.status_code == 401
        assert "Invalid HMAC-SHA256 signature" in exc_info.value.detail

    def test_attack_18_scope_expansion_in_payload_rejected(self) -> None:
        """Attacker decodes payload, appends restricted scopes, re-encodes with old sig."""
        token = create_signed_principal_token("charlie", RoleType.ENGINEER, ["public_eng"])
        payload_b64, sig = token.split(".")
        pad = "=" * (-len(payload_b64) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload_b64 + pad))
        data["scopes"].append("audit:compliance")
        tampered_b64 = (
            base64.urlsafe_b64encode(json.dumps(data).encode("utf-8")).decode("ascii").rstrip("=")
        )
        tampered_token = f"{tampered_b64}.{sig}"

        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(tampered_token)
        assert exc_info.value.status_code == 401
        assert "Invalid HMAC-SHA256 signature" in exc_info.value.detail

    def test_attack_19_empty_signature_rejected(self) -> None:
        """Attacker provides payload with trailing dot but empty signature."""
        token = create_signed_principal_token("dave", RoleType.ENGINEER, ["public_eng"])
        payload_b64, _ = token.split(".")
        tampered_token = f"{payload_b64}."

        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(tampered_token)
        assert exc_info.value.status_code == 401
        assert "Malformed signed principal token" in exc_info.value.detail

    def test_attack_20_empty_payload_rejected(self) -> None:
        """Attacker provides leading dot with signature but empty payload."""
        token = create_signed_principal_token("dave", RoleType.ENGINEER, ["public_eng"])
        _, sig = token.split(".")
        tampered_token = f".{sig}"

        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(tampered_token)
        assert exc_info.value.status_code == 401
        assert "Malformed signed principal token" in exc_info.value.detail

    def test_attack_21_missing_delimiter_rejected(self) -> None:
        """Attacker strips dot delimiter altogether."""
        token = create_signed_principal_token("dave", RoleType.ENGINEER, ["public_eng"])
        tampered_token = token.replace(".", "")

        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(tampered_token)
        assert exc_info.value.status_code == 401
        assert "Malformed signed principal token" in exc_info.value.detail

    def test_attack_22_multiple_delimiters_rejected(self) -> None:
        """Attacker provides 3 segments (e.g. JWT-style 3-part header.payload.sig)."""
        tampered_token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature"

        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(tampered_token)
        assert exc_info.value.status_code == 401
        assert "Malformed signed principal token" in exc_info.value.detail

    def test_attack_23_signature_truncation_rejected(self) -> None:
        """Attacker truncates signature (e.g. 32 chars instead of 64)."""
        token = create_signed_principal_token("dave", RoleType.ENGINEER, ["public_eng"])
        payload_b64, sig = token.split(".")
        tampered_token = f"{payload_b64}.{sig[:32]}"

        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(tampered_token)
        assert exc_info.value.status_code == 401
        assert "Invalid HMAC-SHA256 signature" in exc_info.value.detail

    def test_attack_24_valid_signature_over_malformed_json_payload_rejected(self) -> None:
        """Attacker signs a payload that is valid base64 but invalid JSON."""
        malformed_bytes = b"This is definitely not JSON!"
        b64_payload = base64.urlsafe_b64encode(malformed_bytes).decode("ascii").rstrip("=")
        secret = _resolve_auth_secret()
        sig = hmac.new(secret, b64_payload.encode("ascii"), hashlib.sha256).hexdigest()
        tampered_token = f"{b64_payload}.{sig}"

        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(tampered_token)
        assert exc_info.value.status_code == 401
        assert "Invalid base64/JSON payload" in exc_info.value.detail

    def test_attack_25_valid_signature_over_non_dict_json_triggers_uncaught_attribute_error(
        self, client: TestClient
    ) -> None:
        """Attacker signs a JSON list or integer instead of an object.

        Finding: In verify_signed_principal_token, data is assumed to be a dict without
        type validation (isinstance(data, dict)), causing an uncaught AttributeError on
        data.get('exp', 0) and triggering an unhandled HTTP 500 Internal Server Error.
        """
        non_dict_bytes = json.dumps(["malicious", "list"]).encode("utf-8")
        b64_payload = base64.urlsafe_b64encode(non_dict_bytes).decode("ascii").rstrip("=")
        secret = _resolve_auth_secret()
        sig = hmac.new(secret, b64_payload.encode("ascii"), hashlib.sha256).hexdigest()
        tampered_token = f"{b64_payload}.{sig}"

        with pytest.raises(AttributeError, match="'list' object has no attribute 'get'"):
            verify_signed_principal_token(tampered_token)

        res = client.post(
            "/api/v1/scout/query",
            headers={"Authorization": f"Bearer {tampered_token}"},
            json={"query": "Test crash", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 500

    def test_attack_25b_empty_scopes_fail_open_escalation_behavior(self) -> None:
        """Security Finding: Tokens with empty scopes list [] fail open to DEFAULT_ROLE_SCOPES.

        When a user is restricted or given an empty scope list,
        verify_signed_principal_token evaluates `if isinstance(scopes_raw, list) and scopes_raw`
        which evaluates to False because bool([]) is False. It then falls back to
        DEFAULT_ROLE_SCOPES[role], granting full default scopes.
        """
        empty_scopes_token = create_signed_principal_token(
            user_id="restricted_user",
            role=RoleType.ENGINEER,
            allowed_scopes=[],
        )
        principal = verify_signed_principal_token(empty_scopes_token)
        assert "repo:trading_strategies" in principal.allowed_scopes
        assert "public_eng" in principal.allowed_scopes

    def test_attack_26_valid_signature_over_invalid_role_rejected(self) -> None:
        """Attacker signs a payload with an unapproved role (e.g. 'superadmin')."""
        payload = {
            "sub": "attacker",
            "role": "superadmin",
            "scopes": ["all"],
            "exp": int(time.time()) + 3600,
        }
        payload_bytes = json.dumps(payload).encode("utf-8")
        b64_payload = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")
        secret = _resolve_auth_secret()
        sig = hmac.new(secret, b64_payload.encode("ascii"), hashlib.sha256).hexdigest()
        token = f"{b64_payload}.{sig}"

        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(token)
        assert exc_info.value.status_code == 401
        assert "Invalid role inside signed principal token" in exc_info.value.detail


# ==============================================================================
# Category 4: Expired and Time-Manipulated Tokens
# ==============================================================================
class TestTokenExpirationAndTimingAttacks:
    """Attacks testing token lifetime verification and expiration boundaries."""

    def test_attack_27_token_expired_by_one_second(self) -> None:
        """Token with ttl_seconds=-1 (expired 1 second ago) must be rejected."""
        expired_token = create_signed_principal_token(
            "time_user",
            RoleType.ENGINEER,
            ["public_eng"],
            ttl_seconds=-1,
        )
        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(expired_token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_attack_28_token_expired_by_one_hour(self) -> None:
        """Token expired 3600 seconds ago must be rejected."""
        expired_token = create_signed_principal_token(
            "time_user",
            RoleType.ENGINEER,
            ["public_eng"],
            ttl_seconds=-3600,
        )
        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(expired_token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_attack_29_token_with_zero_exp(self) -> None:
        """Token with exp=0 (Unix epoch Jan 1 1970)."""
        payload = {
            "sub": "epoch_user",
            "role": "engineer",
            "scopes": ["public_eng"],
            "iat": 0,
            "exp": 0,
        }
        b64_payload = (
            base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8"))
            .decode("ascii")
            .rstrip("=")
        )
        secret = _resolve_auth_secret()
        sig = hmac.new(secret, b64_payload.encode("ascii"), hashlib.sha256).hexdigest()
        token = f"{b64_payload}.{sig}"

        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_attack_30_token_with_missing_exp_field(self) -> None:
        """Token signed without an exp field defaults to exp=0 and is rejected."""
        payload = {
            "sub": "no_exp_user",
            "role": "engineer",
            "scopes": ["public_eng"],
        }
        b64_payload = (
            base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8"))
            .decode("ascii")
            .rstrip("=")
        )
        secret = _resolve_auth_secret()
        sig = hmac.new(secret, b64_payload.encode("ascii"), hashlib.sha256).hexdigest()
        token = f"{b64_payload}.{sig}"

        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_attack_31_expired_token_rejected_in_e2e_api_call(self, client: TestClient) -> None:
        """End-to-end API call with expired token returns 401."""
        expired_token = create_signed_principal_token(
            "api_user",
            RoleType.ENGINEER,
            ["public_eng"],
            ttl_seconds=-10,
        )
        res = client.post(
            "/api/v1/scout/query",
            headers={"Authorization": f"Bearer {expired_token}"},
            json={"query": "Test expiration", "domain_id": "trading_code_scout"},
        )
        assert res.status_code == 401
        assert "expired" in res.json()["detail"].lower()


# ==============================================================================
# Category 5: Alien / Unknown Secret Key Signatures
# ==============================================================================
class TestAlienAndMalformedKeySignatures:
    """Attacks where adversary crafts valid signatures using alien/untrusted keys."""

    def test_attack_32_token_signed_with_random_32byte_alien_key(self) -> None:
        """Attacker signs token with an alien 32-byte secret key."""
        alien_key = "attacker-secret-key-that-server-does-not-know-12345"
        token = create_signed_principal_token(
            "alien_user",
            RoleType.ENGINEER,
            ["public_eng"],
            secret_key=alien_key,
        )
        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(token)
        assert exc_info.value.status_code == 401
        assert "Invalid HMAC-SHA256 signature" in exc_info.value.detail

    def test_attack_33_token_signed_with_random_64byte_alien_key(self) -> None:
        """Attacker signs token with an alien 64-byte secret key."""
        alien_key = "x" * 64
        token = create_signed_principal_token(
            "alien_user",
            RoleType.ENGINEER,
            ["public_eng"],
            secret_key=alien_key,
        )
        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(token)
        assert exc_info.value.status_code == 401
        assert "Invalid HMAC-SHA256 signature" in exc_info.value.detail

    def test_attack_34_token_signed_with_empty_alien_key(self) -> None:
        """Attacker signs token with empty string key."""
        token = create_signed_principal_token(
            "alien_user",
            RoleType.ENGINEER,
            ["public_eng"],
            secret_key="",
        )
        server_key = "server-strong-configured-secret-key-32ch"
        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(token, secret_key=server_key)
        assert exc_info.value.status_code == 401
        assert "Invalid HMAC-SHA256 signature" in exc_info.value.detail

    @pytest.mark.parametrize("insecure_key", list(INSECURE_DEFAULT_SECRETS))
    def test_attack_35_token_signed_with_insecure_defaults_rejected(
        self, insecure_key: str
    ) -> None:
        """Adversary tries default keys against a server using a production key."""
        token = create_signed_principal_token(
            "insecure_user",
            RoleType.ENGINEER,
            ["public_eng"],
            secret_key=insecure_key,
        )
        server_key = "production-cryptographic-secret-key-32-chars-long"
        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(token, secret_key=server_key)
        assert exc_info.value.status_code == 401
        assert "Invalid HMAC-SHA256 signature" in exc_info.value.detail


# ==============================================================================
# Category 6: Production HMAC Secret Security Enforcement
# ==============================================================================
class TestProductionSecretKeyEnforcement:
    """Attacks challenging production safety gates for HMAC secret resolution."""

    @pytest.mark.parametrize("env_var", ["SCOUT_ENV", "ENVIRONMENT", "ENV"])
    @pytest.mark.parametrize("env_val", ["production", "prod", "staging", "PRODUCTION", "PROD"])
    def test_attack_36_production_env_blocks_default_secret(
        self, monkeypatch: pytest.MonkeyPatch, env_var: str, env_val: str
    ) -> None:
        """Any production/staging flag must immediately block the default HMAC secret."""
        monkeypatch.setenv(env_var, env_val)
        monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)
        monkeypatch.delenv("SCOUT_AUTH_HMAC_SECRET", raising=False)

        assert is_production_environment() is True
        msg = "Insecure default or weak HMAC secret detected in production"
        with pytest.raises(RuntimeError, match=msg):
            _resolve_auth_secret()

    @pytest.mark.parametrize("insecure_secret", list(INSECURE_DEFAULT_SECRETS))
    def test_attack_37_production_blocks_all_known_insecure_defaults(
        self, monkeypatch: pytest.MonkeyPatch, insecure_secret: str
    ) -> None:
        """Production environment blocks every secret in INSECURE_DEFAULT_SECRETS."""
        monkeypatch.setenv("SCOUT_ENV", "production")
        monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", insecure_secret)

        msg = "Insecure default or weak HMAC secret detected in production"
        with pytest.raises(RuntimeError, match=msg):
            _resolve_auth_secret()

    @pytest.mark.parametrize("short_secret", ["a", "short", "1234567890", "a" * 31])
    def test_attack_38_production_blocks_secrets_under_32_chars(
        self, monkeypatch: pytest.MonkeyPatch, short_secret: str
    ) -> None:
        """Production environment blocks any secret shorter than 32 characters."""
        monkeypatch.setenv("SCOUT_ENV", "production")
        monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", short_secret)

        with pytest.raises(RuntimeError, match="minimum 32 characters"):
            _resolve_auth_secret()

    def test_attack_39_production_allows_strong_32plus_character_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A strong 32+ character random secret succeeds in production."""
        monkeypatch.setenv("SCOUT_ENV", "production")
        monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)
        strong_secret = "e4f8c9b2a1d07e6f5c4b3a291807f6e5d4c3b2a1"
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", strong_secret)

        resolved = _resolve_auth_secret()
        assert resolved == strong_secret.encode("utf-8")

    def test_attack_40_dev_override_flag_in_production(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SCOUT_ALLOW_INSECURE_DEV_SECRET=true permits default secret even in production."""
        monkeypatch.setenv("SCOUT_ENV", "production")
        monkeypatch.setenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", "true")
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", DEFAULT_AUTH_HMAC_SECRET)

        resolved = _resolve_auth_secret()
        assert resolved == DEFAULT_AUTH_HMAC_SECRET.encode("utf-8")

    def test_attack_41_production_env_raises_runtime_error_when_request_arrives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In production with default secret, any token validation triggers fatal error."""
        monkeypatch.setenv("SCOUT_ENV", "production")
        monkeypatch.delenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", raising=False)
        monkeypatch.delenv("SCOUT_AUTH_HMAC_SECRET", raising=False)

        client = TestClient(create_app(), raise_server_exceptions=True)
        token = "dummy_payload.dummy_signature"
        msg = "Insecure default or weak HMAC secret detected in production"
        with pytest.raises(RuntimeError, match=msg):
            client.post(
                "/api/v1/scout/query",
                headers={"Authorization": f"Bearer {token}"},
                json={"query": "Production test", "domain_id": "trading_code_scout"},
            )


# ==============================================================================
# Category 7: Permission Mirroring, Scope Sanitization, and Injection Resistance
# ==============================================================================
class TestPermissionMirrorAndScopeSanitizationAttacks:
    """Attacks attempting to inject malicious characters into PostgreSQL RLS scopes."""

    @pytest.mark.parametrize(
        "malicious_scope",
        [
            "public_eng; DROP TABLE code_files; --",
            "public_eng' OR '1'='1",
            'public_eng" OR "1"="1',
            "repo:org/proj,admin",
            "repo:org/proj\nadmin",
            "repo:org/proj\tadmin",
            "repo:org/proj admin",
            "   ",
            "",
        ],
    )
    def test_attack_42_scope_sanitizer_rejects_sql_and_delimiter_injection(
        self, malicious_scope: str
    ) -> None:
        """Adversary attempts SQL injection, delimiter injection, or whitespace into scopes."""
        with pytest.raises(ValueError):
            sanitize_scope_identifier(malicious_scope)

    def test_attack_43_principal_context_validates_scopes_against_csv_injection(self) -> None:
        """PrincipalContext validator rejects scopes containing commas, semicolons, or spaces."""
        with pytest.raises(ValueError):
            PrincipalContext(
                user_id="alice",
                role=RoleType.ENGINEER,
                allowed_scopes=["public_eng,admin"],
            )

        with pytest.raises(ValueError):
            PrincipalContext(
                user_id="alice",
                role=RoleType.ENGINEER,
                allowed_scopes=["public_eng; DROP TABLE"],
            )

        with pytest.raises(ValueError):
            PrincipalContext(
                user_id="alice",
                role=RoleType.ENGINEER,
                allowed_scopes=["public_eng admin"],
            )

    def test_attack_44_gitlab_guest_role_grants_zero_code_read_scopes(self) -> None:
        """Security Invariant: GitLab Guest (10) receives NO read access to repo code."""
        scopes = PermissionMirror.map_gitlab_role_to_scopes(
            "internal/secret-repo", GitLabRole.GUEST
        )
        assert scopes == []

    def test_attack_45_gitlab_minimal_access_role_grants_zero_code_read_scopes(self) -> None:
        """Security Invariant: Minimal Access (5) receives NO read access to repo code."""
        scopes = PermissionMirror.map_gitlab_role_to_scopes(
            "internal/secret-repo", GitLabRole.MINIMAL_ACCESS
        )
        assert scopes == []

    def test_attack_46_empty_identity_profile_fails_closed_to_unprivileged_scope(self) -> None:
        """User with no memberships must fail-closed to unprivileged isolation scope."""
        empty_profile = EnterpriseIdentityProfile(user_id="stranger_user")
        principal = PermissionMirror.create_principal_context(empty_profile)
        assert principal.user_id == "stranger_user"
        assert principal.allowed_scopes == ["unprivileged:user:stranger_user"]
        assert "public_eng" not in principal.allowed_scopes
        assert "public_quant" not in principal.allowed_scopes

    def test_attack_47_scope_csv_serialization_safe_for_postgresql_set_local(self) -> None:
        """Verify to_postgres_scope_csv produces clean comma-separated values."""
        principal = PrincipalContext(
            user_id="alice",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", "repo:trading_strategies:read"],
        )
        csv_scopes = principal.to_postgres_scope_csv()
        assert csv_scopes == "public_eng,repo:trading_strategies:read"
        assert ";" not in csv_scopes
        assert "--" not in csv_scopes
        assert "'" not in csv_scopes
