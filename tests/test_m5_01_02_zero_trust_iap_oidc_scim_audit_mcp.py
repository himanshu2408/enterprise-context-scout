"""M5-01 & M5-02 Verification: Cloud IAP/OIDC/SCIM, Audit Chain, BetrVG, GMP & MCP."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

import httpx
import jwt
import psycopg
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.testclient import TestClient

from context_scout.agents.session_compactor import ScoutSessionManager
from context_scout.api.dependencies import (
    DEFAULT_ROLE_SCOPES,
    create_signed_principal_token,
    get_repository,
)
from context_scout.api.main import app
from context_scout.api.routes.webhooks import set_task_queue
from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
from context_scout.db.schema import (
    CodeFileRecord,
    OverlayFileDeltaRecord,
    RefSnapshotRecord,
    RepositoryRecord,
)
from context_scout.db.session import PoolRole, RLSConnectionManager
from context_scout.mcp_servers.code_repo_mcp import create_code_repo_mcp_http_app
from context_scout.mcp_servers.common import MCP_PROTOCOL_VERSION_2026_07_28
from context_scout.queue.task_queue import InMemoryTaskQueue
from context_scout.schemas.contracts import (
    DEFAULT_TENANT_ID,
    ExecutionMode,
    ExecutionTelemetryRecord,
    ImplementationContextPack,
    ModificationIntent,
    PreFlightValidationReport,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)
from context_scout.security.audit_chain import (
    AuditEventChainService,
    NDJSONSIEMExporter,
    WebhookSIEMExporter,
    format_gcp_cloud_logging_entry,
    get_audit_chain_service,
    set_audit_chain_service,
)
from context_scout.security.betrvg_privacy import BetrVGPrivacyGuard
from context_scout.security.oidc_iap_auth import (
    CLOUD_IAP_ISSUER,
    GOOGLE_WORKSPACE_ISSUER,
    AuthVerificationError,
    CloudIAPVerifier,
    JWKSCache,
    OIDCJWKSVerifier,
    get_global_jwks_cache,
)
from context_scout.security.permission_mirror import (
    CachedPermissionMirrorService,
    EnterpriseIdentityProfile,
)
from context_scout.security.rbac import FORBIDDEN_TOOL_ARGUMENTS
from context_scout.security.scim import (
    SCIMDirectoryService,
    set_scim_directory_service,
)
from context_scout.telemetry.otel import (
    ATTR_SCOUT_PRINCIPAL_ID,
    ScoutTelemetryTracer,
)
from context_scout.telemetry.prometheus_metrics import (
    get_prometheus_registry,
    reset_prometheus_registry,
)

_EC_KEY_GEN_COUNT = 0


def _generate_ec_p256_keypair(kid: str) -> tuple[ec.EllipticCurvePrivateKey, dict[str, Any]]:
    priv = ec.generate_private_key(ec.SECP256R1())
    pub_jwk: dict[str, Any] = jwt.algorithms.ECAlgorithm.to_jwk(priv.public_key(), as_dict=True)
    pub_jwk["kid"] = kid
    pub_jwk["use"] = "sig"
    pub_jwk["alg"] = "ES256"
    return priv, pub_jwk


def _generate_rsa_2048_keypair(kid: str) -> tuple[rsa.RSAPrivateKey, dict[str, Any]]:
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub_jwk: dict[str, Any] = jwt.algorithms.RSAAlgorithm.to_jwk(priv.public_key(), as_dict=True)
    pub_jwk["kid"] = kid
    pub_jwk["use"] = "sig"
    pub_jwk["alg"] = "RS256"
    return priv, pub_jwk


def _cleanup_pg_tenants(tenant_ids: list[uuid.UUID], *, domain_ids: Sequence[str] = ()) -> None:
    from tests.fixtures.pg_isolation import cleanup_pg_tenants_and_eval_domains

    if not tenant_ids and not domain_ids:
        return
    settings = ScoutSettings()
    eval_domains = [d for d in domain_ids if d.startswith("eval_")]
    other_domains = [d for d in domain_ids if not d.startswith("eval_")]
    with psycopg.connect(settings.db_admin_url) as conn:
        cleanup_pg_tenants_and_eval_domains(
            conn,
            tenant_ids=tuple(tenant_ids),
            domain_ids=tuple(eval_domains),
        )
        for dom_id in other_domains:
            conn.execute(
                "DELETE FROM domain_packs WHERE domain_id = %s AND tenant_id != %s;",
                (dom_id, DEFAULT_TENANT_ID),
            )
        if other_domains:
            conn.commit()


@pytest.fixture(autouse=True)
def _reset_m5_singletons() -> Generator[None, None, None]:
    set_scim_directory_service(None)
    set_audit_chain_service(None)
    set_task_queue(None)
    reset_prometheus_registry()
    cache = get_global_jwks_cache()
    cache.clear()
    orig_loader = cache._jwks_loader
    orig_client = cache._http_client
    yield
    set_scim_directory_service(None)
    set_audit_chain_service(None)
    set_task_queue(None)
    reset_prometheus_registry()
    cache.clear()
    cache._jwks_loader = orig_loader
    cache._http_client = orig_client


def _mcp_headers(
    bearer_token: str,
    *,
    method: str,
    tool_name: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "MCP-Protocol-Version": MCP_PROTOCOL_VERSION_2026_07_28,
        "Mcp-Method": method,
    }
    if tool_name:
        headers["Mcp-Name"] = tool_name
    if extra:
        headers.update(extra)
    return headers


def _mcp_payload(req_id: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    merged_params = dict(params or {})
    merged_params["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": MCP_PROTOCOL_VERSION_2026_07_28,
        "io.modelcontextprotocol/clientCapabilities": {},
    }
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": merged_params,
    }


def _extract_mcp_result(resp: httpx.Response) -> dict[str, Any]:
    ct = resp.headers.get("content-type", "")
    if "application/json" in ct:
        body: dict[str, Any] = resp.json()
        return body
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            parsed: dict[str, Any] = json.loads(line[6:])
            return parsed
    raise AssertionError(f"Unable to decode MCP response ({resp.status_code}): {resp.text!r}")


# =============================================================================
# Pillar 1: Google Cloud IAP ES256 JWKS + GET /api/v1/auth/me
# =============================================================================


def test_cloud_iap_es256_jwks_and_auth_me(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies Cloud IAP ES256 verification and strict negative controls."""
    priv_ec, jwk_ec = _generate_ec_p256_keypair("iap-ec-k1")
    priv_rsa, _ = _generate_rsa_2048_keypair("iap-rsa-k1")
    aud = "/projects/123456789/global/backendServices/scout-api"
    jwks_url = "https://www.gstatic.com/iap/verify/public_key-jwk"
    now = int(time.time())

    cache = get_global_jwks_cache()
    monkeypatch.setattr(cache, "_jwks_loader", lambda _url: {"keys": [jwk_ec]})

    monkeypatch.setenv("SCOUT_IAP_EXPECTED_AUDIENCE", aud)
    monkeypatch.setenv("SCOUT_IAP_ISSUER", CLOUD_IAP_ISSUER)
    monkeypatch.setenv("SCOUT_IAP_JWKS_URL", jwks_url)
    monkeypatch.setenv("SCOUT_IAP_ALLOWED_HOSTED_DOMAINS", "acme.example.com")

    valid_claims = {
        "iss": CLOUD_IAP_ISSUER,
        "aud": aud,
        "sub": "accounts.google.com:10987654321",
        "email": "alice@acme.example.com",
        "hd": "acme.example.com",
        "role": "engineer",
        "groups": ["platform-eng"],
        "iat": now - 5,
        "exp": now + 300,
    }
    valid_jwt = jwt.encode(valid_claims, priv_ec, algorithm="ES256", headers={"kid": "iap-ec-k1"})

    client = TestClient(app)
    res = client.get("/api/v1/auth/me", headers={"X-Goog-IAP-JWT-Assertion": valid_jwt})
    assert res.status_code == 200
    body = res.json()
    assert body["user_id"] == "alice@acme.example.com"
    assert body["role"] == "engineer"
    assert "group:platform-eng" in body["allowed_scopes"]

    # Negative controls: alg=none, HS256 confusion, RS256, wrong aud, wrong iss, expired, bad hd
    for bad_token in (
        f"eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.{valid_jwt.split('.')[1]}.",
        jwt.encode(valid_claims, "shared-secret-key-0123456789abcdef", algorithm="HS256"),
        jwt.encode(valid_claims, priv_rsa, algorithm="RS256", headers={"kid": "iap-ec-k1"}),
        jwt.encode(
            {**valid_claims, "aud": "/projects/999/global/backendServices/other"},
            priv_ec,
            algorithm="ES256",
            headers={"kid": "iap-ec-k1"},
        ),
        jwt.encode(
            {**valid_claims, "iss": "https://evil.example.com"},
            priv_ec,
            algorithm="ES256",
            headers={"kid": "iap-ec-k1"},
        ),
        jwt.encode(
            {**valid_claims, "iat": now - 600, "exp": now - 120},
            priv_ec,
            algorithm="ES256",
            headers={"kid": "iap-ec-k1"},
        ),
    ):
        err_res = client.get("/api/v1/auth/me", headers={"X-Goog-IAP-JWT-Assertion": bad_token})
        assert err_res.status_code == 401

    wrong_hd_jwt = jwt.encode(
        {**valid_claims, "email": "bob@untrusted.org", "hd": "untrusted.org"},
        priv_ec,
        algorithm="ES256",
        headers={"kid": "iap-ec-k1"},
    )
    hd_res = client.get("/api/v1/auth/me", headers={"X-Goog-IAP-JWT-Assertion": wrong_hd_jwt})
    assert hd_res.status_code == 403

    monkeypatch.delenv("SCOUT_IAP_EXPECTED_AUDIENCE", raising=False)
    unset_aud_verifier = CloudIAPVerifier(expected_audience="", jwks_cache=cache)
    with pytest.raises(AuthVerificationError) as exc_info:
        unset_aud_verifier.verify(valid_jwt)
    assert exc_info.value.status_code == 503


# =============================================================================
# Pillar 2: Multi-Provider OIDC JWKS, Kid Rotation & /auth/session Dev Gate
# =============================================================================


def test_multi_provider_oidc_jwks_kid_rotation_and_dev_session_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies Workspace, Keycloak, Okta, Entra group overage, kid rotation & session gate."""
    rsa_k1, jwk_rsa_k1 = _generate_rsa_2048_keypair("oidc-rsa-1")
    ec_k2, jwk_ec_k2 = _generate_ec_p256_keypair("oidc-ec-2")
    now = int(time.time())
    clock_now = [100.0]
    jwks_state: dict[str, Any] = {"keys": [jwk_rsa_k1]}

    cache = JWKSCache(
        cache_ttl_seconds=300.0,
        min_refresh_interval_seconds=10.0,
        jwks_loader=lambda _url: dict(jwks_state),
        clock=lambda: clock_now[0],
    )
    scim = SCIMDirectoryService()
    tenant_id = uuid.uuid4()
    scim.upsert_user(
        tenant_id,
        {
            "id": "entra-user-1",
            "userName": "carol@acme.example.com",
            "emails": [{"value": "carol@acme.example.com"}],
            "active": True,
        },
    )
    scim.upsert_group(
        tenant_id,
        {
            "id": "entra-grp-overage",
            "displayName": "Core Architects",
            "members": [{"value": "entra-user-1", "type": "User"}],
            "mappedScopes": ["domain:code_architecture_scout", "public_eng"],
            "mappedRole": "engineer",
        },
    )

    # 1. Google Workspace RS256
    gws_verifier = OIDCJWKSVerifier(
        provider="google_workspace",
        issuer=GOOGLE_WORKSPACE_ISSUER,
        audience="scout-client-id",
        allowed_hosted_domains=["acme.example.com"],
        jwks_cache=cache,
        scim_service=scim,
    )
    gws_jwt = jwt.encode(
        {
            "iss": GOOGLE_WORKSPACE_ISSUER,
            "aud": "scout-client-id",
            "sub": "gws-u1",
            "email": "dev@acme.example.com",
            "email_verified": True,
            "hd": "acme.example.com",
            "role": "engineer",
            "iat": now - 5,
            "exp": now + 300,
        },
        rsa_k1,
        algorithm="RS256",
        headers={"kid": "oidc-rsa-1"},
    )
    gws_p = gws_verifier.verify(gws_jwt, tenant_id=tenant_id)
    assert gws_p.role == RoleType.ENGINEER

    # 2. Keycloak realm_access.roles
    kc_verifier = OIDCJWKSVerifier(
        provider="keycloak",
        issuer="https://idp.example.com/realms/corp",
        audience="scout-kc",
        jwks_cache=cache,
        scim_service=scim,
    )
    kc_jwt = jwt.encode(
        {
            "iss": "https://idp.example.com/realms/corp",
            "aud": "scout-kc",
            "sub": "kc-u1",
            "preferred_username": "auditor",
            "realm_access": {"roles": ["compliance_officer"]},
            "groups": ["/sec/audit"],
            "iat": now - 5,
            "exp": now + 300,
        },
        rsa_k1,
        algorithm="RS256",
        headers={"kid": "oidc-rsa-1"},
    )
    kc_p = kc_verifier.verify(kc_jwt, tenant_id=tenant_id)
    assert kc_p.role == RoleType.COMPLIANCE_OFFICER
    assert "group:sec/audit" in kc_p.allowed_scopes

    # 3. Okta groups
    okta_verifier = OIDCJWKSVerifier(
        provider="okta",
        issuer="https://corp.okta.com/oauth2/default",
        audience="api://scout",
        jwks_cache=cache,
        group_scope_map={"platform": ["domain:code_architecture_scout"]},
        scim_service=scim,
    )
    okta_jwt = jwt.encode(
        {
            "iss": "https://corp.okta.com/oauth2/default",
            "aud": "api://scout",
            "sub": "okta-u1",
            "uid": "dana@acme.example.com",
            "groups": ["platform"],
            "scp": "public_eng",
            "iat": now - 5,
            "exp": now + 300,
        },
        rsa_k1,
        algorithm="RS256",
        headers={"kid": "oidc-rsa-1"},
    )
    okta_p = okta_verifier.verify(okta_jwt, tenant_id=tenant_id)
    assert "group:platform" in okta_p.allowed_scopes
    assert "domain:code_architecture_scout" in okta_p.allowed_scopes

    # 4. Entra ID group overage (_claim_names) resolved via SCIM + kid rotation cooldown test
    entra_tid = "72f988bf-86f1-41af-91ab-2d7cd011db47"
    entra_verifier = OIDCJWKSVerifier(
        provider="entra_id",
        issuer=f"https://login.microsoftonline.com/{entra_tid}/v2.0",
        audience="scout-entra-app",
        allowed_entra_tenants=[entra_tid],
        jwks_cache=cache,
        scim_service=scim,
    )
    entra_claims = {
        "iss": f"https://login.microsoftonline.com/{entra_tid}/v2.0",
        "aud": "scout-entra-app",
        "sub": "entra-sub-carol",
        "upn": "carol@acme.example.com",
        "tid": entra_tid,
        "_claim_names": {"groups": "src1"},
        "iat": now - 5,
        "exp": now + 300,
    }
    # Populate Entra JWKS cache with k1 at clock_now == 100.0 before key rotation
    initial_entra_jwt = jwt.encode(
        entra_claims, rsa_k1, algorithm="RS256", headers={"kid": "oidc-rsa-1"}
    )
    assert entra_verifier.verify(initial_entra_jwt, tenant_id=tenant_id).role == RoleType.ENGINEER

    # Publish rotated EC key oidc-ec-2 in JWKS
    jwks_state["keys"] = [jwk_rsa_k1, jwk_ec_k2]
    rotated_jwt = jwt.encode(entra_claims, ec_k2, algorithm="ES256", headers={"kid": "oidc-ec-2"})

    # Within min_refresh cooldown (clock_now == 102.0): unknown kid refresh is suppressed -> 401
    clock_now[0] = 102.0
    with pytest.raises(AuthVerificationError) as cooldown_err:
        entra_verifier.verify(rotated_jwt, tenant_id=tenant_id)
    assert cooldown_err.value.status_code == 401

    # Advance monotonic clock past min_refresh -> refresh succeeds and resolves SCIM overage groups
    clock_now[0] = 115.0
    entra_p = entra_verifier.verify(rotated_jwt, tenant_id=tenant_id)
    assert entra_p.role == RoleType.ENGINEER
    assert "domain:code_architecture_scout" in entra_p.allowed_scopes

    # HS256 algorithm confusion rejected even with valid-looking payload
    hs256_confused = jwt.encode(
        entra_claims, "public-pem-confusion-secret-012345", algorithm="HS256"
    )
    with pytest.raises(AuthVerificationError) as alg_err:
        entra_verifier.verify(hs256_confused, tenant_id=tenant_id)
    assert alg_err.value.status_code == 401

    # /api/v1/auth/session blocked in staging/qa/production, allowed in development
    client = TestClient(app)
    session_req = {
        "user_id": "dev_tester",
        "role": "engineer",
        "tenant_id": str(tenant_id),
    }
    for env_name in ("staging", "qa", "production"):
        monkeypatch.setenv("SCOUT_ENVIRONMENT", env_name)
        resp_blocked = client.post("/api/v1/auth/session", json=session_req)
        assert resp_blocked.status_code == 403

    monkeypatch.setenv("SCOUT_ENVIRONMENT", "development")
    resp_dev = client.post("/api/v1/auth/session", json=session_req)
    assert resp_dev.status_code == 200
    assert resp_dev.json()["principal"]["tenant_id"] == str(tenant_id)


# =============================================================================
# Pillar 3: SCIM 2.0 Routes + PG17 FORCE RLS + Cycle-Safe Nested Groups + Eviction
# =============================================================================


@pytest.mark.postgres
def test_scim_routes_pg17_force_rls_nested_groups_and_deprovisioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies fail-closed SCIM, RFC 7644 PatchOp, cycle safety, cache eviction, & PG17 RLS."""
    settings = ScoutSettings()
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    scim_token = "scim-provisioner-secret-token-xyz"
    scim_svc = SCIMDirectoryService(connection_manager=RLSConnectionManager(settings.db_app_url))
    set_scim_directory_service(scim_svc)
    perm_cache = CachedPermissionMirrorService()
    scim_svc._permission_cache = perm_cache

    client = TestClient(app)
    try:
        # 1. Fail-closed 503 when unset, 401 when wrong token
        monkeypatch.delenv("SCOUT_SCIM_BEARER_TOKEN", raising=False)
        res_503 = client.get("/api/v1/scim/v2/Users")
        assert res_503.status_code == 503

        monkeypatch.setenv("SCOUT_SCIM_BEARER_TOKEN", scim_token)
        res_401 = client.get(
            "/api/v1/scim/v2/Users",
            headers={"Authorization": "Bearer wrong-scim-token"},
        )
        assert res_401.status_code == 401

        hdr_a = {
            "Authorization": f"Bearer {scim_token}",
            "X-Scout-Tenant-Id": str(tenant_a),
        }
        hdr_b = {
            "Authorization": f"Bearer {scim_token}",
            "X-Scout-Tenant-Id": str(tenant_b),
        }

        # 2. User + Nested Cyclic Groups under PG17 FORCE RLS
        u_res = client.post(
            "/api/v1/scim/v2/Users",
            headers=hdr_a,
            json={
                "id": "scim-u-alice",
                "userName": "alice@tenant-a.local",
                "emails": [{"value": "alice@tenant-a.local", "primary": True}],
                "active": True,
            },
        )
        assert u_res.status_code == 201
        alice_prof = EnterpriseIdentityProfile(
            user_id="scim-u-alice",
            custom_scopes=["repo:secret"],
        )
        assert perm_cache.get_user_scopes(alice_prof) == ["repo:secret"]
        assert "scim-u-alice" in perm_cache._cache

        g1_res = client.post(
            "/api/v1/scim/v2/Groups",
            headers=hdr_a,
            json={
                "id": "g-leaf",
                "displayName": "Leaf Team",
                "members": [{"value": "scim-u-alice", "type": "User"}],
                "mappedScopes": ["scope:leaf"],
            },
        )
        assert g1_res.status_code == 201
        g2_res = client.post(
            "/api/v1/scim/v2/Groups",
            headers=hdr_a,
            json={
                "id": "g-parent",
                "displayName": "Parent Team",
                "members": [{"value": "g-leaf", "type": "Group"}],
                "mappedScopes": ["scope:parent"],
                "mappedRole": "engineer",
            },
        )
        assert g2_res.status_code == 201
        # Introduce group cycle g-leaf -> g-parent -> g-leaf via PatchOp
        patch_cycle = client.patch(
            "/api/v1/scim/v2/Groups/g-leaf",
            headers=hdr_a,
            json={
                "Operations": [
                    {
                        "op": "add",
                        "path": "members",
                        "value": [{"value": "g-parent", "type": "Group"}],
                    }
                ]
            },
        )
        assert patch_cycle.status_code == 200

        # Resolve cycle-safe nested enrichment
        resolved = scim_svc.resolve_principal_enrichment(
            tenant_id=tenant_a,
            identifiers=["alice@tenant-a.local"],
        )
        assert resolved.role == RoleType.ENGINEER
        assert "scope:leaf" in resolved.scopes
        assert "scope:parent" in resolved.scopes

        # Cross-tenant PG17 FORCE RLS check: tenant_b cannot see tenant_a users or groups
        scim_users_base = "/api/v1/scim/v2/Users"
        alice_uid = "scim-u-alice"
        list_b = client.get(scim_users_base, headers=hdr_b)
        assert list_b.status_code == 200
        assert list_b.json()["totalResults"] == 0
        assert client.get(f"{scim_users_base}/{alice_uid}", headers=hdr_b).status_code == 404

        # PatchOp remove member using SCIM filter expression members[value eq "..."]
        rm_member = client.patch(
            "/api/v1/scim/v2/Groups/g-leaf",
            headers=hdr_a,
            json={
                "Operations": [
                    {"op": "remove", "path": 'members[value eq "g-parent"]'},
                ]
            },
        )
        assert rm_member.status_code == 200

        # Immediate deprovisioning via active=false evicts permission cache & blocks OIDC auth
        deprov_res = client.patch(
            f"{scim_users_base}/{alice_uid}",
            headers=hdr_a,
            json={"Operations": [{"op": "replace", "path": "active", "value": False}]},
        )
        assert deprov_res.status_code == 200
        assert deprov_res.json()["active"] is False

        # Permission cache entry invalidated immediately upon deprovisioning
        assert "scim-u-alice" not in perm_cache._cache

        priv_ec, jwk_ec = _generate_ec_p256_keypair("scim-oidc-k1")
        jwks_cache = JWKSCache(jwks_loader=lambda _u: {"keys": [jwk_ec]})
        oidc_verifier = OIDCJWKSVerifier(
            provider="generic",
            issuer="https://sso.tenant-a.local",
            audience="scout-api",
            jwks_cache=jwks_cache,
            scim_service=scim_svc,
        )
        now = int(time.time())
        alice_oidc_jwt = jwt.encode(
            {
                "iss": "https://sso.tenant-a.local",
                "aud": "scout-api",
                "sub": "scim-u-alice",
                "email": "alice@tenant-a.local",
                "iat": now - 5,
                "exp": now + 300,
            },
            priv_ec,
            algorithm="ES256",
            headers={"kid": "scim-oidc-k1"},
        )
        with pytest.raises(AuthVerificationError) as rev_err:
            oidc_verifier.verify(alice_oidc_jwt, tenant_id=tenant_a)
        assert rev_err.value.status_code == 403
    finally:
        _cleanup_pg_tenants([tenant_a, tenant_b])


# =============================================================================
# Pillar 4: Hash-Chained audit_events + PG17 WORM + GCP Cloud Logging + SIEM
# =============================================================================


@pytest.mark.postgres
def test_audit_chain_pg17_worm_tamper_gcp_logging_and_siem_webhook(
    tmp_path: Path,
) -> None:
    """Verifies SHA-256 hash chain, PG17 WORM, tamper detection, Cloud Logging, & SIEM."""
    settings = ScoutSettings()
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    captured_webhooks: list[tuple[bytes, dict[str, str]]] = []
    siem_secret = "siem-webhook-hmac-secret-at-least-32-chars"

    def _siem_handler(request: httpx.Request) -> httpx.Response:
        captured_webhooks.append((bytes(request.content), dict(request.headers)))
        return httpx.Response(202, json={"accepted": True})

    ndjson_file = tmp_path / "audit.ndjson"
    ndjson_exp = NDJSONSIEMExporter(file_path=ndjson_file, gcp_project="scout-prod-eu")
    webhook_exp = WebhookSIEMExporter(
        "https://siem.example.local/ingest",
        webhook_secret=siem_secret,
        gcp_project="scout-prod-eu",
        http_client=httpx.Client(transport=httpx.MockTransport(_siem_handler)),
    )
    app_mgr = RLSConnectionManager(settings.db_app_url, pool_role=PoolRole.QUERY)
    audit_svc = AuditEventChainService(
        connection_manager=app_mgr,
        ndjson_exporter=ndjson_exp,
        webhook_exporter=webhook_exp,
        gcp_project="scout-prod-eu",
    )
    set_audit_chain_service(audit_svc)

    comp_a = PrincipalContext(
        user_id="compliance_a",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["audit:compliance"],
        tenant_id=tenant_a,
    )
    token_a = create_signed_principal_token(
        user_id="compliance_a",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["audit:compliance"],
        tenant_id=tenant_a,
    )
    token_b = create_signed_principal_token(
        user_id="compliance_b",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["audit:compliance"],
        tenant_id=tenant_b,
    )
    client = TestClient(app)
    try:
        ev1 = audit_svc.record_event(
            principal=comp_a,
            domain_id="code_architecture_scout",
            event_type="auth.login",
            severity="INFO",
            resource_uri="/api/v1/auth/me",
            outcome="allow",
            trace_id="1234567890abcdef1234567890abcdef",
            span_id="1122334455667788",
            payload={"method": "iap"},
        )
        ev2 = audit_svc.record_event(
            principal=comp_a,
            domain_id="code_architecture_scout",
            event_type="scout.query",
            severity="NOTICE",
            resource_uri="/api/v1/scout/query",
            outcome="allow",
            trace_id="1234567890abcdef1234567890abcdef",
            span_id="2233445566778899",
            payload={"slices": 4},
        )
        ev3 = audit_svc.record_event(
            principal=comp_a,
            domain_id="code_architecture_scout",
            event_type="admin.rbac.deny",
            severity="WARNING",
            resource_uri="/api/v1/admin/purge",
            outcome="deny",
            payload={"reason": "test"},
        )
        assert ev2.prev_hash_sha256 == ev1.event_hash_sha256
        assert ev3.prev_hash_sha256 == ev2.event_hash_sha256

        # Route verification & cross-tenant RLS isolation
        ver_a = client.get(
            "/api/v1/admin/audit/verify",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert ver_a.status_code == 200
        assert ver_a.json()["valid"] is True
        assert ver_a.json()["checked_count"] == 3

        ver_b = client.get(
            "/api/v1/admin/audit/verify",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert ver_b.status_code == 200
        assert ver_b.json()["checked_count"] == 0

        # GCP Cloud Logging structured JSON export
        exp_a = client.get(
            "/api/v1/admin/audit/export?format=gcp_logging",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert exp_a.status_code == 200
        exported_entries = [json.loads(ln) for ln in exp_a.text.strip().splitlines()]
        assert len(exported_entries) == 3
        first_entry = exported_entries[0]
        assert first_entry["severity"] == "INFO"
        assert (
            first_entry["logging.googleapis.com/trace"]
            == "projects/scout-prod-eu/traces/1234567890abcdef1234567890abcdef"
        )
        assert first_entry["logging.googleapis.com/spanId"] == "1122334455667788"

        # SIEM signed webhook HMAC verification + NDJSON file output
        assert len(captured_webhooks) == 3
        raw_body, hdrs = captured_webhooks[0]
        expected_sig = (
            "sha256=" + hmac.new(siem_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        )
        sig_header = {k.lower(): v for k, v in hdrs.items()}.get("x-scout-siem-signature-256")
        assert sig_header == expected_sig
        assert len(ndjson_file.read_text(encoding="utf-8").strip().splitlines()) == 3

        # Kernel WORM proof: scout_app_user cannot UPDATE or DELETE audit_events
        with (
            app_mgr.scoped_transaction(comp_a, enable_hnsw_iterative_scan=False) as conn,
            pytest.raises(psycopg.errors.InsufficientPrivilege),
        ):
            conn.execute(
                "UPDATE audit_events SET outcome = 'tampered' WHERE event_id = %s;",
                (ev1.event_id,),
            )
        with (
            app_mgr.scoped_transaction(comp_a, enable_hnsw_iterative_scan=False) as conn,
            pytest.raises(psycopg.errors.InsufficientPrivilege),
        ):
            conn.execute(
                "DELETE FROM audit_events WHERE event_id = %s;",
                (ev1.event_id,),
            )

        # Admin-simulated tampering detected deterministically by SHA-256 chain verifier
        with psycopg.connect(settings.db_admin_url, autocommit=True) as admin_conn:
            admin_conn.execute(
                "UPDATE audit_events SET payload_json = %s::jsonb WHERE event_id = %s;",
                (json.dumps({"slices": 999}), ev2.event_id),
            )
        tamper_report = audit_svc.verify_chain_integrity(comp_a)
        assert tamper_report.valid is False
        assert tamper_report.first_corrupted_seq_no == ev2.seq_no
    finally:
        _cleanup_pg_tenants([tenant_a, tenant_b])

    # Sanity check helper directly
    gcp_entry = format_gcp_cloud_logging_entry(ev1, gcp_project="scout-prod-eu")
    assert gcp_entry["audit"]["event_hash_sha256"] == ev1.event_hash_sha256


# =============================================================================
# Pillar 5: EU Works-Council (BetrVG) Pseudonymization & Telemetry Redaction
# =============================================================================


def test_betrvg_privacy_pseudonymization_and_fail_closed_salt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies BetrVG fail-closed salt, pseudonym consistency, and OTel prompt redaction."""
    monkeypatch.setenv("SCOUT_BETRVG_PRIVACY_MODE", "true")
    monkeypatch.delenv("SCOUT_BETRVG_HMAC_SALT", raising=False)

    with pytest.raises(RuntimeError, match="SCOUT_BETRVG_HMAC_SALT"):
        BetrVGPrivacyGuard().pseudonymize_user_id("worker_alice", DEFAULT_TENANT_ID)

    monkeypatch.setenv("SCOUT_BETRVG_HMAC_SALT", "too-short")
    with pytest.raises(RuntimeError, match="SCOUT_BETRVG_HMAC_SALT"):
        BetrVGPrivacyGuard().pseudonymize_user_id("worker_alice", DEFAULT_TENANT_ID)

    valid_salt = "betrvg-works-council-production-salt-2026-012345"
    monkeypatch.setenv("SCOUT_BETRVG_HMAC_SALT", valid_salt)
    tenant_id = uuid.uuid4()
    raw_user = "worker.klaus@example.de"

    guard = BetrVGPrivacyGuard()
    pseudo = guard.pseudonymize_user_id(raw_user, tenant_id)
    assert pseudo.startswith("betrvg_v1_")
    assert raw_user not in pseudo
    # Deterministic per tenant, different across tenants
    assert guard.pseudonymize_user_id(raw_user, tenant_id) == pseudo
    assert guard.pseudonymize_user_id(raw_user, uuid.uuid4()) != pseudo

    # /api/v1/auth/me exposes pseudonymized actor when BetrVG enabled
    token = create_signed_principal_token(
        user_id=raw_user,
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
        tenant_id=tenant_id,
    )
    res = TestClient(app).get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json()["user_id"] == pseudo

    # Audit event stores pseudonymized actor
    ev = get_audit_chain_service().record_event(
        tenant_id=tenant_id,
        event_type="auth.check",
        actor_id=raw_user,
        actor_role="engineer",
        outcome="allow",
    )
    assert ev.actor_id == pseudo
    assert raw_user not in json.dumps(ev.model_dump(mode="json"))

    # ScoutSessionManager stores pseudonymized principal_id
    mem_repo = InMemoryRLSRepository()
    raw_principal = PrincipalContext(
        user_id=raw_user,
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
        tenant_id=tenant_id,
    )
    sample_pack = ImplementationContextPack(
        trace_id="betrvg-trace-1",
        domain_id="code_architecture_scout",
        user_task="Inspect router authentication",
        architectural_execution_plan=["Inspect auth middleware handler"],
        code_slices=[
            VerbatimCodeSlice(
                file_path="src/auth.py",
                repo_name="core",
                qualified_symbol="core.auth.check",
                selection_reason="Auth middleware target",
                modification_intent=ModificationIntent.MODIFY_TARGET,
                git_commit_sha="a" * 40,
                git_blob_sha="b" * 40,
                start_line=1,
                end_line=2,
                verbatim_code_with_lines="   1 | x = 1\n",
            )
        ],
        preflight_report=PreFlightValidationReport(
            passed=True,
            normalized_rrf_score=0.9,
            citations_verified_100_pct=True,
        ),
        local_tokens_consumed=5000,
        pack_token_count=400,
    )
    saved_session = ScoutSessionManager().record_completed_turn(
        mem_repo,
        raw_principal,
        session_id="sess-betrvg-1",
        domain_id="code_architecture_scout",
        user_prompt="Find sensitive employee salary lookup",
        result=sample_pack,
    )
    assert saved_session is not None
    assert saved_session.principal_id == pseudo

    # OTel StructuredJsonSpanExporter redacts prompt snippets & pseudonymizes actor IDs
    tracer = ScoutTelemetryTracer.reset_instance()
    secret_query = "proprietary payroll query string"
    with tracer.trace_retrieval(domain_id="code_architecture_scout", query=secret_query) as r_ctx:
        r_ctx.set_attribute("scout.tenant_id", str(tenant_id))
        r_ctx.set_attribute(ATTR_SCOUT_PRINCIPAL_ID, raw_user)
        r_ctx.record_retrieval(doc_chunks_count=1, ast_symbols_count=1, query=secret_query)
    json_logs = tracer.json_exporter.get_json_logs()
    assert json_logs
    attrs = json_logs[-1]["attributes"]
    assert attrs["scout.retrieval.query"] == "[REDACTED_BETRVG]"
    assert attrs["scout.retrieval.query_snippet"] == "[REDACTED_BETRVG]"
    assert attrs[ATTR_SCOUT_PRINCIPAL_ID] == pseudo
    serialized = json.dumps(json_logs)
    assert secret_query not in serialized
    assert raw_user not in serialized


# =============================================================================
# Pillar 6: Prometheus / GMP Retrieval-Drift Metrics (/api/v1/telemetry/metrics)
# =============================================================================


def test_prometheus_metrics_auth_drift_alert_and_low_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies Prometheus/GMP exposition auth, EWMA drift alert, and zero high-card labels."""
    client = TestClient(app)
    tenant_id = uuid.uuid4()

    # 1. Auth gates
    assert client.get("/api/v1/telemetry/metrics").status_code == 401

    intern_tok = create_signed_principal_token(
        user_id="pm_restricted_u",
        role=RoleType.PRODUCT_MANAGER,
        allowed_scopes=["public"],
        tenant_id=tenant_id,
    )
    assert (
        client.get(
            "/api/v1/telemetry/metrics",
            headers={"Authorization": f"Bearer {intern_tok}"},
        ).status_code
        == 403
    )

    # 2. Record baseline and drifted observations -> triggers drift alert gauge == 1
    reg = get_prometheus_registry()
    reg.set_baseline_confidence(tenant_id, "code_architecture_scout", 0.90)
    for _ in range(5):
        reg.record_query_outcome(
            tenant_id=tenant_id,
            domain_id="code_architecture_scout",
            outcome="abstain",
            rrf_score=0.20,
            reranker_score=0.15,
            zero_hit=True,
            reason_code="below_floor",
        )

    scrape_secret = "gmp-scrape-bearer-token-2026"
    monkeypatch.setenv("SCOUT_METRICS_BEARER_TOKEN", scrape_secret)
    scrape_res = client.get(
        "/api/v1/telemetry/metrics",
        headers={"Authorization": f"Bearer {scrape_secret}"},
    )
    assert scrape_res.status_code == 200
    assert "text/plain" in scrape_res.headers["content-type"]
    text = scrape_res.text

    expected_alert_line = (
        f'scout_retrieval_drift_alert{{domain_id="code_architecture_scout",'
        f'tenant_id="{tenant_id}"}} 1'
    )
    assert expected_alert_line in text
    assert "scout_queries_total{" in text
    assert "scout_abstention_total{" in text

    # Ensure zero high-cardinality labels
    for forbidden_label in ("user_id=", "principal_id=", "trace_id=", "query=", "email="):
        assert forbidden_label not in text

    eng_tok = create_signed_principal_token(
        user_id="eng_u",
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
        tenant_id=tenant_id,
    )
    assert (
        client.get(
            "/api/v1/telemetry/metrics",
            headers={"Authorization": f"Bearer {eng_tok}"},
        ).status_code
        == 200
    )


# =============================================================================
# Pillar 7: Hardened /webhooks/*, POST /admin/purge, & OpenAPI Schema Fuzzing
# =============================================================================


@pytest.mark.postgres
def test_hardened_webhooks_admin_purge_and_openapi_schema_fuzzing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies fail-closed webhooks, RBAC/RLS DLQ & purge, and zero-500 OpenAPI fuzzing."""
    client = TestClient(app)
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    mem_queue = InMemoryTaskQueue()
    set_task_queue(mem_queue)

    # 1. Fail-closed webhooks (including production ignoring SCOUT_ALLOW_UNSIGNED_DEV_WEBHOOKS)
    for env_key in (
        "SCOUT_WEBHOOK_GITLAB_SECRET",
        "SCOUT_WEBHOOK_GITHUB_SECRET",
        "SCOUT_WEBHOOK_CONFLUENCE_SECRET",
        "SCOUT_WEBHOOK_JIRA_SECRET",
    ):
        monkeypatch.delenv(env_key, raising=False)

    prev_db_app_url = os.environ.get("SCOUT_DB_APP_URL")
    monkeypatch.setenv("SCOUT_ENVIRONMENT", "production")
    monkeypatch.setenv(
        "SCOUT_DB_APP_URL",
        "postgresql://scout_app_user:ProdStrongPass99!@127.0.0.1:54329/context_scout",
    )
    monkeypatch.setenv("SCOUT_ALLOW_UNSIGNED_DEV_WEBHOOKS", "true")
    for vendor in ("gitlab", "github", "confluence", "jira"):
        res_503 = client.post(f"/api/v1/webhooks/{vendor}", json={"domain_id": "core"})
        assert res_503.status_code == 503

    secrets_map = {
        "gitlab": ("SCOUT_WEBHOOK_GITLAB_SECRET", "X-Gitlab-Signature-256"),
        "github": ("SCOUT_WEBHOOK_GITHUB_SECRET", "X-Hub-Signature-256"),
        "confluence": ("SCOUT_WEBHOOK_CONFLUENCE_SECRET", "X-Confluence-Signature-256"),
        "jira": ("SCOUT_WEBHOOK_JIRA_SECRET", "X-Jira-Signature-256"),
    }
    for vendor, (env_var, sig_hdr) in secrets_map.items():
        secret = f"secret-{vendor}-0123456789abcdef"
        monkeypatch.setenv(env_var, secret)
        body = json.dumps({"domain_id": "code_architecture_scout", "ref": "main"}).encode("utf-8")
        bad_res = client.post(
            f"/api/v1/webhooks/{vendor}",
            content=body,
            headers={sig_hdr: "sha256=" + "0" * 64, "Content-Type": "application/json"},
        )
        assert bad_res.status_code == 401

        sig = "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        ok_res = client.post(
            f"/api/v1/webhooks/{vendor}",
            content=body,
            headers={
                sig_hdr: sig,
                "X-Scout-Tenant-Id": str(tenant_a),
                "Content-Type": "application/json",
            },
        )
        assert ok_res.status_code in {200, 202}
        assert ok_res.json()["status"] == "queued"

    monkeypatch.setenv("SCOUT_ENVIRONMENT", "development")
    if prev_db_app_url is None:
        monkeypatch.delenv("SCOUT_DB_APP_URL", raising=False)
    else:
        monkeypatch.setenv("SCOUT_DB_APP_URL", prev_db_app_url)

    # Webhook stats/DLQ RBAC + tenant RLS
    assert client.get("/api/v1/webhooks/stats").status_code == 401
    intern_tok = create_signed_principal_token(
        user_id="pm_restricted",
        role=RoleType.PRODUCT_MANAGER,
        allowed_scopes=["public"],
        tenant_id=tenant_a,
    )
    assert (
        client.get(
            "/api/v1/webhooks/stats",
            headers={"Authorization": f"Bearer {intern_tok}"},
        ).status_code
        == 403
    )

    dlq_task_a = mem_queue.enqueue_task(
        domain_id="code_architecture_scout",
        event_type="gitlab_push",
        payload={"x": 1},
        priority=999,
        max_attempts=1,
        tenant_id=tenant_a,
    )
    claimed_a = mem_queue.fetch_next_task("worker-1", tenant_id=tenant_a)
    assert claimed_a is not None
    assert claimed_a.task_id == dlq_task_a
    # fail_task returns False once max_attempts is exhausted and moves task to DLQ
    assert not mem_queue.fail_task(
        dlq_task_a, error_message="boom", stack_trace="trace", tenant_id=tenant_a
    )

    tok_a_comp = create_signed_principal_token(
        user_id="comp_a",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["audit:compliance"],
        tenant_id=tenant_a,
    )
    tok_b_comp = create_signed_principal_token(
        user_id="comp_b",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["audit:compliance"],
        tenant_id=tenant_b,
    )
    dlq_b = client.get("/api/v1/webhooks/dlq", headers={"Authorization": f"Bearer {tok_b_comp}"})
    assert dlq_b.status_code == 200
    assert dlq_b.json()["total"] == 0
    retry_cross = client.post(
        f"/api/v1/webhooks/dlq/{dlq_task_a}/retry",
        headers={"Authorization": f"Bearer {tok_b_comp}"},
    )
    assert retry_cross.status_code == 404
    retry_ok = client.post(
        f"/api/v1/webhooks/dlq/{dlq_task_a}/retry",
        headers={"Authorization": f"Bearer {tok_a_comp}"},
    )
    assert retry_ok.status_code == 200

    # 2. POST /api/v1/admin/purge RBAC + live PG17 <60s zero-residue purge
    assert (
        client.post(
            "/api/v1/admin/purge",
            json={"domain_id": "core", "doc_id": "d1"},
        ).status_code
        == 401
    )
    eng_tok = create_signed_principal_token(
        user_id="eng_no_purge",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
        tenant_id=tenant_a,
    )
    assert (
        client.post(
            "/api/v1/admin/purge",
            headers={"Authorization": f"Bearer {eng_tok}"},
            json={"domain_id": "core", "doc_id": "d1"},
        ).status_code
        == 403
    )

    eph_domain = f"ephem_m5_purge_{uuid.uuid4().hex[:8]}"
    settings = ScoutSettings()
    ingestion_mgr = RLSConnectionManager(settings.db_app_url, pool_role=PoolRole.INGESTION)
    try:
        with psycopg.connect(settings.db_admin_url, autocommit=True) as admin_conn:
            admin_conn.execute(
                """
                INSERT INTO domain_packs (
                    domain_id, display_name, lora_adapter_id, primary_truth_source,
                    freshness_half_life_days, config_json, tenant_id
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s);
                """,
                (
                    eph_domain,
                    "Ephemeral M5 Purge",
                    "none",
                    "code",
                    180,
                    "{}",
                    tenant_a,
                ),
            )
        comp_principal = PrincipalContext(
            user_id="comp_a",
            role=RoleType.COMPLIANCE_OFFICER,
            allowed_scopes=["audit:compliance"],
            tenant_id=tenant_a,
        )
        pg_repo = PostgresRLSRepository()
        with ingestion_mgr.scoped_transaction(comp_principal) as tx:
            pg_repo.insert_code_file(
                tx,
                CodeFileRecord(
                    file_id=uuid.uuid4(),
                    tenant_id=tenant_a,
                    domain_id=eph_domain,
                    repo_name="purge-target-repo",
                    file_path="secrets/leaked.py",
                    git_commit_sha="1" * 40,
                    git_blob_sha="2" * 40,
                    language="python",
                    allowed_scopes=["audit:compliance"],
                    raw_content="SECRET_MARKER = 'should_be_purged'\n",
                    line_count=1,
                ),
            )

        purge_res = client.post(
            "/api/v1/admin/purge",
            headers={"Authorization": f"Bearer {tok_a_comp}"},
            json={
                "domain_id": eph_domain,
                "repo_name": "purge-target-repo",
                "file_path": "secrets/leaked.py",
            },
        )
        assert purge_res.status_code == 200
        purge_body = purge_res.json()
        assert purge_body["residue_found"] == 0
        assert purge_body["sla_met"] is True
        assert purge_body["pg17_deleted"].get("code_files", 0) == 1

        # 3. OpenAPI schema fuzzing across all FastAPI paths & methods: zero 500s, zero auth bypass
        monkeypatch.setenv("SCOUT_SCIM_BEARER_TOKEN", "fuzz-scim-secret")
        spec = app.openapi()
        monkeypatch.setenv("SCOUT_ENVIRONMENT", "production")
        monkeypatch.setenv(
            "SCOUT_DB_APP_URL",
            "postgresql://scout_app_user:ProdStrongPass99!@127.0.0.1:54329/context_scout",
        )
        public_paths = {"/api/v1/health"}
        for raw_path, path_item in sorted(spec["paths"].items()):
            fuzzed_path = (
                raw_path.replace("{user_id}", str(uuid.uuid4()))
                .replace("{group_id}", str(uuid.uuid4()))
                .replace("{task_id}", str(uuid.uuid4()))
            )
            for http_method in sorted(path_item.keys()):
                if http_method not in {"get", "post", "put", "patch", "delete"}:
                    continue
                unauth = client.request(
                    http_method.upper(),
                    fuzzed_path,
                    json={"invalid_fuzz_key": "../etc/passwd"}
                    if http_method in {"post", "put", "patch"}
                    else None,
                )
                assert unauth.status_code < 500, f"Unauth 5xx on {http_method} {fuzzed_path}"
                if raw_path not in public_paths:
                    assert not (200 <= unauth.status_code < 300), (
                        f"Auth bypass on {http_method} {fuzzed_path}: {unauth.status_code}"
                    )

                authed = client.request(
                    http_method.upper(),
                    fuzzed_path,
                    headers={"Authorization": f"Bearer {tok_a_comp}"},
                    params={"format": "gcp_logging", "limit": 5} if http_method == "get" else None,
                    json={"unexpected_field": [1, 2, 3], "domain_id": "../../etc/passwd"}
                    if http_method in {"post", "put", "patch"}
                    else None,
                )
                assert authed.status_code < 500, (
                    f"Server 5xx on {http_method} {fuzzed_path}: {authed.text}"
                )
    finally:
        set_task_queue(None)
        _cleanup_pg_tenants([tenant_a, tenant_b], domain_ids=[eph_domain])


# =============================================================================
# Pillar 8: Stateless Streamable HTTP MCP 2026-07-28 + IDE CoW Branch/MR Tools
# =============================================================================


@pytest.mark.postgres
async def test_stateless_mcp_2026_07_28_sep_compliance_traceparent_and_cow_mr_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies SEP-2243, SEP-2575, SEP-2663, zero identity args, & live PG17 CoW MR tools."""
    settings = ScoutSettings()
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    app_mgr = RLSConnectionManager(settings.db_app_url, pool_role=PoolRole.QUERY)
    ingestion_mgr = RLSConnectionManager(settings.db_app_url, pool_role=PoolRole.INGESTION)

    tok_a = create_signed_principal_token(
        user_id="mcp_eng_a",
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
        tenant_id=tenant_a,
    )

    # 1. SEP-2243 /.well-known/mcp discovery (schemas redacted unauth, full when authed) & RFC 9728
    api_client = TestClient(app)
    prm_res = api_client.get("/.well-known/oauth-protected-resource")
    assert prm_res.status_code == 200
    prm_body = prm_res.json()
    assert "resource" in prm_body and prm_body["bearer_methods_supported"] == ["header"]

    for wk_path in ("/.well-known/mcp", "/api/v1/.well-known/mcp"):
        wk_unauth = api_client.get(wk_path).json()
        assert wk_unauth["protocolVersion"] == MCP_PROTOCOL_VERSION_2026_07_28
        assert wk_unauth["schemasRedacted"] is True
        assert wk_unauth["tools"] == []
        assert wk_unauth["toolCount"] >= 3

        wk_res = api_client.get(wk_path, headers={"Authorization": f"Bearer {tok_a}"})
        assert wk_res.status_code == 200
        wk = wk_res.json()
        assert wk["schemasRedacted"] is False
        assert wk["protocolVersion"] == MCP_PROTOCOL_VERSION_2026_07_28
        assert wk["transport"]["stateless"] is True
        assert wk["transport"]["sep"] == "SEP-2575"
        assert wk["discovery"]["sep"] == "SEP-2243"
        assert wk["capabilities"]["tracing"]["sep"] == "SEP-2663"
        assert len(wk["tools"]) >= 3
        for tool in wk["tools"]:
            props = tool["inputSchema"]["properties"]
            assert not (set(props.keys()) & FORBIDDEN_TOOL_ARGUMENTS)

    # 2. Two independent stateless HTTP MCP instances (no shared in-memory session state)
    mcp_app_a = create_code_repo_mcp_http_app(
        settings=settings,
        connection_manager=app_mgr,
        domain_id="code_architecture_scout",
    )
    mcp_app_b = create_code_repo_mcp_http_app(
        settings=settings,
        connection_manager=app_mgr,
        domain_id="code_architecture_scout",
    )

    repo_a = "repo-alpha-m5"
    repo_b = "repo-beta-m5"
    tok_b = create_signed_principal_token(
        user_id="mcp_eng_b",
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
        tenant_id=tenant_b,
    )
    p_a = PrincipalContext(
        user_id="mcp_eng_a",
        role=RoleType.ENGINEER,
        allowed_scopes=list(DEFAULT_ROLE_SCOPES[RoleType.ENGINEER]),
        tenant_id=tenant_a,
    )
    p_b = PrincipalContext(
        user_id="mcp_eng_b",
        role=RoleType.ENGINEER,
        allowed_scopes=list(DEFAULT_ROLE_SCOPES[RoleType.ENGINEER]),
        tenant_id=tenant_b,
    )

    try:
        with psycopg.connect(settings.db_admin_url, autocommit=True) as admin_conn:
            for tid in (tenant_a, tenant_b):
                admin_conn.execute(
                    """
                    INSERT INTO domain_packs (
                        domain_id, display_name, lora_adapter_id, primary_truth_source,
                        freshness_half_life_days, config_json, tenant_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (tenant_id, domain_id) DO NOTHING;
                    """,
                    (
                        "code_architecture_scout",
                        "Code Architecture Scout",
                        "none",
                        "code",
                        180,
                        "{}",
                        tid,
                    ),
                )

        pg_repo = PostgresRLSRepository()
        for tid, princ, rname, mr_iid, base_fn, new_fn in (
            (tenant_a, p_a, repo_a, 101, "return 1\n", "return 42\n"),
            (tenant_b, p_b, repo_b, 202, "return 9\n", "return 99\n"),
        ):
            with ingestion_mgr.scoped_transaction(princ) as tx:
                repo_id = uuid.uuid4()
                pg_repo.insert_repository(
                    tx,
                    RepositoryRecord(
                        repo_id=repo_id,
                        tenant_id=tid,
                        domain_id="code_architecture_scout",
                        repo_name=rname,
                        source_type="gitlab",
                        remote_url=f"https://gitlab.local/{rname}",
                        default_branch="main",
                        allowed_scopes=["public_eng"],
                    ),
                )
                pg_repo.insert_code_file(
                    tx,
                    CodeFileRecord(
                        file_id=uuid.uuid4(),
                        tenant_id=tid,
                        domain_id="code_architecture_scout",
                        repo_name=rname,
                        file_path="src/calc.py",
                        git_commit_sha="a" * 40,
                        git_blob_sha="b" * 40,
                        language="python",
                        allowed_scopes=["public_eng"],
                        raw_content=f"def compute() -> int:\n    {base_fn}",
                        line_count=2,
                        repo_id=repo_id,
                    ),
                )
                snap = pg_repo.upsert_ref_snapshot(
                    tx,
                    RefSnapshotRecord(
                        snapshot_id=uuid.uuid4(),
                        tenant_id=tid,
                        domain_id="code_architecture_scout",
                        repo_id=repo_id,
                        repo_name=rname,
                        ref_name=f"mr/{mr_iid}",
                        ref_kind="merge_request",
                        mr_iid=str(mr_iid),
                        base_commit_sha="a" * 40,
                        head_commit_sha="c" * 40,
                        state="open",
                        allowed_scopes=["public_eng"],
                    ),
                )
                pg_repo.upsert_overlay_file_delta(
                    tx,
                    OverlayFileDeltaRecord(
                        delta_id=uuid.uuid4(),
                        snapshot_id=snap.snapshot_id,
                        tenant_id=tid,
                        domain_id="code_architecture_scout",
                        repo_name=rname,
                        file_path="src/calc.py",
                        change_type="modified",
                        overlay_blob_sha="d" * 40,
                        language="python",
                        raw_content=f"def compute() -> int:\n    {new_fn}",
                        line_count=2,
                        allowed_scopes=["public_eng"],
                        symbols_json=[{"qualified_name": "compute"}],
                    ),
                )

        tracer = ScoutTelemetryTracer.reset_instance()
        w3c_trace_hex = "4bf92f3577b34da6a3ce929d0e0e4736"
        w3c_tp = f"00-{w3c_trace_hex}-00f067aa0ba902b7-01"
        cloud_trace_hex = "105445aa7843bc8bf206b12000100000"

        async with (
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=mcp_app_a),
                base_url="http://127.0.0.1",
            ) as cli_a,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=mcp_app_b),
                base_url="http://127.0.0.1",
            ) as cli_b,
        ):
            # Discovery on mounted MCP HTTP app & RFC 9728 + Host allowlist (ECS-001, ECS-045)
            wk_mcp = await cli_a.get("/.well-known/mcp")
            assert wk_mcp.status_code == 200
            assert wk_mcp.json()["protocolVersion"] == MCP_PROTOCOL_VERSION_2026_07_28
            assert wk_mcp.json()["schemasRedacted"] is True

            prm_mcp = await cli_a.get("/.well-known/oauth-protected-resource")
            assert prm_mcp.status_code == 200
            assert "authorization_servers" in prm_mcp.json()

            # Unapproved external Host header rejected with 421; K8s service Host allowed
            bad_host_res = await cli_a.post(
                "/mcp",
                headers=_mcp_headers(
                    tok_a,
                    method="tools/list",
                    extra={"Host": "evil.attacker.example"},
                ),
                json=_mcp_payload(91, "tools/list"),
            )
            assert bad_host_res.status_code == 421

            k8s_host_res = await cli_a.post(
                "/mcp",
                headers=_mcp_headers(
                    tok_a,
                    method="tools/list",
                    extra={"Host": "context-scout-api:8080"},
                ),
                json=_mcp_payload(92, "tools/list"),
            )
            assert k8s_host_res.status_code == 200

            # Unauthenticated -> 401 with RFC 9728 WWW-Authenticate resource_metadata
            unauth_mcp = await cli_a.post(
                "/mcp",
                headers={"Accept": "application/json, text/event-stream"},
                json=_mcp_payload(1, "tools/list"),
            )
            assert unauth_mcp.status_code == 401
            assert "resource_metadata=" in unauth_mcp.headers.get("www-authenticate", "")

            # RFC 8707 expected resource enforcement via SCOUT_MCP_EXPECTED_RESOURCE
            monkeypatch.setenv("SCOUT_MCP_EXPECTED_RESOURCE", "https://mcp.corp.local/mcp")
            mismatch_res = await cli_a.post(
                "/mcp",
                headers=_mcp_headers(tok_a, method="tools/list"),
                json=_mcp_payload(93, "tools/list"),
            )
            assert mismatch_res.status_code == 401
            monkeypatch.delenv("SCOUT_MCP_EXPECTED_RESOURCE", raising=False)

            # Sticky Mcp-Session-Id rejected -> 400 (SEP-2575)
            sticky_res = await cli_a.post(
                "/mcp",
                headers=_mcp_headers(
                    tok_a,
                    method="tools/list",
                    extra={"Mcp-Session-Id": "sticky-session-123"},
                ),
                json=_mcp_payload(2, "tools/list"),
            )
            assert sticky_res.status_code == 400

            # Round-robin call 1 to instance A: tools/list + W3C traceparent
            list_res = await cli_a.post(
                "/mcp",
                headers=_mcp_headers(
                    tok_a,
                    method="tools/list",
                    extra={"traceparent": w3c_tp},
                ),
                json=_mcp_payload(3, "tools/list"),
            )
            assert list_res.status_code == 200
            assert "mcp-session-id" not in {k.lower() for k in list_res.headers}
            assert w3c_trace_hex in list_res.headers.get("traceparent", "")
            list_body = _extract_mcp_result(list_res)
            tool_names = {t["name"] for t in list_body["result"]["tools"]}
            assert {
                "list_repo_branches_and_mrs",
                "inspect_mr_overlay_diff",
                "slice_branch_or_mr_file",
            } <= tool_names

            # Round-robin call 2 to instance B: list_repo_branches_and_mrs + X-Cloud-Trace-Context
            snaps_res = await cli_b.post(
                "/mcp",
                headers=_mcp_headers(
                    tok_a,
                    method="tools/call",
                    tool_name="list_repo_branches_and_mrs",
                    extra={"X-Cloud-Trace-Context": f"{cloud_trace_hex}/42;o=1"},
                ),
                json=_mcp_payload(
                    4,
                    "tools/call",
                    {"name": "list_repo_branches_and_mrs", "arguments": {}},
                ),
            )
            assert snaps_res.status_code == 200
            assert "mcp-session-id" not in {k.lower() for k in snaps_res.headers}
            assert cloud_trace_hex in snaps_res.headers.get("traceparent", "")
            snaps_data = json.loads(_extract_mcp_result(snaps_res)["result"]["content"][0]["text"])
            snap_repos = {s["repo_name"] for s in snaps_data["snapshots"]}
            assert repo_a in snap_repos
            assert repo_b not in snap_repos

            # Round-robin call 3 to instance A: inspect_mr_overlay_diff
            diff_res = await cli_a.post(
                "/mcp",
                headers=_mcp_headers(
                    tok_a, method="tools/call", tool_name="inspect_mr_overlay_diff"
                ),
                json=_mcp_payload(
                    5,
                    "tools/call",
                    {
                        "name": "inspect_mr_overlay_diff",
                        "arguments": {"repo_name": repo_a, "mr_iid": 101},
                    },
                ),
            )
            assert diff_res.status_code == 200
            diff_data = json.loads(_extract_mcp_result(diff_res)["result"]["content"][0]["text"])
            assert diff_data["found"] is True
            assert "-    return 1" in diff_data["deltas"][0]["unified_diff"]
            assert "+    return 42" in diff_data["deltas"][0]["unified_diff"]

            # Round-robin call 4 to instance B: slice_branch_or_mr_file
            slice_res = await cli_b.post(
                "/mcp",
                headers=_mcp_headers(
                    tok_a, method="tools/call", tool_name="slice_branch_or_mr_file"
                ),
                json=_mcp_payload(
                    6,
                    "tools/call",
                    {
                        "name": "slice_branch_or_mr_file",
                        "arguments": {
                            "repo_name": repo_a,
                            "mr_iid": 101,
                            "file_path": "src/calc.py",
                            "start_line": 1,
                            "end_line": 2,
                        },
                    },
                ),
            )
            assert slice_res.status_code == 200
            slice_data = json.loads(_extract_mcp_result(slice_res)["result"]["content"][0]["text"])
            assert "return 42" in "\n".join(ln["content"] for ln in slice_data["lines"])
            assert slice_data["slice_line_count"] == 2

            # Cross-tenant FORCE RLS check: tenant_b cannot see tenant_a's MR
            cross_diff = await cli_a.post(
                "/mcp",
                headers=_mcp_headers(
                    tok_b, method="tools/call", tool_name="inspect_mr_overlay_diff"
                ),
                json=_mcp_payload(
                    7,
                    "tools/call",
                    {
                        "name": "inspect_mr_overlay_diff",
                        "arguments": {"repo_name": repo_a, "mr_iid": 101},
                    },
                ),
            )
            assert cross_diff.status_code == 200
            cross_data = json.loads(_extract_mcp_result(cross_diff)["result"]["content"][0]["text"])
            assert cross_data["found"] is False

            # Fail-closed injection of forbidden allowed_scopes argument (ECS-096)
            inject_res = await cli_a.post(
                "/mcp",
                headers=_mcp_headers(
                    tok_a, method="tools/call", tool_name="list_repo_branches_and_mrs"
                ),
                json=_mcp_payload(
                    8,
                    "tools/call",
                    {
                        "name": "list_repo_branches_and_mrs",
                        "arguments": {"allowed_scopes": ["*"]},
                    },
                ),
            )
            assert inject_res.status_code < 500, (
                f"Expected controlled JSON-RPC/HTTP rejection, got {inject_res.status_code}"
            )
            if inject_res.status_code == 200:
                inj_body = _extract_mcp_result(inject_res)
                assert (
                    inj_body.get("error") is not None
                    or inj_body.get("result", {}).get("isError") is True
                )
            else:
                assert inject_res.status_code in {400, 422}

        recorded_trace_ids = {
            f"{sp.context.trace_id:032x}" for sp in tracer.exporter.get_finished_spans()
        }
        assert w3c_trace_hex in recorded_trace_ids
        assert cloud_trace_hex in recorded_trace_ids
    finally:
        _cleanup_pg_tenants([tenant_a, tenant_b])


# =============================================================================
# Pillar 9: Multi-Tenant Telemetry, Budget & OTel Regression Tests
# =============================================================================


def test_non_default_tenant_telemetry_persists_under_pg17_force_rls() -> None:
    """Non-default tenant _record_live_scout_telemetry writes under PG17 FORCE RLS."""
    from context_scout.api.routes.scout import _record_live_scout_telemetry

    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    princ_a = PrincipalContext(
        user_id="disc025_user_a",
        role=RoleType.ENGINEER,
        allowed_scopes=list(DEFAULT_ROLE_SCOPES[RoleType.ENGINEER]),
        tenant_id=tenant_a,
    )
    tok_a = create_signed_principal_token(
        user_id="disc025_user_a",
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
        tenant_id=tenant_a,
    )
    tok_b = create_signed_principal_token(
        user_id="disc025_user_b",
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
        tenant_id=tenant_b,
    )
    pack = ImplementationContextPack(
        trace_id=f"disc025-{uuid.uuid4().hex}",
        domain_id="code_architecture_scout",
        user_task="Verify telemetry tenant scoping",
        architectural_execution_plan=["Validate tenant isolation in telemetry"],
        code_slices=[
            VerbatimCodeSlice(
                file_path="src/app.py",
                repo_name="core",
                qualified_symbol="core.app.ok",
                selection_reason="Target ok check",
                modification_intent=ModificationIntent.MODIFY_TARGET,
                git_commit_sha="a" * 40,
                git_blob_sha="b" * 40,
                start_line=1,
                end_line=5,
                verbatim_code_with_lines="   1 | def ok() -> bool:\n   2 |     return True\n",
            )
        ],
        preflight_report=PreFlightValidationReport(
            passed=True,
            normalized_rrf_score=0.91,
            citations_verified_100_pct=True,
        ),
        local_tokens_consumed=8000,
        pack_token_count=600,
    )

    from context_scout.api.dependencies import get_domain_registry

    repo = get_repository()
    registry = get_domain_registry()
    client = TestClient(app)
    try:
        _record_live_scout_telemetry(repo, registry, princ_a, pack, latency_ms=25)
        roi_a = client.get("/api/v1/telemetry/roi", headers={"Authorization": f"Bearer {tok_a}"})
        roi_b = client.get("/api/v1/telemetry/roi", headers={"Authorization": f"Bearer {tok_b}"})
        assert roi_a.status_code == 200
        assert roi_b.status_code == 200
        assert roi_a.json()["summary"]["total_queries"] == 1
        assert roi_b.json()["summary"]["total_queries"] == 0
    finally:
        if isinstance(repo, InMemoryRLSRepository):
            tid_uuid = uuid.uuid5(uuid.NAMESPACE_OID, pack.trace_id)
            repo._telemetry.pop(tid_uuid, None)
        else:
            _cleanup_pg_tenants([tenant_a, tenant_b])


def test_session_budget_tracker_and_inmemory_repo_isolate_tenants() -> None:
    """Identical user_id across tenants maintains strictly isolated budgets."""
    from context_scout.telemetry.cost_governance import SessionBudgetTracker

    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    p_a = PrincipalContext(
        user_id="shared_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
        tenant_id=tenant_a,
        max_session_paid_tokens=1_000,
    )
    p_b = PrincipalContext(
        user_id="shared_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
        tenant_id=tenant_b,
        max_session_paid_tokens=1_000,
    )

    tracker = SessionBudgetTracker()
    tracker.record_consumption(p_a, input_tokens=900, output_tokens=50, cost_usd=0.01)
    # p_b must still have full unused budget
    tracker.check_and_reserve(p_b, projected_tokens=800, estimated_cost_usd=0.01)
    assert tracker.get_consumption("shared_eng", tenant_b)["total_tokens"] == 0

    mem_repo = InMemoryRLSRepository()
    mem_repo.record_execution_telemetry(
        ExecutionTelemetryRecord(
            trace_id=uuid.uuid4(),
            tenant_id=tenant_a,
            domain_id="code_architecture_scout",
            principal_id="shared_eng",
            principal_role="engineer",
            execution_mode=ExecutionMode.GEMINI_HANDOFF,
            local_tokens_read=5000,
            context_pack_tokens_sent=950,
            paid_tokens_saved=4050,
            estimated_usd_saved=0.02,
            preflight_passed_first_try=True,
            latency_ms=10,
        )
    )
    assert mem_repo.get_principal_paid_tokens_consumed("shared_eng", tenant_id=tenant_a) == 950
    assert mem_repo.get_principal_paid_tokens_consumed("shared_eng", tenant_id=tenant_b) == 0


def test_otel_middleware_ignores_spoofed_unauthenticated_principal_header() -> None:
    """OTel HTTP span never stamps spoofed X-Principal-User-Id prior to authentication."""
    tracer = ScoutTelemetryTracer.reset_instance()
    client = TestClient(app)

    spoof_res = client.get(
        "/api/v1/auth/me",
        headers={
            "X-Principal-User-Id": "spoofed_root_admin",
            "X-Principal-Role": "compliance_officer",
        },
    )
    assert spoof_res.status_code == 401
    spans = tracer.exporter.get_finished_spans()
    http_spans = [sp for sp in spans if sp.name == "http.get./api/v1/auth/me"]
    assert http_spans
    last_attrs = dict(http_spans[-1].attributes or {})
    assert last_attrs.get(ATTR_SCOUT_PRINCIPAL_ID) == "anonymous"
    assert last_attrs.get(ATTR_SCOUT_PRINCIPAL_ID) != "spoofed_root_admin"

    # Authenticated request binds verified principal ID
    real_tok = create_signed_principal_token(
        user_id="verified_user_99",
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
    )
    ok_res = client.get(
        "/api/v1/auth/me",
        headers={
            "Authorization": f"Bearer {real_tok}",
            "X-Principal-User-Id": "spoofed_override",
        },
    )
    assert ok_res.status_code == 200
    spans2 = tracer.exporter.get_finished_spans()
    http_spans2 = [sp for sp in spans2 if sp.name == "http.get./api/v1/auth/me"]
    last_attrs2 = dict(http_spans2[-1].attributes or {})
    assert last_attrs2.get(ATTR_SCOUT_PRINCIPAL_ID) == "verified_user_99"
    assert os.environ.get("PYTEST_CURRENT_TEST")


@pytest.mark.postgres
def test_admin_purge_fail_closed_and_selector_modes() -> None:
    """/admin/purge rejects missing repo_name (400) and supports doc_id/repo."""
    from datetime import UTC, datetime

    from context_scout.db.schema import DocChunkRecord

    client = TestClient(app)
    tenant_a = uuid.uuid4()
    eph_domain = f"ephem_disc069_{uuid.uuid4().hex[:8]}"
    tok_comp = create_signed_principal_token(
        user_id="comp_disc069",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["audit:compliance"],
        tenant_id=tenant_a,
    )
    comp_principal = PrincipalContext(
        user_id="comp_disc069",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["audit:compliance"],
        tenant_id=tenant_a,
    )
    settings = ScoutSettings()
    ingestion_mgr = RLSConnectionManager(settings.db_app_url, pool_role=PoolRole.INGESTION)
    try:
        with psycopg.connect(settings.db_admin_url, autocommit=True) as admin_conn:
            admin_conn.execute(
                """
                INSERT INTO domain_packs (
                    domain_id, display_name, lora_adapter_id, primary_truth_source,
                    freshness_half_life_days, config_json, tenant_id
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s);
                """,
                (
                    eph_domain,
                    "Ephemeral Selector Purge",
                    "none",
                    "code",
                    180,
                    "{}",
                    tenant_a,
                ),
            )

        pg_repo = PostgresRLSRepository()
        with ingestion_mgr.scoped_transaction(comp_principal) as tx:
            for rname in ("purge-repo-a", "purge-repo-keep"):
                pg_repo.insert_code_file(
                    tx,
                    CodeFileRecord(
                        file_id=uuid.uuid4(),
                        tenant_id=tenant_a,
                        domain_id=eph_domain,
                        repo_name=rname,
                        file_path="secrets/leaked.py",
                        git_commit_sha="1" * 40,
                        git_blob_sha="2" * 40,
                        language="python",
                        allowed_scopes=["audit:compliance"],
                        raw_content="SECRET_MARKER = 'disc069'\n",
                        line_count=1,
                    ),
                )
            pg_repo.insert_doc_chunk(
                tx,
                DocChunkRecord(
                    chunk_id=uuid.uuid4(),
                    domain_id=eph_domain,
                    doc_id="doc.v1.2026",
                    doc_title="Dotted Doc ID",
                    source_uri="urn:docs:doc.v1.2026",
                    market_scope=None,
                    published_date=datetime.now(UTC).date(),
                    authority_weight=1.0,
                    section_heading="Section 1",
                    contextual_header="Section 1",
                    verbatim_content="Dotted doc_id content for selector purge.",
                    metadata_json={},
                    allowed_scopes=["audit:compliance"],
                    embedding=[0.2] * 768,
                    source_type="confluence",
                    tenant_id=tenant_a,
                ),
            )

        # 1. Fail-closed HTTP 400 when file_path lacks repo_name (missing or whitespace)
        for bad_payload in (
            {"domain_id": eph_domain, "file_path": "secrets/leaked.py"},
            {"domain_id": eph_domain, "repo_name": "   ", "file_path": "secrets/leaked.py"},
        ):
            bad_res = client.post(
                "/api/v1/admin/purge",
                headers={"Authorization": f"Bearer {tok_comp}"},
                json=bad_payload,
            )
            assert bad_res.status_code == 400
            assert "file_path purge requires repo_name" in bad_res.json()["detail"]

        # Verify PG17 code_files rows were untouched by rejected purges
        with ingestion_mgr.scoped_transaction(comp_principal) as tx:
            row = tx.execute(
                "SELECT count(*) FROM code_files WHERE domain_id = %s;",
                (eph_domain,),
            ).fetchone()
            assert row is not None and int(str(row[0])) == 2

        # 2. Dotted doc_id purge succeeds without repo_name
        doc_res = client.post(
            "/api/v1/admin/purge",
            headers={"Authorization": f"Bearer {tok_comp}"},
            json={"domain_id": eph_domain, "doc_id": "doc.v1.2026"},
        )
        assert doc_res.status_code == 200
        doc_body = doc_res.json()
        assert doc_body["residue_found"] == 0
        assert doc_body["pg17_deleted"].get("doc_chunks", 0) == 1

        # 3. Repo-only purge removes purge-repo-a while leaving purge-repo-keep intact
        repo_res = client.post(
            "/api/v1/admin/purge",
            headers={"Authorization": f"Bearer {tok_comp}"},
            json={"domain_id": eph_domain, "repo_name": "purge-repo-a"},
        )
        assert repo_res.status_code == 200
        repo_body = repo_res.json()
        assert repo_body["residue_found"] == 0
        assert repo_body["pg17_deleted"].get("code_files", 0) == 1

        with ingestion_mgr.scoped_transaction(comp_principal) as tx:
            rem_rows = tx.execute(
                "SELECT repo_name FROM code_files WHERE domain_id = %s;",
                (eph_domain,),
            ).fetchall()
            assert [str(r[0]) for r in rem_rows] == ["purge-repo-keep"]
    finally:
        _cleanup_pg_tenants([tenant_a], domain_ids=[eph_domain])


@pytest.mark.postgres
def test_admin_purge_rejects_blank_and_ambiguous_selectors() -> None:
    """/admin/purge rejects blank & ambiguous selectors while preserving RBAC."""
    from datetime import UTC, datetime

    from context_scout.db.schema import DocChunkRecord

    client = TestClient(app)
    tenant_a = uuid.uuid4()
    eph_domain = f"ephem_disc070_{uuid.uuid4().hex[:8]}"
    tok_comp = create_signed_principal_token(
        user_id="comp_disc070",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["audit:compliance"],
        tenant_id=tenant_a,
    )
    tok_eng = create_signed_principal_token(
        user_id="eng_disc070",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
        tenant_id=tenant_a,
    )
    comp_principal = PrincipalContext(
        user_id="comp_disc070",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["audit:compliance"],
        tenant_id=tenant_a,
    )
    settings = ScoutSettings()
    ingestion_mgr = RLSConnectionManager(settings.db_app_url, pool_role=PoolRole.INGESTION)
    try:
        with psycopg.connect(settings.db_admin_url, autocommit=True) as admin_conn:
            admin_conn.execute(
                """
                INSERT INTO domain_packs (
                    domain_id, display_name, lora_adapter_id, primary_truth_source,
                    freshness_half_life_days, config_json, tenant_id
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s);
                """,
                (
                    eph_domain,
                    "Ephemeral Ambiguous Selector Purge",
                    "none",
                    "code",
                    180,
                    "{}",
                    tenant_a,
                ),
            )

        pg_repo = PostgresRLSRepository()
        with ingestion_mgr.scoped_transaction(comp_principal) as tx:
            pg_repo.insert_code_file(
                tx,
                CodeFileRecord(
                    file_id=uuid.uuid4(),
                    tenant_id=tenant_a,
                    domain_id=eph_domain,
                    repo_name="acme/keep-repo",
                    file_path="src/keep.py",
                    git_commit_sha="1" * 40,
                    git_blob_sha="2" * 40,
                    language="python",
                    allowed_scopes=["audit:compliance"],
                    raw_content="KEEP_MARKER = 'disc070'\n",
                    line_count=1,
                ),
            )
            pg_repo.insert_doc_chunk(
                tx,
                DocChunkRecord(
                    chunk_id=uuid.uuid4(),
                    domain_id=eph_domain,
                    doc_id="doc.v1.2026",
                    doc_title="Keep Doc",
                    source_uri="urn:docs:doc.v1.2026",
                    market_scope=None,
                    published_date=datetime.now(UTC).date(),
                    authority_weight=1.0,
                    section_heading="Section 1",
                    contextual_header="Section 1",
                    verbatim_content="Keep doc content for selector validation.",
                    metadata_json={},
                    allowed_scopes=["audit:compliance"],
                    embedding=[0.2] * 768,
                    source_type="confluence",
                    tenant_id=tenant_a,
                ),
            )

        # 1. RBAC 401/403 must still take precedence over selector validation 400
        unauth_res = client.post(
            "/api/v1/admin/purge",
            json={"domain_id": eph_domain, "repo_name": "acme/keep-repo", "file_path": "   "},
        )
        assert unauth_res.status_code == 401
        eng_res = client.post(
            "/api/v1/admin/purge",
            headers={"Authorization": f"Bearer {tok_eng}"},
            json={"domain_id": eph_domain, "repo_name": "acme/keep-repo", "file_path": "   "},
        )
        assert eng_res.status_code == 403

        # 2. All blank/whitespace and ambiguous multi-selector combinations must fail closed (400)
        bad_payloads: tuple[dict[str, Any], ...] = (
            {"domain_id": eph_domain, "repo_name": "acme/keep-repo", "file_path": ""},
            {"domain_id": eph_domain, "repo_name": "acme/keep-repo", "file_path": "   "},
            {"domain_id": eph_domain, "repo_name": "acme/keep-repo", "doc_id": ""},
            {"domain_id": eph_domain, "repo_name": "acme/keep-repo", "doc_id": "   "},
            {"domain_id": eph_domain, "repo_name": "acme/keep-repo", "commit_sha": "   "},
            {"domain_id": eph_domain, "repo_name": "acme/keep-repo", "blob_sha": "   "},
            {"domain_id": eph_domain, "repo_name": "acme/keep-repo", "confluence_page_id": "   "},
            {"domain_id": eph_domain, "repo_name": "acme/keep-repo", "jira_issue_key": "   "},
            {"domain_id": eph_domain, "repo_name": "acme/keep-repo", "object_keys": ["   "]},
            {"domain_id": eph_domain, "repo_name": "   "},
            {"domain_id": eph_domain, "doc_id": "doc.v1.2026", "repo_name": "acme/keep-repo"},
            {"domain_id": eph_domain, "repo_name": "acme/keep-repo", "commit_sha": "1" * 40},
            {"domain_id": eph_domain, "repo_name": "acme/keep-repo", "blob_sha": "2" * 40},
            {
                "domain_id": eph_domain,
                "repo_name": "acme/keep-repo",
                "file_path": "src/keep.py",
                "commit_sha": "1" * 40,
            },
            {
                "domain_id": eph_domain,
                "confluence_page_id": "cf-1",
                "jira_issue_key": "SEC-1",
            },
        )
        for bad_payload in bad_payloads:
            res = client.post(
                "/api/v1/admin/purge",
                headers={"Authorization": f"Bearer {tok_comp}"},
                json=bad_payload,
            )
            assert res.status_code == 400, (
                f"Expected 400 for {bad_payload!r}, got {res.status_code}"
            )

        # 2b. OpenAPI list[str] type-mismatch contract: object_keys="str" fails closed (422)
        # (distinct from ArtifactPurgeCoordinator.purge / IndexingWorker which raise ValueError/DLQ)
        type_mismatch_res = client.post(
            "/api/v1/admin/purge",
            headers={"Authorization": f"Bearer {tok_comp}"},
            json={"domain_id": eph_domain, "repo_name": "acme/keep-repo", "object_keys": "str"},
        )
        assert type_mismatch_res.status_code == 422, (
            f"Expected 422 for object_keys='str', got {type_mismatch_res.status_code}"
        )
        detail_items = type_mismatch_res.json().get("detail", [])
        assert any(
            isinstance(item, dict)
            and item.get("loc") == ["body", "object_keys"]
            and item.get("type") == "list_type"
            for item in detail_items
        ), f"Expected RequestValidationError on body.object_keys, got {detail_items!r}"

        # 3. Verify zero rows were deleted in PG17
        with ingestion_mgr.scoped_transaction(comp_principal) as tx:
            cf_cnt = tx.execute(
                "SELECT count(*) FROM code_files WHERE domain_id = %s;",
                (eph_domain,),
            ).fetchone()
            dc_cnt = tx.execute(
                "SELECT count(*) FROM doc_chunks WHERE domain_id = %s;",
                (eph_domain,),
            ).fetchone()
            assert cf_cnt is not None and int(str(cf_cnt[0])) == 1
            assert dc_cnt is not None and int(str(dc_cnt[0])) == 1
    finally:
        _cleanup_pg_tenants([tenant_a], domain_ids=[eph_domain])


@pytest.mark.postgres
def test_admin_purge_cross_tenant_object_key_returns_403() -> None:
    """/admin/purge maps cross-tenant object_keys PermissionError to HTTP 403."""
    from context_scout.api.routes.admin import get_admin_purge_coordinator
    from context_scout.ops.artifact_purger import ArtifactPurgeCoordinator
    from context_scout.storage.object_store import InMemoryObjectStore, build_tenant_domain_key

    client = TestClient(app, raise_server_exceptions=False)
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    eph_domain = f"ephem_disc074_{uuid.uuid4().hex[:8]}"
    tok_comp_a = create_signed_principal_token(
        user_id="comp_disc074_a",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["audit:compliance"],
        tenant_id=tenant_a,
    )
    principal_a = PrincipalContext(
        user_id="comp_disc074_a",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["audit:compliance"],
        tenant_id=tenant_a,
    )
    principal_b = PrincipalContext(
        user_id="comp_disc074_b",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["audit:compliance"],
        tenant_id=tenant_b,
    )
    settings = ScoutSettings()
    ingestion_mgr = RLSConnectionManager(settings.db_app_url, pool_role=PoolRole.INGESTION)
    store = InMemoryObjectStore()
    own_obj_key = build_tenant_domain_key(tenant_a, eph_domain, "repos/acme_keep/own.txt")
    cross_obj_key = build_tenant_domain_key(tenant_b, eph_domain, "repos/acme_secret/secret.txt")
    store.put_bytes(own_obj_key, b"own-secret", expected_tenant_id=tenant_a)
    store.put_bytes(cross_obj_key, b"other-tenant-secret", expected_tenant_id=tenant_b)

    try:
        with psycopg.connect(settings.db_admin_url, autocommit=True) as admin_conn:
            for tid in (tenant_a, tenant_b):
                admin_conn.execute(
                    """
                    INSERT INTO domain_packs (
                        domain_id, display_name, lora_adapter_id, primary_truth_source,
                        freshness_half_life_days, config_json, tenant_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s);
                    """,
                    (
                        eph_domain,
                        "Ephemeral Cross-Tenant Purge",
                        "none",
                        "code",
                        180,
                        "{}",
                        tid,
                    ),
                )

        pg_repo = PostgresRLSRepository()
        for p, rname in ((principal_a, "acme/keep-a"), (principal_b, "acme/keep-b")):
            with ingestion_mgr.scoped_transaction(p) as tx:
                pg_repo.insert_code_file(
                    tx,
                    CodeFileRecord(
                        file_id=uuid.uuid4(),
                        tenant_id=p.tenant_id,
                        domain_id=eph_domain,
                        repo_name=rname,
                        file_path="src/keep.py",
                        git_commit_sha="1" * 40,
                        git_blob_sha="2" * 40,
                        language="python",
                        allowed_scopes=["audit:compliance"],
                        raw_content="SECRET_MARKER = 'disc074'\n",
                        line_count=1,
                    ),
                )

        # 1. Default app route (un-overridden PG17 coordinator): cross-tenant object_keys -> 403
        res_default = client.post(
            "/api/v1/admin/purge",
            headers={"Authorization": f"Bearer {tok_comp_a}"},
            json={"domain_id": eph_domain, "object_keys": [cross_obj_key]},
        )
        assert res_default.status_code == 403, (
            f"Expected 403 for cross-tenant object_keys, got {res_default.status_code}: "
            f"{res_default.text}"
        )
        assert "Cross-tenant key denied" in res_default.json()["detail"]

        # 2. Coordinator wired with InMemoryObjectStore: pure and mixed object_keys -> 403
        app.dependency_overrides[get_admin_purge_coordinator] = lambda: ArtifactPurgeCoordinator(
            repository=PostgresRLSRepository(),
            connection_manager=ingestion_mgr,
            object_store=store,
        )
        for bad_keys in ([cross_obj_key], [own_obj_key, cross_obj_key]):
            res_store = client.post(
                "/api/v1/admin/purge",
                headers={"Authorization": f"Bearer {tok_comp_a}"},
                json={"domain_id": eph_domain, "object_keys": bad_keys},
            )
            assert res_store.status_code == 403
            assert "Cross-tenant key denied" in res_store.json()["detail"]

        # 3. Verify PG17 rows and object store entries remain untouched for both tenants
        assert store.exists(own_obj_key, expected_tenant_id=tenant_a)
        assert store.exists(cross_obj_key, expected_tenant_id=tenant_b)
        for p in (principal_a, principal_b):
            with ingestion_mgr.scoped_transaction(p) as tx:
                cnt = tx.execute(
                    "SELECT count(*) FROM code_files WHERE domain_id = %s;",
                    (eph_domain,),
                ).fetchone()
                assert cnt is not None and int(str(cnt[0])) == 1
    finally:
        app.dependency_overrides.pop(get_admin_purge_coordinator, None)
        _cleanup_pg_tenants([tenant_a, tenant_b], domain_ids=[eph_domain])
