"""Milestone 2: Genuine Schemathesis OpenAPI ASGI Fuzzing & Crash Regression Suite."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import time
import uuid
from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from typing import Any

import psycopg
import pytest
import schemathesis
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, settings
from schemathesis import Case, GenerationMode
from schemathesis.checks import (
    content_type_conformance,
    not_a_server_error,
    response_schema_conformance,
    status_code_conformance,
)

from context_scout.api.dependencies import (
    DEFAULT_ROLE_SCOPES,
    create_signed_principal_token,
    get_repository,
)
from context_scout.api.main import app
from context_scout.api.routes.webhooks import set_task_queue
from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.queue.task_queue import PostgresTaskQueue
from context_scout.schemas.contracts import DEFAULT_TENANT_ID, RoleType
from context_scout.security.audit_chain import (
    get_audit_chain_service,
    set_audit_chain_service,
)
from context_scout.security.oidc_iap_auth import get_global_jwks_cache
from context_scout.security.scim import (
    SCIMDirectoryService,
    get_scim_directory_service,
    set_scim_directory_service,
)

_OPENAPI_CONFORMANCE_CHECKS = [
    not_a_server_error,
    status_code_conformance,
    content_type_conformance,
    response_schema_conformance,
]

_FUZZ_SETTINGS = settings(
    max_examples=10,
    deadline=None,
    derandomize=True,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.filter_too_much,
    ],
)

_WEBHOOK_ENV_AND_HEADERS: dict[str, tuple[str, str, str]] = {
    "/api/v1/webhooks/gitlab": (
        "SCOUT_WEBHOOK_GITLAB_SECRET",
        "X-Gitlab-Signature-256",
        "fuzz-gitlab-secret-0123456789abcdef",
    ),
    "/api/v1/webhooks/github": (
        "SCOUT_WEBHOOK_GITHUB_SECRET",
        "X-Hub-Signature-256",
        "fuzz-github-secret-0123456789abcdef",
    ),
    "/api/v1/webhooks/confluence": (
        "SCOUT_WEBHOOK_CONFLUENCE_SECRET",
        "X-Confluence-Signature-256",
        "fuzz-confluence-secret-0123456789abcdef",
    ),
    "/api/v1/webhooks/jira": (
        "SCOUT_WEBHOOK_JIRA_SECRET",
        "X-Jira-Signature-256",
        "fuzz-jira-secret-0123456789abcdef",
    ),
}


_EPHEMERAL_PRINCIPAL_PREFIXES: tuple[str, ...] = (
    "fuzz_eng_",
    "fuzz_comp_",
    "nul_eng_",
    "nul_comp_",
)


def _cleanup_pg_tenants_and_users(
    tenant_ids: set[uuid.UUID],
    *,
    principal_ids: AbstractSet[str] = frozenset(),
) -> None:
    if DEFAULT_TENANT_ID in tenant_ids:
        raise ValueError("Refusing cleanup of DEFAULT_TENANT_ID in Schemathesis tests.")
    for pid in principal_ids:
        if not pid.startswith(_EPHEMERAL_PRINCIPAL_PREFIXES):
            raise ValueError(
                f"Refusing cleanup of non-ephemeral principal_id {pid!r} in Schemathesis tests."
            )

    repo = get_repository()
    if isinstance(repo, InMemoryRLSRepository):
        for tid_str, rec in list(repo._telemetry.items()):
            if rec.tenant_id in tenant_ids or (
                rec.principal_id in principal_ids and rec.tenant_id != DEFAULT_TENANT_ID
            ):
                repo._telemetry.pop(tid_str, None)
    settings_obj = ScoutSettings()
    with psycopg.connect(settings_obj.db_admin_url, autocommit=True) as conn:
        for tid in tenant_ids:
            for table in (
                "execution_telemetry",
                "audit_events",
                "scim_users",
                "scim_groups",
                "indexing_dead_letter",
                "indexing_queue",
                "overlay_file_deltas",
                "ref_snapshots",
                "code_files",
                "doc_chunks",
                "repositories",
                "domain_packs",
            ):
                conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s;", (tid,))
        for pid in principal_ids:
            conn.execute(
                "DELETE FROM execution_telemetry WHERE principal_id = %s AND tenant_id != %s;",
                (pid, DEFAULT_TENANT_ID),
            )


@pytest.fixture
def api_schema(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Configures fail-closed test secrets/JWKS and returns live Schemathesis ASGI schema."""
    monkeypatch.setenv("SCOUT_ENVIRONMENT", "development")
    monkeypatch.setenv("SCOUT_SCIM_BEARER_TOKEN", "fuzz-scim-bearer-token-0123456789")
    monkeypatch.setenv(
        "SCOUT_IAP_EXPECTED_AUDIENCE",
        "/projects/123456789/global/backendServices/scout-fuzz",
    )
    for env_key, _hdr, secret_val in _WEBHOOK_ENV_AND_HEADERS.values():
        monkeypatch.setenv(env_key, secret_val)

    cache = get_global_jwks_cache()
    cache.clear()
    monkeypatch.setattr(cache, "_jwks_loader", lambda _url: {"keys": []})

    prev_scim = get_scim_directory_service()
    prev_audit = get_audit_chain_service()
    set_scim_directory_service(None)
    set_audit_chain_service(None)
    set_task_queue(None)

    try:
        yield schemathesis.openapi.from_asgi("/openapi.json", app)
    finally:
        set_task_queue(None)
        set_scim_directory_service(prev_scim)
        set_audit_chain_service(prev_audit)


unauth_schema = schemathesis.pytest.from_fixture("api_schema")
auth_schema = schemathesis.pytest.from_fixture("api_schema")


@unauth_schema.parametrize()
@_FUZZ_SETTINGS
def test_schemathesis_unauthenticated_zero_5xx_and_zero_auth_bypass(
    case: Case[Any],
) -> None:
    """Fuzzes all 31 OpenAPI operations unauthenticated: 0 5xx and 0 auth bypasses."""
    headers = dict(case.headers or {})
    # Strip/neutralize any Hypothesis-generated credentials that could match test env secrets.
    for key in list(headers.keys()):
        if key.lower() in {
            "authorization",
            "x-goog-iap-jwt-assertion",
            "x-gitlab-token",
            "x-gitlab-signature-256",
            "x-hub-signature-256",
            "x-confluence-secret",
            "x-confluence-signature-256",
            "x-jira-secret",
            "x-jira-signature-256",
        }:
            headers[key] = "invalid-unauth-fuzz-credential"

    response = case.call(headers=headers)
    assert response.status_code < 500, (
        f"5xx on unauth {case.method} {case.path}: {response.status_code} {response.text}"
    )
    if case.path not in {
        "/api/v1/health",
        "/api/v1/health/live",
        "/api/v1/health/ready",
        "/api/v1/auth/session",
    }:
        assert not (200 <= response.status_code < 300), (
            f"Auth bypass on {case.method} {case.path}: {response.status_code}"
        )
    case.validate_response(response, checks=_OPENAPI_CONFORMANCE_CHECKS)


@pytest.mark.postgres
@auth_schema.parametrize()
@_FUZZ_SETTINGS
def test_schemathesis_authenticated_schema_and_status_conformance(
    case: Case[Any],
) -> None:
    """Fuzzes all OpenAPI operations authenticated with valid credentials and cleans PG17 state."""
    tenant_id = uuid.uuid4()
    eng_uid = f"fuzz_eng_{uuid.uuid4().hex[:10]}"
    comp_uid = f"fuzz_comp_{uuid.uuid4().hex[:10]}"
    tracked_tenants: set[uuid.UUID] = {tenant_id}
    tracked_users: set[str] = {eng_uid, comp_uid}

    eng_token = create_signed_principal_token(
        user_id=eng_uid,
        role=RoleType.ENGINEER,
        allowed_scopes=sorted(
            set(DEFAULT_ROLE_SCOPES[RoleType.ENGINEER]) | {"public", "public_eng", "public_docs"}
        ),
        tenant_id=tenant_id,
    )
    comp_token = create_signed_principal_token(
        user_id=comp_uid,
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=sorted(
            set(DEFAULT_ROLE_SCOPES[RoleType.COMPLIANCE_OFFICER])
            | {"audit:compliance", "admin:purge"}
        ),
        tenant_id=tenant_id,
    )

    headers = dict(case.headers or {})
    # Remove untrusted IAP header unless explicitly testing malformed header handling
    for k in list(headers.keys()):
        if k.lower() == "x-goog-iap-jwt-assertion":
            headers.pop(k, None)

    call_kwargs: dict[str, Any] = {}
    if case.path.startswith("/api/v1/scim/v2/"):
        headers["Authorization"] = "Bearer fuzz-scim-bearer-token-0123456789"
        headers["X-Scout-Tenant-Id"] = str(tenant_id)
    elif case.path in _WEBHOOK_ENV_AND_HEADERS:
        from schemathesis.core import NOT_SET

        _env_key, sig_hdr, secret_val = _WEBHOOK_ENV_AND_HEADERS[case.path]
        raw_body = b"" if case.body is NOT_SET else json.dumps(case.body).encode("utf-8")
        sig = "sha256=" + hmac.new(secret_val.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        headers[sig_hdr] = sig
        headers["Content-Type"] = "application/json"
        headers["X-Scout-Tenant-Id"] = str(tenant_id)
        call_kwargs["data"] = raw_body
    elif case.path.startswith("/api/v1/admin/") or case.path.startswith("/api/v1/webhooks/"):
        headers["Authorization"] = f"Bearer {comp_token}"
    else:
        headers["Authorization"] = f"Bearer {eng_token}"

    if case.path == "/api/v1/auth/session" and isinstance(case.body, dict):
        raw_tid = case.body.get("tenant_id")
        if isinstance(raw_tid, str):
            with contextlib.suppress(ValueError):
                parsed_tid = uuid.UUID(raw_tid)
                if parsed_tid != DEFAULT_TENANT_ID:
                    tracked_tenants.add(parsed_tid)

    try:
        response = case.call(headers=headers, **call_kwargs)
        assert response.status_code < 500, (
            f"5xx on authed {case.method} {case.path}: {response.status_code} {response.text}"
        )
        case.validate_response(response, checks=_OPENAPI_CONFORMANCE_CHECKS)
    finally:
        _cleanup_pg_tenants_and_users(tracked_tenants, principal_ids=tracked_users)


@pytest.mark.postgres
def test_schemathesis_positive_and_negative_modes_explicit(
    api_schema: Any,
) -> None:
    """Verifies Schemathesis POSITIVE + NEGATIVE generation mode execution on key endpoints."""
    from hypothesis import given

    assert GenerationMode.POSITIVE in GenerationMode
    assert GenerationMode.NEGATIVE in GenerationMode
    ops = list(api_schema.get_all_operations())
    assert len(ops) >= 30

    tenant_id = uuid.uuid4()
    eng_uid = f"fuzz_eng_{uuid.uuid4().hex[:8]}"
    eng_tok = create_signed_principal_token(
        user_id=eng_uid,
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
        tenant_id=tenant_id,
    )
    comp_uid = f"fuzz_comp_{uuid.uuid4().hex[:8]}"
    comp_tok = create_signed_principal_token(
        user_id=comp_uid,
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=sorted(
            set(DEFAULT_ROLE_SCOPES[RoleType.COMPLIANCE_OFFICER])
            | {"audit:compliance", "admin:purge"}
        ),
        tenant_id=tenant_id,
    )
    target_paths = [
        ("/api/v1/auth/session", "POST", {}),
        ("/api/v1/scout/query", "POST", {"Authorization": f"Bearer {eng_tok}"}),
        (
            "/api/v1/scim/v2/Users",
            "POST",
            {
                "Authorization": "Bearer fuzz-scim-bearer-token-0123456789",
                "X-Scout-Tenant-Id": str(tenant_id),
            },
        ),
        (
            "/api/v1/webhooks/stats",
            "GET",
            {"Authorization": f"Bearer {comp_tok}"},
        ),
        (
            "/api/v1/admin/audit/verify",
            "GET",
            {"Authorization": f"Bearer {comp_tok}"},
        ),
    ]
    tracked_tenants: set[uuid.UUID] = {tenant_id}
    try:
        for path, method, extra_headers in target_paths:
            op = api_schema[path][method]
            for mode in (GenerationMode.POSITIVE, GenerationMode.NEGATIVE):

                def _make_runner(
                    operation: Any,
                    hdrs_override: dict[str, str],
                    gen_mode: GenerationMode,
                ) -> Any:
                    @given(case=operation.as_strategy(generation_mode=gen_mode))
                    @settings(
                        max_examples=3,
                        deadline=None,
                        database=None,
                        suppress_health_check=[
                            HealthCheck.too_slow,
                            HealthCheck.function_scoped_fixture,
                            HealthCheck.filter_too_much,
                        ],
                    )
                    def _run_case(case: Case[Any]) -> None:
                        hdrs = dict(case.headers or {})
                        hdrs.update(hdrs_override)
                        if case.path == "/api/v1/auth/session" and isinstance(case.body, dict):
                            raw_tid = case.body.get("tenant_id")
                            if isinstance(raw_tid, str):
                                with contextlib.suppress(ValueError):
                                    parsed_tid = uuid.UUID(raw_tid)
                                    if parsed_tid != DEFAULT_TENANT_ID:
                                        tracked_tenants.add(parsed_tid)
                        resp = case.call(headers=hdrs)
                        assert resp.status_code < 500, (
                            f"5xx in {gen_mode.name} mode for {case.method} {case.path}: "
                            f"{resp.status_code} {resp.text}"
                        )
                        case.validate_response(resp, checks=_OPENAPI_CONFORMANCE_CHECKS)

                    return _run_case

                _make_runner(op, extra_headers, mode)()
    finally:
        _cleanup_pg_tenants_and_users(tracked_tenants, principal_ids={eng_uid, comp_uid})


# =============================================================================
# Regression Tests for Discovered FastAPI 500 Crash Bugs
# =============================================================================


@pytest.mark.postgres
def test_webhook_malformed_types_nul_bytes_and_pg17_fk_return_400_never_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: malformed webhook project/repository, priority, NUL, and FK return 400."""
    secret = "webhook-regression-secret-0123456789"
    monkeypatch.setenv("SCOUT_WEBHOOK_GITLAB_SECRET", secret)
    monkeypatch.setenv("SCOUT_WEBHOOK_GITHUB_SECRET", secret)
    monkeypatch.setenv("SCOUT_WEBHOOK_JIRA_SECRET", secret)
    monkeypatch.setenv("SCOUT_WEBHOOK_CONFLUENCE_SECRET", secret)

    settings_obj = ScoutSettings()
    pg_queue = PostgresTaskQueue(settings_obj.db_app_url)
    set_task_queue(pg_queue)
    tenant_id = uuid.uuid4()
    client = TestClient(app)

    def _signed_post(vendor: str, sig_hdr: str, payload: dict[str, Any]) -> Any:
        raw = json.dumps(payload).encode("utf-8")
        sig = "sha256=" + hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
        return client.post(
            f"/api/v1/webhooks/{vendor}",
            content=raw,
            headers={
                sig_hdr: sig,
                "X-Scout-Tenant-Id": str(tenant_id),
                "Content-Type": "application/json",
            },
        )

    try:
        # 1. Non-dict project / repository must return 400 (previously AttributeError 500)
        assert (
            _signed_post("gitlab", "X-Gitlab-Signature-256", {"project": None}).status_code == 400
        )
        assert (
            _signed_post("gitlab", "X-Gitlab-Signature-256", {"project": ["bad"]}).status_code
            == 400
        )
        assert (
            _signed_post("github", "X-Hub-Signature-256", {"repository": "not-a-dict"}).status_code
            == 400
        )

        # 2. Invalid/bool/overflow priority must return 400 (previously ValueError/DataError 500)
        for bad_prio in ("not_an_int", True, None, {"x": 1}, 10**50):
            res = _signed_post(
                "jira",
                "X-Jira-Signature-256",
                {"domain_id": "code_architecture_scout", "priority": bad_prio},
            )
            assert res.status_code == 400, f"Expected 400 for priority={bad_prio!r}, got {res.text}"

        # 3. NUL bytes inside webhook payload must return 400
        assert (
            _signed_post(
                "confluence",
                "X-Confluence-Signature-256",
                {"domain_id": "code_architecture_scout", "nested": {"msg": "bad\x00value"}},
            ).status_code
            == 400
        )

        # 4. Unknown domain_id hitting PG17 ForeignKeyViolation must return 400 (previously 500)
        fk_res = _signed_post(
            "gitlab",
            "X-Gitlab-Signature-256",
            {"domain_id": "nonexistent_fk_domain_xyz"},
        )
        assert fk_res.status_code == 400
    finally:
        set_task_queue(None)
        _cleanup_pg_tenants_and_users({tenant_id})


@pytest.mark.postgres
def test_scim_malformed_scopes_operations_and_nul_bytes_return_400_never_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: non-list SCIM scopes, non-dict Operations, and NUL bytes return 400."""
    scim_tok = "scim-regression-secret-0123456789"
    monkeypatch.setenv("SCOUT_SCIM_BEARER_TOKEN", scim_tok)
    settings_obj = ScoutSettings()
    tenant_id = uuid.uuid4()

    from context_scout.db.session import RLSConnectionManager

    scim_svc = SCIMDirectoryService(
        connection_manager=RLSConnectionManager(settings_obj.db_app_url)
    )
    prev_scim = get_scim_directory_service()
    set_scim_directory_service(scim_svc)
    client = TestClient(app)
    hdrs = {"Authorization": f"Bearer {scim_tok}", "X-Scout-Tenant-Id": str(tenant_id)}

    try:
        # Non-list customScopes / mappedScopes -> 400 (previously TypeError 500)
        assert (
            client.post(
                "/api/v1/scim/v2/Users",
                headers=hdrs,
                json={"userName": "alice@acme.local", "customScopes": 123},
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/v1/scim/v2/Groups",
                headers=hdrs,
                json={"displayName": "Team", "mappedScopes": "not-a-list"},
            ).status_code
            == 400
        )

        # Create valid user & group, then patch with non-dict Operations item -> 400 (not 500)
        assert (
            client.post(
                "/api/v1/scim/v2/Users",
                headers=hdrs,
                json={"id": "scim-u1", "userName": "alice@acme.local"},
            ).status_code
            == 201
        )
        assert (
            client.post(
                "/api/v1/scim/v2/Groups",
                headers=hdrs,
                json={"id": "scim-g1", "displayName": "Eng Team"},
            ).status_code
            == 201
        )
        scim_users_base = "/api/v1/scim/v2/" + "Users"
        assert (
            client.patch(
                f"{scim_users_base}/scim-u1",
                headers=hdrs,
                json={"Operations": ["not-a-dict"]},
            ).status_code
            == 400
        )
        assert (
            client.patch(
                "/api/v1/scim/v2/Groups/scim-g1",
                headers=hdrs,
                json={"Operations": [None]},
            ).status_code
            == 400
        )

        # NUL bytes in SCIM path/payload -> 400 (never PG17 DataError 500)
        assert client.get(f"{scim_users_base}/bad%00id", headers=hdrs).status_code == 400
        assert client.delete("/api/v1/scim/v2/Groups/bad%00id", headers=hdrs).status_code == 400
        assert (
            client.post(
                "/api/v1/scim/v2/Users",
                headers=hdrs,
                json={"userName": "bad\x00user@acme.local"},
            ).status_code
            == 400
        )
    finally:
        set_scim_directory_service(prev_scim)
        _cleanup_pg_tenants_and_users({tenant_id})


@pytest.mark.postgres
def test_nul_bytes_across_auth_admin_scout_and_handoff_return_4xx_never_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: NUL bytes across auth/admin/scout/handoff endpoints return 400/422."""
    tenant_id = uuid.uuid4()
    comp_uid = f"nul_comp_{uuid.uuid4().hex[:8]}"
    eng_uid = f"nul_eng_{uuid.uuid4().hex[:8]}"
    comp_tok = create_signed_principal_token(
        user_id=comp_uid,
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["audit:compliance", "admin:purge"],
        tenant_id=tenant_id,
    )
    eng_tok = create_signed_principal_token(
        user_id=eng_uid,
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
        tenant_id=tenant_id,
    )
    client = TestClient(app)

    try:
        # Prod /auth/session with NUL user_id -> 422/400 (never 500 in audit_chain)
        monkeypatch.setenv("SCOUT_ENVIRONMENT", "production")
        assert client.post(
            "/api/v1/auth/session",
            json={"user_id": "evil\x00user", "tenant_id": str(tenant_id)},
        ).status_code in {400, 422}
        monkeypatch.setenv("SCOUT_ENVIRONMENT", "development")

        # /admin/purge & /admin/audit/export with NUL
        assert client.post(
            "/api/v1/admin/purge",
            headers={"Authorization": f"Bearer {comp_tok}"},
            json={"domain_id": "bad\x00domain"},
        ).status_code in {400, 422}
        assert (
            client.get(
                "/api/v1/admin/audit/export",
                headers={"Authorization": f"Bearer {comp_tok}"},
                params={"domain_id": "bad\x00domain"},
            ).status_code
            == 400
        )

        # /scout/query & /scout/stream with NUL
        for path in ("/api/v1/scout/query", "/api/v1/scout/stream"):
            assert client.post(
                path,
                headers={"Authorization": f"Bearer {eng_tok}"},
                json={"query": "hello\x00world", "domain_id": "code_architecture_scout"},
            ).status_code in {400, 422}

        # /handoff/preview with NUL
        pack_dict = {
            "trace_id": "trace_nul\x00test",
            "domain_id": "code_architecture_scout",
            "user_task": "Task",
            "architectural_execution_plan": ["Plan"],
            "code_slices": [],
            "supporting_doc_citations": [],
            "preflight_report": {
                "passed": True,
                "normalized_rrf_score": 0.9,
                "citations_verified_100_pct": True,
            },
            "local_tokens_consumed": 100,
            "pack_token_count": 50,
        }
        assert client.post(
            "/api/v1/handoff/preview",
            headers={"Authorization": f"Bearer {eng_tok}"},
            json={"pack": pack_dict},
        ).status_code in {400, 422}
    finally:
        _cleanup_pg_tenants_and_users({tenant_id}, principal_ids={comp_uid, eng_uid})


@pytest.mark.postgres
def test_cleanup_pg_tenants_and_users_protects_default_tenant_and_canary_rows() -> None:
    """Regression: Schemathesis cleanup refuses DEFAULT_TENANT_ID and preserves canary rows."""
    with pytest.raises(ValueError, match="DEFAULT_TENANT_ID"):
        _cleanup_pg_tenants_and_users({DEFAULT_TENANT_ID})
    with pytest.raises(ValueError, match="non-ephemeral principal_id"):
        _cleanup_pg_tenants_and_users({uuid.uuid4()}, principal_ids={"e2e_smoke_eng"})

    settings_obj = ScoutSettings()
    ephemeral_tid = uuid.uuid4()
    canary_trace_id = uuid.uuid4()
    ephemeral_trace_id = uuid.uuid4()

    with psycopg.connect(settings_obj.db_admin_url, autocommit=True) as conn:
        packs_before_row = conn.execute(
            "SELECT count(*) FROM domain_packs WHERE tenant_id = %s;",
            (DEFAULT_TENANT_ID,),
        ).fetchone()
        packs_before = int(packs_before_row[0]) if packs_before_row is not None else 0
        conn.execute(
            """
            INSERT INTO execution_telemetry (
                trace_id, tenant_id, domain_id, principal_id, principal_role, execution_mode,
                local_tokens_read, context_pack_tokens_sent, paid_tokens_saved,
                estimated_usd_saved, preflight_passed_first_try, zero_touch_sufficient,
                missing_context_requests, latency_ms
            )
            VALUES (%s, %s, %s, %s, %s, %s, 100, 50, 50, 0.01, TRUE, TRUE, '[]'::jsonb, 10);
            """,
            (
                canary_trace_id,
                DEFAULT_TENANT_ID,
                "code_architecture_scout",
                "e2e_smoke_eng",
                "engineer",
                "chat_qa",
            ),
        )
        conn.execute(
            """
            INSERT INTO execution_telemetry (
                trace_id, tenant_id, domain_id, principal_id, principal_role, execution_mode,
                local_tokens_read, context_pack_tokens_sent, paid_tokens_saved,
                estimated_usd_saved, preflight_passed_first_try, zero_touch_sufficient,
                missing_context_requests, latency_ms
            )
            VALUES (%s, %s, %s, %s, %s, %s, 100, 50, 50, 0.01, TRUE, TRUE, '[]'::jsonb, 10);
            """,
            (
                ephemeral_trace_id,
                ephemeral_tid,
                "code_architecture_scout",
                "fuzz_eng_test",
                "engineer",
                "chat_qa",
            ),
        )

    try:
        _cleanup_pg_tenants_and_users({ephemeral_tid}, principal_ids={"fuzz_eng_test"})
        with psycopg.connect(settings_obj.db_admin_url, autocommit=True) as conn:
            eph_row = conn.execute(
                "SELECT count(*) FROM execution_telemetry WHERE trace_id = %s;",
                (ephemeral_trace_id,),
            ).fetchone()
            canary_row = conn.execute(
                "SELECT count(*) FROM execution_telemetry WHERE trace_id = %s;",
                (canary_trace_id,),
            ).fetchone()
            packs_after_row = conn.execute(
                "SELECT count(*) FROM domain_packs WHERE tenant_id = %s;",
                (DEFAULT_TENANT_ID,),
            ).fetchone()
            assert eph_row is not None and int(eph_row[0]) == 0
            assert canary_row is not None and int(canary_row[0]) == 1
            assert packs_after_row is not None and int(packs_after_row[0]) == packs_before
    finally:
        with psycopg.connect(settings_obj.db_admin_url, autocommit=True) as conn:
            conn.execute(
                "DELETE FROM execution_telemetry WHERE trace_id = %s;",
                (canary_trace_id,),
            )
            conn.execute(
                "DELETE FROM execution_telemetry WHERE trace_id = %s;",
                (ephemeral_trace_id,),
            )


def test_mounted_mcp_http_get_405_well_known_200_and_duplicate_header_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: mounted /mcp/* GET returns 405 (never 500) and rejects duplicate headers."""
    monkeypatch.setenv("SCOUT_ENVIRONMENT", "development")
    tenant_id = uuid.uuid4()
    tok = create_signed_principal_token(
        user_id=f"fuzz_eng_{uuid.uuid4().hex[:8]}",
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
        tenant_id=tenant_id,
    )
    valid_post_body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    def _run_checks(client: TestClient) -> None:
        for mount in ("/mcp/code", "/mcp/docs"):
            assert client.get(f"{mount}/mcp").status_code == 401
            for accept_hdr in ({}, {"Accept": "text/event-stream"}):
                get_res = client.get(
                    f"{mount}/mcp",
                    headers={"Authorization": f"Bearer {tok}", **accept_hdr},
                )
                assert get_res.status_code == 405
                assert get_res.headers.get("allow") == "POST"

            del_res = client.delete(f"{mount}/mcp", headers={"Authorization": f"Bearer {tok}"})
            assert del_res.status_code == 405
            assert client.get(f"{mount}/.well-known/mcp").status_code == 200

            dup_iap = client.post(
                f"{mount}/mcp",
                content=valid_post_body,
                headers=[
                    ("content-type", "application/json"),
                    ("accept", "application/json, text/event-stream"),
                    ("x-goog-iap-jwt-assertion", "invalid"),
                    ("x-goog-iap-jwt-assertion", "second-invalid"),
                ],
            )
            assert dup_iap.status_code == 401

        assert client.get("/.well-known/mcp").status_code == 200

        dup_auth = client.post(
            "/mcp/code/mcp",
            content=valid_post_body,
            headers=[
                ("content-type", "application/json"),
                ("accept", "application/json, text/event-stream"),
                ("authorization", "Bearer invalid"),
                ("authorization", f"Bearer {tok}"),
            ],
        )
        assert dup_auth.status_code == 401

        for ctx_hdr, v1, v2 in (
            ("x-principal-tenant-id", str(tenant_id), str(uuid.uuid4())),
            ("mcp-protocol-version", "2026-07-28", "2025-06-18"),
        ):
            dup_ctx = client.post(
                "/mcp/code/mcp",
                content=valid_post_body,
                headers=[
                    ("content-type", "application/json"),
                    ("accept", "application/json, text/event-stream"),
                    ("authorization", f"Bearer {tok}"),
                    (ctx_hdr, v1),
                    (ctx_hdr, v2),
                ],
            )
            assert dup_ctx.status_code == 400

        ok_post = client.post(
            "/mcp/code/mcp",
            content=valid_post_body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {tok}",
            },
        )
        assert ok_post.status_code == 200
        assert "tools" in ok_post.text

    _run_checks(TestClient(app))
    with TestClient(app) as lifespan_client:
        _run_checks(lifespan_client)


@pytest.mark.postgres
def test_lone_utf16_surrogates_return_4xx_never_500_across_all_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: lone UTF-16 surrogates return 400/422 (never 500) across API routes."""
    monkeypatch.setenv("SCOUT_ENVIRONMENT", "development")
    scim_tok = "surrogate-scim-secret-0123456789"
    wh_secret = "surrogate-webhook-secret-0123456789"
    monkeypatch.setenv("SCOUT_SCIM_BEARER_TOKEN", scim_tok)
    for env_key, _hdr, _sec in _WEBHOOK_ENV_AND_HEADERS.values():
        monkeypatch.setenv(env_key, wh_secret)

    tenant_id = uuid.uuid4()
    comp_uid = f"nul_comp_{uuid.uuid4().hex[:8]}"
    eng_uid = f"nul_eng_{uuid.uuid4().hex[:8]}"
    comp_tok = create_signed_principal_token(
        user_id=comp_uid,
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["audit:compliance", "admin:purge"],
        tenant_id=tenant_id,
    )
    eng_tok = create_signed_principal_token(
        user_id=eng_uid,
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
        tenant_id=tenant_id,
    )

    client = TestClient(app, raise_server_exceptions=False)
    prev_scim = get_scim_directory_service()
    prev_audit = get_audit_chain_service()
    from context_scout.db.session import RLSConnectionManager
    from context_scout.queue.task_queue import InMemoryTaskQueue

    try:
        # 1. Pydantic-validated endpoints with raw JSON lone surrogates
        for raw_json in (
            b'{"user_id":"a\\ud800b"}',
            b'{"user_id":"ok","allowed_scopes":["a\\ud800b"]}',
        ):
            r = client.post(
                "/api/v1/auth/session",
                content=raw_json,
                headers={"Content-Type": "application/json"},
            )
            assert r.status_code in {400, 422}

        for path in ("/api/v1/scout/query", "/api/v1/scout/stream"):
            r = client.post(
                path,
                content=b'{"query":"hi \\ud800","domain_id":"code_architecture_scout"}',
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {eng_tok}",
                },
            )
            assert r.status_code in {400, 422}

        assert client.post(
            "/api/v1/admin/purge",
            content=b'{"domain_id":"d\\ud800"}',
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {comp_tok}",
            },
        ).status_code in {400, 422}

        # /handoff/preview (user_task, model_id, supporting_doc_citations) & /handoff/apply
        for body_bytes in (
            (
                b'{"pack":{"trace_id":"t1","domain_id":"code_architecture_scout",'
                b'"user_task":"bad\\ud800task","architectural_execution_plan":["p"],'
                b'"code_slices":[],"supporting_doc_citations":[],'
                b'"preflight_report":{"passed":true,"normalized_rrf_score":0.9,'
                b'"citations_verified_100_pct":true},"local_tokens_consumed":10,'
                b'"pack_token_count":5}}'
            ),
            (
                b'{"pack":{"trace_id":"t2","domain_id":"code_architecture_scout",'
                b'"user_task":"ok","architectural_execution_plan":["p"],'
                b'"code_slices":[],"supporting_doc_citations":[],'
                b'"preflight_report":{"passed":true,"normalized_rrf_score":0.9,'
                b'"citations_verified_100_pct":true},"local_tokens_consumed":10,'
                b'"pack_token_count":5},"model_id":"m\\ud800"}'
            ),
            (
                b'{"pack":{"trace_id":"t3","domain_id":"code_architecture_scout",'
                b'"user_task":"ok","architectural_execution_plan":["p"],'
                b'"code_slices":[],"supporting_doc_citations":[{'
                b'"chunk_id":"11111111-1111-1111-1111-111111111111","doc_id":"d1",'
                b'"doc_title":"T","source_uri":"u","section_heading":"S",'
                b'"verbatim_quote":"ok","full_section_content":"bad\\ud800"}'
                b'],"preflight_report":{"passed":true,"normalized_rrf_score":0.9,'
                b'"citations_verified_100_pct":true},"local_tokens_consumed":10,'
                b'"pack_token_count":5}}'
            ),
        ):
            r = client.post(
                "/api/v1/handoff/preview",
                content=body_bytes,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {eng_tok}",
                },
            )
            assert r.status_code in {400, 422}

        assert client.post(
            "/api/v1/handoff/apply",
            content=b'{"confirmation_token":"tok\\udfff","confirmed":true}',
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {eng_tok}",
            },
        ).status_code in {400, 422}

        # 2. Webhooks (both InMemoryTaskQueue and PostgresTaskQueue)
        settings_obj = ScoutSettings()
        for queue_impl in (InMemoryTaskQueue(), PostgresTaskQueue(settings_obj.db_app_url)):
            set_task_queue(queue_impl)
            for wh_path, (_ek, sig_hdr, _sec) in _WEBHOOK_ENV_AND_HEADERS.items():
                for raw_wh in (
                    b'{"domain_id":"d\\ud800"}',
                    b'{"domain_id":"code_architecture_scout","nested":{"k":"v\\udfff"}}',
                ):
                    sig = (
                        "sha256="
                        + hmac.new(wh_secret.encode("utf-8"), raw_wh, hashlib.sha256).hexdigest()
                    )
                    res_wh = client.post(
                        wh_path,
                        content=raw_wh,
                        headers={
                            "Content-Type": "application/json",
                            sig_hdr: sig,
                            "X-Scout-Tenant-Id": str(tenant_id),
                        },
                    )
                    assert res_wh.status_code == 400

        # 3. SCIM (both in-memory and PG17 SCIMDirectoryService)
        scim_hdrs = {"Authorization": f"Bearer {scim_tok}", "X-Scout-Tenant-Id": str(tenant_id)}
        for scim_svc in (
            SCIMDirectoryService(),
            SCIMDirectoryService(connection_manager=RLSConnectionManager(settings_obj.db_app_url)),
        ):
            set_scim_directory_service(scim_svc)
            for raw_user in (
                b'{"userName":"a\\ud800b"}',
                b'{"userName":"alice","displayName":"a\\ud800b"}',
                b'{"userName":"alice","customScopes":["a\\ud800b"]}',
                b'{"userName":"alice","metadata":{"k":"a\\ud800b"}}',
            ):
                assert (
                    client.post(
                        "/api/v1/scim/v2/Users",
                        content=raw_user,
                        headers={"Content-Type": "application/json", **scim_hdrs},
                    ).status_code
                    == 400
                )
            for raw_grp in (
                b'{"displayName":"g\\ud800"}',
                b'{"displayName":"grp","mappedScopes":["s\\udfff"]}',
                b'{"displayName":"grp","metadata":{"k":"v\\ud800"}}',
            ):
                assert (
                    client.post(
                        "/api/v1/scim/v2/Groups",
                        content=raw_grp,
                        headers={"Content-Type": "application/json", **scim_hdrs},
                    ).status_code
                    == 400
                )
    finally:
        set_task_queue(None)
        set_scim_directory_service(prev_scim)
        set_audit_chain_service(prev_audit)
        _cleanup_pg_tenants_and_users({tenant_id}, principal_ids={comp_uid, eng_uid})


def test_webhook_nan_inf_unicode_priority_and_duplicate_timestamp_headers_return_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: NaN/inf/underscore timestamps, duplicates, and Unicode priority return 400."""
    from fastapi import HTTPException
    from starlette.requests import Request as StarletteRequest

    from context_scout.api.routes.webhooks import _validate_optional_replay_headers
    from context_scout.queue.task_queue import InMemoryTaskQueue

    secret = "wh-nan-prio-regression-secret-0123456"
    monkeypatch.setenv("SCOUT_WEBHOOK_GITHUB_SECRET", secret)
    monkeypatch.setenv("SCOUT_WEBHOOK_GITLAB_SECRET", secret)
    set_task_queue(InMemoryTaskQueue())
    client = TestClient(app, raise_server_exceptions=False)

    try:
        valid_body = json.dumps({"domain_id": "code_architecture_scout"}).encode("utf-8")
        valid_sig = (
            "sha256=" + hmac.new(secret.encode("utf-8"), valid_body, hashlib.sha256).hexdigest()
        )
        for bad_ts in ("NaN", "nan", "-nan", "inf", "-Infinity", "1e309", "1_000"):
            for ts_hdr in ("X-Webhook-Timestamp", "X-Scout-Webhook-Timestamp"):
                res = client.post(
                    "/api/v1/webhooks/github",
                    content=valid_body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Hub-Signature-256": valid_sig,
                        ts_hdr: bad_ts,
                    },
                )
                assert res.status_code == 400, f"{ts_hdr}={bad_ts!r} returned {res.status_code}"

        # Non-ASCII fullwidth digits rejected by _validate_optional_replay_headers (ASCII-only)
        req_unicode_ts = StarletteRequest(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/v1/webhooks/github",
                "headers": [
                    (b"x-webhook-timestamp", "\uff11\uff12\uff13".encode()),
                ],
            }
        )
        with pytest.raises(HTTPException) as exc_ts:
            _validate_optional_replay_headers(req_unicode_ts, "github")
        assert exc_ts.value.status_code == 400

        dup_ts_res = client.post(
            "/api/v1/webhooks/github",
            content=valid_body,
            headers=[
                ("content-type", "application/json"),
                ("x-hub-signature-256", valid_sig),
                ("x-scout-webhook-timestamp", str(int(time.time()))),
                ("x-webhook-timestamp", str(int(time.time()) - 10)),
            ],
        )
        assert dup_ts_res.status_code == 400

        prio_body = json.dumps(
            {"domain_id": "code_architecture_scout", "priority": "\u00b2"}
        ).encode("utf-8")
        prio_sig = (
            "sha256=" + hmac.new(secret.encode("utf-8"), prio_body, hashlib.sha256).hexdigest()
        )
        prio_res = client.post(
            "/api/v1/webhooks/gitlab",
            content=prio_body,
            headers={"Content-Type": "application/json", "X-Gitlab-Signature-256": prio_sig},
        )
        assert prio_res.status_code == 400
    finally:
        set_task_queue(None)


def test_handoff_preview_rejects_nul_in_nested_citations_and_brief() -> None:
    """Regression: NUL bytes in supporting_doc_citations or domain_knowledge_brief fail 422."""
    tenant_id = uuid.uuid4()
    eng_tok = create_signed_principal_token(
        user_id=f"nul_eng_{uuid.uuid4().hex[:8]}",
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
        tenant_id=tenant_id,
    )
    client = TestClient(app)

    base_pack: dict[str, Any] = {
        "trace_id": "trace_nested_nul_ok",
        "domain_id": "code_architecture_scout",
        "user_task": "Task",
        "architectural_execution_plan": ["Plan"],
        "code_slices": [],
        "supporting_doc_citations": [
            {
                "chunk_id": str(uuid.uuid4()),
                "doc_id": "doc-1",
                "doc_title": "Title",
                "source_uri": "https://docs.internal/doc",
                "section_heading": "Sec",
                "verbatim_quote": "Quote",
                "full_section_content": "Full content",
            }
        ],
        "domain_knowledge_brief": {
            "domain_id": "code_architecture_scout",
            "brief_title": "Brief Title",
            "executive_summary": "Summary",
            "verified_citations": [],
        },
        "preflight_report": {
            "passed": True,
            "normalized_rrf_score": 0.9,
            "citations_verified_100_pct": True,
        },
        "local_tokens_consumed": 100,
        "pack_token_count": 50,
    }

    for field in ("verbatim_quote", "source_uri", "full_section_content"):
        mutated = json.loads(json.dumps(base_pack))
        mutated["supporting_doc_citations"][0][field] = f"bad\x00{field}"
        res = client.post(
            "/api/v1/handoff/preview",
            headers={"Authorization": f"Bearer {eng_tok}"},
            json={"pack": mutated},
        )
        assert res.status_code == 422

    mutated_brief = json.loads(json.dumps(base_pack))
    mutated_brief["domain_knowledge_brief"]["brief_title"] = "bad\x00brief"
    res_brief = client.post(
        "/api/v1/handoff/preview",
        headers={"Authorization": f"Bearer {eng_tok}"},
        json={"pack": mutated_brief},
    )
    assert res_brief.status_code == 422
