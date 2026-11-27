"""Unscripted multi-domain semantic & $0 abstention evaluation suite.

Runs against live PostgreSQL 17 and FastAPI TestClient across all 4 domains
(`data_engineering_scout`, `polyglot_unseen_v2`, `enterprise_docs_scout`,
`code_architecture_scout`), `federated_all` cross-domain routing, and `$0`
out-of-corpus abstention traps.

- When `CONTEXT_SCOUT_USE_LIVE_MLX=1` is set: verifies full live `Gemma-4`
  synthesis (`passed is True` on answerable queries, `is_abstention is True`
  and `paid_tokens_spent == 0` on abstention traps).
- When `CONTEXT_SCOUT_USE_LIVE_MLX` is unset (CI / `make test-full`): verifies
  live PG17 retrieval, domain routing, PreFlight grounding, and `$0`
  out-of-corpus abstention without skipping any test.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from context_scout.api import dependencies
from context_scout.api.main import create_app

pytestmark = pytest.mark.postgres

_ANSWERABLE_CASES: list[tuple[str, str, str, tuple[str, ...]]] = [
    (
        "de_acid_short",
        "data_engineering_scout",
        "What is ACID",
        ("data_engineering_scout",),
    ),
    (
        "de_acid_full",
        "data_engineering_scout",
        "What is ACID in database transactions?",
        ("data_engineering_scout",),
    ),
    (
        "de_cap",
        "data_engineering_scout",
        "Explain CAP theorem",
        ("data_engineering_scout",),
    ),
    (
        "de_iceberg",
        "data_engineering_scout",
        "What is Apache Iceberg?",
        ("data_engineering_scout",),
    ),
    (
        "poly_httpx_pool",
        "polyglot_unseen_v2",
        "How are HTTPX connection pooling limits and keepalive expirations configured?",
        ("polyglot_unseen_v2",),
    ),
    (
        "poly_asgi_lifespan",
        "polyglot_unseen_v2",
        "How does Starlette handle ASGI lifespan startup and shutdown events?",
        ("polyglot_unseen_v2",),
    ),
    (
        "docs_force_rls",
        "enterprise_docs_scout",
        "How does PostgreSQL 17 enforce FORCE ROW LEVEL SECURITY and SET LOCAL tenant isolation?",
        ("enterprise_docs_scout",),
    ),
    (
        "docs_preflight_gate",
        "enterprise_docs_scout",
        (
            "How do the 6 deterministic checks in the Pre-Flight Sufficiency Gate "
            "prevent hallucinations?"
        ),
        ("enterprise_docs_scout",),
    ),
    (
        "code_rls_session",
        "code_architecture_scout",
        "How does RLSConnectionManager enforce SET LOCAL tenant_id and NOBYPASSRLS in session.py?",
        ("code_architecture_scout",),
    ),
    (
        "code_adapter_router",
        "code_architecture_scout",
        "How does AdapterRouter enforce training_scope_ceiling and fail closed to base weights?",
        ("code_architecture_scout",),
    ),
    (
        "fed_cross_domain",
        "federated_all",
        (
            "How does RLSConnectionManager enforce PostgreSQL 17 FORCE ROW LEVEL SECURITY "
            "and SET LOCAL tenant isolation?"
        ),
        ("federated_all", "code_architecture_scout", "enterprise_docs_scout"),
    ),
]

_ABSTENTION_CASES: list[tuple[str, str, str]] = [
    (
        "abstain_mars_rover",
        "data_engineering_scout",
        "What telemetry protocol did NASA Mars Rover Perseverance use on Sol 412?",
    ),
    (
        "abstain_sourdough",
        "enterprise_docs_scout",
        "What hydration percentage and oven steam schedule makes the best sourdough boule?",
    ),
    (
        "abstain_federated_oos",
        "federated_all",
        "What are the 2026 UEFA Champions League quarterfinal tie-breaker rules?",
    ),
]


@pytest.fixture(scope="module")
def api_client() -> Generator[tuple[TestClient, dict[str, str]], None, None]:
    prev_repo = dependencies._shared_repo
    prev_client = dependencies._local_gemma_client
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(
            "SCOUT_AUTH_HMAC_SECRET",
            "scout-test-suite-hmac-secret-32-chars-long-sec04!!",
        )
        mp.setenv("SCOUT_REQUIRE_SIGNED_AUTH", "true")
        dependencies._shared_repo = None
        dependencies._local_gemma_client = None
        try:
            with TestClient(create_app()) as client:
                auth_resp = client.post(
                    "/api/v1/auth/session",
                    json={
                        "user_id": "unscripted_eval_eng",
                        "role": "engineer",
                        "allowed_scopes": [
                            "public_eng",
                            "public_docs",
                            "public_de",
                            "public_polyglot",
                        ],
                    },
                )
                assert auth_resp.status_code == 200, auth_resp.text
                token = str(auth_resp.json()["access_token"])
                yield client, {"Authorization": f"Bearer {token}"}
        finally:
            dependencies._shared_repo = prev_repo
            dependencies._local_gemma_client = prev_client


@pytest.mark.parametrize(
    ("case_id", "requested_domain", "query", "expected_domains"),
    _ANSWERABLE_CASES,
    ids=[c[0] for c in _ANSWERABLE_CASES],
)
def test_unscripted_answerable_multidomain(
    api_client: tuple[TestClient, dict[str, str]],
    case_id: str,
    requested_domain: str,
    query: str,
    expected_domains: tuple[str, ...],
) -> None:
    client, headers = api_client
    resp = client.post(
        "/api/v1/scout/query",
        headers=headers,
        json={"domain_id": requested_domain, "query": query},
    )
    assert resp.status_code == 200, f"{case_id} returned {resp.status_code}: {resp.text[:300]}"
    data = resp.json()
    assert data["domain_id"] in expected_domains, (
        f"{case_id} routed to unexpected domain {data['domain_id']!r}"
    )

    assert data["passed"] is True, (
        f"{case_id} failed answerable PG17 evaluation: {data.get('diagnostic')}"
    )
    assert data["context_pack"] is not None


@pytest.mark.parametrize(
    ("case_id", "requested_domain", "query"),
    _ABSTENTION_CASES,
    ids=[c[0] for c in _ABSTENTION_CASES],
)
def test_unscripted_zero_spend_abstention(
    api_client: tuple[TestClient, dict[str, str]],
    case_id: str,
    requested_domain: str,
    query: str,
) -> None:
    client, headers = api_client
    resp = client.post(
        "/api/v1/scout/query",
        headers=headers,
        json={"domain_id": requested_domain, "query": query},
    )
    assert resp.status_code == 200, f"{case_id} returned {resp.status_code}: {resp.text[:300]}"
    data = resp.json()
    assert data["passed"] is False, f"{case_id} should abstain on out-of-corpus trap"
    diag = data.get("diagnostic") or {}
    assert diag.get("is_abstention") is True
    assert diag.get("paid_tokens_spent", -1) == 0
    assert len(data.get("citations") or []) == 0
