"""Tests for M1-05 Confluence V2 /restriction-state and recursive ancestor VIEW_RESTRICTED ACLs."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
from context_scout.db.session import PoolRole, RLSConnectionManager
from context_scout.plugins.connectors.confluence import (
    ConfluenceConnectorPlugin,
    ConfluencePageDTO,
)
from context_scout.schemas.contracts import PrincipalContext, RoleType


def _make_page(
    page_id: str,
    *,
    space_key: str = "ENG",
    title: str = "Architecture Spec",
    parent_id: str | None = None,
    restriction_state: str = "unrestricted",
    view_restricted_principals: tuple[str, ...] | None = None,
    ancestor_ids: tuple[str, ...] = (),
) -> ConfluencePageDTO:
    now = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
    return ConfluencePageDTO(
        page_id=page_id,
        space_key=space_key,
        title=title,
        version_number=1,
        author_username="author-1",
        source_url=f"https://mock.atlassian.net/wiki/spaces/{space_key}/pages/{page_id}",
        body_storage="<p>Authoritative engineering documentation with sufficient length.</p>",
        parent_id=parent_id,
        restriction_state=restriction_state,
        view_restricted_principals=view_restricted_principals,
        ancestor_ids=ancestor_ids,
        created_at=now,
        last_modified_at=now,
    )


def test_fetch_page_restriction_state_parses_v2_and_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/wiki/api/v2/pages/p-v2/restriction-state":
            return httpx.Response(
                200,
                json={
                    "restrictionState": "VIEW_RESTRICTED",
                    "restrictions": {
                        "read": {
                            "restrictions": {
                                "user": {"results": [{"accountId": "alice"}, {"username": "bob"}]},
                                "group": {"results": [{"name": "eng-leads"}]},
                            }
                        }
                    },
                },
            )
        if path == "/wiki/api/v2/pages/p-ops/restriction-state":
            return httpx.Response(
                200,
                json={
                    "restrictionState": "restricted",
                    "operations": [
                        {
                            "operation": "read",
                            "restrictions": {
                                "user": {"results": [{"accountId": "alice"}]},
                            },
                        }
                    ],
                },
            )
        if path == "/wiki/api/v2/pages/p-open/restriction-state":
            return httpx.Response(200, json={"restrictionState": "unrestricted"})
        if path == "/wiki/api/v2/pages/p-empty/restriction-state":
            return httpx.Response(200, json={"restrictionState": "VIEW_RESTRICTED"})
        if path == "/wiki/api/v2/pages/p-403/restriction-state":
            return httpx.Response(403, json={"message": "Forbidden"})
        if path == "/wiki/api/v2/pages/p-404/restriction-state":
            return httpx.Response(404, json={"message": "Not found"})
        return httpx.Response(500)

    client = httpx.Client(
        base_url="https://mock.atlassian.net",
        transport=httpx.MockTransport(handler),
    )
    plugin = ConfluenceConnectorPlugin(client=client)

    v2 = plugin.fetch_page_restriction_state("p-v2")
    assert v2["is_view_restricted"] is True
    assert v2["restriction_state"] == "view_restricted"
    assert v2["principals"] == ("alice", "bob", "eng-leads")

    ops = plugin.fetch_page_restriction_state("p-ops")
    assert ops["is_view_restricted"] is True
    assert ops["principals"] == ("alice",)

    unrestricted = plugin.fetch_page_restriction_state("p-open")
    assert unrestricted["is_view_restricted"] is False
    assert unrestricted["restriction_state"] == "unrestricted"
    assert unrestricted["principals"] == ()

    empty_restricted = plugin.fetch_page_restriction_state("p-empty")
    assert empty_restricted["is_view_restricted"] is True
    assert empty_restricted["principals"] == ()

    forbidden = plugin.fetch_page_restriction_state("p-403")
    assert forbidden["is_view_restricted"] is True
    assert forbidden["principals"] == ()

    missing = plugin.fetch_page_restriction_state("p-404")
    assert missing["is_view_restricted"] is True
    assert missing["principals"] == ()


def test_four_level_hierarchy_intersects_ancestor_view_restricted_principals() -> None:
    root = _make_page("root", title="Root")
    parent = _make_page(
        "parent",
        title="Parent",
        parent_id="root",
        restriction_state="view_restricted",
        view_restricted_principals=("alice", "bob"),
    )
    child = _make_page(
        "child",
        title="Child",
        parent_id="parent",
        restriction_state="view_restricted",
        view_restricted_principals=("bob", "charlie"),
    )
    grandchild = _make_page(
        "grandchild",
        title="Grandchild",
        parent_id="child",
        restriction_state="unrestricted",
        view_restricted_principals=None,
    )

    pages_by_id = {p.page_id: p for p in (root, parent, child, grandchild)}
    plugin = ConfluenceConnectorPlugin()

    assert plugin.resolve_effective_page_scopes(root, pages_by_id) == ["confluence:space:ENG:read"]
    assert plugin.resolve_effective_page_scopes(parent, pages_by_id) == [
        "confluence:space:ENG:principal:alice:read",
        "confluence:space:ENG:principal:bob:read",
    ]
    expected_bob_only = ["confluence:space:ENG:principal:bob:read"]
    assert plugin.resolve_effective_page_scopes(child, pages_by_id) == expected_bob_only
    assert plugin.resolve_effective_page_scopes(grandchild, pages_by_id) == expected_bob_only


def test_disjoint_ancestor_vs_child_and_empty_principals_fail_closed() -> None:
    parent = _make_page(
        "p1",
        restriction_state="view_restricted",
        view_restricted_principals=("alice",),
    )
    child = _make_page(
        "p2",
        parent_id="p1",
        restriction_state="view_restricted",
        view_restricted_principals=("charlie",),
    )
    empty_child = _make_page(
        "p3",
        parent_id="p1",
        restriction_state="view_restricted",
        view_restricted_principals=(),
    )
    pages_by_id = {p.page_id: p for p in (parent, child, empty_child)}
    plugin = ConfluenceConnectorPlugin()

    assert plugin.resolve_effective_page_scopes(child, pages_by_id) == [
        "confluence:space:ENG:restricted:none"
    ]
    assert plugin.resolve_effective_page_scopes(empty_child, pages_by_id) == [
        "confluence:space:ENG:restricted:none"
    ]


def test_parent_cycle_detection_fails_closed() -> None:
    page_a = _make_page("a", parent_id="b")
    page_b = _make_page("b", parent_id="a")
    self_cycle = _make_page("c", ancestor_ids=("c",))
    pages_by_id = {"a": page_a, "b": page_b, "c": self_cycle}

    plugin = ConfluenceConnectorPlugin()
    assert plugin.resolve_effective_page_scopes(page_a, pages_by_id) == [
        "confluence:space:ENG:restricted:none"
    ]
    assert plugin.resolve_effective_page_scopes(self_cycle, pages_by_id) == [
        "confluence:space:ENG:restricted:none"
    ]


@pytest.mark.parametrize(
    "backend",
    ["memory", pytest.param("postgres", marks=pytest.mark.postgres)],
)
def test_fetch_missing_ancestors_and_sync_propagation_to_in_memory_repo(
    backend: str,
    request: pytest.FixtureRequest,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/wiki/api/v2/pages/anc-parent":
            return httpx.Response(
                200,
                json={
                    "id": "anc-parent",
                    "spaceKey": "ENG",
                    "title": "Restricted Ancestor",
                    "version": {
                        "number": 2,
                        "authorId": "lead",
                        "createdAt": "2026-11-18T10:00:00Z",
                    },
                    "body": {"storage": {"value": "<p>Restricted parent policy document.</p>"}},
                    "status": "current",
                },
            )
        if path == "/wiki/api/v2/pages/anc-parent/restriction-state":
            return httpx.Response(
                200,
                json={
                    "restrictionState": "VIEW_RESTRICTED",
                    "allowedPrincipals": ["alice", "bob"],
                },
            )
        if path == "/wiki/api/v2/pages/anc-403":
            return httpx.Response(403, json={"message": "Forbidden"})
        return httpx.Response(404)

    client = httpx.Client(
        base_url="https://mock.atlassian.net",
        transport=httpx.MockTransport(handler),
    )
    plugin = ConfluenceConnectorPlugin(client=client)

    child = _make_page(
        "child-1",
        title="Child Runbook",
        parent_id="anc-parent",
        restriction_state="view_restricted",
        view_restricted_principals=("bob", "confluence:space:ENG:team_sec:read"),
    )
    unrestricted_grandchild = _make_page(
        "grandchild-1",
        title="Grandchild Spec",
        parent_id="child-1",
    )
    blocked_child = _make_page(
        "blocked-1",
        title="Blocked Child",
        parent_id="anc-403",
    )

    expected_bob = ["confluence:space:ENG:principal:bob:read"]
    expected_none = ["confluence:space:ENG:restricted:none"]

    if backend == "memory":
        repo = InMemoryRLSRepository()
        result = plugin.sync_space_to_repository(
            domain_id="tenant-1",
            space_key="ENG",
            repository=repo,
            current_pages=[child, unrestricted_grandchild, blocked_child],
            fetch_missing_ancestors=True,
        )
        assert result.items_updated == 3

        pages_by_cid = {p.confluence_page_id: p for p in repo._confluence.values()}
        assert pages_by_cid["child-1"].allowed_scopes == expected_bob
        assert pages_by_cid["grandchild-1"].allowed_scopes == expected_bob
        assert pages_by_cid["blocked-1"].allowed_scopes == expected_none

        docs_by_page = {d.doc_id: d for d in repo._docs.values()}
        assert docs_by_page["confluence:child-1"].allowed_scopes == expected_bob
        assert docs_by_page["confluence:grandchild-1"].allowed_scopes == expected_bob
        assert docs_by_page["confluence:blocked-1"].allowed_scopes == expected_none
        return

    domain_id = str(request.getfixturevalue("ephemeral_pg_domain"))
    settings = ScoutSettings()
    ingest_mgr = RLSConnectionManager(settings.db_app_url, pool_role=PoolRole.INGESTION)
    read_mgr = RLSConnectionManager(settings.db_app_url)
    pg_repo = PostgresRLSRepository()
    ingest_principal = PrincipalContext(
        user_id="conf-ingest",
        role=RoleType.ENGINEER,
        allowed_scopes=["system:connector_ingest"],
    )
    with ingest_mgr.scoped_transaction(
        ingest_principal, enable_hnsw_iterative_scan=False
    ) as ingest_conn:
        result = plugin.sync_space_to_repository(
            domain_id=domain_id,
            space_key="ENG",
            repository=pg_repo,
            conn=ingest_conn,
            current_pages=[child, unrestricted_grandchild, blocked_child],
            fetch_missing_ancestors=True,
        )
    assert result.items_updated == 3

    bob_principal = PrincipalContext(
        user_id="bob",
        role=RoleType.ENGINEER,
        allowed_scopes=expected_bob,
    )
    stranger_principal = PrincipalContext(
        user_id="stranger",
        role=RoleType.ENGINEER,
        allowed_scopes=["confluence:space:ENG:read"],
    )
    with read_mgr.scoped_transaction(bob_principal, enable_hnsw_iterative_scan=False) as bob_conn:
        bob_pages = {
            p.confluence_page_id: p
            for p in pg_repo.list_confluence_pages(bob_conn, domain_id, space_key="ENG")
        }
        assert set(bob_pages.keys()) == {"child-1", "grandchild-1"}
        assert bob_pages["child-1"].allowed_scopes == expected_bob
        assert bob_pages["grandchild-1"].allowed_scopes == expected_bob
        bob_docs = {d.doc_id: d for d in pg_repo.list_visible_doc_chunks(bob_conn, domain_id)}
        assert set(bob_docs.keys()) == {"confluence:child-1", "confluence:grandchild-1"}

    with read_mgr.scoped_transaction(
        stranger_principal, enable_hnsw_iterative_scan=False
    ) as stranger_conn:
        assert pg_repo.list_confluence_pages(stranger_conn, domain_id, space_key="ENG") == []
        assert pg_repo.list_visible_doc_chunks(stranger_conn, domain_id) == []


def test_confluence_429_propagates_from_ancestor_and_restriction_fetches() -> None:
    """HTTP 429 in ancestor or restriction-state fetch raises ConnectorRateLimitError."""
    import pytest

    from context_scout.plugins.connector_sdk import ConnectorRateLimitError

    def ancestor_429_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/wiki/api/v2/pages/anc-429":
            return httpx.Response(
                429,
                headers={"Retry-After": "12"},
                json={"message": "rate limited"},
            )
        return httpx.Response(404)

    plugin_anc = ConfluenceConnectorPlugin(
        client=httpx.Client(
            base_url="https://mock.atlassian.net",
            transport=httpx.MockTransport(ancestor_429_handler),
        )
    )
    child_missing_anc = _make_page("child-429", parent_id="anc-429")
    with pytest.raises(ConnectorRateLimitError) as exc_info:
        plugin_anc.resolve_effective_page_scopes(
            child_missing_anc,
            pages_by_id={},
            fetch_missing_ancestors=True,
        )
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after_seconds == 12.0

    def restriction_429_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/wiki/api/v2/pages/anc-ok":
            return httpx.Response(
                200,
                json={
                    "id": "anc-ok",
                    "spaceKey": "ENG",
                    "title": "Ancestor",
                    "version": {"number": 1},
                    "body": {"storage": {"value": "<p>Ancestor page</p>"}},
                },
            )
        if request.url.path == "/wiki/api/v2/pages/anc-ok/restriction-state":
            return httpx.Response(
                429,
                headers={"Retry-After": "9"},
                json={"message": "too many requests"},
            )
        return httpx.Response(404)

    plugin_res = ConfluenceConnectorPlugin(
        client=httpx.Client(
            base_url="https://mock.atlassian.net",
            transport=httpx.MockTransport(restriction_429_handler),
        )
    )
    anc_restricted = _make_page(
        "anc-ok",
        restriction_state="view_restricted",
        view_restricted_principals=None,
    )
    child_of_ok = _make_page("child-ok", parent_id="anc-ok")
    with pytest.raises(ConnectorRateLimitError) as exc_info_2:
        plugin_res.resolve_effective_page_scopes(
            child_of_ok,
            pages_by_id={"anc-ok": anc_restricted},
            fetch_missing_ancestors=True,
        )
    assert exc_info_2.value.status_code == 429
    assert exc_info_2.value.retry_after_seconds == 9.0
