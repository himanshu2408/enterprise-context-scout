"""M1-05 Jira BROWSE ∩ Issue Security Level ∩ Comment Role/Group Visibility ACL tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.plugins.connectors.jira import (
    JiraCommentDTO,
    JiraConnectorPlugin,
    JiraIssueDTO,
)
from context_scout.schemas.contracts import PrincipalContext, RoleType


def _make_principal(user_id: str, scopes: list[str]) -> PrincipalContext:
    return PrincipalContext(
        user_id=user_id,
        role=RoleType.ENGINEER,
        allowed_scopes=scopes,
    )


def _raw_issue(
    *,
    key: str = "SEC-101",
    project_key: str = "SEC",
    security: Any = None,
    comments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "project": {"key": project_key},
        "summary": "Harden credential rotation pipeline",
        "issuetype": {"name": "Story"},
        "status": {"name": "In Progress"},
        "resolution": None,
        "assignee": {"displayName": "Alice"},
        "reporter": {"displayName": "Bob"},
        "created": "2026-04-01T10:00:00Z",
        "updated": "2026-04-02T12:00:00Z",
        "description": "Rotate KMS keys automatically every 24h.",
        "customfield_10014": "All rotated keys pass audit verification.",
        "issuelinks": [],
        "security": security,
        "comment": {"comments": comments or []},
    }
    return {"id": "10101", "key": key, "fields": fields}


def test_standard_issue_comment_role_and_group_isolation() -> None:
    """BROWSE user reads Description/AC/public comments, never role/group restricted comments."""
    raw = _raw_issue(
        key="ENG-200",
        project_key="ENG",
        security=None,
        comments=[
            {
                "id": "c1",
                "author": {"displayName": "Alice"},
                "body": "Public triage note.",
                "created": "2026-04-01T11:00:00Z",
                "updated": "2026-04-01T11:00:00Z",
            },
            {
                "id": "c2",
                "author": {"displayName": "Admin"},
                "body": "Admin-only root cause.",
                "created": "2026-04-01T12:00:00Z",
                "updated": "2026-04-01T12:00:00Z",
                "visibility": {"type": "role", "value": "Administrators"},
            },
            {
                "id": "c3",
                "author": {"displayName": "SecOps"},
                "body": "SecOps-only IOC indicator.",
                "created": "2026-04-01T13:00:00Z",
                "updated": "2026-04-01T13:00:00Z",
                "visibility": {"type": "group", "identifier": "secops"},
            },
        ],
    )
    plugin = JiraConnectorPlugin()
    issue = plugin._parse_jira_issue(raw, fallback_project="ENG")
    chunks = plugin.chunk_issue_verbatim(issue, domain_id="default")

    repo = InMemoryRLSRepository()
    for chunk in chunks:
        repo._docs[chunk.chunk_id] = chunk

    browse_user = _make_principal("user-browse", ["jira:project:ENG:read"])
    visible_browse = repo.list_visible_doc_chunks(browse_user, "default")
    headings_browse = {c.section_heading for c in visible_browse}
    assert headings_browse == {"Description", "Acceptance Criteria", "Comment:c1"}

    admin_user = _make_principal(
        "user-admin",
        ["jira:project:ENG:read", "jira:project:ENG:role:Administrators:read"],
    )
    visible_admin = repo.list_visible_doc_chunks(admin_user, "default")
    headings_admin = {c.section_heading for c in visible_admin}
    assert headings_admin == {"Description", "Acceptance Criteria", "Comment:c1", "Comment:c2"}

    secops_user = _make_principal(
        "user-secops",
        ["jira:project:ENG:read", "jira:project:ENG:group:secops:read"],
    )
    visible_secops = repo.list_visible_doc_chunks(secops_user, "default")
    headings_secops = {c.section_heading for c in visible_secops}
    assert headings_secops == {"Description", "Acceptance Criteria", "Comment:c1", "Comment:c3"}


def test_security_level_and_comment_compound_intersection() -> None:
    """Enforces BROWSE ∩ Issue Security Level ∩ Comment Role/Group Visibility."""
    raw = _raw_issue(
        key="SEC-301",
        project_key="SEC",
        security={"id": "sec_101", "name": "Confidential"},
        comments=[
            {
                "id": "pub1",
                "author": {"displayName": "Alice"},
                "body": "General security-level update.",
                "created": "2026-04-01T11:00:00Z",
                "updated": "2026-04-01T11:00:00Z",
            },
            {
                "id": "priv1",
                "author": {"displayName": "SecLead"},
                "body": "Exploit payload details for SecOps.",
                "created": "2026-04-01T12:00:00Z",
                "updated": "2026-04-01T12:00:00Z",
                "visibility": {"type": "Role", "value": "SecOps"},
            },
        ],
    )
    plugin = JiraConnectorPlugin()
    issue = plugin._parse_jira_issue(raw, fallback_project="SEC")
    chunks = plugin.chunk_issue_verbatim(issue, domain_id="default")

    # Never emit bare project read scope on security-level restricted issues
    for chunk in chunks:
        assert "jira:project:SEC:read" not in chunk.allowed_scopes

    repo = InMemoryRLSRepository()
    for chunk in chunks:
        repo._docs[chunk.chunk_id] = chunk

    # 1. Basic BROWSE user gets 0 chunks
    browse_user = _make_principal("browse-only", ["jira:project:SEC:read"])
    assert repo.list_visible_doc_chunks(browse_user, "default") == []

    # 2. Security-level user reads Description/AC + unrestricted comment, NOT role:SecOps comment
    sec_level_user = _make_principal(
        "sec-level-user",
        ["jira:project:SEC:read", "jira:project:SEC:security_level:sec_101:read"],
    )
    visible_sec = repo.list_visible_doc_chunks(sec_level_user, "default")
    assert {c.section_heading for c in visible_sec} == {
        "Description",
        "Acceptance Criteria",
        "Comment:pub1",
    }

    # 3. User with role:SecOps on standard issue CANNOT read role:SecOps comment without sec_101
    standard_role_user = _make_principal(
        "standard-secops",
        ["jira:project:SEC:read", "jira:project:SEC:role:SecOps:read"],
    )
    assert repo.list_visible_doc_chunks(standard_role_user, "default") == []

    # 4. Compound security-level + role user CAN read the restricted comment
    compound_user = _make_principal(
        "compound-secops",
        [
            "jira:project:SEC:read",
            "jira:project:SEC:security_level:sec_101:read",
            "jira:project:SEC:security_level:sec_101:role:SecOps:read",
        ],
    )
    visible_compound = repo.list_visible_doc_chunks(compound_user, "default")
    assert {c.section_heading for c in visible_compound} == {
        "Description",
        "Acceptance Criteria",
        "Comment:pub1",
        "Comment:priv1",
    }


def test_fail_closed_on_malformed_and_empty_intersections() -> None:
    """Malformed comment/security payloads or empty principal intersections fail closed."""
    plugin = JiraConnectorPlugin()
    now = datetime.now(UTC)

    # 1. Malformed comment visibility (non-dict, invalid type, injection chars)
    raw_bad_comments = _raw_issue(
        key="SEC-401",
        project_key="SEC",
        comments=[
            {
                "id": "bad1",
                "body": "Hidden note 1",
                "visibility": "not-a-dict",
            },
            {
                "id": "bad2",
                "body": "Hidden note 2",
                "visibility": {"type": "team", "value": "alpha"},
            },
            {
                "id": "bad3",
                "body": "Hidden note 3",
                "visibility": {"type": "role", "value": "Admins;DROP TABLE"},
            },
        ],
    )
    issue_bad_comments = plugin._parse_jira_issue(raw_bad_comments, fallback_project="SEC")
    assert all(c.is_visibility_malformed for c in issue_bad_comments.comments)
    for c in issue_bad_comments.comments:
        assert plugin.resolve_comment_allowed_scopes(issue_bad_comments, c) == [
            "jira:project:SEC:restricted:none"
        ]

    # 2. Malformed security payload (non-dict or empty dict) fails closed
    issue_bad_sec_1 = plugin._parse_jira_issue(
        _raw_issue(key="SEC-402", project_key="SEC", security="bad-string"),
        fallback_project="SEC",
    )
    assert plugin.resolve_issue_allowed_scopes(issue_bad_sec_1) == [
        "jira:project:SEC:restricted:none"
    ]
    issue_bad_sec_2 = plugin._parse_jira_issue(
        _raw_issue(key="SEC-403", project_key="SEC", security={}),
        fallback_project="SEC",
    )
    assert plugin.resolve_issue_allowed_scopes(issue_bad_sec_2) == [
        "jira:project:SEC:restricted:none"
    ]

    # 3. Empty BROWSE ∩ SecurityLevel principal intersection fails closed
    disjoint_issue = JiraIssueDTO(
        issue_id="404",
        issue_key="SEC-404",
        project_key="SEC",
        summary="Disjoint principals",
        issue_type="Bug",
        status="Open",
        source_url="https://mock.atlassian.net/browse/SEC-404",
        description_text="Sensitive",
        security_level_id="sec_99",
        security_level_principals=("alice", "bob"),
        browse_principals=("charlie",),
        comments=[
            JiraCommentDTO(
                comment_id="c1",
                body_text="Should also fail closed",
                created_at=now,
                updated_at=now,
            )
        ],
        created_at=now,
        updated_at=now,
    )
    assert plugin.resolve_issue_allowed_scopes(disjoint_issue) == [
        "jira:project:SEC:restricted:none"
    ]
    assert plugin.resolve_comment_allowed_scopes(disjoint_issue, disjoint_issue.comments[0]) == [
        "jira:project:SEC:restricted:none"
    ]

    # 4. Empty browse_principals tuple fails closed
    empty_browse_issue = JiraIssueDTO(
        issue_id="405",
        issue_key="SEC-405",
        project_key="SEC",
        summary="Empty browse",
        issue_type="Bug",
        status="Open",
        source_url="https://mock.atlassian.net/browse/SEC-405",
        description_text="Sensitive",
        browse_principals=(),
        created_at=now,
        updated_at=now,
    )
    assert plugin.resolve_issue_allowed_scopes(empty_browse_issue) == [
        "jira:project:SEC:restricted:none"
    ]


def test_sync_project_retains_all_chunks_and_enforces_issue_record_acls() -> None:
    """Regression test: sync_project_to_repository retains Description + AC + Comments."""
    captured_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        if request.url.path == "/rest/api/3/search":
            captured_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "total": 1,
                    "issues": [
                        _raw_issue(
                            key="SEC-501",
                            project_key="SEC",
                            security={"id": "sec_501", "name": "Restricted"},
                            comments=[
                                {
                                    "id": "c1",
                                    "author": {"displayName": "Alice"},
                                    "body": {
                                        "type": "doc",
                                        "content": [
                                            {
                                                "type": "paragraph",
                                                "content": [
                                                    {"type": "text", "text": "Comment one"}
                                                ],
                                            }
                                        ],
                                    },
                                    "created": "2026-04-01T11:00:00Z",
                                    "updated": "2026-04-01T11:00:00Z",
                                },
                                {
                                    "id": "c2",
                                    "author": {"displayName": "Bob"},
                                    "body": "Comment two (role restricted)",
                                    "created": "2026-04-01T12:00:00Z",
                                    "updated": "2026-04-01T12:00:00Z",
                                    "visibility": {"type": "role", "value": "SecOps"},
                                },
                            ],
                        )
                    ],
                },
            )
        return httpx.Response(404)

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://mock.atlassian.net"
    )
    plugin = JiraConnectorPlugin(client=client)
    repo = InMemoryRLSRepository()

    result = plugin.sync_project_to_repository("default", "SEC", repo)
    assert result.items_scanned == 1
    assert result.items_indexed == 4  # Description + AC + 2 Comments

    # Verify search_issues_incremental requested security and comment fields
    assert len(captured_payloads) == 1
    assert "security" in captured_payloads[0]["fields"]
    assert "comment" in captured_payloads[0]["fields"]

    # Verify all 4 chunks were retained in repo._docs (no premature purge inside insert loop)
    assert len(repo._docs) == 4
    assert {c.section_heading for c in repo._docs.values()} == {
        "Description",
        "Acceptance Criteria",
        "Comment:c1",
        "Comment:c2",
    }

    # Verify JiraIssueRecord uses security-level scopes, NOT bare project read
    assert len(repo._jira) == 1
    issue_rec = next(iter(repo._jira.values()))
    assert issue_rec.allowed_scopes == ["jira:project:SEC:security_level:sec_501:read"]

    # Re-sync replaces old chunks cleanly (still 4 chunks, not 8)
    plugin.sync_project_to_repository("default", "SEC", repo)
    assert len(repo._docs) == 4
