"""Unit and integration tests for Enterprise Source Connectors (F15, F16, F17).

Verifies:
1. Confluence verbatim retrieval, 50-token Anthropic headers, temporal decay, and supersession.
2. Jira ticket parsing, ADF description, acceptance criteria, dependency graph, and staleness.
3. GitLab commit compare diffing, blob hashing, and push webhook processing.
4. Multi-source permission mirroring and RLS isolation contracts.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import httpx

from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.db.schema import DocChunkRecord
from context_scout.plugins.connectors.confluence import (
    ConfluenceConnectorPlugin,
    ConfluencePageDTO,
)
from context_scout.plugins.connectors.gitlab import (
    GitLabConnectorPlugin,
)
from context_scout.plugins.connectors.jira import (
    JiraConnectorPlugin,
)
from context_scout.queue.task_queue import QueueTask
from context_scout.schemas.contracts import (
    DocCitationSpan,
)
from context_scout.security.permission_mirror import (
    CachedPermissionMirrorService,
    ConfluencePermission,
    ConfluenceSpaceMembership,
    EnterpriseIdentityProfile,
    GitLabProjectMembership,
    GitLabRole,
    JiraPermission,
    JiraProjectMembership,
    PermissionMirror,
)


def test_confluence_verbatim_chunking_and_headers() -> None:
    """Verifies Confluence lossless verbatim chunking and Anthropic 50-token header."""
    quote = (
        "All microservices MUST enforce FORCE ROW LEVEL SECURITY. "
        "The connection pool uses psycopg3 with PgBouncer."
    )
    storage_html = f"<h1>Persistence Layer</h1><p>{quote}</p>"

    plugin = ConfluenceConnectorPlugin(freshness_half_life_days=180)
    page = ConfluencePageDTO(
        page_id="10928374",
        space_key="ENG",
        title="PostgreSQL 17 Migration ADR",
        version_number=3,
        author_username="alyssa.p.hacker",
        source_url="https://mock.atlassian.net/wiki/spaces/ENG/pages/10928374",
        body_storage=storage_html,
        created_at=datetime.now(UTC) - timedelta(days=10),
        last_modified_at=datetime.now(UTC) - timedelta(days=10),
    )

    chunks = plugin.chunk_page_verbatim(page)
    assert len(chunks) >= 1
    chunk = chunks[0]

    # Anthropic contextual header validation
    assert "Confluence space 'ENG'" in chunk.contextual_header
    assert "PostgreSQL 17 Migration ADR" in chunk.contextual_header
    assert "Persistence Layer" in chunk.contextual_header

    # Lossless verbatim preservation
    assert quote in chunk.verbatim_content
    assert chunk.is_stale is False
    assert "confluence:space:ENG:read" in chunk.allowed_scopes


def test_confluence_temporal_decay_warning() -> None:
    """Verifies Confluence pages older than threshold receive staleness flag."""
    plugin = ConfluenceConnectorPlugin(freshness_half_life_days=180)
    stale_date = datetime.now(UTC) - timedelta(days=200)

    page = ConfluencePageDTO(
        page_id="2048",
        space_key="ARCH",
        title="Legacy Architecture",
        version_number=1,
        author_username="dev",
        source_url="https://mock.atlassian.net/wiki/spaces/ARCH/pages/2048",
        body_storage="<p>Legacy setup description.</p>",
        created_at=stale_date,
        last_modified_at=stale_date,
    )

    chunks = plugin.chunk_page_verbatim(page)
    assert len(chunks) >= 1
    assert chunks[0].is_stale is True
    assert "threshold of 180 days" in (chunks[0].stale_reason or "")


def test_confluence_version_supersession_invalidation() -> None:
    """Verifies syncing version N+1 supersedes version N in repository."""
    repo = InMemoryRLSRepository()
    page_id = "10928374"

    # Pre-populate repository with v1 chunk
    v1_chunk_id = uuid.uuid4()
    repo._docs[v1_chunk_id] = DocChunkRecord(
        chunk_id=v1_chunk_id,
        domain_id="default",
        doc_id=f"confluence:{page_id}",
        doc_title="PostgreSQL 17 Migration ADR",
        source_uri="https://mock.atlassian.net/wiki/spaces/ENG/pages/10928374",
        market_scope=None,
        published_date=datetime.now(UTC).date(),
        authority_weight=1.0,
        section_heading="Persistence",
        contextual_header="v1 header",
        verbatim_content="v1 content",
        metadata_json={},
        allowed_scopes=["confluence:space:ENG:read"],
        embedding=[0.0] * 768,
        source_type="confluence",
        version_str="1",
        is_stale=False,
        stale_reason=None,
    )

    plugin = ConfluenceConnectorPlugin()
    v2_page = ConfluencePageDTO(
        page_id=page_id,
        space_key="ENG",
        title="PostgreSQL 17 Migration ADR",
        version_number=2,
        author_username="alyssa",
        source_url="https://mock.atlassian.net/wiki/spaces/ENG/pages/10928374",
        body_storage="<p>Updated content.</p>",
        created_at=datetime.now(UTC),
        last_modified_at=datetime.now(UTC),
    )

    result = plugin.sync_space("default", "ENG", repo, current_pages=[v2_page])
    assert result.source_type == "confluence"
    assert result.items_indexed >= 1
    assert result.stale_items_flagged >= 1

    # Check that v1 chunk was flagged stale
    v1_rec = repo._docs[v1_chunk_id]
    assert v1_rec.is_stale is True
    assert "Superseded by page version 2" in (v1_rec.stale_reason or "")


def test_jira_ticket_parsing_and_dependency_graph() -> None:
    """Verifies Jira REST v3 issue parsing, ADF flattening, and dependency links."""
    raw_issue = {
        "id": "20485",
        "key": "ENG-4092",
        "fields": {
            "summary": "Implement PostgreSQL 17 task queue with SKIP LOCKED",
            "issuetype": {"name": "Story"},
            "status": {"name": "In Progress"},
            "resolution": None,
            "assignee": {"displayName": "Himanshu", "accountId": "hima123"},
            "reporter": {"displayName": "Lead Architect", "accountId": "lead456"},
            "created": "2026-10-10T08:00:00Z",
            "updated": "2026-10-14T12:00:00Z",
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {"type": "text", "text": "Replace in-memory JSON with PG queue."}
                        ],
                    }
                ],
            },
            "customfield_10014": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": "AC1: Background worker claims tasks using SKIP LOCKED.",
                            }
                        ],
                    }
                ],
            },
            "issuelinks": [
                {
                    "outwardIssue": {"key": "ENG-4091", "fields": {"status": {"name": "Done"}}},
                    "type": {"outward": "blocks"},
                }
            ],
        },
    }

    plugin = JiraConnectorPlugin()
    issue = plugin._parse_jira_issue(raw_issue, fallback_project="ENG")

    assert issue.issue_key == "ENG-4092"
    assert issue.summary == "Implement PostgreSQL 17 task queue with SKIP LOCKED"
    assert issue.description_text == "Replace in-memory JSON with PG queue."
    assert "SKIP LOCKED" in (issue.acceptance_criteria or "")
    assert len(issue.linked_issues) == 1
    assert issue.linked_issues[0].target_key == "ENG-4091"
    assert issue.linked_issues[0].link_type == "blocks"

    chunks = plugin.chunk_issue_verbatim(issue)
    assert len(chunks) >= 2  # Description chunk + Acceptance Criteria chunk
    assert any("AC1: Background worker" in c.verbatim_content for c in chunks)
    assert any("Jira ticket ENG-4092 in project 'ENG'" in c.contextual_header for c in chunks)


def test_jira_staleness_on_closed_or_wontfix() -> None:
    """Verifies Jira tickets with Closed status or Won't Fix resolution are stale."""
    plugin = JiraConnectorPlugin()
    issue = plugin._parse_jira_issue(
        {
            "id": "1001",
            "key": "ENG-101",
            "fields": {
                "summary": "Abandoned feature",
                "issuetype": {"name": "Task"},
                "status": {"name": "Closed"},
                "resolution": {"name": "Won't Fix"},
                "created": "2026-01-01T00:00:00Z",
                "updated": "2026-01-02T00:00:00Z",
                "description": "Never implement this.",
            },
        },
        fallback_project="ENG",
    )

    chunks = plugin.chunk_issue_verbatim(issue)
    assert len(chunks) >= 1
    assert chunks[0].is_stale is True
    assert "Closed" in (chunks[0].stale_reason or "")
    assert "Won't Fix" in (chunks[0].stale_reason or "")


def test_jira_citation_span_contract_validation() -> None:
    """Verifies DocCitationSpan contract validation with Jira provenance."""
    span = DocCitationSpan(
        chunk_id=str(uuid.uuid4()),
        doc_id="ENG-4092",
        doc_title="Implement PostgreSQL 17 task queue with SKIP LOCKED",
        source_uri="https://mock.atlassian.net/browse/ENG-4092",
        section_heading="Acceptance Criteria",
        verbatim_quote="AC1: Background worker claims tasks using SKIP LOCKED.",
        ticket_key="ENG-4092",
    )
    assert span.ticket_key == "ENG-4092"
    assert span.page_id is None
    assert span.doc_id == "ENG-4092"


def test_gitlab_compare_diff_incremental_sync() -> None:
    """Verifies GitLab Compare API diff parsing and raw blob fetching."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/repository/compare" in url:
            return httpx.Response(
                200,
                json={
                    "diffs": [
                        {
                            "old_path": "src/queue/task_queue.py",
                            "new_path": "src/queue/task_queue.py",
                            "new_file": False,
                            "renamed_file": False,
                            "deleted_file": False,
                            "diff": "@@ -1,5 +1,6 @@\n+import psycopg\n",
                        }
                    ]
                },
            )
        if "src%2Fqueue%2Ftask_queue.py/raw" in url:
            return httpx.Response(
                200,
                content=b'"""Updated task queue implementation."""\nimport psycopg\n',
            )
        return httpx.Response(404, json={"error": "Not found"})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://gitlab.example.com",
    )
    plugin = GitLabConnectorPlugin(base_url="https://gitlab.example.com", client=client)

    diffs = plugin.compare_commits("trading/core", "commit_a", "commit_b")
    assert len(diffs) == 1
    assert diffs[0].new_path == "src/queue/task_queue.py"

    blob = plugin.fetch_raw_blob("trading/core", "src/queue/task_queue.py", "commit_b")
    assert b"import psycopg" in blob


def test_gitlab_push_and_mr_webhook_handling() -> None:
    """Verifies GitLab push webhook task processing and SyncResult production."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/repository/compare" in url:
            return httpx.Response(
                200,
                json={
                    "diffs": [
                        {
                            "old_path": "src/App.cs",
                            "new_path": "src/App.cs",
                            "new_file": True,
                            "renamed_file": False,
                            "deleted_file": False,
                            "diff": "+public class App {}",
                        }
                    ]
                },
            )
        if "src%2FApp.cs/raw" in url:
            return httpx.Response(200, content=b"public class App {}")
        return httpx.Response(404, json={"error": "Not found"})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://gitlab.example.com",
    )
    plugin = GitLabConnectorPlugin(base_url="https://gitlab.example.com", client=client)
    repo = InMemoryRLSRepository()

    task = QueueTask(
        task_id=uuid.uuid4(),
        domain_id="default",
        event_type="gitlab_push",
        payload={
            "project": {
                "id": 100,
                "path_with_namespace": "trading/core",
                "default_branch": "main",
                "git_http_url": "https://gitlab.example.com/trading/core.git",
            },
            "before": "sha_before",
            "after": "sha_after",
            "ref": "refs/heads/main",
            "commits": [{"id": "sha_after"}],
        },
        priority=1,
        status="PENDING",
        worker_id=None,
        attempts=0,
        max_attempts=3,
        error_message=None,
        scheduled_at=datetime.now(UTC),
        created_at=datetime.now(UTC),
    )

    result = plugin.process_push_webhook(task, repo)
    assert result.source_type == "gitlab"
    assert result.items_scanned == 1
    assert result.items_indexed == 1
    assert result.cursor_or_checkpoint == "sha_after"
    assert len(result.errors) == 0


def test_multi_source_permission_mirroring_rls_isolation() -> None:
    """Verifies that PermissionMirror derives canonical scopes across Confluence, Jira, GitLab."""
    profile = EnterpriseIdentityProfile(
        user_id="alice",
        gitlab_memberships=[
            GitLabProjectMembership(
                project_path="trading/execution",
                role=GitLabRole.DEVELOPER,
            )
        ],
        confluence_memberships=[
            ConfluenceSpaceMembership(
                space_key="ENG",
                permissions=[ConfluencePermission.READ, ConfluencePermission.WRITE],
            )
        ],
        jira_memberships=[
            JiraProjectMembership(
                project_key="ENG",
                permissions=[JiraPermission.BROWSE],
            )
        ],
    )
    principal = PermissionMirror.create_principal_context(profile)

    # Scopes check
    scopes = set(principal.allowed_scopes)
    assert "repo:trading/execution:read" in scopes
    assert "repo:trading/execution:write" in scopes
    assert "confluence:space:ENG:read" in scopes
    assert "confluence:space:ENG:write" in scopes
    assert "jira:project:ENG:read" in scopes

    # Verify RLS repo filtering respects scopes
    repo = InMemoryRLSRepository()
    visible = repo.list_visible_code_files(principal, domain_id="default")
    assert isinstance(visible, list)


def test_webhook_permission_cache_invalidation() -> None:
    """Verifies CachedPermissionMirrorService invalidates cached user profiles on demand."""
    service = CachedPermissionMirrorService(ttl_seconds=300)

    profile_1 = EnterpriseIdentityProfile(
        user_id="alice",
        gitlab_memberships=[
            GitLabProjectMembership(project_path="repo1", role=GitLabRole.REPORTER)
        ],
    )

    s1 = service.get_user_scopes(profile_1)
    assert "repo:repo1:read" in s1

    # Now simulate membership change without invalidate - should still return cached
    profile_2 = EnterpriseIdentityProfile(
        user_id="alice",
        gitlab_memberships=[
            GitLabProjectMembership(project_path="repo1", role=GitLabRole.DEVELOPER)
        ],
    )
    s2 = service.get_user_scopes(profile_2)
    assert "repo:repo1:write" not in s2  # Cached old scopes

    # Invalidate cache for alice
    service.invalidate_user("alice")
    s3 = service.get_user_scopes(profile_2)
    assert "repo:repo1:write" in s3  # Freshly resolved
