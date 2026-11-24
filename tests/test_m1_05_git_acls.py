"""Tests for M1-05 GitLab 18.7+ :read_code and GitHub Enterprise App/Team ACLs."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import httpx
import pytest

from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.plugins.connectors import GitHubEnterpriseConnectorPlugin, GitLabConnectorPlugin
from context_scout.plugins.connectors.github import GitHubAppInstallationACL
from context_scout.plugins.registry import PluginRegistry
from context_scout.queue.task_queue import QueueTask
from context_scout.security.permission_mirror import GitLabRole


def _make_task(event_type: str, payload: dict[str, object]) -> QueueTask:
    now = datetime.now(UTC)
    return QueueTask(
        task_id=uuid.uuid4(),
        domain_id="default",
        event_type=event_type,
        payload=payload,
        priority=1,
        status="PENDING",
        worker_id=None,
        attempts=0,
        max_attempts=3,
        error_message=None,
        scheduled_at=now,
        created_at=now,
    )


def test_gitlab_18_7_guest_and_custom_read_code_scopes() -> None:
    plugin = GitLabConnectorPlugin()

    # Guest without :read_code receives 0 code scopes
    assert plugin.resolve_project_member_scopes("org/core", access_level=GitLabRole.GUEST) == []
    assert plugin.mirror_permissions("org/core", role=GitLabRole.GUEST) == []

    # Guest with GitLab 18.7+ custom role :read_code=True receives read scope
    assert plugin.resolve_project_member_scopes(
        "org/core",
        access_level=GitLabRole.GUEST,
        read_code=True,
    ) == ["repo:org/core:read"]

    # NO_ACCESS / MINIMAL_ACCESS always fail closed even if read_code=True
    assert (
        plugin.resolve_project_member_scopes(
            "org/core",
            access_level=GitLabRole.NO_ACCESS,
            read_code=True,
        )
        == []
    )
    assert (
        plugin.resolve_project_member_scopes(
            "org/core",
            access_level=GitLabRole.MINIMAL_ACCESS,
            read_code=True,
        )
        == []
    )


def test_gitlab_18_7_fail_closed_and_shared_groups() -> None:
    plugin = GitLabConnectorPlugin()

    # Backward compatibility on default call
    assert plugin.mirror_permissions("org/repo") == ["repo:org/repo:read"]

    # Developer with explicit read_code=False fails closed
    assert (
        plugin.resolve_project_member_scopes(
            "org/core",
            access_level=GitLabRole.DEVELOPER,
            read_code=False,
        )
        == []
    )

    # Disabled repository_access_level fails closed
    assert (
        plugin.resolve_project_member_scopes(
            "org/core",
            access_level=GitLabRole.MAINTAINER,
            repository_access_level="disabled",
        )
        == []
    )

    # Transitive shared_with_groups emits group scopes for >= Reporter or Guest+:read_code
    scopes = plugin.resolve_project_member_scopes(
        "org/core",
        access_level=GitLabRole.REPORTER,
        shared_with_groups=[
            {"group_full_path": "sec/reviewers", "group_access_level": GitLabRole.REPORTER},
            {"group_full_path": "sec/guests", "group_access_level": GitLabRole.GUEST},
            {
                "group_full_path": "sec/custom-guests",
                "group_access_level": GitLabRole.GUEST,
                "member_role": {"read_code": True},
            },
        ],
    )
    assert scopes == [
        "repo:org/core:read",
        "gitlab:group:sec/reviewers:read",
        "gitlab:group:sec/custom-guests:read",
    ]


def test_gitlab_fetch_member_read_code_permission() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/members/all/1"):
            return httpx.Response(200, json={"access_level": 10})
        if url.endswith("/members/all/2"):
            return httpx.Response(
                200,
                json={"access_level": 10, "member_role": {"read_code": True}},
            )
        if url.endswith("/members/all/3"):
            return httpx.Response(
                200,
                json={"access_level": 30, "permissions": {"read_code": False}},
            )
        if url.endswith("/members/all/4"):
            return httpx.Response(200, json={"access_level": 30})
        if url.endswith("/members/all/403"):
            return httpx.Response(403, json={"message": "Forbidden"})
        return httpx.Response(404, json={"message": "Not found"})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://gitlab.example.com",
    )
    plugin = GitLabConnectorPlugin(client=client)

    assert plugin.fetch_member_read_code_permission("org/core", 1) is False
    assert plugin.fetch_member_read_code_permission("org/core", 2) is True
    assert plugin.fetch_member_read_code_permission("org/core", 3) is False
    assert plugin.fetch_member_read_code_permission("org/core", 4) is True
    assert plugin.fetch_member_read_code_permission("org/core", 403) is False
    assert plugin.fetch_member_read_code_permission("org/core", 999) is False


def test_gitlab_push_webhook_fails_closed_on_disabled_repo() -> None:
    plugin = GitLabConnectorPlugin()
    repo = InMemoryRLSRepository()

    task_disabled = _make_task(
        "gitlab_push",
        {
            "project": {
                "id": 101,
                "path_with_namespace": "org/secret-repo",
                "repository_access_level": "disabled",
            },
            "before": "sha1",
            "after": "sha2",
        },
    )
    res_disabled = plugin.process_push_webhook(task_disabled, repo)
    assert res_disabled.items_indexed == 0
    assert len(res_disabled.errors) == 1
    assert len(repo._repos) == 0
    assert len(repo._files) == 0

    task_denied_code = _make_task(
        "gitlab_push",
        {
            "project": {
                "id": 102,
                "path_with_namespace": "org/no-code",
                "read_code": False,
            },
            "before": "sha1",
            "after": "sha2",
        },
    )
    res_denied = plugin.process_push_webhook(task_denied_code, repo)
    assert res_denied.items_indexed == 0
    assert len(res_denied.errors) == 1
    assert len(repo._repos) == 0


def test_github_app_acl_fail_closed_and_team_collaborator_mapping() -> None:
    plugin = GitHubEnterpriseConnectorPlugin()

    # Suspended installation fails closed
    suspended_acl = GitHubAppInstallationACL(
        installation_id=42,
        repository_selection="all",
        permissions={"contents": "read"},
        suspended=True,
    )
    assert plugin.resolve_repo_allowed_scopes("acme/api", installation_acl=suspended_acl) == []

    # Missing contents permission fails closed
    no_contents_acl = GitHubAppInstallationACL(
        installation_id=42,
        repository_selection="all",
        permissions={"metadata": "read"},
    )
    assert plugin.resolve_repo_allowed_scopes("acme/api", installation_acl=no_contents_acl) == []

    # Unselected repository fails closed
    selected_acl = GitHubAppInstallationACL(
        installation_id=42,
        repository_selection="selected",
        repositories=["acme/web"],
        permissions={"contents": "read"},
    )
    assert plugin.resolve_repo_allowed_scopes("acme/api", installation_acl=selected_acl) == []
    assert plugin.resolve_repo_allowed_scopes("acme/web", installation_acl=selected_acl) == [
        "repo:acme/web:read"
    ]

    # Explicit teams/collaborators with no read permission fail closed
    assert (
        plugin.resolve_repo_allowed_scopes(
            "acme/web",
            installation_acl=selected_acl,
            teams=[{"slug": "contractors", "permission": "none"}],
            collaborators=[{"login": "bob", "permission": "pull", "read_contents": False}],
        )
        == []
    )

    # Team and collaborator scopes exclude broad repo scope so unauthorized org users cannot read
    scoped = plugin.resolve_repo_allowed_scopes(
        "acme/web",
        installation_acl=selected_acl,
        teams=[
            {"slug": "platform", "permission": "push"},
            {"slug": "auditors", "permission": "pull"},
            {"slug": "blocked", "permission": "none"},
        ],
        collaborators=[{"login": "alice", "permission": "maintain"}],
    )
    assert "repo:acme/web:read" not in scoped
    assert scoped == [
        "github:team:acme/platform:read",
        "github:team:acme/platform:write",
        "github:team:acme/auditors:read",
        "github:user:alice:read",
        "github:user:alice:write",
    ]


def test_github_api_endpoints_and_push_webhook_ast_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeEmbedder:
        def embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
            return [[0.25, 0.75, 0.5, 0.1] for _ in texts]

    monkeypatch.setattr(
        "context_scout.retrieval.embedder.get_embedding_provider",
        lambda: _FakeEmbedder(),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/installation/repositories"):
            return httpx.Response(200, json={"repositories": [{"full_name": "acme/service"}]})
        if url.endswith("/repos/acme/service/teams"):
            return httpx.Response(200, json=[{"slug": "core", "permission": "pull"}])
        if url.endswith("/repos/acme/service/collaborators/alice/permission"):
            return httpx.Response(200, json={"permission": "write"})
        if url.endswith("/repos/acme/service"):
            return httpx.Response(200, json={"full_name": "acme/service", "private": True})
        if "/compare/sha1...sha2" in url:
            return httpx.Response(
                200,
                json={
                    "files": [
                        {
                            "filename": "src/app.py",
                            "status": "modified",
                            "patch": "@@ -1 +1 @@\n+def hello(): pass",
                        }
                    ]
                },
            )
        if "/contents/src/app.py" in url:
            return httpx.Response(
                200,
                content=b"def compute_risk(x: int) -> int:\n    return x * 2\n",
            )
        return httpx.Response(404, json={"message": "Not found"})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://github.corp.example/api/v3",
    )
    plugin = GitHubEnterpriseConnectorPlugin(
        base_url="https://github.corp.example/api/v3",
        installation_token="ghs_test_token",
        installation_id=99,
        client=client,
    )

    assert plugin.fetch_app_installation_repos() == ["acme/service"]
    assert plugin.fetch_repo_teams("acme/service") == [{"slug": "core", "permission": "pull"}]
    assert plugin.fetch_collaborator_permission("acme/service", "alice") == {"permission": "write"}
    assert plugin.fetch_item("acme/service") == {"full_name": "acme/service", "private": True}

    repo = InMemoryRLSRepository()
    acl = GitHubAppInstallationACL(
        installation_id=99,
        repository_selection="selected",
        repositories=["acme/service"],
        permissions={"contents": "read"},
    )
    task = _make_task(
        "github_push",
        {
            "repository": {"id": 500, "full_name": "acme/service", "default_branch": "main"},
            "before": "sha1",
            "after": "sha2",
            "ref": "refs/heads/main",
            "commits": [{"id": "sha2", "message": "feat: add compute_risk"}],
        },
    )

    res = plugin.process_push_webhook(task, repo, installation_acl=acl)
    assert res.source_type == "github"
    assert res.items_indexed == 1
    assert len(res.errors) == 0
    assert len(repo._files) == 1
    assert len(repo._symbols) >= 1
    for sym in repo._symbols.values():
        assert sym.embedding == [0.25, 0.75, 0.5, 0.1]
        assert sym.allowed_scopes == ["repo:acme/service:read"]

    # Denied installation ACL fails closed without indexing
    denied_acl = GitHubAppInstallationACL(
        installation_id=99,
        repository_selection="selected",
        repositories=["acme/other"],
        permissions={"contents": "read"},
    )
    res_denied = plugin.process_push_webhook(task, repo, installation_acl=denied_acl)
    assert res_denied.items_indexed == 0
    assert len(res_denied.errors) == 1


def test_github_connector_registered_in_plugin_registry() -> None:
    PluginRegistry.reset()
    connector = PluginRegistry.get_connector("github")
    assert isinstance(connector, GitHubEnterpriseConnectorPlugin)
    assert connector.source_id == "github"


def test_gitlab_push_webhook_propagates_embedder_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GitLab process_push_webhook does not silently swallow embedder failures."""
    from context_scout.agents.scout_workflow import ModelUnavailableError

    def handler(request: httpx.Request) -> httpx.Response:
        if "/repository/compare" in request.url.path:
            return httpx.Response(
                200,
                json={"diffs": [{"old_path": "app.py", "new_path": "app.py", "new_file": True}]},
            )
        if "/repository/files/" in request.url.path:
            return httpx.Response(
                200, text="def compute_signal(x: int) -> int:\n    return x + 1\n"
            )
        return httpx.Response(404)

    class _FailingEmbedder:
        def embed_batch(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
            _ = (texts, is_query)
            raise ModelUnavailableError("Local embedder offline")

    monkeypatch.setattr(
        "context_scout.retrieval.embedder.get_embedding_provider",
        lambda *_a, **_kw: _FailingEmbedder(),
    )

    plugin = GitLabConnectorPlugin(
        client=httpx.Client(
            base_url="https://gitlab.example.com",
            transport=httpx.MockTransport(handler),
        )
    )
    repo = InMemoryRLSRepository()
    task = _make_task(
        "gitlab_push",
        {
            "project": {"id": 42, "path_with_namespace": "acme/core"},
            "before": "sha1",
            "after": "sha2",
            "ref": "refs/heads/main",
            "commits": [{"id": "sha2", "message": "add signal"}],
        },
    )
    with pytest.raises(ModelUnavailableError, match="Local embedder offline"):
        plugin.process_push_webhook(task, repo)
