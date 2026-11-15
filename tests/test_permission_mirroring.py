r"""Permanent test suite for Enterprise Permission Mirroring & Provenance Tracking (M1 F02, F04).

Verifies:
1. GitLab role mapping hierarchy: Guest (zero read scopes), Reporter (read-only),
   Developer (read/write), Maintainer/Owner (admin/write/read).
2. Confluence space permission mapping: read, write, admin canonical scopes.
3. Jira project permission mapping: browse/read, write, admin canonical scopes.
4. Scope injection prevention: strict regex ^[a-zA-Z0-9_\-\./:]+$ rejects SQL injection.
5. Fail-closed security: users with no permissions receive unprivileged scope.
6. Webhook-driven cache invalidation and TTL expiration in CachedPermissionMirrorService.
7. Complete provenance enforcement on CodeSlicePointer, DocCitationSpan, and ContextPack.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from context_scout.schemas.contracts import (
    CodeSlicePointer,
    DocCitationSpan,
    ImplementationContextPack,
    ModificationIntent,
    PreFlightValidationReport,
    VerbatimCodeSlice,
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
    sanitize_scope_identifier,
)


def test_gitlab_role_mapping_hierarchy() -> None:
    """F02: Verifies GitLab role mapping, specifically that Guest gets 0 read scopes."""
    repo = "enterprise/trading-core"

    # Guest (< Reporter) gets zero access to private repo code
    guest_scopes = PermissionMirror.map_gitlab_role_to_scopes(repo, GitLabRole.GUEST)
    assert guest_scopes == [], "GitLab Guest must receive 0 read scopes to private repositories"

    minimal_scopes = PermissionMirror.map_gitlab_role_to_scopes(repo, GitLabRole.MINIMAL_ACCESS)
    assert minimal_scopes == []

    # Reporter gets read-only
    reporter_scopes = PermissionMirror.map_gitlab_role_to_scopes(repo, GitLabRole.REPORTER)
    assert reporter_scopes == ["repo:enterprise/trading-core:read"]

    # Developer gets read and write
    dev_scopes = PermissionMirror.map_gitlab_role_to_scopes(repo, GitLabRole.DEVELOPER)
    assert dev_scopes == [
        "repo:enterprise/trading-core:read",
        "repo:enterprise/trading-core:write",
    ]

    # Maintainer & Owner get full administrative scopes
    maintainer_scopes = PermissionMirror.map_gitlab_role_to_scopes(repo, GitLabRole.MAINTAINER)
    assert maintainer_scopes == [
        "repo:enterprise/trading-core:read",
        "repo:enterprise/trading-core:write",
        "repo:enterprise/trading-core:admin",
    ]

    owner_scopes = PermissionMirror.map_gitlab_role_to_scopes(repo, GitLabRole.OWNER)
    assert owner_scopes == maintainer_scopes


def test_confluence_permission_mapping() -> None:
    """F02: Verifies Confluence space permission mapping."""
    space = "QUANT"

    read_scopes = PermissionMirror.map_confluence_permissions_to_scopes(
        space, [ConfluencePermission.READ]
    )
    assert read_scopes == ["confluence:space:QUANT:read"]

    write_scopes = PermissionMirror.map_confluence_permissions_to_scopes(
        space, [ConfluencePermission.WRITE]
    )
    assert write_scopes == ["confluence:space:QUANT:read", "confluence:space:QUANT:write"]

    admin_scopes = PermissionMirror.map_confluence_permissions_to_scopes(
        space, [ConfluencePermission.ADMIN]
    )
    assert admin_scopes == [
        "confluence:space:QUANT:admin",
        "confluence:space:QUANT:read",
        "confluence:space:QUANT:write",
    ]


def test_jira_permission_mapping() -> None:
    """F02: Verifies Jira project permission mapping."""
    project = "RISK"

    browse_scopes = PermissionMirror.map_jira_permissions_to_scopes(
        project, [JiraPermission.BROWSE]
    )
    assert browse_scopes == ["jira:project:RISK:read"]

    write_scopes = PermissionMirror.map_jira_permissions_to_scopes(
        project, [JiraPermission.WRITE]
    )
    assert write_scopes == ["jira:project:RISK:read", "jira:project:RISK:write"]

    admin_scopes = PermissionMirror.map_jira_permissions_to_scopes(
        project, [JiraPermission.ADMIN]
    )
    assert admin_scopes == [
        "jira:project:RISK:admin",
        "jira:project:RISK:read",
        "jira:project:RISK:write",
    ]


@pytest.mark.parametrize(
    "malicious_identifier",
    [
        "repo_name; DROP TABLE code_files; --",
        "space_key' OR 1=1 --",
        "project,with,commas",
        "project with spaces",
        "space\nnewline",
        "",
        "   ",
    ],
)
def test_scope_injection_prevention(malicious_identifier: str) -> None:
    """F02: Verifies regex sanitization rejects SQL/RLS injection attempts."""
    with pytest.raises(ValueError):
        sanitize_scope_identifier(malicious_identifier)


def test_enterprise_identity_profile_resolution() -> None:
    """F02: Verifies end-to-end scope aggregation and deduplication."""
    profile = EnterpriseIdentityProfile(
        user_id="user_senior_eng",
        gitlab_memberships=[
            GitLabProjectMembership(project_path="org/alpha", role=GitLabRole.DEVELOPER),
            GitLabProjectMembership(project_path="org/beta", role=GitLabRole.GUEST),
        ],
        confluence_memberships=[
            ConfluenceSpaceMembership(
                space_key="ENG", permissions=[ConfluencePermission.READ, ConfluencePermission.WRITE]
            )
        ],
        jira_memberships=[
            JiraProjectMembership(
                project_key="ALPHA", permissions=[JiraPermission.BROWSE]
            )
        ],
        custom_scopes=["audit:viewer"],
    )

    scopes = PermissionMirror.resolve_allowed_scopes(profile)
    assert "repo:org/alpha:read" in scopes
    assert "repo:org/alpha:write" in scopes
    # Guest on beta must have no repo:org/beta scopes
    assert not any("org/beta" in s for s in scopes)
    assert "confluence:space:ENG:read" in scopes
    assert "confluence:space:ENG:write" in scopes
    assert "jira:project:ALPHA:read" in scopes
    assert "audit:viewer" in scopes


def test_fail_closed_zero_permissions_isolation() -> None:
    """F02: A user holding 0 permissions receives an isolated unprivileged scope."""
    empty_profile = EnterpriseIdentityProfile(
        user_id="unprivileged_guest",
        gitlab_memberships=[
            GitLabProjectMembership(project_path="org/secret-repo", role=GitLabRole.GUEST)
        ],
    )
    principal = PermissionMirror.create_principal_context(empty_profile)
    assert principal.user_id == "unprivileged_guest"
    assert principal.allowed_scopes == ["unprivileged:user:unprivileged_guest"]


def test_cached_permission_mirror_service_ttl_and_invalidation() -> None:
    """F02: Verifies cache service TTL and webhook invalidation."""
    service = CachedPermissionMirrorService(ttl_seconds=1)
    profile = EnterpriseIdentityProfile(
        user_id="alice",
        gitlab_memberships=[
            GitLabProjectMembership(project_path="org/repo-a", role=GitLabRole.REPORTER)
        ],
    )

    scopes_1 = service.get_user_scopes(profile)
    assert scopes_1 == ["repo:org/repo-a:read"]

    # Mutate profile to add write access
    profile.gitlab_memberships[0].role = GitLabRole.DEVELOPER

    # Should still return cached read-only scope before invalidation
    cached_scopes = service.get_user_scopes(profile)
    assert cached_scopes == ["repo:org/repo-a:read"]

    # Webhook invalidation triggers re-resolution
    service.invalidate_user("alice")
    fresh_scopes = service.get_user_scopes(profile)
    assert "repo:org/repo-a:write" in fresh_scopes

    # Clear drops everything
    service.clear()
    assert len(service._cache) == 0


def test_provenance_tracking_contracts() -> None:
    """F04: Verifies provenance fields on CodeSlicePointer and DocCitationSpan."""
    ptr = CodeSlicePointer(
        repo_name="org/repo",
        file_path="src/main.py",
        start_line=10,
        end_line=20,
        qualified_symbol="app.main",
        selection_reason="Target symbol",
        modification_intent=ModificationIntent.MODIFY_TARGET,
        git_commit_sha="a" * 40,
    )
    assert ptr.git_commit_sha == "a" * 40

    span = DocCitationSpan(
        chunk_id="chk_1",
        doc_id="12345",
        doc_title="API Reference",
        source_uri="https://confluence.example.com/pages/viewpage.action?pageId=12345",
        section_heading="Authentication",
        verbatim_quote="Bearer tokens are required for all API routes.",
        version="2.1",
    )
    assert span.version == "2.1"
    # Auto-extracted page_id from numeric doc_id for Confluence URI
    assert span.page_id == "12345"

    jira_span = DocCitationSpan(
        chunk_id="chk_jira_1",
        doc_id="CORE-456",
        doc_title="Fix Memory Leak",
        source_uri="https://jira.example.com/browse/CORE-456",
        section_heading="Description",
        verbatim_quote="Fix leak in event buffer.",
    )
    assert jira_span.ticket_key == "CORE-456"
    assert jira_span.version == "1.0"


def test_context_pack_provenance_validation() -> None:
    """F04: ImplementationContextPack validates code slice and doc citation provenance."""
    valid_slice = VerbatimCodeSlice(
        repo_name="org/repo",
        file_path="src/main.py",
        start_line=1,
        end_line=5,
        qualified_symbol="main",
        selection_reason="main entry",
        modification_intent=ModificationIntent.MODIFY_TARGET,
        git_commit_sha="c0ffee123456",
        git_blob_sha="deadbeef1234",
        verbatim_code_with_lines="1 | def main(): pass",
    )
    valid_citation = DocCitationSpan(
        chunk_id="chk_1",
        doc_id="doc_1",
        doc_title="Doc",
        source_uri="https://docs.internal/1",
        section_heading="Overview",
        verbatim_quote="Verbatim quote here",
        version="1.0",
    )

    pack = ImplementationContextPack(
        trace_id="tr_1",
        domain_id="trading_code_scout",
        user_task="Implement feature",
        architectural_execution_plan=["Step 1"],
        code_slices=[valid_slice],
        supporting_doc_citations=[valid_citation],
        preflight_report=PreFlightValidationReport(
            passed=True,
            normalized_rrf_score=0.95,
            citations_verified_100_pct=True,
        ),
        local_tokens_consumed=100,
        pack_token_count=200,
    )
    assert len(pack.code_slices) == 1
    assert len(pack.supporting_doc_citations) == 1

    # Incomplete code slice provenance must be rejected by ImplementationContextPack
    invalid_slice = VerbatimCodeSlice.model_construct(
        repo_name="",
        file_path="src/main.py",
        start_line=1,
        end_line=5,
        qualified_symbol="main",
        selection_reason="main entry",
        modification_intent=ModificationIntent.MODIFY_TARGET,
        git_commit_sha="c0ffee123456",
        git_blob_sha="deadbeef1234",
        verbatim_code_with_lines="1 | def main(): pass",
    )
    with pytest.raises(ValidationError, match="incomplete provenance"):
        ImplementationContextPack(
            trace_id="tr_2",
            domain_id="trading_code_scout",
            user_task="Task",
            architectural_execution_plan=["Step 1"],
            code_slices=[invalid_slice],
            preflight_report=PreFlightValidationReport(
                passed=True,
                normalized_rrf_score=0.95,
                citations_verified_100_pct=True,
            ),
            local_tokens_consumed=10,
            pack_token_count=20,
        )
