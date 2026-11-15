"""Empirical Challenger Test Suite for Milestone 1 Security & Authentication (M1 F01-F04).

Adversarial stress-testing across 5 required dimensions:
1. Scope injection payloads: `,`, `;`, `'`, `"`, `--`, `\0`, newline, whitespace, and special chars.
2. GitLab role boundary conditions: verify that Guest never receives private repo read access.
3. Fail-closed behavior on unmapped roles or empty memberships.
4. Provenance tampering: create ImplementationContextPack instances with empty git_commit_sha,
   empty version, or missing source_uri and assert Pydantic validation rejects them.
5. EU residency bypass: attempt instantiating ScoutSettings or GeminiImplementationExecutor with
   us-central1, asia-northeast1, etc., asserting strict rejection.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import ValidationError

from context_scout.agents.gemini_executor import GeminiImplementationExecutor
from context_scout.config import ScoutSettings, validate_vertex_location
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    DocCitationSpan,
    ImplementationContextPack,
    ModificationIntent,
    PreFlightValidationReport,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)
from context_scout.security.permission_mirror import (
    EnterpriseIdentityProfile,
    GitLabProjectMembership,
    GitLabRole,
    PermissionMirror,
    sanitize_scope_identifier,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clean_env() -> Iterator[None]:
    """Ensures test isolation by snapshotting and restoring environment variables."""
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


@pytest.fixture
def valid_code_slice() -> VerbatimCodeSlice:
    return VerbatimCodeSlice(
        repo_name="org/trading-core",
        file_path="src/execution.py",
        start_line=10,
        end_line=25,
        qualified_symbol="execution.process_order",
        selection_reason="Order routing target",
        modification_intent=ModificationIntent.MODIFY_TARGET,
        git_commit_sha="a1b2c3d4e5f67890",
        git_blob_sha="deadbeef12345678",
        verbatim_code_with_lines="10 | def process_order(): pass",
    )


@pytest.fixture
def valid_doc_citation() -> DocCitationSpan:
    return DocCitationSpan(
        chunk_id="chk_001",
        doc_id="DOC-991",
        doc_title="Risk Governance Specs",
        source_uri="https://confluence.internal/spaces/RISK/pages/991",
        section_heading="Capital Adequacy",
        verbatim_quote="Leverage must not exceed 4x under standard regime.",
        version="2.0",
    )


@pytest.fixture
def base_context_pack_kwargs(
    valid_code_slice: VerbatimCodeSlice,
    valid_doc_citation: DocCitationSpan,
) -> dict[str, Any]:
    return {
        "trace_id": "tr_adversarial_m1",
        "domain_id": "trading_code_scout",
        "user_task": "Harden permission mirroring",
        "architectural_execution_plan": ["Verify invariants", "Challenge edge cases"],
        "code_slices": [valid_code_slice],
        "supporting_doc_citations": [valid_doc_citation],
        "preflight_report": PreFlightValidationReport(
            passed=True,
            normalized_rrf_score=0.98,
            citations_verified_100_pct=True,
        ),
        "local_tokens_consumed": 250,
        "pack_token_count": 500,
    }


# =============================================================================
# Dimension 1: Scope Injection Payloads
# =============================================================================


class TestScopeInjectionAdversarial:
    """Stress-tests scope sanitization against SQL/RLS injection delimiters and meta-characters."""

    @pytest.mark.parametrize(
        "payload",
        [
            "repo,admin",
            ",repo",
            "repo,",
            "group/repo,repo:other:read",
            "a,b,c",
            "scope1,scope2",
        ],
    )
    def test_comma_injection_payloads_strictly_rejected(self, payload: str) -> None:
        """Comma is PostgreSQL array delimiter in string_to_array(..., ','); must be rejected."""
        with pytest.raises(ValueError, match="Invalid characters detected"):
            sanitize_scope_identifier(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            "repo; DROP TABLE code_files;",
            "repo;admin",
            ";repo",
            "repo;--",
            "select;1",
        ],
    )
    def test_semicolon_injection_payloads_strictly_rejected(self, payload: str) -> None:
        """Semicolon terminates SQL commands; must be rejected."""
        with pytest.raises(ValueError, match="Invalid characters detected"):
            sanitize_scope_identifier(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            "repo' OR '1'='1",
            "repo'--",
            "repo' UNION SELECT * FROM users--",
            "'",
            "'''",
            "repo'admin",
        ],
    )
    def test_single_quote_sql_injection_payloads_rejected(self, payload: str) -> None:
        """Single quote escapes SQL string literals; must be rejected."""
        with pytest.raises(ValueError, match="Invalid characters detected"):
            sanitize_scope_identifier(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            'repo" OR 1=1',
            'repo"admin',
            '"',
            '""',
            'repo"inject',
        ],
    )
    def test_double_quote_injection_payloads_rejected(self, payload: str) -> None:
        """Double quote escapes SQL identifiers; must be rejected."""
        with pytest.raises(ValueError, match="Invalid characters detected"):
            sanitize_scope_identifier(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            "repo\0admin",
            "\0repo",
            "repo\x00malicious",
            "repo\0",
        ],
    )
    def test_null_byte_injection_payloads_rejected(self, payload: str) -> None:
        """Null bytes truncate C-strings and SQL buffers; must be rejected."""
        with pytest.raises(ValueError, match="Invalid characters detected"):
            sanitize_scope_identifier(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            "repo\nadmin",
            "repo\r\nadmin",
            "repo\radmin",
            "repo\n\nadmin",
            "repo\r\radmin",
        ],
    )
    def test_internal_newline_injection_payloads_rejected(self, payload: str) -> None:
        """Internal newlines break log/header parsing and query boundaries; must be rejected."""
        with pytest.raises(ValueError, match="Invalid characters detected"):
            sanitize_scope_identifier(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            "repo admin",
            "repo\tadmin",
            "repo \t admin",
            "repo  admin",
        ],
    )
    def test_internal_whitespace_payloads_rejected(self, payload: str) -> None:
        """Internal spaces and tabs must be rejected."""
        with pytest.raises(ValueError, match="Invalid characters detected"):
            sanitize_scope_identifier(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            "",
            "   ",
            "\t\n\r",
            "\n\n",
            "\t\t",
        ],
    )
    def test_empty_and_all_whitespace_payloads_rejected(self, payload: str) -> None:
        """Empty strings and all-whitespace strings must be rejected."""
        with pytest.raises(ValueError):
            sanitize_scope_identifier(payload)

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (" repo", "repo"),
            ("repo ", "repo"),
            ("\nrepo", "repo"),
            ("repo\r", "repo"),
            ("\t\nrepo\r\n\t", "repo"),
        ],
    )
    def test_leading_trailing_whitespace_stripped(self, payload: str, expected: str) -> None:
        """Sanitizer cleans outer whitespace via strip() before regex matching."""
        assert sanitize_scope_identifier(payload) == expected

    @pytest.mark.parametrize(
        "payload",
        [
            "repo$admin",
            "repo#admin",
            "repo%admin",
            "repo^admin",
            "repo&admin",
            "repo*admin",
            "repo(admin)",
            "repo{admin}",
            "repo[admin]",
            "repo|admin",
            "repo\\admin",
            "repo<admin>",
            "repo?admin",
            "repo~admin",
            "repo`admin",
            "repo!admin",
            "repo+admin",
            "repo=admin",
        ],
    )
    def test_special_characters_strictly_rejected(self, payload: str) -> None:
        """Shell and regex metacharacters must be rejected."""
        with pytest.raises(ValueError, match="Invalid characters detected"):
            sanitize_scope_identifier(payload)

    def test_sql_comment_double_dash_probe(self) -> None:
        r"""Adversarial Probe: Checks whether SQL comment sequence '--' is permitted or rejected.

        Worker claim: 'Any token failing regex or containing SQL/RLS meta-characters
        (\', \", ;, --, space) raises ValueError'.
        Empirical discovery: ^[a-zA-Z0-9_\-\./:]+$ includes '-' so '--' matches the pattern
        unless explicitly checked.
        """
        raw_dash = "--"
        is_rejected = False
        try:
            sanitize_scope_identifier(raw_dash)
        except ValueError:
            is_rejected = True

        # Document finding: '--' is permitted by regex because '-' is an allowed character
        err_msg = "Expected sanitize_scope_identifier('--') to not raise under current regex"
        assert not is_rejected, err_msg
        assert sanitize_scope_identifier(raw_dash) == "--"
        assert sanitize_scope_identifier("--droptable") == "--droptable"
        assert sanitize_scope_identifier("repo--name") == "repo--name"

    def test_principal_context_scope_validation_delimiters(self) -> None:
        """Verifies PrincipalContext field_validator rejects commas, semicolons, and spaces."""
        with pytest.raises(ValidationError):
            PrincipalContext(
                user_id="alice",
                role=RoleType.ENGINEER,
                allowed_scopes=["repo:legit:read,repo:admin:read"],
            )

        with pytest.raises(ValidationError):
            PrincipalContext(
                user_id="alice",
                role=RoleType.ENGINEER,
                allowed_scopes=["repo:legit:read;DROP TABLE"],
            )

        with pytest.raises(ValidationError):
            PrincipalContext(
                user_id="alice",
                role=RoleType.ENGINEER,
                allowed_scopes=["repo:legit:read repo:admin:read"],
            )


# =============================================================================
# Dimension 2: GitLab Role Boundary Conditions
# =============================================================================


class TestGitLabRoleBoundaryConditions:
    """Stress-tests GitLab role mapping to ensure Guest never obtains read access."""

    @pytest.mark.parametrize(
        "project_path",
        [
            "gitlab-org/gitlab",
            "enterprise/secret-trading-engine",
            "alpha/beta/gamma/deep-nested-service",
            "singleproject",
            "group.name/sub_group-1/repo.git",
        ],
    )
    def test_guest_role_never_receives_read_scopes(self, project_path: str) -> None:
        """Security Invariant: Guest (10) MUST receive [] scopes to private repositories."""
        scopes = PermissionMirror.map_gitlab_role_to_scopes(project_path, GitLabRole.GUEST)
        assert scopes == [], f"GitLab Guest was granted scopes on {project_path}: {scopes}"

    def test_minimal_and_no_access_roles_receive_zero_scopes(self) -> None:
        """Roles below Guest (NO_ACCESS=0, MINIMAL_ACCESS=5) must also return []."""
        p = "enterprise/vault"
        assert PermissionMirror.map_gitlab_role_to_scopes(p, GitLabRole.NO_ACCESS) == []
        assert PermissionMirror.map_gitlab_role_to_scopes(p, GitLabRole.MINIMAL_ACCESS) == []
        assert PermissionMirror.map_gitlab_role_to_scopes(p, -1) == []  # type: ignore[arg-type]

    def test_reporter_role_boundary_is_strictly_read_only(self) -> None:
        """Reporter (20) must receive read-only scope, never write or admin."""
        p = "enterprise/core"
        scopes = PermissionMirror.map_gitlab_role_to_scopes(p, GitLabRole.REPORTER)
        assert scopes == [f"repo:{p}:read"]
        assert f"repo:{p}:write" not in scopes
        assert f"repo:{p}:admin" not in scopes

    def test_developer_role_boundary_is_read_write_without_admin(self) -> None:
        """Developer (30) must receive read and write scopes, never admin."""
        p = "enterprise/core"
        scopes = PermissionMirror.map_gitlab_role_to_scopes(p, GitLabRole.DEVELOPER)
        assert scopes == [f"repo:{p}:read", f"repo:{p}:write"]
        assert f"repo:{p}:admin" not in scopes

    def test_maintainer_and_owner_receive_admin_scopes(self) -> None:
        """Maintainer (40) and Owner (50) receive read, write, and admin scopes."""
        p = "enterprise/core"
        maintainer_scopes = PermissionMirror.map_gitlab_role_to_scopes(p, GitLabRole.MAINTAINER)
        owner_scopes = PermissionMirror.map_gitlab_role_to_scopes(p, GitLabRole.OWNER)

        expected = [f"repo:{p}:read", f"repo:{p}:write", f"repo:{p}:admin"]
        assert maintainer_scopes == expected
        assert owner_scopes == expected

    def test_multi_repo_membership_guest_isolation(self) -> None:
        """In a multi-repo profile, Guest permissions on one repo do not bleed into others."""
        profile = EnterpriseIdentityProfile(
            user_id="developer_bob",
            gitlab_memberships=[
                GitLabProjectMembership(project_path="open/docs", role=GitLabRole.DEVELOPER),
                GitLabProjectMembership(project_path="secret/core", role=GitLabRole.GUEST),
                GitLabProjectMembership(project_path="secret/algo", role=GitLabRole.MINIMAL_ACCESS),
            ],
        )
        resolved = PermissionMirror.resolve_allowed_scopes(profile)
        assert "repo:open/docs:read" in resolved
        assert "repo:open/docs:write" in resolved
        assert not any("secret/core" in s for s in resolved)
        assert not any("secret/algo" in s for s in resolved)


# =============================================================================
# Dimension 3: Fail-Closed Behavior on Unmapped Roles & Empty Memberships
# =============================================================================


class TestFailClosedBehavior:
    """Stress-tests fail-closed security across zero-permission profiles and unmapped roles."""

    def test_empty_enterprise_profile_produces_zero_resource_scopes(self) -> None:
        """An identity profile with no memberships resolves to zero scopes."""
        profile = EnterpriseIdentityProfile(user_id="empty_user")
        scopes = PermissionMirror.resolve_allowed_scopes(profile)
        assert scopes == []

    def test_empty_profile_principal_context_fails_closed_in_rls(self) -> None:
        """When scopes are empty, assigns an unprivileged isolation scope in RLS."""
        profile = EnterpriseIdentityProfile(user_id="isolated_user")
        principal = PermissionMirror.create_principal_context(profile)
        assert principal.user_id == "isolated_user"
        assert principal.allowed_scopes == ["unprivileged:user:isolated_user"]
        for s in principal.allowed_scopes:
            assert not s.startswith("repo:")
            assert not s.startswith("confluence:")
            assert not s.startswith("jira:")

    def test_guest_only_profile_fails_closed_in_principal_context(self) -> None:
        """A profile with only Guest memberships fails closed to unprivileged scope."""
        profile = EnterpriseIdentityProfile(
            user_id="guest_only",
            gitlab_memberships=[
                GitLabProjectMembership(project_path="corp/vault", role=GitLabRole.GUEST),
                GitLabProjectMembership(project_path="corp/auth", role=GitLabRole.MINIMAL_ACCESS),
            ],
        )
        principal = PermissionMirror.create_principal_context(profile)
        assert principal.allowed_scopes == ["unprivileged:user:guest_only"]

    def test_confluence_unmapped_permission_fails_closed(self) -> None:
        """Unrecognized Confluence permission strings must resolve to empty scopes."""
        scopes = PermissionMirror.map_confluence_permissions_to_scopes(
            "ENG", ["unmapped_permission", "auditor", "guest"]
        )
        assert scopes == []

    def test_jira_unmapped_permission_fails_closed(self) -> None:
        """Unrecognized Jira permission strings must resolve to empty scopes."""
        scopes = PermissionMirror.map_jira_permissions_to_scopes(
            "CORE", ["unknown_perm", "viewer", "reporter"]
        )
        assert scopes == []

    def test_unmapped_gitlab_role_integer_probe(self) -> None:
        """Adversarial Probe: Evaluates behavior when an unmapped integer role is supplied.

        If a caller supplies an unmapped integer (e.g. 25, 99) to map_gitlab_role_to_scopes:
        Fail-closed security requires that unknown roles MUST NOT grant administrative permissions.
        """
        scopes_25 = PermissionMirror.map_gitlab_role_to_scopes("corp/repo", 25)  # type: ignore[arg-type]
        scopes_99 = PermissionMirror.map_gitlab_role_to_scopes("corp/repo", 99)  # type: ignore[arg-type]

        grants_admin_on_25 = "repo:corp/repo:admin" in scopes_25
        grants_admin_on_99 = "repo:corp/repo:admin" in scopes_99

        # Document finding for challenger report: unmapped roles fall through to MAINTAINER/OWNER
        assert grants_admin_on_25, "Defect verified: role 25 falls through to admin scopes"
        assert grants_admin_on_99, "Defect verified: role 99 falls through to admin scopes"


# =============================================================================
# Dimension 4: Provenance Tampering & Validation Invariants
# =============================================================================


class TestProvenanceTampering:
    """Stress-tests provenance integrity on VerbatimCodeSlice, DocCitationSpan, and ContextPack."""

    def test_code_slice_empty_commit_sha_rejected_at_pointer(self) -> None:
        """CodeSlicePointer rejects commit SHA with length < 7 if provided."""
        with pytest.raises(ValidationError):
            CodeSlicePointer(
                repo_name="org/repo",
                file_path="src/a.py",
                start_line=1,
                end_line=5,
                qualified_symbol="sym",
                selection_reason="reason",
                modification_intent=ModificationIntent.MODIFY_TARGET,
                git_commit_sha="abc",  # < 7 characters
            )

    def test_verbatim_code_slice_empty_commit_sha_rejected(self) -> None:
        """VerbatimCodeSlice requires mandatory git_commit_sha with min_length=7."""
        with pytest.raises(ValidationError):
            VerbatimCodeSlice(
                repo_name="org/repo",
                file_path="src/a.py",
                start_line=1,
                end_line=5,
                qualified_symbol="sym",
                selection_reason="reason",
                modification_intent=ModificationIntent.MODIFY_TARGET,
                git_commit_sha="",  # empty string rejected
                git_blob_sha="deadbeef1234",
                verbatim_code_with_lines="1 | pass",
            )

    def test_context_pack_rejects_model_constructed_empty_sha(
        self, base_context_pack_kwargs: dict[str, Any]
    ) -> None:
        """ImplementationContextPack catches empty commit_sha via model_construct."""
        tampered_slice = VerbatimCodeSlice.model_construct(
            repo_name="org/repo",
            file_path="src/a.py",
            start_line=1,
            end_line=5,
            qualified_symbol="sym",
            selection_reason="reason",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="",
            git_blob_sha="deadbeef1234",
            verbatim_code_with_lines="1 | pass",
        )
        kwargs = dict(base_context_pack_kwargs)
        kwargs["code_slices"] = [tampered_slice]

        with pytest.raises(ValidationError, match="incomplete provenance"):
            ImplementationContextPack(**kwargs)

    def test_doc_citation_empty_version_rejected(self) -> None:
        """DocCitationSpan rejects empty version string."""
        with pytest.raises(ValidationError):
            DocCitationSpan(
                chunk_id="chk_1",
                doc_id="DOC-1",
                doc_title="Title",
                source_uri="https://docs.corp/1",
                section_heading="Heading",
                verbatim_quote="Quote",
                version="",  # min_length=1
            )

    def test_context_pack_rejects_empty_citation_version(
        self, base_context_pack_kwargs: dict[str, Any]
    ) -> None:
        """ImplementationContextPack field_validator rejects citation with empty version."""
        tampered_citation = DocCitationSpan.model_construct(
            chunk_id="chk_1",
            doc_id="DOC-1",
            doc_title="Title",
            source_uri="https://docs.corp/1",
            section_heading="Heading",
            verbatim_quote="Quote",
            version="",
        )
        kwargs = dict(base_context_pack_kwargs)
        kwargs["supporting_doc_citations"] = [tampered_citation]

        with pytest.raises(ValidationError, match="missing core provenance fields"):
            ImplementationContextPack(**kwargs)

    def test_doc_citation_missing_source_uri_rejected(self) -> None:
        """DocCitationSpan requires source_uri field; omitting it raises ValidationError."""
        with pytest.raises(ValidationError):
            DocCitationSpan(  # type: ignore[call-arg]
                chunk_id="chk_1",
                doc_id="DOC-1",
                doc_title="Title",
                section_heading="Heading",
                verbatim_quote="Quote",
            )

    def test_context_pack_rejects_empty_source_uri(
        self, base_context_pack_kwargs: dict[str, Any]
    ) -> None:
        """ImplementationContextPack field_validator rejects citation with empty source_uri."""
        tampered_citation = DocCitationSpan.model_construct(
            chunk_id="chk_1",
            doc_id="DOC-1",
            doc_title="Title",
            source_uri="",
            section_heading="Heading",
            verbatim_quote="Quote",
            version="1.0",
        )
        kwargs = dict(base_context_pack_kwargs)
        kwargs["supporting_doc_citations"] = [tampered_citation]

        with pytest.raises(ValidationError, match="missing core provenance fields"):
            ImplementationContextPack(**kwargs)

    def test_whitespace_commit_sha_bypass_probe(
        self, base_context_pack_kwargs: dict[str, Any]
    ) -> None:
        """Adversarial Probe: Checks whether whitespace-only commit_sha passes truthiness checks.

        In contracts.py line 330: `if not s.git_commit_sha:` evaluates truthiness,
        meaning '       ' (7 spaces) satisfies min_length=7 and evaluates as truthy.
        """
        whitespace_sha_slice = VerbatimCodeSlice(
            repo_name="org/repo",
            file_path="src/a.py",
            start_line=1,
            end_line=5,
            qualified_symbol="sym",
            selection_reason="reason",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="       ",  # 7 spaces satisfies min_length=7
            git_blob_sha="deadbeef1234",
            verbatim_code_with_lines="1 | pass",
        )
        kwargs = dict(base_context_pack_kwargs)
        kwargs["code_slices"] = [whitespace_sha_slice]

        pack = ImplementationContextPack(**kwargs)
        assert pack.code_slices[0].git_commit_sha == "       "

    def test_inverted_line_bounds_rejected(self) -> None:
        """CodeSlicePointer and VerbatimCodeSlice reject end_line < start_line."""
        with pytest.raises(ValidationError, match=r"end_line .* must be >= start_line"):
            CodeSlicePointer(
                repo_name="org/repo",
                file_path="src/a.py",
                start_line=20,
                end_line=10,
                qualified_symbol="sym",
                selection_reason="reason",
                modification_intent=ModificationIntent.MODIFY_TARGET,
            )

    def test_zero_or_negative_line_numbers_rejected(self) -> None:
        """Lines must be 1-indexed (ge=1)."""
        with pytest.raises(ValidationError):
            CodeSlicePointer(
                repo_name="org/repo",
                file_path="src/a.py",
                start_line=0,
                end_line=5,
                qualified_symbol="sym",
                selection_reason="reason",
                modification_intent=ModificationIntent.MODIFY_TARGET,
            )


# =============================================================================
# Dimension 5: EU Data Residency Bypass & Location Validation
# =============================================================================


class TestEUResidencyBypass:
    """Stress-tests EU data residency across ScoutSettings and GeminiImplementationExecutor."""

    @pytest.mark.parametrize(
        "forbidden_region",
        [
            "us-central1",
            "us-east4",
            "us-west1",
            "asia-northeast1",
            "asia-south1",
            "southamerica-east1",
            "australia-southeast1",
            "me-central1",
            "africa-south1",
        ],
    )
    def test_scout_settings_rejects_non_eu_regions(self, forbidden_region: str) -> None:
        """ScoutSettings must reject non-EU locations with ValueError under default policy."""
        with pytest.raises(ValueError, match="violates EU data residency policy"):
            ScoutSettings(gcp_location=forbidden_region)

    @pytest.mark.parametrize(
        "forbidden_region",
        [
            "us-central1",
            "us-east1",
            "asia-northeast1",
            "australia-southeast1",
        ],
    )
    def test_gemini_executor_rejects_non_eu_regions(self, forbidden_region: str) -> None:
        """GeminiImplementationExecutor must reject non-EU locations with ValueError."""
        with pytest.raises(ValueError, match="violates EU data residency policy"):
            GeminiImplementationExecutor(location=forbidden_region, vertexai=True)

    def test_environment_variable_non_eu_location_rejected(self, clean_env: None) -> None:
        """SCOUT_GCP_LOCATION and GOOGLE_CLOUD_LOCATION environment overrides are validated."""
        os.environ["SCOUT_GCP_LOCATION"] = "us-central1"
        with pytest.raises(ValueError, match="violates EU data residency policy"):
            ScoutSettings()

        os.environ["GOOGLE_CLOUD_LOCATION"] = "asia-northeast1"
        with pytest.raises(ValueError, match="violates EU data residency policy"):
            GeminiImplementationExecutor(vertexai=True)

    @pytest.mark.parametrize(
        "valid_eu_region",
        [
            "europe-west1",
            "europe-west2",
            "europe-west3",
            "europe-west4",
            "europe-west6",
            "europe-west8",
            "europe-west9",
            "europe-west10",
            "europe-west12",
            "europe-north1",
            "europe-central2",
            "europe-southwest1",
            " EUROPE-WEST3 ",
        ],
    )
    def test_valid_eu_regions_accepted(self, valid_eu_region: str) -> None:
        """Legitimate European regions must be accepted without error."""
        cfg = ScoutSettings(gcp_location=valid_eu_region)
        assert cfg.gcp_location == valid_eu_region.strip().lower()

    def test_explicit_override_allows_non_eu_regions(self) -> None:
        """Non-EU regions are permitted ONLY when allow_non_eu_regions=True is explicit."""
        cfg = ScoutSettings(gcp_location="us-central1", allow_non_eu_regions=True)
        assert cfg.gcp_location == "us-central1"

        executor = GeminiImplementationExecutor(
            location="us-central1", allow_non_eu_regions=True, vertexai=True
        )
        assert executor.location == "us-central1"

    def test_region_prefix_probe(self) -> None:
        """Adversarial Probe: Checks if validate_vertex_location checks ALLOWED_EU_REGIONS.

        In config.py line 53, `cleaned.startswith('europe-')` allows arbitrary strings
        such as 'europe-fake-region' to bypass the ALLOWED_EU_REGIONS frozenset.
        """
        fabricated = "europe-fake-region"
        res = validate_vertex_location(fabricated)
        assert res == "europe-fake-region", "Loose prefix check allows arbitrary europe-* strings"
