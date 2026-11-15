"""Enterprise Permission Mirroring and PostgreSQL RLS Scope Resolution (SEC-02, F02)."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from enum import IntEnum, StrEnum
from typing import Final

from pydantic import BaseModel, Field, field_validator

from context_scout.schemas.contracts import PrincipalContext, RoleType


class GitLabRole(IntEnum):
    """GitLab project access levels."""

    NO_ACCESS = 0
    MINIMAL_ACCESS = 5
    GUEST = 10
    REPORTER = 20
    DEVELOPER = 30
    MAINTAINER = 40
    OWNER = 50


class ConfluencePermission(StrEnum):
    """Confluence space permission types."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class JiraPermission(StrEnum):
    """Jira project permission types."""

    BROWSE = "browse"
    WRITE = "write"
    ADMIN = "admin"


_SCOPE_CHAR_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9_\-\./:]+$")


def sanitize_scope_identifier(raw: str) -> str:
    """Sanitizes workspace/repo/space/project identifiers against scope injection."""
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("Scope identifier cannot be empty")
    if not _SCOPE_CHAR_PATTERN.match(cleaned):
        raise ValueError(
            f"Invalid characters detected in scope identifier '{raw}'. "
            "Must match ^[a-zA-Z0-9_\\-\\./:]+$ (no commas, semicolons, or spaces)."
        )
    return cleaned


class GitLabProjectMembership(BaseModel):
    project_path: str = Field(min_length=1)  # e.g. "gitlab-org/gitlab"
    role: GitLabRole

    @field_validator("project_path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        return sanitize_scope_identifier(v)


class ConfluenceSpaceMembership(BaseModel):
    space_key: str = Field(min_length=1)  # e.g. "ENG", "PROD"
    permissions: list[ConfluencePermission] = Field(min_length=1)

    @field_validator("space_key")
    @classmethod
    def _validate_key(cls, v: str) -> str:
        return sanitize_scope_identifier(v)


class JiraProjectMembership(BaseModel):
    project_key: str = Field(min_length=1)  # e.g. "CORE", "TRAD"
    permissions: list[JiraPermission] = Field(min_length=1)

    @field_validator("project_key")
    @classmethod
    def _validate_key(cls, v: str) -> str:
        return sanitize_scope_identifier(v)


class EnterpriseIdentityProfile(BaseModel):
    """Aggregated enterprise identity containing all provider memberships."""

    user_id: str = Field(min_length=1)
    gitlab_memberships: list[GitLabProjectMembership] = Field(default_factory=list)
    confluence_memberships: list[ConfluenceSpaceMembership] = Field(default_factory=list)
    jira_memberships: list[JiraProjectMembership] = Field(default_factory=list)
    custom_scopes: list[str] = Field(default_factory=list)


class PermissionMirror:
    """Translates enterprise identity profiles into canonical PostgreSQL RLS allowed_scopes."""

    @staticmethod
    def map_gitlab_role_to_scopes(project_path: str, role: GitLabRole) -> list[str]:
        """Maps GitLab project roles to repo scopes.

        Security Invariant:
        - Guest (10) has NO read access to private repository code (returns []).
        - Reporter (20) receives read-only access.
        - Developer (30) receives read and write access.
        - Maintainer (40) / Owner (50) receive read, write, and admin access.
        """
        clean_path = sanitize_scope_identifier(project_path)
        if role < GitLabRole.REPORTER:
            return []  # Guests and minimal access have zero code access in private repos
        if role == GitLabRole.REPORTER:
            return [f"repo:{clean_path}:read"]
        if role == GitLabRole.DEVELOPER:
            return [f"repo:{clean_path}:read", f"repo:{clean_path}:write"]
        # Maintainer (40) and Owner (50)
        return [
            f"repo:{clean_path}:read",
            f"repo:{clean_path}:write",
            f"repo:{clean_path}:admin",
        ]

    @staticmethod
    def map_confluence_permissions_to_scopes(
        space_key: str, permissions: Sequence[ConfluencePermission | str]
    ) -> list[str]:
        """Maps Confluence space permissions to space scopes."""
        clean_key = sanitize_scope_identifier(space_key)
        scopes: set[str] = set()
        perm_set = {str(p).lower() for p in permissions}
        if "read" in perm_set or "view" in perm_set:
            scopes.add(f"confluence:space:{clean_key}:read")
        if "write" in perm_set or "edit" in perm_set:
            scopes.add(f"confluence:space:{clean_key}:read")
            scopes.add(f"confluence:space:{clean_key}:write")
        if "admin" in perm_set:
            scopes.add(f"confluence:space:{clean_key}:read")
            scopes.add(f"confluence:space:{clean_key}:write")
            scopes.add(f"confluence:space:{clean_key}:admin")
        return sorted(scopes)

    @staticmethod
    def map_jira_permissions_to_scopes(
        project_key: str, permissions: Sequence[JiraPermission | str]
    ) -> list[str]:
        """Maps Jira project permissions to project scopes."""
        clean_key = sanitize_scope_identifier(project_key)
        scopes: set[str] = set()
        perm_set = {str(p).lower() for p in permissions}
        if "browse" in perm_set or "read" in perm_set or "view" in perm_set:
            scopes.add(f"jira:project:{clean_key}:read")
        if "write" in perm_set or "create" in perm_set or "edit" in perm_set:
            scopes.add(f"jira:project:{clean_key}:read")
            scopes.add(f"jira:project:{clean_key}:write")
        if "admin" in perm_set:
            scopes.add(f"jira:project:{clean_key}:read")
            scopes.add(f"jira:project:{clean_key}:write")
            scopes.add(f"jira:project:{clean_key}:admin")
        return sorted(scopes)

    @classmethod
    def resolve_allowed_scopes(cls, profile: EnterpriseIdentityProfile) -> list[str]:
        """Resolves the complete, deduplicated set of RLS allowed_scopes for an enterprise user."""
        all_scopes: set[str] = set()

        for gl in profile.gitlab_memberships:
            all_scopes.update(cls.map_gitlab_role_to_scopes(gl.project_path, gl.role))

        for conf in profile.confluence_memberships:
            all_scopes.update(
                cls.map_confluence_permissions_to_scopes(conf.space_key, conf.permissions)
            )

        for jira in profile.jira_memberships:
            all_scopes.update(
                cls.map_jira_permissions_to_scopes(jira.project_key, jira.permissions)
            )

        for custom in profile.custom_scopes:
            clean_custom = sanitize_scope_identifier(custom)
            all_scopes.add(clean_custom)

        return sorted(all_scopes)

    @classmethod
    def create_principal_context(
        cls,
        profile: EnterpriseIdentityProfile,
        role: RoleType = RoleType.ENGINEER,
        max_session_paid_tokens: int = 16_000,
    ) -> PrincipalContext:
        """Constructs an authenticated PrincipalContext.

        Fail-Closed Security:
        If user holds zero permitted scopes, populates a dummy unprivileged scope
        that matches 0 rows in PostgreSQL RLS.
        """
        scopes = cls.resolve_allowed_scopes(profile)
        if not scopes:
            # Provide an unprivileged isolation scope so PrincipalContext passes validation
            # while guaranteeing 0 rows match in RLS.
            scopes = [f"unprivileged:user:{profile.user_id}"]

        return PrincipalContext(
            user_id=profile.user_id,
            role=role,
            allowed_scopes=scopes,
            max_session_paid_tokens=max_session_paid_tokens,
        )


class CachedPermissionMirrorService:
    """Thread-safe caching service with TTL and webhook-driven cache invalidation."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, list[str]]] = {}

    def get_user_scopes(
        self, profile: EnterpriseIdentityProfile, force_refresh: bool = False
    ) -> list[str]:
        now = time.time()
        if not force_refresh and profile.user_id in self._cache:
            ts, scopes = self._cache[profile.user_id]
            if now - ts < self.ttl_seconds:
                return scopes

        resolved = PermissionMirror.resolve_allowed_scopes(profile)
        self._cache[profile.user_id] = (now, resolved)
        return resolved

    def invalidate_user(self, user_id: str) -> None:
        """Called upon receiving GitLab/Confluence/Jira membership change webhooks."""
        self._cache.pop(user_id, None)

    def clear(self) -> None:
        self._cache.clear()
