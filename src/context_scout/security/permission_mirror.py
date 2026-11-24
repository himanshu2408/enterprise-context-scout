"""Enterprise Permission Mirroring and PostgreSQL RLS Scope Resolution (SEC-02, F02, M1-05)."""

from __future__ import annotations

import contextlib
import json
import re
import socket
import ssl
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from enum import IntEnum, StrEnum
from typing import Any, Final, Protocol, runtime_checkable
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field, field_validator

from context_scout.schemas.contracts import PrincipalContext, RoleType

DEFAULT_ACL_CACHE_TTL_SECONDS: Final[int] = 60
MAX_ACL_CACHE_TTL_SECONDS: Final[int] = 60


class GitLabRole(IntEnum):
    """GitLab project access levels."""

    NO_ACCESS = 0
    MINIMAL_ACCESS = 5
    GUEST = 10
    REPORTER = 20
    DEVELOPER = 30
    MAINTAINER = 40
    OWNER = 50


class GitHubPermission(StrEnum):
    """GitHub repository / GitHub App permission levels."""

    NONE = "none"
    PULL = "pull"
    TRIAGE = "triage"
    PUSH = "push"
    MAINTAIN = "maintain"
    ADMIN = "admin"


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
    read_code: bool | None = None
    repository_access_level: str = "enabled"

    @field_validator("project_path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        return sanitize_scope_identifier(v)


class GitHubRepoMembership(BaseModel):
    repo_full_name: str = Field(min_length=1)  # e.g. "org/repo"
    permission: GitHubPermission | str = GitHubPermission.PULL
    team_slugs: list[str] = Field(default_factory=list)
    installation_id: int | str | None = None
    read_contents: bool = True

    @field_validator("repo_full_name")
    @classmethod
    def _validate_repo(cls, v: str) -> str:
        return sanitize_scope_identifier(v)

    @field_validator("team_slugs")
    @classmethod
    def _validate_teams(cls, v: list[str]) -> list[str]:
        return [sanitize_scope_identifier(slug) for slug in v]


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
    github_memberships: list[GitHubRepoMembership] = Field(default_factory=list)
    confluence_memberships: list[ConfluenceSpaceMembership] = Field(default_factory=list)
    jira_memberships: list[JiraProjectMembership] = Field(default_factory=list)
    custom_scopes: list[str] = Field(default_factory=list)


class PermissionMirror:
    """Translates enterprise identity profiles into canonical PostgreSQL RLS allowed_scopes."""

    @staticmethod
    def map_gitlab_role_to_scopes(
        project_path: str,
        role: GitLabRole,
        *,
        read_code: bool | None = None,
        repository_access_level: str = "enabled",
    ) -> list[str]:
        """Maps GitLab project roles to repo scopes with GitLab 18.7+ :read_code support.

        Security Invariant:
        - Disabled repository_access_level or explicit read_code=False fails closed ([]).
        - NO_ACCESS (0) and MINIMAL_ACCESS (5) always return [].
        - Guest (10) returns [] by default; ONLY if read_code is explicitly True
          (GitLab 18.7+ custom role granting :read_code), grants read-only scope.
        - Reporter (20) receives read-only access.
        - Developer (30) receives read and write access.
        - Maintainer (40) / Owner (50) receive read, write, and admin access.
        """
        clean_path = sanitize_scope_identifier(project_path)
        if repository_access_level.strip().lower() == "disabled" or read_code is False:
            return []
        if role <= GitLabRole.MINIMAL_ACCESS:
            return []
        if role == GitLabRole.GUEST:
            if read_code is True:
                return [f"repo:{clean_path}:read"]
            return []
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
    def map_github_permissions_to_scopes(
        repo_full_name: str,
        permission: GitHubPermission | str,
        *,
        team_slugs: Sequence[str] = (),
        installation_id: int | str | None = None,
        read_contents: bool = True,
    ) -> list[str]:
        """Maps GitHub Enterprise / GitHub App repository and team permissions to scopes."""
        del installation_id  # Validated on membership; not emitted as a wildcard scope.
        clean_repo = sanitize_scope_identifier(repo_full_name)
        if not read_contents:
            return []

        perm = str(permission).strip().lower()
        scopes: set[str] = set()
        if perm in {"pull", "triage", "read"}:
            scopes.add(f"repo:{clean_repo}:read")
        elif perm in {"push", "write"}:
            scopes.add(f"repo:{clean_repo}:read")
            scopes.add(f"repo:{clean_repo}:write")
        elif perm in {"maintain", "admin"}:
            scopes.add(f"repo:{clean_repo}:read")
            scopes.add(f"repo:{clean_repo}:write")
            scopes.add(f"repo:{clean_repo}:admin")
        else:
            return []

        for slug in team_slugs:
            clean_slug = sanitize_scope_identifier(slug)
            scopes.add(f"github:team:{clean_slug}:read")

        return sorted(scopes)

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
    def intersect_confluence_ancestor_scopes(
        space_key: str,
        user_principals: Sequence[str],
        ancestor_restrictions: Sequence[Sequence[str]],
        page_restrictions: Sequence[str] = (),
    ) -> list[str]:
        """Evaluates Confluence V2 recursive ancestor & page VIEW_RESTRICTED intersections.

        Every non-empty restriction list in the ancestor chain (or on the target page)
        must contain at least one principal held by the user; otherwise fails closed ([]).
        """
        clean_space = sanitize_scope_identifier(space_key)
        user_set = {sanitize_scope_identifier(p) for p in user_principals if p.strip()}

        all_levels: list[Sequence[str]] = list(ancestor_restrictions)
        if page_restrictions:
            all_levels.append(page_restrictions)

        for level_principals in all_levels:
            allowed = {sanitize_scope_identifier(r) for r in level_principals if r.strip()}
            if allowed and not user_set.intersection(allowed):
                return []

        return [f"confluence:space:{clean_space}:read"]

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

    @staticmethod
    def resolve_jira_issue_scopes(
        project_key: str,
        project_permissions: Sequence[JiraPermission | str],
        *,
        user_security_levels: Sequence[str] = (),
        issue_security_level: str | None = None,
    ) -> list[str]:
        """Resolves Jira issue scopes enforcing BROWSE + Issue Security Level."""
        base_scopes = PermissionMirror.map_jira_permissions_to_scopes(
            project_key, project_permissions
        )
        if not base_scopes:
            return []

        if issue_security_level is not None and issue_security_level.strip():
            clean_level = sanitize_scope_identifier(issue_security_level)
            user_levels = {
                sanitize_scope_identifier(lvl) for lvl in user_security_levels if lvl.strip()
            }
            if clean_level not in user_levels:
                return []
            return sorted([*base_scopes, f"jira:security_level:{clean_level}:read"])

        return base_scopes

    @staticmethod
    def resolve_jira_comment_scopes(
        project_key: str,
        project_permissions: Sequence[JiraPermission | str],
        *,
        user_roles: Sequence[str] = (),
        user_groups: Sequence[str] = (),
        comment_visibility_type: str | None = None,
        comment_visibility_value: str | None = None,
        user_security_levels: Sequence[str] = (),
        issue_security_level: str | None = None,
    ) -> list[str]:
        """Resolves Jira comment visibility (role/group) on top of issue security scopes."""
        issue_scopes = PermissionMirror.resolve_jira_issue_scopes(
            project_key,
            project_permissions,
            user_security_levels=user_security_levels,
            issue_security_level=issue_security_level,
        )
        if not issue_scopes:
            return []

        if comment_visibility_type is not None and comment_visibility_type.strip():
            vis_type = comment_visibility_type.strip().lower()
            if comment_visibility_value is None or not comment_visibility_value.strip():
                return []
            vis_val = sanitize_scope_identifier(comment_visibility_value)
            if vis_type == "role":
                roles = {sanitize_scope_identifier(r) for r in user_roles if r.strip()}
                if vis_val not in roles:
                    return []
            elif vis_type == "group":
                groups = {sanitize_scope_identifier(g) for g in user_groups if g.strip()}
                if vis_val not in groups:
                    return []
            else:
                return []

        return issue_scopes

    @classmethod
    def resolve_allowed_scopes(cls, profile: EnterpriseIdentityProfile) -> list[str]:
        """Resolves the complete, deduplicated set of RLS allowed_scopes for an enterprise user."""
        all_scopes: set[str] = set()

        for gl in profile.gitlab_memberships:
            all_scopes.update(
                cls.map_gitlab_role_to_scopes(
                    gl.project_path,
                    gl.role,
                    read_code=gl.read_code,
                    repository_access_level=gl.repository_access_level,
                )
            )

        for gh in profile.github_memberships:
            all_scopes.update(
                cls.map_github_permissions_to_scopes(
                    gh.repo_full_name,
                    gh.permission,
                    team_slugs=gh.team_slugs,
                    installation_id=gh.installation_id,
                    read_contents=gh.read_contents,
                )
            )

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


@runtime_checkable
class ACLCacheBackend(Protocol):
    """Protocol for distributed ACL cache backends (e.g. Redis / GCP Memorystore)."""

    def get(self, key: str) -> list[str] | None: ...

    def set(self, key: str, scopes: Sequence[str], ttl_seconds: int) -> None: ...

    def delete(self, key: str) -> None: ...

    def delete_prefix(self, prefix: str) -> int: ...

    def clear(self) -> None: ...


class _StdlibRESP2Client:
    """Thread-safe zero-dependency stdlib socket/ssl RESP2 client for Redis / Memorystore."""

    def __init__(
        self,
        redis_url: str,
        *,
        timeout: float = 2.0,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        parsed = urlparse(redis_url)
        if parsed.scheme not in {"redis", "rediss"}:
            raise ValueError(f"Unsupported Redis URL scheme: '{parsed.scheme}'")
        self._host = parsed.hostname or "localhost"
        self._port = parsed.port or 6379
        self._use_tls = parsed.scheme == "rediss"
        self._username = unquote(parsed.username) if parsed.username else None
        self._password = unquote(parsed.password) if parsed.password else None
        db_str = parsed.path.lstrip("/")
        self._db = int(db_str) if db_str else 0
        self._timeout = timeout
        self._ssl_context = ssl_context
        self._lock = threading.RLock()
        self._sock: socket.socket | None = None
        self._reader: Any = None

    def _connect(self) -> None:
        raw_sock = socket.create_connection((self._host, self._port), timeout=self._timeout)
        if self._use_tls:
            ctx = self._ssl_context or ssl.create_default_context()
            raw_sock = ctx.wrap_socket(raw_sock, server_hostname=self._host)
        self._sock = raw_sock
        self._reader = raw_sock.makefile("rb")
        if self._password is not None:
            if self._username:
                self._execute_unlocked("AUTH", self._username, self._password)
            else:
                self._execute_unlocked("AUTH", self._password)
        if self._db != 0:
            self._execute_unlocked("SELECT", str(self._db))

    def _ensure_connected(self) -> None:
        if self._sock is None or self._reader is None:
            self._connect()

    def _read_line(self) -> bytes:
        line = self._reader.readline()
        if not line or not line.endswith(b"\r\n"):
            raise ConnectionError("Unexpected EOF from Redis server")
        return bytes(line[:-2])

    def _read_response(self) -> Any:
        line = self._read_line()
        prefix = line[:1]
        payload = line[1:]
        if prefix == b"+":
            return payload.decode("utf-8", errors="replace")
        if prefix == b"-":
            raise RuntimeError(f"Redis error: {payload.decode('utf-8', errors='replace')}")
        if prefix == b":":
            return int(payload)
        if prefix == b"$":
            length = int(payload)
            if length == -1:
                return None
            data = self._reader.read(length)
            crlf = self._reader.read(2)
            if len(data) != length or crlf != b"\r\n":
                raise ConnectionError("Truncated RESP2 bulk string")
            return bytes(data).decode("utf-8")
        if prefix == b"*":
            count = int(payload)
            if count == -1:
                return None
            return [self._read_response() for _ in range(count)]
        raise RuntimeError(f"Invalid RESP2 type prefix: {prefix!r}")

    def _execute_unlocked(self, *args: str) -> Any:
        self._ensure_connected()
        encoded_parts = [f"*{len(args)}\r\n".encode("ascii")]
        for arg in args:
            raw = arg.encode("utf-8")
            encoded_parts.append(f"${len(raw)}\r\n".encode("ascii"))
            encoded_parts.append(raw + b"\r\n")
        assert self._sock is not None
        self._sock.sendall(b"".join(encoded_parts))
        return self._read_response()

    def _execute(self, *args: str) -> Any:
        with self._lock:
            try:
                return self._execute_unlocked(*args)
            except (OSError, ConnectionError):
                self.close()
                self._connect()
                return self._execute_unlocked(*args)

    def get(self, key: str) -> str | None:
        res = self._execute("GET", key)
        return str(res) if res is not None else None

    def set(self, key: str, value: str, *, ex: int | None = None) -> Any:
        if ex is not None:
            return self.setex(key, int(ex), value)
        return self._execute("SET", key, value)

    def setex(self, key: str, ttl_seconds: int, value: str) -> Any:
        return self._execute("SETEX", key, str(ttl_seconds), value)

    def incrby(self, key: str, amount: int = 1) -> int:
        return int(self._execute("INCRBY", key, str(int(amount))) or 0)

    def expire(self, key: str, ttl_seconds: int) -> int:
        return int(self._execute("EXPIRE", key, str(max(1, int(ttl_seconds)))) or 0)

    def ping(self) -> str:
        return str(self._execute("PING") or "")

    def flushdb(self) -> str:
        return str(self._execute("FLUSHDB") or "")

    def flushall(self) -> str:
        return str(self._execute("FLUSHALL") or "")

    def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        return int(self._execute("DEL", *keys) or 0)

    def scan_iter(self, match: str = "*", count: int = 100) -> list[str]:
        cursor = "0"
        collected: list[str] = []
        while True:
            res = self._execute("SCAN", cursor, "MATCH", match, "COUNT", str(count))
            if not isinstance(res, list) or len(res) != 2:
                break
            next_cursor, batch = res[0], res[1]
            cursor = str(next_cursor)
            if isinstance(batch, list):
                collected.extend(str(item) for item in batch)
            if cursor == "0":
                break
        return collected

    def keys(self, pattern: str = "*") -> list[str]:
        res = self._execute("KEYS", pattern)
        if isinstance(res, list):
            return [str(item) for item in res]
        return []

    def close(self) -> None:
        with self._lock:
            if self._reader is not None:
                with contextlib.suppress(OSError):
                    self._reader.close()
                self._reader = None
            if self._sock is not None:
                with contextlib.suppress(OSError):
                    self._sock.close()
                self._sock = None


class RedisACLCacheBackend:
    """Redis / GCP Memorystore ACL cache backend with <= 60s clamped TTL."""

    def __init__(
        self,
        client: Any | None = None,
        redis_url: str | None = None,
        *,
        key_prefix: str = "",
        socket_timeout: float = 2.0,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if client is None and redis_url is None:
            raise ValueError("Either client or redis_url must be provided")
        self.key_prefix = key_prefix
        self._client: Any = (
            client
            if client is not None
            else _StdlibRESP2Client(
                redis_url,  # type: ignore[arg-type]
                timeout=socket_timeout,
                ssl_context=ssl_context,
            )
        )

    def _full_key(self, key: str) -> str:
        if self.key_prefix and not key.startswith(self.key_prefix):
            return f"{self.key_prefix}{key}"
        return key

    def get(self, key: str) -> list[str] | None:
        raw = self._client.get(self._full_key(key))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        decoded = json.loads(raw)
        if not isinstance(decoded, list):
            return None
        return [str(item) for item in decoded]

    def set(self, key: str, scopes: Sequence[str], ttl_seconds: int) -> None:
        clamped_ttl = max(1, min(int(ttl_seconds), MAX_ACL_CACHE_TTL_SECONDS))
        payload = json.dumps([str(s) for s in scopes])
        full_key = self._full_key(key)
        if hasattr(self._client, "setex"):
            self._client.setex(full_key, clamped_ttl, payload)
        else:
            self._client.set(full_key, payload, ex=clamped_ttl)

    def delete(self, key: str) -> None:
        self._client.delete(self._full_key(key))

    def delete_prefix(self, prefix: str) -> int:
        pattern = f"{self._full_key(prefix)}*"
        if hasattr(self._client, "scan_iter"):
            matched = list(self._client.scan_iter(match=pattern))
        else:
            matched = list(self._client.keys(pattern))
        if not matched:
            return 0
        normalized = [k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in matched]
        return int(self._client.delete(*normalized) or 0)

    def clear(self) -> None:
        self.delete_prefix(self.key_prefix)


def _extract_str(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for k in keys:
        val = mapping.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, int):
            return str(val)
    return None


class CachedPermissionMirrorService:
    """Thread-safe caching service with <= 60s TTL, Redis backend, and webhook invalidation."""

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_ACL_CACHE_TTL_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
        redis_backend: ACLCacheBackend | None = None,
        redis_client: Any | None = None,
        redis_url: str | None = None,
        key_prefix: str = "scout:acl:",
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.ttl_seconds: int = min(int(ttl_seconds), MAX_ACL_CACHE_TTL_SECONDS)
        self._clock = clock
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[float, list[str]]] = {}
        self._scope_to_users: dict[str, set[str]] = {}
        self.key_prefix = key_prefix
        if redis_backend is not None:
            self._backend: ACLCacheBackend | None = redis_backend
        elif redis_client is not None or redis_url is not None:
            self._backend = RedisACLCacheBackend(client=redis_client, redis_url=redis_url)
        else:
            self._backend = None

    def _cache_key(self, user_id: str) -> str:
        return f"{self.key_prefix}{user_id}"

    def _remove_user_from_reverse_index(self, user_id: str) -> None:
        entry = self._cache.pop(user_id, None)
        if entry is None:
            return
        _, old_scopes = entry
        for scope in old_scopes:
            users = self._scope_to_users.get(scope)
            if users is not None:
                users.discard(user_id)
                if not users:
                    self._scope_to_users.pop(scope, None)

    def _store_in_memory(self, user_id: str, ts: float, scopes: Sequence[str]) -> list[str]:
        self._remove_user_from_reverse_index(user_id)
        copied = list(scopes)
        self._cache[user_id] = (ts, copied)
        for scope in copied:
            self._scope_to_users.setdefault(scope, set()).add(user_id)
        return list(copied)

    def get_user_scopes(
        self, profile: EnterpriseIdentityProfile, force_refresh: bool = False
    ) -> list[str]:
        with self._lock:
            now = self._clock()
            cache_key = self._cache_key(profile.user_id)

            if not force_refresh:
                cached = self._cache.get(profile.user_id)
                if cached is not None:
                    ts, scopes = cached
                    if now - ts < self.ttl_seconds:
                        return list(scopes)
                    self._remove_user_from_reverse_index(profile.user_id)

                if self._backend is not None:
                    backend_scopes = self._backend.get(cache_key)
                    if backend_scopes is not None:
                        return self._store_in_memory(profile.user_id, now, backend_scopes)

            resolved = PermissionMirror.resolve_allowed_scopes(profile)
            stored = self._store_in_memory(profile.user_id, now, resolved)
            if self._backend is not None:
                self._backend.set(cache_key, stored, self.ttl_seconds)
            return list(stored)

    def invalidate_user(self, user_id: str) -> None:
        """Called upon receiving GitLab/GitHub/Confluence/Jira membership change webhooks."""
        with self._lock:
            self._remove_user_from_reverse_index(user_id)
            if self._backend is not None:
                self._backend.delete(self._cache_key(user_id))

    def invalidate_scope_prefix(self, scope_prefix: str) -> int:
        """Evicts all cached users holding any scope matching scope_prefix."""
        with self._lock:
            affected_users: set[str] = set()
            for scope, users in self._scope_to_users.items():
                if scope.startswith(scope_prefix):
                    affected_users.update(users)
            for user_id in affected_users:
                self.invalidate_user(user_id)
            return len(affected_users)

    def process_revocation_webhook(self, source_type: str, payload: Mapping[str, Any]) -> list[str]:
        """Processes membership/permission/restriction revocation webhooks across providers."""
        src = source_type.strip().lower()
        supported = {
            "gitlab",
            "gitlab_enterprise",
            "github",
            "github_app",
            "github_enterprise",
            "ghe",
            "confluence",
            "confluence_cloud",
            "confluence_dc",
            "jira",
            "jira_cloud",
            "jira_dc",
        }
        if src not in supported:
            raise ValueError(f"Unsupported revocation webhook source_type: '{source_type}'")

        with self._lock:
            invalidated: set[str] = set()
            scope_prefixes: list[str] = []

            direct_user = _extract_str(
                payload, "user_id", "user_username", "username", "account_id"
            )
            if direct_user:
                invalidated.add(direct_user)

            user_obj = payload.get("user")
            if isinstance(user_obj, Mapping):
                nested_user = _extract_str(
                    user_obj, "user_id", "username", "login", "accountId", "name"
                )
                if nested_user:
                    invalidated.add(nested_user)

            if src in {"gitlab", "gitlab_enterprise"}:
                project_path = _extract_str(payload, "project_path", "path_with_namespace")
                if not project_path and isinstance(payload.get("project"), Mapping):
                    project_path = _extract_str(payload["project"], "path_with_namespace", "path")
                if project_path:
                    scope_prefixes.append(f"repo:{sanitize_scope_identifier(project_path)}:")

                group_path = _extract_str(payload, "group_path", "full_path")
                if group_path:
                    scope_prefixes.append(f"repo:{sanitize_scope_identifier(group_path)}/")

            elif src in {"github", "github_app", "github_enterprise", "ghe"}:
                member_obj = payload.get("member")
                if isinstance(member_obj, Mapping):
                    member_login = _extract_str(member_obj, "login", "username")
                    if member_login:
                        invalidated.add(member_login)
                membership_obj = payload.get("membership")
                if isinstance(membership_obj, Mapping) and isinstance(
                    membership_obj.get("user"), Mapping
                ):
                    m_login = _extract_str(membership_obj["user"], "login", "username")
                    if m_login:
                        invalidated.add(m_login)

                repo_name = _extract_str(payload, "repo_full_name", "repository_full_name")
                if not repo_name and isinstance(payload.get("repository"), Mapping):
                    repo_name = _extract_str(payload["repository"], "full_name")
                if repo_name:
                    scope_prefixes.append(f"repo:{sanitize_scope_identifier(repo_name)}:")

                removed_repos = payload.get("repositories_removed")
                if isinstance(removed_repos, Sequence) and not isinstance(removed_repos, str):
                    for item in removed_repos:
                        if isinstance(item, Mapping):
                            r_name = _extract_str(item, "full_name")
                            if r_name:
                                scope_prefixes.append(f"repo:{sanitize_scope_identifier(r_name)}:")
                        elif isinstance(item, str) and item.strip():
                            scope_prefixes.append(f"repo:{sanitize_scope_identifier(item)}:")

                team_obj = payload.get("team")
                if isinstance(team_obj, Mapping):
                    team_slug = _extract_str(team_obj, "slug")
                    if team_slug:
                        scope_prefixes.append(
                            f"github:team:{sanitize_scope_identifier(team_slug)}:"
                        )

            elif src in {"confluence", "confluence_cloud", "confluence_dc"}:
                space_key = _extract_str(payload, "space_key", "spaceKey")
                if not space_key and isinstance(payload.get("space"), Mapping):
                    space_key = _extract_str(payload["space"], "key")
                if space_key:
                    scope_prefixes.append(
                        f"confluence:space:{sanitize_scope_identifier(space_key)}:"
                    )

            elif src in {"jira", "jira_cloud", "jira_dc"}:
                project_key = _extract_str(payload, "project_key", "projectKey")
                if not project_key and isinstance(payload.get("project"), Mapping):
                    project_key = _extract_str(payload["project"], "key")
                if project_key:
                    scope_prefixes.append(f"jira:project:{sanitize_scope_identifier(project_key)}:")

            for prefix in scope_prefixes:
                for scope, users in list(self._scope_to_users.items()):
                    if scope.startswith(prefix):
                        invalidated.update(users)

            for uid in invalidated:
                self.invalidate_user(uid)

            return sorted(invalidated)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._scope_to_users.clear()
            if self._backend is not None:
                self._backend.clear()
