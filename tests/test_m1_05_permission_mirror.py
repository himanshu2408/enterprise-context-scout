"""M1-05 Permission Mirror tests: RLock concurrency, <=60s TTL, Redis RESP2, and webhooks."""

from __future__ import annotations

import fnmatch
import socket
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from context_scout.schemas.contracts import DEFAULT_TENANT_ID
from context_scout.security.permission_mirror import (
    DEFAULT_ACL_CACHE_TTL_SECONDS,
    MAX_ACL_CACHE_TTL_SECONDS,
    ACLCacheBackend,
    CachedPermissionMirrorService,
    ConfluencePermission,
    ConfluenceSpaceMembership,
    EnterpriseIdentityProfile,
    GitHubPermission,
    GitHubRepoMembership,
    GitLabProjectMembership,
    GitLabRole,
    JiraPermission,
    JiraProjectMembership,
    PermissionMirror,
    RedisACLCacheBackend,
)


class _FakeClock:
    def __init__(self, initial: float = 1000.0) -> None:
        self.now = initial

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _MockRedisClient:
    """In-memory mock mimicking redis-py client methods."""

    def __init__(self) -> None:
        self.store: dict[str, tuple[int, str]] = {}
        self.last_ttl: dict[str, int] = {}

    def get(self, key: str) -> str | None:
        item = self.store.get(key)
        return item[1] if item is not None else None

    def setex(self, key: str, ttl_seconds: int, value: str) -> bool:
        self.store[key] = (ttl_seconds, value)
        self.last_ttl[key] = ttl_seconds
        return True

    def delete(self, *keys: str) -> int:
        deleted = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                deleted += 1
        return deleted

    def scan_iter(self, match: str = "*") -> list[str]:
        return [k for k in list(self.store.keys()) if fnmatch.fnmatch(k, match)]


class _LocalRESP2TestServer:
    """Minimal thread-safe RESP2 TCP server for testing _StdlibRESP2Client."""

    def __init__(self, expected_password: str = "secret") -> None:
        self.expected_password = expected_password
        self._lock = threading.Lock()
        self._dbs: dict[int, dict[str, str]] = {0: {}}
        self._ttls: dict[str, int] = {}
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        self._sock.settimeout(0.1)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            t = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
            t.start()
            self._threads.append(t)

    @staticmethod
    def _read_resp_command(reader: Any) -> list[str] | None:
        header = reader.readline()
        if not header:
            return None
        if not header.startswith(b"*"):
            raise ValueError(f"Unexpected header: {header!r}")
        count = int(header[1:-2])
        args: list[str] = []
        for _ in range(count):
            len_line = reader.readline()
            length = int(len_line[1:-2])
            data = reader.read(length)
            reader.read(2)  # CRLF
            args.append(data.decode("utf-8"))
        return args

    def _handle_client(self, conn: socket.socket) -> None:
        current_db = 0
        authed = self.expected_password == ""
        reader = conn.makefile("rb")
        try:
            while not self._stop.is_set():
                cmd = self._read_resp_command(reader)
                if cmd is None:
                    break
                op = cmd[0].upper()
                if op == "AUTH":
                    pw = cmd[-1]
                    if pw == self.expected_password:
                        authed = True
                        conn.sendall(b"+OK\r\n")
                    else:
                        conn.sendall(b"-ERR invalid password\r\n")
                elif not authed:
                    conn.sendall(b"-NOAUTH Authentication required\r\n")
                elif op == "SELECT":
                    current_db = int(cmd[1])
                    with self._lock:
                        self._dbs.setdefault(current_db, {})
                    conn.sendall(b"+OK\r\n")
                elif op == "GET":
                    with self._lock:
                        val = self._dbs.get(current_db, {}).get(cmd[1])
                    if val is None:
                        conn.sendall(b"$-1\r\n")
                    else:
                        raw = val.encode("utf-8")
                        conn.sendall(f"${len(raw)}\r\n".encode("ascii") + raw + b"\r\n")
                elif op == "SETEX":
                    key, ttl, val = cmd[1], int(cmd[2]), cmd[3]
                    with self._lock:
                        self._dbs.setdefault(current_db, {})[key] = val
                        self._ttls[key] = ttl
                    conn.sendall(b"+OK\r\n")
                elif op == "DEL":
                    deleted = 0
                    with self._lock:
                        db = self._dbs.setdefault(current_db, {})
                        for k in cmd[1:]:
                            if k in db:
                                del db[k]
                                deleted += 1
                    conn.sendall(f":{deleted}\r\n".encode("ascii"))
                elif op == "SCAN":
                    pattern = "*"
                    if "MATCH" in cmd:
                        pattern = cmd[cmd.index("MATCH") + 1]
                    with self._lock:
                        keys = [
                            k for k in self._dbs.get(current_db, {}) if fnmatch.fnmatch(k, pattern)
                        ]
                    parts = [b"*2\r\n$1\r\n0\r\n", f"*{len(keys)}\r\n".encode("ascii")]
                    for k in keys:
                        kb = k.encode("utf-8")
                        parts.append(f"${len(kb)}\r\n".encode("ascii") + kb + b"\r\n")
                    conn.sendall(b"".join(parts))
                elif op == "KEYS":
                    pattern = cmd[1]
                    with self._lock:
                        keys = [
                            k for k in self._dbs.get(current_db, {}) if fnmatch.fnmatch(k, pattern)
                        ]
                    parts = [f"*{len(keys)}\r\n".encode("ascii")]
                    for k in keys:
                        kb = k.encode("utf-8")
                        parts.append(f"${len(kb)}\r\n".encode("ascii") + kb + b"\r\n")
                    conn.sendall(b"".join(parts))
                else:
                    conn.sendall(f"-ERR unknown command '{op}'\r\n".encode())
        finally:
            reader.close()
            conn.close()

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
        self._accept_thread.join(timeout=1.0)


def test_ttl_validation_clamping_and_monotonic_expiration() -> None:
    assert DEFAULT_ACL_CACHE_TTL_SECONDS == 60
    assert MAX_ACL_CACHE_TTL_SECONDS == 60

    with pytest.raises(ValueError, match="positive"):
        CachedPermissionMirrorService(ttl_seconds=0)
    with pytest.raises(ValueError, match="positive"):
        CachedPermissionMirrorService(ttl_seconds=-5)

    clock = _FakeClock(100.0)
    svc = CachedPermissionMirrorService(ttl_seconds=300, clock=clock)
    assert svc.ttl_seconds == 60

    profile = EnterpriseIdentityProfile(
        user_id="alice",
        gitlab_memberships=[
            GitLabProjectMembership(project_path="org/repo", role=GitLabRole.REPORTER)
        ],
    )
    scopes_1 = svc.get_user_scopes(profile)
    assert scopes_1 == ["repo:org/repo:read"]

    # Defensive copy check: mutating returned list must not corrupt cache
    scopes_1.append("repo:org/evil:admin")
    assert svc.get_user_scopes(profile) == ["repo:org/repo:read"]

    # Mutate profile and advance clock just before 60s -> still cached
    profile.gitlab_memberships[0].role = GitLabRole.DEVELOPER
    clock.advance(59.9)
    assert svc.get_user_scopes(profile) == ["repo:org/repo:read"]

    # Advance past 60s -> expired, re-resolves
    clock.advance(0.2)
    assert svc.get_user_scopes(profile) == ["repo:org/repo:read", "repo:org/repo:write"]


def test_thread_pool_rlock_stress_concurrency() -> None:
    svc = CachedPermissionMirrorService(ttl_seconds=60)
    profiles = [
        EnterpriseIdentityProfile(
            user_id=f"user-{i}",
            gitlab_memberships=[
                GitLabProjectMembership(project_path=f"org/repo-{i % 4}", role=GitLabRole.DEVELOPER)
            ],
        )
        for i in range(12)
    ]

    def worker(idx: int) -> None:
        p = profiles[idx % len(profiles)]
        for step in range(25):
            scopes = svc.get_user_scopes(p)
            assert any(s.startswith("repo:org/repo-") for s in scopes)
            if step % 7 == 0:
                svc.invalidate_user(p.user_id)
            elif step % 11 == 0:
                svc.invalidate_scope_prefix(f"repo:org/repo-{idx % 4}:")

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(worker, i) for i in range(24)]
        for fut in futures:
            fut.result()


def test_redis_acl_cache_backend_mock_and_ttl_clamping() -> None:
    mock_client = _MockRedisClient()
    backend = RedisACLCacheBackend(client=mock_client)
    assert isinstance(backend, ACLCacheBackend)

    svc = CachedPermissionMirrorService(ttl_seconds=300, redis_backend=backend)
    profile = EnterpriseIdentityProfile(
        user_id="carol",
        github_memberships=[
            GitHubRepoMembership(repo_full_name="acme/api", permission=GitHubPermission.MAINTAIN)
        ],
    )
    scopes = svc.get_user_scopes(profile)
    assert scopes == ["repo:acme/api:admin", "repo:acme/api:read", "repo:acme/api:write"]
    expected_key = f"scout:acl:{DEFAULT_TENANT_ID}:carol"
    assert mock_client.last_ttl[expected_key] == 60

    # New service instance with empty local cache reads from Redis backend
    svc_2 = CachedPermissionMirrorService(ttl_seconds=60, redis_backend=backend)
    assert svc_2.get_user_scopes(profile) == scopes

    svc_2.invalidate_user("carol")
    assert mock_client.get(expected_key) is None


def test_cross_tenant_identical_user_id_non_collision_in_redis_and_memory() -> None:
    """ECS-031: Tenants A and B sharing identical user_id never collide in ACL cache."""
    server = _LocalRESP2TestServer(expected_password="s3cr3t")
    try:
        url = f"redis://:s3cr3t@127.0.0.1:{server.port}/3"
        svc_a = CachedPermissionMirrorService(ttl_seconds=60, redis_url=url)
        svc_b = CachedPermissionMirrorService(ttl_seconds=60, redis_url=url)
        tenant_a = uuid.uuid4()
        tenant_b = uuid.uuid4()

        prof_a = EnterpriseIdentityProfile(
            user_id="alice",
            tenant_id=tenant_a,
            gitlab_memberships=[
                GitLabProjectMembership(project_path="tenant-a/core", role=GitLabRole.DEVELOPER)
            ],
        )
        prof_b = EnterpriseIdentityProfile(
            user_id="alice",
            tenant_id=tenant_b,
            github_memberships=[
                GitHubRepoMembership(
                    repo_full_name="tenant-b/secret", permission=GitHubPermission.ADMIN
                )
            ],
        )
        scopes_a = svc_a.get_user_scopes(prof_a)
        scopes_b = svc_a.get_user_scopes(prof_b)
        assert scopes_a == ["repo:tenant-a/core:read", "repo:tenant-a/core:write"]
        assert scopes_b == [
            "repo:tenant-b/secret:admin",
            "repo:tenant-b/secret:read",
            "repo:tenant-b/secret:write",
        ]
        assert f"scout:acl:{tenant_a}:alice" in server._dbs[3]
        assert f"scout:acl:{tenant_b}:alice" in server._dbs[3]
        assert PermissionMirror.create_principal_context(prof_a).tenant_id == tenant_a
        assert PermissionMirror.create_principal_context(prof_b).tenant_id == tenant_b

        # Cold second service reads isolated per-tenant ACLs from Redis
        assert svc_b.get_user_scopes(prof_a) == scopes_a
        assert svc_b.get_user_scopes(prof_b) == scopes_b

        # Evicting tenant_a leaves tenant_b untouched in both memory & Redis
        svc_a.invalidate_user("alice", tenant_id=tenant_a)
        assert f"scout:acl:{tenant_a}:alice" not in server._dbs[3]
        assert f"scout:acl:{tenant_b}:alice" in server._dbs[3]
        assert f"{tenant_a}:alice" not in svc_a._cache
        assert f"{tenant_b}:alice" in svc_a._cache
    finally:
        server.close()


def test_redis_resp2_socket_server_round_trip() -> None:
    server = _LocalRESP2TestServer(expected_password="s3cr3t")
    try:
        url = f"redis://:s3cr3t@127.0.0.1:{server.port}/2"
        svc = CachedPermissionMirrorService(ttl_seconds=60, redis_url=url)
        profile = EnterpriseIdentityProfile(
            user_id="dave",
            confluence_memberships=[
                ConfluenceSpaceMembership(space_key="ARCH", permissions=[ConfluencePermission.READ])
            ],
        )
        assert svc.get_user_scopes(profile) == ["confluence:space:ARCH:read"]

        # Second service reads from the same RESP2 server DB 2
        svc_2 = CachedPermissionMirrorService(ttl_seconds=60, redis_url=url)
        assert svc_2.get_user_scopes(profile) == ["confluence:space:ARCH:read"]

        # Prefix invalidation evicts both local cache and Redis key
        evicted = svc_2.invalidate_scope_prefix("confluence:space:ARCH:")
        assert evicted == 1
        assert server._dbs[2] == {}

        # Clear works cleanly
        svc.get_user_scopes(profile)
        svc.clear()
        assert server._dbs[2] == {}
    finally:
        server.close()


def test_revocation_webhooks_across_all_sources() -> None:
    mock_client = _MockRedisClient()
    svc = CachedPermissionMirrorService(ttl_seconds=60, redis_client=mock_client)

    p_alice = EnterpriseIdentityProfile(
        user_id="alice",
        gitlab_memberships=[
            GitLabProjectMembership(project_path="fin/ledger", role=GitLabRole.DEVELOPER)
        ],
        confluence_memberships=[
            ConfluenceSpaceMembership(space_key="SEC", permissions=[ConfluencePermission.READ])
        ],
    )
    p_bob = EnterpriseIdentityProfile(
        user_id="bob",
        gitlab_memberships=[
            GitLabProjectMembership(project_path="fin/ledger", role=GitLabRole.REPORTER)
        ],
        github_memberships=[
            GitHubRepoMembership(
                repo_full_name="acme/vault",
                permission=GitHubPermission.PUSH,
                team_slugs=["sec-ops"],
            )
        ],
        jira_memberships=[
            JiraProjectMembership(project_key="COMP", permissions=[JiraPermission.BROWSE])
        ],
    )
    svc.get_user_scopes(p_alice)
    svc.get_user_scopes(p_bob)

    # GitLab project revocation evicts both alice & bob via reverse index
    evicted_gl = svc.process_revocation_webhook(
        "gitlab", {"project": {"path_with_namespace": "fin/ledger"}}
    )
    assert evicted_gl == ["alice", "bob"]
    assert mock_client.get(f"scout:acl:{DEFAULT_TENANT_ID}:alice") is None
    assert mock_client.get(f"scout:acl:{DEFAULT_TENANT_ID}:bob") is None

    # Re-populate and test GitHub team/repo revocation
    svc.get_user_scopes(p_bob)
    evicted_gh = svc.process_revocation_webhook(
        "github_app",
        {"repository": {"full_name": "acme/vault"}, "member": {"login": "bob"}},
    )
    assert evicted_gh == ["bob"]

    # Confluence space revocation
    svc.get_user_scopes(p_alice)
    evicted_conf = svc.process_revocation_webhook("confluence", {"space": {"key": "SEC"}})
    assert evicted_conf == ["alice"]

    # Jira project revocation
    svc.get_user_scopes(p_bob)
    evicted_jira = svc.process_revocation_webhook("jira", {"project": {"key": "COMP"}})
    assert evicted_jira == ["bob"]

    with pytest.raises(ValueError, match="Unsupported revocation webhook"):
        svc.process_revocation_webhook("unknown_provider", {})


def test_transitive_acl_edge_cases() -> None:
    # GitLab: NO_ACCESS and MINIMAL_ACCESS ignore read_code=True
    assert (
        PermissionMirror.map_gitlab_role_to_scopes(
            "org/repo", GitLabRole.MINIMAL_ACCESS, read_code=True
        )
        == []
    )
    # GitHub: unknown permission or NONE fails closed
    assert (
        PermissionMirror.map_github_permissions_to_scopes("org/repo", GitHubPermission.NONE) == []
    )
    assert PermissionMirror.map_github_permissions_to_scopes("org/repo", "unknown_perm") == []

    # Jira comment visibility: group & missing value fail closed
    assert (
        PermissionMirror.resolve_jira_comment_scopes(
            "CORE",
            [JiraPermission.BROWSE],
            user_groups=["eng"],
            comment_visibility_type="group",
            comment_visibility_value="security-team",
        )
        == []
    )
    assert PermissionMirror.resolve_jira_comment_scopes(
        "CORE",
        [JiraPermission.BROWSE],
        user_groups=["security-team"],
        comment_visibility_type="group",
        comment_visibility_value="security-team",
    ) == ["jira:project:CORE:read"]
    assert (
        PermissionMirror.resolve_jira_comment_scopes(
            "CORE",
            [JiraPermission.BROWSE],
            comment_visibility_type="group",
            comment_visibility_value=None,
        )
        == []
    )
