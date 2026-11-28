"""M6-02a verification: PITR StateReconciler and Testcontainers Chaos DR Suite."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import socketserver
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from uuid import UUID
from xml.sax.saxutils import escape as xml_escape

import httpx
import numpy as np
import psycopg
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from safetensors.numpy import save_file as save_safetensors_numpy
from testcontainers.community.postgres import PostgresContainer
from testcontainers.core.container import DockerContainer
from tests.fixtures.pg_isolation import (
    _embed_fixture_text,
    cleanup_ephemeral_pg_domain,
    create_ephemeral_pg_domain,
    make_ephemeral_pg_domain_id,
)

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.llm_backend import (
    AdapterRouter,
    GPUOverloadError,
    GPUPriorityClass,
    RedisGPUQuotaManager,
)
from context_scout.agents.open_weights_vllm_executor import OpenWeightsVLLMExecutor
from context_scout.agents.scout_workflow import ScoutWorkflow
from context_scout.cli import cmd_reconcile_state
from context_scout.cli import main as cli_main
from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
from context_scout.db.schema import (
    ASTSymbolRecord,
    CodeFileRecord,
    ConnectorWatermarkRecord,
    DocChunkRecord,
    DomainSyncManifestRecord,
    EmbeddingSpaceRecord,
    ModelArtifactRecord,
    RepoCommitRecord,
    RepositoryRecord,
)
from context_scout.db.session import PoolRole, RLSConnectionManager
from context_scout.ops.state_reconciler import StateReconciler
from context_scout.plugins.base import SyncResult
from context_scout.plugins.connector_sdk import (
    ConnectorRateLimitError,
    ConnectorSyncCoordinator,
    RedisRateLimitCoordinator,
)
from context_scout.plugins.connectors.confluence import (
    ConfluenceConnectorPlugin,
    ConfluencePageDTO,
)
from context_scout.plugins.connectors.github import (
    GitHubAppInstallationACL,
    GitHubEnterpriseConnectorPlugin,
)
from context_scout.plugins.connectors.gitlab import GitLabConnectorPlugin
from context_scout.plugins.connectors.jira import JiraConnectorPlugin, JiraIssueDTO
from context_scout.queue.indexing_worker import IndexingWorker
from context_scout.queue.task_queue import InMemoryTaskQueue, PostgresTaskQueue, QueueTask
from context_scout.retrieval.embedder import LocalSemanticEmbedder, normalize_embedding
from context_scout.retrieval.zoekt_client import LocalTrigramZoektIndex, ZoektGrpcJsonClient
from context_scout.schemas.contracts import (
    DEFAULT_TENANT_ID,
    AuthorityRules,
    DomainPackManifest,
    ImplementationContextPack,
    PrincipalContext,
    RoleType,
)
from context_scout.security.permission_mirror import (
    CachedPermissionMirrorService,
    EnterpriseIdentityProfile,
    GitLabProjectMembership,
    GitLabRole,
    RedisACLCacheBackend,
    _StdlibRESP2Client,
)
from context_scout.serving.adapter_sync_sidecar import (
    AdapterSyncSidecar,
    CosignBlobVerifier,
    sign_blob_for_cosign,
)
from context_scout.storage.object_store import (
    GCSObjectStore,
    InMemoryObjectStore,
    S3CompatibleObjectStore,
    build_tenant_domain_key,
)

# ---------------------------------------------------------------------------
# Helpers & Real Loopback TCP / HTTP Wire Chaos Servers + Testcontainers
# ---------------------------------------------------------------------------


def _docker_daemon_available() -> bool:
    for sock_path in (
        Path("/var/run/docker.sock"),
        Path.home() / ".docker" / "run" / "docker.sock",
    ):
        if sock_path.exists():
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(0.25)
            try:
                sock.connect(str(sock_path))
                return True
            except OSError:
                continue
            finally:
                sock.close()
    return False


class ChaosTestcontainersOrchestrator:
    """Manages optional Testcontainers PG17 + Redis 7.4 lifecycle with local PG17 fallback."""

    pg_image: str = "pgvector/pgvector:pg17"
    redis_image: str = "redis:7.4-alpine"
    redis_port: int = 6379

    def __init__(self, *, use_containers: bool | None = None) -> None:
        env_flag = os.environ.get("SCOUT_CHAOS_USE_CONTAINERS", "").strip() == "1"
        self.use_containers = (
            use_containers
            if use_containers is not None
            else (env_flag and _docker_daemon_available())
        )
        self.pg_container: PostgresContainer | None = None
        self.redis_container: DockerContainer | None = None
        self.app_db_url: str = ""
        self.admin_db_url: str = ""
        self.redis_url: str = ""

    def build_containers(self) -> tuple[PostgresContainer, DockerContainer]:
        pg = PostgresContainer(self.pg_image)
        redis = DockerContainer(self.redis_image).with_exposed_ports(self.redis_port)
        return pg, redis

    def __enter__(self) -> ChaosTestcontainersOrchestrator:
        if self.use_containers:
            from urllib.parse import quote, urlparse, urlunparse

            from context_scout.db.migrations_runner import run_alembic_upgrade

            pg, redis = self.build_containers()
            pg.start()
            redis.start()
            self.pg_container = pg
            self.redis_container = redis
            raw_pg = pg.get_connection_url()
            if raw_pg.startswith("postgresql+"):
                raw_pg = "postgresql://" + raw_pg.split("://", 1)[1]
            self.admin_db_url = raw_pg
            app_pw = f"ChaosApp_{uuid.uuid4().hex[:12]}!"
            run_alembic_upgrade(self.admin_db_url, app_role_password=app_pw)
            parsed = urlparse(self.admin_db_url)
            host_port = parsed.netloc.rsplit("@", 1)[-1]
            self.app_db_url = urlunparse(
                parsed._replace(netloc=f"scout_app_user:{quote(app_pw, safe='')}@{host_port}")
            )
            host = redis.get_container_host_ip()
            port = redis.get_exposed_port(6379)
            self.redis_url = f"redis://{host}:{port}/0"
        else:
            cfg = ScoutSettings()
            self.admin_db_url = cfg.db_admin_url
            self.app_db_url = cfg.db_app_url
        return self

    def __exit__(self, *_args: object) -> None:
        if self.redis_container is not None:
            self.redis_container.stop()
        if self.pg_container is not None:
            self.pg_container.stop()


class _RESP2TCPHandler(socketserver.StreamRequestHandler):
    server: FaultInjectingRESP2Server

    def _read_resp_command(self) -> list[str] | None:
        first_line = self.rfile.readline()
        if not first_line:
            return None
        if not first_line.startswith(b"*"):
            parts = first_line.decode("utf-8", errors="replace").strip().split()
            return parts or None
        count = int(first_line[1:].strip())
        args: list[str] = []
        for _ in range(count):
            len_line = self.rfile.readline()
            if not len_line.startswith(b"$"):
                return None
            blen = int(len_line[1:].strip())
            payload = self.rfile.read(blen)
            _crlf = self.rfile.read(2)
            args.append(payload.decode("utf-8"))
        return args

    def handle(self) -> None:
        self.server.register_socket(self.connection)
        try:
            while True:
                cmd = self._read_resp_command()
                if cmd is None:
                    break
                response = self.server.execute_command(cmd)
                self.wfile.write(response)
                self.wfile.flush()
        except OSError:
            return
        finally:
            self.server.unregister_socket(self.connection)


class FaultInjectingRESP2Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Real multi-threaded loopback TCP server speaking RESP2 with chaos toggles."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self) -> None:
        self._state_lock = threading.RLock()
        self._kv: dict[str, tuple[str, float | None]] = {}
        self._sockets: set[socket.socket] = set()
        self.command_log: list[tuple[str, ...]] = []
        self.reject_commands = False
        super().__init__(("127.0.0.1", 0), _RESP2TCPHandler)
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    @property
    def redis_url(self) -> str:
        return f"redis://127.0.0.1:{self.port}/0"

    def register_socket(self, sock: socket.socket) -> None:
        with self._state_lock:
            self._sockets.add(sock)

    def unregister_socket(self, sock: socket.socket) -> None:
        with self._state_lock:
            self._sockets.discard(sock)

    def drop_active_connections(self) -> None:
        with self._state_lock:
            for sock in list(self._sockets):
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)
                with contextlib.suppress(OSError):
                    sock.close()
            self._sockets.clear()

    def flush_all(self) -> None:
        with self._state_lock:
            self._kv.clear()

    def snapshot_keys(self) -> dict[str, str]:
        now = time.time()
        with self._state_lock:
            out: dict[str, str] = {}
            for k, (val, exp) in list(self._kv.items()):
                if exp is not None and now >= exp:
                    self._kv.pop(k, None)
                    continue
                out[k] = val
            return out

    def _get_live(self, key: str) -> str | None:
        entry = self._kv.get(key)
        if entry is None:
            return None
        val, exp = entry
        if exp is not None and time.time() >= exp:
            self._kv.pop(key, None)
            return None
        return val

    @staticmethod
    def _encode_bulk(val: str | None) -> bytes:
        if val is None:
            return b"$-1\r\n"
        raw = val.encode("utf-8")
        return f"${len(raw)}\r\n".encode("ascii") + raw + b"\r\n"

    def execute_command(self, cmd: list[str]) -> bytes:
        op = cmd[0].upper()
        with self._state_lock:
            self.command_log.append(tuple(cmd))
            if self.reject_commands:
                return b"-ERR CHAOS_REDIS_UNAVAILABLE\r\n"

            if op == "PING":
                return b"+PONG\r\n"
            if op in {"AUTH", "SELECT"}:
                return b"+OK\r\n"
            if op == "GET" and len(cmd) >= 2:
                return self._encode_bulk(self._get_live(cmd[1]))
            if op == "SET" and len(cmd) >= 3:
                exp: float | None = None
                if len(cmd) >= 5 and cmd[3].upper() == "EX":
                    exp = time.time() + max(1, int(cmd[4]))
                self._kv[cmd[1]] = (cmd[2], exp)
                return b"+OK\r\n"
            if op == "SETEX" and len(cmd) >= 4:
                ttl = max(1, int(cmd[2]))
                self._kv[cmd[1]] = (cmd[3], time.time() + ttl)
                return b"+OK\r\n"
            if op == "INCRBY" and len(cmd) >= 3:
                cur_raw = self._get_live(cmd[1])
                cur_int = int(cur_raw) if cur_raw is not None else 0
                nxt = cur_int + int(cmd[2])
                prev_exp = self._kv[cmd[1]][1] if cmd[1] in self._kv else None
                self._kv[cmd[1]] = (str(nxt), prev_exp)
                return f":{nxt}\r\n".encode("ascii")
            if op == "EXPIRE" and len(cmd) >= 3:
                cur_raw = self._get_live(cmd[1])
                if cur_raw is None:
                    return b":0\r\n"
                self._kv[cmd[1]] = (cur_raw, time.time() + max(1, int(cmd[2])))
                return b":1\r\n"
            if op == "DEL":
                deleted = 0
                for k in cmd[1:]:
                    if self._kv.pop(k, None) is not None:
                        deleted += 1
                return f":{deleted}\r\n".encode("ascii")
            if op in {"FLUSHDB", "FLUSHALL"}:
                self._kv.clear()
                return b"+OK\r\n"
            if op == "KEYS":
                pat = cmd[1] if len(cmd) >= 2 else "*"
                prefix = pat[:-1] if pat.endswith("*") else pat
                matched = sorted(
                    k
                    for k in list(self._kv)
                    if self._get_live(k) is not None
                    and (k.startswith(prefix) if pat.endswith("*") else k == pat)
                )
                parts = [f"*{len(matched)}\r\n".encode("ascii")]
                parts.extend(self._encode_bulk(k) for k in matched)
                return b"".join(parts)
            if op == "SCAN":
                match_pat = "*"
                for i, tok in enumerate(cmd[1:], start=1):
                    if tok.upper() == "MATCH" and i + 1 < len(cmd):
                        match_pat = cmd[i + 1]
                prefix = match_pat[:-1] if match_pat.endswith("*") else match_pat
                matched = sorted(
                    k
                    for k in list(self._kv)
                    if self._get_live(k) is not None
                    and (k.startswith(prefix) if match_pat.endswith("*") else k == match_pat)
                )
                parts = [b"*2\r\n", self._encode_bulk("0"), f"*{len(matched)}\r\n".encode("ascii")]
                parts.extend(self._encode_bulk(k) for k in matched)
                return b"".join(parts)
            return b"-ERR unsupported command\r\n"

    def close_server(self) -> None:
        self.drop_active_connections()
        self.shutdown()
        self.server_close()


class LoopbackS3AndGCSChaosServer:
    """Real HTTP server handling both AWS S3 XML REST (boto3) & GCP GCS JSON API v1 (httpx)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.s3_objects: dict[tuple[str, str], tuple[bytes, dict[str, str], str]] = {}
        self.gcs_objects: dict[tuple[str, str], tuple[bytes, dict[str, Any]]] = {}

        outer = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _fmt: str, *_args: object) -> None:
                return

            def _send(
                self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None
            ) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                if extra:
                    for hk, hv in extra.items():
                        self.send_header(hk, hv)
                self.end_headers()
                if self.command != "HEAD" and body:
                    self.wfile.write(body)

            def _read_body(self) -> bytes:
                clen = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(clen) if clen > 0 else b""

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                raw_body = self._read_body()
                if parsed.path.startswith("/upload/storage/v1/b/"):
                    rest = parsed.path[len("/upload/storage/v1/b/") :]
                    bucket = unquote(rest.split("/o", 1)[0])
                    ctype = self.headers.get("Content-Type", "")
                    boundary = ctype.split("boundary=", 1)[-1].strip()
                    parts = raw_body.split(f"--{boundary}".encode("ascii"))
                    data_parts: list[bytes] = []
                    for part in parts[1:]:
                        stripped = part.lstrip(b"\r\n")
                        if not stripped.startswith(b"Content-Type:") or b"\r\n\r\n" not in stripped:
                            continue
                        _hdr, body_raw = stripped.split(b"\r\n\r\n", 1)
                        if body_raw.endswith(b"\r\n"):
                            body_raw = body_raw[:-2]
                        data_parts.append(body_raw)
                    meta_obj: dict[str, Any] = (
                        json.loads(data_parts[0].decode("utf-8")) if data_parts else {}
                    )
                    data_bytes = data_parts[1] if len(data_parts) > 1 else b""
                    qs = parse_qs(parsed.query)
                    obj_name = str(meta_obj.get("name") or (qs.get("name") or [""])[0])
                    with outer._lock:
                        outer.gcs_objects[(bucket, obj_name)] = (data_bytes, meta_obj)
                    self._send(200, json.dumps(meta_obj).encode("utf-8"), "application/json")
                    return
                self._send(404, b'{"error":"not_found"}', "application/json")

            def do_PUT(self) -> None:
                parsed = urlparse(self.path)
                raw_body = self._read_body()
                clean_path = parsed.path.lstrip("/")
                if "/" not in clean_path:
                    self._send(400, b"", "application/xml")
                    return
                bucket, encoded_key = clean_path.split("/", 1)
                key = unquote(encoded_key)
                meta = {
                    k[len("x-amz-meta-") :].lower(): v
                    for k, v in self.headers.items()
                    if k.lower().startswith("x-amz-meta-")
                }
                ctype = self.headers.get("Content-Type") or "application/octet-stream"
                with outer._lock:
                    outer.s3_objects[(unquote(bucket), key)] = (raw_body, meta, ctype)
                etag = hashlib.md5(raw_body).hexdigest()
                self._send(200, b"", "application/xml", {"ETag": f'"{etag}"'})

            def do_HEAD(self) -> None:
                parsed = urlparse(self.path)
                clean_path = parsed.path.lstrip("/")
                if "/" not in clean_path:
                    self._send(404, b"", "application/xml")
                    return
                bucket, encoded_key = clean_path.split("/", 1)
                key = unquote(encoded_key)
                with outer._lock:
                    entry = outer.s3_objects.get((unquote(bucket), key))
                if entry is None:
                    self._send(404, b"", "application/xml")
                    return
                raw_body, meta, ctype = entry
                headers = {f"x-amz-meta-{mk}": mv for mk, mv in meta.items()}
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw_body)))
                for hk, hv in headers.items():
                    self.send_header(hk, hv)
                self.end_headers()

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                qs = parse_qs(parsed.query)
                if parsed.path.startswith("/storage/v1/b/"):
                    rest = parsed.path[len("/storage/v1/b/") :]
                    if "/o/" in rest:
                        bkt_enc, key_enc = rest.split("/o/", 1)
                        bkt, key = unquote(bkt_enc), unquote(key_enc)
                        with outer._lock:
                            entry = outer.gcs_objects.get((bkt, key))
                        if entry is None:
                            self._send(404, b'{"error":"not_found"}', "application/json")
                            return
                        data_bytes, meta_obj = entry
                        if (qs.get("alt") or [""])[0] == "media":
                            self._send(200, data_bytes, "application/octet-stream")
                        else:
                            self._send(
                                200, json.dumps(meta_obj).encode("utf-8"), "application/json"
                            )
                        return
                    if rest.endswith("/o"):
                        bkt = unquote(rest[:-2])
                        prefix = (qs.get("prefix") or [""])[0]
                        with outer._lock:
                            names = sorted(
                                k
                                for (b, k) in outer.gcs_objects
                                if b == bkt and k.startswith(prefix)
                            )
                        body = json.dumps({"items": [{"name": n} for n in names]}).encode("utf-8")
                        self._send(200, body, "application/json")
                        return

                # AWS S3 ListObjectsV2 or GetObject
                clean_path = parsed.path.lstrip("/")
                if "list-type" in qs:
                    bucket = unquote(clean_path.rstrip("/"))
                    prefix = (qs.get("prefix") or [""])[0]
                    with outer._lock:
                        matched = sorted(
                            (k, len(v[0]), hashlib.md5(v[0]).hexdigest())
                            for (b, k), v in outer.s3_objects.items()
                            if b == bucket and k.startswith(prefix)
                        )
                    items_xml = "".join(
                        f"<Contents><Key>{xml_escape(k)}</Key><Size>{sz}</Size>"
                        f'<ETag>"{etag}"</ETag></Contents>'
                        for k, sz, etag in matched
                    )
                    xml_body = (
                        '<?xml version="1.0" encoding="UTF-8"?>'
                        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                        f"<Name>{xml_escape(bucket)}</Name>"
                        f"<Prefix>{xml_escape(prefix)}</Prefix>"
                        f"<KeyCount>{len(matched)}</KeyCount>"
                        "<MaxKeys>1000</MaxKeys>"
                        "<IsTruncated>false</IsTruncated>"
                        f"{items_xml}</ListBucketResult>"
                    ).encode()
                    self._send(200, xml_body, "application/xml")
                    return

                if "/" not in clean_path:
                    self._send(404, b"<Error><Code>NoSuchKey</Code></Error>", "application/xml")
                    return
                bucket, encoded_key = clean_path.split("/", 1)
                key = unquote(encoded_key)
                with outer._lock:
                    entry_s3 = outer.s3_objects.get((unquote(bucket), key))
                if entry_s3 is None:
                    self._send(
                        404,
                        b"<Error><Code>NoSuchKey</Code><Message>Not found</Message></Error>",
                        "application/xml",
                    )
                    return
                raw_body, meta, ctype = entry_s3
                headers = {f"x-amz-meta-{mk}": mv for mk, mv in meta.items()}
                self._send(200, raw_body, ctype, headers)

            def do_DELETE(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path.startswith("/storage/v1/b/"):
                    rest = parsed.path[len("/storage/v1/b/") :]
                    if "/o/" in rest:
                        bkt_enc, key_enc = rest.split("/o/", 1)
                        bkt, key = unquote(bkt_enc), unquote(key_enc)
                        with outer._lock:
                            removed = outer.gcs_objects.pop((bkt, key), None)
                        self._send(204 if removed is not None else 404, b"", "application/json")
                        return
                clean_path = parsed.path.lstrip("/")
                if "/" in clean_path:
                    bucket, encoded_key = clean_path.split("/", 1)
                    key = unquote(encoded_key)
                    with outer._lock:
                        outer.s3_objects.pop((unquote(bucket), key), None)
                    self._send(204, b"", "application/xml")
                    return
                self._send(404, b"", "application/xml")

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        port = int(self._httpd.server_address[1])
        return f"http://127.0.0.1:{port}"

    def make_gcs_store(self, bucket: str) -> GCSObjectStore:
        return GCSObjectStore(bucket=bucket, emulator_host=self.base_url)

    def make_s3_store(self, bucket: str) -> S3CompatibleObjectStore:
        return S3CompatibleObjectStore(
            bucket=bucket,
            endpoint_url=self.base_url,
            aws_access_key_id="chaos-test-ak",
            aws_secret_access_key="chaos-test-sk",
            force_path_style=True,
        )

    def corrupt_s3_bytes(self, bucket: str, key: str, corrupted_bytes: bytes) -> None:
        with self._lock:
            _old, meta, ctype = self.s3_objects[(bucket, key)]
            self.s3_objects[(bucket, key)] = (corrupted_bytes, meta, ctype)

    def drop_gcs_raw(self, bucket: str, key: str) -> None:
        with self._lock:
            self.gcs_objects.pop((bucket, key), None)

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


class LoopbackZoektAndVLLMChaosServer:
    """Real HTTP server hosting remote Zoekt gRPC-JSON + multi-pod vLLM endpoints."""

    def __init__(self, *, base_model_id: str, pod_count: int = 2) -> None:
        self._lock = threading.RLock()
        self.base_model_id = base_model_id
        self.zoekt_files: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self.pod_loras: dict[str, set[str]] = {f"pod{i}": set() for i in range(pod_count)}
        self.pod_offline: set[str] = set()
        self.completion_mode: str = "ok"  # "ok" | "429" | "503" | "timeout"
        self.timeout_sleep_seconds: float = 0.8
        self.completion_calls: list[dict[str, Any]] = []
        self.custom_completion_text: str | None = None

        outer = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _fmt: str, *_args: object) -> None:
                return

            def _send_json(self, code: int, payload: Any) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, Any]:
                clen = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(clen) if clen > 0 else b"{}"
                data = json.loads(raw.decode("utf-8"))
                return dict(data) if isinstance(data, dict) else {}

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                segs = [s for s in parsed.path.split("/") if s]
                if len(segs) == 3 and segs[1] == "v1" and segs[2] == "models":
                    pod = segs[0]
                    with outer._lock:
                        if pod in outer.pod_offline:
                            self._send_json(503, {"error": "pod offline"})
                            return
                        loras = sorted(outer.pod_loras.get(pod, set()))
                    data = [
                        {"id": outer.base_model_id, "object": "model", "max_model_len": 32768},
                        *[
                            {"id": lid, "object": "model", "parent": outer.base_model_id}
                            for lid in loras
                        ],
                    ]
                    self._send_json(200, {"object": "list", "data": data})
                    return
                self._send_json(404, {"error": "not_found"})

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                body = self._read_json()

                # Zoekt endpoints
                if path == "/api/list":
                    tid = str(body.get("tenant_id") or "")
                    did = str(body.get("domain_id") or "")
                    with outer._lock:
                        files = [
                            dict(v)
                            for (ftid, fdid, _rn, _fp), v in outer.zoekt_files.items()
                            if ftid == tid and fdid == did
                        ]
                        repos = sorted({f["repo_name"] for f in files})
                    self._send_json(200, {"repos": repos, "files": files})
                    return

                if path == "/api/admin/purge":
                    tid = str(body.get("tenant_id") or "")
                    did = str(body.get("domain_id") or "")
                    rname = str(body.get("repo_name") or "")
                    removed = 0
                    with outer._lock:
                        for k in list(outer.zoekt_files):
                            if k[0] == tid and k[1] == did and k[2] == rname:
                                outer.zoekt_files.pop(k, None)
                                removed += 1
                    self._send_json(200, {"deleted": removed})
                    return

                if path == "/api/admin/reindex":
                    tid = str(body.get("tenant_id") or "")
                    did = str(body.get("domain_id") or "")
                    purged = 0
                    reindexed = 0
                    with outer._lock:
                        for df in body.get("delete_files") or []:
                            rn = str(df.get("repo_name") or "")
                            fp = str(df.get("file_path") or "")
                            if outer.zoekt_files.pop((tid, did, rn, fp), None) is not None:
                                purged += 1
                        for uf in body.get("upsert_files") or []:
                            rn = str(uf.get("repo_name") or "")
                            fp = str(uf.get("file_path") or "")
                            outer.zoekt_files[(tid, did, rn, fp)] = {
                                "repo_name": rn,
                                "file_path": fp,
                                "checksum": str(uf.get("checksum") or ""),
                                "content": str(uf.get("content") or ""),
                                "language": str(uf.get("language") or "python"),
                            }
                            reindexed += 1
                    self._send_json(200, {"purged": purged, "reindexed": reindexed})
                    return

                if path.endswith("/Search"):
                    q = str(body.get("Q") or "")
                    with outer._lock:
                        matched_files = []
                        for (_tid, _did, rn, fp), item in outer.zoekt_files.items():
                            if rn in q and any(
                                tok in item["content"] for tok in q.split() if len(tok) >= 3
                            ):
                                matched_files.append(
                                    {
                                        "Repository": rn,
                                        "FileName": fp,
                                        "Language": item.get("language", "python"),
                                        "Checksum": item.get("checksum", ""),
                                        "Score": 5.0,
                                        "LineMatches": [
                                            {"LineNumber": 1, "Line": item["content"][:120]}
                                        ],
                                    }
                                )
                    self._send_json(
                        200,
                        {
                            "Result": {
                                "Files": matched_files,
                                "Stats": {"FilesConsidered": len(matched_files)},
                            }
                        },
                    )
                    return

                # vLLM pod endpoints
                segs = [s for s in path.split("/") if s]
                if len(segs) == 3 and segs[1] == "v1":
                    pod, op = segs[0], segs[2]
                    with outer._lock:
                        if pod in outer.pod_offline:
                            self._send_json(503, {"error": "pod offline"})
                            return
                        if op == "load_lora_adapter":
                            lname = str(body.get("lora_name") or "")
                            if lname in outer.pod_loras.setdefault(pod, set()):
                                msg = f"The lora adapter '{lname}' has already been loaded."
                                self._send_json(400, {"message": msg})
                                return
                            outer.pod_loras[pod].add(lname)
                            self._send_json(200, {"status": "loaded", "lora_name": lname})
                            return
                        if op == "unload_lora_adapter":
                            lname = str(body.get("lora_name") or "")
                            if lname in outer.pod_loras.setdefault(pod, set()):
                                outer.pod_loras[pod].discard(lname)
                                self._send_json(200, {"status": "unloaded", "lora_name": lname})
                            else:
                                self._send_json(404, {"error": f"'{lname}' not loaded"})
                            return
                        if op == "chat":
                            self._send_json(404, {"error": "unknown"})
                            return
                if (
                    len(segs) == 4
                    and segs[1] == "v1"
                    and segs[2] == "chat"
                    and segs[3] == "completions"
                ):
                    pod = segs[0]
                    with outer._lock:
                        outer.completion_calls.append(body)
                        mode = outer.completion_mode
                        sleep_s = outer.timeout_sleep_seconds
                        custom = outer.custom_completion_text
                        offline = pod in outer.pod_offline
                    if offline or mode == "503":
                        self._send_json(503, {"error": "vLLM server overloaded"})
                        return
                    if mode == "429":
                        self._send_json(429, {"error": "rate limit exceeded"})
                        return
                    if mode == "timeout":
                        time.sleep(sleep_s)
                        self._send_json(504, {"error": "gateway timeout"})
                        return
                    reply = custom or json.dumps({"status": "ok"})
                    if body.get("response_format"):
                        reply = f"<json_output>{reply}</json_output>"
                    self._send_json(
                        200,
                        {
                            "id": "chatcmpl-chaos",
                            "object": "chat.completion",
                            "choices": [
                                {"index": 0, "message": {"role": "assistant", "content": reply}}
                            ],
                            "usage": {
                                "prompt_tokens": 32,
                                "completion_tokens": 16,
                                "total_tokens": 48,
                            },
                        },
                    )
                    return

                self._send_json(404, {"error": "not_found"})

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        port = int(self._httpd.server_address[1])
        return f"http://127.0.0.1:{port}"

    def zoekt_url(self) -> str:
        return self.base_url

    def pod_v1_url(self, pod_idx: int = 0) -> str:
        return f"{self.base_url}/pod{pod_idx}/v1"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def _build_safetensors_bytes(seed_scale: float = 1.0) -> bytes:
    tensors = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": (
            np.ones((4, 4), dtype=np.float32) * seed_scale
        ),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": (
            np.ones((4, 4), dtype=np.float32) * (seed_scale + 1.0)
        ),
    }
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        save_safetensors_numpy(tensors, str(tmp_path))
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def _make_unit_vec(dim: int, hot_index: int = 0) -> list[float]:
    raw = [0.02] * dim
    raw[hot_index % dim] = 1.0
    return normalize_embedding(raw, target_dim=dim, supports_matryoshka=True)


def _cleanup_pg_tenants_and_domains(
    tenant_ids: list[UUID],
    domain_ids: list[str],
) -> None:
    cfg = ScoutSettings()
    with psycopg.connect(cfg.db_admin_url) as conn:
        for dom in domain_ids:
            cleanup_ephemeral_pg_domain(conn, dom)
        for tid in tenant_ids:
            for tbl in (
                "audit_events",
                "scim_groups",
                "scim_users",
                "model_artifacts",
                "scout_sessions",
                "overlay_file_deltas",
                "ref_snapshots",
                "side_effects",
                "ast_edges",
                "ast_symbols",
                "code_files",
                "repo_commits",
                "repositories",
                "confluence_pages",
                "jira_issues",
                "doc_chunks",
                "execution_telemetry",
                "indexing_queue",
                "indexing_dead_letter",
                "connector_sync_watermarks",
                "embedding_spaces",
                "domain_sync_manifests",
                "domain_packs",
            ):
                conn.execute(f"DELETE FROM {tbl} WHERE tenant_id = %s;", (tid,))
        conn.commit()


# ---------------------------------------------------------------------------
# 1. Regression Tests for Redis Quota Failover & vLLM Degraded Mode
# ---------------------------------------------------------------------------


def test_resp2_atomic_incrby_and_redis_flush_failover_quota_reconciliation() -> None:
    """Proves _StdlibRESP2Client uses atomic INCRBY/EXPIRE & survives FLUSHALL + failover."""
    srv = FaultInjectingRESP2Server()
    try:
        client = _StdlibRESP2Client(srv.redis_url, timeout=1.0)
        assert client.ping() == "PONG"
        qm = RedisGPUQuotaManager(
            redis_client=client,
            max_global_inflight=2,
            max_tenant_inflight=2,
            max_team_inflight=2,
            max_user_inflight=2,
            key_prefix="scout:disc028",
        )
        principal = PrincipalContext(
            user_id="eng_disc028",
            role=RoleType.ENGINEER,
            allowed_scopes=["public"],
            tenant_id=uuid.uuid4(),
        )

        # 1. Normal lease uses real atomic INCRBY + EXPIRE commands over TCP
        with qm.lease(principal, priority=GPUPriorityClass.P0, estimated_tokens=512) as lease:
            assert lease.used_redis is True
            snap = qm.snapshot(principal)
            assert snap["global"] == 1
            assert snap["tenant_inf"] == 1

        cmds = [c[0].upper() for c in srv.command_log]
        assert "INCRBY" in cmds
        assert "EXPIRE" in cmds
        assert qm.snapshot(principal)["global"] == 0

        # 2. Mid-lease FLUSHALL must NOT leave negative counters or block subsequent releases
        with qm.lease(principal, priority=GPUPriorityClass.P0, estimated_tokens=256):
            srv.flush_all()
            assert srv.snapshot_keys() == {}
        # After release, key must remain clean (>=0, deleted on negative clamp)
        assert srv.snapshot_keys() == {}
        assert qm.snapshot(principal)["global"] == 0

        # 3. Acquire while Redis rejects commands -> fallback to _memory_counters,
        #    then Redis recovers BEFORE release -> _memory_counters MUST still decrement cleanly.
        srv.reject_commands = True
        with qm.lease(principal, priority=GPUPriorityClass.P0, estimated_tokens=256) as mem_lease:
            assert mem_lease.used_redis is False
            assert qm.snapshot(principal)["global"] == 1
            # Redis recovers mid-lease
            srv.reject_commands = False
        # Even if Redis later goes back down, local fallback counters must NOT be wedged!
        srv.reject_commands = True
        assert qm.snapshot(principal)["global"] == 0
        srv.reject_commands = False
        client.close()
    finally:
        srv.close_server()


def test_vllm_llm_backend_unreachable_and_priority_propagation_degraded_mode() -> None:
    """Proves delegated OpenWeightsVLLMExecutor propagates priority & emits degraded mode."""
    vllm_srv = LoopbackZoektAndVLLMChaosServer(base_model_id="google/gemma-4-26B-A4B-it")
    try:
        vllm_srv.pod_offline.add("pod0")
        qm = RedisGPUQuotaManager(max_global_inflight=4)
        executor = OpenWeightsVLLMExecutor(
            model_id="google/gemma-4-26B-A4B-it",
            endpoint_url=vllm_srv.pod_v1_url(0),
            runtime_backend="vllm",
            timeout_seconds=0.5,
            gpu_quota_manager=qm,
            auto_discover_max_len=False,
        )
        assert executor.is_online() is False

        mem_repo = InMemoryRLSRepository()
        tid = uuid.uuid4()
        principal = PrincipalContext(
            user_id="eng_disc029",
            role=RoleType.ENGINEER,
            allowed_scopes=["public", "public_eng"],
            tenant_id=tid,
        )
        fid = uuid.uuid4()
        mem_repo.insert_code_file(
            principal,
            CodeFileRecord(
                file_id=fid,
                domain_id="code_architecture_scout",
                repo_name="repo-disc029",
                file_path="src/auth_token.py",
                git_commit_sha="c1" * 20,
                git_blob_sha="b1" * 20,
                language="python",
                allowed_scopes=["public", "public_eng"],
                raw_content="def verify_hmac_token(tok: str) -> bool:\n    return len(tok) > 10\n",
                line_count=2,
                tenant_id=tid,
            ),
        )
        mem_repo.insert_ast_symbol(
            principal,
            ASTSymbolRecord(
                symbol_id=uuid.uuid4(),
                file_id=fid,
                domain_id="code_architecture_scout",
                qualified_name="verify_hmac_token",
                symbol_kind="function",
                start_line=1,
                end_line=2,
                signature_skeleton="def verify_hmac_token(tok: str) -> bool:",
                contextual_summary="Verifies HMAC token length.",
                allowed_scopes=["public", "public_eng"],
                embedding=_make_unit_vec(768, 3),
                tenant_id=tid,
            ),
        )

        # Default LocalGemmaClient endpoint stays http://127.0.0.1:8000/v1, delegated llm_backend
        wf = ScoutWorkflow(repo=mem_repo, llm_backend=executor)
        assert wf.gpu_quota_manager is qm

        pack = wf.run(
            user_prompt="Inspect verify_hmac_token in src/auth_token.py",
            principal=principal,
            domain_id="code_architecture_scout",
        )
        assert isinstance(pack, ImplementationContextPack)
        assert pack.is_abstention is False
        assert pack.synthesis_mode == "degraded_extractive_reranked"

        # Priority propagation check: while P0 is active, P2 call via LocalGemmaClient is preempted
        vllm_srv.pod_offline.clear()
        vllm_srv.custom_completion_text = json.dumps(
            [
                {
                    "repo_name": "repo-disc029",
                    "file_path": "src/auth_token.py",
                    "start_line": 1,
                    "end_line": 2,
                    "qualified_symbol": "verify_hmac_token",
                    "selection_reason": "Matched",
                    "modification_intent": "reference_only",
                }
            ]
        )
        other_principal = PrincipalContext(
            user_id="p0_caller",
            role=RoleType.ENGINEER,
            allowed_scopes=["public"],
            tenant_id=uuid.uuid4(),
        )
        with qm.lease(other_principal, priority=GPUPriorityClass.P0, estimated_tokens=100):
            wf.local_client.generate_code_pointers(
                user_prompt="Inspect verify_hmac_token",
                domain_id="code_architecture_scout",
                repo=mem_repo,
                principal=principal,
                priority=GPUPriorityClass.P2,
            )
            assert wf.local_client._gpu_degraded_reason == "gpu_quota_overload:preempted_by_p0"
        executor.close()
    finally:
        vllm_srv.close()


# ---------------------------------------------------------------------------
# 2. Testcontainers Orchestrator & Fail-Closed NOBYPASSRLS Guard
# ---------------------------------------------------------------------------


def test_testcontainers_orchestrator_unit_and_reconciler_rejects_admin_bypassrls() -> None:
    """Verifies Testcontainers PG17/Redis container specs & StateReconciler NOBYPASSRLS check."""
    orch = ChaosTestcontainersOrchestrator(use_containers=False)
    assert orch.pg_image == "pgvector/pgvector:pg17"
    assert orch.redis_image == "redis:7.4-alpine"
    assert orch.redis_port == 6379
    assert issubclass(PostgresContainer, DockerContainer)
    if _docker_daemon_available():
        pg_container, redis_container = orch.build_containers()
        assert "pgvector/pgvector:pg17" in str(pg_container.image)
        assert "redis:7.4-alpine" in str(redis_container.image)

    with orch as ctx:
        assert "postgres" in ctx.admin_db_url
        assert "scout_app_user" in ctx.app_db_url

    cfg = ScoutSettings()
    admin_mgr = RLSConnectionManager(cfg.db_admin_url, use_pool=False)
    with pytest.raises(PermissionError, match="privileged DB role"):
        StateReconciler(repository=PostgresRLSRepository(), connection_manager=admin_mgr)


# ---------------------------------------------------------------------------
# 3. Live PG17 PITR Rollback -> StateReconciler across Zoekt, GCS, S3, & vLLM
# ---------------------------------------------------------------------------


@pytest.mark.postgres
def test_chaos_postgres_pitr_rollback_reconciles_zoekt_gcs_s3_vllm(
    tmp_path: Path,
) -> None:
    """Proves full PITR healing across PG17, local+remote Zoekt, GCS gs://, S3 s3://, & vLLM."""
    cfg = ScoutSettings()
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    dom_a = make_ephemeral_pg_domain_id("m6_02_pitr_a")
    dom_b = make_ephemeral_pg_domain_id("m6_02_pitr_b")

    scopes = ["public", "public_eng", "public_docs"]
    principal_a = PrincipalContext(
        user_id="pitr_ops_a",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=scopes,
        tenant_id=tenant_a,
    )
    principal_b = PrincipalContext(
        user_id="pitr_ops_b",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=scopes,
        tenant_id=tenant_b,
    )

    base_model = "google/gemma-4-26B-A4B-it"
    pinned_rev = hashlib.sha256(b"gemma-4-26B-A4B-it-pitr-rev").hexdigest()
    ed_priv = ed25519.Ed25519PrivateKey.generate()
    ed_pub_pem = (
        ed_priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    verifier = CosignBlobVerifier(trusted_public_keys_pem={"pitr-ed25519": ed_pub_pem})

    s3_gcs_srv = LoopbackS3AndGCSChaosServer()
    zoekt_vllm_srv = LoopbackZoektAndVLLMChaosServer(base_model_id=base_model, pod_count=2)
    gcs_store = s3_gcs_srv.make_gcs_store("scout-pitr-gcs")
    s3_store = s3_gcs_srv.make_s3_store("scout-pitr-s3")
    local_zoekt = LocalTrigramZoektIndex()
    remote_zoekt = ZoektGrpcJsonClient(zoekt_vllm_srv.zoekt_url())
    embedder = LocalSemanticEmbedder()

    app_mgr = RLSConnectionManager(cfg.db_app_url, use_pool=False, pool_role=PoolRole.INGESTION)
    pg_repo = PostgresRLSRepository()
    queue = PostgresTaskQueue(connection_manager=app_mgr, default_tenant_id=tenant_a)

    try:
        with psycopg.connect(cfg.db_admin_url) as admin_conn:
            create_ephemeral_pg_domain(admin_conn, domain_id=dom_a, tenant_id=tenant_a)
            create_ephemeral_pg_domain(admin_conn, domain_id=dom_b, tenant_id=tenant_b)

        sha_c1 = "1" * 40
        sha_c2 = "2" * 40
        blob_c1 = "a" * 40
        blob_c2 = "b" * 40
        repo_a_id = uuid.uuid4()
        repo_b_id = uuid.uuid4()
        file_a1_id = uuid.uuid4()
        sym_a1_id = uuid.uuid4()
        chunk_a1_id = uuid.uuid4()
        file_b1_id = uuid.uuid4()

        # Upload Adapter v1 (survived PITR) & v2 (rolled back from DB, stranded on GCS/S3)
        w1 = _build_safetensors_bytes(1.0)
        w1_sha = hashlib.sha256(w1).hexdigest()
        w2 = _build_safetensors_bytes(2.0)
        cfg_bytes = json.dumps(
            {
                "base_model_name_or_path": base_model,
                "base_model_revision_sha256": pinned_rev,
                "r": 16,
                "lora_alpha": 32,
                "use_rslora": True,
                "use_dora": False,
            }
        ).encode("utf-8")

        v1_w_key = build_tenant_domain_key(
            tenant_a, dom_a, "adapters/lora_v1/adapter_model.safetensors"
        )
        v1_c_key = build_tenant_domain_key(tenant_a, dom_a, "adapters/lora_v1/adapter_config.json")
        v1_s_key = build_tenant_domain_key(
            tenant_a, dom_a, "adapters/lora_v1/adapter_model.safetensors.sig"
        )
        v1_sig = sign_blob_for_cosign(ed_priv, w1)
        gcs_store.put_bytes(v1_w_key, w1, expected_tenant_id=tenant_a)
        gcs_store.put_bytes(v1_c_key, cfg_bytes, expected_tenant_id=tenant_a)
        gcs_store.put_bytes(v1_s_key, v1_sig.encode("utf-8"), expected_tenant_id=tenant_a)

        # Orphaned post-PITR adapter_v2 artifacts left behind in GCS and S3
        v2_w_key = build_tenant_domain_key(
            tenant_a, dom_a, "adapters/lora_v2/adapter_model.safetensors"
        )
        v2_c_key = build_tenant_domain_key(tenant_a, dom_a, "adapters/lora_v2/adapter_config.json")
        gcs_store.put_bytes(v2_w_key, w2, expected_tenant_id=tenant_a)
        s3_store.put_bytes(v2_c_key, cfg_bytes, expected_tenant_id=tenant_a)

        # Canary tenant_b object in GCS that MUST remain untouched
        canary_key = build_tenant_domain_key(
            tenant_b, dom_b, "adapters/canary_v1/adapter_config.json"
        )
        gcs_store.put_bytes(canary_key, cfg_bytes, expected_tenant_id=tenant_b)

        # Seed Tenant A T1 state in PG17 (post-PITR database snapshot)
        v1_vec = _embed_fixture_text("def compute_checksum_v1(): pass", dim=768)
        doc_vec = _embed_fixture_text("Runbook section 1 for PITR service", dim=768)
        with app_mgr.scoped_transaction(principal_a, enable_hnsw_iterative_scan=False) as conn:
            pg_repo.register_embedding_space(
                conn,
                EmbeddingSpaceRecord(
                    tenant_id=tenant_a,
                    domain_id=dom_a,
                    space_id="space_v1",
                    model_id="google/embeddinggemma-2",
                    dimensions=768,
                    status="active",
                    is_active=True,
                ),
            )
            pg_repo.insert_repository(
                conn,
                RepositoryRecord(
                    repo_id=repo_a_id,
                    domain_id=dom_a,
                    repo_name="svc-core",
                    source_type="gitlab",
                    remote_url="https://gitlab.internal/svc-core",
                    default_branch="main",
                    allowed_scopes=scopes,
                    head_commit_sha=sha_c1,
                    tenant_id=tenant_a,
                ),
            )
            pg_repo.insert_repo_commit(
                conn,
                RepoCommitRecord(
                    commit_id=uuid.uuid4(),
                    repo_id=repo_a_id,
                    commit_sha=sha_c1,
                    parent_shas=[],
                    author_name="dev",
                    author_email="dev@corp.local",
                    commit_message="T1 baseline",
                    committed_at=datetime.now(UTC) - timedelta(hours=1),
                    tenant_id=tenant_a,
                ),
            )
            pg_repo.insert_code_file(
                conn,
                CodeFileRecord(
                    file_id=file_a1_id,
                    domain_id=dom_a,
                    repo_name="svc-core",
                    file_path="src/core.py",
                    git_commit_sha=sha_c1,
                    git_blob_sha=blob_c1,
                    language="python",
                    allowed_scopes=scopes,
                    raw_content="def compute_checksum_v1(x: bytes) -> str:\n    return 'ok'\n",
                    line_count=2,
                    repo_id=repo_a_id,
                    tenant_id=tenant_a,
                ),
            )
            pg_repo.insert_ast_symbol(
                conn,
                ASTSymbolRecord(
                    symbol_id=sym_a1_id,
                    file_id=file_a1_id,
                    domain_id=dom_a,
                    qualified_name="compute_checksum_v1",
                    symbol_kind="function",
                    start_line=1,
                    end_line=2,
                    signature_skeleton="def compute_checksum_v1(x: bytes) -> str:",
                    contextual_summary="Computes checksum for T1.",
                    allowed_scopes=scopes,
                    embedding=v1_vec,
                    repo_id=repo_a_id,
                    tenant_id=tenant_a,
                    embedding_space_id="space_v1",
                ),
            )
            pg_repo.insert_doc_chunk(
                conn,
                DocChunkRecord(
                    chunk_id=chunk_a1_id,
                    domain_id=dom_a,
                    doc_id="runbook-1",
                    doc_title="Service Runbook",
                    source_uri="https://wiki.internal/runbook-1",
                    market_scope="GLOBAL",
                    published_date=datetime.now(UTC).date(),
                    authority_weight=1.0,
                    section_heading="Recovery",
                    contextual_header="PITR recovery procedure",
                    verbatim_content="Verify PG17 state reconciler after Cloud SQL failover.",
                    metadata_json={},
                    allowed_scopes=scopes,
                    embedding=doc_vec,
                    tenant_id=tenant_a,
                    embedding_space_id="space_v1",
                ),
            )
            # Inject PITR embedding watermark drift & orphaned shadow space
            hv_lit = "[" + ",".join(f"{v:.6f}" for v in v1_vec) + "]"
            conn.execute(
                """
                UPDATE ast_symbols
                SET embedding_space_id = 'space_v2_stale',
                    shadow_embedding_space_id = 'space_v2_orphan',
                    shadow_embedding_halfvec = %s::halfvec
                WHERE domain_id = %s AND symbol_id = %s;
                """,
                (hv_lit, dom_a, sym_a1_id),
            )
            conn.execute(
                """
                UPDATE doc_chunks
                SET embedding_space_id = 'space_v2_stale',
                    shadow_embedding_space_id = 'space_v2_orphan',
                    shadow_embedding_halfvec = %s::halfvec
                WHERE domain_id = %s AND chunk_id = %s;
                """,
                (hv_lit, dom_a, chunk_a1_id),
            )
            # Stale connector watermark ahead at T2 commit sha_c2
            pg_repo.upsert_connector_watermark(
                conn,
                ConnectorWatermarkRecord(
                    tenant_id=tenant_a,
                    domain_id=dom_a,
                    source_id="gitlab",
                    container_key="svc-core",
                    watermark_cursor=sha_c2,
                    sync_tier="webhook",
                    items_synced=2,
                ),
            )
            # Stale domain sync manifest pointing to T2
            pg_repo.upsert_domain_sync_manifest(
                conn,
                DomainSyncManifestRecord(
                    tenant_id=tenant_a,
                    domain_id=dom_a,
                    kind="code",
                    repo_commit_shas={"svc-core": sha_c2},
                    code_files_indexed=2,
                ),
            )
            # Register lora_v1 as active in PG17 (post-PITR rolled back from lora_v2)
            pg_repo.register_model_artifact(
                conn,
                ModelArtifactRecord(
                    tenant_id=tenant_a,
                    domain_id=dom_a,
                    artifact_id="lora_v1",
                    base_model_id=base_model,
                    base_model_revision_sha256=pinned_rev,
                    storage_uri=f"gs://scout-pitr-gcs/{v1_w_key}",
                    weights_object_key=v1_w_key,
                    config_object_key=v1_c_key,
                    signature_object_key=v1_s_key,
                    weights_sha256=w1_sha,
                    cosign_signature_b64=v1_sig,
                    cosign_key_id="pitr-ed25519",
                    training_scope_ceiling=["public_eng"],
                    allowed_scopes=scopes,
                    status="active",
                    is_active=True,
                ),
            )

        # Seed canary Tenant B file in PG17 + local Zoekt
        with app_mgr.scoped_transaction(principal_b, enable_hnsw_iterative_scan=False) as conn:
            pg_repo.insert_repository(
                conn,
                RepositoryRecord(
                    repo_id=repo_b_id,
                    domain_id=dom_b,
                    repo_name="canary-svc",
                    source_type="github",
                    remote_url="https://github.internal/canary-svc",
                    default_branch="main",
                    allowed_scopes=scopes,
                    head_commit_sha=sha_c1,
                    tenant_id=tenant_b,
                ),
            )
            pg_repo.insert_code_file(
                conn,
                CodeFileRecord(
                    file_id=file_b1_id,
                    domain_id=dom_b,
                    repo_name="canary-svc",
                    file_path="src/canary.py",
                    git_commit_sha=sha_c1,
                    git_blob_sha=blob_c1,
                    language="python",
                    allowed_scopes=scopes,
                    raw_content="def canary_ok() -> bool:\n    return True\n",
                    line_count=2,
                    repo_id=repo_b_id,
                    tenant_id=tenant_b,
                ),
            )
        local_zoekt.index_file(
            tenant_id=tenant_b,
            domain_id=dom_b,
            repo_name="canary-svc",
            file_path="src/canary.py",
            content="def canary_ok() -> bool:\n    return True\n",
            allowed_scopes=scopes,
            checksum=blob_c1,
        )

        # Simulate stale T2 state outside PG17:
        # 1. Local Zoekt has phantom T2 file + phantom T2 repo (missing T1 svc-core/src/core.py)
        local_zoekt.index_file(
            tenant_id=tenant_a,
            domain_id=dom_a,
            repo_name="svc-core",
            file_path="src/phantom_t2.py",
            content="def phantom_t2_fn():\n    return 2\n",
            allowed_scopes=scopes,
            checksum=blob_c2,
        )
        local_zoekt.index_file(
            tenant_id=tenant_a,
            domain_id=dom_a,
            repo_name="phantom-repo-t2",
            file_path="src/deleted.py",
            content="def ghost(): pass\n",
            allowed_scopes=scopes,
            checksum=blob_c2,
        )
        # 2. Remote Zoekt server has phantom T2 file + phantom repo (missing svc-core/src/core.py)
        zoekt_vllm_srv.zoekt_files[(str(tenant_a), dom_a, "svc-core", "src/phantom_t2.py")] = {
            "repo_name": "svc-core",
            "file_path": "src/phantom_t2.py",
            "checksum": blob_c2,
            "content": "def phantom_t2_fn(): return 2",
        }
        zoekt_vllm_srv.zoekt_files[(str(tenant_a), dom_a, "phantom-repo-t2", "src/ghost.py")] = {
            "repo_name": "phantom-repo-t2",
            "file_path": "src/ghost.py",
            "checksum": blob_c2,
            "content": "def ghost(): pass",
        }
        remote_zoekt.acl_registry.register_repo(tenant_a, dom_a, "phantom-repo-t2", scopes)

        # 3. GCS gs:// & S3 s3:// hold T2 sync_manifest.json
        manifest_key = build_tenant_domain_key(tenant_a, dom_a, "manifests/sync_manifest.json")
        t2_manifest_bytes = json.dumps(
            {
                "domain_id": dom_a,
                "repo_commit_shas": {"svc-core": sha_c2, "phantom-repo-t2": sha_c2},
            }
        ).encode("utf-8")
        gcs_store.put_bytes(manifest_key, t2_manifest_bytes, expected_tenant_id=tenant_a)
        s3_store.put_bytes(manifest_key, t2_manifest_bytes, expected_tenant_id=tenant_a)

        # 4. Both vLLM pods hold stranded lora_v2 from before PITR rollback
        zoekt_vllm_srv.pod_loras["pod0"].add("lora_v2")
        zoekt_vllm_srv.pod_loras["pod1"].add("lora_v2")

        # 5. Stale PROCESSING task left orphaned in PG17 indexing_queue after crash
        stale_tid = queue.enqueue_task(
            dom_a,
            "watermark_sync",
            {"source_id": "gitlab", "container_key": "svc-core"},
            tenant_id=tenant_a,
        )
        claimed = queue.fetch_next_task("crashed-worker", tenant_id=tenant_a)
        assert claimed is not None and claimed.task_id == stale_tid
        with app_mgr.scoped_transaction(principal_a, enable_hnsw_iterative_scan=False) as conn:
            conn.execute(
                "UPDATE indexing_queue SET started_at = NOW() - INTERVAL '30 seconds' "
                "WHERE task_id = %s;",
                (stale_tid,),
            )

        sidecar = AdapterSyncSidecar(
            expected_base_model_id=base_model,
            expected_base_model_revision_sha256=pinned_rev,
            staging_root=tmp_path / "pitr_stage",
            cosign_verifier=verifier,
            tenant_id=tenant_a,
            sidecar_scopes=scopes,
            rls_conn_manager=app_mgr,
            object_stores_by_scheme={"gs": gcs_store, "s3": s3_store},
            vllm_pod_endpoints=[zoekt_vllm_srv.pod_v1_url(0), zoekt_vllm_srv.pod_v1_url(1)],
        )

        reconciler = StateReconciler(
            repository=pg_repo,
            connection_manager=app_mgr,
            zoekt_index=local_zoekt,
            zoekt_clients=[remote_zoekt],
            object_stores_by_scheme={"gs": gcs_store, "s3": s3_store},
            adapter_sidecar=sidecar,
            task_queue=queue,
            embedder=embedder,
            prune_orphan_objects=True,
            stale_task_seconds=5,
        )

        report = reconciler.reconcile(principal_a, dom_a)
        assert report.drift_detected is True
        assert report.healed is True
        assert report.warnings == []

        # Verify Plane 1: Commit watermarks rewound, manifests repaired on GCS+S3, Zoekt healed
        assert report.watermarks_rewound >= 2
        assert report.reconcile_tasks_enqueued >= 2
        assert len(report.object_manifests_repaired) == 2
        assert any(u.startswith("gs://") for u in report.object_manifests_repaired)
        assert any(u.startswith("s3://") for u in report.object_manifests_repaired)

        repaired_gcs = json.loads(
            gcs_store.get_bytes(manifest_key, expected_tenant_id=tenant_a).decode("utf-8")
        )
        repaired_s3 = json.loads(
            s3_store.get_bytes(manifest_key, expected_tenant_id=tenant_a).decode("utf-8")
        )
        assert repaired_gcs["repo_commit_shas"] == {"svc-core": sha_c1}
        assert repaired_s3["repo_commit_shas"] == {"svc-core": sha_c1}

        assert report.zoekt_purged_repos == 2  # phantom-repo-t2 purged from local + remote Zoekt
        assert report.zoekt_purged_files >= 4  # phantom_t2.py + phantom-repo-t2 files across both
        assert report.zoekt_reindexed_files == 2  # svc-core/src/core.py reindexed on local + remote

        # Verify local & remote Zoekt no longer match phantom T2, and DO match T1
        assert (
            local_zoekt.search(
                principal=principal_a, domain_id=dom_a, query="phantom_t2_fn"
            ).file_matches
            == ()
        )
        t1_hits = local_zoekt.search(
            principal=principal_a, domain_id=dom_a, query="compute_checksum_v1"
        ).file_matches
        assert [f.file_path for f in t1_hits] == ["src/core.py"]
        assert (str(tenant_a), dom_a, "svc-core", "src/core.py") in zoekt_vllm_srv.zoekt_files
        assert (
            str(tenant_a),
            dom_a,
            "phantom-repo-t2",
            "src/ghost.py",
        ) not in zoekt_vllm_srv.zoekt_files

        # Verify Plane 2: Embeddings & orphaned shadow cleared & real halfvec restored
        assert report.embedding_drift.active_space_id == "space_v1"
        assert report.embedding_drift.orphaned_shadow_rows_cleared == 2
        assert report.embedding_drift.drifted_symbol_rows_found == 1
        assert report.embedding_drift.drifted_chunk_rows_found == 1
        assert report.embedding_drift.symbol_rows_reembedded == 1
        assert report.embedding_drift.chunk_rows_reembedded == 1
        assert report.embedding_drift.unhealed_rows == 0

        # Verify Plane 3: Stranded lora_v2 pruned from GCS+S3 & unloaded from pods; lora_v1 synced
        assert sorted(report.adapter_drift.pruned_object_keys) == sorted([v2_w_key, v2_c_key])
        assert report.adapter_drift.sidecar_synced_pods == 2
        assert zoekt_vllm_srv.pod_loras["pod0"] == {"lora_v1"}
        assert zoekt_vllm_srv.pod_loras["pod1"] == {"lora_v1"}

        # Verify Plane 4: Stale task reclaimed & watermark rewound in PG17
        assert report.stale_tasks_reclaimed == 1
        with app_mgr.scoped_transaction(principal_a, enable_hnsw_iterative_scan=False) as conn:
            wm_after = pg_repo.get_connector_watermark(conn, dom_a, "gitlab", "svc-core")
            assert wm_after is not None and wm_after.watermark_cursor == sha_c1
            assert wm_after.sync_tier == "reconcile"

        # Canary Tenant B MUST remain 100% untouched across PG17, Zoekt, and GCS
        assert gcs_store.exists(canary_key, expected_tenant_id=tenant_b) is True
        canary_hits = local_zoekt.search(
            principal=principal_b, domain_id=dom_b, query="canary_ok"
        ).file_matches
        assert [f.file_path for f in canary_hits] == ["src/canary.py"]

        # Pass 2 immediately after MUST detect zero drift (idempotent)
        report_2 = reconciler.reconcile(principal_a, dom_a)
        assert report_2.drift_detected is False
        assert report_2.healed is True
        assert report_2.warnings == []

        sidecar.close()
        remote_zoekt.close()
    finally:
        s3_gcs_srv.close()
        zoekt_vllm_srv.close()
        _cleanup_pg_tenants_and_domains([tenant_a, tenant_b], [dom_a, dom_b])


def test_chaos_corrupted_active_adapter_in_s3_and_gcs_fails_closed_to_last_known_good(
    tmp_path: Path,
) -> None:
    """Corrupted active adapter_v2 in S3/GCS is retired fail-closed and last-known-good promoted."""
    tenant_a = uuid.uuid4()
    dom = "eval_m6_02_adapter_failover"
    scopes = ["public", "public_eng"]
    principal = PrincipalContext(
        user_id="mlops_failover",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=scopes,
        tenant_id=tenant_a,
    )
    base_model = "google/gemma-4-26B-A4B-it"
    pinned_rev = hashlib.sha256(b"gemma-4-adapter-dr").hexdigest()
    ed_priv = ed25519.Ed25519PrivateKey.generate()
    ed_pub_pem = (
        ed_priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    verifier = CosignBlobVerifier(trusted_public_keys_pem={"dr-key": ed_pub_pem})

    s3_gcs_srv = LoopbackS3AndGCSChaosServer()
    vllm_srv = LoopbackZoektAndVLLMChaosServer(base_model_id=base_model, pod_count=2)
    s3_store = s3_gcs_srv.make_s3_store("scout-dr-s3")
    gcs_store = s3_gcs_srv.make_gcs_store("scout-dr-gcs")
    mem_repo = InMemoryRLSRepository()
    router = AdapterRouter(default_base_model_id=base_model)

    try:
        w1 = _build_safetensors_bytes(1.0)
        w2 = _build_safetensors_bytes(2.0)
        cfg_bytes = json.dumps(
            {
                "base_model_name_or_path": base_model,
                "base_model_revision_sha256": pinned_rev,
                "r": 16,
                "lora_alpha": 32,
                "use_rslora": True,
                "use_dora": False,
            }
        ).encode("utf-8")

        v1_w = build_tenant_domain_key(tenant_a, dom, "adapters/lora_v1/adapter_model.safetensors")
        v1_c = build_tenant_domain_key(tenant_a, dom, "adapters/lora_v1/adapter_config.json")
        v1_s = build_tenant_domain_key(
            tenant_a, dom, "adapters/lora_v1/adapter_model.safetensors.sig"
        )
        sig1 = sign_blob_for_cosign(ed_priv, w1)
        gcs_store.put_bytes(v1_w, w1, expected_tenant_id=tenant_a)
        gcs_store.put_bytes(v1_c, cfg_bytes, expected_tenant_id=tenant_a)
        gcs_store.put_bytes(v1_s, sig1.encode("utf-8"), expected_tenant_id=tenant_a)

        v2_w = build_tenant_domain_key(tenant_a, dom, "adapters/lora_v2/adapter_model.safetensors")
        v2_c = build_tenant_domain_key(tenant_a, dom, "adapters/lora_v2/adapter_config.json")
        v2_s = build_tenant_domain_key(
            tenant_a, dom, "adapters/lora_v2/adapter_model.safetensors.sig"
        )
        sig2 = sign_blob_for_cosign(ed_priv, w2)
        s3_store.put_bytes(v2_w, w2, expected_tenant_id=tenant_a)
        s3_store.put_bytes(v2_c, cfg_bytes, expected_tenant_id=tenant_a)
        s3_store.put_bytes(v2_s, sig2.encode("utf-8"), expected_tenant_id=tenant_a)

        now = datetime.now(UTC)
        mem_repo.register_model_artifact(
            principal,
            ModelArtifactRecord(
                tenant_id=tenant_a,
                domain_id=dom,
                artifact_id="lora_v1",
                base_model_id=base_model,
                base_model_revision_sha256=pinned_rev,
                storage_uri=f"gs://scout-dr-gcs/{v1_w}",
                weights_object_key=v1_w,
                config_object_key=v1_c,
                signature_object_key=v1_s,
                weights_sha256=hashlib.sha256(w1).hexdigest(),
                cosign_signature_b64=sig1,
                cosign_key_id="dr-key",
                training_scope_ceiling=["public_eng"],
                allowed_scopes=scopes,
                status="retired",
                is_active=False,
                created_at=now - timedelta(hours=2),
            ),
        )
        mem_repo.register_model_artifact(
            principal,
            ModelArtifactRecord(
                tenant_id=tenant_a,
                domain_id=dom,
                artifact_id="lora_v2",
                base_model_id=base_model,
                base_model_revision_sha256=pinned_rev,
                storage_uri=f"s3://scout-dr-s3/{v2_w}",
                weights_object_key=v2_w,
                config_object_key=v2_c,
                signature_object_key=v2_s,
                weights_sha256=hashlib.sha256(w2).hexdigest(),
                cosign_signature_b64=sig2,
                cosign_key_id="dr-key",
                training_scope_ceiling=["public_eng"],
                allowed_scopes=scopes,
                status="active",
                is_active=True,
                created_at=now - timedelta(minutes=5),
            ),
        )
        # Corrupt active lora_v2 weights in S3 & pretend pods currently hold lora_v2
        s3_gcs_srv.corrupt_s3_bytes("scout-dr-s3", v2_w, b"CORRUPTED_WEIGHTS_BLOB")
        vllm_srv.pod_loras["pod0"].add("lora_v2")
        vllm_srv.pod_loras["pod1"].add("lora_v2")

        sidecar = AdapterSyncSidecar(
            expected_base_model_id=base_model,
            expected_base_model_revision_sha256=pinned_rev,
            staging_root=tmp_path / "dr_stage",
            cosign_verifier=verifier,
            tenant_id=tenant_a,
            sidecar_scopes=scopes,
            in_memory_repo=mem_repo,
            object_stores_by_scheme={"gs": gcs_store, "s3": s3_store},
            vllm_pod_endpoints=[vllm_srv.pod_v1_url(0), vllm_srv.pod_v1_url(1)],
            adapter_router=router,
        )
        reconciler = StateReconciler(
            repository=mem_repo,
            object_stores_by_scheme={"gs": gcs_store, "s3": s3_store},
            adapter_sidecar=sidecar,
            adapter_router=router,
            promote_last_known_good_adapter=True,
        )

        report = reconciler.reconcile(principal, dom)
        assert report.healed is True
        assert report.adapter_drift.active_artifact_id_before == "lora_v2"
        assert report.adapter_drift.active_artifact_id_after == "lora_v1"
        assert report.adapter_drift.retired_broken_artifacts == ["lora_v2"]
        assert report.adapter_drift.promoted_artifact_id == "lora_v1"
        assert report.adapter_drift.sidecar_synced_pods == 2
        assert vllm_srv.pod_loras["pod0"] == {"lora_v1"}
        assert vllm_srv.pod_loras["pod1"] == {"lora_v1"}
        sidecar.close()
    finally:
        s3_gcs_srv.close()
        vllm_srv.close()


# ---------------------------------------------------------------------------
# 4. Chaos: Dropped Webhooks + Worker Crash Reclaim + Tier-2/3 Convergence
# ---------------------------------------------------------------------------


@pytest.mark.postgres
def test_chaos_dropped_webhooks_healed_by_tier2_watermarks_and_tier3_reconcile() -> None:
    """Proves dropped webhooks + worker crash recover to 100% parity via Tier-2/Tier-3 sync."""
    cfg = ScoutSettings()
    tenant_a = uuid.uuid4()
    dom = make_ephemeral_pg_domain_id("m6_02_webhooks")
    scopes = ["public", "public_eng", "public_docs", "jira:project:ENG:read"]
    principal = PrincipalContext(
        user_id="webhook_dr_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes,
        tenant_id=tenant_a,
    )
    app_mgr = RLSConnectionManager(cfg.db_app_url, use_pool=False, pool_role=PoolRole.INGESTION)
    pg_repo = PostgresRLSRepository()
    queue = PostgresTaskQueue(connection_manager=app_mgr, default_tenant_id=tenant_a)

    now = datetime.now(UTC) - timedelta(minutes=15)
    after = now + timedelta(minutes=10)

    class _ChaosConfluence(ConfluenceConnectorPlugin):
        def __init__(self) -> None:
            super().__init__()
            self.remote_pages: dict[str, ConfluencePageDTO] = {
                "cf-100": ConfluencePageDTO(
                    page_id="cf-100",
                    space_key="ENG",
                    title="DR Runbook",
                    version_number=1,
                    author_username="ops",
                    created_at=now,
                    last_modified_at=now,
                    source_url="https://wiki.internal/cf-100",
                    body_storage="<p>Initial DR baseline v1.</p>",
                ),
                "cf-101": ConfluencePageDTO(
                    page_id="cf-101",
                    space_key="ENG",
                    title="Deprecated Runbook",
                    version_number=1,
                    author_username="ops",
                    created_at=now,
                    last_modified_at=now,
                    source_url="https://wiki.internal/cf-101",
                    body_storage="<p>Legacy runbook slated for deletion.</p>",
                ),
            }

        def sync_space_to_repository(
            self,
            domain_id: str,
            space_key: str,
            repository: Any,
            conn: Any = None,
            principal: Any = None,
            current_pages: Any = None,
            *,
            since_updated_at: datetime | None = None,
            deleted_page_ids: list[str] | None = None,
            reconcile_deletes: bool = False,
            **kwargs: Any,
        ) -> SyncResult:
            pages = [
                p
                for p in self.remote_pages.values()
                if since_updated_at is None or p.last_modified_at >= since_updated_at
            ]
            return super().sync_space_to_repository(
                domain_id,
                space_key,
                repository,
                conn=conn,
                principal=principal,
                current_pages=pages,
                since_updated_at=since_updated_at,
                deleted_page_ids=deleted_page_ids,
                reconcile_deletes=reconcile_deletes,
                space_allowed_scopes=["public", "public_docs"],
            )

    class _ChaosJira(JiraConnectorPlugin):
        def __init__(self) -> None:
            super().__init__()
            self.remote_issues: list[JiraIssueDTO] = [
                JiraIssueDTO(
                    issue_id="10042",
                    issue_key="ENG-42",
                    project_key="ENG",
                    summary="Missed webhook Jira issue",
                    description_text=(
                        "Recovered via Tier-2 watermark polling after dropped webhook."
                    ),
                    issue_type="Bug",
                    status="Open",
                    source_url="https://jira.internal/ENG-42",
                    created_at=now,
                    updated_at=after,
                )
            ]

        def search_issues_incremental(
            self,
            project_key: str,
            since_datetime: datetime | None = None,
            batch_size: int = 100,
        ) -> list[JiraIssueDTO]:
            _ = batch_size
            return [
                iss
                for iss in self.remote_issues
                if iss.project_key == project_key
                and (since_datetime is None or iss.updated_at >= since_datetime)
            ]

    cf_conn = _ChaosConfluence()
    jira_conn = _ChaosJira()
    coord = ConnectorSyncCoordinator(
        repository=pg_repo,
        connection_manager=app_mgr,
        confluence_connector=cf_conn,
        jira_connector=jira_conn,
    )
    worker = IndexingWorker(
        queue,
        repository=pg_repo,
        connection_manager=app_mgr,
        sync_coordinator=coord,
        build_repo_wiki=False,
        allowed_scopes=scopes,
    )

    try:
        with psycopg.connect(cfg.db_admin_url) as admin_conn:
            create_ephemeral_pg_domain(admin_conn, domain_id=dom, tenant_id=tenant_a)

        # Initial baseline sync for Confluence ENG at T1
        queue.enqueue_task(dom, "confluence_update", {"space_key": "ENG"}, tenant_id=tenant_a)
        assert worker.run_once() is True

        # CHAOS 1: Upstream creates cf-102, updates cf-100, deletes cf-101, and creates ENG-42,
        #          but ALL Tier-1 webhooks are DROPPED!
        cf_conn.remote_pages.pop("cf-101")
        cf_conn.remote_pages["cf-100"] = ConfluencePageDTO(
            page_id="cf-100",
            space_key="ENG",
            title="DR Runbook",
            version_number=2,
            author_username="ops",
            created_at=now,
            last_modified_at=after,
            source_url="https://wiki.internal/cf-100",
            body_storage="<p>Updated DR procedure after dropped webhook.</p>",
        )
        cf_conn.remote_pages["cf-102"] = ConfluencePageDTO(
            page_id="cf-102",
            space_key="ENG",
            title="New DR Service",
            version_number=1,
            author_username="ops",
            created_at=after,
            last_modified_at=after,
            source_url="https://wiki.internal/cf-102",
            body_storage="<p>Newly created service doc missed by webhook.</p>",
        )

        # CHAOS 2: Worker claims a watermark_sync task and crashes mid-flight (stuck PROCESSING)
        crashed_tid = queue.enqueue_task(
            dom,
            "watermark_sync",
            {"source_id": "confluence", "container_key": "ENG"},
            tenant_id=tenant_a,
        )
        stuck = queue.fetch_next_task("crashed-pod-1", tenant_id=tenant_a)
        assert stuck is not None and stuck.task_id == crashed_tid
        with app_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
            conn.execute(
                "UPDATE indexing_queue SET started_at = NOW() - INTERVAL '45 seconds' "
                "WHERE task_id = %s;",
                (crashed_tid,),
            )

        # Reclaim stale task & run Tier-2 watermark recovery + Tier-3 deletion reconciliation
        reclaimed = queue.reclaim_stale_tasks(stale_threshold_seconds=10, tenant_id=tenant_a)
        assert reclaimed == 1
        assert worker.run_once() is True  # executes reclaimed confluence watermark_sync

        queue.enqueue_task(
            dom,
            "watermark_sync",
            {"source_id": "jira", "container_key": "ENG"},
            tenant_id=tenant_a,
        )
        assert worker.run_once() is True

        queue.enqueue_task(
            dom,
            "reconcile_sweep",
            {
                "source_id": "confluence",
                "container_key": "ENG",
                "deleted_ids": ["cf-101"],
                "reconcile_contracts": False,
                "rebuild_wiki": False,
            },
            tenant_id=tenant_a,
        )
        assert worker.run_once() is True
        assert queue.list_dead_letter_records(tenant_id=tenant_a) == []

        with app_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
            chunks = pg_repo.list_visible_doc_chunks(conn, dom)
            doc_ids = {c.doc_id for c in chunks}
            assert "confluence:cf-100" in doc_ids
            assert "confluence:cf-102" in doc_ids
            assert "confluence:cf-101" not in doc_ids
            assert "jira:ENG-42" in doc_ids
            assert any(
                "Updated DR procedure after dropped webhook" in c.verbatim_content
                for c in chunks
                if c.doc_id == "confluence:cf-100"
            )
            wm_cf = pg_repo.get_connector_watermark(conn, dom, "confluence", "ENG")
            wm_jira = pg_repo.get_connector_watermark(conn, dom, "jira", "ENG")
            assert wm_cf is not None and wm_cf.sync_tier == "reconcile"
            assert wm_jira is not None and wm_jira.sync_tier == "watermark"
    finally:
        _cleanup_pg_tenants_and_domains([tenant_a], [dom])


# ---------------------------------------------------------------------------
# 5. Chaos: Redis / Memorystore FLUSHALL & TCP Disconnect Fail-Closed Recovery
# ---------------------------------------------------------------------------


def test_chaos_redis_memorystore_flushall_acl_cache_and_rate_limit_fail_closed() -> None:
    """Proves ACL cache, rate-limit coordinator, & GPU quotas fail closed after Redis FLUSHALL."""
    srv = FaultInjectingRESP2Server()
    try:
        tid_a = uuid.uuid4()
        cache_client = _StdlibRESP2Client(srv.redis_url, timeout=1.0)
        acl_backend = RedisACLCacheBackend(client=cache_client, key_prefix="acl:")
        mirror_pod1 = CachedPermissionMirrorService(ttl_seconds=60, redis_backend=acl_backend)
        mirror_pod2 = CachedPermissionMirrorService(ttl_seconds=60, redis_backend=acl_backend)

        alice_full = EnterpriseIdentityProfile(
            user_id="alice",
            gitlab_memberships=[
                GitLabProjectMembership(project_path="core/public-svc", role=GitLabRole.DEVELOPER),
                GitLabProjectMembership(
                    project_path="secret/alpha-enclave", role=GitLabRole.DEVELOPER
                ),
            ],
        )
        bob_profile = EnterpriseIdentityProfile(
            user_id="bob",
            gitlab_memberships=[
                GitLabProjectMembership(
                    project_path="secret/beta-enclave", role=GitLabRole.DEVELOPER
                ),
            ],
        )

        # Warm distributed Redis ACL cache on pod 1; pod 2 reads from Redis
        alice_scopes = mirror_pod1.get_user_scopes(alice_full)
        bob_scopes = mirror_pod1.get_user_scopes(bob_profile)
        assert "repo:secret/alpha-enclave:read" in alice_scopes
        assert "repo:secret/beta-enclave:read" in bob_scopes
        assert mirror_pod2.get_user_scopes(alice_full) == alice_scopes
        assert any(k.startswith("acl:") for k in srv.snapshot_keys())

        # CHAOS: Upstream revokes alice's secret repo + Redis FLUSHALL + drop TCP connections
        alice_revoked = EnterpriseIdentityProfile(
            user_id="alice",
            gitlab_memberships=[
                GitLabProjectMembership(project_path="core/public-svc", role=GitLabRole.DEVELOPER),
            ],
        )
        srv.flush_all()
        srv.drop_active_connections()
        assert srv.snapshot_keys() == {}

        # Next resolution on pod 2 (no local hit after invalidate_user / fresh pod) reconnects
        # cleanly, re-resolves upstream fail-closed, and prevents cross-tenant/revoked leakage
        mirror_pod2.invalidate_user("alice")
        alice_after = mirror_pod2.get_user_scopes(alice_revoked)
        assert "repo:core/public-svc:read" in alice_after
        assert "repo:secret/alpha-enclave:read" not in alice_after
        assert "repo:secret/beta-enclave:read" not in alice_after

        # Rate-limit coordinator survives FLUSHALL + TCP outage & resumes enforcement
        rl_client = _StdlibRESP2Client(srv.redis_url, timeout=1.0)
        rl = RedisRateLimitCoordinator(redis_client=rl_client, key_prefix="rl")
        rl.record_rate_limit(tid_a, "gitlab", retry_after_seconds=30.0, container_key="svc-core")
        with pytest.raises(ConnectorRateLimitError):
            rl.check_or_raise(tid_a, "gitlab", container_key="svc-core")

        srv.flush_all()
        srv.drop_active_connections()
        # Local instance fallback still enforces fail-closed cooldown even after Redis wipe
        with pytest.raises(ConnectorRateLimitError):
            rl.check_or_raise(tid_a, "gitlab", container_key="svc-core")

        # Separate pod after Redis wipe sees unlocked state and re-registers cleanly on reconnect
        rl_pod2 = RedisRateLimitCoordinator(redis_client=rl_client, key_prefix="rl")
        rl_pod2.check_or_raise(tid_a, "gitlab", container_key="svc-core")
        rl_pod2.record_rate_limit(
            tid_a, "gitlab", retry_after_seconds=25.0, container_key="svc-core"
        )
        assert rl_pod2.remaining_backoff_seconds(tid_a, "gitlab", container_key="svc-core") > 0.0

        cache_client.close()
        rl_client.close()
    finally:
        srv.close_server()


# ---------------------------------------------------------------------------
# 6. Chaos: vLLM Overload (429/503/Timeout + Noisy Neighbor) -> degraded_extractive_reranked
# ---------------------------------------------------------------------------


@pytest.mark.postgres
def test_chaos_vllm_overload_429_503_timeout_and_noisy_neighbor_degraded_extractive_reranked() -> (
    None
):
    """Proves HTTP 429/503/timeout & noisy-neighbor overload degrade cleanly with $0 spend."""
    cfg = ScoutSettings()
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    dom = make_ephemeral_pg_domain_id("m6_02_vllm")
    scopes = ["public", "public_eng", "public_docs"]
    principal_a = PrincipalContext(
        user_id="noisy_user_a",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes,
        tenant_id=tenant_a,
    )
    principal_b = PrincipalContext(
        user_id="interactive_user_b",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes,
        tenant_id=tenant_b,
    )

    app_mgr = RLSConnectionManager(cfg.db_app_url, use_pool=False, pool_role=PoolRole.INGESTION)
    pg_repo = PostgresRLSRepository()
    vllm_srv = LoopbackZoektAndVLLMChaosServer(base_model_id="google/gemma-4-26B-A4B-it")
    vllm_srv.timeout_sleep_seconds = 0.6

    try:
        with psycopg.connect(cfg.db_admin_url) as admin_conn:
            create_ephemeral_pg_domain(admin_conn, domain_id=dom, tenant_id=tenant_a)
            create_ephemeral_pg_domain(admin_conn, domain_id=dom, tenant_id=tenant_b)

        for tid, p_ctx, tag in (
            (tenant_a, principal_a, "alpha_tenant"),
            (tenant_b, principal_b, "beta_tenant"),
        ):
            repo_id = uuid.uuid4()
            file_id = uuid.uuid4()
            sym_id = uuid.uuid4()
            vec = _embed_fixture_text(f"def verify_circuit_breaker_{tag}(): pass", dim=768)
            dvec = _embed_fixture_text(f"Circuit breaker architecture doc for {tag}", dim=768)
            with app_mgr.scoped_transaction(p_ctx, enable_hnsw_iterative_scan=False) as conn:
                pg_repo.insert_repository(
                    conn,
                    RepositoryRecord(
                        repo_id=repo_id,
                        domain_id=dom,
                        repo_name=f"repo-{tag}",
                        source_type="github",
                        remote_url=f"https://github.internal/{tag}",
                        default_branch="main",
                        allowed_scopes=scopes,
                        head_commit_sha="c" * 40,
                        tenant_id=tid,
                    ),
                )
                pg_repo.insert_code_file(
                    conn,
                    CodeFileRecord(
                        file_id=file_id,
                        domain_id=dom,
                        repo_name=f"repo-{tag}",
                        file_path="src/breaker.py",
                        git_commit_sha="c" * 40,
                        git_blob_sha="d" * 40,
                        language="python",
                        allowed_scopes=scopes,
                        raw_content=(
                            f"def verify_circuit_breaker_{tag}(failures: int) -> bool:\n"
                            "    return failures >= 3\n"
                        ),
                        line_count=2,
                        repo_id=repo_id,
                        tenant_id=tid,
                    ),
                )
                pg_repo.insert_ast_symbol(
                    conn,
                    ASTSymbolRecord(
                        symbol_id=sym_id,
                        file_id=file_id,
                        domain_id=dom,
                        qualified_name=f"verify_circuit_breaker_{tag}",
                        symbol_kind="function",
                        start_line=1,
                        end_line=2,
                        signature_skeleton=(
                            f"def verify_circuit_breaker_{tag}(failures: int) -> bool:"
                        ),
                        contextual_summary=f"Checks circuit breaker threshold for {tag}.",
                        allowed_scopes=scopes,
                        embedding=vec,
                        repo_id=repo_id,
                        tenant_id=tid,
                    ),
                )
                pg_repo.insert_doc_chunk(
                    conn,
                    DocChunkRecord(
                        chunk_id=uuid.uuid4(),
                        domain_id=dom,
                        doc_id=f"doc-{tag}",
                        doc_title=f"Circuit Breaker Guide ({tag})",
                        source_uri=f"https://docs.internal/{tag}",
                        market_scope="GLOBAL",
                        published_date=datetime.now(UTC).date(),
                        authority_weight=1.0,
                        section_heading="Degraded Mode",
                        contextual_header="Fallback policy",
                        verbatim_content=(
                            f"Circuit breaker {tag} trips when vLLM returns 429 or 503."
                        ),
                        metadata_json={},
                        allowed_scopes=scopes,
                        embedding=dvec,
                        tenant_id=tid,
                    ),
                )

        qm = RedisGPUQuotaManager(
            max_global_inflight=4,
            max_tenant_inflight=1,
            max_team_inflight=2,
            max_user_inflight=2,
        )
        executor = OpenWeightsVLLMExecutor(
            model_id="google/gemma-4-26B-A4B-it",
            endpoint_url=vllm_srv.pod_v1_url(0),
            runtime_backend="vllm",
            timeout_seconds=0.25,
            gpu_quota_manager=qm,
            auto_discover_max_len=False,
        )

        # Build live PG17 BoundGraphRepo-compatible adapters per principal via PostgresRLSRepository
        class _PGScopedWorkflowRepo:
            def __init__(self, mgr: RLSConnectionManager, repo: PostgresRLSRepository) -> None:
                self._mgr = mgr
                self._repo = repo

            def list_visible_code_files(self, p: PrincipalContext, d: str) -> list[CodeFileRecord]:
                with self._mgr.scoped_transaction(p) as conn:
                    return self._repo.list_visible_code_file_records(conn, d)

            def list_visible_ast_symbols(
                self, p: PrincipalContext, d: str
            ) -> list[ASTSymbolRecord]:
                with self._mgr.scoped_transaction(p) as conn:
                    return self._repo.list_visible_ast_symbol_records(conn, d)

            def list_visible_ast_edges(self, p: PrincipalContext, d: str) -> list[Any]:
                with self._mgr.scoped_transaction(p) as conn:
                    return self._repo.list_visible_ast_edge_records(conn, d)

            def list_visible_doc_chunks(self, p: PrincipalContext, d: str) -> list[DocChunkRecord]:
                with self._mgr.scoped_transaction(p) as conn:
                    return self._repo.list_visible_doc_chunks(conn, d)

            def search_visible_doc_chunks(
                self, p: PrincipalContext, d: str, query_text: str = "", limit: int = 10
            ) -> list[DocChunkRecord]:
                with self._mgr.scoped_transaction(p) as conn:
                    return self._repo.list_visible_doc_chunks(conn, d)[:limit]

            def search_visible_ast_symbols(
                self, p: PrincipalContext, d: str, query_text: str = "", limit: int = 10
            ) -> list[ASTSymbolRecord]:
                with self._mgr.scoped_transaction(p) as conn:
                    return self._repo.search_visible_ast_symbols(conn, d, query_text, limit=limit)

        registry = DomainRegistry()
        registry.load_all()
        eval_manifest = DomainPackManifest(
            domain_id=dom,
            display_name="M6-02 Chaos Test Domain",
            base_model_id="google/gemma-4-26B-A4B-it",
            lora_adapter_id="eval_none",
            lora_adapter_path=f"domains/{dom}/adapters/eval_none",
            output_contract="ImplementationContextPack",
            lora_enabled=False,
            default_scopes=scopes,
            authority_rules=AuthorityRules(primary_truth_source="code"),
            mcp_read_tools_allowed=[
                "code_repo_mcp:assemble_verified_context_pack",
                "code_repo_mcp:search_code_trigram",
                "code_repo_mcp:lookup_ast_symbol",
                "code_repo_mcp:expand_ast_1hop_closure",
                "code_repo_mcp:slice_file_lines",
                "code_repo_mcp:search_ast_symbols",
                "code_repo_mcp:get_symbol_definition",
            ],
        )
        registry.register(eval_manifest)
        registry.register(eval_manifest, tenant_id=tenant_a)
        registry.register(eval_manifest, tenant_id=tenant_b)

        wf_repo = _PGScopedWorkflowRepo(app_mgr, pg_repo)
        wf = ScoutWorkflow(
            registry=registry,
            repo=wf_repo,
            llm_backend=executor,
            gpu_quota_manager=qm,
        )

        # 1. Live HTTP 503, 429, and Timeout on vLLM -> all degrade cleanly with 0 paid tokens
        for mode in ("503", "429", "timeout"):
            vllm_srv.completion_mode = mode
            pack = wf.run(
                user_prompt="Inspect verify_circuit_breaker_alpha_tenant in src/breaker.py",
                principal=principal_a,
                domain_id=dom,
            )
            assert isinstance(pack, ImplementationContextPack)
            assert pack.is_abstention is False
            assert pack.synthesis_mode == "degraded_extractive_reranked"
            assert any(
                "verify_circuit_breaker_alpha_tenant" in s.verbatim_code_with_lines
                for s in pack.code_slices
            )

        # 2. Healthy vLLM + Tenant A saturates tenant_a quota (noisy neighbor):
        #    Tenant A degrades to degraded_extractive_reranked while Tenant B runs cleanly!
        vllm_srv.completion_mode = "ok"
        vllm_srv.custom_completion_text = json.dumps(
            [
                {
                    "repo_name": "repo-beta_tenant",
                    "file_path": "src/breaker.py",
                    "start_line": 1,
                    "end_line": 2,
                    "qualified_symbol": "verify_circuit_breaker_beta_tenant",
                    "selection_reason": "Matched",
                    "modification_intent": "reference_only",
                }
            ]
        )
        with qm.lease(
            PrincipalContext(
                user_id="noisy_worker_a2",
                role=RoleType.ENGINEER,
                allowed_scopes=scopes,
                tenant_id=tenant_a,
            ),
            priority=GPUPriorityClass.P2,
            estimated_tokens=512,
        ):
            noisy_pack_a = wf.run(
                user_prompt="Inspect verify_circuit_breaker_alpha_tenant in src/breaker.py",
                principal=principal_a,
                domain_id=dom,
            )
            assert isinstance(noisy_pack_a, ImplementationContextPack)
            assert noisy_pack_a.synthesis_mode == "degraded_extractive_reranked"

            # Tenant B is completely unimpacted by Tenant A's saturated quota
            pack_b = wf.run(
                user_prompt="Inspect verify_circuit_breaker_beta_tenant in src/breaker.py",
                principal=principal_b,
                domain_id=dom,
            )
            assert isinstance(pack_b, ImplementationContextPack)
            assert pack_b.is_abstention is False
            assert pack_b.synthesis_mode in {"ast_extractive", "hybrid_federated"}
            assert any(
                "verify_circuit_breaker_beta_tenant" in s.verbatim_code_with_lines
                for s in pack_b.code_slices
            )
            assert all("alpha_tenant" not in s.verbatim_code_with_lines for s in pack_b.code_slices)

        # 3. P0 preempts P2
        with qm.lease(principal_b, priority=GPUPriorityClass.P0, estimated_tokens=100):
            with pytest.raises(GPUOverloadError) as exc_info:
                qm.acquire(principal_a, priority=GPUPriorityClass.P2, estimated_tokens=100)
            assert exc_info.value.scope == "preempted_by_p0"

        executor.close()
    finally:
        vllm_srv.close()
        _cleanup_pg_tenants_and_domains([tenant_a, tenant_b], [dom])


# ---------------------------------------------------------------------------
# 7. CLI `reconcile-state` & InMemoryRLSRepository Parity
# ---------------------------------------------------------------------------


@pytest.mark.postgres
def test_cli_reconcile_state_and_inmemory_parity(capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies InMemoryRLSRepository reconciliation parity and `context-scout reconcile-state`."""
    tid = uuid.uuid4()
    dom = "eval_m6_02_mem_cli"
    scopes = ["public", "public_eng", "public_docs"]
    principal = PrincipalContext(
        user_id="cli:reconcile-state",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=scopes,
        tenant_id=tid,
    )
    mem_repo = InMemoryRLSRepository()
    mem_store = InMemoryObjectStore(bucket="mem-cli-reconcile")
    mem_zoekt = LocalTrigramZoektIndex()
    mem_q = InMemoryTaskQueue(default_tenant_id=tid)
    embedder = LocalSemanticEmbedder()

    repo_id = uuid.uuid4()
    file_id = uuid.uuid4()
    sym_id = uuid.uuid4()
    sha1 = "e" * 40
    sha2 = "f" * 40
    mem_repo.register_embedding_space(
        principal,
        EmbeddingSpaceRecord(
            tenant_id=tid,
            domain_id=dom,
            space_id="space_mem_v1",
            model_id="google/embeddinggemma-2",
            dimensions=768,
            status="active",
            is_active=True,
        ),
    )
    mem_repo.insert_repository(
        principal,
        RepositoryRecord(
            repo_id=repo_id,
            domain_id=dom,
            repo_name="mem-svc",
            source_type="local_git",
            remote_url="file:///mem-svc",
            default_branch="main",
            allowed_scopes=scopes,
            head_commit_sha=sha1,
            tenant_id=tid,
        ),
    )
    mem_repo.insert_code_file(
        principal,
        CodeFileRecord(
            file_id=file_id,
            domain_id=dom,
            repo_name="mem-svc",
            file_path="app.py",
            git_commit_sha=sha1,
            git_blob_sha="9" * 40,
            language="python",
            allowed_scopes=scopes,
            raw_content="def mem_entry() -> int:\n    return 1\n",
            line_count=2,
            repo_id=repo_id,
            tenant_id=tid,
        ),
    )
    mem_repo.insert_ast_symbol(
        principal,
        ASTSymbolRecord(
            symbol_id=sym_id,
            file_id=file_id,
            domain_id=dom,
            qualified_name="mem_entry",
            symbol_kind="function",
            start_line=1,
            end_line=2,
            signature_skeleton="def mem_entry() -> int:",
            contextual_summary="Memory entry function.",
            allowed_scopes=scopes,
            embedding=_make_unit_vec(768, 1),
            repo_id=repo_id,
            tenant_id=tid,
            embedding_space_id="space_drifted_old",
        ),
    )
    mem_repo.upsert_connector_watermark(
        principal,
        ConnectorWatermarkRecord(
            tenant_id=tid,
            domain_id=dom,
            source_id="local_git",
            container_key="mem-svc",
            watermark_cursor=sha2,
            sync_tier="watermark",
        ),
    )
    mem_zoekt.index_file(
        tenant_id=tid,
        domain_id=dom,
        repo_name="mem-svc",
        file_path="ghost.py",
        content="def ghost(): pass\n",
        allowed_scopes=scopes,
    )

    out = cmd_reconcile_state(
        dom,
        tenant_id=tid,
        allowed_scopes=scopes,
        repository=mem_repo,
        zoekt_index=mem_zoekt,
        object_store=mem_store,
        task_queue=mem_q,
        embedder=embedder,
    )
    assert out["healed"] is True
    assert out["drift_detected"] is True
    assert out["watermarks_rewound"] == 1
    assert out["embedding_drift"]["symbol_rows_reembedded"] == 1
    assert out["zoekt_purged_files"] == 1
    assert out["zoekt_reindexed_files"] == 1

    # Also exercise real CLI parser `reconcile-state` against live PG17 for a clean eval domain
    cfg = ScoutSettings()
    cli_dom = make_ephemeral_pg_domain_id("m6_02_cli_pg")
    try:
        with psycopg.connect(cfg.db_admin_url) as admin_conn:
            create_ephemeral_pg_domain(admin_conn, domain_id=cli_dom, tenant_id=DEFAULT_TENANT_ID)
        rc = cli_main(["reconcile-state", "--domain-id", cli_dom])
        assert rc == 0
        cli_json = json.loads(capsys.readouterr().out)
        assert cli_json["domain_id"] == cli_dom
        assert cli_json["healed"] is True
    finally:
        _cleanup_pg_tenants_and_domains([], [cli_dom])


# ---------------------------------------------------------------------------
# 8. Regression Test for Connector Non-Default Tenant Propagation
# ---------------------------------------------------------------------------


@pytest.mark.postgres
def test_connectors_propagate_non_default_tenant_id_under_pg17_force_rls() -> None:
    """Proves GitLab, GitHub, Confluence, and Jira propagate non-default tenant_id in PG17."""
    cfg = ScoutSettings()
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    assert tenant_a != DEFAULT_TENANT_ID and tenant_b != DEFAULT_TENANT_ID
    dom = make_ephemeral_pg_domain_id("m6_02_disc030")
    before_sha = "0" * 40
    gl_after = "a1" * 20
    gh_after = "b2" * 20
    gl_src = b"def gl_tenant_fn(x: int) -> int:\n    return x + 1\n"
    gh_src = b"def gh_tenant_fn(x: int) -> int:\n    return x + 2\n"

    gl_transport = httpx.MockTransport(
        lambda req: (
            httpx.Response(
                200,
                json=[{"path": "src/gl_mod.py", "type": "blob", "mode": "100644"}],
            )
            if "/repository/tree" in req.url.path
            else httpx.Response(200, content=gl_src)
        )
    )
    gh_transport = httpx.MockTransport(
        lambda req: (
            httpx.Response(
                200,
                json={"tree": [{"path": "src/gh_mod.py", "type": "blob", "mode": "100644"}]},
            )
            if "/git/trees/" in req.url.path
            else httpx.Response(200, content=gh_src)
        )
    )

    gl_plugin = GitLabConnectorPlugin(
        client=httpx.Client(base_url="https://gl.local", transport=gl_transport)
    )
    gh_plugin = GitHubEnterpriseConnectorPlugin(
        installation_id=77,
        client=httpx.Client(base_url="https://gh.local", transport=gh_transport),
    )

    gl_scopes = gl_plugin.resolve_project_member_scopes(
        "grp/gl-repo",
        access_level=GitLabRole.REPORTER,
    )
    gh_acl = GitHubAppInstallationACL(
        installation_id=77,
        repository_selection="selected",
        repositories=["org/gh-repo"],
        permissions={"contents": "read"},
    )
    gh_scopes = gh_plugin.resolve_repo_allowed_scopes("org/gh-repo", installation_acl=gh_acl)
    cf_scopes = ["confluence:space:eng:read"]
    jira_scopes = ["jira:project:PLAT:read"]
    scopes_a = sorted({*gl_scopes, *gh_scopes, *cf_scopes, *jira_scopes, "public"})

    principal_a = PrincipalContext(
        user_id="disc030_a",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes_a,
        tenant_id=tenant_a,
    )
    principal_b = PrincipalContext(
        user_id="disc030_b",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes_a,
        tenant_id=tenant_b,
    )
    app_mgr = RLSConnectionManager(cfg.db_app_url, use_pool=False, pool_role=PoolRole.INGESTION)
    pg_repo = PostgresRLSRepository()

    now = datetime.now(UTC)

    class _CF(ConfluenceConnectorPlugin):
        def fetch_space_pages(
            self,
            space_key: str,
            since_updated_at: datetime | None = None,
            *,
            fetch_restrictions: bool = True,
        ) -> list[ConfluencePageDTO]:
            del since_updated_at, fetch_restrictions
            return [
                ConfluencePageDTO(
                    page_id="cf-t030",
                    space_key=space_key,
                    title="Tenant Isolated Runbook",
                    version_number=1,
                    author_username="ops",
                    created_at=now,
                    last_modified_at=now,
                    source_url="https://wiki.local/cf-t030",
                    body_storage="<p>Confluence tenant isolation verified.</p>",
                )
            ]

    class _Jira(JiraConnectorPlugin):
        def search_issues_incremental(
            self, project_key: str, since_datetime: datetime | None = None, batch_size: int = 100
        ) -> list[JiraIssueDTO]:
            del since_datetime, batch_size
            return [
                JiraIssueDTO(
                    issue_id="9030",
                    issue_key=f"{project_key}-30",
                    project_key=project_key,
                    summary="Verify Jira tenant RLS propagation",
                    description_text="Jira issue inserted with non-default tenant_id.",
                    issue_type="Task",
                    status="Open",
                    source_url=f"https://jira.local/{project_key}-30",
                    created_at=now,
                    updated_at=now,
                )
            ]

    try:
        with psycopg.connect(cfg.db_admin_url) as admin_conn:
            create_ephemeral_pg_domain(admin_conn, domain_id=dom, tenant_id=tenant_a)
            create_ephemeral_pg_domain(admin_conn, domain_id=dom, tenant_id=tenant_b)

        with app_mgr.scoped_transaction(principal_a, enable_hnsw_iterative_scan=False) as conn:
            gl_res = gl_plugin.process_push_webhook(
                QueueTask(
                    task_id=uuid.uuid4(),
                    domain_id=dom,
                    event_type="gitlab_push",
                    payload={
                        "project": {
                            "id": 101,
                            "path_with_namespace": "grp/gl-repo",
                            "default_branch": "main",
                        },
                        "ref": "refs/heads/main",
                        "before": before_sha,
                        "after": gl_after,
                        "commits": [
                            {
                                "id": gl_after,
                                "message": "init gl",
                                "author": {"name": "a", "email": "a@example.com"},
                                "timestamp": now.isoformat(),
                            }
                        ],
                    },
                    tenant_id=tenant_a,
                    priority=10,
                    status="PROCESSING",
                    worker_id="disc030-worker",
                    attempts=1,
                    max_attempts=3,
                    error_message=None,
                    scheduled_at=now,
                    created_at=now,
                ),
                pg_repo,
                conn=conn,
                principal=principal_a,
            )
            assert gl_res.errors == []

            gh_res = gh_plugin.process_push_webhook(
                QueueTask(
                    task_id=uuid.uuid4(),
                    domain_id=dom,
                    event_type="github_push",
                    payload={
                        "repository": {
                            "id": 202,
                            "full_name": "org/gh-repo",
                            "default_branch": "main",
                        },
                        "ref": "refs/heads/main",
                        "before": before_sha,
                        "after": gh_after,
                        "commits": [
                            {
                                "id": gh_after,
                                "message": "init gh",
                                "author": {"name": "b", "email": "b@example.com"},
                                "timestamp": now.isoformat(),
                            }
                        ],
                        "installation_acl": gh_acl.model_dump(),
                    },
                    tenant_id=tenant_a,
                    priority=10,
                    status="PROCESSING",
                    worker_id="disc030-worker",
                    attempts=1,
                    max_attempts=3,
                    error_message=None,
                    scheduled_at=now,
                    created_at=now,
                ),
                pg_repo,
                conn=conn,
                principal=principal_a,
            )
            assert gh_res.errors == []

            cf_plugin = _CF()
            cf_res = cf_plugin.sync_space_to_repository(
                dom,
                "ENG",
                pg_repo,
                conn=conn,
                principal=principal_a,
                space_allowed_scopes=cf_scopes,
            )
            assert cf_res.items_updated == 1

            jira_plugin = _Jira()
            jira_res = jira_plugin.sync_project_to_repository(
                dom, "PLAT", pg_repo, conn=conn, principal=principal_a
            )
            assert jira_res.items_updated == 1

        with app_mgr.scoped_transaction(principal_a, enable_hnsw_iterative_scan=False) as conn:
            repos_a = {r.repo_name: r for r in pg_repo.list_repositories(conn, dom)}
            assert {"grp/gl-repo", "org/gh-repo"}.issubset(repos_a.keys())
            assert pg_repo.get_repo_commits(conn, repos_a["grp/gl-repo"].repo_id)
            assert pg_repo.get_repo_commits(conn, repos_a["org/gh-repo"].repo_id)
            syms_a = {s.qualified_name for s in pg_repo.list_visible_ast_symbol_records(conn, dom)}
            assert {"grp/gl-repo::gl_tenant_fn", "org/gh-repo::gh_tenant_fn"}.issubset(syms_a)
            assert len(pg_repo.list_confluence_pages(conn, dom, space_key="ENG")) == 1
            chunks_a = {c.doc_id for c in pg_repo.list_visible_doc_chunks(conn, dom)}
            assert {"confluence:cf-t030", "jira:PLAT-30"}.issubset(chunks_a)

        with app_mgr.scoped_transaction(principal_b, enable_hnsw_iterative_scan=False) as conn:
            assert pg_repo.list_repositories(conn, dom) == []
            assert pg_repo.list_visible_code_file_records(conn, dom) == []
            assert pg_repo.list_visible_ast_symbol_records(conn, dom) == []
            assert pg_repo.list_confluence_pages(conn, dom, space_key="ENG") == []
            assert pg_repo.list_visible_doc_chunks(conn, dom) == []
    finally:
        gl_plugin.client.close()
        gh_plugin.client.close()
        _cleanup_pg_tenants_and_domains([tenant_a, tenant_b], [dom])


def test_pr5_resilience_guards_reconciler_sidecar_and_object_store(
    tmp_path: Path,
) -> None:
    """Verifies batched reconciler, sidecar disk/stale-dir guards, and ObjectStore size cap."""
    from context_scout.serving.adapter_sync_sidecar import (
        AdapterStagingCapacityError,
        AdapterSyncSidecar,
        CosignBlobVerifier,
    )
    from context_scout.storage.object_store import InMemoryObjectStore, ObjectTooLargeError

    # 1. ObjectStore max_object_bytes enforcement
    store = InMemoryObjectStore(max_object_bytes=16)
    key = f"tenants/{uuid.uuid4()}/domains/code_architecture_scout/small.bin"
    store.put_bytes(key, b"1234567890")
    assert store.get_bytes(key) == b"1234567890"
    with pytest.raises(ObjectTooLargeError):
        store.put_bytes(key, b"x" * 32)

    # 2. AdapterSyncSidecar stale .staging_* sweep + disk_usage fail-closed guard
    staging_root = tmp_path / "adapters"
    domain_parent = staging_root / str(DEFAULT_TENANT_ID) / "code_architecture_scout"
    domain_parent.mkdir(parents=True)
    stale_dir = domain_parent / ".staging_old_1234"
    stale_dir.mkdir()
    old_ts = time.time() - 3600
    os.utime(stale_dir, (old_ts, old_ts))

    sidecar = AdapterSyncSidecar(
        expected_base_model_id="google/gemma-4-26B-A4B-it",
        expected_base_model_revision_sha256="a" * 64,
        staging_root=staging_root,
        cosign_verifier=CosignBlobVerifier(),
        in_memory_repo=InMemoryRLSRepository(),
        min_staging_free_bytes=1024 * 1024,
        stale_staging_ttl_seconds=60.0,
    )
    assert sidecar._sweep_stale_staging_dirs(domain_parent) == 1
    assert not stale_dir.exists()
    sidecar.close()
    assert issubclass(AdapterStagingCapacityError, RuntimeError)


def test_chaos_testcontainers_live_alembic_and_nobypassrls_reconciler() -> None:
    """Exercises ChaosTestcontainersOrchestrator(use_containers=True) when Docker is running."""
    assert _docker_daemon_available(), "Docker daemon must be running for live Testcontainers gate"
    with ChaosTestcontainersOrchestrator(use_containers=True) as orch:
        assert "scout_app_user" in orch.app_db_url
        assert orch.app_db_url != orch.admin_db_url
        with psycopg.connect(orch.app_db_url) as conn:
            row = conn.execute(
                "SELECT current_user, r.rolbypassrls "
                "FROM pg_roles r WHERE r.rolname = current_user;"
            ).fetchone()
            assert row is not None
            assert row[0] == "scout_app_user"
            assert row[1] is False
            rls_row = conn.execute(
                "SELECT relrowsecurity, relforcerowsecurity "
                "FROM pg_class WHERE relname = 'doc_chunks';"
            ).fetchone()
            assert rls_row == (True, True)
