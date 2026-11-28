"""S3/MinIO and keyless GCP GCS artifact storage SPI with tenant keys and SHA-256 checks."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol
from urllib.parse import quote
from uuid import UUID, uuid4

import boto3
import httpx
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from context_scout.config import ScoutSettings

logger = logging.getLogger(__name__)

_SAFE_DOMAIN_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9_]+$")
_TENANT_KEY_RE: Final[re.Pattern[str]] = re.compile(
    r"^tenants/[0-9a-fA-F-]{36}/domains/[a-z0-9_]+/.+$"
)
_GCS_DEFAULT_API_ROOT: Final[str] = "https://storage.googleapis.com"
_GCS_RW_SCOPE: Final[str] = "https://www.googleapis.com/auth/devstorage.read_write"
DEFAULT_MAX_OBJECT_BYTES: Final[int] = 512 * 1024 * 1024


class ObjectTooLargeError(ValueError):
    """Raised fail-closed when an object exceeds the configured max_object_bytes bound."""


def validate_object_key(key: str, *, expected_tenant_id: UUID | None = None) -> str:
    """Validates object key structure and prevents cross-tenant or path traversal."""
    cleaned = key.strip()
    if not cleaned or "\x00" in cleaned or cleaned.startswith("/") or "\\" in cleaned:
        raise ValueError(f"Invalid object storage key: {key!r}")
    parts = cleaned.split("/")
    if any(p in {"", ".", ".."} for p in parts):
        raise ValueError(f"Path traversal or empty segment rejected in object key: {key!r}")
    if not _TENANT_KEY_RE.fullmatch(cleaned):
        raise ValueError(
            f"Object key must use 'tenants/<tenant_id>/domains/<domain_id>/<path>' format: {key!r}"
        )
    if expected_tenant_id is not None:
        key_tenant = UUID(parts[1])
        if key_tenant != expected_tenant_id:
            raise PermissionError(
                f"Cross-tenant key denied: expected {expected_tenant_id}, got {key_tenant}"
            )
    return cleaned


def validate_object_prefix(prefix: str, *, expected_tenant_id: UUID | None = None) -> str:
    """Validates a list_keys prefix against path traversal and cross-tenant access."""
    if "\x00" in prefix or "\\" in prefix:
        raise ValueError(f"Invalid object storage prefix: {prefix!r}")
    clean_prefix = prefix.strip().lstrip("/")
    segments = [seg for seg in clean_prefix.split("/") if seg]
    if any(seg in {".", ".."} for seg in segments):
        raise ValueError(f"Path traversal rejected in object storage prefix: {prefix!r}")
    if expected_tenant_id is not None:
        expected_root = f"tenants/{expected_tenant_id}/"
        if not clean_prefix.startswith(expected_root):
            raise PermissionError(
                f"Cross-tenant list_keys prefix denied for {expected_tenant_id}: {prefix!r}"
            )
    return clean_prefix


def build_tenant_domain_key(tenant_id: UUID, domain_id: str, relative_key: str) -> str:
    """Constructs a validated tenant-prefixed object key."""
    slug = domain_id.strip()
    if not _SAFE_DOMAIN_SLUG_RE.fullmatch(slug):
        raise ValueError(f"Invalid domain_id slug for object key: {domain_id!r}")
    rel = relative_key.strip().lstrip("/")
    return validate_object_key(
        f"tenants/{tenant_id}/domains/{slug}/{rel}",
        expected_tenant_id=tenant_id,
    )


@dataclass(frozen=True)
class ObjectMetadata:
    """Immutable metadata returned after storing an object."""

    uri: str
    key: str
    sha256: str
    size_bytes: int
    content_type: str = "application/octet-stream"


class ArtifactObjectStore(Protocol):
    """Protocol for tenant-scoped artifact and manifest object storage."""

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        expected_tenant_id: UUID | None = None,
    ) -> ObjectMetadata: ...

    def get_bytes(self, key: str, *, expected_tenant_id: UUID | None = None) -> bytes: ...

    def exists(self, key: str, *, expected_tenant_id: UUID | None = None) -> bool: ...

    def delete(self, key: str, *, expected_tenant_id: UUID | None = None) -> None: ...

    def list_keys(self, prefix: str, *, expected_tenant_id: UUID | None = None) -> list[str]: ...


class InMemoryObjectStore:
    """Thread-safe in-memory ArtifactObjectStore for offline development and tests."""

    def __init__(
        self,
        bucket: str = "local-inmemory",
        *,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
    ) -> None:
        self._bucket = bucket
        self._max_object_bytes = max(1, int(max_object_bytes))
        self._lock = threading.Lock()
        self._objects: dict[str, tuple[bytes, ObjectMetadata]] = {}

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        expected_tenant_id: UUID | None = None,
    ) -> ObjectMetadata:
        clean_key = validate_object_key(key, expected_tenant_id=expected_tenant_id)
        if len(data) > self._max_object_bytes:
            raise ObjectTooLargeError(
                f"Object {clean_key!r} size {len(data)} exceeds "
                f"max_object_bytes={self._max_object_bytes}"
            )
        digest = hashlib.sha256(data).hexdigest()
        meta = ObjectMetadata(
            uri=f"memory://{self._bucket}/{clean_key}",
            key=clean_key,
            sha256=digest,
            size_bytes=len(data),
            content_type=content_type,
        )
        with self._lock:
            self._objects[clean_key] = (bytes(data), meta)
        return meta

    def get_bytes(self, key: str, *, expected_tenant_id: UUID | None = None) -> bytes:
        clean_key = validate_object_key(key, expected_tenant_id=expected_tenant_id)
        with self._lock:
            entry = self._objects.get(clean_key)
        if entry is None:
            raise KeyError(f"Object not found in InMemoryObjectStore: {clean_key}")
        data, meta = entry
        if hashlib.sha256(data).hexdigest() != meta.sha256:
            raise ValueError(f"Object SHA-256 integrity mismatch for {clean_key}")
        return data

    def exists(self, key: str, *, expected_tenant_id: UUID | None = None) -> bool:
        clean_key = validate_object_key(key, expected_tenant_id=expected_tenant_id)
        with self._lock:
            return clean_key in self._objects

    def delete(self, key: str, *, expected_tenant_id: UUID | None = None) -> None:
        clean_key = validate_object_key(key, expected_tenant_id=expected_tenant_id)
        with self._lock:
            self._objects.pop(clean_key, None)

    def list_keys(self, prefix: str, *, expected_tenant_id: UUID | None = None) -> list[str]:
        clean_prefix = validate_object_prefix(prefix, expected_tenant_id=expected_tenant_id)
        with self._lock:
            return sorted(k for k in self._objects if k.startswith(clean_prefix))


class S3CompatibleObjectStore:
    """S3 / MinIO ArtifactObjectStore backed by boto3 with s3v4 and SHA-256 verification."""

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region_name: str = "eu-central-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        force_path_style: bool = True,
        client: Any | None = None,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
    ) -> None:
        if not bucket or not bucket.strip():
            raise ValueError("S3 bucket name must be non-empty.")
        self._bucket = bucket.strip()
        self._max_object_bytes = max(1, int(max_object_bytes))
        if client is not None:
            self._client = client
        else:
            cfg = BotocoreConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path" if force_path_style else "auto"},
                connect_timeout=3,
                read_timeout=15,
                retries={"max_attempts": 2, "mode": "standard"},
            )
            self._client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                config=cfg,
            )

    @property
    def client(self) -> Any:
        return self._client

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        expected_tenant_id: UUID | None = None,
    ) -> ObjectMetadata:
        clean_key = validate_object_key(key, expected_tenant_id=expected_tenant_id)
        if len(data) > self._max_object_bytes:
            raise ObjectTooLargeError(
                f"Object {clean_key!r} size {len(data)} exceeds "
                f"max_object_bytes={self._max_object_bytes}"
            )
        digest = hashlib.sha256(data).hexdigest()
        self._client.put_object(
            Bucket=self._bucket,
            Key=clean_key,
            Body=data,
            ContentType=content_type,
            Metadata={"sha256": digest},
        )
        return ObjectMetadata(
            uri=f"s3://{self._bucket}/{clean_key}",
            key=clean_key,
            sha256=digest,
            size_bytes=len(data),
            content_type=content_type,
        )

    def get_bytes(self, key: str, *, expected_tenant_id: UUID | None = None) -> bytes:
        clean_key = validate_object_key(key, expected_tenant_id=expected_tenant_id)
        resp = self._client.get_object(Bucket=self._bucket, Key=clean_key)
        content_len = resp.get("ContentLength")
        if content_len is not None and int(content_len) > self._max_object_bytes:
            raise ObjectTooLargeError(
                f"S3 object s3://{self._bucket}/{clean_key} ContentLength={content_len} "
                f"exceeds max_object_bytes={self._max_object_bytes}"
            )
        body_stream = resp["Body"]
        try:
            body = body_stream.read(self._max_object_bytes + 1)
        except TypeError:
            body = body_stream.read()
        data = body if isinstance(body, bytes) else bytes(body)
        if len(data) > self._max_object_bytes:
            raise ObjectTooLargeError(
                f"S3 object s3://{self._bucket}/{clean_key} stream size exceeds "
                f"max_object_bytes={self._max_object_bytes}"
            )
        expected_sha = str((resp.get("Metadata") or {}).get("sha256", "")).strip()
        if expected_sha and hashlib.sha256(data).hexdigest() != expected_sha:
            raise ValueError(f"S3 SHA-256 integrity mismatch for s3://{self._bucket}/{clean_key}")
        return data

    def exists(self, key: str, *, expected_tenant_id: UUID | None = None) -> bool:
        clean_key = validate_object_key(key, expected_tenant_id=expected_tenant_id)
        try:
            self._client.head_object(Bucket=self._bucket, Key=clean_key)
            return True
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def delete(self, key: str, *, expected_tenant_id: UUID | None = None) -> None:
        clean_key = validate_object_key(key, expected_tenant_id=expected_tenant_id)
        self._client.delete_object(Bucket=self._bucket, Key=clean_key)

    def list_keys(self, prefix: str, *, expected_tenant_id: UUID | None = None) -> list[str]:
        clean_prefix = validate_object_prefix(prefix, expected_tenant_id=expected_tenant_id)
        keys: list[str] = []
        continuation_token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": clean_prefix}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            resp = self._client.list_objects_v2(**kwargs)
            for item in resp.get("Contents") or []:
                item_key = str(item.get("Key", ""))
                if item_key:
                    keys.append(item_key)
            if not resp.get("IsTruncated"):
                break
            continuation_token = resp.get("NextContinuationToken")
        return sorted(keys)


def _normalize_gcs_bucket_name(raw_bucket: str) -> str:
    cleaned = raw_bucket.strip()
    if cleaned.startswith("gs://"):
        cleaned = cleaned[len("gs://") :]
        if cleaned.startswith("/"):
            raise ValueError("GCS bucket name must be non-empty.")
        cleaned = cleaned.split("/", 1)[0].strip()
    else:
        cleaned = cleaned.strip("/")
        if "/" in cleaned:
            cleaned = cleaned.split("/", 1)[0].strip()
    if not cleaned or cleaned in {".", ".."} or "\x00" in cleaned or "\\" in cleaned:
        raise ValueError("GCS bucket name must be non-empty.")
    return cleaned


class GCSObjectStore:
    """Keyless Google Cloud Storage ArtifactObjectStore via ADC / GKE Workload Identity.

    Authenticates strictly via Application Default Credentials (`google.auth.default`)
    or GKE Workload Identity Federation (`iam.gke.io/gcp-service-account`) with zero
    static HMAC or service-account JSON keys (`iam.disableServiceAccountKeyCreation`),
    plus native `STORAGE_EMULATOR_HOST` support for air-gapped and offline testing.
    """

    def __init__(
        self,
        bucket: str,
        *,
        project: str | None = None,
        emulator_host: str | None = None,
        credentials: Any | None = None,
        http_client: httpx.Client | None = None,
        client: Any | None = None,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
    ) -> None:
        self._bucket = _normalize_gcs_bucket_name(bucket)
        self._max_object_bytes = max(1, int(max_object_bytes))
        self._project = project.strip() if project and project.strip() else None
        raw_emulator = (
            emulator_host if emulator_host is not None else os.environ.get("STORAGE_EMULATOR_HOST")
        )
        if raw_emulator and raw_emulator.strip():
            em = raw_emulator.strip().rstrip("/")
            if not em.startswith(("http://", "https://")):
                em = f"http://{em}"
            self._emulator_host: str | None = em
        else:
            self._emulator_host = None
        self._credentials = credentials
        self._sdk_client = client
        self._http_client = http_client or httpx.Client(timeout=15.0)
        self._cred_lock = threading.Lock()

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def project(self) -> str | None:
        return self._project

    @property
    def emulator_host(self) -> str | None:
        return self._emulator_host

    @property
    def uses_keyless_adc(self) -> bool:
        return True

    def _api_root(self) -> str:
        return self._emulator_host if self._emulator_host is not None else _GCS_DEFAULT_API_ROOT

    def _resolve_auth_headers(self) -> dict[str, str]:
        if self._emulator_host is not None and self._credentials is None:
            headers = {"Authorization": "Bearer gcs-emulator-token"}
            if self._project:
                headers["x-goog-user-project"] = self._project
            return headers

        with self._cred_lock:
            creds = self._credentials
            if creds is None:
                import google.auth

                creds, discovered_project = google.auth.default(scopes=[_GCS_RW_SCOPE])
                self._credentials = creds
                if self._project is None and discovered_project:
                    self._project = str(discovered_project)

            valid = getattr(creds, "valid", False)
            token = getattr(creds, "token", None)
            if not valid or not token:
                from google.auth.transport.requests import Request as GoogleAuthRequest

                creds.refresh(GoogleAuthRequest())
                token = getattr(creds, "token", None)

        if not token:
            raise RuntimeError(
                "Keyless GCP Application Default Credentials did not yield an access token."
            )
        headers = {"Authorization": f"Bearer {token}"}
        if self._project:
            headers["x-goog-user-project"] = self._project
        return headers

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        expected_tenant_id: UUID | None = None,
    ) -> ObjectMetadata:
        clean_key = validate_object_key(key, expected_tenant_id=expected_tenant_id)
        if len(data) > self._max_object_bytes:
            raise ObjectTooLargeError(
                f"Object {clean_key!r} size {len(data)} exceeds "
                f"max_object_bytes={self._max_object_bytes}"
            )
        digest = hashlib.sha256(data).hexdigest()

        if self._sdk_client is not None and hasattr(self._sdk_client, "bucket"):
            bkt = self._sdk_client.bucket(self._bucket)
            blob = bkt.blob(clean_key)
            blob.metadata = {"sha256": digest}
            blob.upload_from_string(data, content_type=content_type)
            return ObjectMetadata(
                uri=f"gs://{self._bucket}/{clean_key}",
                key=clean_key,
                sha256=digest,
                size_bytes=len(data),
                content_type=content_type,
            )

        boundary = f"===============scout_gcs_multipart_{uuid4().hex}=="
        while boundary.encode("ascii") in data:
            boundary = f"===============scout_gcs_multipart_{uuid4().hex}=="
        meta_json = json.dumps(
            {
                "name": clean_key,
                "contentType": content_type,
                "metadata": {"sha256": digest},
            }
        ).encode("utf-8")
        body = (
            (f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n").encode()
            + meta_json
            + (f"\r\n--{boundary}\r\nContent-Type: {content_type}\r\n\r\n").encode()
            + data
            + f"\r\n--{boundary}--\r\n".encode()
        )

        headers = self._resolve_auth_headers()
        headers["Content-Type"] = f"multipart/related; boundary={boundary}"
        url = f"{self._api_root()}/upload/storage/v1/b/{quote(self._bucket, safe='')}/o"
        resp = self._http_client.post(
            url,
            params={"uploadType": "multipart", "name": clean_key},
            content=body,
            headers=headers,
        )
        resp.raise_for_status()
        return ObjectMetadata(
            uri=f"gs://{self._bucket}/{clean_key}",
            key=clean_key,
            sha256=digest,
            size_bytes=len(data),
            content_type=content_type,
        )

    def get_bytes(self, key: str, *, expected_tenant_id: UUID | None = None) -> bytes:
        clean_key = validate_object_key(key, expected_tenant_id=expected_tenant_id)
        if self._sdk_client is not None and hasattr(self._sdk_client, "bucket"):
            bkt = self._sdk_client.bucket(self._bucket)
            blob = bkt.blob(clean_key)
            if hasattr(blob, "exists") and not blob.exists():
                raise KeyError(
                    f"Object not found in GCSObjectStore: gs://{self._bucket}/{clean_key}"
                )
            if hasattr(blob, "reload"):
                blob.reload()
            blob_size = getattr(blob, "size", None)
            if blob_size is not None and int(blob_size) > self._max_object_bytes:
                raise ObjectTooLargeError(
                    f"GCS object gs://{self._bucket}/{clean_key} size={blob_size} "
                    f"exceeds max_object_bytes={self._max_object_bytes}"
                )
            raw_bytes = blob.download_as_bytes()
            data = raw_bytes if isinstance(raw_bytes, bytes) else bytes(raw_bytes)
            if len(data) > self._max_object_bytes:
                raise ObjectTooLargeError(
                    f"GCS object gs://{self._bucket}/{clean_key} exceeds "
                    f"max_object_bytes={self._max_object_bytes}"
                )
            expected_sha = str((getattr(blob, "metadata", None) or {}).get("sha256", "")).strip()
            if expected_sha and hashlib.sha256(data).hexdigest() != expected_sha:
                raise ValueError(
                    f"GCS SHA-256 integrity mismatch for gs://{self._bucket}/{clean_key}"
                )
            return data

        headers = self._resolve_auth_headers()
        encoded_obj = quote(clean_key, safe="")
        obj_url = f"{self._api_root()}/storage/v1/b/{quote(self._bucket, safe='')}/o/{encoded_obj}"

        meta_resp = self._http_client.get(obj_url, headers=headers)
        if meta_resp.status_code == 404:
            raise KeyError(f"Object not found in GCSObjectStore: gs://{self._bucket}/{clean_key}")
        meta_resp.raise_for_status()
        meta_payload = meta_resp.json() if meta_resp.content else {}
        meta_size = meta_payload.get("size")
        if meta_size is not None and int(meta_size) > self._max_object_bytes:
            raise ObjectTooLargeError(
                f"GCS object gs://{self._bucket}/{clean_key} size={meta_size} "
                f"exceeds max_object_bytes={self._max_object_bytes}"
            )
        expected_sha = str((meta_payload.get("metadata") or {}).get("sha256", "")).strip()

        media_resp = self._http_client.get(
            obj_url,
            params={"alt": "media"},
            headers=headers,
        )
        if media_resp.status_code == 404:
            raise KeyError(f"Object not found in GCSObjectStore: gs://{self._bucket}/{clean_key}")
        media_resp.raise_for_status()
        data = bytes(media_resp.content)
        if len(data) > self._max_object_bytes:
            raise ObjectTooLargeError(
                f"GCS object gs://{self._bucket}/{clean_key} exceeds "
                f"max_object_bytes={self._max_object_bytes}"
            )
        if expected_sha and hashlib.sha256(data).hexdigest() != expected_sha:
            raise ValueError(f"GCS SHA-256 integrity mismatch for gs://{self._bucket}/{clean_key}")
        return data

    def exists(self, key: str, *, expected_tenant_id: UUID | None = None) -> bool:
        clean_key = validate_object_key(key, expected_tenant_id=expected_tenant_id)
        if self._sdk_client is not None and hasattr(self._sdk_client, "bucket"):
            bkt = self._sdk_client.bucket(self._bucket)
            blob = bkt.blob(clean_key)
            return bool(blob.exists())

        headers = self._resolve_auth_headers()
        encoded_obj = quote(clean_key, safe="")
        obj_url = f"{self._api_root()}/storage/v1/b/{quote(self._bucket, safe='')}/o/{encoded_obj}"
        resp = self._http_client.get(obj_url, headers=headers)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    def delete(self, key: str, *, expected_tenant_id: UUID | None = None) -> None:
        clean_key = validate_object_key(key, expected_tenant_id=expected_tenant_id)
        if self._sdk_client is not None and hasattr(self._sdk_client, "bucket"):
            bkt = self._sdk_client.bucket(self._bucket)
            blob = bkt.blob(clean_key)
            if hasattr(blob, "exists") and not blob.exists():
                return
            blob.delete()
            return

        headers = self._resolve_auth_headers()
        encoded_obj = quote(clean_key, safe="")
        obj_url = f"{self._api_root()}/storage/v1/b/{quote(self._bucket, safe='')}/o/{encoded_obj}"
        resp = self._http_client.delete(obj_url, headers=headers)
        if resp.status_code == 404:
            return
        resp.raise_for_status()

    def list_keys(self, prefix: str, *, expected_tenant_id: UUID | None = None) -> list[str]:
        clean_prefix = validate_object_prefix(prefix, expected_tenant_id=expected_tenant_id)
        if self._sdk_client is not None and hasattr(self._sdk_client, "list_blobs"):
            blobs = self._sdk_client.list_blobs(self._bucket, prefix=clean_prefix)
            return sorted(str(getattr(b, "name", "")) for b in blobs if getattr(b, "name", ""))

        headers = self._resolve_auth_headers()
        list_url = f"{self._api_root()}/storage/v1/b/{quote(self._bucket, safe='')}/o"
        keys: list[str] = []
        page_token: str | None = None
        while True:
            params: dict[str, str] = {"prefix": clean_prefix}
            if page_token:
                params["pageToken"] = page_token
            resp = self._http_client.get(list_url, params=params, headers=headers)
            resp.raise_for_status()
            payload = resp.json() if resp.content else {}
            for item in payload.get("items") or []:
                name = str(item.get("name", ""))
                if name:
                    keys.append(name)
            next_token = payload.get("nextPageToken")
            if not next_token:
                break
            page_token = str(next_token)
        return sorted(keys)


_STORE_LOCK = threading.Lock()
_SHARED_INMEMORY_STORE = InMemoryObjectStore()


def get_artifact_object_store(settings: ScoutSettings | None = None) -> ArtifactObjectStore:
    """Returns GCSObjectStore (gs:// / SCOUT_GCS_BUCKET), S3CompatibleObjectStore, or InMemory."""
    from context_scout.config import ScoutSettings

    cfg = settings or ScoutSettings()
    if cfg.gcs_bucket and cfg.gcs_bucket.strip():
        return GCSObjectStore(
            bucket=cfg.gcs_bucket.strip(),
            project=cfg.gcs_project or cfg.gcp_project,
            emulator_host=cfg.gcs_emulator_host,
        )
    if cfg.s3_bucket and cfg.s3_bucket.strip():
        raw_s3 = cfg.s3_bucket.strip()
        if raw_s3.startswith("gs://"):
            return GCSObjectStore(
                bucket=raw_s3,
                project=cfg.gcs_project or cfg.gcp_project,
                emulator_host=cfg.gcs_emulator_host,
            )
        return S3CompatibleObjectStore(
            bucket=raw_s3,
            endpoint_url=cfg.s3_endpoint_url,
            region_name=cfg.s3_region,
            aws_access_key_id=cfg.s3_access_key_id,
            aws_secret_access_key=cfg.s3_secret_access_key,
            force_path_style=cfg.s3_force_path_style,
        )
    with _STORE_LOCK:
        return _SHARED_INMEMORY_STORE
