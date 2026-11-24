"""Per-Pod & Fleet-Coordinated vLLM Multi-LoRA AdapterSyncSidecar (vllm#12174).

Watches active ``model_artifacts`` in PostgreSQL 17 (strictly via ``NOBYPASSRLS`` runtime pool)
and fetches governed LoRA cartridges from ``ArtifactObjectStore`` — supporting both keyless
GCP ``GCSObjectStore`` (``gs://`` via GKE Workload Identity / ADC) and portable
``S3CompatibleObjectStore`` (``s3://`` / MinIO).

Before exposing any cartridge to ``vLLM`` (``127.0.0.1:8001/v1/load_lora_adapter``),
``AdapterSyncSidecar`` enforces fail-closed supply-chain & architectural gates:
1. ``base_model_id`` & 64-hex ``base_model_revision_sha256`` exact match against the serving pod.
2. Tenant-isolated object key validation & ``weights_sha256`` digest verification.
3. Cryptographic ``cosign`` signature / Sigstore ``.bundle`` verification (ECDSA P-256/P-384,
   Ed25519, RSA, or optional ``cosign verify-blob`` CLI).
4. ``.safetensors``-only weights, ``trust_remote_code=False``, ``r in {8, 16, 32}`` (``r <= 32``),
   ``use_rslora=True``, ``use_dora=False``, non-empty ``training_scope_ceiling``,
   ``canary_leak_count == 0``, and zero MoE router gate or Mamba SSM state tensors.
5. Two-phase atomic loading across ALL ``vLLM`` pod endpoints (remediating ``vllm#12174``
   ClusterIP desync) with automatic rollback on partial pod failure and blue/green unload of
   superseded/retired cartridges.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse
from uuid import UUID, uuid4

import httpx
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from context_scout.agents.llm_backend import AdapterRouter
from context_scout.agents.open_weights_vllm_executor import (
    ModelArchitectureAdapter,
    validate_lora_safetensors_for_vllm,
)
from context_scout.config import ModelGovernancePolicy
from context_scout.db.repositories import (
    InMemoryRLSRepository,
    ModelArtifactRecord,
    PostgresRLSRepository,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.schemas.contracts import DEFAULT_TENANT_ID, PrincipalContext, RoleType
from context_scout.storage.object_store import (
    ArtifactObjectStore,
    GCSObjectStore,
    InMemoryObjectStore,
    S3CompatibleObjectStore,
    validate_object_key,
)

logger = logging.getLogger(__name__)

DEFAULT_VLLM_ADMIN_ENDPOINT: Final[str] = "http://127.0.0.1:8001"
_FORBIDDEN_ADMIN_DB_ROLES: Final[frozenset[str]] = frozenset(
    {"postgres", "supabase_admin", "rdsadmin", "cloudsqlsuperuser"}
)
_SAFE_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ALLOWED_STORAGE_SCHEMES: Final[frozenset[str]] = frozenset({"gs", "s3", "memory"})


class AdapterSyncError(RuntimeError):
    """Base error raised by AdapterSyncSidecar."""


class AdapterVerificationError(AdapterSyncError):
    """Raised fail-closed when cartridge metadata, SHA-256, or LoRA policy checks fail."""


class CosignVerificationError(AdapterVerificationError):
    """Raised fail-closed when cosign cryptographic signature verification fails."""


class MultiPodAtomicLoadError(AdapterSyncError):
    """Raised when loading across all vLLM pods fails and rollback has been executed."""


@dataclass(frozen=True)
class LoadedPodCartridge:
    """Tracks a verified LoRA cartridge currently loaded across all vLLM pods."""

    tenant_id: UUID
    domain_id: str
    artifact_id: str
    weights_sha256: str
    base_model_revision_sha256: str
    staged_dir: Path
    pod_endpoints: tuple[str, ...]
    training_scope_ceiling: tuple[str, ...]
    lora_rank: int


@dataclass(frozen=True)
class AdapterSyncReport:
    """Summary of a single AdapterSyncSidecar reconciliation cycle."""

    tenant_id: UUID
    active_in_db: tuple[str, ...]
    loaded_artifacts: tuple[str, ...]
    newly_loaded: tuple[str, ...]
    unloaded_retired: tuple[str, ...]
    failed_artifacts: Mapping[str, str] = field(default_factory=dict)
    pod_count: int = 1


def sign_blob_for_cosign(
    private_key: Any, payload: bytes, *, as_sigstore_bundle: bool = False
) -> str:
    """Signs ``payload`` with ECDSA, Ed25519, or RSA, returning ``.sig`` or ``.bundle``."""
    if isinstance(private_key, ec.EllipticCurvePrivateKey):
        digest_alg: hashes.HashAlgorithm = (
            hashes.SHA384() if private_key.curve.key_size >= 384 else hashes.SHA256()
        )
        raw_sig = private_key.sign(payload, ec.ECDSA(digest_alg))
    elif isinstance(private_key, ed25519.Ed25519PrivateKey):
        raw_sig = private_key.sign(payload)
    elif isinstance(private_key, rsa.RSAPrivateKey):
        raw_sig = private_key.sign(payload, padding.PKCS1v15(), hashes.SHA256())
    else:
        raise TypeError(f"Unsupported private key type for cosign signing: {type(private_key)!r}")

    sig_b64 = base64.b64encode(raw_sig).decode("ascii")
    if not as_sigstore_bundle:
        return sig_b64

    sha256_b64 = base64.b64encode(hashlib.sha256(payload).digest()).decode("ascii")
    bundle = {
        "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
        "messageSignature": {
            "messageDigest": {
                "algorithm": "SHA2_256",
                "digest": sha256_b64,
            },
            "signature": sig_b64,
        },
    }
    return json.dumps(bundle, separators=(",", ":"), sort_keys=True)


class CosignBlobVerifier:
    """Verifies ``cosign`` detached ``.sig`` & Sigstore ``.bundle`` signatures fail-closed."""

    def __init__(
        self,
        trusted_public_keys_pem: Mapping[str, str | bytes] | None = None,
        *,
        trusted_ca_certs_pem: Sequence[str | bytes] | None = None,
        cosign_binary: str | None = None,
        cosign_public_key_paths: Mapping[str, Path] | None = None,
    ) -> None:
        self._public_keys: dict[str, Any] = {}
        for key_id, raw_pem in (trusted_public_keys_pem or {}).items():
            pem_bytes = raw_pem.encode("utf-8") if isinstance(raw_pem, str) else bytes(raw_pem)
            self._public_keys[str(key_id)] = load_pem_public_key(pem_bytes)
        self._ca_certs: list[x509.Certificate] = []
        for raw_ca in trusted_ca_certs_pem or ():
            ca_bytes = raw_ca.encode("utf-8") if isinstance(raw_ca, str) else bytes(raw_ca)
            self._ca_certs.append(x509.load_pem_x509_certificate(ca_bytes))
        self._cosign_binary = cosign_binary
        self._cosign_key_paths = dict(cosign_public_key_paths or {})

    @property
    def has_trusted_keys(self) -> bool:
        return bool(
            self._public_keys or self._ca_certs or (self._cosign_binary and self._cosign_key_paths)
        )

    def _extract_leaf_key_from_bundle(self, bundle: Mapping[str, Any]) -> Any | None:
        ver_mat = bundle.get("verificationMaterial")
        if not isinstance(ver_mat, Mapping) or not self._ca_certs:
            return None
        raw_cert_b64 = ""
        cert_obj = ver_mat.get("certificate")
        if isinstance(cert_obj, Mapping):
            raw_cert_b64 = str(cert_obj.get("rawBytes") or "").strip()
        elif isinstance(ver_mat.get("x509CertificateChain"), Mapping):
            certs = ver_mat["x509CertificateChain"].get("certificates") or []
            if isinstance(certs, list) and certs and isinstance(certs[0], Mapping):
                raw_cert_b64 = str(certs[0].get("rawBytes") or "").strip()
        if not raw_cert_b64:
            return None
        try:
            leaf_der = base64.b64decode(raw_cert_b64, validate=True)
            leaf_cert = x509.load_der_x509_certificate(leaf_der)
        except Exception as exc:
            raise CosignVerificationError(f"Malformed X.509 certificate in bundle: {exc}") from exc

        now = datetime.now(UTC)
        if now < leaf_cert.not_valid_before_utc or now > leaf_cert.not_valid_after_utc:
            raise CosignVerificationError("Bundle X.509 signing certificate is outside validity.")

        for ca_cert in self._ca_certs:
            try:
                leaf_cert.verify_directly_issued_by(ca_cert)
                return leaf_cert.public_key()
            except Exception as exc:
                logger.debug(
                    "Bundle leaf certificate did not match CA candidate (%s)",
                    type(exc).__name__,
                )
                continue
        raise CosignVerificationError(
            "Bundle X.509 certificate is not signed by any trusted CA root."
        )

    def _extract_raw_signature_and_bundle_key(
        self, signature_text: str, payload: bytes
    ) -> tuple[bytes, Any | None]:
        cleaned = signature_text.strip()
        if not cleaned:
            raise CosignVerificationError("Missing or empty cosign signature.")

        if cleaned.startswith("{"):
            try:
                bundle = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise CosignVerificationError(f"Malformed Sigstore JSON bundle: {exc}") from exc
            if not isinstance(bundle, Mapping):
                raise CosignVerificationError("Sigstore bundle root must be a JSON object.")
            msg_sig = bundle.get("messageSignature")
            if not isinstance(msg_sig, Mapping):
                raise CosignVerificationError("Sigstore bundle missing 'messageSignature' section.")
            msg_digest = msg_sig.get("messageDigest")
            if not isinstance(msg_digest, Mapping):
                raise CosignVerificationError(
                    "Sigstore bundle missing required 'messageDigest' object."
                )
            alg = str(msg_digest.get("algorithm") or "SHA2_256").upper()
            if alg not in {"SHA2_256", "SHA256"}:
                raise CosignVerificationError(
                    f"Unsupported Sigstore messageDigest algorithm: {alg!r}"
                )
            expected_digest_b64 = str(msg_digest.get("digest") or "").strip()
            actual_digest_b64 = base64.b64encode(hashlib.sha256(payload).digest()).decode("ascii")
            if not expected_digest_b64 or expected_digest_b64 != actual_digest_b64:
                raise CosignVerificationError(
                    "Sigstore bundle messageDigest mismatch against .safetensors payload."
                )
            raw_sig_b64 = str(msg_sig.get("signature") or "").strip()
            if not raw_sig_b64:
                raise CosignVerificationError("Sigstore bundle missing base64 'signature' field.")
            try:
                sig_bytes = base64.b64decode(raw_sig_b64, validate=True)
            except Exception as exc:
                raise CosignVerificationError(
                    f"Invalid base64 signature in Sigstore bundle: {exc}"
                ) from exc
            return sig_bytes, self._extract_leaf_key_from_bundle(bundle)

        try:
            return base64.b64decode(cleaned, validate=True), None
        except Exception as exc:
            raise CosignVerificationError(f"Invalid base64 cosign signature: {exc}") from exc

    def verify(
        self,
        payload: bytes,
        signature_text: str,
        *,
        key_id: str = "default",
    ) -> None:
        """Verifies ``signature_text`` over ``payload`` using ``key_id``; raises on any failure."""
        if not self.has_trusted_keys:
            raise CosignVerificationError(
                "No trusted cosign public keys configured; failing closed."
            )

        if self._cosign_binary and key_id in self._cosign_key_paths:
            key_path = self._cosign_key_paths[key_id]
            cleaned_sig = signature_text.strip()
            is_bundle = cleaned_sig.startswith("{")
            with tempfile.TemporaryDirectory(prefix="scout_cosign_") as tmp:
                tmp_dir = Path(tmp)
                blob_file = tmp_dir / "artifact.safetensors"
                sig_file = tmp_dir / ("artifact.bundle" if is_bundle else "artifact.sig")
                blob_file.write_bytes(payload)
                sig_file.write_text(cleaned_sig, encoding="utf-8")
                sig_flag = "--bundle" if is_bundle else "--signature"
                cmd = [
                    self._cosign_binary,
                    "verify-blob",
                    "--offline",
                    "--key",
                    str(key_path),
                    sig_flag,
                    str(sig_file),
                    str(blob_file),
                ]
                try:
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, check=False, timeout=15
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise CosignVerificationError(
                        f"cosign verify-blob execution failed ({type(exc).__name__}): {exc}"
                    ) from exc
                if proc.returncode != 0:
                    raise CosignVerificationError(
                        f"cosign verify-blob failed (exit {proc.returncode}): "
                        f"{proc.stderr or proc.stdout}"
                    )
                return

        sig_bytes, bundle_pub_key = self._extract_raw_signature_and_bundle_key(
            signature_text, payload
        )
        pub_key = bundle_pub_key if bundle_pub_key is not None else self._public_keys.get(key_id)
        if pub_key is None:
            raise CosignVerificationError(
                f"Untrusted or unknown cosign key_id '{key_id}'; known keys: "
                f"{sorted(self._public_keys)}"
            )
        try:
            if isinstance(pub_key, ec.EllipticCurvePublicKey):
                digest_alg: hashes.HashAlgorithm = (
                    hashes.SHA384() if pub_key.curve.key_size >= 384 else hashes.SHA256()
                )
                pub_key.verify(sig_bytes, payload, ec.ECDSA(digest_alg))
            elif isinstance(pub_key, ed25519.Ed25519PublicKey):
                pub_key.verify(sig_bytes, payload)
            elif isinstance(pub_key, rsa.RSAPublicKey):
                pub_key.verify(sig_bytes, payload, padding.PKCS1v15(), hashes.SHA256())
            else:
                raise CosignVerificationError(
                    f"Unsupported cosign public key algorithm: {type(pub_key).__name__}"
                )
        except InvalidSignature as exc:
            raise CosignVerificationError(
                f"Invalid cosign cryptographic signature for key_id '{key_id}'."
            ) from exc


class AdapterSyncSidecar:
    """Watches Postgres ``model_artifacts`` & GCS/S3 object stores, syncing all ``vLLM`` pods."""

    def __init__(
        self,
        *,
        expected_base_model_id: str,
        expected_base_model_revision_sha256: str,
        staging_root: Path,
        cosign_verifier: CosignBlobVerifier,
        tenant_id: UUID = DEFAULT_TENANT_ID,
        sidecar_scopes: Sequence[str] = ("tenant_common", "public_eng"),
        rls_conn_manager: RLSConnectionManager | None = None,
        in_memory_repo: InMemoryRLSRepository | None = None,
        default_object_store: ArtifactObjectStore | None = None,
        object_stores_by_scheme: Mapping[str, ArtifactObjectStore] | None = None,
        gcs_project: str | None = None,
        gcs_emulator_host: str | None = None,
        s3_endpoint_url: str | None = None,
        s3_region: str = "eu-central-1",
        vllm_pod_endpoints: Sequence[str] = (DEFAULT_VLLM_ADMIN_ENDPOINT,),
        vllm_admin_api_key: str | None = None,
        http_client: httpx.Client | None = None,
        adapter_router: AdapterRouter | None = None,
        poll_interval_seconds: float = 10.0,
        governance_policy: ModelGovernancePolicy | None = None,
        owns_rls_conn_manager: bool = False,
    ) -> None:
        self.policy = governance_policy or ModelGovernancePolicy.default()
        self.expected_base_model_id = self.policy.validate_local_model_id(expected_base_model_id)
        rev_clean = expected_base_model_revision_sha256.strip().lower()
        if len(rev_clean) != 64 or any(ch not in "0123456789abcdef" for ch in rev_clean):
            raise ValueError(
                "expected_base_model_revision_sha256 must be a 64-character hex SHA-256 digest."
            )
        self.expected_base_model_revision_sha256 = rev_clean

        if rls_conn_manager is None and in_memory_repo is None:
            raise ValueError(
                "AdapterSyncSidecar requires either rls_conn_manager (NOBYPASSRLS runtime pool) "
                "or in_memory_repo."
            )
        if rls_conn_manager is not None:
            self._validate_non_admin_rls_manager(rls_conn_manager)

        if not vllm_pod_endpoints:
            raise ValueError("vllm_pod_endpoints must contain at least one vLLM admin endpoint.")

        self.tenant_id = tenant_id
        self._principal = PrincipalContext(
            user_id="adapter_sync_sidecar",
            role=RoleType.ENGINEER,
            allowed_scopes=list(sidecar_scopes) or ["tenant_common"],
            tenant_id=tenant_id,
        )
        self._rls_conn_manager = rls_conn_manager
        self._owns_rls_conn_manager = owns_rls_conn_manager and rls_conn_manager is not None
        self._in_memory_repo = in_memory_repo
        self._pg_repo = PostgresRLSRepository()
        self.staging_root = staging_root.resolve()
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.cosign_verifier = cosign_verifier
        self._default_object_store = default_object_store
        self._stores_by_scheme: dict[str, ArtifactObjectStore] = {
            k.lower().rstrip(":/"): v for k, v in (object_stores_by_scheme or {}).items()
        }
        self._dynamic_bucket_stores: dict[tuple[str, str], ArtifactObjectStore] = {}
        self._gcs_project = gcs_project
        self._gcs_emulator_host = gcs_emulator_host
        self._s3_endpoint_url = s3_endpoint_url
        self._s3_region = s3_region

        self.vllm_pod_endpoints: tuple[str, ...] = tuple(
            ModelArchitectureAdapter.normalize_v1_base_url(ep) for ep in vllm_pod_endpoints
        )
        self._vllm_admin_api_key = vllm_admin_api_key or os.environ.get("SCOUT_VLLM_ADMIN_API_KEY")
        self._http_client = http_client or httpx.Client(timeout=15.0)
        self._owns_http_client = http_client is None
        self._adapter_router = adapter_router
        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))

        self._lock = threading.RLock()
        self._loaded_by_domain: dict[str, LoadedPodCartridge] = {}
        self._dirty_pod_adapters: set[tuple[str, str]] = set()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    @staticmethod
    def _validate_non_admin_rls_manager(mgr: RLSConnectionManager) -> None:
        raw_dsn = str(
            getattr(mgr, "_app_db_url", "")
            or getattr(mgr, "_conninfo", "")
            or getattr(mgr, "db_url", "")
            or ""
        )
        if raw_dsn:
            parsed = urlparse(raw_dsn)
            user = (parsed.username or "").strip().lower()
            if user in _FORBIDDEN_ADMIN_DB_ROLES:
                raise PermissionError(
                    f"AdapterSyncSidecar MUST NOT connect as privileged DB role '{user}'; "
                    "use RLS runtime pool ('scout_app_user' with NOBYPASSRLS)."
                )

    @property
    def loaded_cartridges(self) -> dict[str, LoadedPodCartridge]:
        with self._lock:
            return dict(self._loaded_by_domain)

    def close(self) -> None:
        self.stop()
        if self._owns_http_client:
            self._http_client.close()
        if self._owns_rls_conn_manager and self._rls_conn_manager is not None:
            close_fn = getattr(self._rls_conn_manager, "close", None)
            if callable(close_fn):
                close_fn()

    def _vllm_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._vllm_admin_api_key:
            headers["Authorization"] = f"Bearer {self._vllm_admin_api_key}"
        return headers

    def _fetch_active_artifacts_from_db(self) -> list[ModelArtifactRecord]:
        if self._rls_conn_manager is not None:
            with self._rls_conn_manager.scoped_transaction(
                self._principal, enable_hnsw_iterative_scan=False
            ) as conn:
                return self._pg_repo.list_active_model_artifacts(conn, artifact_kind="lora_adapter")
        assert self._in_memory_repo is not None
        return self._in_memory_repo.list_active_model_artifacts(
            self._principal, artifact_kind="lora_adapter"
        )

    @staticmethod
    def _validate_slug(value: str, field_name: str) -> str:
        cleaned = value.strip()
        if not _SAFE_SLUG_RE.fullmatch(cleaned) or ".." in cleaned:
            raise AdapterVerificationError(
                f"Invalid or unsafe {field_name} '{value}': must match single-segment slug "
                f"{_SAFE_SLUG_RE.pattern} without path traversal."
            )
        return cleaned

    @staticmethod
    def _assert_store_bucket_matches(
        store: ArtifactObjectStore,
        expected_bucket: str,
        storage_uri: str,
    ) -> ArtifactObjectStore:
        configured_bucket = str(
            getattr(store, "bucket", None) or getattr(store, "_bucket", None) or ""
        ).strip()
        if configured_bucket and configured_bucket != expected_bucket:
            raise AdapterVerificationError(
                f"Storage bucket mismatch for {storage_uri!r}: store configured for "
                f"'{configured_bucket}', URI requested '{expected_bucket}'."
            )
        return store

    def _resolve_object_store_for_uri(self, storage_uri: str) -> ArtifactObjectStore:
        parsed = urlparse(storage_uri.strip())
        scheme = parsed.scheme.lower()
        bucket = parsed.netloc.strip()

        if scheme not in _ALLOWED_STORAGE_SCHEMES:
            raise AdapterVerificationError(
                f"Unsupported storage_uri scheme '{scheme}://' for {storage_uri!r}; "
                "expected 'gs://' (GCSObjectStore) or 's3://' (S3CompatibleObjectStore)."
            )
        if not bucket or "/" in bucket or "\\" in bucket or bucket in {".", ".."}:
            raise AdapterVerificationError(
                f"Missing or invalid bucket in storage_uri: {storage_uri!r}"
            )

        if scheme in self._stores_by_scheme:
            return self._assert_store_bucket_matches(
                self._stores_by_scheme[scheme], bucket, storage_uri
            )

        if self._default_object_store is not None:
            if scheme == "gs" and isinstance(self._default_object_store, GCSObjectStore):
                return self._assert_store_bucket_matches(
                    self._default_object_store, bucket, storage_uri
                )
            if scheme == "s3" and isinstance(self._default_object_store, S3CompatibleObjectStore):
                return self._assert_store_bucket_matches(
                    self._default_object_store, bucket, storage_uri
                )
            if scheme == "memory" and isinstance(self._default_object_store, InMemoryObjectStore):
                return self._assert_store_bucket_matches(
                    self._default_object_store, bucket, storage_uri
                )

        cache_key = (scheme, bucket)
        if cache_key in self._dynamic_bucket_stores:
            return self._dynamic_bucket_stores[cache_key]

        if scheme == "gs":
            gcs_store = GCSObjectStore(
                bucket=bucket,
                project=self._gcs_project,
                emulator_host=self._gcs_emulator_host,
            )
            self._dynamic_bucket_stores[cache_key] = gcs_store
            return gcs_store

        if scheme == "s3":
            s3_store = S3CompatibleObjectStore(
                bucket=bucket,
                endpoint_url=self._s3_endpoint_url,
                region_name=self._s3_region,
            )
            self._dynamic_bucket_stores[cache_key] = s3_store
            return s3_store

        raise AdapterVerificationError(
            f"Unconfigured storage_uri scheme '{scheme}://' for {storage_uri!r}."
        )

    def _verify_and_stage_artifact(self, artifact: ModelArtifactRecord) -> Path:
        if artifact.tenant_id != self.tenant_id:
            raise AdapterVerificationError(
                f"Cross-tenant artifact rejected: expected {self.tenant_id}, "
                f"got {artifact.tenant_id}"
            )
        safe_domain_id = self._validate_slug(artifact.domain_id, "domain_id")
        safe_artifact_id = self._validate_slug(artifact.artifact_id, "artifact_id")
        if artifact.base_model_id != self.expected_base_model_id:
            raise AdapterVerificationError(
                f"base_model_id mismatch for '{artifact.artifact_id}': pod serves "
                f"'{self.expected_base_model_id}', artifact targets '{artifact.base_model_id}'."
            )
        if (
            artifact.base_model_revision_sha256.strip().lower()
            != self.expected_base_model_revision_sha256
        ):
            raise AdapterVerificationError(
                f"base_model_revision_sha256 mismatch for '{artifact.artifact_id}': pod pinned to "
                f"{self.expected_base_model_revision_sha256[:12]}..., artifact built against "
                f"{artifact.base_model_revision_sha256[:12]}..."
            )

        ceiling = tuple(s.strip() for s in artifact.training_scope_ceiling if s and s.strip())
        if not ceiling:
            raise AdapterVerificationError(
                f"Artifact '{artifact.artifact_id}' missing training_scope_ceiling (OWASP ASI06)."
            )
        if artifact.canary_leak_count != 0:
            raise AdapterVerificationError(
                f"Artifact '{artifact.artifact_id}' has non-zero canary_leak_count="
                f"{artifact.canary_leak_count} (OWASP ASI06)."
            )

        weights_key = validate_object_key(
            artifact.weights_object_key, expected_tenant_id=self.tenant_id
        )
        config_key = validate_object_key(
            artifact.config_object_key, expected_tenant_id=self.tenant_id
        )
        if not weights_key.lower().endswith(".safetensors"):
            raise AdapterVerificationError(
                f"Supply-chain policy requires .safetensors weights key, got '{weights_key}'."
            )

        store = self._resolve_object_store_for_uri(artifact.storage_uri)
        weights_bytes = store.get_bytes(weights_key, expected_tenant_id=self.tenant_id)
        actual_sha256 = hashlib.sha256(weights_bytes).hexdigest()
        if actual_sha256 != artifact.weights_sha256.strip().lower():
            raise AdapterVerificationError(
                f"weights_sha256 mismatch for '{artifact.artifact_id}': expected "
                f"{artifact.weights_sha256}, computed {actual_sha256}."
            )

        sig_text = artifact.cosign_signature_b64.strip()
        if artifact.signature_object_key.strip():
            sig_key = validate_object_key(
                artifact.signature_object_key, expected_tenant_id=self.tenant_id
            )
            sig_bytes = store.get_bytes(sig_key, expected_tenant_id=self.tenant_id)
            sig_text = sig_bytes.decode("utf-8").strip()
        if not sig_text:
            raise CosignVerificationError(
                f"Missing cosign signature for artifact '{artifact.artifact_id}'."
            )
        self.cosign_verifier.verify(
            weights_bytes,
            sig_text,
            key_id=artifact.cosign_key_id or "default",
        )

        config_bytes = store.get_bytes(config_key, expected_tenant_id=self.tenant_id)
        expected_cfg_sha = str(artifact.metadata_json.get("config_sha256") or "").strip().lower()
        if expected_cfg_sha and hashlib.sha256(config_bytes).hexdigest() != expected_cfg_sha:
            raise AdapterVerificationError(
                f"adapter_config.json SHA-256 mismatch for '{artifact.artifact_id}'."
            )
        try:
            config_dict = json.loads(config_bytes.decode("utf-8"))
        except Exception as exc:
            raise AdapterVerificationError(
                f"Malformed JSON in adapter_config.json for '{artifact.artifact_id}': {exc}"
            ) from exc
        if not isinstance(config_dict, Mapping):
            raise AdapterVerificationError("adapter_config.json root must be a JSON object.")

        cfg_base = str(config_dict.get("base_model_name_or_path") or "").strip()
        if cfg_base and cfg_base != self.expected_base_model_id:
            raise AdapterVerificationError(
                f"adapter_config.json base_model_name_or_path ('{cfg_base}') does not match "
                f"pod base model ('{self.expected_base_model_id}')."
            )
        cfg_rev = str(config_dict.get("base_model_revision_sha256") or "").strip().lower()
        if cfg_rev and cfg_rev != self.expected_base_model_revision_sha256:
            raise AdapterVerificationError(
                "adapter_config.json base_model_revision_sha256 does not match pod revision."
            )

        self.policy.validate_weights_safety(
            [Path(weights_key).name, Path(config_key).name],
            trust_remote_code=False,
            config_payload=config_dict,
        )
        verified_rank, _tensor_names = validate_lora_safetensors_for_vllm(
            weights_bytes,
            config_dict,
            base_model_id=self.expected_base_model_id,
        )
        if verified_rank != artifact.lora_rank:
            raise AdapterVerificationError(
                f"Rank mismatch between model_artifacts ({artifact.lora_rank}) and "
                f"adapter_config.json ({verified_rank})."
            )

        tenant_root = (self.staging_root / str(self.tenant_id)).resolve()
        if not tenant_root.is_relative_to(self.staging_root):
            raise AdapterVerificationError("Invalid tenant staging root.")
        domain_parent = (tenant_root / safe_domain_id).resolve()
        if not domain_parent.is_relative_to(tenant_root) or domain_parent == tenant_root:
            raise AdapterVerificationError("Path traversal detected in domain staging path.")
        final_dir = (domain_parent / safe_artifact_id).resolve()
        if not final_dir.is_relative_to(domain_parent) or final_dir == domain_parent:
            raise AdapterVerificationError("Path traversal detected in artifact staging path.")

        domain_parent.mkdir(parents=True, exist_ok=True)
        staging_tmp = domain_parent / f".staging_{safe_artifact_id}_{uuid4().hex[:8]}"
        staging_tmp.mkdir(parents=True, exist_ok=True)
        try:
            weights_filename = Path(weights_key).name
            (staging_tmp / weights_filename).write_bytes(weights_bytes)
            if weights_filename != "adapter_model.safetensors":
                # Ensure standard PEFT/vLLM filename exists alongside custom safetensors name
                (staging_tmp / "adapter_model.safetensors").write_bytes(weights_bytes)
            (staging_tmp / "adapter_config.json").write_bytes(config_bytes)

            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(staging_tmp, final_dir)
        except Exception:
            if staging_tmp.exists():
                shutil.rmtree(staging_tmp, ignore_errors=True)
            raise

        return final_dir

    def _load_across_all_vllm_pods(self, lora_name: str, staged_dir: Path) -> None:
        """Atomically loads ``lora_name`` across ALL vLLM pods; rolls back on partial failure."""
        succeeded_pods: list[str] = []
        payload = {
            "lora_name": lora_name,
            "lora_path": str(staged_dir),
        }
        headers = self._vllm_headers()

        for pod_v1 in self.vllm_pod_endpoints:
            try:
                resp = self._http_client.post(
                    f"{pod_v1}/load_lora_adapter",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code == 400 and "already been loaded" in resp.text.lower():
                    # Pod already holds lora_name (e.g. sidecar restart or prior failed rollback):
                    # unload and reload atomically so staged_dir weights are guaranteed current.
                    unload_first = self._http_client.post(
                        f"{pod_v1}/unload_lora_adapter",
                        headers=headers,
                        json={"lora_name": lora_name},
                    )
                    if unload_first.status_code not in {200, 204, 404}:
                        raise AdapterSyncError(
                            f"Pod {pod_v1} failed pre-reload /unload_lora_adapter "
                            f"(HTTP {unload_first.status_code}): {unload_first.text}"
                        )
                    resp = self._http_client.post(
                        f"{pod_v1}/load_lora_adapter",
                        headers=headers,
                        json=payload,
                    )
                if resp.status_code >= 400:
                    raise AdapterSyncError(
                        f"Pod {pod_v1} rejected /load_lora_adapter "
                        f"(HTTP {resp.status_code}): {resp.text}"
                    )
                self._dirty_pod_adapters.discard((pod_v1, lora_name))
                succeeded_pods.append(pod_v1)
            except Exception as exc:
                for loaded_pod in reversed(succeeded_pods):
                    try:
                        rb_resp = self._http_client.post(
                            f"{loaded_pod}/unload_lora_adapter",
                            headers=headers,
                            json={"lora_name": lora_name},
                        )
                        if rb_resp.status_code not in {200, 204, 404}:
                            self._dirty_pod_adapters.add((loaded_pod, lora_name))
                            logger.error(
                                "Rollback /unload_lora_adapter returned HTTP %d on %s for '%s'",
                                rb_resp.status_code,
                                loaded_pod,
                                lora_name,
                            )
                    except Exception as rb_exc:
                        self._dirty_pod_adapters.add((loaded_pod, lora_name))
                        logger.error(
                            "Rollback /unload_lora_adapter failed on pod %s for '%s' (%s): %s",
                            loaded_pod,
                            lora_name,
                            type(rb_exc).__name__,
                            rb_exc,
                        )
                raise MultiPodAtomicLoadError(
                    f"Atomic multi-pod LoRA load failed on {pod_v1} for '{lora_name}' "
                    f"(rolled back {len(succeeded_pods)} pod(s), vllm#12174): {exc}"
                ) from exc

    def _unload_across_all_vllm_pods(self, lora_name: str, staged_dir: Path | None = None) -> None:
        headers = self._vllm_headers()
        for pod_v1 in self.vllm_pod_endpoints:
            try:
                resp = self._http_client.post(
                    f"{pod_v1}/unload_lora_adapter",
                    headers=headers,
                    json={"lora_name": lora_name},
                )
                if resp.status_code in {200, 204, 404}:
                    self._dirty_pod_adapters.discard((pod_v1, lora_name))
                else:
                    self._dirty_pod_adapters.add((pod_v1, lora_name))
                    logger.warning(
                        "Pod %s returned HTTP %d on /unload_lora_adapter for '%s': %s",
                        pod_v1,
                        resp.status_code,
                        lora_name,
                        resp.text,
                    )
            except Exception as exc:
                self._dirty_pod_adapters.add((pod_v1, lora_name))
                logger.warning(
                    "Unload request to %s failed for '%s' (%s): %s",
                    pod_v1,
                    lora_name,
                    type(exc).__name__,
                    exc,
                )
        if (
            staged_dir is not None
            and staged_dir.exists()
            and staged_dir.is_relative_to(self.staging_root)
        ):
            shutil.rmtree(staged_dir, ignore_errors=True)

    def _sync_router_registration(
        self,
        artifact: ModelArtifactRecord,
        staged_dir: Path,
    ) -> None:
        if self._adapter_router is None:
            return
        self._adapter_router.register_specialization(
            artifact.domain_id,
            {
                "tenant_id": str(artifact.tenant_id),
                "domain_id": artifact.domain_id,
                "base_model_id": artifact.base_model_id,
                "lora_adapter_id": artifact.artifact_id,
                "lora_adapter_path": str(staged_dir),
                "training_scope_ceiling": list(artifact.training_scope_ceiling),
                "lora_policy": {
                    "enabled": True,
                    "rank": artifact.lora_rank,
                    "lora_alpha": artifact.lora_alpha,
                    "use_rslora": artifact.use_rslora,
                    "use_dora": artifact.use_dora,
                },
            },
            tenant_id=artifact.tenant_id,
        )

    def verify_multi_pod_parity(self) -> dict[str, set[str]]:
        """Queries ``GET /v1/models`` across all configured vLLM pods and asserts adapter parity."""
        pod_models: dict[str, set[str]] = {}
        headers = self._vllm_headers()
        for pod_v1 in self.vllm_pod_endpoints:
            resp = self._http_client.get(f"{pod_v1}/models", headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            ids = {
                str(item.get("id"))
                for item in (payload.get("data") or [])
                if isinstance(item, Mapping) and item.get("id")
            }
            pod_models[pod_v1] = ids

        expected_loaded = {c.artifact_id for c in self._loaded_by_domain.values()}
        pod_lora_sets: dict[str, set[str]] = {
            pod_v1: (model_ids - {self.expected_base_model_id})
            for pod_v1, model_ids in pod_models.items()
        }
        for pod_v1, lora_ids in pod_lora_sets.items():
            missing = expected_loaded - lora_ids
            unexpected = lora_ids - expected_loaded
            if missing or unexpected:
                raise AdapterSyncError(
                    f"vLLM multi-pod LoRA parity drift on {pod_v1} (vllm#12174): "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
        return pod_models

    def reconcile_once(self) -> AdapterSyncReport:
        """Executes one reconciliation pass against Postgres ``model_artifacts`` & GCS/S3."""
        with self._lock:
            active_records = self._fetch_active_artifacts_from_db()
            desired_by_domain = {rec.domain_id: rec for rec in active_records}

            newly_loaded: list[str] = []
            unloaded_retired: list[str] = []
            failures: dict[str, str] = {}

            # 1. Load new or rotated cartridges FIRST (blue/green zero-downtime rotation)
            for domain_id, artifact in sorted(desired_by_domain.items()):
                current = self._loaded_by_domain.get(domain_id)
                if (
                    current is not None
                    and current.artifact_id == artifact.artifact_id
                    and current.weights_sha256 == artifact.weights_sha256.strip().lower()
                ):
                    continue

                try:
                    staged_dir = self._verify_and_stage_artifact(artifact)
                    self._load_across_all_vllm_pods(artifact.artifact_id, staged_dir)
                except Exception as exc:
                    logger.error(
                        "Rejected or failed loading model_artifact '%s' for domain '%s': %s",
                        artifact.artifact_id,
                        domain_id,
                        exc,
                    )
                    failures[artifact.artifact_id] = str(exc)
                    continue

                # Unload superseded previous version ONLY after new cartridge succeeded on ALL pods
                if current is not None and current.artifact_id != artifact.artifact_id:
                    self._unload_across_all_vllm_pods(current.artifact_id, current.staged_dir)
                    unloaded_retired.append(current.artifact_id)

                loaded_entry = LoadedPodCartridge(
                    tenant_id=artifact.tenant_id,
                    domain_id=domain_id,
                    artifact_id=artifact.artifact_id,
                    weights_sha256=artifact.weights_sha256.strip().lower(),
                    base_model_revision_sha256=artifact.base_model_revision_sha256.strip().lower(),
                    staged_dir=staged_dir,
                    pod_endpoints=self.vllm_pod_endpoints,
                    training_scope_ceiling=tuple(artifact.training_scope_ceiling),
                    lora_rank=artifact.lora_rank,
                )
                self._loaded_by_domain[domain_id] = loaded_entry
                self._sync_router_registration(artifact, staged_dir)
                newly_loaded.append(artifact.artifact_id)

            # 2. Unload cartridges for domains whose active artifact was retired or revoked
            retired_domains = sorted(set(self._loaded_by_domain) - set(desired_by_domain))
            for domain_id in retired_domains:
                retired = self._loaded_by_domain.pop(domain_id)
                self._unload_across_all_vllm_pods(retired.artifact_id, retired.staged_dir)
                unloaded_retired.append(retired.artifact_id)
                if self._adapter_router is not None:
                    self._adapter_router.register_specialization(
                        domain_id,
                        {
                            "tenant_id": str(self.tenant_id),
                            "domain_id": domain_id,
                            "base_model_id": self.expected_base_model_id,
                            "training_scope_ceiling": list(retired.training_scope_ceiling),
                            "lora_policy": {
                                "enabled": False,
                                "rank": retired.lora_rank,
                                "use_rslora": True,
                            },
                        },
                        tenant_id=self.tenant_id,
                    )

            # 3. Sweep any dirty rollback/unload adapters that are not currently active
            currently_active_ids = {c.artifact_id for c in self._loaded_by_domain.values()}
            headers = self._vllm_headers()
            for dirty_pod, dirty_lora in list(self._dirty_pod_adapters):
                if dirty_lora in currently_active_ids:
                    self._dirty_pod_adapters.discard((dirty_pod, dirty_lora))
                    continue
                try:
                    sweep_resp = self._http_client.post(
                        f"{dirty_pod}/unload_lora_adapter",
                        headers=headers,
                        json={"lora_name": dirty_lora},
                    )
                    if sweep_resp.status_code in {200, 204, 404}:
                        self._dirty_pod_adapters.discard((dirty_pod, dirty_lora))
                except Exception as sweep_exc:
                    logger.debug(
                        "Dirty adapter sweep deferred on %s for '%s' (%s)",
                        dirty_pod,
                        dirty_lora,
                        type(sweep_exc).__name__,
                    )

            return AdapterSyncReport(
                tenant_id=self.tenant_id,
                active_in_db=tuple(sorted(r.artifact_id for r in active_records)),
                loaded_artifacts=tuple(
                    sorted(c.artifact_id for c in self._loaded_by_domain.values())
                ),
                newly_loaded=tuple(newly_loaded),
                unloaded_retired=tuple(unloaded_retired),
                failed_artifacts=failures,
                pod_count=len(self.vllm_pod_endpoints),
            )

    def start(self) -> None:
        """Starts the background sidecar polling loop."""
        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stop_event.clear()

            def _loop() -> None:
                while not self._stop_event.is_set():
                    try:
                        self.reconcile_once()
                    except Exception as exc:
                        logger.error(
                            "AdapterSyncSidecar background cycle error (%s): %s",
                            type(exc).__name__,
                            exc,
                        )
                    self._stop_event.wait(self.poll_interval_seconds)

            self._worker_thread = threading.Thread(
                target=_loop,
                name=f"adapter-sync-sidecar-{str(self.tenant_id)[:8]}",
                daemon=True,
            )
            self._worker_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stops the background sidecar polling loop cleanly."""
        self._stop_event.set()
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._worker_thread = None


def parse_vllm_endpoints_from_env(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Resolves vLLM pod endpoints from SCOUT_VLLM_ENDPOINTS or SCOUT_VLLM_ADMIN_ENDPOINT."""
    env = os.environ if environ is None else environ
    raw_multi = str(env.get("SCOUT_VLLM_ENDPOINTS") or "").strip()
    if raw_multi:
        parts = tuple(ep.strip() for ep in raw_multi.split(",") if ep.strip())
        if parts:
            return parts
    raw_single = str(env.get("SCOUT_VLLM_ADMIN_ENDPOINT") or "").strip()
    if raw_single:
        return (raw_single,)
    return (DEFAULT_VLLM_ADMIN_ENDPOINT,)


def build_sidecar_from_env(environ: Mapping[str, str] | None = None) -> AdapterSyncSidecar:
    """Constructs a fail-closed AdapterSyncSidecar configured from environment variables."""
    env = os.environ if environ is None else environ

    db_app_url = str(env.get("SCOUT_DB_APP_URL") or "").strip()
    if not db_app_url:
        raise AdapterVerificationError(
            "SCOUT_DB_APP_URL must be set for AdapterSyncSidecar (NOBYPASSRLS runtime pool)."
        )

    expected_base_model_id = (
        str(env.get("SCOUT_PRIMARY_MODEL_ID") or "google/gemma-4-26B-A4B-it").strip()
        or "google/gemma-4-26B-A4B-it"
    )
    expected_rev_sha = str(env.get("SCOUT_BASE_MODEL_REVISION_SHA256") or "").strip()
    if not expected_rev_sha:
        raise AdapterVerificationError(
            "SCOUT_BASE_MODEL_REVISION_SHA256 must be set to a 64-hex SHA-256 digest."
        )

    trusted_keys: dict[str, str | bytes] = {}
    raw_keys_json = str(env.get("SCOUT_COSIGN_PUBLIC_KEYS_JSON") or "").strip()
    if raw_keys_json:
        parsed_keys = json.loads(raw_keys_json)
        if not isinstance(parsed_keys, Mapping):
            raise AdapterVerificationError("SCOUT_COSIGN_PUBLIC_KEYS_JSON must be a JSON object.")
        for key_id, pem_val in parsed_keys.items():
            if str(pem_val).strip():
                trusted_keys[str(key_id)] = str(pem_val)
    raw_single_pem = str(env.get("SCOUT_COSIGN_PUBLIC_KEY_PEM") or "").strip()
    if raw_single_pem and "default" not in trusted_keys:
        trusted_keys["default"] = raw_single_pem

    raw_ca_pem = str(env.get("SCOUT_COSIGN_CA_CERTS_PEM") or "").strip()
    ca_certs = [raw_ca_pem] if raw_ca_pem else []
    cosign_verifier = CosignBlobVerifier(
        trusted_public_keys_pem=trusted_keys,
        trusted_ca_certs_pem=ca_certs,
    )
    if not cosign_verifier.has_trusted_keys:
        raise AdapterVerificationError(
            "No trusted cosign public keys or CA certs configured "
            "(SCOUT_COSIGN_PUBLIC_KEYS_JSON / SCOUT_COSIGN_PUBLIC_KEY_PEM / "
            "SCOUT_COSIGN_CA_CERTS_PEM); failing closed."
        )

    raw_tenant = str(env.get("SCOUT_TENANT_ID") or "").strip()
    tenant_id = UUID(raw_tenant) if raw_tenant else DEFAULT_TENANT_ID
    staging_dir = Path(
        str(env.get("SCOUT_ADAPTER_STAGING_DIR") or "/var/run/scout/adapters").strip()
        or "/var/run/scout/adapters"
    )
    poll_interval = float(str(env.get("SCOUT_ADAPTER_POLL_INTERVAL_SECONDS") or "15.0").strip())

    rls_mgr = RLSConnectionManager(db_app_url, use_pool=True)
    return AdapterSyncSidecar(
        expected_base_model_id=expected_base_model_id,
        expected_base_model_revision_sha256=expected_rev_sha,
        staging_root=staging_dir,
        cosign_verifier=cosign_verifier,
        tenant_id=tenant_id,
        rls_conn_manager=rls_mgr,
        gcs_project=str(env.get("SCOUT_GCS_PROJECT") or "").strip() or None,
        gcs_emulator_host=str(env.get("STORAGE_EMULATOR_HOST") or "").strip() or None,
        s3_endpoint_url=str(env.get("SCOUT_S3_ENDPOINT_URL") or "").strip() or None,
        s3_region=str(env.get("SCOUT_S3_REGION") or "eu-central-1").strip() or "eu-central-1",
        vllm_pod_endpoints=parse_vllm_endpoints_from_env(env),
        vllm_admin_api_key=str(env.get("SCOUT_VLLM_ADMIN_API_KEY") or "").strip() or None,
        poll_interval_seconds=poll_interval,
        owns_rls_conn_manager=True,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the vLLM multi-LoRA AdapterSyncSidecar container."""
    parser = argparse.ArgumentParser(
        prog="adapter-sync-sidecar",
        description="Sync verified LoRA cartridges from PG17 + GCS/S3 to vLLM pods.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single reconciliation pass and exit.",
    )
    args = parser.parse_args(argv)

    try:
        sidecar = build_sidecar_from_env()
    except Exception as exc:
        logger.error("AdapterSyncSidecar configuration failed closed: %s", exc)
        return 2

    if args.once:
        try:
            report = sidecar.reconcile_once()
            print(
                json.dumps(
                    {
                        "tenant_id": str(report.tenant_id),
                        "active_in_db": list(report.active_in_db),
                        "loaded_artifacts": list(report.loaded_artifacts),
                        "newly_loaded": list(report.newly_loaded),
                        "unloaded_retired": list(report.unloaded_retired),
                        "failed_artifacts": dict(report.failed_artifacts),
                        "pod_count": report.pod_count,
                    },
                    indent=2,
                )
            )
            return 0 if not report.failed_artifacts else 1
        finally:
            sidecar.close()

    stop_event = threading.Event()

    def _handle_sig(signum: int, _frame: Any) -> None:
        logger.info("AdapterSyncSidecar received signal %s; shutting down.", signum)
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, _handle_sig)
        signal.signal(signal.SIGTERM, _handle_sig)
    except (ValueError, AttributeError):
        pass

    sidecar.start()
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        sidecar.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
