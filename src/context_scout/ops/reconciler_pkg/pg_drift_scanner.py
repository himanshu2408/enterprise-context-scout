"""PG17 NOBYPASSRLS checks, object store resolution, and commit watermark drift scanner."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from context_scout.config import ScoutSettings
from context_scout.db.schema import (
    CodeFileRecord,
    RepositoryRecord,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.ops.reconciler_pkg.base import _StateReconcilerBase
from context_scout.ops.reconciler_pkg.models import (
    _ALLOWED_SCHEMES,
    _DOMAIN_SLUG_RE,
    _FORBIDDEN_ADMIN_DB_ROLES,
    StateReconcilerError,
)
from context_scout.storage.object_store import (
    ArtifactObjectStore,
    GCSObjectStore,
    InMemoryObjectStore,
    S3CompatibleObjectStore,
    validate_object_key,
)

logger = logging.getLogger(__name__)


class _PgDriftScannerMixin(_StateReconcilerBase):
    """PG17 NOBYPASSRLS checks, object store resolution, and commit watermark drift scanner."""

    @staticmethod
    def _validate_non_admin_rls_manager(
        mgr: RLSConnectionManager,
        settings: ScoutSettings | None = None,
    ) -> None:
        raw_dsn = str(
            getattr(mgr, "_app_db_url", "")
            or getattr(mgr, "_conninfo", "")
            or getattr(mgr, "db_url", "")
            or ""
        ).strip()
        if not raw_dsn:
            return
        parsed = urlparse(raw_dsn)
        user = (parsed.username or "").strip().lower()
        if user in _FORBIDDEN_ADMIN_DB_ROLES or user.startswith("postgres"):
            raise PermissionError(
                f"StateReconciler MUST NOT connect as privileged DB role '{user}'; "
                "use RLS runtime pool ('scout_app_user' with NOBYPASSRLS)."
            )
        if settings is not None:
            admin_parsed = urlparse(settings.db_admin_url.strip())
            if raw_dsn == settings.db_admin_url.strip() or (
                parsed.hostname == admin_parsed.hostname
                and parsed.port == admin_parsed.port
                and parsed.path == admin_parsed.path
                and user
                and user == (admin_parsed.username or "").strip().lower()
            ):
                raise PermissionError(
                    "StateReconciler MUST NOT use db_admin_url; use RLS runtime pool (NOBYPASSRLS)."
                )

    @staticmethod
    def _assert_pg_session_non_bypass_rls(conn: Any) -> None:
        row = conn.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user;"
        ).fetchone()
        if row is not None and (bool(row[0]) or bool(row[1])):
            raise PermissionError(
                "StateReconciler refuses execution on PostgreSQL role with SUPERUSER or BYPASSRLS."
            )

    @staticmethod
    def _validate_domain_id(domain_id: str) -> str:
        cleaned = domain_id.strip()
        if not cleaned or not _DOMAIN_SLUG_RE.fullmatch(cleaned):
            raise ValueError(
                f"Invalid domain_id {domain_id!r}; "
                f"expected lowercase slug {_DOMAIN_SLUG_RE.pattern}."
            )
        return cleaned

    @staticmethod
    def _infer_store_scheme(store: ArtifactObjectStore) -> str:
        if isinstance(store, GCSObjectStore):
            return "gs"
        if isinstance(store, S3CompatibleObjectStore):
            return "s3"
        if isinstance(store, InMemoryObjectStore):
            return "memory"
        cls_name = type(store).__name__.lower()
        if "gcs" in cls_name:
            return "gs"
        if "s3" in cls_name:
            return "s3"
        return "memory"

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
            raise StateReconcilerError(
                f"Storage bucket mismatch for {storage_uri!r}: store configured for "
                f"'{configured_bucket}', URI requested '{expected_bucket}'."
            )
        return store

    def _resolve_object_store_for_uri(self, storage_uri: str) -> ArtifactObjectStore:
        parsed = urlparse(storage_uri.strip())
        scheme = parsed.scheme.lower()
        bucket = parsed.netloc.strip()
        if scheme not in _ALLOWED_SCHEMES:
            raise StateReconcilerError(
                f"Unsupported storage_uri scheme '{scheme}://' for {storage_uri!r}; "
                "expected 'gs://' (GCSObjectStore) or 's3://' (S3CompatibleObjectStore)."
            )
        if not bucket or "/" in bucket or "\\" in bucket or bucket in {".", ".."}:
            raise StateReconcilerError(f"Missing or invalid bucket in storage_uri: {storage_uri!r}")

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
                project=self._settings.gcs_project or self._settings.gcp_project,
                emulator_host=self._settings.gcs_emulator_host,
            )
            self._dynamic_bucket_stores[cache_key] = gcs_store
            return gcs_store

        if scheme == "s3":
            s3_store = S3CompatibleObjectStore(
                bucket=bucket,
                endpoint_url=self._settings.s3_endpoint_url,
                region_name=self._settings.s3_region,
                aws_access_key_id=self._settings.s3_access_key_id,
                aws_secret_access_key=self._settings.s3_secret_access_key,
                force_path_style=self._settings.s3_force_path_style,
            )
            self._dynamic_bucket_stores[cache_key] = s3_store
            return s3_store

        raise StateReconcilerError(
            f"Unconfigured storage_uri scheme '{scheme}://' for {storage_uri!r}."
        )

    def _target_stores(self) -> list[tuple[str, ArtifactObjectStore]]:
        stores: list[tuple[str, ArtifactObjectStore]] = []
        seen_ids: set[int] = set()
        for scheme, store in sorted(self._stores_by_scheme.items()):
            if id(store) not in seen_ids:
                stores.append((scheme, store))
                seen_ids.add(id(store))
        if (
            self._default_object_store is not None
            and id(self._default_object_store) not in seen_ids
        ):
            stores.append(
                (self._infer_store_scheme(self._default_object_store), self._default_object_store)
            )
            seen_ids.add(id(self._default_object_store))
        return stores

    @staticmethod
    def _extract_key_from_uri(uri: str, expected_tenant_id: UUID) -> str | None:
        parsed = urlparse(uri.strip())
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            return None
        rel = parsed.path.lstrip("/")
        if not rel:
            return None
        return validate_object_key(rel, expected_tenant_id=expected_tenant_id)

    def _read_external_manifest_payload(
        self,
        store: ArtifactObjectStore,
        key: str,
        *,
        tenant_id: UUID,
        warnings: list[str],
        label: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            if not store.exists(key, expected_tenant_id=tenant_id):
                return None, None
            raw = store.get_bytes(key, expected_tenant_id=tenant_id)
        except Exception as exc:
            warnings.append(f"Failed reading {label} manifest '{key}': {type(exc).__name__}: {exc}")
            return None, None

        digest = hashlib.sha256(raw).hexdigest()
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            warnings.append(f"Corrupted JSON in {label} manifest '{key}': {type(exc).__name__}")
            return None, digest
        if not isinstance(parsed, dict):
            warnings.append(f"Unexpected root type in {label} manifest '{key}'.")
            return None, digest
        return parsed, digest

    def _compute_authoritative_pg_commits(
        self,
        *,
        repos: Sequence[RepositoryRecord],
        code_files: Sequence[CodeFileRecord],
        repo_commits_map: Mapping[UUID, Sequence[Any]],
    ) -> dict[str, str]:
        files_by_repo: dict[str, list[CodeFileRecord]] = {}
        for cf in code_files:
            files_by_repo.setdefault(cf.repo_name, []).append(cf)

        pg_repo_shas: dict[str, str] = {}
        for r in repos:
            if not r.is_active:
                continue
            commits = list(repo_commits_map.get(r.repo_id) or [])
            commit_shas_known = {str(c.commit_sha) for c in commits if getattr(c, "commit_sha", "")}
            r_files = files_by_repo.get(r.repo_name, [])
            file_commit_shas = [cf.git_commit_sha for cf in r_files if cf.git_commit_sha]

            resolved_sha = r.head_commit_sha.strip()
            if commits:
                latest_commit_sha = str(commits[0].commit_sha).strip()
                if not resolved_sha or (
                    resolved_sha not in commit_shas_known and resolved_sha not in file_commit_shas
                ):
                    resolved_sha = latest_commit_sha
            elif file_commit_shas:
                if not resolved_sha or resolved_sha not in file_commit_shas:
                    resolved_sha = file_commit_shas[0]
            pg_repo_shas[r.repo_name] = resolved_sha
        return pg_repo_shas
