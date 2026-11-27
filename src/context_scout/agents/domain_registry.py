"""Declarative DomainPack registry (PG17 + GCS/S3 object store) and LoRA AdapterPool."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml

from context_scout.db.repositories import PostgresRLSRepository
from context_scout.db.session import (
    PoolRole,
    RLSConnectionManager,
    get_ingestion_pool,
    get_query_pool,
)
from context_scout.schemas.contracts import (
    DEFAULT_TENANT_ID,
    DomainPackManifest,
    PrincipalContext,
    RoleType,
)
from context_scout.storage.object_store import (
    ArtifactObjectStore,
    ObjectMetadata,
    build_tenant_domain_key,
    get_artifact_object_store,
)

logger = logging.getLogger(__name__)


def audit_safetensors_adapter_weights(safetensors_path: Path) -> tuple[bool, str | None]:
    """Validates a .safetensors adapter: non-corrupt, zero router/SSM keys, non-zero lora_b."""
    if not safetensors_path.is_file() or safetensors_path.stat().st_size < 8:
        return False, f"Corrupt or empty safetensors weights at {safetensors_path}"
    try:
        from safetensors.numpy import load_file as load_safetensors_numpy

        tensors = load_safetensors_numpy(str(safetensors_path))
    except Exception as exc:
        return False, f"Corrupt safetensors weights at {safetensors_path}: {exc}"

    if not tensors:
        return False, f"No tensors found in safetensors weights at {safetensors_path}"

    for key in tensors:
        key_l = key.lower()
        if "router" in key_l or "ssm" in key_l:
            return False, (f"Prohibited frozen router/SSM tensor key '{key}' in {safetensors_path}")

    lora_b_arrays = [arr for key, arr in tensors.items() if "lora_b" in key.lower()]
    if lora_b_arrays and not any(float((arr * arr).sum()) > 0.0 for arr in lora_b_arrays):
        return False, (
            f"Zero-delta lora_b weights (all norms == 0.0) rejected at {safetensors_path}"
        )
    return True, None


@dataclass(frozen=True)
class AdapterSlot:
    """Represents a pre-warmed in-memory QLoRA adapter slot."""

    domain_id: str
    adapter_id: str
    adapter_path: str
    base_model_id: str
    max_kv_tokens: int = 32_768
    is_verified: bool = False


class AdapterPool:
    """Thread-safe in-memory registry pinning active QLoRA adapters (ARC-02 & HW-01)."""

    def __init__(
        self,
        max_slots: int = 16,
        max_kv_tokens: int = 32_768,
        *,
        object_store: ArtifactObjectStore | None = None,
    ) -> None:
        if max_kv_tokens > 32_768:
            raise ValueError(
                f"max_kv_tokens ({max_kv_tokens}) exceeds Apple M5 Pro 48GB safety cap (32768)"
            )
        self._max_slots = max_slots
        self._max_kv_tokens = max_kv_tokens
        self._object_store = object_store
        self._slots: dict[str, AdapterSlot] = {}
        self._lock = threading.Lock()

    def register_from_manifest(self, manifest: DomainPackManifest) -> AdapterSlot:
        with self._lock:
            if manifest.domain_id not in self._slots and len(self._slots) >= self._max_slots:
                raise RuntimeError(
                    f"AdapterPool capacity ({self._max_slots}) reached for '{manifest.domain_id}'"
                )
            slot = AdapterSlot(
                domain_id=manifest.domain_id,
                adapter_id=manifest.lora_adapter_id,
                adapter_path=manifest.lora_adapter_path,
                base_model_id=manifest.base_model_id,
                max_kv_tokens=self._max_kv_tokens,
            )
            self._slots[manifest.domain_id] = slot
            return slot

    def get_slot(self, domain_id: str) -> AdapterSlot:
        with self._lock:
            if domain_id not in self._slots:
                raise KeyError(f"Domain '{domain_id}' is not registered in AdapterPool")
            return self._slots[domain_id]

    def unregister_slot(self, domain_id: str) -> None:
        with self._lock:
            self._slots.pop(domain_id, None)

    def list_slots(self) -> list[AdapterSlot]:
        with self._lock:
            return list(self._slots.values())

    def _verify_adapter_from_object_store(
        self,
        slot: AdapterSlot,
        store: ArtifactObjectStore,
        tenant_id: UUID,
    ) -> dict[str, Any] | None:
        domain_id = slot.domain_id
        try:
            weights_key = build_tenant_domain_key(
                tenant_id, domain_id, f"adapters/{slot.adapter_id}/adapters.safetensors"
            )
            alt_weights_key = build_tenant_domain_key(
                tenant_id, domain_id, f"adapters/{slot.adapter_id}/adapter_model.safetensors"
            )
            cfg_key = build_tenant_domain_key(
                tenant_id, domain_id, f"adapters/{slot.adapter_id}/adapter_config.json"
            )
            metrics_key = build_tenant_domain_key(
                tenant_id, domain_id, f"adapters/{slot.adapter_id}/training_metrics.json"
            )
        except ValueError:
            return None

        if not store.exists(weights_key, expected_tenant_id=tenant_id):
            if store.exists(alt_weights_key, expected_tenant_id=tenant_id):
                weights_key = alt_weights_key
            else:
                return None
        if not store.exists(cfg_key, expected_tenant_id=tenant_id):
            return {
                "domain_id": domain_id,
                "is_valid": False,
                "adapter_path": cfg_key,
                "error_message": f"Missing adapter_config.json in object store: {cfg_key}",
            }

        weights_bytes = store.get_bytes(weights_key, expected_tenant_id=tenant_id)
        if not weights_bytes:
            return {
                "domain_id": domain_id,
                "is_valid": False,
                "adapter_path": weights_key,
                "error_message": f"Empty adapters.safetensors in object store: {weights_key}",
            }

        try:
            cfg = json.loads(store.get_bytes(cfg_key, expected_tenant_id=tenant_id).decode("utf-8"))
            r = int(cfg.get("r", 0))
            lora_alpha = int(cfg.get("lora_alpha", 0))
            is_moe = bool(cfg.get("is_moe", False))
        except Exception as exc:
            return {
                "domain_id": domain_id,
                "is_valid": False,
                "adapter_path": cfg_key,
                "error_message": f"Corrupt adapter_config.json in object store: {exc}",
            }

        final_loss = None
        if store.exists(metrics_key, expected_tenant_id=tenant_id):
            try:
                metrics = json.loads(
                    store.get_bytes(metrics_key, expected_tenant_id=tenant_id).decode("utf-8")
                )
                final_loss = metrics.get("final_loss")
            except Exception as exc:
                logger.debug(
                    "Failed reading training_metrics.json from object store (%s): %s",
                    type(exc).__name__,
                    exc,
                )

        return {
            "domain_id": domain_id,
            "is_valid": True,
            "adapter_path": weights_key,
            "safetensors_bytes": len(weights_bytes),
            "r": r,
            "lora_alpha": lora_alpha,
            "is_moe": is_moe,
            "final_loss": final_loss,
            "error_message": None,
        }

    def _read_adapter_config_and_metrics(
        self,
        adapter_dir: Path,
    ) -> tuple[dict[str, Any] | None, float | None, str | None]:
        config_path = adapter_dir / "adapter_config.json"
        metrics_path = adapter_dir / "training_metrics.json"
        cfg: dict[str, Any] | None = None
        if config_path.exists():
            if config_path.stat().st_size == 0:
                return None, None, f"Empty adapter_config.json at {config_path}"
            try:
                with open(config_path, encoding="utf-8") as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict):
                    return None, None, f"Invalid adapter_config.json object at {config_path}"
                if loaded.get("trust_remote_code") is True or "auto_map" in loaded:
                    return None, None, f"Prohibited trust_remote_code/auto_map in {config_path}"
                cfg_rank = loaded.get("r")
                if cfg_rank is not None and int(cfg_rank) > 32:
                    return None, None, f"Adapter rank r={cfg_rank} > 32 in {config_path}"
                cfg = loaded
            except Exception as exc:
                return None, None, f"Corrupt adapter_config.json: {exc}"

        final_loss: float | None = None
        if metrics_path.exists():
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    metrics = json.load(f)
                raw_loss = metrics.get("final_loss") if isinstance(metrics, dict) else None
                final_loss = float(raw_loss) if raw_loss is not None else None
            except Exception as exc:
                logger.debug(
                    "Failed reading training_metrics.json (%s): %s",
                    type(exc).__name__,
                    exc,
                )
        return cfg, final_loss, None

    def verify_adapter(
        self,
        domain_id: str,
        base_dir: Path | str | None = None,
        *,
        object_store: ArtifactObjectStore | None = None,
        tenant_id: UUID = DEFAULT_TENANT_ID,
    ) -> dict[str, Any]:
        """Verifies adapter artifacts on read-only local bundle or in S3/MinIO object store."""
        slot = self.get_slot(domain_id)
        root = Path(base_dir) if base_dir else Path.cwd()
        adapter_dir = (root / slot.adapter_path).resolve()
        display_path = str(adapter_dir)
        try:
            cwd = Path.cwd().resolve()
            if adapter_dir.resolve().is_relative_to(cwd):
                display_path = str(adapter_dir.resolve().relative_to(cwd))
        except OSError as exc:
            logger.debug(
                "Could not compute relative adapter path (%s): %s",
                type(exc).__name__,
                exc,
            )

        if not adapter_dir.exists():
            direct_path = Path(slot.adapter_path).resolve()
            if direct_path.exists():
                adapter_dir = direct_path
                try:
                    cwd = Path.cwd().resolve()
                    if adapter_dir.resolve().is_relative_to(cwd):
                        display_path = str(adapter_dir.resolve().relative_to(cwd))
                except OSError as exc:
                    logger.debug(
                        "Could not compute relative direct adapter path (%s): %s",
                        type(exc).__name__,
                        exc,
                    )
            else:
                active_store = object_store or self._object_store
                if active_store is not None:
                    remote_res = self._verify_adapter_from_object_store(
                        slot, active_store, tenant_id
                    )
                    if remote_res is not None:
                        return remote_res
                return {
                    "domain_id": domain_id,
                    "is_valid": True,
                    "adapter_present": False,
                    "adapters_present": False,
                    "serving_mode": "base_open_weights",
                    "adapter_state": "base_open_weights",
                    "weight_format": "base_open_weights",
                    "safetensors_bytes": 0,
                    "adapter_path": display_path,
                    "error_message": None,
                }

        safetensors_path = adapter_dir / "adapters.safetensors"
        if not safetensors_path.exists():
            alt_safetensors = adapter_dir / "adapter_model.safetensors"
            if alt_safetensors.exists():
                safetensors_path = alt_safetensors

        cfg, final_loss, cfg_err = self._read_adapter_config_and_metrics(adapter_dir)
        if not safetensors_path.exists():
            if cfg_err is not None:
                return {
                    "domain_id": domain_id,
                    "is_valid": False,
                    "adapter_present": False,
                    "adapters_present": False,
                    "serving_mode": "base_open_weights",
                    "adapter_state": "corrupt",
                    "weight_format": "base_open_weights",
                    "safetensors_bytes": 0,
                    "adapter_path": display_path,
                    "error_message": cfg_err,
                }
            return {
                "domain_id": domain_id,
                "is_valid": True,
                "adapter_present": False,
                "adapters_present": False,
                "serving_mode": "base_open_weights",
                "adapter_state": "base_open_weights",
                "weight_format": "base_open_weights",
                "safetensors_bytes": 0,
                "r": int(cfg.get("r", 0)) if cfg else None,
                "lora_alpha": int(cfg.get("lora_alpha", 0)) if cfg else None,
                "is_moe": bool(cfg.get("is_moe", False)) if cfg else None,
                "final_loss": final_loss,
                "adapter_path": display_path,
                "error_message": None,
            }

        from context_scout.agents.llm_backend import detect_adapter_weight_format

        weight_format = detect_adapter_weight_format(safetensors_path)
        if safetensors_path.stat().st_size == 0 or weight_format == "corrupt":
            return {
                "domain_id": domain_id,
                "is_valid": False,
                "adapter_present": True,
                "adapters_present": True,
                "serving_mode": "qlora_adapter",
                "adapter_state": "corrupt",
                "weight_format": "corrupt",
                "safetensors_bytes": safetensors_path.stat().st_size,
                "adapter_path": display_path,
                "error_message": f"Corrupt or empty safetensors weights at {safetensors_path}",
            }

        weights_ok, weights_err = audit_safetensors_adapter_weights(safetensors_path)
        if not weights_ok:
            return {
                "domain_id": domain_id,
                "is_valid": False,
                "adapter_present": True,
                "adapters_present": True,
                "serving_mode": "qlora_adapter",
                "adapter_state": "corrupt",
                "weight_format": "corrupt",
                "safetensors_bytes": safetensors_path.stat().st_size,
                "adapter_path": display_path,
                "error_message": weights_err,
            }

        if cfg is None:
            return {
                "domain_id": domain_id,
                "is_valid": False,
                "adapter_present": True,
                "adapters_present": True,
                "serving_mode": "qlora_adapter",
                "adapter_state": "corrupt",
                "weight_format": weight_format,
                "safetensors_bytes": safetensors_path.stat().st_size,
                "adapter_path": display_path,
                "error_message": cfg_err
                or f"Missing or empty adapter_config.json at {adapter_dir / 'adapter_config.json'}",
            }

        return {
            "domain_id": domain_id,
            "is_valid": True,
            "adapter_present": True,
            "adapters_present": True,
            "serving_mode": "qlora_adapter",
            "adapter_state": "ready",
            "weight_format": weight_format,
            "adapter_path": display_path,
            "safetensors_bytes": safetensors_path.stat().st_size,
            "r": int(cfg.get("r", 0)),
            "lora_alpha": int(cfg.get("lora_alpha", 0)),
            "is_moe": bool(cfg.get("is_moe", False)),
            "final_loss": final_loss,
            "error_message": None,
        }

    def verify_all_adapters(
        self,
        base_dir: Path | str | None = None,
        *,
        object_store: ArtifactObjectStore | None = None,
        tenant_id: UUID = DEFAULT_TENANT_ID,
    ) -> dict[str, dict[str, Any]]:
        """Verifies artifacts for all registered adapters in pool."""
        results: dict[str, dict[str, Any]] = {}
        for slot in self.list_slots():
            results[slot.domain_id] = self.verify_adapter(
                slot.domain_id,
                base_dir=base_dir,
                object_store=object_store,
                tenant_id=tenant_id,
            )
        return results


class DomainRegistry:
    """Stateless PG17 + S3/MinIO DomainPack registry with read-only seed template loading."""

    def __init__(
        self,
        domains_dir: Path | None = None,
        adapter_pool: AdapterPool | None = None,
        *,
        connection_manager: RLSConnectionManager | None = None,
        ingestion_manager: RLSConnectionManager | None = None,
        object_store: ArtifactObjectStore | None = None,
        tenant_id: UUID = DEFAULT_TENANT_ID,
        enable_pg: bool = True,
    ) -> None:
        self._domains_dir = domains_dir
        self._object_store = object_store or get_artifact_object_store()
        self._adapter_pool = adapter_pool or AdapterPool(object_store=self._object_store)
        self._connection_manager = connection_manager
        self._ingestion_manager = ingestion_manager
        self._tenant_id = tenant_id
        self._enable_pg = enable_pg
        self._manifests: dict[str, DomainPackManifest] = {}
        self._tenant_manifests: dict[tuple[UUID, str], DomainPackManifest] = {}

    @property
    def domains_dir(self) -> Path | None:
        return self._domains_dir

    @property
    def adapter_pool(self) -> AdapterPool:
        return self._adapter_pool

    @property
    def object_store(self) -> ArtifactObjectStore:
        return self._object_store

    def _effective_principal(self, principal: PrincipalContext | None) -> PrincipalContext:
        if principal is not None:
            return principal
        return PrincipalContext(
            user_id="domain_registry_service",
            role=RoleType.ENGINEER,
            allowed_scopes=["public"],
            tenant_id=self._tenant_id,
        )

    def _query_mgr(self) -> RLSConnectionManager:
        if self._connection_manager is not None:
            return self._connection_manager
        return get_query_pool()

    def _ingest_mgr(self) -> RLSConnectionManager:
        if self._ingestion_manager is not None:
            return self._ingestion_manager
        if self._connection_manager is not None:
            return self._connection_manager
        return get_ingestion_pool()

    def _load_pg_manifest_entries(
        self, principal: PrincipalContext
    ) -> list[tuple[DomainPackManifest, bool]]:
        if not self._enable_pg:
            return []
        try:
            pg_repo = PostgresRLSRepository()
            if self._connection_manager is not None:
                mgr = self._connection_manager
            else:
                # Use a short-timeout non-pooled probe so startup/import does not block
                # for ConnectionPool's 30s retry window when PG17 is temporarily unreachable.
                from context_scout.config import ScoutSettings

                cfg = ScoutSettings()
                mgr = RLSConnectionManager(
                    cfg.db_app_url,
                    use_pool=False,
                    pool_role=PoolRole.QUERY,
                    statement_timeout_ms=cfg.query_statement_timeout_ms,
                    lock_timeout_ms=cfg.query_lock_timeout_ms,
                    idle_tx_timeout_ms=cfg.query_idle_tx_timeout_ms,
                    connect_timeout_s=2,
                )
            with mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
                return pg_repo.list_domain_pack_entries(conn)
        except Exception as exc:
            logger.debug(
                "PG17 domain_packs load skipped (%s): %s",
                type(exc).__name__,
                exc,
            )
            return []

    def _load_pg_manifests(self, principal: PrincipalContext) -> list[DomainPackManifest]:
        return [manifest for manifest, _ in self._load_pg_manifest_entries(principal)]

    def load_all(self, principal: PrincipalContext | None = None) -> dict[str, DomainPackManifest]:
        effective_principal = self._effective_principal(principal)
        disk_manifests: dict[str, DomainPackManifest] = {}
        has_disk_dir = self._domains_dir is not None and self._domains_dir.exists()

        if has_disk_dir and self._domains_dir is not None:
            for yaml_path in sorted(self._domains_dir.glob("*/domain.yaml")):
                with open(yaml_path, encoding="utf-8") as f:
                    raw_data = yaml.safe_load(f)
                manifest = DomainPackManifest.model_validate(raw_data)
                disk_manifests[manifest.domain_id] = manifest
                self.register(manifest, tenant_id=effective_principal.tenant_id)

        pg_entries = self._load_pg_manifest_entries(effective_principal)
        for manifest, is_persistent in pg_entries:
            # Never let non-S3 legacy DB seed config override authoritative domain.yaml templates
            if manifest.domain_id in disk_manifests and not is_persistent:
                continue
            self.register(manifest, tenant_id=effective_principal.tenant_id)

        if self._domains_dir is not None and not has_disk_dir and not pg_entries:
            raise FileNotFoundError(f"Domains directory not found: {self._domains_dir}")

        if effective_principal.tenant_id == self._tenant_id:
            return dict(self._manifests)
        return {
            d_id: m
            for (tid, d_id), m in self._tenant_manifests.items()
            if tid == effective_principal.tenant_id
        }

    def register(
        self,
        manifest: DomainPackManifest,
        *,
        tenant_id: UUID | None = None,
    ) -> None:
        resolved_tid = tenant_id or self._tenant_id
        self._tenant_manifests[(resolved_tid, manifest.domain_id)] = manifest
        if resolved_tid == self._tenant_id:
            self._manifests[manifest.domain_id] = manifest
            self._adapter_pool.register_from_manifest(manifest)
        elif manifest.domain_id not in self._manifests and (
            manifest.domain_id not in {s.domain_id for s in self._adapter_pool.list_slots()}
        ):
            self._adapter_pool.register_from_manifest(manifest)

    def register_persistent(
        self,
        manifest: DomainPackManifest,
        principal: PrincipalContext | None = None,
        *,
        bundle_bytes: bytes | None = None,
    ) -> ObjectMetadata:
        """Persists a DomainPackManifest into PG17 domain_packs and ArtifactObjectStore."""
        effective_principal = self._effective_principal(principal)
        payload_bytes = (
            bundle_bytes
            if bundle_bytes is not None
            else manifest.model_dump_json(indent=2).encode("utf-8")
        )
        obj_key = build_tenant_domain_key(
            effective_principal.tenant_id, manifest.domain_id, "domain.json"
        )
        meta = self._object_store.put_bytes(
            obj_key,
            payload_bytes,
            content_type="application/json",
            expected_tenant_id=effective_principal.tenant_id,
        )
        if self._enable_pg:
            pg_repo = PostgresRLSRepository()
            with self._ingest_mgr().scoped_transaction(
                effective_principal, enable_hnsw_iterative_scan=False
            ) as conn:
                pg_repo.upsert_domain_pack(
                    conn,
                    manifest,
                    tenant_id=effective_principal.tenant_id,
                    s3_bundle_uri=meta.uri,
                    bundle_sha256=meta.sha256,
                )
        self.register(manifest, tenant_id=effective_principal.tenant_id)
        return meta

    def unregister(self, domain_id: str, *, tenant_id: UUID | None = None) -> None:
        resolved_tid = tenant_id or self._tenant_id
        self._tenant_manifests.pop((resolved_tid, domain_id), None)
        if resolved_tid == self._tenant_id:
            self._manifests.pop(domain_id, None)
            self._adapter_pool.unregister_slot(domain_id)

    def get(
        self,
        domain_id: str,
        principal: PrincipalContext | None = None,
    ) -> DomainPackManifest:
        effective_principal = self._effective_principal(principal)
        key = (effective_principal.tenant_id, domain_id)
        if key not in self._tenant_manifests:
            self.load_all(principal=effective_principal)
        if key in self._tenant_manifests:
            return self._tenant_manifests[key]
        if effective_principal.tenant_id == self._tenant_id and domain_id in self._manifests:
            return self._manifests[domain_id]
        registered_for_tenant = sorted(
            d for t, d in self._tenant_manifests if t == effective_principal.tenant_id
        )
        raise KeyError(
            f"Unknown domain_id '{domain_id}' for tenant {effective_principal.tenant_id}. "
            f"Registered: {registered_for_tenant}"
        )

    def is_mcp_tool_allowed(
        self,
        domain_id: str,
        tool_qualified_name: str,
        principal: PrincipalContext | None = None,
    ) -> bool:
        manifest = self.get(domain_id, principal=principal)
        return tool_qualified_name in manifest.mcp_read_tools_allowed
