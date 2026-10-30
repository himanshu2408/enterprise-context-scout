"""Declarative DomainPack registry and in-memory LoRA AdapterPool (ARC-02)."""

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from context_scout.schemas.contracts import DomainPackManifest


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
    """Thread-safe in-memory registry pinning active QLoRA adapters (ARC-02 & HW-01).

    Prevents per-request disk I/O thrashing when alternating between Model 1
    (trading_code_scout) and Model 2 (quant_research_scout).
    """

    def __init__(self, max_slots: int = 4, max_kv_tokens: int = 32_768) -> None:
        if max_kv_tokens > 32_768:
            raise ValueError(
                f"max_kv_tokens ({max_kv_tokens}) exceeds Apple M5 Pro 48GB safety cap (32768)"
            )
        self._max_slots = max_slots
        self._max_kv_tokens = max_kv_tokens
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

    def list_slots(self) -> list[AdapterSlot]:
        with self._lock:
            return list(self._slots.values())

    def verify_adapter(
        self,
        domain_id: str,
        base_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Verifies on-disk artifacts for a registered domain adapter.

        Inspects adapters.safetensors, adapter_config.json, and training_metrics.json.
        """
        slot = self.get_slot(domain_id)
        root = Path(base_dir) if base_dir else Path.cwd()
        adapter_dir = (root / slot.adapter_path).resolve()
        display_path = str(adapter_dir)
        try:
            cwd = Path.cwd().resolve()
            if adapter_dir.resolve().is_relative_to(cwd):
                display_path = str(adapter_dir.resolve().relative_to(cwd))
        except Exception:
            pass

        if not adapter_dir.exists():
            direct_path = Path(slot.adapter_path).resolve()
            if direct_path.exists():
                adapter_dir = direct_path
                try:
                    cwd = Path.cwd().resolve()
                    if adapter_dir.resolve().is_relative_to(cwd):
                        display_path = str(adapter_dir.resolve().relative_to(cwd))
                except Exception:
                    pass
            else:
                return {
                    "domain_id": domain_id,
                    "is_valid": False,
                    "adapter_path": display_path,
                    "error_message": f"Adapter directory not found: {adapter_dir}",
                }

        safetensors_path = adapter_dir / "adapters.safetensors"
        config_path = adapter_dir / "adapter_config.json"
        metrics_path = adapter_dir / "training_metrics.json"

        if not safetensors_path.exists() or safetensors_path.stat().st_size == 0:
            return {
                "domain_id": domain_id,
                "is_valid": False,
                "adapter_path": display_path,
                "error_message": f"Missing or empty adapters.safetensors at {safetensors_path}",
            }

        if not config_path.exists() or config_path.stat().st_size == 0:
            return {
                "domain_id": domain_id,
                "is_valid": False,
                "adapter_path": display_path,
                "error_message": f"Missing or empty adapter_config.json at {config_path}",
            }

        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            r = int(cfg.get("r", 0))
            lora_alpha = int(cfg.get("lora_alpha", 0))
            is_moe = bool(cfg.get("is_moe", False))
        except Exception as exc:
            return {
                "domain_id": domain_id,
                "is_valid": False,
                "adapter_path": display_path,
                "error_message": f"Corrupt adapter_config.json: {exc}",
            }

        final_loss = None
        if metrics_path.exists():
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    metrics = json.load(f)
                final_loss = metrics.get("final_loss")
            except Exception:
                pass

        return {
            "domain_id": domain_id,
            "is_valid": True,
            "adapter_path": display_path,
            "safetensors_bytes": safetensors_path.stat().st_size,
            "r": r,
            "lora_alpha": lora_alpha,
            "is_moe": is_moe,
            "final_loss": final_loss,
            "error_message": None,
        }

    def verify_all_adapters(
        self,
        base_dir: Path | str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Verifies artifacts for all registered adapters in pool."""
        results: dict[str, dict[str, Any]] = {}
        for slot in self.list_slots():
            results[slot.domain_id] = self.verify_adapter(slot.domain_id, base_dir=base_dir)
        return results



class DomainRegistry:
    """Loads and validates declarative DomainPack YAML plugins without core code changes."""

    def __init__(self, domains_dir: Path, adapter_pool: AdapterPool | None = None) -> None:
        self._domains_dir = domains_dir
        self._manifests: dict[str, DomainPackManifest] = {}
        self._adapter_pool = adapter_pool or AdapterPool()

    @property
    def adapter_pool(self) -> AdapterPool:
        return self._adapter_pool

    def load_all(self) -> dict[str, DomainPackManifest]:
        if not self._domains_dir.exists():
            raise FileNotFoundError(f"Domains directory not found: {self._domains_dir}")

        for yaml_path in sorted(self._domains_dir.glob("*/domain.yaml")):
            with open(yaml_path, encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
            manifest = DomainPackManifest.model_validate(raw_data)
            self.register(manifest)

        return dict(self._manifests)

    def register(self, manifest: DomainPackManifest) -> None:
        self._manifests[manifest.domain_id] = manifest
        self._adapter_pool.register_from_manifest(manifest)

    def get(self, domain_id: str) -> DomainPackManifest:
        if not self._manifests and self._domains_dir.exists():
            self.load_all()
        if domain_id not in self._manifests:
            raise KeyError(f"Unknown domain_id '{domain_id}'. Registered: {list(self._manifests)}")
        return self._manifests[domain_id]

    def is_mcp_tool_allowed(self, domain_id: str, tool_qualified_name: str) -> bool:
        manifest = self.get(domain_id)
        return tool_qualified_name in manifest.mcp_read_tools_allowed
