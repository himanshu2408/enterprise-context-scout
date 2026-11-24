"""Health check endpoint: database connectivity, adapter verification, and host hardware."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from context_scout import __version__
from context_scout.agents.domain_registry import DomainRegistry
from context_scout.api.dependencies import (
    get_domain_registry,
    get_repo_root,
)
from context_scout.api.schemas import HealthResponse
from context_scout.config import ScoutSettings, detect_hardware_profile
from context_scout.db.session import RLSConnectionManager
from context_scout.schemas.contracts import PrincipalContext, RoleType

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


def _detect_cpu_brand_fallback() -> str | None:
    """Detect host CPU/chip string dynamically on macOS or Linux without hardcoded fakes."""
    sys_name = platform.system()
    if sys_name == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            ).strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("macOS CPU brand probe skipped (%s): %s", type(exc).__name__, exc)
    elif sys_name == "Linux" and Path("/proc/cpuinfo").exists():
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    return line.split(":", 1)[1].strip()
        except OSError as exc:
            logger.debug("Linux /proc/cpuinfo read skipped (%s): %s", type(exc).__name__, exc)
    proc = platform.processor().strip()
    return proc or None


def _detect_linux_meminfo_gb() -> float:
    """Fallback RAM detection from /proc/meminfo on Linux hosts."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return 0.0
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return round(int(parts[1]) / (1024 * 1024), 1)
    except (OSError, ValueError) as exc:
        logger.debug("Linux /proc/meminfo read skipped (%s): %s", type(exc).__name__, exc)
    return 0.0


def probe_runtime_environment(
    registry: DomainRegistry,
    repo_root: Path,
    settings: ScoutSettings | None = None,
) -> dict[str, Any]:
    """Probe PostgreSQL 17 connectivity/version, dynamic hardware, and runtime models."""
    cfg = settings or ScoutSettings()
    db_connected = False
    db_status = "offline_in_memory"
    postgres_version: str | None = None
    try:
        conn_mgr = RLSConnectionManager(cfg.db_app_url)
        probe = PrincipalContext(
            user_id="health_check_probe",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        with conn_mgr.scoped_transaction(probe) as conn:
            row = conn.execute("SELECT version();").fetchone()
            if row and row[0]:
                postgres_version = str(row[0]).split(",")[0].strip()
        db_connected = True
        db_status = "connected"
    except Exception as exc:
        logger.debug("Database probe fallback to in-memory: %s", exc)
        db_connected = False
        db_status = "offline_in_memory"

    manifests = registry.load_all()
    registered_domains = list(manifests.keys())
    adapter_status: dict[str, Any] = registry.adapter_pool.verify_all_adapters(base_dir=repo_root)

    active_lora_domains = sum(1 for m in manifests.values() if m.lora_enabled)
    hardware_profile = detect_hardware_profile(
        settings=cfg,
        lora_slots_allocated=active_lora_domains,
    )
    mem_gb = float(hardware_profile.get("memory_gb") or 0.0)
    if mem_gb <= 0.0:
        mem_gb = _detect_linux_meminfo_gb()
        if mem_gb > 0.0:
            hardware_profile["memory_gb"] = mem_gb
            hardware_profile["unified_memory"] = f"{mem_gb:.1f} GB"

    cpu_brand = _detect_cpu_brand_fallback()
    if cpu_brand and str(hardware_profile.get("device", "")).startswith("Apple Silicon ("):
        hardware_profile["device"] = cpu_brand

    runtime_mode = (
        "live_mlx"
        if os.environ.get("CONTEXT_SCOUT_USE_LIVE_MLX", "").strip() == "1"
        else "offline_deterministic"
    )
    hardware_profile["arch"] = platform.machine() or "unknown"
    hardware_profile["os_name"] = f"{platform.system()} {platform.release()}".strip()
    hardware_profile["postgres_version"] = postgres_version
    hardware_profile["runtime_mode"] = runtime_mode

    runtime_models: dict[str, Any] = {}
    overall_status = "healthy"
    for domain_id, manifest in manifests.items():
        adapter_info = adapter_status.get(domain_id, {})
        is_adapter_valid = (
            bool(adapter_info.get("is_valid", False)) if isinstance(adapter_info, dict) else False
        )
        adapter_present = (
            bool(adapter_info.get("adapter_present", is_adapter_valid))
            if isinstance(adapter_info, dict)
            else False
        )
        adapter_state = (
            str(
                adapter_info.get("adapter_state")
                or ("ready" if is_adapter_valid else "no_trained_adapter")
            )
            if isinstance(adapter_info, dict)
            else "no_trained_adapter"
        )
        uses_lora = bool(manifest.lora_enabled and is_adapter_valid)
        runtime_models[domain_id] = {
            "base_model_id": manifest.base_model_id,
            "contextualizer_model_id": manifest.contextualizer_model_id,
            "embedding_model_id": manifest.embedding_model_id,
            "lora_enabled": manifest.lora_enabled,
            "lora_adapter_id": manifest.lora_adapter_id if uses_lora else None,
            "adapter_valid": is_adapter_valid,
            "adapter_state": adapter_state,
            "execution_mode": "qlora_adapter" if uses_lora else "base_gemma4",
        }
        # Serving on base open-weights when no adapter has been trained yet is healthy;
        # only degrade when an existing adapter artifact on disk fails integrity checks.
        if manifest.lora_enabled and adapter_present and not is_adapter_valid:
            overall_status = "degraded"

    return {
        "overall_status": overall_status,
        "database_connected": db_connected,
        "database_status": db_status,
        "registered_domains": registered_domains,
        "adapter_pool_status": adapter_status,
        "hardware_profile": hardware_profile,
        "runtime_models": runtime_models,
        "runtime_mode": runtime_mode,
    }


@router.get("/health", response_model=HealthResponse)
def get_health(
    registry: Annotated[DomainRegistry, Depends(get_domain_registry)],
) -> HealthResponse:
    """Returns service health, DB connectivity, adapter pool status, and hardware profile."""
    repo_root = get_repo_root()
    settings = ScoutSettings()
    probed = probe_runtime_environment(registry=registry, repo_root=repo_root, settings=settings)

    return HealthResponse(
        version=__version__,
        status=probed["overall_status"],
        overall_status=probed["overall_status"],
        database_connected=probed["database_connected"],
        database_status=probed["database_status"],
        registered_domains=probed["registered_domains"],
        adapter_pool_status=probed["adapter_pool_status"],
        hardware_profile=probed["hardware_profile"],
        runtime_models=probed["runtime_models"],
        timestamp=datetime.now(UTC).isoformat(),
    )
