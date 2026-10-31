"""Health check endpoint: database connectivity, adapter verification, and M5 Pro profile."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends

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


@router.get("/health", response_model=HealthResponse)
def get_health(
    registry: Annotated[DomainRegistry, Depends(get_domain_registry)],
) -> HealthResponse:
    """Returns service health, DB connectivity, adapter pool status, and hardware profile."""
    repo_root = get_repo_root()
    settings = ScoutSettings()

    # Database connectivity probe
    db_connected = False
    db_status = "offline_in_memory"
    try:
        conn_mgr = RLSConnectionManager(settings.db_app_url)
        probe = PrincipalContext(
            user_id="health_check_probe",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        with conn_mgr.scoped_transaction(probe) as conn:
            conn.execute("SELECT 1;")
        db_connected = True
        db_status = "connected"
    except Exception as exc:
        logger.debug("Database probe fallback to in-memory: %s", exc)
        db_connected = False
        db_status = "offline_in_memory"

    # Adapter pool verification
    registered_domains = list(registry.load_all().keys())
    adapter_status: dict[str, Any] = registry.adapter_pool.verify_all_adapters(base_dir=repo_root)

    # Auto-detected or .env-configured hardware profile
    hardware_profile = detect_hardware_profile(
        settings=settings,
        lora_slots_allocated=len(registry.adapter_pool.list_slots()),
    )

    overall_status = "healthy"
    for v in adapter_status.values():
        if isinstance(v, dict) and not v.get("is_valid", False):
            overall_status = "degraded"
            break

    return HealthResponse(
        status=overall_status,  # type: ignore[arg-type]
        database_connected=db_connected,
        database_status=db_status,
        registered_domains=registered_domains,
        adapter_pool_status=adapter_status,
        hardware_profile=hardware_profile,
        timestamp=datetime.now(UTC).isoformat(),
    )
