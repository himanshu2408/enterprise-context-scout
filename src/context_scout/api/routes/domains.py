"""DomainPack manifest and adapter pool registry endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.api.dependencies import (
    get_domain_registry,
    get_repo_root,
)
from context_scout.api.schemas import DomainDetail, DomainsResponse

router = APIRouter(tags=["Domains"])


@router.get("/domains", response_model=DomainsResponse)
def list_domains(
    registry: Annotated[DomainRegistry, Depends(get_domain_registry)],
) -> DomainsResponse:
    """Returns registered DomainPack manifests, adapter verification, and training metrics."""
    repo_root = get_repo_root()
    manifests = registry.load_all()
    domain_details: list[DomainDetail] = []

    for domain_id, manifest in manifests.items():
        adapter_verification: dict[str, Any] = registry.adapter_pool.verify_adapter(
            domain_id, base_dir=repo_root
        )

        training_metrics: dict[str, Any] | None = None
        adapter_path = Path(manifest.lora_adapter_path)
        metrics_candidates = [
            repo_root / adapter_path / "training_metrics.json",
            adapter_path / "training_metrics.json",
        ]
        for candidate in metrics_candidates:
            if candidate.exists():
                try:
                    with open(candidate, encoding="utf-8") as f:
                        training_metrics = json.load(f)
                    break
                except Exception:
                    pass

        domain_details.append(
            DomainDetail(
                domain_id=domain_id,
                display_name=manifest.display_name,
                manifest=manifest,
                adapter_verification=adapter_verification,
                training_metrics=training_metrics,
            )
        )

    return DomainsResponse(
        domains=domain_details,
        count=len(domain_details),
    )
