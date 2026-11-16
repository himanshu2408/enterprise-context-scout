"""DomainPack manifest, live RLS corpus counts, and adapter pool registry endpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.api.dependencies import (
    get_domain_registry,
    get_principal_context,
    get_repo_root,
)
from context_scout.api.schemas import DomainDetail, DomainsResponse
from context_scout.config import ScoutSettings
from context_scout.db.session import RLSConnectionManager
from context_scout.schemas.contracts import PrincipalContext

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Domains"])


def _query_rls_domain_counts(principal: PrincipalContext) -> dict[str, dict[str, int]]:
    """Query live PostgreSQL 17 row counts per domain under RLS for principal."""
    counts: dict[str, dict[str, int]] = {}
    try:
        settings = ScoutSettings()
        conn_mgr = RLSConnectionManager(settings.db_app_url)
        with conn_mgr.scoped_transaction(principal) as conn:
            doc_rows = conn.execute(
                "SELECT domain_id, COUNT(*), COUNT(DISTINCT doc_id) "
                "FROM doc_chunks GROUP BY domain_id;"
            ).fetchall()
            for domain_id, chunk_count, doc_count in doc_rows:
                entry = counts.setdefault(str(domain_id), {})
                entry["doc_chunks_count"] = (
                    int(chunk_count) if isinstance(chunk_count, (int, str)) else 0
                )
                entry["documents_count"] = (
                    int(doc_count) if isinstance(doc_count, (int, str)) else 0
                )
            for table, key in (
                ("code_files", "code_files_count"),
                ("ast_symbols", "ast_symbols_count"),
            ):
                rows = conn.execute(
                    f"SELECT domain_id, COUNT(*) FROM {table} GROUP BY domain_id;"
                ).fetchall()
                for domain_id, row_count in rows:
                    entry = counts.setdefault(str(domain_id), {})
                    entry[key] = int(row_count) if isinstance(row_count, (int, str)) else 0
    except Exception as exc:
        logger.debug("Live domain count query skipped (%s): %s", type(exc).__name__, exc)
    return counts


def build_domain_details(
    registry: DomainRegistry,
    principal: PrincipalContext,
) -> list[DomainDetail]:
    """Build enriched DomainDetail records combining manifests, adapter health, and PG17 counts."""
    repo_root = get_repo_root()
    manifests = registry.load_all()
    rls_counts = _query_rls_domain_counts(principal)
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
                except Exception as exc:
                    logger.warning(
                        "Failed loading training metrics from %s (%s): %s",
                        candidate,
                        type(exc).__name__,
                        exc,
                    )

        domain_counts = rls_counts.get(domain_id, {})
        modalities = list(manifest.modalities)
        if not modalities:
            modalities = [manifest.authority_rules.primary_truth_source]

        domain_details.append(
            DomainDetail(
                domain_id=domain_id,
                display_name=manifest.display_name,
                description=manifest.description,
                modalities=modalities,
                sample_queries=list(manifest.sample_queries),
                lora_enabled=manifest.lora_enabled,
                default_scopes=list(manifest.default_scopes),
                doc_chunks_count=domain_counts.get("doc_chunks_count", 0),
                documents_count=domain_counts.get("documents_count", 0),
                code_files_count=domain_counts.get("code_files_count", 0),
                ast_symbols_count=domain_counts.get("ast_symbols_count", 0),
                manifest=manifest,
                adapter_verification=adapter_verification,
                training_metrics=training_metrics,
            )
        )

    return domain_details


@router.get("/domains", response_model=DomainsResponse)
def list_domains(
    registry: Annotated[DomainRegistry, Depends(get_domain_registry)],
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
) -> DomainsResponse:
    """Returns registered DomainPack manifests, RLS corpus counts, and adapter health."""
    domain_details = build_domain_details(registry=registry, principal=principal)
    return DomainsResponse(
        domains=domain_details,
        count=len(domain_details),
    )
