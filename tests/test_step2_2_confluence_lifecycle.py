"""Tests for Step 2.2: Confluence lifecycle, noise filtering, tombstones, and conflict detection."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import uuid4

import httpx
import psycopg
import pytest
from tests.fixtures.pg_isolation import (
    DEFAULT_TENANT_ID,
    cleanup_ephemeral_pg_domain,
    create_ephemeral_pg_domain,
)

from context_scout.agents.conflict_detector import ConflictDetector
from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
from context_scout.db.schema import DocChunkRecord
from context_scout.db.session import PoolRole, RLSConnectionManager
from context_scout.plugins.connectors.confluence import (
    ConfluenceConnectorPlugin,
    ConfluenceNoisePolicy,
    ConfluencePageDTO,
)
from context_scout.retrieval.embedder import get_embedding_provider
from context_scout.schemas.contracts import (
    DocumentScope,
    ModificationIntent,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
    compute_normalized_rrf,
)


def _snapshot_shared_counts(conn: psycopg.Connection[Any]) -> dict[str, int]:
    queries = {
        "code_architecture_scout_code_files": (
            "SELECT COUNT(*) FROM code_files "
            "WHERE tenant_id = %s AND domain_id = 'code_architecture_scout'"
        ),
        "code_architecture_scout_ast_symbols": (
            "SELECT COUNT(*) FROM ast_symbols "
            "WHERE tenant_id = %s AND domain_id = 'code_architecture_scout'"
        ),
        "code_architecture_scout_ast_edges": (
            "SELECT COUNT(*) FROM ast_edges "
            "WHERE tenant_id = %s AND domain_id = 'code_architecture_scout'"
        ),
        "enterprise_docs_scout_doc_chunks": (
            "SELECT COUNT(*) FROM doc_chunks "
            "WHERE tenant_id = %s AND domain_id = 'enterprise_docs_scout'"
        ),
    }
    counts: dict[str, int] = {}
    for key, sql in queries.items():
        row = conn.execute(sql, (DEFAULT_TENANT_ID,)).fetchone()
        counts[key] = int(row[0]) if row else 0
    return counts


@pytest.fixture
def s22_pg_domain() -> Iterator[tuple[str, ScoutSettings]]:
    """Provides an isolated eval_conf_s22_<uuid8> PG17 domain with guaranteed FK teardown."""
    settings = ScoutSettings()
    domain_id = f"eval_conf_s22_{uuid4().hex[:8]}"
    with psycopg.connect(settings.db_admin_url) as admin_conn:
        baseline = _snapshot_shared_counts(admin_conn)
        create_ephemeral_pg_domain(admin_conn, domain_id=domain_id)

    try:
        yield domain_id, settings
    finally:
        with psycopg.connect(settings.db_admin_url) as admin_conn:
            cleanup_ephemeral_pg_domain(admin_conn, domain_id)
            rem_domains = admin_conn.execute(
                "SELECT COUNT(*) FROM domain_packs WHERE domain_id = %s",
                (domain_id,),
            ).fetchone()
            rem_pages = admin_conn.execute(
                "SELECT COUNT(*) FROM confluence_pages WHERE domain_id = %s",
                (domain_id,),
            ).fetchone()
            rem_chunks = admin_conn.execute(
                "SELECT COUNT(*) FROM doc_chunks WHERE domain_id = %s",
                (domain_id,),
            ).fetchone()
            assert rem_domains is not None and int(rem_domains[0]) == 0
            assert rem_pages is not None and int(rem_pages[0]) == 0
            assert rem_chunks is not None and int(rem_chunks[0]) == 0

            after = _snapshot_shared_counts(admin_conn)
            for key, before_val in baseline.items():
                assert after[key] == before_val, (
                    f"Shared domain {key} count mutated ({before_val} -> {after[key]})!"
                )


def _make_doc_chunk(
    *,
    domain_id: str,
    doc_id: str,
    title: str,
    content: str,
    authority_weight: float = 1.0,
    market_scope: DocumentScope | str | None = None,
    source_type: str = "confluence",
    version_str: str = "1",
    is_stale: bool = False,
    stale_reason: str | None = None,
    allowed_scopes: list[str] | None = None,
    published_date: date | None = None,
    section_heading: str = "Overview",
    space_key: str = "ENG",
) -> DocChunkRecord:
    pub_dt = published_date or date(2026, 10, 15)
    header = (
        f"This document is from Confluence space '{space_key}' > '{title}' "
        f"(Page ID: {doc_id}, Version: {version_str}, Last Modified: {pub_dt.isoformat()} "
        f"by alice). Section: '{section_heading}'."
    )
    return DocChunkRecord(
        chunk_id=uuid4(),
        domain_id=domain_id,
        doc_id=doc_id,
        doc_title=title,
        source_uri=f"https://example.atlassian.net/wiki/spaces/{space_key}/pages/{doc_id}",
        market_scope=market_scope,
        published_date=pub_dt,
        authority_weight=authority_weight,
        section_heading=section_heading,
        contextual_header=header,
        verbatim_content=content,
        metadata_json={"space_key": space_key, "version": version_str},
        allowed_scopes=allowed_scopes or ["confluence_space_eng"],
        embedding=get_embedding_provider(dimensions=768).embed_text(
            f"{header} {content}", is_query=False
        ),
        source_type=source_type,
        version_str=version_str,
        is_stale=is_stale,
        stale_reason=stale_reason,
    )


def test_repositories_inmemory_persists_lifecycle_market_scopes_and_authority_rrf() -> None:
    """InMemoryRLSRepository persists lifecycle fields, parses market_scope, and ranks authority."""
    repo = InMemoryRLSRepository()
    domain_id = "eval_conf_s22_mem"
    principal = PrincipalContext(
        user_id="eng_1",
        role=RoleType.ENGINEER,
        allowed_scopes=["confluence_space_eng"],
    )

    c_std = _make_doc_chunk(
        domain_id=domain_id,
        doc_id="confluence:101",
        title="Standard Gateway Guide",
        content="Gateway circuit breaker config with steady fallback threshold.",
        authority_weight=1.0,
        market_scope="CONFLUENCE",
        source_type="confluence",
        version_str="2",
    )
    c_boost = _make_doc_chunk(
        domain_id=domain_id,
        doc_id="confluence:102",
        title="Gateway Incident Runbook",
        content="Gateway circuit breaker config with escalation procedure.",
        authority_weight=1.35,
        market_scope=None,
        source_type="confluence",
        version_str="3",
    )
    c_demoted = _make_doc_chunk(
        domain_id=domain_id,
        doc_id="confluence:103",
        title="[DRAFT] Gateway Notes",
        content="Gateway circuit breaker config tentative draft.",
        authority_weight=0.20,
        market_scope="ENTERPRISE_DOC",
        source_type="confluence",
        version_str="1",
    )
    c_stale = _make_doc_chunk(
        domain_id=domain_id,
        doc_id="confluence:104",
        title="Gateway Old Spec",
        content="Gateway circuit breaker config deprecated value.",
        authority_weight=1.5,
        market_scope="JIRA",
        source_type="confluence",
        version_str="1",
        is_stale=True,
        stale_reason="Superseded by page version 2",
    )

    for rec in (c_std, c_boost, c_demoted, c_stale):
        repo.insert_doc_chunk(principal, rec)

    active_rows = repo.list_visible_doc_chunks(principal, domain_id)
    assert {r.chunk_id for r in active_rows} == {
        c_std.chunk_id,
        c_boost.chunk_id,
        c_demoted.chunk_id,
    }

    all_rows = repo.list_visible_doc_chunks(principal, domain_id, include_stale=True)
    assert {r.chunk_id for r in all_rows} == {
        c_std.chunk_id,
        c_boost.chunk_id,
        c_demoted.chunk_id,
        c_stale.chunk_id,
    }
    by_id = {r.chunk_id: r for r in all_rows}
    assert by_id[c_std.chunk_id].market_scope == "CONFLUENCE"
    assert by_id[c_boost.chunk_id].market_scope is None
    assert by_id[c_demoted.chunk_id].market_scope == "ENTERPRISE_DOC"
    assert by_id[c_stale.chunk_id].market_scope == "JIRA"
    assert by_id[c_stale.chunk_id].is_stale is True

    hits_unit = repo.search_doc_chunks_hybrid_rrf(
        principal,
        domain_id=domain_id,
        query_text="Gateway circuit breaker config",
        query_embedding=c_std.embedding,
        top_k=10,
    )
    hit_ids = [h.item_id for h in hits_unit]
    assert c_stale.chunk_id not in hit_ids
    assert hit_ids[0] == c_boost.chunk_id
    assert hit_ids[-1] == c_demoted.chunk_id

    # Verify exact 6-decimal equality on authority_weight == 1.0 baseline isolated domain
    unit_domain = "eval_conf_s22_unit"
    u1 = _make_doc_chunk(
        domain_id=unit_domain,
        doc_id="confluence:201",
        title="Retry Policy",
        content="Retry policy exponential backoff baseline config.",
        authority_weight=1.0,
    )
    repo.insert_doc_chunk(principal, u1)
    unit_hits = repo.search_doc_chunks_hybrid_rrf(
        principal,
        domain_id=unit_domain,
        query_text="Retry policy exponential backoff",
        query_embedding=u1.embedding,
        top_k=5,
    )
    assert len(unit_hits) == 1
    hit0 = unit_hits[0]
    expected_exact = compute_normalized_rrf(
        [hit0.trigram_rank, hit0.bm25_rank, hit0.vector_rank],
        num_modalities=3,
    )
    assert math.isclose(hit0.normalized_rrf_score, expected_exact, rel_tol=0.0, abs_tol=1e-6)


@pytest.mark.postgres
def test_repositories_pg17_persists_lifecycle_columns_null_market_scope_and_authority_rrf(
    s22_pg_domain: tuple[str, ScoutSettings],
) -> None:
    """Live PG17 persists source_type/version/stale, SQL NULL market_scope, and weighted RRF."""
    domain_id, settings = s22_pg_domain
    mgr = RLSConnectionManager(settings.db_app_url)
    repo = PostgresRLSRepository()
    principal = PrincipalContext(
        user_id="eng_pg",
        role=RoleType.ENGINEER,
        allowed_scopes=["confluence_space_eng"],
    )

    c_unit = _make_doc_chunk(
        domain_id=domain_id,
        doc_id="confluence:pg1",
        title="Standard Service Manual",
        content="Unified gateway timeout configuration standard value.",
        authority_weight=1.0,
        market_scope=None,
        source_type="confluence",
        version_str="3",
    )
    c_boost = _make_doc_chunk(
        domain_id=domain_id,
        doc_id="confluence:pg2",
        title="Production Runbook",
        content="Unified gateway timeout configuration incident response.",
        authority_weight=1.35,
        market_scope="CONFLUENCE",
        source_type="confluence",
        version_str="4",
    )
    c_demoted = _make_doc_chunk(
        domain_id=domain_id,
        doc_id="confluence:pg3",
        title="[WIP] Draft Spec",
        content="Unified gateway timeout configuration draft thoughts.",
        authority_weight=0.20,
        market_scope="ENTERPRISE_DOC",
        source_type="confluence",
        version_str="1",
    )
    c_stale = _make_doc_chunk(
        domain_id=domain_id,
        doc_id="confluence:pg4",
        title="Superseded Spec",
        content="Unified gateway timeout configuration legacy v1.",
        authority_weight=1.8,
        market_scope="JIRA",
        source_type="confluence",
        version_str="1",
        is_stale=True,
        stale_reason="Superseded by page version 2",
    )

    with mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
        repo.insert_doc_chunks_batch(conn, [c_unit, c_boost, c_demoted, c_stale])

    # Confirm raw SQL NULL for market_scope and non-null lifecycle columns
    with psycopg.connect(settings.db_admin_url) as admin_conn:
        row = admin_conn.execute(
            """
            SELECT market_scope IS NULL, source_type, version_str, is_stale, stale_reason
            FROM doc_chunks
            WHERE chunk_id = %s
            """,
            (c_unit.chunk_id,),
        ).fetchone()
        assert row is not None
        assert row[0] is True
        assert row[1] == "confluence"
        assert row[2] == "3"
        assert row[3] is False
        assert row[4] is None

    with mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=True) as conn:
        active = repo.list_visible_doc_chunks(conn, domain_id)
        assert {c.chunk_id for c in active} == {
            c_unit.chunk_id,
            c_boost.chunk_id,
            c_demoted.chunk_id,
        }
        all_rows = repo.list_visible_doc_chunks(conn, domain_id, include_stale=True)
        assert {c.chunk_id for c in all_rows} == {
            c_unit.chunk_id,
            c_boost.chunk_id,
            c_demoted.chunk_id,
            c_stale.chunk_id,
        }
        parsed = {c.chunk_id: c for c in all_rows}
        assert parsed[c_unit.chunk_id].market_scope is None
        assert parsed[c_boost.chunk_id].market_scope == "CONFLUENCE"
        assert parsed[c_demoted.chunk_id].market_scope == "ENTERPRISE_DOC"
        assert parsed[c_stale.chunk_id].market_scope == "JIRA"
        assert parsed[c_stale.chunk_id].is_stale is True
        assert parsed[c_stale.chunk_id].stale_reason == "Superseded by page version 2"

        hits = repo.search_doc_chunks_hybrid_rrf(
            conn,
            domain_id=domain_id,
            query_text="Unified gateway timeout configuration",
            query_embedding=c_unit.embedding,
            top_k=10,
        )
        hit_ids = [h.item_id for h in hits]
        assert c_stale.chunk_id not in hit_ids
        assert hit_ids[0] == c_boost.chunk_id
        assert hit_ids[-1] == c_demoted.chunk_id

    # Check exact compute_normalized_rrf 6-decimal equality on authority_weight == 1.0
    unit_domain = f"eval_conf_s22_{uuid4().hex[:8]}"
    with psycopg.connect(settings.db_admin_url) as admin_conn:
        create_ephemeral_pg_domain(admin_conn, domain_id=unit_domain)
    try:
        u1 = _make_doc_chunk(
            domain_id=unit_domain,
            doc_id="confluence:unit1",
            title="Exact Score Page",
            content="Exact deterministic retrieval check phrase.",
            authority_weight=1.0,
        )
        with mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=True) as conn:
            repo.insert_doc_chunk(conn, u1)
            unit_hits = repo.search_doc_chunks_hybrid_rrf(
                conn,
                domain_id=unit_domain,
                query_text="Exact deterministic retrieval check phrase",
                query_embedding=u1.embedding,
                top_k=5,
            )
            assert len(unit_hits) == 1
            pg_hit = unit_hits[0]
            expected = compute_normalized_rrf(
                [pg_hit.trigram_rank, pg_hit.bm25_rank, pg_hit.vector_rank],
                num_modalities=3,
            )
            assert math.isclose(pg_hit.normalized_rrf_score, expected, rel_tol=0.0, abs_tol=1e-6)
    finally:
        with psycopg.connect(settings.db_admin_url) as admin_conn:
            cleanup_ephemeral_pg_domain(admin_conn, unit_domain)


def test_confluence_noise_policy_filters_demotes_and_boosts_cleanly() -> None:
    """ConfluenceNoisePolicy filters personal/archived/meeting/stubs and weights drafts/ADRs."""
    now = datetime.now(UTC)
    connector = ConfluenceConnectorPlugin()

    def _dto(
        pid: str,
        title: str,
        body: str,
        *,
        space_key: str = "ENG",
        status: str = "current",
        labels: tuple[str, ...] = (),
    ) -> ConfluencePageDTO:
        return ConfluencePageDTO(
            page_id=pid,
            space_key=space_key,
            title=title,
            version_number=1,
            author_username="alice",
            source_url=f"https://example.atlassian.net/wiki/spaces/{space_key}/pages/{pid}",
            body_storage=body,
            labels=labels,
            status=status,
            created_at=now,
            last_modified_at=now,
        )

    pages = [
        _dto(
            "p1", "Alice Scratchpad", "<p>Substantive personal notes here.</p>", space_key="~alice"
        ),
        _dto("p2", "Old Gateway", "<p>Substantive archived page.</p>", status="archived"),
        _dto("p3", "[DEPRECATED] V1 Auth", "<p>Legacy auth endpoint specification.</p>"),
        _dto("p4", "Sprint 14 Meeting Notes", "<p>Discussed backlog and velocity items.</p>"),
        _dto("p5", "Weekly Retro", "<p>What went well and action items.</p>", labels=("retro",)),
        _dto("p6", "Empty Page", "<p>Tiny</p>"),
        _dto(
            "p7",
            "Template Page",
            "<ac:placeholder>Click here to edit this template</ac:placeholder>",
        ),
        _dto("p8", "Service Spec", "<p>TBD</p>"),
        _dto("p9", "Legacy Setup", "<p>Legacy setup description.</p>"),  # 25 chars -> standard 1.0
        _dto("p10", "[DRAFT] Queue Design", "<p>Proposed broker topology details.</p>"),
        _dto(
            "p11",
            "Checkout Runbook",
            "<p>Production alert remediation steps for checkout service.</p>",
        ),
        _dto(
            "p12",
            "ADR-019 Caching",
            "<p>Architectural decision record for cache invalidation.</p>",
            labels=("adr",),
        ),
    ]

    repo = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="eng_1",
        role=RoleType.ENGINEER,
        allowed_scopes=connector.mirror_permissions("ENG"),
    )
    res = connector.sync_space_to_repository(
        domain_id="eval_conf_s22_noise",
        space_key="ENG",
        repository=repo,
        principal=principal,
        current_pages=pages,
    )

    assert res.items_scanned == 12
    assert res.noise_filtered_count == 8
    assert res.noise_demoted_count == 1
    assert res.items_updated == 4  # p9 standard + p10 draft + p11 runbook + p12 ADR
    assert res.filter_breakdown["personal_space"] == 1
    assert res.filter_breakdown["status_excluded"] == 1
    assert res.filter_breakdown["deprecated_or_archived"] == 1
    assert res.filter_breakdown["meeting_notes"] == 2
    assert res.filter_breakdown["stub_too_short"] == 1
    assert res.filter_breakdown["template_boilerplate"] == 2

    docs_by_id = {d.doc_id: d for d in repo._docs.values()}
    assert "confluence:p9" in docs_by_id
    assert math.isclose(docs_by_id["confluence:p9"].authority_weight, 1.0)
    assert math.isclose(docs_by_id["confluence:p10"].authority_weight, 0.20)
    assert math.isclose(docs_by_id["confluence:p11"].authority_weight, 1.35)
    assert math.isclose(docs_by_id["confluence:p12"].authority_weight, 1.35)


@pytest.mark.postgres
def test_confluence_lifecycle_pg17_supersession_same_version_tombstone_cursor_and_scope_narrowing(
    s22_pg_domain: tuple[str, ScoutSettings],
) -> None:
    """PG17 lifecycle: replace, supersede, cursor, RLS narrowing, tombstone under scout_app_user."""
    domain_id, settings = s22_pg_domain
    mgr = RLSConnectionManager(settings.db_app_url)
    ingest_mgr = RLSConnectionManager(settings.db_app_url, pool_role=PoolRole.INGESTION)
    ingest_principal = PrincipalContext(
        user_id="confluence_sync_worker",
        role=RoleType.ENGINEER,
        allowed_scopes=["system:connector_ingest"],
    )
    repo = PostgresRLSRepository()

    t1 = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 10, 10, 10, 0, tzinfo=UTC)

    def _mk_page(
        pid: str,
        title: str,
        ver: int,
        body: str,
        mod: datetime,
        status: str = "current",
    ) -> ConfluencePageDTO:
        return ConfluencePageDTO(
            page_id=pid,
            space_key="ENG",
            title=title,
            version_number=ver,
            author_username="alice",
            source_url=f"https://example.atlassian.net/wiki/spaces/ENG/pages/{pid}",
            body_storage=body,
            status=status,
            created_at=mod,
            last_modified_at=mod,
        )

    connector = ConfluenceConnectorPlugin()

    # 1. Initial sync of two pages via scout_app_user (PoolRole.INGESTION, zero db_admin_url!)
    p1_v1 = _mk_page("5001", "Checkout Runbook", 1, "<p>Timeout is 30s initial baseline.</p>", t1)
    p2_v1 = _mk_page(
        "5002", "Gateway Overview", 1, "<p>Legacy gateway architecture overview.</p>", t1
    )
    with ingest_mgr.scoped_transaction(
        ingest_principal, enable_hnsw_iterative_scan=False
    ) as ingest_conn:
        res1 = connector.sync_space_to_repository(
            domain_id=domain_id,
            space_key="ENG",
            repository=repo,
            conn=ingest_conn,
            current_pages=[p1_v1, p2_v1],
        )
    assert res1.items_updated == 2

    # 2. Same-version re-sync replaces v1 chunks atomically (no duplicates!) under scout_app_user
    p1_v1_edit = _mk_page(
        "5001", "Checkout Runbook", 1, "<p>Timeout is 25s corrected baseline.</p>", t1
    )
    with ingest_mgr.scoped_transaction(
        ingest_principal, enable_hnsw_iterative_scan=False
    ) as ingest_conn:
        connector.sync_space_to_repository(
            domain_id=domain_id,
            space_key="ENG",
            repository=repo,
            conn=ingest_conn,
            current_pages=[p1_v1_edit, p2_v1],
        )
        count_row = ingest_conn.execute(
            "SELECT COUNT(*) FROM doc_chunks WHERE domain_id = %s AND doc_id = 'confluence:5001'",
            (domain_id,),
        ).fetchone()
        assert count_row is not None and int(str(count_row[0])) == 1

    # Cross-tenant system:connector_ingest MUST NOT see tenant A's rows
    other_tenant_ingest = PrincipalContext(
        user_id="confluence_sync_worker",
        role=RoleType.ENGINEER,
        allowed_scopes=["system:connector_ingest"],
        tenant_id=uuid4(),
    )
    with ingest_mgr.scoped_transaction(
        other_tenant_ingest, enable_hnsw_iterative_scan=False
    ) as cross_conn:
        assert repo.list_confluence_pages(cross_conn, domain_id, space_key="ENG") == []
        assert repo.list_visible_doc_chunks(cross_conn, domain_id) == []

    public_principal = PrincipalContext(
        user_id="public_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=connector.mirror_permissions("ENG"),
    )
    with mgr.scoped_transaction(public_principal, enable_hnsw_iterative_scan=False) as conn:
        before_narrowing = [
            d
            for d in repo.list_visible_doc_chunks(conn, domain_id)
            if d.doc_id == "confluence:5001"
        ]
        assert len(before_narrowing) == 1

    # 3. Upgrade 5001 v1 -> v2 + narrow allowed_scopes on 5001 under scout_app_user
    p1_v2 = _mk_page("5001", "Checkout Runbook", 2, "<p>Timeout is 10s strict SLA.</p>", t2)
    with ingest_mgr.scoped_transaction(
        ingest_principal, enable_hnsw_iterative_scan=False
    ) as ingest_conn:
        res2 = connector.sync_space_to_repository(
            domain_id=domain_id,
            space_key="ENG",
            repository=repo,
            conn=ingest_conn,
            current_pages=[p1_v2, p2_v1],
            since_updated_at=t1,  # cursor skips unchanged 5002 at t1 and fetches only 5001 at t2
            page_allowed_scopes_map={"5001": ["confluence_restricted_lead"]},
        )
    assert res2.items_scanned == 1  # cursor skipped p2_v1
    assert res2.stale_items_flagged == 1  # v1 flagged stale

    # Verify PG17 forced RLS enforces narrowed allowed_scopes immediately!
    lead_principal = PrincipalContext(
        user_id="lead_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=["confluence_restricted_lead"],
    )
    with mgr.scoped_transaction(public_principal, enable_hnsw_iterative_scan=False) as conn:
        pub_docs = [
            d
            for d in repo.list_visible_doc_chunks(conn, domain_id)
            if d.doc_id == "confluence:5001"
        ]
        assert pub_docs == []
    with mgr.scoped_transaction(lead_principal, enable_hnsw_iterative_scan=False) as conn:
        lead_docs = [
            d
            for d in repo.list_visible_doc_chunks(conn, domain_id)
            if d.doc_id == "confluence:5001"
        ]
        assert len(lead_docs) == 1
        assert "10s strict SLA" in lead_docs[0].verbatim_content

    # 4. Reconcile deletion of 5002 and archive status of 5001 under scout_app_user
    p1_archived = _mk_page(
        "5001",
        "Checkout Runbook",
        3,
        "<p>Archived runbook.</p>",
        t2 + timedelta(days=1),
        status="archived",
    )
    with ingest_mgr.scoped_transaction(
        ingest_principal, enable_hnsw_iterative_scan=False
    ) as ingest_conn:
        res3 = connector.sync_space_to_repository(
            domain_id=domain_id,
            space_key="ENG",
            repository=repo,
            conn=ingest_conn,
            current_pages=[p1_archived],
            reconcile_deletes=True,
        )
        active_pages = repo.list_confluence_pages(ingest_conn, domain_id, space_key="ENG")
        assert active_pages == []
        tomb_rows = ingest_conn.execute(
            """
            SELECT confluence_page_id, is_deleted, status
            FROM confluence_pages
            WHERE domain_id = %s
            ORDER BY confluence_page_id
            """,
            (domain_id,),
        ).fetchall()
        assert tomb_rows == [("5001", True, "archived"), ("5002", True, "deleted")]

    assert res3.tombstoned_count == 2
    assert res3.items_deleted == 2

    with mgr.scoped_transaction(lead_principal, enable_hnsw_iterative_scan=False) as conn:
        assert repo.list_visible_doc_chunks(conn, domain_id) == []

    # 5. Verify HTTP fetch_space_pages parses labels, status, and since_updated_at cursor
    def _mock_http(request: httpx.Request) -> httpx.Response:
        del request
        payload = {
            "results": [
                {
                    "id": "9001",
                    "title": "Old Page",
                    "status": "current",
                    "labels": {"results": [{"name": "architecture"}]},
                    "version": {
                        "number": 1,
                        "authorId": "bob",
                        "createdAt": "2026-10-01T10:00:00Z",
                    },
                    "body": {"storage": {"value": "<p>Older unchanged content.</p>"}},
                },
                {
                    "id": "9002",
                    "title": "Newer ADR",
                    "status": "current",
                    "labels": {"results": [{"name": "adr"}]},
                    "version": {
                        "number": 2,
                        "authorId": "bob",
                        "createdAt": "2026-10-12T10:00:00Z",
                    },
                    "body": {"storage": {"value": "<p>Updated ADR decision text.</p>"}},
                },
            ],
            "_links": {},
        }
        return httpx.Response(200, content=json.dumps(payload))

    http_connector = ConfluenceConnectorPlugin(
        base_url="https://example.atlassian.net",
        client=httpx.Client(
            base_url="https://example.atlassian.net",
            transport=httpx.MockTransport(_mock_http),
        ),
    )
    fetched = http_connector.fetch_space_pages("ENG", since_updated_at=t1)
    assert [p.page_id for p in fetched] == ["9002"]
    assert fetched[0].labels == ("adr",)
    assert fetched[0].status == "current"


def test_conflict_detector_confluence_pages_and_head_code_vs_docs() -> None:
    """ConflictDetector catches cross-page Confluence conflicts/age gaps and HEAD drift."""
    detector = ConflictDetector()
    domain_id = "eval_conf_s22_conflicts"

    auth_runbook = _make_doc_chunk(
        domain_id=domain_id,
        doc_id="confluence:r1",
        title="Checkout Runbook",
        section_heading="Timeouts",
        content="Set timeout: 5s and SLA: 99.95% for checkout orchestration.",
        authority_weight=1.35,
        published_date=date(2026, 10, 15),
        space_key="ENG",
    )
    older_setup = _make_doc_chunk(
        domain_id=domain_id,
        doc_id="confluence:r2",
        title="Legacy Checkout Setup",
        section_heading="Config",
        content="Default timeout: 30s for checkout calls.",
        authority_weight=1.0,
        published_date=date(2025, 1, 10),
        space_key="ENG",
    )
    stale_draft = _make_doc_chunk(
        domain_id=domain_id,
        doc_id="confluence:r3",
        title="Stale Superseded Draft",
        section_heading="Config",
        content="Default timeout: 120s obsolete setting.",
        authority_weight=1.0,
        published_date=date(2024, 1, 1),
        space_key="ENG",
        is_stale=True,
        stale_reason="Superseded",
    )

    page_warnings = detector.detect_confluence_page_conflicts(
        [auth_runbook, older_setup, stale_draft]
    )
    joined_page_warnings = "\n".join(page_warnings)
    assert "CONFLUENCE_CONFIG_CONFLICT (timeout)" in joined_page_warnings
    assert "CONFLUENCE_AGE_DIVERGENCE" in joined_page_warnings
    assert (
        "[Checkout Runbook > Timeouts | Space: ENG | Last Updated: 2026-10-15]"
        in joined_page_warnings
    )
    assert (
        "[Legacy Checkout Setup > Config | Space: ENG | Last Updated: 2025-01-10]"
        in joined_page_warnings
    )
    assert "120s" not in joined_page_warnings  # stale chunk excluded

    # Code vs Doc conflicts: live HEAD code wins implementation state
    code_slice = VerbatimCodeSlice(
        repo_name="context-service",
        file_path="src/gateway/client.py",
        start_line=10,
        end_line=22,
        qualified_symbol="GatewayClient",
        selection_reason="runtime config",
        modification_intent=ModificationIntent.REFERENCE_ONLY,
        git_commit_sha="deadbeefcafebabe1234567890abcdef12345678",
        git_blob_sha="feedface1234567890abcdef1234567890abcdef",
        verbatim_code_with_lines=(
            "10: REQUEST_TIMEOUT_SECONDS = 12\n"
            "11: MAX_RETRIES = 4\n"
            "12: CONTEXT_ENDPOINT = '/api/v2/context'\n"
            "13: POOL_SIZE = int(os.getenv('SCOUT_POOL_LIMIT', '32'))\n"
        ),
    )
    adr_doc = _make_doc_chunk(
        domain_id=domain_id,
        doc_id="confluence:adr9",
        title="ADR-009 Gateway Contract",
        section_heading="Runtime Defaults",
        content=(
            "Gateway specifies timeout: 30s, max_retries: 2, endpoint /api/v1/context, "
            "and SCOUT_POOL_LIMIT = 10."
        ),
        authority_weight=1.35,
        published_date=date(2026, 9, 1),
        space_key="ARCH",
    )

    code_warnings = detector.detect_code_vs_doc_conflicts([code_slice], [adr_doc])
    joined_code_warnings = "\n".join(code_warnings)
    assert "CODE_VS_DOC_CONFLICT (timeout)" in joined_code_warnings
    assert "CODE_VS_DOC_CONFLICT (retries)" in joined_code_warnings
    assert "CODE_VS_DOC_ENDPOINT_DRIFT (context)" in joined_code_warnings
    assert "CODE_VS_DOC_ENV_CONFLICT (SCOUT_POOL_LIMIT)" in joined_code_warnings
    assert "HEAD code wins implementation state" in joined_code_warnings

    # Equivalent units (5000ms == 5s) must NOT raise a false conflict
    equiv_slice = VerbatimCodeSlice(
        repo_name="context-service",
        file_path="src/gateway/units.py",
        start_line=1,
        end_line=3,
        qualified_symbol="UnitConfig",
        selection_reason="unit check",
        modification_intent=ModificationIntent.REFERENCE_ONLY,
        git_commit_sha="abcdef1234567890abcdef1234567890abcdef12",
        git_blob_sha="1234567890abcdef1234567890abcdef12345678",
        verbatim_code_with_lines="1: REQUEST_TIMEOUT_MS = 5000\n",
    )
    equiv_doc = _make_doc_chunk(
        domain_id=domain_id,
        doc_id="confluence:u5",
        title="Unit Match ADR",
        section_heading="Timeout",
        content="Service timeout: 5s.",
        published_date=date(2026, 10, 1),
    )
    assert detector.detect_code_vs_doc_conflicts([equiv_slice], [equiv_doc]) == []


def test_negative_control_weakened_stale_and_noise_gate_fails() -> None:
    """Negative control: weakening include_stale or noise filters leaks stale/noise pages."""
    repo = InMemoryRLSRepository()
    domain_id = "eval_conf_s22_negctrl"
    principal = PrincipalContext(
        user_id="eng_1",
        role=RoleType.ENGINEER,
        allowed_scopes=["confluence_space_eng"],
    )
    now = datetime.now(UTC)

    stale_doc = _make_doc_chunk(
        domain_id=domain_id,
        doc_id="confluence:stale1",
        title="Legacy Retry Spec",
        content="Obsolete retry timeout fallback specification.",
        is_stale=True,
        stale_reason="Superseded by page version 2",
    )
    repo.insert_doc_chunk(principal, stale_doc)

    # Strong gate (default include_stale=False) blocks stale doc
    strict_hits = repo.search_doc_chunks_hybrid_rrf(
        principal,
        domain_id=domain_id,
        query_text="Legacy Retry Spec",
        query_embedding=stale_doc.embedding,
        top_k=5,
    )
    assert strict_hits == []

    # Deliberately weakened search gate leaks stale doc
    weakened_hits = repo.search_doc_chunks_hybrid_rrf(
        principal,
        domain_id=domain_id,
        query_text="Legacy Retry Spec",
        query_embedding=stale_doc.embedding,
        top_k=5,
        include_stale=True,
    )
    assert len(weakened_hits) == 1
    assert weakened_hits[0].item_id == stale_doc.chunk_id

    # Noise gate negative control: personal/deprecated pages pass when policy is weakened
    noise_page = ConfluencePageDTO(
        page_id="n1",
        space_key="~alice",
        title="[DEPRECATED] Old Scratch",
        version_number=1,
        author_username="alice",
        source_url="https://example.atlassian.net/wiki/spaces/~alice/pages/n1",
        body_storage="<p>Short stub.</p>",
        created_at=now,
        last_modified_at=now,
    )
    strict_connector = ConfluenceConnectorPlugin()
    assert strict_connector.evaluate_page_noise(noise_page).action == "exclude"

    weak_connector = ConfluenceConnectorPlugin(
        noise_policy=ConfluenceNoisePolicy(
            exclude_personal_spaces=False,
            deprecated_tokens=(),
            min_substantive_chars=1,
        )
    )
    # Without personal space exclusion, a weakened noise gate lets personal pages pass as standard.
    personal_only = noise_page.model_copy(update={"title": "Personal Draftless Note"})
    assert strict_connector.evaluate_page_noise(personal_only).action == "exclude"
    assert weak_connector.evaluate_page_noise(personal_only).action == "standard"
