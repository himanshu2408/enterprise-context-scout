"""Live PostgreSQL 17 + pgvector + pg_trgm + Forced RLS integration tests for M3 doc_chunks."""

import uuid
from datetime import date

import pytest

from context_scout.config import ScoutSettings
from context_scout.db.repositories import DocChunkRecord, PostgresRLSRepository
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.git_merkle_indexer import generate_deterministic_embedding
from context_scout.schemas.contracts import MarketRegime, PrincipalContext, RoleType


@pytest.fixture(scope="module")
def settings() -> ScoutSettings:
    return ScoutSettings()


def test_live_doc_chunks_forced_rls_isolation(settings: ScoutSettings) -> None:
    """Verifies that live PostgreSQL 17 doc_chunks enforces forced RLS scope isolation."""
    mgr = RLSConnectionManager(settings.db_app_url)
    admin_mgr = RLSConnectionManager(settings.db_admin_url)
    repo = PostgresRLSRepository()

    # 1. Setup domain pack prerequisite
    admin_principal = PrincipalContext(
        user_id="admin_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "quant_restricted"],
    )
    with admin_mgr.scoped_transaction(admin_principal, enable_hnsw_iterative_scan=False) as conn:
        conn.execute(
            """
            INSERT INTO domain_packs (
                domain_id, display_name, lora_adapter_id,
                primary_truth_source, freshness_half_life_days, config_json
            )
            VALUES (
                'quant_research_scout', 'Quant Research Scout', 'gemma4_26b_quant_scout',
                'document', 180, '{"m3_test": true}'::jsonb
            )
            ON CONFLICT (domain_id) DO NOTHING;
            """
        )

    # 2. Insert restricted doc chunk using quant_restricted principal
    quant_principal = PrincipalContext(
        user_id="quant_trader_restricted",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["quant_restricted"],
    )
    chunk_id = uuid.uuid4()
    dummy_emb = [0.0] * 768

    restricted_record = DocChunkRecord(
        chunk_id=chunk_id,
        domain_id="quant_research_scout",
        doc_id="arxiv:2403.99999",
        doc_title="Proprietary Alpha Discovery in Indian Equities",
        source_uri="https://arxiv.org/abs/2403.99999",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        published_date=date(2024, 3, 15),
        authority_weight=1.5,
        section_heading="Confidential Alpha",
        contextual_header="[Context: arxiv:2403.99999 | Market: IN_NSE | Restricted]",
        verbatim_content="Confidential proprietary momentum formulation with STT delivery cost.",
        metadata_json={"confidential": True},
        allowed_scopes=["quant_restricted"],
        embedding=dummy_emb,
    )

    with mgr.scoped_transaction(quant_principal, enable_hnsw_iterative_scan=False) as conn:
        repo.insert_doc_chunk(conn, restricted_record)

    try:
        # 3. Principal with only public_eng scope must see ZERO rows
        public_principal = PrincipalContext(
            user_id="public_user_1",
            role=RoleType.PRODUCT_MANAGER,
            allowed_scopes=["public_eng"],
        )
        with mgr.scoped_transaction(public_principal, enable_hnsw_iterative_scan=False) as conn:
            visible_public = repo.list_visible_doc_chunks(conn, "quant_research_scout")
            assert not any(c.chunk_id == chunk_id for c in visible_public), (
                "RLS Security Violation: Public principal saw quant_restricted doc chunk!"
            )

        # 4. Quant principal MUST see the row
        with mgr.scoped_transaction(quant_principal, enable_hnsw_iterative_scan=False) as conn:
            visible_quant = repo.list_visible_doc_chunks(conn, "quant_research_scout")
            matching = [c for c in visible_quant if c.chunk_id == chunk_id]
            assert len(matching) == 1, "Quant principal failed to see authorized doc chunk"
            assert matching[0].doc_id == "arxiv:2403.99999"

    finally:
        with admin_mgr.scoped_transaction(
            admin_principal, enable_hnsw_iterative_scan=False
        ) as conn:
            conn.execute("DELETE FROM doc_chunks WHERE chunk_id = %s;", (chunk_id,))


def test_live_doc_chunks_3way_hybrid_rrf_query(settings: ScoutSettings) -> None:
    """Verifies live PostgreSQL 17 3-way hybrid RRF query using FTS, pgvector, and pg_trgm."""
    mgr = RLSConnectionManager(settings.db_app_url)
    admin_mgr = RLSConnectionManager(settings.db_admin_url)
    repo = PostgresRLSRepository()

    admin_principal = PrincipalContext(
        user_id="admin_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_quant"],
    )
    with admin_mgr.scoped_transaction(admin_principal, enable_hnsw_iterative_scan=False) as conn:
        conn.execute(
            """
            INSERT INTO domain_packs (
                domain_id, display_name, lora_adapter_id,
                primary_truth_source, freshness_half_life_days, config_json
            )
            VALUES (
                'quant_research_scout', 'Quant Research Scout', 'gemma4_26b_quant_scout',
                'document', 180, '{"m3_test": true}'::jsonb
            )
            ON CONFLICT (domain_id) DO NOTHING;
            """
        )

    principal = PrincipalContext(
        user_id="quant_trader_live",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["public_quant"],
    )

    chunk_id_nse = uuid.uuid4()
    chunk_id_xetra = uuid.uuid4()
    chunk_id_us = uuid.uuid4()

    emb_nse = generate_deterministic_embedding(
        "Securities Transaction Tax STT Nifty delivery Indian equities", 768
    )
    emb_xetra = generate_deterministic_embedding(
        "Deutsche Börse Xetra closing auction DAX liquidity tick size", 768
    )
    emb_us = generate_deterministic_embedding(
        "NYSE NASDAQ Almgren-Chriss order flow imbalance US equities", 768
    )

    record_nse = DocChunkRecord(
        chunk_id=chunk_id_nse,
        domain_id="quant_research_scout",
        doc_id="arxiv:2403.01182",
        doc_title="Momentum with Indian STT",
        source_uri="https://arxiv.org/abs/2403.01182",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        published_date=date(2024, 3, 15),
        authority_weight=1.5,
        section_heading="Empirical Results",
        contextual_header=(
            "[Context: arxiv:2403.01182 (2024) | Market: IN_NSE | Friction: 15 bps]"
        ),
        verbatim_content=(
            "Securities Transaction Tax (STT) requires 15 bps delivery friction on NSE."
        ),
        metadata_json={"sample_period": "2018-2023"},
        allowed_scopes=["public_quant"],
        embedding=emb_nse,
    )
    record_xetra = DocChunkRecord(
        chunk_id=chunk_id_xetra,
        domain_id="quant_research_scout",
        doc_id="arxiv:2311.08941",
        doc_title="Xetra DAX Liquidity",
        source_uri="https://arxiv.org/abs/2311.08941",
        market_scope=MarketRegime.GERMANY_XETRA_DAX,
        published_date=date(2023, 11, 20),
        authority_weight=1.2,
        section_heading="Market Microstructure",
        contextual_header=(
            "[Context: arxiv:2311.08941 (2023) | Market: DE_XETRA | Friction: 10 bps]"
        ),
        verbatim_content=(
            "Deutsche Börse Xetra closing auction dynamics under MiFID II tick sizes."
        ),
        metadata_json={"sample_period": "2019-2023"},
        allowed_scopes=["public_quant"],
        embedding=emb_xetra,
    )
    record_us = DocChunkRecord(
        chunk_id=chunk_id_us,
        domain_id="quant_research_scout",
        doc_id="arxiv:2401.05432",
        doc_title="US Order Flow Imbalance",
        source_uri="https://arxiv.org/abs/2401.05432",
        market_scope=MarketRegime.US_NYSE_NASDAQ,
        published_date=date(2024, 1, 10),
        authority_weight=1.0,
        section_heading="Order Flow Imbalance",
        contextual_header=(
            "[Context: arxiv:2401.05432 (2024) | Market: US_EQUITY | Friction: 5 bps]"
        ),
        verbatim_content=(
            "Almgren-Chriss institutional order execution across NYSE and NASDAQ equities."
        ),
        metadata_json={"sample_period": "2020-2023"},
        allowed_scopes=["public_quant"],
        embedding=emb_us,
    )

    with mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
        repo.insert_doc_chunks_batch(conn, [record_nse, record_xetra, record_us])

    try:
        # Execute 3-way hybrid search with HNSW iterative scan active
        with mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=True) as conn:
            hits = repo.search_doc_chunks_hybrid_rrf(
                conn=conn,
                domain_id="quant_research_scout",
                query_text="Securities Transaction Tax STT delivery NSE",
                query_embedding=emb_nse,
                market_scope=MarketRegime.INDIA_NSE_BSE,
                top_k=5,
            )

            assert len(hits) >= 1, f"Expected at least 1 hit for IN_NSE, got {len(hits)}"
            matching_hits = [h for h in hits if h.item_id == chunk_id_nse]
            assert len(matching_hits) == 1, "Expected inserted NSE chunk in search hits"
            hit = matching_hits[0]
            assert 0.0 <= hit.normalized_rrf_score <= 1.0
            assert hit.bm25_rank is not None and hit.bm25_rank >= 1
            assert hit.vector_rank is not None and hit.vector_rank >= 1

            # Search across all markets without market_scope filter
            all_hits = repo.search_doc_chunks_hybrid_rrf(
                conn=conn,
                domain_id="quant_research_scout",
                query_text="equity execution",
                query_embedding=emb_nse,
                market_scope=None,
                top_k=5,
            )
            assert (
                len(all_hits) >= 3
            ), f"Expected at least 3 hits across all markets, got {len(all_hits)}"
            for h in all_hits:
                assert 0.0 <= h.normalized_rrf_score <= 1.0

    finally:
        with admin_mgr.scoped_transaction(
            admin_principal, enable_hnsw_iterative_scan=False
        ) as conn:
            conn.execute(
                "DELETE FROM doc_chunks WHERE chunk_id IN (%s, %s, %s);",
                (chunk_id_nse, chunk_id_xetra, chunk_id_us),
            )
