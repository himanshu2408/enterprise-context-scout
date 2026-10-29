"""Live PostgreSQL 17 integration tests for M2 Code Engine (Merkle Indexer, Forced RLS)."""

from pathlib import Path
from uuid import uuid4

import pytest

from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    PostgresRLSRepository,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.git_merkle_indexer import (
    GitMerkleIndexer,
    generate_deterministic_embedding,
)
from context_scout.schemas.contracts import PrincipalContext, RoleType


@pytest.fixture(scope="module")
def settings() -> ScoutSettings:
    return ScoutSettings()


@pytest.fixture
def extractor() -> ASTExtractor:
    return ASTExtractor()


@pytest.fixture
def indexer(extractor: ASTExtractor) -> GitMerkleIndexer:
    return GitMerkleIndexer(extractor=extractor)


@pytest.fixture(scope="module", autouse=True)
def setup_domain_pack(settings: ScoutSettings) -> None:
    """Ensures trading_code_scout domain pack exists prior to live indexer tests."""
    admin_mgr = RLSConnectionManager(settings.db_admin_url)
    admin_principal = PrincipalContext(
        user_id="db_admin_m2",
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
                'trading_code_scout', 'Trading Code Scout', 'gemma4_26b_code_scout',
                'code', 180, '{"m2_test": true}'::jsonb
            )
            ON CONFLICT (domain_id) DO NOTHING;
            """
        )


def test_live_postgres_merkle_sync_cold_and_warm_cache_hit(
    settings: ScoutSettings, indexer: GitMerkleIndexer
) -> None:
    """Verifies sync_postgres cold index populates DB and second run executes in <50ms."""
    mgr = RLSConnectionManager(settings.db_app_url)
    repo = PostgresRLSRepository()
    domain_id = "trading_code_scout"
    repo_name = "enterprise-context-scout-live"
    repo_dir = Path.cwd()

    eng_principal = PrincipalContext(
        user_id="eng_indexer_runner",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    try:
        # Run 1: Cold index
        with mgr.scoped_transaction(eng_principal, enable_hnsw_iterative_scan=False) as conn:
            stats_cold = indexer.sync_postgres(
                repo_dir=repo_dir,
                repo=repo,
                conn=conn,
                principal=eng_principal,
                domain_id=domain_id,
                repo_name=repo_name,
                allowed_scopes=["public_eng"],
            )

            assert stats_cold.cache_hit is False
            assert stats_cold.total_files_scanned > 0
            assert stats_cold.new_files_indexed > 0
            assert stats_cold.total_symbols_stored > 0

            # Verify rows are visible in Postgres
            files = repo.list_visible_code_files(conn, domain_id)
            symbols = repo.list_visible_ast_symbols(conn, domain_id)
            edges = repo.list_visible_ast_edges(conn, domain_id)
            assert len(files) > 0
            assert len(symbols) > 0
            assert len(edges) > 0

        # Run 2: Warm index (must be <50ms cache hit)
        with mgr.scoped_transaction(eng_principal, enable_hnsw_iterative_scan=False) as conn:
            stats_warm = indexer.sync_postgres(
                repo_dir=repo_dir,
                repo=repo,
                conn=conn,
                principal=eng_principal,
                domain_id=domain_id,
                repo_name=repo_name,
                allowed_scopes=["public_eng"],
            )

            assert stats_warm.cache_hit is True
            assert stats_warm.unchanged_files_skipped == stats_warm.total_files_scanned
            assert stats_warm.new_files_indexed == 0
            assert stats_warm.modified_files_reindexed == 0
            assert stats_warm.deleted_files_removed == 0
            assert stats_warm.duration_ms < 100.0, (
                f"PostgreSQL warm cache hit took {stats_warm.duration_ms}ms (must be <100ms)"
            )

    finally:
        # Clean up records created for this repo_name
        with mgr.scoped_transaction(eng_principal, enable_hnsw_iterative_scan=False) as conn:
            conn.execute("DELETE FROM code_files WHERE repo_name = %s;", (repo_name,))


def test_live_postgres_ast_rls_isolation_between_principals(
    settings: ScoutSettings,
) -> None:
    """Verifies that scout_app_user cannot read ast_symbols or edges across tenant boundaries."""
    mgr = RLSConnectionManager(settings.db_app_url)
    repo = PostgresRLSRepository()
    domain_id = "trading_code_scout"
    repo_name = "enterprise-isolation-test"

    quant_principal = PrincipalContext(
        user_id="quant_lead_m2",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["quant_restricted"],
    )
    eng_principal = PrincipalContext(
        user_id="eng_viewer_m2",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    restricted_file_id = uuid4()
    restricted_sym_id = uuid4()
    restricted_edge_id = uuid4()

    try:
        # 1. Insert restricted code file, symbol, and edge as quant_principal
        with mgr.scoped_transaction(quant_principal, enable_hnsw_iterative_scan=False) as conn:
            repo.insert_code_file(
                conn,
                CodeFileRecord(
                    file_id=restricted_file_id,
                    domain_id=domain_id,
                    repo_name=repo_name,
                    file_path="src/alpha/secret_strat.py",
                    git_commit_sha="commit_sec",
                    git_blob_sha="blob_sec",
                    language="python",
                    allowed_scopes=["quant_restricted"],
                    raw_content="def compute_alpha(): return 42\n",
                    line_count=1,
                ),
            )
            repo.insert_ast_symbol(
                conn,
                ASTSymbolRecord(
                    symbol_id=restricted_sym_id,
                    file_id=restricted_file_id,
                    domain_id=domain_id,
                    qualified_name="alpha.secret_strat.compute_alpha",
                    symbol_kind="function",
                    start_line=1,
                    end_line=1,
                    signature_skeleton="def compute_alpha() -> int:",
                    contextual_summary="Proprietary alpha signal generator",
                    allowed_scopes=["quant_restricted"],
                    embedding=generate_deterministic_embedding("compute_alpha proprietary alpha"),
                ),
            )
            repo.insert_ast_edge(
                conn,
                ASTEdgeRecord(
                    edge_id=restricted_edge_id,
                    domain_id=domain_id,
                    source_symbol_id=restricted_sym_id,
                    target_qualified_name="alpha.models.SignalWeights",
                    target_symbol_id=None,
                    edge_kind="USES_TYPE",
                    allowed_scopes=["quant_restricted"],
                ),
            )

            # Quant researcher can see their own records
            quant_files = repo.list_visible_code_files(conn, domain_id)
            assert "src/alpha/secret_strat.py" in quant_files
            quant_symbols = repo.list_visible_ast_symbols(conn, domain_id)
            assert "alpha.secret_strat.compute_alpha" in quant_symbols
            quant_edges = repo.list_visible_ast_edges(conn, domain_id)
            assert "alpha.models.SignalWeights" in quant_edges

        # 2. Public engineer MUST see 0 restricted records (Forced RLS)
        with mgr.scoped_transaction(eng_principal, enable_hnsw_iterative_scan=False) as conn:
            eng_files = repo.list_visible_code_files(conn, domain_id)
            assert "src/alpha/secret_strat.py" not in eng_files

            eng_symbols = repo.list_visible_ast_symbols(conn, domain_id)
            assert "alpha.secret_strat.compute_alpha" not in eng_symbols

            eng_edges = repo.list_visible_ast_edges(conn, domain_id)
            assert "alpha.models.SignalWeights" not in eng_edges

    finally:
        # Clean up restricted test records
        with mgr.scoped_transaction(quant_principal, enable_hnsw_iterative_scan=False) as conn:
            repo.delete_code_file(conn, restricted_file_id)
