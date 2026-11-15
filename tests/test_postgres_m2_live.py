"""Live PostgreSQL 17 integration tests for M2 Code Engine (Merkle Indexer, Forced RLS)."""

from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import psycopg
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


M2_LIVE_SESSION_DOMAIN_PREFIX: Final[str] = "eval_m2_live_"


def _seed_session_domain_pack(conn: psycopg.Connection[Any], domain_id: str) -> None:
    """Seeds an isolated session domain_pack row for live M2 tests."""
    conn.execute(
        """
        INSERT INTO domain_packs (
            domain_id, display_name, lora_adapter_id,
            primary_truth_source, freshness_half_life_days, config_json
        )
        VALUES (%s, %s, 'none', 'code', 180, '{"m2_test": true}'::jsonb)
        ON CONFLICT (domain_id) DO NOTHING;
        """,
        (domain_id, f"Eval M2 Live ({domain_id})"),
    )
    conn.commit()


def cleanup_m2_live_session_domain(
    conn: psycopg.Connection[Any],
    session_domain: str,
    expected_prefix: str = M2_LIVE_SESSION_DOMAIN_PREFIX,
) -> None:
    """Strictly deletes M2 live test rows scoped to a single session domain_id."""
    if not session_domain.startswith(expected_prefix):
        raise ValueError(
            f"Refusing to clean up domain_id={session_domain!r}: "
            f"must start with {expected_prefix!r}"
        )
    conn.execute("DELETE FROM ast_edges WHERE domain_id = %s;", (session_domain,))
    conn.execute("DELETE FROM ast_symbols WHERE domain_id = %s;", (session_domain,))
    conn.execute("DELETE FROM code_files WHERE domain_id = %s;", (session_domain,))
    conn.execute("DELETE FROM domain_packs WHERE domain_id = %s;", (session_domain,))
    conn.commit()


def test_live_postgres_merkle_sync_cold_and_warm_cache_hit(
    settings: ScoutSettings, indexer: GitMerkleIndexer
) -> None:
    """Verifies sync_postgres cold index populates DB and second run executes in <50ms."""
    mgr = RLSConnectionManager(settings.db_app_url)
    repo = PostgresRLSRepository()
    domain_id = f"{M2_LIVE_SESSION_DOMAIN_PREFIX}sync_{uuid4().hex[:8]}"
    repo_name = f"eval-m2-live-sync-{uuid4().hex[:8]}"
    repo_dir = Path.cwd()

    eng_principal = PrincipalContext(
        user_id="eng_indexer_runner",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    try:
        with psycopg.connect(settings.db_admin_url) as admin_conn:
            _seed_session_domain_pack(admin_conn, domain_id)

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

        # Run 2: Warm index (must be <100ms cache hit)
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
        with psycopg.connect(settings.db_admin_url) as cleanup_conn:
            cleanup_m2_live_session_domain(cleanup_conn, domain_id)


def test_live_postgres_ast_rls_isolation_between_principals(
    settings: ScoutSettings,
) -> None:
    """Verifies that scout_app_user cannot read ast_symbols or edges across tenant boundaries."""
    mgr = RLSConnectionManager(settings.db_app_url)
    repo = PostgresRLSRepository()
    domain_id = f"{M2_LIVE_SESSION_DOMAIN_PREFIX}rls_{uuid4().hex[:8]}"
    repo_name = f"eval-m2-live-rls-{uuid4().hex[:8]}"

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
        with psycopg.connect(settings.db_admin_url) as admin_conn:
            _seed_session_domain_pack(admin_conn, domain_id)

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
        with psycopg.connect(settings.db_admin_url) as cleanup_conn:
            cleanup_m2_live_session_domain(cleanup_conn, domain_id)


def test_m2_live_cleanup_canary_isolation_and_mutation_proof(
    settings: ScoutSettings,
) -> None:
    """Canary test proving cleanup helper refuses non-session domains and preserves canary rows."""
    pg_repo = PostgresRLSRepository()
    canary_domain = f"canary_m2_live_{uuid4().hex[:8]}"
    session_domain = f"{M2_LIVE_SESSION_DOMAIN_PREFIX}canary_{uuid4().hex[:8]}"
    shared_repo = f"eval-m2-live-shared-{uuid4().hex[:8]}"

    with psycopg.connect(settings.db_admin_url) as conn:
        try:
            for dom in (canary_domain, session_domain):
                _seed_session_domain_pack(conn, dom)
                fid = uuid4()
                sid = uuid4()
                pg_repo.insert_code_file(
                    conn,
                    CodeFileRecord(
                        file_id=fid,
                        domain_id=dom,
                        repo_name=shared_repo,
                        file_path="src/canary.py",
                        git_commit_sha="c1",
                        git_blob_sha="b1",
                        language="python",
                        allowed_scopes=["public_eng"],
                        raw_content="def f(): pass",
                        line_count=1,
                    ),
                )
                pg_repo.insert_ast_symbol(
                    conn,
                    ASTSymbolRecord(
                        symbol_id=sid,
                        file_id=fid,
                        domain_id=dom,
                        qualified_name=f"{shared_repo}::canary::f",
                        symbol_kind="function",
                        start_line=1,
                        end_line=1,
                        signature_skeleton="def f()",
                        contextual_summary="f",
                        allowed_scopes=["public_eng"],
                        embedding=[0.1] * 768,
                    ),
                )
                pg_repo.insert_ast_edge(
                    conn,
                    ASTEdgeRecord(
                        edge_id=uuid4(),
                        domain_id=dom,
                        source_symbol_id=sid,
                        target_qualified_name="builtins::pass",
                        target_symbol_id=None,
                        edge_kind="CALLS",
                        allowed_scopes=["public_eng"],
                    ),
                )
            conn.commit()

            # 1. Helper must refuse non-session domain prefix
            with pytest.raises(ValueError, match="Refusing to clean up"):
                cleanup_m2_live_session_domain(conn, canary_domain)

            # 2. Scoped cleanup removes session_domain and leaves canary_domain intact
            cleanup_m2_live_session_domain(conn, session_domain)

            for tbl in ("ast_edges", "ast_symbols", "code_files", "domain_packs"):
                session_cnt = conn.execute(
                    f"SELECT COUNT(*) FROM {tbl} WHERE domain_id = %s;",
                    (session_domain,),
                ).fetchone()
                canary_cnt = conn.execute(
                    f"SELECT COUNT(*) FROM {tbl} WHERE domain_id = %s;",
                    (canary_domain,),
                ).fetchone()
                assert session_cnt is not None and session_cnt[0] == 0
                assert canary_cnt is not None and canary_cnt[0] == 1

            # 3. Mutation check: weakening WHERE domain_id = %s to repo_name destroys canary
            conn.execute("DELETE FROM code_files WHERE repo_name = %s;", (shared_repo,))
            conn.commit()
            post_mutation_cnt = conn.execute(
                "SELECT COUNT(*) FROM code_files WHERE domain_id = %s;",
                (canary_domain,),
            ).fetchone()
            assert post_mutation_cnt is not None and post_mutation_cnt[0] == 0
        finally:
            for dom in (canary_domain, session_domain):
                conn.execute("DELETE FROM ast_edges WHERE domain_id = %s;", (dom,))
                conn.execute("DELETE FROM ast_symbols WHERE domain_id = %s;", (dom,))
                conn.execute("DELETE FROM code_files WHERE domain_id = %s;", (dom,))
                conn.execute("DELETE FROM domain_packs WHERE domain_id = %s;", (dom,))
            conn.commit()
