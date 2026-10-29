"""Live PostgreSQL 17 + pgvector + Forced RLS integration tests (SEC-01, SEC-02)."""

import uuid
from typing import Any

import pytest

from context_scout.config import ScoutSettings
from context_scout.db.session import RLSConnectionManager
from context_scout.schemas.contracts import PrincipalContext, RoleType


@pytest.fixture(scope="module")
def settings() -> ScoutSettings:
    return ScoutSettings()


def test_live_postgres_version_and_extensions(settings: ScoutSettings) -> None:
    """Verifies live database is PostgreSQL 17 with pgvector and pg_trgm active."""
    mgr = RLSConnectionManager(settings.db_app_url)
    principal = PrincipalContext(
        user_id="test_runner",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    with mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
        version_row = conn.execute("SHOW server_version;").fetchone()
        assert version_row is not None
        server_version: str = str(version_row[0])
        assert server_version.startswith("17."), f"Expected PostgreSQL 17, got {server_version}"

        ext_rows = conn.execute(
            "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_trgm');"
        ).fetchall()
        ext_dict = {row[0]: row[1] for row in ext_rows}
        assert "vector" in ext_dict, "pgvector extension is not installed in live PostgreSQL"
        assert "pg_trgm" in ext_dict, "pg_trgm extension is not installed in live PostgreSQL"


def test_live_forced_row_level_security_catalog(settings: ScoutSettings) -> None:
    """Verifies ENABLE ROW LEVEL SECURITY and FORCE ROW LEVEL SECURITY in pg_class."""
    mgr = RLSConnectionManager(settings.db_app_url)
    principal = PrincipalContext(
        user_id="test_runner",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    with mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
        rows = conn.execute(
            """
            SELECT relname, relrowsecurity, relforcerowsecurity
            FROM pg_class
            WHERE relname IN ('code_files', 'ast_symbols', 'ast_edges', 'doc_chunks')
            ORDER BY relname;
            """
        ).fetchall()

        assert len(rows) == 4, f"Expected 4 tenant tables, got {len(rows)}"
        for relname, relrowsecurity, relforcerowsecurity in rows:
            assert relrowsecurity is True, f"Table {relname} does not have ENABLE RLS"
            assert relforcerowsecurity is True, f"Table {relname} does not have FORCE RLS"


def test_live_rls_isolation_between_principals(settings: ScoutSettings) -> None:
    """Verifies that scout_app_user cannot read rows across tenant/scope boundaries."""
    mgr = RLSConnectionManager(settings.db_app_url)

    # 1. Setup prerequisite domain pack
    admin_mgr = RLSConnectionManager(settings.db_admin_url)
    admin_principal = PrincipalContext(
        user_id="db_admin",
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
                'code', 180, '{"test": true}'::jsonb
            )
            ON CONFLICT (domain_id) DO NOTHING;
            """
        )

    # 2. Insert restricted code file using quant_restricted principal
    quant_principal = PrincipalContext(
        user_id="quant_dev_1",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["quant_restricted"],
    )
    file_id = uuid.uuid4()

    with mgr.scoped_transaction(quant_principal, enable_hnsw_iterative_scan=False) as conn:
        conn.execute(
            """
            INSERT INTO code_files (
                file_id, domain_id, repo_name, file_path,
                git_commit_sha, git_blob_sha, language,
                allowed_scopes, raw_content, line_count
            )
            VALUES (%s, 'trading_code_scout', 'strat-core', 'src/alpha/secret_arb.py',
                    'commit123', 'blob456', 'python',
                    %s, 'def proprietary_alpha(): return 42\n', 1);
            """,
            (file_id, ["quant_restricted"]),
        )

    try:
        # 3. Product Manager with only public_eng scope must see ZERO rows
        pm_principal = PrincipalContext(
            user_id="pm_user_1",
            role=RoleType.PRODUCT_MANAGER,
            allowed_scopes=["public_eng"],
        )
        with mgr.scoped_transaction(pm_principal, enable_hnsw_iterative_scan=False) as conn:
            pm_results = conn.execute(
                "SELECT file_id, file_path FROM code_files WHERE file_id = %s;",
                (file_id,),
            ).fetchall()
            assert len(pm_results) == 0, "RLS Security Violation: PM saw quant_restricted file!"

        # 4. Quant principal MUST see the row
        with mgr.scoped_transaction(quant_principal, enable_hnsw_iterative_scan=False) as conn:
            quant_results = conn.execute(
                "SELECT file_id, file_path FROM code_files WHERE file_id = %s;",
                (file_id,),
            ).fetchall()
            assert len(quant_results) == 1, "Quant principal failed to see authorized file"
            assert quant_results[0][1] == "src/alpha/secret_arb.py"

    finally:
        # 5. Clean up test row
        with admin_mgr.scoped_transaction(
            admin_principal, enable_hnsw_iterative_scan=False
        ) as conn:
            conn.execute("DELETE FROM code_files WHERE file_id = %s;", (file_id,))


def test_live_pgvector_cosine_distance_query(settings: ScoutSettings) -> None:
    """Verifies live pgvector 768-dim similarity search and HNSW iterative scan setting."""
    mgr = RLSConnectionManager(settings.db_app_url)
    admin_mgr = RLSConnectionManager(settings.db_admin_url)

    principal = PrincipalContext(
        user_id="quant_dev_2",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["public_eng"],
    )

    file_id = uuid.uuid4()
    sym_id = uuid.uuid4()
    test_vec = [1.0] + [0.0] * 767
    vec_literal = f"[{','.join(str(x) for x in test_vec)}]"

    # Setup parent file
    with mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
        conn.execute(
            """
            INSERT INTO code_files (
                file_id, domain_id, repo_name, file_path,
                git_commit_sha, git_blob_sha, language,
                allowed_scopes, raw_content, line_count
            )
            VALUES (%s, 'trading_code_scout', 'strat-core', 'src/vector_test.py',
                    'commit_vec', 'blob_vec', 'python',
                    %s, 'def test_vec(): pass\n', 1);
            """,
            (file_id, ["public_eng"]),
        )
        conn.execute(
            """
            INSERT INTO ast_symbols (
                symbol_id, file_id, domain_id, qualified_name,
                symbol_kind, start_line, end_line, signature_skeleton,
                contextual_summary, allowed_scopes, embedding
            )
            VALUES (%s, %s, 'trading_code_scout', 'strat.vector_test.test_vec',
                    'function', 1, 1, 'def test_vec() -> None:',
                    'Vector similarity test fixture', %s, %s::vector);
            """,
            (sym_id, file_id, ["public_eng"], vec_literal),
        )

    try:
        # Test vector search with HNSW relaxed_order iterative scan
        with mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=True) as conn:
            query_vec_literal = f"[{','.join(str(x) for x in test_vec)}]"
            hit: tuple[Any, ...] | None = conn.execute(
                """
                SELECT symbol_id, qualified_name, (embedding <=> %s::vector) AS cosine_dist
                FROM ast_symbols
                WHERE symbol_id = %s;
                """,
                (query_vec_literal, sym_id),
            ).fetchone()

            assert hit is not None
            assert hit[1] == "strat.vector_test.test_vec"
            assert float(hit[2]) < 1e-5, f"Expected ~0.0 cosine distance, got {hit[2]}"

    finally:
        with admin_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
            conn.execute("DELETE FROM ast_symbols WHERE symbol_id = %s;", (sym_id,))
            conn.execute("DELETE FROM code_files WHERE file_id = %s;", (file_id,))
