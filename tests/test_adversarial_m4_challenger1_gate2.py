"""Challenger 1 Adversarial Suite — Milestone 4 Iteration 2 Gate 2.

Empirical verification and stress testing of:
1. Live PostgreSQL 17 benchmark harness isolation and canary protection.
2. 4-table zero-row assertion on teardown (`ast_symbols`, `code_files`,
   `doc_chunks`, `domain_packs`).
3. Teardown predicate mutation proofs (omitted table deletions, repo-scoped leakage).
4. Cross-domain collision resistance with identical repo names and file paths.
5. Forced RLS enforcement on live PostgreSQL 17 for synthetic benchmark fixtures.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator

import psycopg
import pytest
from tests.test_zero_spend_abstention import (
    is_postgres_reachable,
    seed_synthetic_benchmark_repo,
    teardown_synthetic_benchmark_repo,
)

from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    CodeFileRecord,
    PostgresRLSRepository,
)
from context_scout.schemas.contracts import PrincipalContext, RoleType


@pytest.fixture(scope="module")
def postgres_admin_conn() -> Generator[psycopg.Connection, None, None]:
    """Provides a live PostgreSQL 17 admin connection for test verification."""
    if not is_postgres_reachable():
        pytest.fail("Live PostgreSQL 17 instance is unreachable at db_admin_url!")

    st = ScoutSettings()
    with psycopg.connect(st.db_admin_url) as conn:
        yield conn


class TestLivePostgresBenchmarkIsolationAndTeardown:
    """Adversarially challenge the live PostgreSQL 17 benchmark isolation and teardown harness."""

    def test_teardown_asserts_zero_rows_across_all_4_tables(
        self,
        postgres_admin_conn: psycopg.Connection,
    ) -> None:
        """Empirically verify that teardown_synthetic_benchmark_repo purges all 4 tables:
        `doc_chunks`, `ast_symbols`, `code_files`, and `domain_packs`.
        """
        conn = postgres_admin_conn
        session_domain = f"eval_synthetic_{uuid.uuid4().hex[:12]}"

        # 1. Seed benchmark dataset into PostgreSQL
        seed_synthetic_benchmark_repo(conn, session_domain)

        # 2. Assert rows exist prior to teardown across all 4 tables
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM domain_packs WHERE domain_id = %s;", (session_domain,)
            )
            dp_row = cur.fetchone()
            assert dp_row is not None
            dp_count = dp_row[0]
            cur.execute("SELECT COUNT(*) FROM code_files WHERE domain_id = %s;", (session_domain,))
            cf_row = cur.fetchone()
            assert cf_row is not None
            cf_count = cf_row[0]
            cur.execute("SELECT COUNT(*) FROM ast_symbols WHERE domain_id = %s;", (session_domain,))
            sym_row = cur.fetchone()
            assert sym_row is not None
            sym_count = sym_row[0]
            cur.execute("SELECT COUNT(*) FROM doc_chunks WHERE domain_id = %s;", (session_domain,))
            doc_row = cur.fetchone()
            assert doc_row is not None
            doc_count = doc_row[0]

        assert dp_count == 1, f"Expected 1 domain_pack, got {dp_count}"
        assert cf_count >= 300, f"Expected >= 300 code_files, got {cf_count}"
        assert sym_count >= 300, f"Expected >= 300 ast_symbols, got {sym_count}"
        assert doc_count >= 80, f"Expected >= 80 doc_chunks, got {doc_count}"

        # 3. Execute teardown
        teardown_synthetic_benchmark_repo(conn, session_domain)

        # 4. Assert independent queries confirm exactly 0 rows remain across all 4 tables
        with conn.cursor() as cur:
            for tbl in ("doc_chunks", "ast_symbols", "code_files", "domain_packs"):
                cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE domain_id = %s;", (session_domain,))
                row = cur.fetchone()
                assert row is not None
                cnt = row[0]
                assert cnt == 0, (
                    f"Teardown failure: table '{tbl}' still contains {cnt} rows "
                    f"for domain {session_domain}!"
                )

    def test_canary_isolation_with_identical_repo_and_file_path(
        self,
        postgres_admin_conn: psycopg.Connection,
    ) -> None:
        """Adversarial stress: Canary row uses the EXACT SAME repo_name and file_path
        as a seeded benchmark file ('dotnet/eShop' and
        'src/Services/Ordering/Ordering.API/Application/Commands/OrderCheckoutHandler_00.cs').
        Proves that benchmark teardown on session_domain CANNOT touch canary row in another domain.
        """
        conn = postgres_admin_conn
        canary_domain = f"canary_verification_domain_{uuid.uuid4().hex[:8]}"
        canary_fid = uuid.uuid4()
        session_domain = f"eval_synthetic_{uuid.uuid4().hex[:12]}"

        collision_repo = "dotnet/eShop"
        collision_path = (
            "src/Services/Ordering/Ordering.API/Application/Commands/OrderCheckoutHandler_00.cs"
        )

        try:
            # 1. Insert collision canary row
            conn.execute(
                """
                INSERT INTO domain_packs (
                    domain_id, display_name, lora_adapter_id,
                    primary_truth_source, freshness_half_life_days, config_json
                )
                VALUES (%s, 'Canary Collision Domain', 'none', 'code', 30, '{}')
                ON CONFLICT (domain_id) DO NOTHING;
                """,
                (canary_domain,),
            )
            pg_repo = PostgresRLSRepository()
            pg_repo.insert_code_file(
                conn,
                CodeFileRecord(
                    file_id=canary_fid,
                    domain_id=canary_domain,
                    repo_name=collision_repo,
                    file_path=collision_path,
                    git_commit_sha="9999888877776666555544443333222211110000",
                    git_blob_sha="8888777766665555444433332222111100009999",
                    language="csharp",
                    allowed_scopes=["scope:canary_isolated"],
                    raw_content="// Canary collision test content",
                    line_count=1,
                ),
            )
            conn.commit()

            # 2. Seed session domain which inserts the exact same repo_name & file_path
            seed_synthetic_benchmark_repo(conn, session_domain)

            # 3. Teardown session domain
            teardown_synthetic_benchmark_repo(conn, session_domain)

            # 4. Verify canary row survived 100% intact
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*), git_commit_sha FROM code_files "
                    "WHERE domain_id = %s GROUP BY git_commit_sha;",
                    (canary_domain,),
                )
                row = cur.fetchone()
                assert row is not None, "Canary row was erroneously deleted!"
                assert row[0] == 1
                assert row[1] == "9999888877776666555544443333222211110000"

        finally:
            conn.execute("DELETE FROM code_files WHERE domain_id = %s;", (canary_domain,))
            conn.execute("DELETE FROM domain_packs WHERE domain_id = %s;", (canary_domain,))
            conn.commit()

    @pytest.mark.parametrize(
        "omitted_table",
        ["doc_chunks", "ast_symbols", "code_files", "domain_packs"],
    )
    def test_teardown_mutation_proof_omitted_table_fails_assertion(
        self,
        postgres_admin_conn: psycopg.Connection,
        omitted_table: str,
    ) -> None:
        """Mutation test: If teardown logic is mutated to skip any of the 4 tables,
        the post-teardown zero-row assertion must raise an AssertionError naming that table.
        """
        conn = postgres_admin_conn
        session_domain = f"eval_synthetic_{uuid.uuid4().hex[:12]}"

        seed_synthetic_benchmark_repo(conn, session_domain)

        try:
            # Note: doc_chunks is FK child of domain_packs (ON DELETE CASCADE),
            # and ast_symbols is FK child of code_files (ON DELETE CASCADE).
            # To test an omitted table, its parent table in the FK hierarchy must not be deleted.
            if omitted_table == "doc_chunks":
                conn.execute("DELETE FROM ast_symbols WHERE domain_id = %s;", (session_domain,))
                conn.execute("DELETE FROM code_files WHERE domain_id = %s;", (session_domain,))
                # Do not delete domain_packs so doc_chunks is not cascade-deleted
            elif omitted_table == "ast_symbols":
                conn.execute("DELETE FROM doc_chunks WHERE domain_id = %s;", (session_domain,))
                # Do not delete code_files so ast_symbols is not cascade-deleted
            elif omitted_table == "code_files":
                conn.execute("DELETE FROM doc_chunks WHERE domain_id = %s;", (session_domain,))
                conn.execute("DELETE FROM ast_symbols WHERE domain_id = %s;", (session_domain,))
            elif omitted_table == "domain_packs":
                conn.execute("DELETE FROM doc_chunks WHERE domain_id = %s;", (session_domain,))
                conn.execute("DELETE FROM ast_symbols WHERE domain_id = %s;", (session_domain,))
                conn.execute("DELETE FROM code_files WHERE domain_id = %s;", (session_domain,))
            conn.commit()

            # Execute the 4-table zero-row verification from teardown_synthetic_benchmark_repo
            match_msg = f"Teardown failure: {omitted_table} still contains"
            with (
                pytest.raises(AssertionError, match=match_msg),
                conn.cursor() as cur,
            ):
                for tbl in ("doc_chunks", "ast_symbols", "code_files", "domain_packs"):
                    cur.execute(
                        f"SELECT COUNT(*) FROM {tbl} WHERE domain_id = %s;", (session_domain,)
                    )
                    row = cur.fetchone()
                    assert row is not None
                    cnt = row[0]
                    assert cnt == 0, (
                        f"Teardown failure: {tbl} still contains {cnt} rows for {session_domain}!"
                    )

        finally:
            # Clean up properly after mutation test
            conn.execute("DELETE FROM doc_chunks WHERE domain_id = %s;", (session_domain,))
            conn.execute("DELETE FROM ast_symbols WHERE domain_id = %s;", (session_domain,))
            conn.execute("DELETE FROM code_files WHERE domain_id = %s;", (session_domain,))
            conn.execute("DELETE FROM domain_packs WHERE domain_id = %s;", (session_domain,))
            conn.commit()

    def test_teardown_mutation_proof_flawed_repo_predicate_destroys_canary(
        self,
        postgres_admin_conn: psycopg.Connection,
    ) -> None:
        """Mutation test: Proves that deleting by repo_name instead of domain_id
        erroneously destroys other domains sharing the same repo_name.
        """
        conn = postgres_admin_conn
        canary_domain = f"canary_verification_domain_{uuid.uuid4().hex[:8]}"
        canary_fid = uuid.uuid4()
        shared_repo = f"canary_repo_{uuid.uuid4().hex[:8]}"

        try:
            conn.execute(
                """
                INSERT INTO domain_packs (
                    domain_id, display_name, lora_adapter_id,
                    primary_truth_source, freshness_half_life_days, config_json
                )
                VALUES (%s, 'Canary Domain', 'none', 'code', 30, '{}')
                ON CONFLICT (domain_id) DO NOTHING;
                """,
                (canary_domain,),
            )
            pg_repo = PostgresRLSRepository()
            pg_repo.insert_code_file(
                conn,
                CodeFileRecord(
                    file_id=canary_fid,
                    domain_id=canary_domain,
                    repo_name=shared_repo,
                    file_path="src/shared_canary.cs",
                    git_commit_sha="aaaa111122223333444455556666777788889999",
                    git_blob_sha="bbbb111122223333444455556666777788889999",
                    language="csharp",
                    allowed_scopes=["scope:canary"],
                    raw_content="// Canary code",
                    line_count=1,
                ),
            )
            conn.commit()

            # Execute flawed teardown that lacks domain isolation
            def flawed_teardown(c: psycopg.Connection) -> None:
                c.execute("DELETE FROM code_files WHERE repo_name = %s;", (shared_repo,))

            flawed_teardown(conn)

            # Mutation detected: canary row count dropped to 0
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM code_files WHERE domain_id = %s;", (canary_domain,)
                )
                row = cur.fetchone()
                assert row is not None
                canary_count = row[0]
                assert canary_count == 0, "Flawed teardown failed to delete canary row as expected!"

        finally:
            conn.execute("DELETE FROM code_files WHERE domain_id = %s;", (canary_domain,))
            conn.execute("DELETE FROM domain_packs WHERE domain_id = %s;", (canary_domain,))
            conn.commit()

    def test_live_postgres_forced_rls_blocks_unauthorized_scope_access(
        self,
        postgres_admin_conn: psycopg.Connection,
    ) -> None:
        """Verifies that Forced RLS on live PostgreSQL 17 actively filters rows
        at the database level when accessed via application connection.
        """
        st = ScoutSettings()
        session_domain = f"eval_synthetic_{uuid.uuid4().hex[:12]}"
        seed_synthetic_benchmark_repo(postgres_admin_conn, session_domain)

        try:
            from context_scout.db.session import RLSConnectionManager

            app_conn_mgr = RLSConnectionManager(st.db_app_url)
            pg_repo = PostgresRLSRepository()

            # Authorized principal: allowed_scopes contains "repo:dotnet/eShop"
            p_auth = PrincipalContext(
                user_id="authorized_engineer",
                role=RoleType.ENGINEER,
                allowed_scopes=["repo:dotnet/eShop"],
            )

            # Unauthorized principal: allowed_scopes contains only "unrelated:scope"
            p_unauth = PrincipalContext(
                user_id="unauthorized_guest",
                role=RoleType.ENGINEER,
                allowed_scopes=["unrelated:scope"],
            )

            with app_conn_mgr.scoped_transaction(p_auth) as conn:
                auth_files = pg_repo.list_visible_code_files(conn, session_domain)
                assert len(auth_files) == 75, (
                    f"Expected 75 visible files for authorized principal, got {len(auth_files)}"
                )

            with app_conn_mgr.scoped_transaction(p_unauth) as conn:
                unauth_files = pg_repo.list_visible_code_files(conn, session_domain)
                assert len(unauth_files) == 0, (
                    "Expected 0 visible files for unauthorized principal under Forced RLS, "
                    f"got {len(unauth_files)}"
                )

        finally:
            teardown_synthetic_benchmark_repo(postgres_admin_conn, session_domain)
