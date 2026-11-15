"""Challenger 2 Adversarial Suite — Milestone 4 Iteration 3 Gate Verification.

Empirical verification and stress testing of:
1. Concurrent Database Isolation:
   Execute canary mutation test (flawed_teardown) concurrently with active benchmark
   queries on test_zero_spend_abstention.py fixtures to confirm that private repo scoping
   (f"canary_repo_{uuid.uuid4().hex[:8]}") completely eliminates cross-test interference.
2. Transaction Rollback Under Abort Fault Injection:
   Inject an intentional SQL syntax error inside try: during canary setup, and assert
   that finally: gracefully rolls back via conn.rollback() without raising
   InFailedSqlTransaction or abandoning residual rows.
3. Database Zero Residual Row Audit:
   Verify via SQL queries that after all test runs, 0 rows remain across
   code_files, ast_symbols, doc_chunks, and domain_packs matching
   'eval_synthetic_%' or 'canary_%'.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from collections.abc import Generator

import psycopg
import pytest
from psycopg.errors import InFailedSqlTransaction
from psycopg.errors import SyntaxError as PgSyntaxError
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
from context_scout.db.session import RLSConnectionManager
from context_scout.mcp_servers.common import PostgresScoutGraphAdapter
from context_scout.schemas.contracts import PrincipalContext, RoleType


@pytest.fixture(scope="module")
def postgres_admin_conn() -> Generator[psycopg.Connection, None, None]:
    """Provides a live PostgreSQL 17 admin connection for test verification."""
    if not is_postgres_reachable():
        pytest.fail("Live PostgreSQL 17 instance is unreachable at db_admin_url!")

    st = ScoutSettings()
    with psycopg.connect(st.db_admin_url) as conn:
        yield conn


class TestConcurrentCanaryIsolation:
    """Stress-test concurrent database isolation between benchmark queries and canary mutation."""

    def test_concurrent_canary_flawed_teardown_with_private_repo_scoping_eliminates_interference(
        self,
        postgres_admin_conn: psycopg.Connection,
    ) -> None:
        """Proves that running flawed_teardown with private repo scoping
        (f"canary_repo_{uuid.uuid4().hex[:8]}") concurrently with active benchmark queries
        causes zero interference, zero lost rows, and zero query failures.
        """
        st = ScoutSettings()
        session_domain = f"eval_synthetic_{uuid.uuid4().hex[:12]}"

        # Seed benchmark dataset in PostgreSQL
        with psycopg.connect(st.db_admin_url) as seed_conn:
            seed_synthetic_benchmark_repo(seed_conn, session_domain)

        try:
            conn_mgr = RLSConnectionManager(st.db_app_url)
            adapter = PostgresScoutGraphAdapter(conn_mgr, st)
            eng_p = PrincipalContext(
                user_id="eng_user", role=RoleType.ENGINEER, allowed_scopes=["public_eng"]
            )

            stop_event = threading.Event()
            benchmark_ready = threading.Event()
            benchmark_errors: list[Exception] = []
            benchmark_query_counts = {"queries": 0, "visible_files": 0, "eshop_files": 0}

            def benchmark_query_worker() -> None:
                try:
                    with psycopg.connect(st.db_admin_url) as q_conn:
                        while not stop_event.is_set():
                            # 1. Direct SQL check on benchmark domain code files
                            with q_conn.cursor() as cur:
                                cur.execute(
                                    "SELECT COUNT(*) FROM code_files WHERE domain_id = %s;",
                                    (session_domain,),
                                )
                                total_row = cur.fetchone()
                                assert total_row is not None
                                total = total_row[0]
                                # 300 public_eng + 1 billing + 1 vault = 302 total files
                                assert total == 302, f"Expected 302 total files, got {total}"

                                cur.execute(
                                    "SELECT COUNT(*) FROM code_files "
                                    "WHERE domain_id = %s AND repo_name = 'dotnet/eShop';",
                                    (session_domain,),
                                )
                                eshop_row = cur.fetchone()
                                assert eshop_row is not None
                                eshop_cnt = eshop_row[0]
                                assert eshop_cnt == 75, f"Expected 75 eShop files, got {eshop_cnt}"

                            # 2. Adapter check via RLS (public_eng sees 300 files)
                            visible = adapter.list_visible_code_files(eng_p, session_domain)
                            assert len(visible) == 300, (
                                f"Expected 300 visible files, got {len(visible)}"
                            )

                            benchmark_query_counts["queries"] += 1
                            benchmark_query_counts["visible_files"] = len(visible)
                            benchmark_query_counts["eshop_files"] = eshop_cnt
                            benchmark_ready.set()
                            time.sleep(0.01)
                except Exception as exc:
                    benchmark_errors.append(exc)

            canary_errors: list[Exception] = []
            canary_cycles_completed = [0]

            def canary_mutation_worker() -> None:
                try:
                    with psycopg.connect(st.db_admin_url) as c_conn:
                        for _ in range(15):
                            canary_domain = f"canary_verification_domain_{uuid.uuid4().hex[:8]}"
                            canary_fid = uuid.uuid4()
                            # PRIVATE REPO SCOPING (from b93a2db fix)
                            private_repo = f"canary_repo_{uuid.uuid4().hex[:8]}"

                            try:
                                c_conn.execute(
                                    """
                                    INSERT INTO domain_packs (
                                        domain_id, display_name, lora_adapter_id,
                                        primary_truth_source, freshness_half_life_days, config_json
                                    )
                                    VALUES (%s, 'Concurrent Canary', 'none', 'code', 30, '{}')
                                    ON CONFLICT (domain_id) DO NOTHING;
                                    """,
                                    (canary_domain,),
                                )
                                pg_repo = PostgresRLSRepository()
                                pg_repo.insert_code_file(
                                    c_conn,
                                    CodeFileRecord(
                                        file_id=canary_fid,
                                        domain_id=canary_domain,
                                        repo_name=private_repo,
                                        file_path="src/canary.py",
                                        git_commit_sha="1111222233334444555566667777888899990000",
                                        git_blob_sha="2222333344445555666677778888999900001111",
                                        language="python",
                                        allowed_scopes=["scope:canary"],
                                        raw_content='print("concurrent canary")',
                                        line_count=1,
                                    ),
                                )
                                c_conn.commit()

                                # Flawed teardown mutation (deletes by repo)
                                c_conn.execute(
                                    "DELETE FROM code_files WHERE repo_name = %s;",
                                    (private_repo,),
                                )

                                # Verify that canary row was deleted
                                with c_conn.cursor() as cur:
                                    cur.execute(
                                        "SELECT COUNT(*) FROM code_files WHERE domain_id = %s;",
                                        (canary_domain,),
                                    )
                                    cnt_row = cur.fetchone()
                                    assert cnt_row is not None
                                    cnt = cnt_row[0]
                                    assert cnt == 0, f"Flawed teardown failed: {cnt}"

                                canary_cycles_completed[0] += 1
                            finally:
                                with contextlib.suppress(Exception):
                                    c_conn.rollback()
                                c_conn.execute(
                                    "DELETE FROM code_files WHERE domain_id = %s;",
                                    (canary_domain,),
                                )
                                c_conn.execute(
                                    "DELETE FROM domain_packs WHERE domain_id = %s;",
                                    (canary_domain,),
                                )
                                c_conn.commit()
                            time.sleep(0.01)
                except Exception as exc:
                    canary_errors.append(exc)

            # Launch both threads concurrently
            t_benchmark = threading.Thread(target=benchmark_query_worker)
            t_canary = threading.Thread(target=canary_mutation_worker)

            t_benchmark.start()
            assert benchmark_ready.wait(timeout=5.0), (
                "Benchmark worker failed to initialize queries"
            )
            t_canary.start()

            # Wait for canary cycles to finish
            t_canary.join(timeout=10.0)
            stop_event.set()
            t_benchmark.join(timeout=10.0)

            # Assert no errors occurred in either thread
            assert not canary_errors, f"Canary worker errors: {canary_errors}"
            assert not benchmark_errors, (
                f"Benchmark worker encountered cross-test interference: {benchmark_errors}"
            )
            assert canary_cycles_completed[0] == 15, (
                f"Completed {canary_cycles_completed[0]}/15 canary cycles"
            )
            assert benchmark_query_counts["queries"] > 0, "Benchmark worker executed 0 queries"
            assert benchmark_query_counts["visible_files"] == 300
            assert benchmark_query_counts["eshop_files"] == 75

        finally:
            with psycopg.connect(st.db_admin_url) as cleanup_conn:
                teardown_synthetic_benchmark_repo(cleanup_conn, session_domain)

    def test_unscoped_flawed_teardown_cross_test_interference_demonstration(
        self,
        postgres_admin_conn: psycopg.Connection,
    ) -> None:
        """Counter-proof: Demonstrates that if flawed_teardown used a non-private shared
        repo (e.g. 'dotnet/eShop'), it WOULD delete benchmark rows and corrupt concurrent
        benchmark queries. This empirically proves that private repo scoping
        (canary_repo_<uuid>) is strictly necessary.
        """
        st = ScoutSettings()
        session_domain = f"eval_synthetic_{uuid.uuid4().hex[:12]}"

        # Seed benchmark dataset
        with psycopg.connect(st.db_admin_url) as seed_conn:
            seed_synthetic_benchmark_repo(seed_conn, session_domain)

        canary_domain = f"canary_verification_domain_{uuid.uuid4().hex[:8]}"
        canary_fid = uuid.uuid4()
        unscoped_repo = "dotnet/eShop"  # UNSCOPED: shared with benchmark

        try:
            with psycopg.connect(st.db_admin_url) as c_conn:
                c_conn.execute(
                    """
                    INSERT INTO domain_packs (
                        domain_id, display_name, lora_adapter_id,
                        primary_truth_source, freshness_half_life_days, config_json
                    )
                    VALUES (%s, 'Unscoped Canary Domain', 'none', 'code', 30, '{}')
                    ON CONFLICT (domain_id) DO NOTHING;
                    """,
                    (canary_domain,),
                )
                pg_repo = PostgresRLSRepository()
                pg_repo.insert_code_file(
                    c_conn,
                    CodeFileRecord(
                        file_id=canary_fid,
                        domain_id=canary_domain,
                        repo_name=unscoped_repo,
                        file_path="src/unscoped_canary.cs",
                        git_commit_sha="3333444455556666777788889999000011112222",
                        git_blob_sha="4444555566667777888899990000111122223333",
                        language="csharp",
                        allowed_scopes=["scope:canary"],
                        raw_content="// Unscoped canary",
                        line_count=1,
                    ),
                )
                c_conn.commit()

                # Prior to flawed teardown: 75 benchmark files + 1 canary file = 76
                with c_conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM code_files WHERE repo_name = %s;",
                        (unscoped_repo,),
                    )
                    before_row = cur.fetchone()
                    assert before_row is not None
                    cnt_before = before_row[0]
                    assert cnt_before == 76

                # Flawed teardown deleting by unscoped repo_name
                c_conn.execute("DELETE FROM code_files WHERE repo_name = %s;", (unscoped_repo,))
                c_conn.commit()

                # After flawed teardown: all 75 benchmark files in session_domain are deleted
                with c_conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM code_files WHERE domain_id = %s AND repo_name = %s;",
                        (session_domain, unscoped_repo),
                    )
                    after_row = cur.fetchone()
                    assert after_row is not None
                    cnt_benchmark_after = after_row[0]
                    # Proves that unscoped flawed teardown wipes out benchmark rows!
                    assert cnt_benchmark_after == 0, (
                        "Expected unscoped flawed teardown to wipe out benchmark files, "
                        f"but {cnt_benchmark_after} remain"
                    )
        finally:
            with psycopg.connect(st.db_admin_url) as cleanup_conn:
                with contextlib.suppress(Exception):
                    cleanup_conn.rollback()
                cleanup_conn.execute(
                    "DELETE FROM code_files WHERE domain_id = %s;", (canary_domain,)
                )
                cleanup_conn.execute(
                    "DELETE FROM domain_packs WHERE domain_id = %s;", (canary_domain,)
                )
                cleanup_conn.commit()
                teardown_synthetic_benchmark_repo(cleanup_conn, session_domain)


class TestTransactionRollbackUnderAbortFaultInjection:
    """Stress-test transaction rollback behavior under SQL error during canary setup."""

    def test_sql_error_during_setup_recovers_gracefully_in_finally_via_rollback(
        self,
        postgres_admin_conn: psycopg.Connection,
    ) -> None:
        """Injects an intentional SQL syntax error inside try: during canary setup,
        and proves that finally: with conn.rollback() successfully cleans up
        without raising InFailedSqlTransaction or leaving residual rows.
        """
        st = ScoutSettings()
        canary_domain = f"canary_verification_domain_{uuid.uuid4().hex[:8]}"

        with psycopg.connect(st.db_admin_url) as conn:
            # Insert domain pack
            conn.execute(
                """
                INSERT INTO domain_packs (
                    domain_id, display_name, lora_adapter_id,
                    primary_truth_source, freshness_half_life_days, config_json
                )
                VALUES (%s, 'Abort Canary Domain', 'none', 'code', 30, '{}')
                ON CONFLICT (domain_id) DO NOTHING;
                """,
                (canary_domain,),
            )
            # Intentionally NOT committing yet so transaction is active

            # Fault injection: execute intentional SQL syntax error inside try:
            error_caught = False
            try:
                conn.execute("SELECT * FROM non_existent_table WHERE SYNTAX ERROR !!!")
            except PgSyntaxError:
                error_caught = True
            finally:
                # The b93a2db fix: rollback aborted transaction before executing cleanup
                with contextlib.suppress(Exception):
                    conn.rollback()

                # Cleanup commands must execute successfully without InFailedSqlTransaction
                conn.execute("DELETE FROM code_files WHERE domain_id = %s;", (canary_domain,))
                conn.execute("DELETE FROM domain_packs WHERE domain_id = %s;", (canary_domain,))
                conn.commit()

            assert error_caught is True, "Intentional SQL syntax error was not triggered"

            # Assert 0 residual rows exist for canary_domain
            with conn.cursor() as cur:
                for tbl in ("code_files", "ast_symbols", "doc_chunks", "domain_packs"):
                    cur.execute(
                        f"SELECT COUNT(*) FROM {tbl} WHERE domain_id = %s;",
                        (canary_domain,),
                    )
                    row = cur.fetchone()
                    assert row is not None
                    cnt = row[0]
                    assert cnt == 0, f"Residual rows found in {tbl} for {canary_domain}: {cnt}"

    def test_omitting_rollback_on_aborted_transaction_raises_in_failed_sql_transaction(
        self,
        postgres_admin_conn: psycopg.Connection,
    ) -> None:
        """Counter-proof: Demonstrates that without conn.rollback(), attempting to execute
        cleanup queries on an aborted transaction raises InFailedSqlTransaction.
        This proves the necessity of the rollback in finally:.
        """
        st = ScoutSettings()
        canary_domain = f"canary_verification_domain_{uuid.uuid4().hex[:8]}"

        with psycopg.connect(st.db_admin_url) as conn:
            with contextlib.suppress(PgSyntaxError):
                conn.execute("SELECT SYNTAX ERROR !!!")

            # Attempting to execute cleanup WITHOUT conn.rollback() MUST raise
            # InFailedSqlTransaction
            with pytest.raises(InFailedSqlTransaction):
                conn.execute("DELETE FROM domain_packs WHERE domain_id = %s;", (canary_domain,))

            # Now perform rollback to cleanly exit
            conn.rollback()


class TestDatabaseZeroResidualRowAudit:
    """Verifies that no lingering eval_synthetic_% or canary_% rows remain across all tables."""

    def test_database_zero_residual_rows_matching_eval_synthetic_and_canary(
        self,
        postgres_admin_conn: psycopg.Connection,
    ) -> None:
        """Audits live PostgreSQL database for residual test rows across 4 tables."""
        conn = postgres_admin_conn
        with conn.cursor() as cur:
            tables = ("code_files", "ast_symbols", "doc_chunks", "domain_packs")
            residual_counts = {}

            for tbl in tables:
                cur.execute(
                    f"SELECT COUNT(*) FROM {tbl} "
                    "WHERE domain_id LIKE 'eval_synthetic_%' OR domain_id LIKE 'canary_%';"
                )
                row = cur.fetchone()
                assert row is not None
                cnt = row[0]
                residual_counts[tbl] = cnt

            for tbl, cnt in residual_counts.items():
                assert cnt == 0, (
                    f"Zero residual row audit FAILED: {tbl} contains {cnt} lingering rows "
                    f"matching 'eval_synthetic_%' or 'canary_%'!"
                )
