"""Canary and guard tests for ephemeral PostgreSQL domain fixtures."""

from __future__ import annotations

import uuid
from collections.abc import Callable

import psycopg
import pytest
from tests.fixtures.pg_isolation import (
    ALL_RLS_TABLES_FK_DELETE_ORDER,
    cleanup_ephemeral_pg_domain,
    create_ephemeral_pg_domain,
    make_ephemeral_pg_domain_id,
)

from context_scout.config import ScoutSettings

pytestmark = pytest.mark.postgres


def test_all_rls_tables_fk_delete_order_matches_pg17_catalog() -> None:
    """ALL_RLS_TABLES_FK_DELETE_ORDER must cover 100% of public PG17 application tables."""
    settings = ScoutSettings()
    with psycopg.connect(settings.db_admin_url) as conn:
        rows = conn.execute(
            """
            SELECT c.relname
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
              AND c.relkind = 'r'
              AND c.relname <> 'alembic_version'
            ORDER BY c.relname
            """
        ).fetchall()
    pg_tables = {str(r[0]) for r in rows}
    assert len(ALL_RLS_TABLES_FK_DELETE_ORDER) == 23
    assert set(ALL_RLS_TABLES_FK_DELETE_ORDER) == pg_tables


def test_cleanup_refuses_non_ephemeral_domains() -> None:
    """Cleanup guard must fail fast on protected or non-eval_ domain identifiers."""
    settings = ScoutSettings()
    with psycopg.connect(settings.db_admin_url) as conn:
        for forbidden in (
            "code_architecture_scout",
            "enterprise_docs_scout",
            "polyglot_unseen_v2",
            "eval_",
            "",
            "prod_tenant",
        ):
            with pytest.raises(ValueError, match="non-ephemeral"):
                cleanup_ephemeral_pg_domain(conn, forbidden)


def test_ephemeral_pg_domain_canary_isolation(ephemeral_pg_domain: str) -> None:
    """Cleaning one eval_ domain must remove its rows without touching canary rows."""
    assert ephemeral_pg_domain.startswith("eval_")
    settings = ScoutSettings()
    canary_domain = make_ephemeral_pg_domain_id("canary_keep")
    session_file_id = str(uuid.uuid4())
    canary_file_id = str(uuid.uuid4())
    session_chunk_id = str(uuid.uuid4())
    canary_chunk_id = str(uuid.uuid4())

    with psycopg.connect(settings.db_admin_url) as conn:
        create_ephemeral_pg_domain(conn, domain_id=canary_domain)
        try:
            for dom_id, f_id, c_id in (
                (ephemeral_pg_domain, session_file_id, session_chunk_id),
                (canary_domain, canary_file_id, canary_chunk_id),
            ):
                conn.execute(
                    """
                    INSERT INTO code_files
                        (file_id, domain_id, repo_name, file_path, git_commit_sha, git_blob_sha,
                         language, allowed_scopes, raw_content, line_count)
                    VALUES (%s::uuid, %s, %s, %s, 'deadbeef', 'blob1234',
                            'python', %s, 'def f(): pass\n', 1)
                    """,
                    (f_id, dom_id, f"{dom_id}/repo", "src/mod.py", [f"domain:{dom_id}"]),
                )
                conn.execute(
                    """
                    INSERT INTO doc_chunks
                        (chunk_id, domain_id, doc_id, doc_title, source_uri, market_scope,
                         section_heading, contextual_header, verbatim_content, allowed_scopes,
                         embedding)
                    VALUES (%s::uuid, %s, 'doc-1', 'Canary Doc', 'docs/a.md', 'GLOBAL',
                            'Intro', 'Header', 'body text', %s, %s)
                    """,
                    (c_id, dom_id, [f"domain:{dom_id}"], "[" + ",".join(["0.05"] * 768) + "]"),
                )
            conn.commit()

            deleted = cleanup_ephemeral_pg_domain(conn, ephemeral_pg_domain)
            assert deleted["code_files"] == 1
            assert deleted["doc_chunks"] == 1
            assert deleted["domain_packs"] == 1

            canary_files = conn.execute(
                "SELECT COUNT(*) FROM code_files WHERE domain_id = %s",
                (canary_domain,),
            ).fetchone()
            canary_chunks = conn.execute(
                "SELECT COUNT(*) FROM doc_chunks WHERE domain_id = %s",
                (canary_domain,),
            ).fetchone()
            assert canary_files is not None and canary_files[0] == 1
            assert canary_chunks is not None and canary_chunks[0] == 1
        finally:
            conn.rollback()
            cleanup_ephemeral_pg_domain(conn, canary_domain)


def test_ephemeral_pg_domain_factory_registers_distinct_domains(
    ephemeral_pg_domain_factory: Callable[[str], str],
) -> None:
    """Factory creates distinct eval_* domain_packs rows ready for multi-tenant tests."""
    tenant_a = ephemeral_pg_domain_factory("alpha")
    tenant_b = ephemeral_pg_domain_factory("beta")
    assert tenant_a != tenant_b
    assert tenant_a.startswith("eval_")
    assert tenant_b.startswith("eval_")

    settings = ScoutSettings()
    with psycopg.connect(settings.db_admin_url) as conn:
        rows = conn.execute(
            "SELECT domain_id FROM domain_packs WHERE domain_id = ANY(%s) ORDER BY domain_id",
            ([tenant_a, tenant_b],),
        ).fetchall()
        assert {row[0] for row in rows} == {tenant_a, tenant_b}


def test_parallel_bootstrap_preserves_active_eval_domain_under_advisory_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under SCOUT_PARALLEL_PYTEST=1, bootstrap never drops active eval_* domains."""
    from tests.fixtures.pg_isolation import ensure_pg17_schema_and_seed_corpora

    settings = ScoutSettings()
    active_eval = make_ephemeral_pg_domain_id("parallel_active")
    monkeypatch.setenv("SCOUT_PARALLEL_PYTEST", "1")
    with psycopg.connect(settings.db_admin_url) as conn:
        create_ephemeral_pg_domain(conn, domain_id=active_eval)
    try:
        ensure_pg17_schema_and_seed_corpora(
            db_admin_url=settings.db_admin_url,
            db_app_url=settings.db_app_url,
            repo_root=__import__("pathlib").Path(__file__).resolve().parents[1],
        )
        with psycopg.connect(settings.db_admin_url) as conn:
            row = conn.execute(
                "SELECT 1 FROM domain_packs WHERE domain_id = %s",
                (active_eval,),
            ).fetchone()
            assert row is not None, "Active eval_* domain was deleted under SCOUT_PARALLEL_PYTEST=1"
    finally:
        with psycopg.connect(settings.db_admin_url) as conn:
            cleanup_ephemeral_pg_domain(conn, active_eval)
