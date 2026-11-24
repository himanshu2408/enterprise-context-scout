"""M1-01 & M1-02 live verification: Alembic+squawk, 23-table FORCE RLS, pools, PG+S3 registry."""

from __future__ import annotations

import ast
import datetime
import hashlib
import io
import os
import stat
import uuid
from pathlib import Path

import boto3
import psycopg
import pytest
from botocore.stub import Stubber

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.preflight_gate import (
    CUSTOM_CORPUS_INDEX_PATHS,
    BoundGraphRepository,
    read_custom_sync_manifest,
    sync_custom_domain_corpus_if_needed,
)
from context_scout.config import (
    INSECURE_DB_PASSWORDS,
    ScoutSettings,
    validate_db_app_url,
)
from context_scout.db.migrations_runner import (
    run_alembic_upgrade,
    verify_migrations_with_squawk,
)
from context_scout.db.repositories import PostgresRLSRepository
from context_scout.db.session import (
    PoolRole,
    get_ingestion_pool,
    get_query_pool,
)
from context_scout.retrieval.embedder import get_embedding_provider
from context_scout.schemas.contracts import (
    DEFAULT_TENANT_ID,
    AuthorityRules,
    DomainPackManifest,
    PrincipalContext,
    RoleType,
)
from context_scout.storage.object_store import (
    InMemoryObjectStore,
    S3CompatibleObjectStore,
    build_tenant_domain_key,
    validate_object_key,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_PG_TABLES: tuple[str, ...] = (
    "domain_packs",
    "domain_sync_manifests",
    "embedding_spaces",
    "connector_sync_watermarks",
    "repositories",
    "repo_commits",
    "ref_snapshots",
    "overlay_file_deltas",
    "side_effects",
    "code_files",
    "ast_symbols",
    "ast_edges",
    "doc_chunks",
    "confluence_pages",
    "jira_issues",
    "execution_telemetry",
    "indexing_queue",
    "indexing_dead_letter",
    "scout_sessions",
    "model_artifacts",
    "audit_events",
    "scim_users",
    "scim_groups",
)

DELETE_ORDER_PG_TABLES: tuple[str, ...] = (
    "audit_events",
    "scim_groups",
    "scim_users",
    "model_artifacts",
    "scout_sessions",
    "overlay_file_deltas",
    "ref_snapshots",
    "side_effects",
    "ast_edges",
    "ast_symbols",
    "code_files",
    "repo_commits",
    "repositories",
    "confluence_pages",
    "jira_issues",
    "doc_chunks",
    "execution_telemetry",
    "indexing_dead_letter",
    "indexing_queue",
    "connector_sync_watermarks",
    "embedding_spaces",
    "domain_sync_manifests",
    "domain_packs",
)


def _format_vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{val:.8f}" for val in vec) + "]"


def _cleanup_tenant_domains(
    admin_db_url: str,
    tenant_ids: tuple[uuid.UUID, ...],
    domain_id: str,
) -> None:
    """Deletes synthetic M1 test rows across all 23 tables in strict child-to-parent order."""
    with psycopg.connect(admin_db_url) as conn:
        for table in DELETE_ORDER_PG_TABLES:
            if table == "repo_commits":
                conn.execute(
                    "DELETE FROM repo_commits WHERE tenant_id = ANY(%s);",
                    (list(tenant_ids),),
                )
            else:
                conn.execute(
                    f"DELETE FROM {table} WHERE tenant_id = ANY(%s) AND domain_id = %s;",
                    (list(tenant_ids), domain_id),
                )
        conn.commit()


def test_m1_squawk_zero_violations_and_idempotent_upgrade() -> None:
    """Verifies squawk PG17 passes all M1 migrations with zero squawk-ignore bypasses."""
    linted_revs = verify_migrations_with_squawk()
    assert linted_revs == [
        "0002_m1_add_tenant",
        "0003_m1_validate_nn",
        "0004_m1_conc_idx",
        "0005_m1_swap_keys",
        "0006_m1_force_rls",
        "0007_m2_cow_tables",
        "0008_m2_conc_idx",
        "0009_m2_swap_checks",
        "0010_m2_force_rls",
        "0011_m2_side_effects",
        "0012_m2_se_conc_idx",
        "0013_m2_se_swap_chk",
        "0014_m2_se_force_rls",
        "0015_m3_scout_sessions",
        "0016_m3_sess_conc_idx",
        "0017_m3_sess_force_rls",
        "0018_m4_model_artifacts",
        "0019_m4_art_conc_idx",
        "0020_m4_art_force_rls",
        "0021_m5_audit_scim",
        "0022_m5_audit_conc_idx",
        "0023_m5_audit_force_rls",
    ]

    scan_dirs = (
        REPO_ROOT / "src" / "context_scout" / "db" / "migrations",
        REPO_ROOT / "sql",
    )
    for scan_dir in scan_dirs:
        for path in scan_dir.rglob("*.py"):
            txt = path.read_text(encoding="utf-8")
            assert "squawk-ignore" not in txt.lower(), f"Prohibited squawk-ignore in {path}"
        for path in scan_dir.rglob("*.sql"):
            txt = path.read_text(encoding="utf-8")
            assert "squawk-ignore" not in txt.lower(), f"Prohibited squawk-ignore in {path}"
            assert "scout_local_dev_pw" not in txt, f"Hardcoded password literal in {path}"

    for src_py in (REPO_ROOT / "src").rglob("*.py"):
        if src_py.name == "config.py":
            continue
        assert "scout_local_dev_pw" not in src_py.read_text(encoding="utf-8"), (
            f"Hardcoded password literal in {src_py}"
        )

    admin_url = ScoutSettings().db_admin_url
    run_alembic_upgrade(admin_url)
    run_alembic_upgrade(admin_url)


def test_m1_all_pg_tables_force_rls_and_fail_closed_unset_tenant() -> None:
    """Proves 23/23 public tables enforce PostgreSQL 17 FORCE RLS + fail-closed unset tenant."""
    settings = ScoutSettings()
    with psycopg.connect(settings.db_admin_url) as admin_conn:
        cat_rows = admin_conn.execute(
            """
            SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity,
                   COALESCE(a.attnotnull, FALSE) AS tenant_nn,
                   COALESCE(format_type(a.atttypid, a.atttypmod), '') AS tenant_type,
                   (SELECT COUNT(*) FROM pg_policy p WHERE p.polrelid = c.oid) AS pol_cnt
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            LEFT JOIN pg_attribute a
              ON a.attrelid = c.oid AND a.attname = 'tenant_id' AND NOT a.attisdropped
            WHERE n.nspname = 'public'
              AND c.relkind = 'r'
              AND c.relname <> 'alembic_version'
            ORDER BY c.relname;
            """
        ).fetchall()
        role_row = admin_conn.execute(
            """
            SELECT rolsuper, rolbypassrls, rolinherit, rolcreatedb, rolcreaterole
            FROM pg_roles
            WHERE rolname = 'scout_app_user';
            """
        ).fetchone()

    assert {str(r[0]) for r in cat_rows} == set(ALL_PG_TABLES)
    for relname, relrowsecurity, relforcerowsecurity, tenant_nn, tenant_type, pol_cnt in cat_rows:
        assert relrowsecurity is True, f"{relname} missing ENABLE ROW LEVEL SECURITY"
        assert relforcerowsecurity is True, f"{relname} missing FORCE ROW LEVEL SECURITY"
        assert tenant_nn is True, f"{relname}.tenant_id must be NOT NULL"
        assert tenant_type == "uuid", f"{relname}.tenant_id must be uuid, got {tenant_type}"
        assert int(str(pol_cnt)) >= 1, f"{relname} missing RLS policy in pg_policy"

    assert role_row == (False, False, False, False, False), (
        f"scout_app_user must have NOBYPASSRLS NOINHERIT minimal privileges, got {role_row}"
    )

    shared_domain = f"eval_m1_unset_{uuid.uuid4().hex[:8]}"
    try:
        with psycopg.connect(settings.db_admin_url) as admin_conn:
            admin_conn.execute(
                """
                INSERT INTO domain_packs (
                    tenant_id, domain_id, display_name, lora_adapter_id,
                    primary_truth_source, config_json
                )
                VALUES (%s, %s, 'Unset Guard Domain', 'none', 'code', '{}'::jsonb)
                """,
                (DEFAULT_TENANT_ID, shared_domain),
            )
            admin_conn.commit()

        with psycopg.connect(settings.db_app_url) as app_conn:
            with app_conn.transaction():
                app_conn.execute("RESET app.tenant_id;")
                app_conn.execute(
                    "SELECT set_config('app.principal_scopes', 'public,public_eng', true);"
                )
                for table in ALL_PG_TABLES:
                    cnt_row = app_conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()
                    assert cnt_row == (0,), (
                        f"Table {table} leaked {cnt_row} rows when app.tenant_id was unset!"
                    )

            with (
                pytest.raises(
                    (
                        psycopg.errors.InsufficientPrivilege,
                        psycopg.errors.InvalidTextRepresentation,
                    )
                ),
                app_conn.transaction(),
            ):
                app_conn.execute("RESET app.tenant_id;")
                app_conn.execute(
                    """
                    INSERT INTO domain_packs (
                        tenant_id, domain_id, display_name, lora_adapter_id,
                        primary_truth_source, config_json
                    )
                    VALUES (%s, %s, 'Unauthorized Insert', 'none', 'code', '{}'::jsonb)
                    """,
                    (DEFAULT_TENANT_ID, f"eval_m1_unauth_{uuid.uuid4().hex[:8]}"),
                )
    finally:
        _cleanup_tenant_domains(settings.db_admin_url, (DEFAULT_TENANT_ID,), shared_domain)


def test_m1_cross_tenant_isolation_all_pg_tables_and_shared_domain_slug(
    tmp_path: Path,
) -> None:
    """Isolates two tenants with identical domain_id across all 23 PG17 FORCE RLS tables."""
    settings = ScoutSettings()
    query_mgr = get_query_pool(settings)
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    shared_domain = f"eval_m1_shared_{uuid.uuid4().hex[:8]}"
    shared_sync_domain = f"eval_m1_sync_{uuid.uuid4().hex[:8]}"

    scopes_a = ["public", "public_eng", f"domain:{shared_domain}"]
    scopes_b = ["public", "public_eng", f"domain:{shared_domain}"]
    principal_a = PrincipalContext(
        user_id="user_tenant_a",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes_a,
        tenant_id=tenant_a,
    )
    principal_b = PrincipalContext(
        user_id="user_tenant_b",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes_b,
        tenant_id=tenant_b,
    )

    now_dt = datetime.datetime.now(datetime.UTC)
    ids_by_tenant: dict[uuid.UUID, dict[str, uuid.UUID]] = {}
    vec_literal = _format_vector_literal(
        get_embedding_provider().embed_text("tenant isolation vector", is_query=False)
    )

    try:
        for tid, principal, label in (
            (tenant_a, principal_a, "tenant_a_secret_marker"),
            (tenant_b, principal_b, "tenant_b_secret_marker"),
        ):
            repo_id = uuid.uuid4()
            commit_id = uuid.uuid4()
            file_id = uuid.uuid4()
            sym_id = uuid.uuid4()
            edge_id = uuid.uuid4()
            chunk_id = uuid.uuid4()
            page_id = uuid.uuid4()
            issue_id = uuid.uuid4()
            trace_id = uuid.uuid4()
            task_id = uuid.uuid4()
            dlq_id = uuid.uuid4()
            snap_id = uuid.uuid4()
            delta_id = uuid.uuid4()
            side_effect_id = uuid.uuid4()
            ids_by_tenant[tid] = {
                "repo_id": repo_id,
                "commit_id": commit_id,
                "file_id": file_id,
                "sym_id": sym_id,
                "edge_id": edge_id,
                "chunk_id": chunk_id,
                "page_id": page_id,
                "issue_id": issue_id,
                "trace_id": trace_id,
                "task_id": task_id,
                "dlq_id": dlq_id,
                "snap_id": snap_id,
                "delta_id": delta_id,
                "side_effect_id": side_effect_id,
            }

            with query_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
                conn.execute(
                    """
                    INSERT INTO domain_packs (
                        tenant_id, domain_id, display_name, lora_adapter_id,
                        primary_truth_source, config_json
                    )
                    VALUES (%s, %s, %s, 'none', 'code', '{}'::jsonb)
                    """,
                    (tid, shared_domain, label),
                )
                conn.execute(
                    """
                    INSERT INTO domain_sync_manifests (tenant_id, domain_id, kind, source_dir)
                    VALUES (%s, %s, 'code', %s)
                    """,
                    (tid, shared_domain, f"/repos/{label}"),
                )
                conn.execute(
                    """
                    INSERT INTO embedding_spaces (
                        tenant_id, domain_id, space_id, model_id, provider,
                        dimensions, quantization, status, is_active
                    )
                    VALUES (
                        %s, %s, 'embeddinggemma-2-768', 'google/embeddinggemma-2',
                        'sentence-transformers', 768, 'halfvec', 'active', TRUE
                    )
                    """,
                    (tid, shared_domain),
                )
                conn.execute(
                    """
                    INSERT INTO repositories (
                        repo_id, tenant_id, domain_id, repo_name, source_type,
                        remote_url, default_branch, head_commit_sha, allowed_scopes
                    )
                    VALUES (
                        %s, %s, %s, 'shared-repo', 'github', %s, 'main',
                        'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', %s
                    )
                    """,
                    (repo_id, tid, shared_domain, f"https://example.invalid/{label}", scopes_a),
                )
                conn.execute(
                    """
                    INSERT INTO repo_commits (
                        commit_id, tenant_id, repo_id, commit_sha, author_name,
                        author_email, commit_message, committed_at
                    )
                    VALUES (
                        %s, %s, %s, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                        'Dev', 'dev@example.invalid', %s, %s
                    )
                    """,
                    (commit_id, tid, repo_id, label, now_dt),
                )
                conn.execute(
                    """
                    INSERT INTO code_files (
                        file_id, tenant_id, domain_id, repo_id, repo_name, file_path,
                        git_commit_sha, git_blob_sha, language, allowed_scopes,
                        raw_content, line_count
                    )
                    VALUES (
                        %s, %s, %s, %s, 'shared-repo', 'src/app.py',
                        '1', '2', 'python', %s, %s, 1
                    )
                    """,
                    (file_id, tid, shared_domain, repo_id, scopes_a, f"# {label}\n"),
                )
                conn.execute(
                    """
                    INSERT INTO ast_symbols (
                        symbol_id, tenant_id, file_id, domain_id, repo_id, qualified_name,
                        symbol_kind, start_line, end_line, signature_skeleton,
                        contextual_summary, allowed_scopes, embedding
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, 'function', 1, 2,
                        'def f():', %s, %s, %s::vector
                    )
                    """,
                    (
                        sym_id,
                        tid,
                        file_id,
                        shared_domain,
                        repo_id,
                        f"mod.{label}",
                        label,
                        scopes_a,
                        vec_literal,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO ast_edges (
                        edge_id, tenant_id, domain_id, source_symbol_id,
                        target_qualified_name, target_symbol_id, edge_kind, allowed_scopes
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, 'CALLS', %s)
                    """,
                    (edge_id, tid, shared_domain, sym_id, label, sym_id, scopes_a),
                )
                conn.execute(
                    """
                    INSERT INTO doc_chunks (
                        chunk_id, tenant_id, domain_id, doc_id, doc_title, source_uri,
                        market_scope, authority_weight, section_heading,
                        contextual_header, verbatim_content, allowed_scopes, embedding
                    )
                    VALUES (
                        %s, %s, %s, 'doc-1', %s, 'urn:doc:1', 'ENTERPRISE_DOC',
                        1.0, 'Sec', 'Hdr', %s, %s, %s::vector
                    )
                    """,
                    (chunk_id, tid, shared_domain, label, label, scopes_a, vec_literal),
                )
                conn.execute(
                    """
                    INSERT INTO confluence_pages (
                        page_id, tenant_id, domain_id, confluence_page_id, space_key,
                        title, version_number, author_username, source_url,
                        allowed_scopes, last_modified_at
                    )
                    VALUES (
                        %s, %s, %s, 'page-1', 'ENG', %s, 1, 'alice',
                        'https://example.invalid/wiki', %s, %s
                    )
                    """,
                    (page_id, tid, shared_domain, label, scopes_a, now_dt),
                )
                conn.execute(
                    """
                    INSERT INTO jira_issues (
                        issue_id, tenant_id, domain_id, issue_key, project_key,
                        summary, issue_type, status, source_url, allowed_scopes, updated_at
                    )
                    VALUES (
                        %s, %s, %s, 'ENG-1', 'ENG', %s, 'Task', 'Open',
                        'https://example.invalid/jira', %s, %s
                    )
                    """,
                    (issue_id, tid, shared_domain, label, scopes_a, now_dt),
                )
                conn.execute(
                    """
                    INSERT INTO execution_telemetry (
                        trace_id, tenant_id, domain_id, principal_id, principal_role,
                        execution_mode, local_tokens_read, context_pack_tokens_sent,
                        paid_tokens_saved, estimated_usd_saved,
                        preflight_passed_first_try, latency_ms
                    )
                    VALUES (%s, %s, %s, %s, 'engineer', 'chat_qa', 100, 50, 50, 0.001, TRUE, 12)
                    """,
                    (trace_id, tid, shared_domain, principal.user_id),
                )
                conn.execute(
                    """
                    INSERT INTO indexing_queue (
                        task_id, tenant_id, domain_id, event_type, payload_json, status
                    )
                    VALUES (%s, %s, %s, 'github_push', %s::jsonb, 'PENDING')
                    """,
                    (task_id, tid, shared_domain, f'{{"marker": "{label}"}}'),
                )
                conn.execute(
                    """
                    INSERT INTO indexing_dead_letter (
                        dlq_id, tenant_id, task_id, domain_id, event_type,
                        payload_json, failure_reason, stack_trace, attempts
                    )
                    VALUES (%s, %s, %s, %s, 'github_push', '{}'::jsonb, %s, 'trace', 3)
                    """,
                    (dlq_id, tid, task_id, shared_domain, label),
                )
                conn.execute(
                    """
                    INSERT INTO ref_snapshots (
                        snapshot_id, tenant_id, domain_id, repo_id, repo_name, ref_name,
                        ref_kind, mr_iid, base_commit_sha, head_commit_sha, state, allowed_scopes
                    )
                    VALUES (
                        %s, %s, %s, %s, 'shared-repo', %s, 'merge_request', '101',
                        'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                        'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', 'open', %s
                    )
                    """,
                    (snap_id, tid, shared_domain, repo_id, f"refs/mr/{label}", scopes_a),
                )
                conn.execute(
                    """
                    INSERT INTO overlay_file_deltas (
                        delta_id, tenant_id, snapshot_id, domain_id, repo_name, file_path,
                        change_type, overlay_blob_sha, language, raw_content, line_count,
                        allowed_scopes
                    )
                    VALUES (
                        %s, %s, %s, %s, 'shared-repo', 'src/overlay.py',
                        'added', 'cccccccccccccccccccccccccccccccccccccccc', 'python',
                        %s, 1, %s
                    )
                    """,
                    (delta_id, tid, snap_id, shared_domain, f"# {label}\n", scopes_a),
                )
                conn.execute(
                    """
                    INSERT INTO side_effects (
                        side_effect_id, tenant_id, domain_id, repo_name, file_id, symbol_id,
                        file_path, qualified_symbol, start_line, end_line,
                        effect_category, target_identifier, provider_or_driver, allowed_scopes
                    )
                    VALUES (
                        %s, %s, %s, 'shared-repo', %s, %s, 'src/service.py', 'service.notify',
                        1, 5, 'webhook', %s, 'httpx', %s
                    )
                    """,
                    (
                        side_effect_id,
                        tid,
                        shared_domain,
                        file_id,
                        sym_id,
                        f"https://hooks.internal/{label}",
                        scopes_a,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO connector_sync_watermarks (
                        tenant_id, domain_id, source_id, container_key,
                        watermark_cursor, sync_tier, items_synced
                    )
                    VALUES (%s, %s, 'gitlab', %s, 'sha-001', 'watermark', 1)
                    """,
                    (tid, shared_domain, f"org/{label}"),
                )
                conn.execute(
                    """
                    INSERT INTO scout_sessions (
                        tenant_id, session_id, domain_id, principal_id, allowed_scopes,
                        turn_count, compacted_summary
                    )
                    VALUES (%s, %s, %s, %s, %s, 1, %s)
                    """,
                    (tid, f"sess-{label}", shared_domain, f"user-{label}", scopes_a, label),
                )
                conn.execute(
                    """
                    INSERT INTO model_artifacts (
                        tenant_id, domain_id, artifact_id, artifact_kind,
                        base_model_id, base_model_revision_sha256,
                        storage_uri, weights_object_key, config_object_key,
                        weights_sha256, training_scope_ceiling, allowed_scopes,
                        lora_rank, lora_alpha, use_rslora, use_dora, status, is_active
                    )
                    VALUES (
                        %s, %s, %s, 'lora_adapter',
                        'google/gemma-4-26B-A4B-it', %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        16, 32, TRUE, FALSE, 'active', TRUE
                    )
                    """,
                    (
                        tid,
                        shared_domain,
                        f"lora-{label}",
                        "a" * 64,
                        f"gs://scout-models/tenants/{tid}/domains/{shared_domain}/adapters/lora-{label}",
                        f"tenants/{tid}/domains/{shared_domain}/adapters/lora-{label}/adapter_model.safetensors",
                        f"tenants/{tid}/domains/{shared_domain}/adapters/lora-{label}/adapter_config.json",
                        "b" * 64,
                        scopes_a,
                        scopes_a,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO audit_events (
                        tenant_id, seq_no, event_id, domain_id, event_type, severity,
                        actor_id, actor_role, resource_uri, outcome,
                        prev_hash_sha256, event_hash_sha256
                    )
                    VALUES (
                        %s, 1, %s, %s, 'auth.session', 'INFO',
                        %s, 'engineer', '/api/v1/auth/me', 'allow',
                        %s, %s
                    )
                    """,
                    (
                        tid,
                        uuid.uuid4(),
                        shared_domain,
                        f"user-{label}",
                        "0" * 64,
                        "c" * 64,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO scim_users (
                        tenant_id, scim_user_id, domain_id, user_name,
                        email, display_name, active, custom_scopes
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, TRUE, %s)
                    """,
                    (
                        tid,
                        f"scim-u-{label}",
                        shared_domain,
                        f"user_{label}@example.invalid",
                        f"user_{label}@example.invalid",
                        label,
                        scopes_a,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO scim_groups (
                        tenant_id, scim_group_id, domain_id, display_name,
                        member_user_ids, mapped_scopes
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        tid,
                        f"scim-g-{label}",
                        shared_domain,
                        f"group-{label}",
                        [f"scim-u-{label}"],
                        scopes_a,
                    ),
                )

        for viewer_tid, other_tid, viewer_principal in (
            (tenant_a, tenant_b, principal_a),
            (tenant_b, tenant_a, principal_b),
        ):
            with query_mgr.scoped_transaction(
                viewer_principal, enable_hnsw_iterative_scan=False
            ) as conn:
                for table in ALL_PG_TABLES:
                    rows = conn.execute(
                        f"SELECT tenant_id FROM {table} WHERE tenant_id = ANY(%s);",
                        ([tenant_a, tenant_b],),
                    ).fetchall()
                    assert len(rows) == 1, (
                        f"{table}: expected 1 row for tenant {viewer_tid}, got {len(rows)}"
                    )
                    assert uuid.UUID(str(rows[0][0])) == viewer_tid

                upd_cur = conn.execute(
                    "UPDATE domain_packs SET display_name = 'PWNED' WHERE tenant_id = %s;",
                    (other_tid,),
                )
                assert upd_cur.rowcount == 0
                del_cur = conn.execute(
                    "DELETE FROM code_files WHERE file_id = %s;",
                    (ids_by_tenant[other_tid]["file_id"],),
                )
                assert del_cur.rowcount == 0

                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    conn.execute("SAVEPOINT sp_spoof;")
                    try:
                        conn.execute(
                            """
                            INSERT INTO indexing_queue (
                                task_id, tenant_id, domain_id, event_type, payload_json
                            )
                            VALUES (%s, %s, %s, 'github_push', '{}'::jsonb)
                            """,
                            (uuid.uuid4(), other_tid, shared_domain),
                        )
                    except Exception:
                        conn.execute("ROLLBACK TO SAVEPOINT sp_spoof;")
                        raise

        dir_a = tmp_path / "tenant_a_repo"
        dir_b = tmp_path / "tenant_b_repo"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "service.py").write_text(
            "def handle_tenant_a() -> str:\n    return 'A'\n", encoding="utf-8"
        )
        (dir_b / "service.py").write_text(
            "def handle_tenant_b() -> str:\n    return 'B'\n", encoding="utf-8"
        )

        sync_dom_a = tmp_path / "domains_a" / shared_sync_domain
        sync_dom_b = tmp_path / "domains_b" / shared_sync_domain
        sync_dom_a.mkdir(parents=True)
        sync_dom_b.mkdir(parents=True)
        for dom_dir, src_dir in ((sync_dom_a, dir_a), (sync_dom_b, dir_b)):
            (dom_dir / "domain.yaml").write_text(
                f"domain_id: {shared_sync_domain}\n"
                "display_name: Shared Sync Slug\n"
                "output_contract: ImplementationContextPack\n"
                f"source_dir: {src_dir}\n",
                encoding="utf-8",
            )

        res_a = sync_custom_domain_corpus_if_needed(
            shared_sync_domain,
            index_path=sync_dom_a / "sync_manifest.json",
            tenant_id=tenant_a,
        )
        res_b = sync_custom_domain_corpus_if_needed(
            shared_sync_domain,
            index_path=sync_dom_b / "sync_manifest.json",
            tenant_id=tenant_b,
        )
        assert res_a is not None and res_a["synced"] is True
        assert res_b is not None and res_b["synced"] is True

        bound = BoundGraphRepository(PostgresRLSRepository())
        sync_p_a = PrincipalContext(
            user_id="sync_a",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", f"domain:{shared_sync_domain}"],
            tenant_id=tenant_a,
        )
        sync_p_b = PrincipalContext(
            user_id="sync_b",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", f"domain:{shared_sync_domain}"],
            tenant_id=tenant_b,
        )
        files_a = bound.list_visible_code_files(sync_p_a, shared_sync_domain)
        files_b = bound.list_visible_code_files(sync_p_b, shared_sync_domain)
        assert len(files_a) == 1 and "handle_tenant_a" in files_a[0].raw_content
        assert "handle_tenant_b" not in files_a[0].raw_content
        assert len(files_b) == 1 and "handle_tenant_b" in files_b[0].raw_content
        assert "handle_tenant_a" not in files_b[0].raw_content

        man_a = read_custom_sync_manifest(shared_sync_domain, principal=sync_p_a)
        man_b = read_custom_sync_manifest(shared_sync_domain, principal=sync_p_b)
        assert man_a is not None and str(dir_a.resolve()) in str(man_a["source_dir"])
        assert man_b is not None and str(dir_b.resolve()) in str(man_b["source_dir"])
    finally:
        CUSTOM_CORPUS_INDEX_PATHS.pop(shared_sync_domain, None)
        _cleanup_tenant_domains(settings.db_admin_url, (tenant_a, tenant_b), shared_domain)
        _cleanup_tenant_domains(settings.db_admin_url, (tenant_a, tenant_b), shared_sync_domain)


def test_m1_query_vs_ingestion_pools_and_zero_runtime_admin_url() -> None:
    """Separate query_pool vs ingestion_pool timeouts and zero runtime db_admin_url."""
    settings = ScoutSettings()
    query_mgr = get_query_pool(settings)
    ingest_mgr = get_ingestion_pool(settings)

    assert query_mgr.pool_role == PoolRole.QUERY
    assert ingest_mgr.pool_role == PoolRole.INGESTION
    assert query_mgr.get_pool() is get_query_pool(settings).get_pool()
    assert query_mgr.get_pool() is not ingest_mgr.get_pool()

    principal = PrincipalContext(
        user_id="pool_verifier",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    with query_mgr.scoped_transaction(principal) as qconn:
        q_stmt = str((qconn.execute("SHOW statement_timeout;").fetchone() or ("",))[0])
        q_lock = str((qconn.execute("SHOW lock_timeout;").fetchone() or ("",))[0])
        q_idle = str(
            (qconn.execute("SHOW idle_in_transaction_session_timeout;").fetchone() or ("",))[0]
        )
        assert q_stmt == "5s"
        assert q_lock == "2s"
        assert q_idle == "10s"

    with (
        pytest.raises(psycopg.errors.QueryCanceled),
        query_mgr.scoped_transaction(principal) as qconn,
    ):
        qconn.execute("SELECT pg_sleep(5.5);")

    with ingest_mgr.scoped_transaction(principal) as iconn:
        i_stmt = str((iconn.execute("SHOW statement_timeout;").fetchone() or ("",))[0])
        i_lock = str((iconn.execute("SHOW lock_timeout;").fetchone() or ("",))[0])
        i_idle = str(
            (iconn.execute("SHOW idle_in_transaction_session_timeout;").fetchone() or ("",))[0]
        )
        assert i_stmt == "5min"
        assert i_lock == "10s"
        assert i_idle == "5min"

    runtime_paths = (
        REPO_ROOT / "src" / "context_scout" / "api" / "routes" / "telemetry.py",
        REPO_ROOT / "src" / "context_scout" / "tuning" / "teacher_distiller.py",
        REPO_ROOT / "src" / "context_scout" / "agents" / "preflight_gate.py",
    )
    for py_path in runtime_paths:
        src_text = py_path.read_text(encoding="utf-8")
        tree = ast.parse(src_text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != "db_admin_url", f"{py_path} must not access db_admin_url"
            if isinstance(node, ast.Name):
                assert node.id != "db_admin_url", f"{py_path} must not reference db_admin_url"
            if (
                isinstance(node, ast.keyword)
                and node.arg == "use_pool"
                and isinstance(node.value, ast.Constant)
                and node.value.value is False
            ):
                raise AssertionError(f"{py_path} must not pass use_pool=False")


def test_m1_stateless_domain_registry_and_s3_object_store(tmp_path: Path) -> None:
    """Read-only domains_dir + S3CompatibleObjectStore checks."""
    settings = ScoutSettings()
    tenant_id = uuid.uuid4()
    domain_id = f"eval_m1_s3_{uuid.uuid4().hex[:8]}"

    src_dir = tmp_path / "corpus_src"
    src_dir.mkdir()
    (src_dir / "worker.py").write_text("def run_job() -> int:\n    return 42\n", encoding="utf-8")

    ro_domains_dir = tmp_path / "ro_domains"
    ro_dom_dir = ro_domains_dir / domain_id
    ro_dom_dir.mkdir(parents=True)
    (ro_dom_dir / "domain.yaml").write_text(
        f"domain_id: {domain_id}\n"
        "display_name: Read Only Domain\n"
        "lora_adapter_id: gemma4_test_ro\n"
        f"lora_adapter_path: domains/{domain_id}/adapters/gemma4_test_ro\n"
        "output_contract: ImplementationContextPack\n"
        f"source_dir: {src_dir}\n",
        encoding="utf-8",
    )

    store = InMemoryObjectStore()
    registry = DomainRegistry(domains_dir=ro_domains_dir, object_store=store)
    principal = PrincipalContext(
        user_id="s3_tester",
        role=RoleType.ENGINEER,
        allowed_scopes=["public", "public_eng", f"domain:{domain_id}"],
        tenant_id=tenant_id,
    )
    manifest = DomainPackManifest(
        domain_id=domain_id,
        display_name="Read Only Domain",
        base_model_id="google/gemma-4-26B-A4B-it",
        lora_adapter_id="gemma4_test_ro",
        lora_adapter_path=f"domains/{domain_id}/adapters/gemma4_test_ro",
        mcp_read_tools_allowed=["code_repo_mcp:assemble_verified_context_pack"],
        authority_rules=AuthorityRules(primary_truth_source="code", freshness_half_life_days=90),
        output_contract="ImplementationContextPack",
    )

    try:
        os.chmod(ro_dom_dir, stat.S_IRUSR | stat.S_IXUSR)
        os.chmod(ro_domains_dir, stat.S_IRUSR | stat.S_IXUSR)

        meta = registry.register_persistent(manifest, principal=principal)
        expected_key = build_tenant_domain_key(tenant_id, domain_id, "domain.json")
        assert meta.uri == f"memory://local-inmemory/{expected_key}"

        reloaded_registry = DomainRegistry(
            domains_dir=None, object_store=store, tenant_id=tenant_id
        )
        loaded_map = reloaded_registry.load_all(principal=principal)
        assert domain_id in loaded_map

        sync_res = sync_custom_domain_corpus_if_needed(
            domain_id,
            index_path=ro_dom_dir / "sync_manifest.json",
            tenant_id=tenant_id,
        )
        assert sync_res is not None and sync_res["synced"] is True
        assert sorted(ro_domains_dir.rglob("sync_manifest.json")) == []
        assert read_custom_sync_manifest(domain_id, principal=principal) is not None
    finally:
        CUSTOM_CORPUS_INDEX_PATHS.pop(domain_id, None)
        if ro_domains_dir.exists():
            os.chmod(ro_domains_dir, stat.S_IRWXU)
        if ro_dom_dir.exists():
            os.chmod(ro_dom_dir, stat.S_IRWXU)
        _cleanup_tenant_domains(settings.db_admin_url, (tenant_id,), domain_id)

    raw_client = boto3.client(
        "s3",
        region_name="eu-central-1",
        aws_access_key_id="test_ak",
        aws_secret_access_key="test_sk",
    )
    stubber = Stubber(raw_client)
    s3_store = S3CompatibleObjectStore("context-scout-artifacts", client=raw_client)
    obj_key = build_tenant_domain_key(tenant_id, domain_id, "sync_manifest.json")
    payload_bytes = b'{"ok": true}'
    digest = hashlib.sha256(payload_bytes).hexdigest()

    stubber.add_response(
        "put_object",
        {"ETag": '"abc"'},
        {
            "Bucket": "context-scout-artifacts",
            "Key": obj_key,
            "Body": payload_bytes,
            "ContentType": "application/json",
            "Metadata": {"sha256": digest},
        },
    )
    stubber.add_response(
        "get_object",
        {"Body": io.BytesIO(payload_bytes), "Metadata": {"sha256": digest}},
        {"Bucket": "context-scout-artifacts", "Key": obj_key},
    )
    stubber.add_response(
        "get_object",
        {"Body": io.BytesIO(b'{"ok": false}'), "Metadata": {"sha256": digest}},
        {"Bucket": "context-scout-artifacts", "Key": obj_key},
    )

    with stubber:
        put_meta = s3_store.put_bytes(
            obj_key,
            payload_bytes,
            content_type="application/json",
            expected_tenant_id=tenant_id,
        )
        assert put_meta.uri == f"s3://context-scout-artifacts/{obj_key}"
        assert s3_store.get_bytes(obj_key, expected_tenant_id=tenant_id) == payload_bytes
        with pytest.raises(ValueError, match="SHA-256 integrity mismatch"):
            s3_store.get_bytes(obj_key, expected_tenant_id=tenant_id)

    with pytest.raises(PermissionError, match="Cross-tenant"):
        s3_store.get_bytes(obj_key, expected_tenant_id=uuid.uuid4())
    with pytest.raises(ValueError, match="object key"):
        validate_object_key("../escape")


def test_m1_production_db_app_url_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production env fails closed on unset, superuser, or weak DB URL."""
    monkeypatch.setenv("SCOUT_ENV", "production")
    monkeypatch.delenv("SCOUT_DB_APP_URL", raising=False)
    with pytest.raises(RuntimeError, match="SCOUT_DB_APP_URL must be explicitly configured"):
        ScoutSettings()

    monkeypatch.setenv(
        "SCOUT_DB_APP_URL",
        "postgresql://postgres:strong_pw_123@10.0.0.5:5432/context_scout",
    )
    with pytest.raises(RuntimeError, match="privileged role"):
        ScoutSettings()

    monkeypatch.setenv(
        "SCOUT_DB_APP_URL", "postgresql://scout_app_user@10.0.0.5:5432/context_scout"
    )
    with pytest.raises(RuntimeError, match="Passwordless"):
        ScoutSettings()

    for weak_pw in sorted(pw for pw in INSECURE_DB_PASSWORDS if pw):
        monkeypatch.setenv(
            "SCOUT_DB_APP_URL",
            f"postgresql://scout_app_user:{weak_pw}@10.0.0.5:5432/context_scout",
        )
        with pytest.raises(RuntimeError, match="Insecure default database password"):
            ScoutSettings()

    valid_prod_url = "postgresql://scout_app_user:Prod_Str0ng_Vault_Pw!@10.0.0.5:5432/context_scout"
    monkeypatch.setenv("SCOUT_DB_APP_URL", valid_prod_url)
    validate_db_app_url(valid_prod_url)
    prod_settings = ScoutSettings()
    assert prod_settings.db_app_url == valid_prod_url
