"""Validate composite FKs, enforce ENABLE + FORCE RLS on 13/13 tables, and harden runtime role.

Revision ID: 0006_m1_force_rls
Revises: 0005_m1_swap_keys
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0006_m1_force_rls"
down_revision: str | Sequence[str] | None = "0005_m1_swap_keys"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_PRED = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid"
_SCOPE_PRED = (
    "allowed_scopes && "
    "string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')"
)
_REPO_COMMITS_PRED = (
    f"{_TENANT_PRED} AND EXISTS ("
    "SELECT 1 FROM repositories r "
    "WHERE r.repo_id = repo_commits.repo_id "
    "AND r.tenant_id = repo_commits.tenant_id)"
)

_ALL_TABLES: tuple[str, ...] = (
    "domain_packs",
    "domain_sync_manifests",
    "embedding_spaces",
    "code_files",
    "ast_symbols",
    "ast_edges",
    "doc_chunks",
    "repositories",
    "repo_commits",
    "confluence_pages",
    "jira_issues",
    "execution_telemetry",
    "indexing_queue",
    "indexing_dead_letter",
)

_ACL_SCOPED_POLICY_MAP: tuple[tuple[str, str], ...] = (
    ("code_files", "rls_code_files_scope"),
    ("ast_symbols", "rls_ast_symbols_scope"),
    ("ast_edges", "rls_ast_edges_scope"),
    ("doc_chunks", "rls_doc_chunks_scope"),
    ("repositories", "rls_repositories_scope"),
    ("confluence_pages", "rls_confluence_scope"),
    ("jira_issues", "rls_jira_scope"),
)

_TENANT_ONLY_POLICY_MAP: tuple[tuple[str, str], ...] = (
    ("domain_packs", "rls_domain_packs_tenant"),
    ("domain_sync_manifests", "rls_domain_sync_manifests_tenant"),
    ("embedding_spaces", "rls_embedding_spaces_tenant"),
    ("execution_telemetry", "rls_execution_telemetry_tenant"),
    ("indexing_queue", "rls_indexing_queue_tenant"),
    ("indexing_dead_letter", "rls_indexing_dead_letter_tenant"),
)


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '600s';")

    op.execute("ALTER TABLE doc_chunks VALIDATE CONSTRAINT doc_chunks_market_scope_check;")
    for table in (
        "domain_sync_manifests",
        "embedding_spaces",
        "code_files",
        "doc_chunks",
        "repositories",
        "confluence_pages",
        "jira_issues",
        "indexing_queue",
    ):
        op.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT fk_{table}_tenant_domain;")

    op.execute(
        """
        INSERT INTO embedding_spaces (
            tenant_id, domain_id, space_id, model_id, provider,
            dimensions, quantization, status, is_active, activated_at
        )
        SELECT
            tenant_id, domain_id, 'embeddinggemma-2-768', 'google/embeddinggemma-2',
            'sentence-transformers', 768, 'halfvec', 'active', TRUE, NOW()
        FROM domain_packs
        ON CONFLICT (tenant_id, domain_id, space_id) DO NOTHING;
        """
    )

    for table in _ALL_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;")

    for table, pol_name in _TENANT_ONLY_POLICY_MAP:
        op.execute(f"DROP POLICY IF EXISTS {pol_name} ON {table};")
        op.execute(
            f"CREATE POLICY {pol_name} ON {table} "
            f"USING ({_TENANT_PRED}) WITH CHECK ({_TENANT_PRED});"
        )

    for table, pol_name in _ACL_SCOPED_POLICY_MAP:
        pred = f"({_TENANT_PRED}) AND ({_SCOPE_PRED})"
        op.execute(f"DROP POLICY IF EXISTS {pol_name} ON {table};")
        op.execute(f"CREATE POLICY {pol_name} ON {table} USING ({pred}) WITH CHECK ({pred});")

    op.execute("DROP POLICY IF EXISTS rls_repo_commits_scope ON repo_commits;")
    op.execute(
        f"CREATE POLICY rls_repo_commits_scope ON repo_commits "
        f"USING ({_REPO_COMMITS_PRED}) WITH CHECK ({_REPO_COMMITS_PRED});"
    )

    op.execute(
        "ALTER ROLE scout_app_user NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOINHERIT;"
    )
    op.execute("GRANT USAGE ON SCHEMA public TO scout_app_user;")
    table_list = ", ".join(_ALL_TABLES)
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table_list} TO scout_app_user;")


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")
    for table, pol_name in _TENANT_ONLY_POLICY_MAP:
        op.execute(f"DROP POLICY IF EXISTS {pol_name} ON {table};")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;")
    op.execute("DROP POLICY IF EXISTS rls_repo_commits_scope ON repo_commits;")
    op.execute("ALTER TABLE repo_commits NO FORCE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE repo_commits DISABLE ROW LEVEL SECURITY;")
