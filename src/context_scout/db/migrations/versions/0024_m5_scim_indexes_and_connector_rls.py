"""Add SCIM JSONB/B-tree indexes and tenant-bounded connector ingest RLS policy.

Revision ID: 0024_m5_scim_idx_conn_rls
Revises: 0023_m5_audit_force_rls
Create Date: 2026-11-27
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0024_m5_scim_idx_conn_rls"
down_revision: str | Sequence[str] | None = "0023_m5_audit_force_rls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_PRED = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid"
_SCOPE_ARRAY_EXPR = (
    "string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')"
)
_SCOPE_OR_CONNECTOR_PRED = (
    f"((allowed_scopes && {_SCOPE_ARRAY_EXPR}) "
    f"OR ('system:connector_ingest' = ANY({_SCOPE_ARRAY_EXPR})))"
)
_OLD_SCOPE_PRED = f"allowed_scopes && {_SCOPE_ARRAY_EXPR}"

_CONNECTOR_RLS_POLICY_MAP: tuple[tuple[str, str], ...] = (
    ("code_files", "rls_code_files_scope"),
    ("ast_symbols", "rls_ast_symbols_scope"),
    ("ast_edges", "rls_ast_edges_scope"),
    ("doc_chunks", "rls_doc_chunks_scope"),
    ("repositories", "rls_repositories_scope"),
    ("confluence_pages", "rls_confluence_scope"),
    ("jira_issues", "rls_jira_scope"),
)

_CONCURRENT_INDEX_DDL: tuple[str, ...] = (
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_idx_scim_users_username "
    "ON scim_users (tenant_id, lower(user_name));",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scim_users_external_id "
    "ON scim_users (tenant_id, external_id);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scim_users_email_lower "
    "ON scim_users (tenant_id, lower(email));",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scim_users_emails_gin "
    "ON scim_users USING GIN (emails_json jsonb_path_ops);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scim_groups_members_json_gin "
    "ON scim_groups USING GIN (members_json jsonb_path_ops);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scim_groups_member_groups_gin "
    "ON scim_groups USING GIN (member_group_ids);",
)

_NEW_INDEX_NAMES: tuple[str, ...] = (
    "idx_scim_groups_member_groups_gin",
    "idx_scim_groups_members_json_gin",
    "idx_scim_users_emails_gin",
    "idx_scim_users_email_lower",
    "idx_scim_users_external_id",
)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("SET lock_timeout = '4s';")
        op.execute("SET statement_timeout = '30s';")
        op.execute(
            "ALTER TABLE scim_users "
            "ADD COLUMN IF NOT EXISTS emails_json JSONB NOT NULL DEFAULT '[]'::jsonb;"
        )
        op.execute(
            "ALTER TABLE scim_groups "
            "ADD COLUMN IF NOT EXISTS members_json JSONB NOT NULL DEFAULT '[]'::jsonb;"
        )
        for table, pol_name in _CONNECTOR_RLS_POLICY_MAP:
            pred = f"({_TENANT_PRED}) AND {_SCOPE_OR_CONNECTOR_PRED}"
            op.execute(f"DROP POLICY IF EXISTS {pol_name} ON {table};")
            op.execute(f"CREATE POLICY {pol_name} ON {table} USING ({pred}) WITH CHECK ({pred});")
        op.execute("RESET lock_timeout;")
        op.execute("RESET statement_timeout;")
        for stmt in _CONCURRENT_INDEX_DDL:
            op.execute(stmt)


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("SET lock_timeout = '4s';")
        op.execute("SET statement_timeout = '120s';")
        for idx_name in _NEW_INDEX_NAMES:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {idx_name};")
        for table, pol_name in _CONNECTOR_RLS_POLICY_MAP:
            pred = f"({_TENANT_PRED}) AND ({_OLD_SCOPE_PRED})"
            op.execute(f"DROP POLICY IF EXISTS {pol_name} ON {table};")
            op.execute(f"CREATE POLICY {pol_name} ON {table} USING ({pred}) WITH CHECK ({pred});")
        op.execute("ALTER TABLE scim_groups DROP COLUMN IF EXISTS members_json;")
        op.execute("ALTER TABLE scim_users DROP COLUMN IF EXISTS emails_json;")
        op.execute("RESET lock_timeout;")
        op.execute("RESET statement_timeout;")
