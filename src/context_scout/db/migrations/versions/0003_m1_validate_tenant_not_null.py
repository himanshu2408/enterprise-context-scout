"""Validate tenant_id NOT VALID constraints and apply metadata-only NOT NULL.

Revision ID: 0003_m1_validate_nn
Revises: 0002_m1_add_tenant
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003_m1_validate_nn"
down_revision: str | Sequence[str] | None = "0002_m1_add_tenant"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BASELINE_TABLES: tuple[str, ...] = (
    "domain_packs",
    "repositories",
    "repo_commits",
    "confluence_pages",
    "jira_issues",
    "code_files",
    "ast_symbols",
    "ast_edges",
    "doc_chunks",
    "execution_telemetry",
    "indexing_queue",
    "indexing_dead_letter",
)


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '600s';")

    for table in _BASELINE_TABLES:
        op.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT chk_{table}_tenant_nn;")
        op.execute(f"ALTER TABLE {table} ALTER COLUMN tenant_id SET NOT NULL;")

    op.execute("ALTER TABLE domain_packs VALIDATE CONSTRAINT chk_domain_packs_updated_at_nn;")
    op.execute("ALTER TABLE domain_packs ALTER COLUMN updated_at SET NOT NULL;")
    op.execute("ALTER TABLE ast_symbols VALIDATE CONSTRAINT chk_ast_symbols_nonzero_vec;")
    op.execute("ALTER TABLE doc_chunks VALIDATE CONSTRAINT chk_doc_chunks_nonzero_vec;")


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")
    op.execute("ALTER TABLE domain_packs ALTER COLUMN updated_at DROP NOT NULL;")
    for table in reversed(_BASELINE_TABLES):
        op.execute(f"ALTER TABLE {table} ALTER COLUMN tenant_id DROP NOT NULL;")
