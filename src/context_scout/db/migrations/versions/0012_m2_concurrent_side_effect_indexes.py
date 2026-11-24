"""Create concurrent indexes for side_effects, connector_sync_watermarks, and Jira hierarchy.

Revision ID: 0012_m2_se_conc_idx
Revises: 0011_m2_side_effects
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0012_m2_se_conc_idx"
down_revision: str | Sequence[str] | None = "0011_m2_side_effects"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONCURRENT_INDEX_DDL: tuple[str, ...] = (
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_idx_side_effects_natural "
    "ON side_effects ("
    "tenant_id, domain_id, repo_name, file_path, qualified_symbol, "
    "effect_category, target_identifier, start_line"
    ");",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_side_effects_domain_cat "
    "ON side_effects (tenant_id, domain_id, effect_category);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_side_effects_scopes "
    "ON side_effects USING GIN (allowed_scopes);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_jira_issues_hierarchy "
    "ON jira_issues (tenant_id, domain_id, epic_key, parent_issue_key) "
    "WHERE epic_key IS NOT NULL OR parent_issue_key IS NOT NULL;",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_connector_watermarks_lookup "
    "ON connector_sync_watermarks (tenant_id, domain_id, source_id);",
)

_INDEX_NAMES: tuple[str, ...] = (
    "uq_idx_side_effects_natural",
    "idx_side_effects_domain_cat",
    "idx_side_effects_scopes",
    "idx_jira_issues_hierarchy",
    "idx_connector_watermarks_lookup",
)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("SET lock_timeout = '2s';")
        op.execute("SET statement_timeout = '600s';")
        for stmt in _CONCURRENT_INDEX_DDL:
            op.execute(stmt)


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("SET lock_timeout = '2s';")
        op.execute("SET statement_timeout = '120s';")
        for idx_name in reversed(_INDEX_NAMES):
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {idx_name};")
