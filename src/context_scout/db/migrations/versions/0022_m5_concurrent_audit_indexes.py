"""Create concurrent indexes for audit_events, scim_users, and scim_groups.

Revision ID: 0022_m5_audit_conc_idx
Revises: 0021_m5_audit_scim
Create Date: 2026-11-19
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0022_m5_audit_conc_idx"
down_revision: str | Sequence[str] | None = "0021_m5_audit_scim"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONCURRENT_INDEX_DDL: tuple[str, ...] = (
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_idx_audit_events_event_id "
    "ON audit_events (tenant_id, event_id);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_events_domain_created "
    "ON audit_events (tenant_id, domain_id, created_at DESC);",
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_idx_scim_users_username "
    "ON scim_users (tenant_id, lower(user_name));",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scim_groups_members_gin "
    "ON scim_groups USING GIN (member_user_ids);",
)

_INDEX_NAMES: tuple[str, ...] = (
    "uq_idx_audit_events_event_id",
    "idx_audit_events_domain_created",
    "uq_idx_scim_users_username",
    "idx_scim_groups_members_gin",
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
