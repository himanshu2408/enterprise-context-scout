"""Create concurrent indexes for scout_sessions lookup and scope GIN filtering.

Revision ID: 0016_m3_sess_conc_idx
Revises: 0015_m3_scout_sessions
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0016_m3_sess_conc_idx"
down_revision: str | Sequence[str] | None = "0015_m3_scout_sessions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONCURRENT_INDEX_DDL: tuple[str, ...] = (
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scout_sessions_domain_user "
    "ON scout_sessions (tenant_id, domain_id, principal_id, updated_at DESC);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scout_sessions_scopes "
    "ON scout_sessions USING GIN (allowed_scopes);",
)

_INDEX_NAMES: tuple[str, ...] = (
    "idx_scout_sessions_domain_user",
    "idx_scout_sessions_scopes",
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
