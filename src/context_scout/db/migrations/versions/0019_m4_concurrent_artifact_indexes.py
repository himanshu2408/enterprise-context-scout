"""Create concurrent indexes for model_artifacts active lookup and one-active uniqueness.

Revision ID: 0019_m4_art_conc_idx
Revises: 0018_m4_model_artifacts
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0019_m4_art_conc_idx"
down_revision: str | Sequence[str] | None = "0018_m4_model_artifacts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONCURRENT_INDEX_DDL: tuple[str, ...] = (
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_idx_model_artifacts_one_active "
    "ON model_artifacts (tenant_id, domain_id, artifact_kind) WHERE is_active = TRUE;",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_artifacts_active_lookup "
    "ON model_artifacts (tenant_id, is_active, updated_at DESC);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_artifacts_ceiling_gin "
    "ON model_artifacts USING GIN (training_scope_ceiling);",
)

_INDEX_NAMES: tuple[str, ...] = (
    "uq_idx_model_artifacts_one_active",
    "idx_model_artifacts_active_lookup",
    "idx_model_artifacts_ceiling_gin",
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
