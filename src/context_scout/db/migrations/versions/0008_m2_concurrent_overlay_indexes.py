"""Create concurrent unique, B-tree, GIN, and trigram indexes for CoW MR overlays.

Revision ID: 0008_m2_conc_idx
Revises: 0007_m2_cow_tables
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0008_m2_conc_idx"
down_revision: str | Sequence[str] | None = "0007_m2_cow_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONCURRENT_INDEX_DDL: tuple[str, ...] = (
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_idx_ref_snapshots_tenant_domain_repo_ref "
    "ON ref_snapshots (tenant_id, domain_id, repo_name, ref_name);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ref_snapshots_tenant_domain_mr "
    "ON ref_snapshots (tenant_id, domain_id, repo_name, mr_iid) WHERE mr_iid IS NOT NULL;",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ref_snapshots_scopes "
    "ON ref_snapshots USING GIN (allowed_scopes);",
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_idx_overlay_deltas_tenant_snap_path "
    "ON overlay_file_deltas (tenant_id, snapshot_id, file_path);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_overlay_deltas_tenant_domain_repo "
    "ON overlay_file_deltas (tenant_id, domain_id, repo_name, snapshot_id);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_overlay_deltas_scopes "
    "ON overlay_file_deltas USING GIN (allowed_scopes);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_overlay_deltas_trgm "
    "ON overlay_file_deltas USING GIN (raw_content gin_trgm_ops) WHERE raw_content IS NOT NULL;",
)

_INDEX_NAMES: tuple[str, ...] = (
    "uq_idx_ref_snapshots_tenant_domain_repo_ref",
    "idx_ref_snapshots_tenant_domain_mr",
    "idx_ref_snapshots_scopes",
    "uq_idx_overlay_deltas_tenant_snap_path",
    "idx_overlay_deltas_tenant_domain_repo",
    "idx_overlay_deltas_scopes",
    "idx_overlay_deltas_trgm",
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
