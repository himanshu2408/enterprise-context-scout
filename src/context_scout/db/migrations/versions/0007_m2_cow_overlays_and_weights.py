"""Add CoW ref_snapshots, overlay_file_deltas, and code_files ranking weights.

Revision ID: 0007_m2_cow_tables
Revises: 0006_m1_force_rls
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0007_m2_cow_tables"
down_revision: str | Sequence[str] | None = "0006_m1_force_rls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_DEFAULT_EXPR = (
    "COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, "
    "'00000000-0000-0000-0000-000000000001'::uuid)"
)


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '600s';")

    op.execute(
        "ALTER TABLE code_files ADD COLUMN IF NOT EXISTS "
        "ranking_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0;"
    )
    op.execute(
        "ALTER TABLE code_files ADD COLUMN IF NOT EXISTS "
        "is_generated BOOLEAN NOT NULL DEFAULT FALSE;"
    )
    op.execute(
        "ALTER TABLE code_files ADD COLUMN IF NOT EXISTS "
        "is_vendored BOOLEAN NOT NULL DEFAULT FALSE;"
    )
    op.execute(
        "ALTER TABLE code_files ADD CONSTRAINT chk_code_files_ranking_weight "
        "CHECK (ranking_weight >= 0.0 AND ranking_weight <= 2.0) NOT VALID;"
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS ref_snapshots (
            snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id UUID NOT NULL DEFAULT {_TENANT_DEFAULT_EXPR},
            domain_id TEXT NOT NULL,
            repo_id UUID,
            repo_name TEXT NOT NULL,
            ref_name TEXT NOT NULL,
            ref_kind TEXT NOT NULL CHECK (
                ref_kind IN ('base_branch', 'feature_branch', 'merge_request', 'pull_request')
            ),
            mr_iid TEXT,
            base_snapshot_id UUID,
            base_commit_sha TEXT NOT NULL,
            head_commit_sha TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'open' CHECK (
                state IN ('active', 'open', 'merged', 'closed')
            ),
            allowed_scopes TEXT[] NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS overlay_file_deltas (
            delta_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id UUID NOT NULL DEFAULT {_TENANT_DEFAULT_EXPR},
            snapshot_id UUID NOT NULL,
            domain_id TEXT NOT NULL,
            repo_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            previous_file_path TEXT,
            change_type TEXT NOT NULL CHECK (
                change_type IN ('added', 'modified', 'deleted', 'renamed')
            ),
            base_blob_sha TEXT,
            overlay_blob_sha TEXT,
            language TEXT NOT NULL DEFAULT 'text',
            raw_content TEXT,
            line_count BIGINT NOT NULL DEFAULT 0 CHECK (line_count >= 0),
            ranking_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0 CHECK (
                ranking_weight >= 0.0 AND ranking_weight <= 2.0
            ),
            is_generated BOOLEAN NOT NULL DEFAULT FALSE,
            is_vendored BOOLEAN NOT NULL DEFAULT FALSE,
            symbols_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            edges_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            allowed_scopes TEXT[] NOT NULL,
            indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CHECK (
                (change_type = 'deleted' AND raw_content IS NULL)
                OR (
                    change_type <> 'deleted'
                    AND raw_content IS NOT NULL
                    AND overlay_blob_sha IS NOT NULL
                )
            )
        );
        """
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")
    op.execute("DROP TABLE IF EXISTS overlay_file_deltas;")
    op.execute("DROP TABLE IF EXISTS ref_snapshots;")
    op.execute("ALTER TABLE code_files DROP CONSTRAINT IF EXISTS chk_code_files_ranking_weight;")
    op.execute("ALTER TABLE code_files DROP COLUMN IF EXISTS is_vendored;")
    op.execute("ALTER TABLE code_files DROP COLUMN IF EXISTS is_generated;")
    op.execute("ALTER TABLE code_files DROP COLUMN IF EXISTS ranking_weight;")
