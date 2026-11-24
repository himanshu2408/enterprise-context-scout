"""Add scout_sessions table for multi-turn Graph 0 compaction & commit-drift tracking.

Revision ID: 0015_m3_scout_sessions
Revises: 0014_m2_se_force_rls
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0015_m3_scout_sessions"
down_revision: str | Sequence[str] | None = "0014_m2_se_force_rls"
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
        f"""
        CREATE TABLE IF NOT EXISTS scout_sessions (
            tenant_id UUID NOT NULL DEFAULT {_TENANT_DEFAULT_EXPR},
            session_id TEXT NOT NULL,
            domain_id TEXT NOT NULL,
            principal_id TEXT NOT NULL,
            allowed_scopes TEXT[] NOT NULL,
            turn_count BIGINT NOT NULL DEFAULT 0 CHECK (turn_count >= 0),
            compacted_summary TEXT NOT NULL DEFAULT '',
            recent_turns_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            pinned_file_blobs_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            pinned_symbols_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            total_tokens_compacted BIGINT NOT NULL DEFAULT 0 CHECK (total_tokens_compacted >= 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (tenant_id, session_id)
        );
        """
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")
    op.execute("DROP TABLE IF EXISTS scout_sessions;")
