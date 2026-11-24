"""Add audit_events, scim_users, and scim_groups tables for M5 zero-trust auth & SIEM.

Revision ID: 0021_m5_audit_scim
Revises: 0020_m4_art_force_rls
Create Date: 2026-11-19
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0021_m5_audit_scim"
down_revision: str | Sequence[str] | None = "0020_m4_art_force_rls"
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
        CREATE TABLE IF NOT EXISTS audit_events (
            tenant_id UUID NOT NULL DEFAULT {_TENANT_DEFAULT_EXPR},
            seq_no BIGINT NOT NULL CHECK (seq_no >= 1),
            event_id UUID NOT NULL,
            domain_id TEXT NOT NULL DEFAULT 'default',
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'INFO' CHECK (
                severity IN (
                    'DEFAULT', 'DEBUG', 'INFO', 'NOTICE',
                    'WARNING', 'ERROR', 'CRITICAL', 'ALERT', 'EMERGENCY'
                )
            ),
            actor_id TEXT NOT NULL,
            actor_role TEXT NOT NULL,
            resource_uri TEXT NOT NULL DEFAULT '',
            outcome TEXT NOT NULL CHECK (
                outcome IN ('allow', 'deny', 'success', 'failure', 'abstain')
            ),
            trace_id TEXT NOT NULL DEFAULT '',
            span_id TEXT NOT NULL DEFAULT '',
            prev_hash_sha256 TEXT NOT NULL CHECK (char_length(prev_hash_sha256) = 64),
            event_hash_sha256 TEXT NOT NULL CHECK (char_length(event_hash_sha256) = 64),
            payload_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (tenant_id, seq_no)
        );
        """
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS scim_users (
            tenant_id UUID NOT NULL DEFAULT {_TENANT_DEFAULT_EXPR},
            scim_user_id TEXT NOT NULL,
            domain_id TEXT NOT NULL DEFAULT 'default',
            external_id TEXT NOT NULL DEFAULT '',
            user_name TEXT NOT NULL,
            email TEXT NOT NULL DEFAULT '',
            display_name TEXT NOT NULL DEFAULT '',
            active BOOLEAN NOT NULL DEFAULT TRUE,
            role_override TEXT,
            custom_scopes TEXT[] NOT NULL DEFAULT '{{}}'::text[],
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (tenant_id, scim_user_id)
        );
        """
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS scim_groups (
            tenant_id UUID NOT NULL DEFAULT {_TENANT_DEFAULT_EXPR},
            scim_group_id TEXT NOT NULL,
            domain_id TEXT NOT NULL DEFAULT 'default',
            external_id TEXT NOT NULL DEFAULT '',
            display_name TEXT NOT NULL,
            member_user_ids TEXT[] NOT NULL DEFAULT '{{}}'::text[],
            member_group_ids TEXT[] NOT NULL DEFAULT '{{}}'::text[],
            mapped_scopes TEXT[] NOT NULL DEFAULT '{{}}'::text[],
            mapped_role TEXT,
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (tenant_id, scim_group_id)
        );
        """
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")
    op.execute("DROP TABLE IF EXISTS scim_groups;")
    op.execute("DROP TABLE IF EXISTS scim_users;")
    op.execute("DROP TABLE IF EXISTS audit_events;")
