"""Add side_effects catalog, connector_sync_watermarks, and Jira hierarchy columns.

Revision ID: 0011_m2_side_effects
Revises: 0010_m2_force_rls
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0011_m2_side_effects"
down_revision: str | Sequence[str] | None = "0010_m2_force_rls"
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
        CREATE TABLE IF NOT EXISTS side_effects (
            side_effect_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id UUID NOT NULL DEFAULT {_TENANT_DEFAULT_EXPR},
            domain_id TEXT NOT NULL,
            repo_name TEXT NOT NULL,
            file_id UUID,
            symbol_id UUID,
            file_path TEXT NOT NULL,
            qualified_symbol TEXT NOT NULL,
            start_line BIGINT NOT NULL CHECK (start_line >= 1),
            end_line BIGINT NOT NULL CHECK (end_line >= start_line),
            effect_category TEXT NOT NULL CHECK (
                effect_category IN (
                    'email', 'sms', 'push_notification', 'webhook',
                    'pubsub_kafka_event', 'cron_job', 'feature_flag', 'db_write'
                )
            ),
            target_identifier TEXT NOT NULL,
            provider_or_driver TEXT NOT NULL DEFAULT '',
            trigger_condition TEXT NOT NULL DEFAULT '',
            template_or_schema_ref TEXT,
            severity_tier TEXT NOT NULL DEFAULT 'external_io' CHECK (
                severity_tier IN (
                    'external_comms', 'external_io', 'async_messaging',
                    'scheduled_job', 'runtime_toggle', 'state_mutation'
                )
            ),
            confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0 CHECK (
                confidence >= 0.0 AND confidence <= 1.0
            ),
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            allowed_scopes TEXT[] NOT NULL,
            indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS connector_sync_watermarks (
            tenant_id UUID NOT NULL DEFAULT {_TENANT_DEFAULT_EXPR},
            domain_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            container_key TEXT NOT NULL,
            watermark_cursor TEXT NOT NULL DEFAULT '',
            sync_tier TEXT NOT NULL DEFAULT 'watermark' CHECK (
                sync_tier IN ('webhook', 'watermark', 'reconcile', 'bundle')
            ),
            items_synced BIGINT NOT NULL DEFAULT 0 CHECK (items_synced >= 0),
            last_webhook_at TIMESTAMPTZ,
            last_watermark_sync_at TIMESTAMPTZ,
            last_reconciliation_at TIMESTAMPTZ,
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (tenant_id, domain_id, source_id, container_key)
        );
        """
    )

    op.execute("ALTER TABLE jira_issues ADD COLUMN IF NOT EXISTS epic_key TEXT;")
    op.execute("ALTER TABLE jira_issues ADD COLUMN IF NOT EXISTS parent_issue_key TEXT;")
    op.execute(
        "ALTER TABLE jira_issues ADD COLUMN IF NOT EXISTS "
        "linked_issues_json JSONB NOT NULL DEFAULT '[]'::jsonb;"
    )
    op.execute(
        "ALTER TABLE jira_issues ADD COLUMN IF NOT EXISTS "
        "linked_mrs_json JSONB NOT NULL DEFAULT '[]'::jsonb;"
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")
    op.execute("ALTER TABLE jira_issues DROP COLUMN IF EXISTS linked_mrs_json;")
    op.execute("ALTER TABLE jira_issues DROP COLUMN IF EXISTS linked_issues_json;")
    op.execute("ALTER TABLE jira_issues DROP COLUMN IF EXISTS parent_issue_key;")
    op.execute("ALTER TABLE jira_issues DROP COLUMN IF EXISTS epic_key;")
    op.execute("DROP TABLE IF EXISTS connector_sync_watermarks;")
    op.execute("DROP TABLE IF EXISTS side_effects;")
