"""Add model_artifacts table for governed LoRA cartridges and AdapterSyncSidecar.

Revision ID: 0018_m4_model_artifacts
Revises: 0017_m3_sess_force_rls
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0018_m4_model_artifacts"
down_revision: str | Sequence[str] | None = "0017_m3_sess_force_rls"
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
        CREATE TABLE IF NOT EXISTS model_artifacts (
            tenant_id UUID NOT NULL DEFAULT {_TENANT_DEFAULT_EXPR},
            domain_id TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            artifact_kind TEXT NOT NULL DEFAULT 'lora_adapter' CHECK (
                artifact_kind IN ('lora_adapter', 'bi_encoder', 'merged_checkpoint')
            ),
            base_model_id TEXT NOT NULL,
            base_model_revision_sha256 TEXT NOT NULL CHECK (
                char_length(base_model_revision_sha256) = 64
            ),
            storage_uri TEXT NOT NULL,
            weights_object_key TEXT NOT NULL,
            config_object_key TEXT NOT NULL,
            signature_object_key TEXT NOT NULL DEFAULT '',
            weights_sha256 TEXT NOT NULL CHECK (char_length(weights_sha256) = 64),
            cosign_signature_b64 TEXT NOT NULL DEFAULT '',
            cosign_key_id TEXT NOT NULL DEFAULT 'default',
            lora_rank BIGINT NOT NULL DEFAULT 16 CHECK (lora_rank IN (8, 16, 32)),
            lora_alpha BIGINT NOT NULL DEFAULT 32 CHECK (lora_alpha > 0),
            use_rslora BOOLEAN NOT NULL DEFAULT TRUE CHECK (use_rslora IS TRUE),
            use_dora BOOLEAN NOT NULL DEFAULT FALSE CHECK (use_dora IS FALSE),
            training_scope_ceiling TEXT[] NOT NULL CHECK (
                cardinality(training_scope_ceiling) >= 1
            ),
            allowed_scopes TEXT[] NOT NULL,
            canary_leak_count BIGINT NOT NULL DEFAULT 0 CHECK (canary_leak_count = 0),
            status TEXT NOT NULL DEFAULT 'staged' CHECK (
                status IN ('staged', 'active', 'retired', 'revoked')
            ),
            is_active BOOLEAN NOT NULL DEFAULT FALSE,
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            activated_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (tenant_id, domain_id, artifact_id)
        );
        """
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")
    op.execute("DROP TABLE IF EXISTS model_artifacts;")
