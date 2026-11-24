"""Add nullable tenant_id, backfill default tenant, add NOT VALID checks, and domain_sync_manifests.

Revision ID: 0002_m1_add_tenant
Revises: 0001_legacy_baseline
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002_m1_add_tenant"
down_revision: str | Sequence[str] | None = "0001_legacy_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_DEFAULT_EXPR = (
    "COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, "
    "'00000000-0000-0000-0000-000000000001'::uuid)"
)

_BASELINE_TABLES: tuple[str, ...] = (
    "domain_packs",
    "repositories",
    "repo_commits",
    "confluence_pages",
    "jira_issues",
    "code_files",
    "ast_symbols",
    "ast_edges",
    "doc_chunks",
    "execution_telemetry",
    "indexing_queue",
    "indexing_dead_letter",
)


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '600s';")

    for table in _BASELINE_TABLES:
        op.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS tenant_id UUID "
            "DEFAULT '00000000-0000-0000-0000-000000000001'::uuid;"
        )
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN tenant_id SET DEFAULT {_TENANT_DEFAULT_EXPR};"
        )
        op.execute(
            f"UPDATE {table} SET tenant_id = '00000000-0000-0000-0000-000000000001'::uuid "
            "WHERE tenant_id IS NULL;"
        )
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT chk_{table}_tenant_nn "
            "CHECK (tenant_id IS NOT NULL) NOT VALID;"
        )

    op.execute("ALTER TABLE domain_packs ADD COLUMN IF NOT EXISTS s3_bundle_uri TEXT;")
    op.execute("ALTER TABLE domain_packs ADD COLUMN IF NOT EXISTS bundle_sha256 TEXT;")
    op.execute("ALTER TABLE domain_packs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;")
    op.execute("ALTER TABLE domain_packs ALTER COLUMN updated_at SET DEFAULT NOW();")
    op.execute("UPDATE domain_packs SET updated_at = NOW() WHERE updated_at IS NULL;")
    op.execute(
        "ALTER TABLE domain_packs ADD CONSTRAINT chk_domain_packs_updated_at_nn "
        "CHECK (updated_at IS NOT NULL) NOT VALID;"
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS domain_sync_manifests (
            tenant_id UUID NOT NULL DEFAULT {_TENANT_DEFAULT_EXPR},
            domain_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            source_dir TEXT NOT NULL DEFAULT '',
            repo_commit_shas JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            empty_doc_shas JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            code_files_indexed BIGINT NOT NULL DEFAULT 0,
            ast_symbols_indexed BIGINT NOT NULL DEFAULT 0,
            ast_edges_indexed BIGINT NOT NULL DEFAULT 0,
            doc_chunks_indexed BIGINT NOT NULL DEFAULT 0,
            s3_manifest_uri TEXT,
            manifest_sha256 TEXT,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (tenant_id, domain_id)
        );
        """
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS embedding_spaces (
            tenant_id UUID NOT NULL DEFAULT {_TENANT_DEFAULT_EXPR},
            domain_id TEXT NOT NULL,
            space_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT 'sentence-transformers',
            dimensions BIGINT NOT NULL CHECK (dimensions BETWEEN 128 AND 4096),
            quantization TEXT NOT NULL DEFAULT 'halfvec' CHECK (
                quantization IN ('halfvec', 'vector')
            ),
            status TEXT NOT NULL DEFAULT 'shadow' CHECK (
                status IN ('active', 'shadow', 'retired')
            ),
            is_active BOOLEAN NOT NULL DEFAULT FALSE,
            index_operator_class TEXT NOT NULL DEFAULT 'halfvec_cosine_ops',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            activated_at TIMESTAMPTZ,
            PRIMARY KEY (tenant_id, domain_id, space_id)
        );
        """
    )

    for vec_table in ("ast_symbols", "doc_chunks"):
        op.execute(
            f"ALTER TABLE {vec_table} ADD COLUMN IF NOT EXISTS embedding_space_id TEXT "
            "DEFAULT 'embeddinggemma-2-768';"
        )
        op.execute(f"ALTER TABLE {vec_table} ADD COLUMN IF NOT EXISTS embedding_halfvec halfvec;")
        op.execute(
            f"ALTER TABLE {vec_table} ADD COLUMN IF NOT EXISTS shadow_embedding_space_id TEXT;"
        )
        op.execute(
            f"ALTER TABLE {vec_table} ADD COLUMN IF NOT EXISTS shadow_embedding_halfvec halfvec;"
        )
        op.execute(
            f"ALTER TABLE {vec_table} ADD CONSTRAINT chk_{vec_table}_nonzero_vec CHECK ("
            "vector_norm(embedding) > 0.0 "
            "AND (embedding_halfvec IS NULL OR l2_norm(embedding_halfvec) > 0.0) "
            "AND (shadow_embedding_halfvec IS NULL OR l2_norm(shadow_embedding_halfvec) > 0.0)"
            ") NOT VALID;"
        )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION sync_default_embedding_halfvec() RETURNS trigger AS $$
        BEGIN
            IF NEW.embedding_space_id IS NULL OR btrim(NEW.embedding_space_id) = '' THEN
                NEW.embedding_space_id := 'embeddinggemma-2-768';
            END IF;
            IF NEW.embedding IS NOT NULL AND (
                NEW.embedding_halfvec IS NULL
                OR (
                    TG_OP = 'UPDATE'
                    AND NEW.embedding IS DISTINCT FROM OLD.embedding
                    AND NEW.embedding_halfvec IS NOT DISTINCT FROM OLD.embedding_halfvec
                )
            ) THEN
                NEW.embedding_halfvec := NEW.embedding::halfvec;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    for vec_table in ("ast_symbols", "doc_chunks"):
        op.execute(f"DROP TRIGGER IF EXISTS trg_{vec_table}_sync_halfvec ON {vec_table};")
        op.execute(
            f"CREATE TRIGGER trg_{vec_table}_sync_halfvec "
            f"BEFORE INSERT OR UPDATE OF embedding, embedding_halfvec, embedding_space_id "
            f"ON {vec_table} FOR EACH ROW EXECUTE FUNCTION sync_default_embedding_halfvec();"
        )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")
    for vec_table in ("doc_chunks", "ast_symbols"):
        op.execute(f"DROP TRIGGER IF EXISTS trg_{vec_table}_sync_halfvec ON {vec_table};")
        op.execute(
            f"ALTER TABLE {vec_table} DROP CONSTRAINT IF EXISTS chk_{vec_table}_nonzero_vec;"
        )
        op.execute(f"ALTER TABLE {vec_table} DROP COLUMN IF EXISTS shadow_embedding_halfvec;")
        op.execute(f"ALTER TABLE {vec_table} DROP COLUMN IF EXISTS shadow_embedding_space_id;")
        op.execute(f"ALTER TABLE {vec_table} DROP COLUMN IF EXISTS embedding_halfvec;")
        op.execute(f"ALTER TABLE {vec_table} DROP COLUMN IF EXISTS embedding_space_id;")
    op.execute("DROP FUNCTION IF EXISTS sync_default_embedding_halfvec();")
    op.execute("DROP TABLE IF EXISTS embedding_spaces;")
    op.execute("DROP TABLE IF EXISTS domain_sync_manifests;")
    op.execute("ALTER TABLE domain_packs DROP CONSTRAINT IF EXISTS chk_domain_packs_updated_at_nn;")
    op.execute("ALTER TABLE domain_packs DROP COLUMN IF EXISTS updated_at;")
    op.execute("ALTER TABLE domain_packs DROP COLUMN IF EXISTS bundle_sha256;")
    op.execute("ALTER TABLE domain_packs DROP COLUMN IF EXISTS s3_bundle_uri;")
    for table in reversed(_BASELINE_TABLES):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS chk_{table}_tenant_nn;")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS tenant_id;")
