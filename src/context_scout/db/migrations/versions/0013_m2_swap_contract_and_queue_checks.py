"""Attach NOT VALID FKs and widen edge, queue event, and repository source checks.

Revision ID: 0013_m2_se_swap_chk
Revises: 0012_m2_se_conc_idx
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0013_m2_se_swap_chk"
down_revision: str | Sequence[str] | None = "0012_m2_se_conc_idx"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")

    op.execute(
        "ALTER TABLE side_effects ADD CONSTRAINT uq_side_effects_natural "
        "UNIQUE USING INDEX uq_idx_side_effects_natural;"
    )

    op.execute(
        "ALTER TABLE side_effects ADD CONSTRAINT fk_side_effects_tenant_domain "
        "FOREIGN KEY (tenant_id, domain_id) REFERENCES domain_packs (tenant_id, domain_id) "
        "ON DELETE CASCADE NOT VALID;"
    )
    op.execute(
        "ALTER TABLE side_effects ADD CONSTRAINT fk_side_effects_file "
        "FOREIGN KEY (file_id) REFERENCES code_files (file_id) "
        "ON DELETE CASCADE NOT VALID;"
    )
    op.execute(
        "ALTER TABLE side_effects ADD CONSTRAINT fk_side_effects_symbol "
        "FOREIGN KEY (symbol_id) REFERENCES ast_symbols (symbol_id) "
        "ON DELETE SET NULL NOT VALID;"
    )
    op.execute(
        "ALTER TABLE connector_sync_watermarks ADD CONSTRAINT "
        "fk_connector_watermarks_tenant_domain "
        "FOREIGN KEY (tenant_id, domain_id) REFERENCES domain_packs (tenant_id, domain_id) "
        "ON DELETE CASCADE NOT VALID;"
    )

    op.execute("ALTER TABLE ast_edges DROP CONSTRAINT IF EXISTS ast_edges_edge_kind_check;")
    op.execute(
        """
        ALTER TABLE ast_edges ADD CONSTRAINT ast_edges_edge_kind_check CHECK (
            edge_kind IN (
                'CALLS', 'IMPORTS', 'USES_TYPE', 'INHERITS', 'TESTED_BY',
                'PROVISIONED_BY', 'DISPATCHES', 'SELECTS', 'DOCUMENTS_COMPONENT',
                'OWNED_BY_TEAM', 'PART_OF_SYSTEM',
                'PUBLISHES_EVENT', 'SUBSCRIBES_EVENT', 'READS_TABLE', 'WRITES_TABLE',
                'RENDERS_TEMPLATE', 'IMPLEMENTS_SPEC', 'IMPLEMENTS_ISSUE',
                'BINDS_SERVICE_ACCOUNT', 'ACCESSES_SECRET'
            )
        ) NOT VALID;
        """
    )

    op.execute(
        "ALTER TABLE indexing_queue DROP CONSTRAINT IF EXISTS indexing_queue_event_type_check;"
    )
    op.execute(
        """
        ALTER TABLE indexing_queue ADD CONSTRAINT indexing_queue_event_type_check CHECK (
            event_type IN (
                'gitlab_push', 'github_push', 'gitlab_mr', 'github_pr',
                'confluence_update', 'jira_update', 'cross_repo_relink',
                'watermark_sync', 'reconcile_sweep', 'purge_artifact'
            )
        ) NOT VALID;
        """
    )

    op.execute("ALTER TABLE repositories DROP CONSTRAINT IF EXISTS repositories_source_type_check;")
    op.execute(
        """
        ALTER TABLE repositories ADD CONSTRAINT repositories_source_type_check CHECK (
            source_type IN ('gitlab', 'github', 'local_git', 'airgap_bundle')
        ) NOT VALID;
        """
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")

    op.execute("ALTER TABLE repositories DROP CONSTRAINT IF EXISTS repositories_source_type_check;")
    op.execute(
        """
        ALTER TABLE repositories ADD CONSTRAINT repositories_source_type_check CHECK (
            source_type IN ('gitlab', 'github', 'local_git')
        ) NOT VALID;
        """
    )
    op.execute(
        "ALTER TABLE indexing_queue DROP CONSTRAINT IF EXISTS indexing_queue_event_type_check;"
    )
    op.execute(
        """
        ALTER TABLE indexing_queue ADD CONSTRAINT indexing_queue_event_type_check CHECK (
            event_type IN (
                'gitlab_push', 'github_push', 'confluence_update', 'jira_update',
                'cross_repo_relink'
            )
        ) NOT VALID;
        """
    )
    op.execute("ALTER TABLE ast_edges DROP CONSTRAINT IF EXISTS ast_edges_edge_kind_check;")
    op.execute(
        """
        ALTER TABLE ast_edges ADD CONSTRAINT ast_edges_edge_kind_check CHECK (
            edge_kind IN (
                'CALLS', 'IMPORTS', 'USES_TYPE', 'INHERITS', 'TESTED_BY',
                'PROVISIONED_BY', 'DISPATCHES', 'SELECTS', 'DOCUMENTS_COMPONENT',
                'OWNED_BY_TEAM', 'PART_OF_SYSTEM'
            )
        ) NOT VALID;
        """
    )

    op.execute(
        "ALTER TABLE connector_sync_watermarks DROP CONSTRAINT IF EXISTS "
        "fk_connector_watermarks_tenant_domain;"
    )
    op.execute("ALTER TABLE side_effects DROP CONSTRAINT IF EXISTS fk_side_effects_symbol;")
    op.execute("ALTER TABLE side_effects DROP CONSTRAINT IF EXISTS fk_side_effects_file;")
    op.execute("ALTER TABLE side_effects DROP CONSTRAINT IF EXISTS fk_side_effects_tenant_domain;")
    op.execute("ALTER TABLE side_effects DROP CONSTRAINT IF EXISTS uq_side_effects_natural;")
