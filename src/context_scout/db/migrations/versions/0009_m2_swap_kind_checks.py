"""Promote overlay unique indexes, attach NOT VALID FKs, and widen ownership checks.

Revision ID: 0009_m2_swap_checks
Revises: 0008_m2_conc_idx
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0009_m2_swap_checks"
down_revision: str | Sequence[str] | None = "0008_m2_conc_idx"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")

    op.execute(
        "ALTER TABLE ref_snapshots ADD CONSTRAINT uq_ref_snapshots_tenant_domain_repo_ref "
        "UNIQUE USING INDEX uq_idx_ref_snapshots_tenant_domain_repo_ref;"
    )
    op.execute(
        "ALTER TABLE overlay_file_deltas ADD CONSTRAINT uq_overlay_deltas_tenant_snap_path "
        "UNIQUE USING INDEX uq_idx_overlay_deltas_tenant_snap_path;"
    )

    op.execute(
        "ALTER TABLE ref_snapshots ADD CONSTRAINT fk_ref_snapshots_tenant_domain "
        "FOREIGN KEY (tenant_id, domain_id) REFERENCES domain_packs (tenant_id, domain_id) "
        "ON DELETE CASCADE NOT VALID;"
    )
    op.execute(
        "ALTER TABLE ref_snapshots ADD CONSTRAINT fk_ref_snapshots_base_snapshot "
        "FOREIGN KEY (base_snapshot_id) REFERENCES ref_snapshots (snapshot_id) "
        "ON DELETE SET NULL NOT VALID;"
    )
    op.execute(
        "ALTER TABLE overlay_file_deltas ADD CONSTRAINT fk_overlay_file_deltas_tenant_domain "
        "FOREIGN KEY (tenant_id, domain_id) REFERENCES domain_packs (tenant_id, domain_id) "
        "ON DELETE CASCADE NOT VALID;"
    )
    op.execute(
        "ALTER TABLE overlay_file_deltas ADD CONSTRAINT fk_overlay_file_deltas_snapshot "
        "FOREIGN KEY (snapshot_id) REFERENCES ref_snapshots (snapshot_id) "
        "ON DELETE CASCADE NOT VALID;"
    )

    op.execute("ALTER TABLE ast_symbols DROP CONSTRAINT IF EXISTS ast_symbols_symbol_kind_check;")
    op.execute(
        """
        ALTER TABLE ast_symbols ADD CONSTRAINT ast_symbols_symbol_kind_check CHECK (
            symbol_kind IN (
                'function', 'class', 'method', 'interface', 'type_alias',
                'hook', 'slice', 'selector', 'reducer', 'story',
                'tf_resource', 'tf_module', 'gcp_resource',
                'component', 'system', 'team', 'service', 'module', 'api', 'domain'
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


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")

    op.execute("ALTER TABLE ast_edges DROP CONSTRAINT IF EXISTS ast_edges_edge_kind_check;")
    op.execute(
        """
        ALTER TABLE ast_edges ADD CONSTRAINT ast_edges_edge_kind_check CHECK (
            edge_kind IN (
                'CALLS', 'IMPORTS', 'USES_TYPE', 'INHERITS', 'TESTED_BY',
                'PROVISIONED_BY', 'DISPATCHES', 'SELECTS', 'DOCUMENTS_COMPONENT'
            )
        ) NOT VALID;
        """
    )
    op.execute("ALTER TABLE ast_symbols DROP CONSTRAINT IF EXISTS ast_symbols_symbol_kind_check;")
    op.execute(
        """
        ALTER TABLE ast_symbols ADD CONSTRAINT ast_symbols_symbol_kind_check CHECK (
            symbol_kind IN (
                'function', 'class', 'method', 'interface', 'type_alias',
                'hook', 'slice', 'selector', 'reducer', 'story',
                'tf_resource', 'tf_module', 'gcp_resource'
            )
        ) NOT VALID;
        """
    )
    op.execute(
        "ALTER TABLE overlay_file_deltas DROP CONSTRAINT IF EXISTS fk_overlay_file_deltas_snapshot;"
    )
    op.execute(
        "ALTER TABLE overlay_file_deltas DROP CONSTRAINT IF EXISTS "
        "fk_overlay_file_deltas_tenant_domain;"
    )
    op.execute(
        "ALTER TABLE ref_snapshots DROP CONSTRAINT IF EXISTS fk_ref_snapshots_base_snapshot;"
    )
    op.execute(
        "ALTER TABLE ref_snapshots DROP CONSTRAINT IF EXISTS fk_ref_snapshots_tenant_domain;"
    )
    op.execute(
        "ALTER TABLE overlay_file_deltas DROP CONSTRAINT IF EXISTS "
        "uq_overlay_deltas_tenant_snap_path;"
    )
    op.execute(
        "ALTER TABLE ref_snapshots DROP CONSTRAINT IF EXISTS "
        "uq_ref_snapshots_tenant_domain_repo_ref;"
    )
