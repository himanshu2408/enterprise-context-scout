"""Validate M2 constraints and enforce ENABLE + FORCE RLS on overlay tables.

Revision ID: 0010_m2_force_rls
Revises: 0009_m2_swap_checks
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0010_m2_force_rls"
down_revision: str | Sequence[str] | None = "0009_m2_swap_checks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_PRED = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid"
_SCOPE_PRED = (
    "allowed_scopes && "
    "string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')"
)
_ACL_PRED = f"({_TENANT_PRED}) AND ({_SCOPE_PRED})"

_OVERLAY_TABLES: tuple[tuple[str, str], ...] = (
    ("ref_snapshots", "rls_ref_snapshots_scope"),
    ("overlay_file_deltas", "rls_overlay_file_deltas_scope"),
)


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '600s';")

    op.execute("ALTER TABLE code_files VALIDATE CONSTRAINT chk_code_files_ranking_weight;")
    op.execute("ALTER TABLE ast_symbols VALIDATE CONSTRAINT ast_symbols_symbol_kind_check;")
    op.execute("ALTER TABLE ast_edges VALIDATE CONSTRAINT ast_edges_edge_kind_check;")
    op.execute("ALTER TABLE ref_snapshots VALIDATE CONSTRAINT fk_ref_snapshots_tenant_domain;")
    op.execute("ALTER TABLE ref_snapshots VALIDATE CONSTRAINT fk_ref_snapshots_base_snapshot;")
    op.execute(
        "ALTER TABLE overlay_file_deltas VALIDATE CONSTRAINT fk_overlay_file_deltas_tenant_domain;"
    )
    op.execute(
        "ALTER TABLE overlay_file_deltas VALIDATE CONSTRAINT fk_overlay_file_deltas_snapshot;"
    )

    for table, policy_name in _OVERLAY_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;")
        op.execute(f"DROP POLICY IF EXISTS {policy_name} ON {table};")
        op.execute(
            f"CREATE POLICY {policy_name} ON {table} USING ({_ACL_PRED}) WITH CHECK ({_ACL_PRED});"
        )

    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON ref_snapshots, overlay_file_deltas "
        "TO scout_app_user;"
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")
    for table, policy_name in reversed(_OVERLAY_TABLES):
        op.execute(f"DROP POLICY IF EXISTS {policy_name} ON {table};")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;")
