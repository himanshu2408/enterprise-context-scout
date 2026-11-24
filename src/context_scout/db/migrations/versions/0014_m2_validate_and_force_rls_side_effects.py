"""Validate M2-03/04 constraints and enforce ENABLE + FORCE RLS on new tables.

Revision ID: 0014_m2_se_force_rls
Revises: 0013_m2_se_swap_chk
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0014_m2_se_force_rls"
down_revision: str | Sequence[str] | None = "0013_m2_se_swap_chk"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_PRED = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid"
_SCOPE_PRED = (
    "allowed_scopes && "
    "string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')"
)
_ACL_PRED = f"({_TENANT_PRED}) AND ({_SCOPE_PRED})"


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '600s';")

    op.execute("ALTER TABLE side_effects VALIDATE CONSTRAINT fk_side_effects_tenant_domain;")
    op.execute("ALTER TABLE side_effects VALIDATE CONSTRAINT fk_side_effects_file;")
    op.execute("ALTER TABLE side_effects VALIDATE CONSTRAINT fk_side_effects_symbol;")
    op.execute(
        "ALTER TABLE connector_sync_watermarks VALIDATE CONSTRAINT "
        "fk_connector_watermarks_tenant_domain;"
    )
    op.execute("ALTER TABLE ast_edges VALIDATE CONSTRAINT ast_edges_edge_kind_check;")
    op.execute("ALTER TABLE indexing_queue VALIDATE CONSTRAINT indexing_queue_event_type_check;")
    op.execute("ALTER TABLE repositories VALIDATE CONSTRAINT repositories_source_type_check;")

    op.execute("ALTER TABLE side_effects ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE side_effects FORCE ROW LEVEL SECURITY;")
    op.execute("DROP POLICY IF EXISTS rls_side_effects_scope ON side_effects;")
    op.execute(
        f"CREATE POLICY rls_side_effects_scope ON side_effects "
        f"USING ({_ACL_PRED}) WITH CHECK ({_ACL_PRED});"
    )

    op.execute("ALTER TABLE connector_sync_watermarks ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE connector_sync_watermarks FORCE ROW LEVEL SECURITY;")
    op.execute(
        "DROP POLICY IF EXISTS rls_connector_sync_watermarks_tenant ON connector_sync_watermarks;"
    )
    op.execute(
        f"CREATE POLICY rls_connector_sync_watermarks_tenant ON connector_sync_watermarks "
        f"USING ({_TENANT_PRED}) WITH CHECK ({_TENANT_PRED});"
    )

    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON side_effects, connector_sync_watermarks "
        "TO scout_app_user;"
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")
    op.execute(
        "DROP POLICY IF EXISTS rls_connector_sync_watermarks_tenant ON connector_sync_watermarks;"
    )
    op.execute("ALTER TABLE connector_sync_watermarks NO FORCE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE connector_sync_watermarks DISABLE ROW LEVEL SECURITY;")
    op.execute("DROP POLICY IF EXISTS rls_side_effects_scope ON side_effects;")
    op.execute("ALTER TABLE side_effects NO FORCE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE side_effects DISABLE ROW LEVEL SECURITY;")
