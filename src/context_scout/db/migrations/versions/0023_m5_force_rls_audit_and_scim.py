"""Enforce ENABLE + FORCE ROW LEVEL SECURITY and least-privilege grants on M5 tables.

Revision ID: 0023_m5_audit_force_rls
Revises: 0022_m5_audit_conc_idx
Create Date: 2026-11-19
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0023_m5_audit_force_rls"
down_revision: str | Sequence[str] | None = "0022_m5_audit_conc_idx"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_PRED = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid"


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '600s';")

    for table_name in ("audit_events", "scim_users", "scim_groups"):
        op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY;")
        op.execute(f"DROP POLICY IF EXISTS rls_{table_name}_tenant ON {table_name};")
        op.execute(
            f"CREATE POLICY rls_{table_name}_tenant ON {table_name} "
            f"USING ({_TENANT_PRED}) WITH CHECK ({_TENANT_PRED});"
        )

    # audit_events is kernel-enforced append-only WORM for scout_app_user (no UPDATE/DELETE)
    op.execute("REVOKE ALL ON audit_events FROM scout_app_user;")
    op.execute("GRANT SELECT, INSERT ON audit_events TO scout_app_user;")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON scim_users TO scout_app_user;")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON scim_groups TO scout_app_user;")


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")
    for table_name in ("scim_groups", "scim_users", "audit_events"):
        op.execute(f"DROP POLICY IF EXISTS rls_{table_name}_tenant ON {table_name};")
        op.execute(f"ALTER TABLE {table_name} NO FORCE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table_name} DISABLE ROW LEVEL SECURITY;")
