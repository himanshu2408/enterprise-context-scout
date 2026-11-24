"""Enforce ENABLE + FORCE ROW LEVEL SECURITY and runtime grants on scout_sessions.

Revision ID: 0017_m3_sess_force_rls
Revises: 0016_m3_sess_conc_idx
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0017_m3_sess_force_rls"
down_revision: str | Sequence[str] | None = "0016_m3_sess_conc_idx"
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

    op.execute("ALTER TABLE scout_sessions ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE scout_sessions FORCE ROW LEVEL SECURITY;")
    op.execute("DROP POLICY IF EXISTS rls_scout_sessions_scope ON scout_sessions;")
    op.execute(
        f"CREATE POLICY rls_scout_sessions_scope ON scout_sessions "
        f"USING ({_ACL_PRED}) WITH CHECK ({_ACL_PRED});"
    )
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON scout_sessions TO scout_app_user;")


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")
    op.execute("DROP POLICY IF EXISTS rls_scout_sessions_scope ON scout_sessions;")
    op.execute("ALTER TABLE scout_sessions NO FORCE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE scout_sessions DISABLE ROW LEVEL SECURITY;")
