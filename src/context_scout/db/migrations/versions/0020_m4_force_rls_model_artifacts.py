"""Enforce ENABLE + FORCE ROW LEVEL SECURITY and runtime grants on model_artifacts.

Revision ID: 0020_m4_art_force_rls
Revises: 0019_m4_art_conc_idx
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0020_m4_art_force_rls"
down_revision: str | Sequence[str] | None = "0019_m4_art_conc_idx"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_PRED = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid"


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '600s';")

    op.execute("ALTER TABLE model_artifacts ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE model_artifacts FORCE ROW LEVEL SECURITY;")
    op.execute("DROP POLICY IF EXISTS rls_model_artifacts_tenant ON model_artifacts;")
    op.execute(
        f"CREATE POLICY rls_model_artifacts_tenant ON model_artifacts "
        f"USING ({_TENANT_PRED}) WITH CHECK ({_TENANT_PRED});"
    )
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON model_artifacts TO scout_app_user;")


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")
    op.execute("DROP POLICY IF EXISTS rls_model_artifacts_tenant ON model_artifacts;")
    op.execute("ALTER TABLE model_artifacts NO FORCE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE model_artifacts DISABLE ROW LEVEL SECURITY;")
