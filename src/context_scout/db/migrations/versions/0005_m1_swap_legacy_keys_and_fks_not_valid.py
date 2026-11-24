"""Swap legacy single-tenant constraints to composite (tenant_id, domain_id) keys and NOT VALID FKs.

Revision ID: 0005_m1_swap_keys
Revises: 0004_m1_conc_idx
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0005_m1_swap_keys"
down_revision: str | Sequence[str] | None = "0004_m1_conc_idx"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")

    # 1. Drop legacy single-column FKs referencing domain_packs(domain_id)
    op.execute("ALTER TABLE code_files DROP CONSTRAINT IF EXISTS code_files_domain_id_fkey;")
    op.execute("ALTER TABLE doc_chunks DROP CONSTRAINT IF EXISTS doc_chunks_domain_id_fkey;")
    op.execute("ALTER TABLE repositories DROP CONSTRAINT IF EXISTS repositories_domain_id_fkey;")
    op.execute(
        "ALTER TABLE confluence_pages DROP CONSTRAINT IF EXISTS confluence_pages_domain_id_fkey;"
    )
    op.execute("ALTER TABLE jira_issues DROP CONSTRAINT IF EXISTS jira_issues_domain_id_fkey;")
    op.execute(
        "ALTER TABLE indexing_queue DROP CONSTRAINT IF EXISTS indexing_queue_domain_id_fkey;"
    )

    # 2. Drop legacy single-tenant unique constraints & primary key
    op.execute(
        "ALTER TABLE code_files DROP CONSTRAINT IF EXISTS "
        "code_files_domain_id_repo_name_file_path_key;"
    )
    op.execute(
        "ALTER TABLE repositories DROP CONSTRAINT IF EXISTS repositories_domain_id_repo_name_key;"
    )
    op.execute(
        "ALTER TABLE confluence_pages DROP CONSTRAINT IF EXISTS "
        "confluence_pages_domain_id_confluence_page_id_key;"
    )
    op.execute(
        "ALTER TABLE jira_issues DROP CONSTRAINT IF EXISTS jira_issues_domain_id_issue_key_key;"
    )
    op.execute("ALTER TABLE doc_chunks DROP CONSTRAINT IF EXISTS doc_chunks_market_scope_check;")
    op.execute("ALTER TABLE domain_packs DROP CONSTRAINT IF EXISTS domain_packs_pkey;")

    # 3. Promote pre-built CONCURRENTLY unique indexes to composite PK / UNIQUE constraints
    op.execute(
        "ALTER TABLE domain_packs ADD CONSTRAINT domain_packs_pkey "
        "PRIMARY KEY USING INDEX uq_idx_domain_packs_tenant_domain;"
    )
    op.execute(
        "ALTER TABLE code_files ADD CONSTRAINT uq_code_files_tenant_domain_repo_path "
        "UNIQUE USING INDEX uq_idx_code_files_tenant_domain_repo_path;"
    )
    op.execute(
        "ALTER TABLE repositories ADD CONSTRAINT uq_repositories_tenant_domain_repo "
        "UNIQUE USING INDEX uq_idx_repositories_tenant_domain_repo;"
    )
    op.execute(
        "ALTER TABLE confluence_pages ADD CONSTRAINT uq_confluence_tenant_domain_page "
        "UNIQUE USING INDEX uq_idx_confluence_tenant_domain_page;"
    )
    op.execute(
        "ALTER TABLE jira_issues ADD CONSTRAINT uq_jira_tenant_domain_issue "
        "UNIQUE USING INDEX uq_idx_jira_tenant_domain_issue;"
    )

    # 4. Relax doc_chunks market_scope check & attach composite (tenant_id, domain_id) FKs NOT VALID
    op.execute(
        "ALTER TABLE doc_chunks ADD CONSTRAINT doc_chunks_market_scope_check "
        "CHECK (market_scope IS NULL OR length(btrim(market_scope)) > 0) NOT VALID;"
    )
    for table in (
        "domain_sync_manifests",
        "embedding_spaces",
        "code_files",
        "doc_chunks",
        "repositories",
        "confluence_pages",
        "jira_issues",
        "indexing_queue",
    ):
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT fk_{table}_tenant_domain "
            "FOREIGN KEY (tenant_id, domain_id) REFERENCES domain_packs (tenant_id, domain_id) "
            "ON DELETE CASCADE NOT VALID;"
        )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '2s';")
    op.execute("SET LOCAL statement_timeout = '30s';")
    for table in (
        "indexing_queue",
        "jira_issues",
        "confluence_pages",
        "repositories",
        "doc_chunks",
        "code_files",
        "embedding_spaces",
        "domain_sync_manifests",
    ):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS fk_{table}_tenant_domain;")
    op.execute("ALTER TABLE jira_issues DROP CONSTRAINT IF EXISTS uq_jira_tenant_domain_issue;")
    op.execute(
        "ALTER TABLE confluence_pages DROP CONSTRAINT IF EXISTS uq_confluence_tenant_domain_page;"
    )
    op.execute(
        "ALTER TABLE repositories DROP CONSTRAINT IF EXISTS uq_repositories_tenant_domain_repo;"
    )
    op.execute(
        "ALTER TABLE code_files DROP CONSTRAINT IF EXISTS uq_code_files_tenant_domain_repo_path;"
    )
    op.execute("ALTER TABLE domain_packs DROP CONSTRAINT IF EXISTS domain_packs_pkey;")
