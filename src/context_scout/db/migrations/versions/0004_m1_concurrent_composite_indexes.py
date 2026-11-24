"""Create concurrent composite unique and tenant-leading B-tree indexes.

Revision ID: 0004_m1_conc_idx
Revises: 0003_m1_validate_nn
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0004_m1_conc_idx"
down_revision: str | Sequence[str] | None = "0003_m1_validate_nn"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONCURRENT_INDEX_DDL: tuple[str, ...] = (
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_idx_domain_packs_tenant_domain "
    "ON domain_packs (tenant_id, domain_id);",
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_idx_code_files_tenant_domain_repo_path "
    "ON code_files (tenant_id, domain_id, repo_name, file_path);",
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_idx_repositories_tenant_domain_repo "
    "ON repositories (tenant_id, domain_id, repo_name);",
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_idx_confluence_tenant_domain_page "
    "ON confluence_pages (tenant_id, domain_id, confluence_page_id);",
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_idx_jira_tenant_domain_issue "
    "ON jira_issues (tenant_id, domain_id, issue_key);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ast_symbols_tenant_domain "
    "ON ast_symbols (tenant_id, domain_id);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ast_edges_tenant_domain "
    "ON ast_edges (tenant_id, domain_id);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_doc_chunks_tenant_domain "
    "ON doc_chunks (tenant_id, domain_id);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_repo_commits_tenant_repo "
    "ON repo_commits (tenant_id, repo_id);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_execution_telemetry_tenant_domain "
    "ON execution_telemetry (tenant_id, domain_id);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_indexing_queue_tenant_claim "
    "ON indexing_queue (tenant_id, status, scheduled_at);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_indexing_dlq_tenant_domain "
    "ON indexing_dead_letter (tenant_id, domain_id);",
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_idx_embedding_spaces_one_active "
    "ON embedding_spaces (tenant_id, domain_id) WHERE is_active = TRUE;",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ast_symbols_halfvec_768 "
    "ON ast_symbols USING hnsw ((embedding_halfvec::halfvec(768)) halfvec_cosine_ops) "
    "WHERE vector_dims(embedding_halfvec) = 768;",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_doc_chunks_halfvec_768 "
    "ON doc_chunks USING hnsw ((embedding_halfvec::halfvec(768)) halfvec_cosine_ops) "
    "WHERE vector_dims(embedding_halfvec) = 768;",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ast_symbols_halfvec_1024 "
    "ON ast_symbols USING hnsw ((embedding_halfvec::halfvec(1024)) halfvec_cosine_ops) "
    "WHERE vector_dims(embedding_halfvec) = 1024;",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_doc_chunks_halfvec_1024 "
    "ON doc_chunks USING hnsw ((embedding_halfvec::halfvec(1024)) halfvec_cosine_ops) "
    "WHERE vector_dims(embedding_halfvec) = 1024;",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ast_symbols_shadow_halfvec_768 "
    "ON ast_symbols USING hnsw ((shadow_embedding_halfvec::halfvec(768)) halfvec_cosine_ops) "
    "WHERE vector_dims(shadow_embedding_halfvec) = 768;",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_doc_chunks_shadow_halfvec_768 "
    "ON doc_chunks USING hnsw ((shadow_embedding_halfvec::halfvec(768)) halfvec_cosine_ops) "
    "WHERE vector_dims(shadow_embedding_halfvec) = 768;",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ast_symbols_shadow_halfvec_1024 "
    "ON ast_symbols USING hnsw ((shadow_embedding_halfvec::halfvec(1024)) halfvec_cosine_ops) "
    "WHERE vector_dims(shadow_embedding_halfvec) = 1024;",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_doc_chunks_shadow_halfvec_1024 "
    "ON doc_chunks USING hnsw ((shadow_embedding_halfvec::halfvec(1024)) halfvec_cosine_ops) "
    "WHERE vector_dims(shadow_embedding_halfvec) = 1024;",
)

_INDEX_NAMES: tuple[str, ...] = (
    "uq_idx_domain_packs_tenant_domain",
    "uq_idx_code_files_tenant_domain_repo_path",
    "uq_idx_repositories_tenant_domain_repo",
    "uq_idx_confluence_tenant_domain_page",
    "uq_idx_jira_tenant_domain_issue",
    "idx_ast_symbols_tenant_domain",
    "idx_ast_edges_tenant_domain",
    "idx_doc_chunks_tenant_domain",
    "idx_repo_commits_tenant_repo",
    "idx_execution_telemetry_tenant_domain",
    "idx_indexing_queue_tenant_claim",
    "idx_indexing_dlq_tenant_domain",
    "uq_idx_embedding_spaces_one_active",
    "idx_ast_symbols_halfvec_768",
    "idx_doc_chunks_halfvec_768",
    "idx_ast_symbols_halfvec_1024",
    "idx_doc_chunks_halfvec_1024",
    "idx_ast_symbols_shadow_halfvec_768",
    "idx_doc_chunks_shadow_halfvec_768",
    "idx_ast_symbols_shadow_halfvec_1024",
    "idx_doc_chunks_shadow_halfvec_1024",
)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("SET lock_timeout = '2s';")
        op.execute("SET statement_timeout = '600s';")
        for stmt in _CONCURRENT_INDEX_DDL:
            op.execute(stmt)


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("SET lock_timeout = '2s';")
        op.execute("SET statement_timeout = '120s';")
        for idx_name in reversed(_INDEX_NAMES):
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {idx_name};")
