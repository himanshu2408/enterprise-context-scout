"""Baseline pre-M1 schema matching sql/001 and sql/002 without embedded secrets.

Revision ID: 0001_legacy_baseline
Revises:
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001_legacy_baseline"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BASELINE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS domain_packs (
    domain_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    lora_adapter_id TEXT NOT NULL,
    primary_truth_source TEXT NOT NULL CHECK (primary_truth_source IN ('code', 'document')),
    freshness_half_life_days INTEGER NOT NULL DEFAULT 180,
    config_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS repositories (
    repo_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id TEXT NOT NULL REFERENCES domain_packs(domain_id) ON DELETE CASCADE,
    repo_name TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('gitlab', 'github', 'local_git')),
    remote_url TEXT NOT NULL,
    default_branch TEXT NOT NULL DEFAULT 'main',
    head_commit_sha VARCHAR(40) NOT NULL,
    allowed_scopes TEXT[] NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (domain_id, repo_name)
);

CREATE INDEX IF NOT EXISTS idx_repositories_domain ON repositories (domain_id);
CREATE INDEX IF NOT EXISTS idx_repositories_scopes ON repositories USING GIN (allowed_scopes);

CREATE TABLE IF NOT EXISTS repo_commits (
    commit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id UUID NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    commit_sha VARCHAR(40) NOT NULL,
    parent_shas VARCHAR(40)[] NOT NULL DEFAULT '{}',
    author_name TEXT NOT NULL,
    author_email TEXT NOT NULL,
    commit_message TEXT NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (repo_id, commit_sha)
);

CREATE INDEX IF NOT EXISTS idx_repo_commits_sha ON repo_commits (repo_id, commit_sha);

CREATE TABLE IF NOT EXISTS confluence_pages (
    page_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id TEXT NOT NULL REFERENCES domain_packs(domain_id) ON DELETE CASCADE,
    confluence_page_id TEXT NOT NULL,
    space_key TEXT NOT NULL,
    title TEXT NOT NULL,
    version_number INTEGER NOT NULL CHECK (version_number >= 1),
    author_username TEXT NOT NULL,
    source_url TEXT NOT NULL,
    allowed_scopes TEXT[] NOT NULL,
    last_modified_at TIMESTAMPTZ NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status TEXT NOT NULL DEFAULT 'current',
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (domain_id, confluence_page_id)
);

CREATE INDEX IF NOT EXISTS idx_confluence_lookup
    ON confluence_pages (domain_id, space_key, confluence_page_id);
CREATE INDEX IF NOT EXISTS idx_confluence_scopes ON confluence_pages USING GIN (allowed_scopes);

CREATE TABLE IF NOT EXISTS jira_issues (
    issue_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id TEXT NOT NULL REFERENCES domain_packs(domain_id) ON DELETE CASCADE,
    issue_key TEXT NOT NULL,
    project_key TEXT NOT NULL,
    summary TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    status TEXT NOT NULL,
    resolution TEXT,
    assignee_username TEXT,
    source_url TEXT NOT NULL,
    allowed_scopes TEXT[] NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (domain_id, issue_key)
);

CREATE INDEX IF NOT EXISTS idx_jira_lookup ON jira_issues (domain_id, project_key, issue_key);
CREATE INDEX IF NOT EXISTS idx_jira_scopes ON jira_issues USING GIN (allowed_scopes);

CREATE TABLE IF NOT EXISTS code_files (
    file_id UUID PRIMARY KEY,
    domain_id TEXT NOT NULL REFERENCES domain_packs(domain_id) ON DELETE CASCADE,
    repo_id UUID REFERENCES repositories(repo_id) ON DELETE CASCADE,
    repo_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    git_commit_sha TEXT NOT NULL,
    git_blob_sha TEXT NOT NULL,
    language TEXT NOT NULL,
    allowed_scopes TEXT[] NOT NULL,
    raw_content TEXT NOT NULL,
    line_count INTEGER NOT NULL CHECK (line_count >= 0),
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (domain_id, repo_name, file_path)
);

CREATE INDEX IF NOT EXISTS idx_code_files_blob_sha
    ON code_files (domain_id, repo_name, git_blob_sha);
CREATE INDEX IF NOT EXISTS idx_code_files_scopes ON code_files USING GIN (allowed_scopes);
CREATE INDEX IF NOT EXISTS idx_code_files_trgm ON code_files USING GIN (raw_content gin_trgm_ops);

CREATE TABLE IF NOT EXISTS ast_symbols (
    symbol_id UUID PRIMARY KEY,
    file_id UUID NOT NULL REFERENCES code_files(file_id) ON DELETE CASCADE,
    domain_id TEXT NOT NULL,
    repo_id UUID REFERENCES repositories(repo_id) ON DELETE CASCADE,
    qualified_name TEXT NOT NULL,
    repo_qualified_id TEXT,
    bare_name TEXT,
    symbol_kind TEXT NOT NULL CHECK (symbol_kind IN (
        'function', 'class', 'method', 'interface', 'type_alias',
        'hook', 'slice', 'selector', 'reducer', 'story',
        'tf_resource', 'tf_module', 'gcp_resource'
    )),
    start_line INTEGER NOT NULL CHECK (start_line >= 1),
    end_line INTEGER NOT NULL CHECK (end_line >= start_line),
    signature_skeleton TEXT NOT NULL,
    contextual_summary TEXT NOT NULL,
    allowed_scopes TEXT[] NOT NULL,
    fts_doc TSVECTOR GENERATED ALWAYS AS (
        to_tsvector(
            'english',
            qualified_name || ' ' || signature_skeleton || ' ' || contextual_summary
        )
    ) STORED,
    embedding vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ast_symbols_file_id ON ast_symbols (file_id);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_domain_qual ON ast_symbols (domain_id, qualified_name);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_repo_qual ON ast_symbols (domain_id, repo_qualified_id);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_bare_name ON ast_symbols (bare_name);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_fts ON ast_symbols USING GIN (fts_doc);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_trgm
    ON ast_symbols USING GIN (qualified_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_scopes ON ast_symbols USING GIN (allowed_scopes);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_vec
    ON ast_symbols USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS ast_edges (
    edge_id UUID PRIMARY KEY,
    domain_id TEXT NOT NULL,
    source_symbol_id UUID NOT NULL REFERENCES ast_symbols(symbol_id) ON DELETE CASCADE,
    target_qualified_name TEXT NOT NULL,
    target_symbol_id UUID REFERENCES ast_symbols(symbol_id) ON DELETE SET NULL,
    edge_kind TEXT NOT NULL CHECK (edge_kind IN (
        'CALLS', 'IMPORTS', 'USES_TYPE', 'INHERITS', 'TESTED_BY',
        'PROVISIONED_BY', 'DISPATCHES', 'SELECTS', 'DOCUMENTS_COMPONENT'
    )),
    allowed_scopes TEXT[] NOT NULL,
    target_repo_qualified_id TEXT,
    is_cross_repo BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_ast_edges_domain ON ast_edges (domain_id);
CREATE INDEX IF NOT EXISTS idx_ast_edges_source ON ast_edges (source_symbol_id, edge_kind);
CREATE INDEX IF NOT EXISTS idx_ast_edges_target ON ast_edges (target_symbol_id, edge_kind);
CREATE INDEX IF NOT EXISTS idx_ast_edges_target_qual ON ast_edges (target_repo_qualified_id);
CREATE INDEX IF NOT EXISTS idx_ast_edges_domain_target_qname
    ON ast_edges (domain_id, target_qualified_name, edge_kind);
CREATE INDEX IF NOT EXISTS idx_ast_edges_domain_source_kind
    ON ast_edges (domain_id, source_symbol_id, edge_kind);
CREATE INDEX IF NOT EXISTS idx_ast_edges_scopes ON ast_edges USING GIN (allowed_scopes);

CREATE TABLE IF NOT EXISTS doc_chunks (
    chunk_id UUID PRIMARY KEY,
    domain_id TEXT NOT NULL REFERENCES domain_packs(domain_id) ON DELETE CASCADE,
    doc_id TEXT NOT NULL,
    doc_title TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    market_scope TEXT CHECK (market_scope IS NULL OR length(btrim(market_scope)) > 0),
    published_date DATE,
    authority_weight REAL NOT NULL DEFAULT 1.0,
    section_heading TEXT NOT NULL,
    contextual_header TEXT NOT NULL,
    verbatim_content TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    allowed_scopes TEXT[] NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'enterprise_doc',
    version_str TEXT NOT NULL DEFAULT '1',
    is_stale BOOLEAN NOT NULL DEFAULT FALSE,
    stale_reason TEXT,
    fts_doc TSVECTOR GENERATED ALWAYS AS (
        to_tsvector(
            'english',
            contextual_header || ' ' || section_heading || ' ' || verbatim_content
        )
    ) STORED,
    embedding vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_fts ON doc_chunks USING GIN (fts_doc);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_trgm
    ON doc_chunks USING GIN (verbatim_content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_scopes ON doc_chunks USING GIN (allowed_scopes);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_market ON doc_chunks (domain_id, market_scope);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_vec
    ON doc_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS execution_telemetry (
    trace_id UUID PRIMARY KEY,
    domain_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    principal_role TEXT NOT NULL,
    execution_mode TEXT NOT NULL CHECK (execution_mode IN ('chat_qa', 'gemini_handoff')),
    local_tokens_read INTEGER NOT NULL CHECK (local_tokens_read >= 0),
    context_pack_tokens_sent INTEGER NOT NULL CHECK (context_pack_tokens_sent >= 0),
    paid_tokens_saved INTEGER NOT NULL,
    estimated_usd_saved NUMERIC(10, 6) NOT NULL,
    preflight_passed_first_try BOOLEAN NOT NULL,
    zero_touch_sufficient BOOLEAN,
    missing_context_requests JSONB NOT NULL DEFAULT '[]'::jsonb,
    latency_ms INTEGER NOT NULL CHECK (latency_ms >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS indexing_queue (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id TEXT NOT NULL REFERENCES domain_packs(domain_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL CHECK (event_type IN (
        'gitlab_push', 'github_push', 'confluence_update', 'jira_update', 'cross_repo_relink'
    )),
    payload_json JSONB NOT NULL,
    priority INTEGER NOT NULL DEFAULT 10,
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK (
        status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')
    ),
    worker_id TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    error_message TEXT,
    scheduled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    processed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_indexing_queue_claim
    ON indexing_queue (scheduled_at, priority DESC, created_at ASC)
    WHERE status = 'PENDING';
CREATE INDEX IF NOT EXISTS idx_indexing_queue_domain_status ON indexing_queue (domain_id, status);

CREATE TABLE IF NOT EXISTS indexing_dead_letter (
    dlq_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL,
    domain_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    failure_reason TEXT NOT NULL,
    stack_trace TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    failed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_indexing_dlq_domain
    ON indexing_dead_letter (domain_id, failed_at DESC);
CREATE INDEX IF NOT EXISTS idx_indexing_dlq_event
    ON indexing_dead_letter (event_type, failed_at DESC);

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'scout_app_user') THEN
        CREATE ROLE scout_app_user LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOINHERIT;
    END IF;
END
$$;
"""


def upgrade() -> None:
    op.execute(_BASELINE_SQL)


def downgrade() -> None:
    for table in (
        "indexing_dead_letter",
        "indexing_queue",
        "execution_telemetry",
        "doc_chunks",
        "ast_edges",
        "ast_symbols",
        "code_files",
        "jira_issues",
        "confluence_pages",
        "repo_commits",
        "repositories",
        "domain_packs",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
