-- =============================================================================
-- Enterprise Context Scout: Production Enterprise Schema Migration (Milestone 2)
-- Target: PostgreSQL 17 + pgvector + pg_trgm + btree_gin + Forced RLS
-- File: sql/002_production_enterprise_schema.sql
-- =============================================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------------
-- 1. Repository Catalog (F05)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS repositories (
    repo_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id TEXT NOT NULL REFERENCES domain_packs(domain_id) ON DELETE CASCADE,
    repo_name TEXT NOT NULL,                         -- e.g. "billing/invoice-engine"
    source_type TEXT NOT NULL CHECK (source_type IN ('gitlab', 'github', 'local_git')),
    remote_url TEXT NOT NULL,
    default_branch TEXT NOT NULL DEFAULT 'main',
    head_commit_sha VARCHAR(40) NOT NULL,
    allowed_scopes TEXT[] NOT NULL,                  -- e.g. ARRAY['gitlab:billing/invoice-engine:developer']
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (domain_id, repo_name)
);

CREATE INDEX IF NOT EXISTS idx_repositories_domain ON repositories (domain_id);
CREATE INDEX IF NOT EXISTS idx_repositories_scopes ON repositories USING GIN (allowed_scopes);

-- -----------------------------------------------------------------------------
-- 2. Repository Commits History (Merkle Tree Tracking) (F05 & F08)
-- -----------------------------------------------------------------------------
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

-- -----------------------------------------------------------------------------
-- 3. Confluence Spaces & Pages (F05)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS confluence_pages (
    page_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id TEXT NOT NULL REFERENCES domain_packs(domain_id) ON DELETE CASCADE,
    confluence_page_id TEXT NOT NULL,               -- Upstream page ID e.g. "10928374"
    space_key TEXT NOT NULL,                        -- e.g. "RISK"
    title TEXT NOT NULL,
    version_number INTEGER NOT NULL CHECK (version_number >= 1),
    author_username TEXT NOT NULL,
    source_url TEXT NOT NULL,
    allowed_scopes TEXT[] NOT NULL,
    last_modified_at TIMESTAMPTZ NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (domain_id, confluence_page_id)
);

CREATE INDEX IF NOT EXISTS idx_confluence_lookup ON confluence_pages (domain_id, space_key, confluence_page_id);
CREATE INDEX IF NOT EXISTS idx_confluence_scopes ON confluence_pages USING GIN (allowed_scopes);

-- -----------------------------------------------------------------------------
-- 4. Jira Projects & Issues (F05)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS jira_issues (
    issue_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id TEXT NOT NULL REFERENCES domain_packs(domain_id) ON DELETE CASCADE,
    issue_key TEXT NOT NULL,                        -- e.g. "ENG-4092"
    project_key TEXT NOT NULL,                      -- e.g. "ENG"
    summary TEXT NOT NULL,
    issue_type TEXT NOT NULL,                       -- 'Story', 'Bug', 'Epic', 'Task'
    status TEXT NOT NULL,                           -- 'In Progress', 'Done', 'Closed'
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

-- -----------------------------------------------------------------------------
-- 5. Backward-Compatible Evolution of Existing Tables (F05, F08, F09)
-- -----------------------------------------------------------------------------
-- A. code_files: add repo_id FK (nullable to preserve 001 test inserts)
ALTER TABLE code_files ADD COLUMN IF NOT EXISTS repo_id UUID REFERENCES repositories(repo_id) ON DELETE CASCADE;

-- B. ast_symbols: add repo_id FK, repo_qualified_id, bare_name, expand symbol_kind
ALTER TABLE ast_symbols ADD COLUMN IF NOT EXISTS repo_id UUID REFERENCES repositories(repo_id) ON DELETE CASCADE;
ALTER TABLE ast_symbols ADD COLUMN IF NOT EXISTS repo_qualified_id TEXT;
ALTER TABLE ast_symbols ADD COLUMN IF NOT EXISTS bare_name TEXT;

ALTER TABLE ast_symbols DROP CONSTRAINT IF EXISTS ast_symbols_symbol_kind_check;
ALTER TABLE ast_symbols ADD CONSTRAINT ast_symbols_symbol_kind_check CHECK (symbol_kind IN (
    'function', 'class', 'method', 'interface', 'type_alias',
    'hook', 'slice', 'selector', 'reducer', 'story',
    'tf_resource', 'tf_module', 'gcp_resource'
));

CREATE INDEX IF NOT EXISTS idx_ast_symbols_repo_qual ON ast_symbols (domain_id, repo_qualified_id);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_bare_name ON ast_symbols (bare_name);

-- C. ast_edges: add target_repo_qualified_id, is_cross_repo, expand edge_kind
ALTER TABLE ast_edges ADD COLUMN IF NOT EXISTS target_repo_qualified_id TEXT;
ALTER TABLE ast_edges ADD COLUMN IF NOT EXISTS is_cross_repo BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE ast_edges DROP CONSTRAINT IF EXISTS ast_edges_edge_kind_check;
ALTER TABLE ast_edges ADD CONSTRAINT ast_edges_edge_kind_check CHECK (edge_kind IN (
    'CALLS', 'IMPORTS', 'USES_TYPE', 'INHERITS', 'TESTED_BY',
    'PROVISIONED_BY', 'DISPATCHES', 'SELECTS', 'DOCUMENTS_COMPONENT'
));

CREATE INDEX IF NOT EXISTS idx_ast_edges_target_qual ON ast_edges (target_repo_qualified_id);

-- D. doc_chunks: add enterprise metadata columns & relax market_scope constraint
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'quant_paper';
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS version_str TEXT NOT NULL DEFAULT '1';
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS is_stale BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS stale_reason TEXT;

-- E. confluence_pages: lifecycle status and soft-delete tombstone columns
ALTER TABLE confluence_pages ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'current';
ALTER TABLE confluence_pages ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE doc_chunks ALTER COLUMN market_scope DROP NOT NULL;
ALTER TABLE doc_chunks DROP CONSTRAINT IF EXISTS doc_chunks_market_scope_check;
ALTER TABLE doc_chunks ADD CONSTRAINT doc_chunks_market_scope_check CHECK (
    market_scope IS NULL OR market_scope IN (
        'IN_NSE', 'DE_XETRA', 'US_EQUITY', 'GLOBAL', 'INTERNAL_ADR',
        'ENTERPRISE_DOC', 'CONFLUENCE', 'JIRA'
    )
);

-- -----------------------------------------------------------------------------
-- 6. Transactional Event Ingestion Queue & DLQ (F07)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS indexing_queue (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id TEXT NOT NULL REFERENCES domain_packs(domain_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL CHECK (event_type IN (
        'gitlab_push', 'github_push', 'confluence_update', 'jira_update', 'cross_repo_relink'
    )),
    payload_json JSONB NOT NULL,
    priority INTEGER NOT NULL DEFAULT 10,
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')),
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

CREATE INDEX IF NOT EXISTS idx_indexing_queue_domain_status
ON indexing_queue (domain_id, status);

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

-- -----------------------------------------------------------------------------
-- 7. FORCE Row-Level Security (RLS) on All Tenant & Catalog Tables (F05)
-- -----------------------------------------------------------------------------
ALTER TABLE repositories ENABLE ROW LEVEL SECURITY;
ALTER TABLE repositories FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_repositories_scope ON repositories;
CREATE POLICY rls_repositories_scope ON repositories
    USING (allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ','));

ALTER TABLE confluence_pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE confluence_pages FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_confluence_scope ON confluence_pages;
CREATE POLICY rls_confluence_scope ON confluence_pages
    USING (allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ','));

ALTER TABLE jira_issues ENABLE ROW LEVEL SECURITY;
ALTER TABLE jira_issues FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_jira_scope ON jira_issues;
CREATE POLICY rls_jira_scope ON jira_issues
    USING (allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ','));

-- -----------------------------------------------------------------------------
-- 8. Grant Permissions to scout_app_user Role
-- -----------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON repositories, repo_commits, confluence_pages, jira_issues, indexing_queue, indexing_dead_letter TO scout_app_user;

COMMIT;
