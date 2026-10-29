-- =============================================================================
-- Enterprise Context Scout: PostgreSQL 17 Schema + HNSW + pg_trgm + Forced RLS
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 1. DomainPack Registry
CREATE TABLE IF NOT EXISTS domain_packs (
    domain_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    lora_adapter_id TEXT NOT NULL,
    primary_truth_source TEXT NOT NULL CHECK (primary_truth_source IN ('code', 'document')),
    freshness_half_life_days INTEGER NOT NULL DEFAULT 180,
    config_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Content-Addressed Git Files (Merkle Blob-SHA Tracking)
CREATE TABLE IF NOT EXISTS code_files (
    file_id UUID PRIMARY KEY,
    domain_id TEXT NOT NULL REFERENCES domain_packs(domain_id) ON DELETE CASCADE,
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

CREATE INDEX IF NOT EXISTS idx_code_files_blob_sha ON code_files (domain_id, repo_name, git_blob_sha);
CREATE INDEX IF NOT EXISTS idx_code_files_scopes ON code_files USING GIN (allowed_scopes);
CREATE INDEX IF NOT EXISTS idx_code_files_trgm ON code_files USING GIN (raw_content gin_trgm_ops);

-- 3. Tree-sitter AST Symbol Nodes (Functions, Classes, Methods, Interfaces)
CREATE TABLE IF NOT EXISTS ast_symbols (
    symbol_id UUID PRIMARY KEY,
    file_id UUID NOT NULL REFERENCES code_files(file_id) ON DELETE CASCADE,
    domain_id TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    symbol_kind TEXT NOT NULL CHECK (symbol_kind IN ('function', 'class', 'method', 'interface', 'type_alias')),
    start_line INTEGER NOT NULL CHECK (start_line >= 1),
    end_line INTEGER NOT NULL CHECK (end_line >= start_line),
    signature_skeleton TEXT NOT NULL,
    contextual_summary TEXT NOT NULL,
    allowed_scopes TEXT[] NOT NULL,
    fts_doc TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', qualified_name || ' ' || signature_skeleton || ' ' || contextual_summary)
    ) STORED,
    embedding vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ast_symbols_fts ON ast_symbols USING GIN (fts_doc);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_trgm ON ast_symbols USING GIN (qualified_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_scopes ON ast_symbols USING GIN (allowed_scopes);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_vec ON ast_symbols USING hnsw (embedding vector_cosine_ops);

-- 4. Directed AST Call & Dependency Graph (SEC-02: Includes allowed_scopes & Forced RLS)
CREATE TABLE IF NOT EXISTS ast_edges (
    edge_id UUID PRIMARY KEY,
    domain_id TEXT NOT NULL,
    source_symbol_id UUID NOT NULL REFERENCES ast_symbols(symbol_id) ON DELETE CASCADE,
    target_qualified_name TEXT NOT NULL,
    target_symbol_id UUID REFERENCES ast_symbols(symbol_id) ON DELETE SET NULL,
    edge_kind TEXT NOT NULL CHECK (edge_kind IN ('CALLS', 'IMPORTS', 'USES_TYPE', 'INHERITS', 'TESTED_BY')),
    allowed_scopes TEXT[] NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ast_edges_source ON ast_edges (source_symbol_id, edge_kind);
CREATE INDEX IF NOT EXISTS idx_ast_edges_target ON ast_edges (target_symbol_id, edge_kind);
CREATE INDEX IF NOT EXISTS idx_ast_edges_scopes ON ast_edges USING GIN (allowed_scopes);

-- 5. Contextual Document & Quant Research Paper Chunks
CREATE TABLE IF NOT EXISTS doc_chunks (
    chunk_id UUID PRIMARY KEY,
    domain_id TEXT NOT NULL REFERENCES domain_packs(domain_id) ON DELETE CASCADE,
    doc_id TEXT NOT NULL,
    doc_title TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    market_scope TEXT NOT NULL CHECK (market_scope IN ('IN_NSE', 'DE_XETRA', 'US_EQUITY', 'GLOBAL', 'INTERNAL_ADR')),
    published_date DATE,
    authority_weight REAL NOT NULL DEFAULT 1.0,
    section_heading TEXT NOT NULL,
    contextual_header TEXT NOT NULL,
    verbatim_content TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    allowed_scopes TEXT[] NOT NULL,
    fts_doc TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', contextual_header || ' ' || section_heading || ' ' || verbatim_content)
    ) STORED,
    embedding vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_fts ON doc_chunks USING GIN (fts_doc);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_trgm ON doc_chunks USING GIN (verbatim_content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_scopes ON doc_chunks USING GIN (allowed_scopes);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_market ON doc_chunks (domain_id, market_scope);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_vec ON doc_chunks USING hnsw (embedding vector_cosine_ops);

-- 6. Enterprise ROI, Token Savings & Context-Miss Flywheel Ledger
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

-- =============================================================================
-- SEC-01 & SEC-02: Enable AND Force Row-Level Security Across All Tenant Tables
-- =============================================================================
ALTER TABLE code_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE code_files FORCE ROW LEVEL SECURITY;

ALTER TABLE ast_symbols ENABLE ROW LEVEL SECURITY;
ALTER TABLE ast_symbols FORCE ROW LEVEL SECURITY;

ALTER TABLE ast_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE ast_edges FORCE ROW LEVEL SECURITY;

ALTER TABLE doc_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE doc_chunks FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS rls_code_files_scope ON code_files;
CREATE POLICY rls_code_files_scope ON code_files
    USING (allowed_scopes && string_to_array(current_setting('app.principal_scopes', true), ','));

DROP POLICY IF EXISTS rls_ast_symbols_scope ON ast_symbols;
CREATE POLICY rls_ast_symbols_scope ON ast_symbols
    USING (allowed_scopes && string_to_array(current_setting('app.principal_scopes', true), ','));

DROP POLICY IF EXISTS rls_ast_edges_scope ON ast_edges;
CREATE POLICY rls_ast_edges_scope ON ast_edges
    USING (allowed_scopes && string_to_array(current_setting('app.principal_scopes', true), ','));

DROP POLICY IF EXISTS rls_doc_chunks_scope ON doc_chunks;
CREATE POLICY rls_doc_chunks_scope ON doc_chunks
    USING (allowed_scopes && string_to_array(current_setting('app.principal_scopes', true), ','));

-- Create non-superuser runtime role (SEC-01)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'scout_app_user') THEN
        CREATE ROLE scout_app_user LOGIN PASSWORD 'scout_local_dev_pw' NOSUPERUSER NOCREATEDB NOCREATEROLE;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO scout_app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON domain_packs, code_files, ast_symbols, ast_edges, doc_chunks, execution_telemetry TO scout_app_user;
