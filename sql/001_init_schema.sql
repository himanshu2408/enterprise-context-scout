-- =============================================================================
-- Enterprise Context Scout: Canonical PostgreSQL 17 Core Schema (Part 1 of 2)
-- Unified with Alembic head (0024_m5_scim_idx_conn_rls): PG17 + halfvec + Forced RLS
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------------
-- 1. DomainPack Registry
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS domain_packs (
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    lora_adapter_id TEXT NOT NULL,
    primary_truth_source TEXT NOT NULL CHECK (primary_truth_source IN ('code', 'document')),
    freshness_half_life_days INTEGER NOT NULL DEFAULT 180,
    config_json JSONB NOT NULL,
    s3_bundle_uri TEXT,
    bundle_sha256 TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT domain_packs_pkey PRIMARY KEY (tenant_id, domain_id),
    CONSTRAINT chk_domain_packs_tenant_nn CHECK (tenant_id IS NOT NULL),
    CONSTRAINT chk_domain_packs_updated_at_nn CHECK (updated_at IS NOT NULL)
);

-- -----------------------------------------------------------------------------
-- 2. Per-Domain Embedding Space Registry (halfvec + Blue/Green Migration)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embedding_spaces (
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
    space_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'sentence-transformers',
    dimensions BIGINT NOT NULL CHECK (dimensions >= 128 AND dimensions <= 4096),
    quantization TEXT NOT NULL DEFAULT 'halfvec' CHECK (quantization IN ('halfvec', 'vector')),
    status TEXT NOT NULL DEFAULT 'shadow' CHECK (status IN ('active', 'shadow', 'retired')),
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    index_operator_class TEXT NOT NULL DEFAULT 'halfvec_cosine_ops',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_at TIMESTAMPTZ,
    CONSTRAINT embedding_spaces_pkey PRIMARY KEY (tenant_id, domain_id, space_id),
    CONSTRAINT fk_embedding_spaces_tenant_domain FOREIGN KEY (tenant_id, domain_id)
        REFERENCES domain_packs(tenant_id, domain_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_idx_embedding_spaces_one_active
    ON embedding_spaces (tenant_id, domain_id) WHERE (is_active = TRUE);

-- -----------------------------------------------------------------------------
-- 3. Domain Sync Manifests (Deterministic Fingerprint & S3 Bundle Ledger)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS domain_sync_manifests (
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    source_dir TEXT NOT NULL DEFAULT '',
    repo_commit_shas JSONB NOT NULL DEFAULT '{}'::jsonb,
    empty_doc_shas JSONB NOT NULL DEFAULT '{}'::jsonb,
    code_files_indexed BIGINT NOT NULL DEFAULT 0,
    ast_symbols_indexed BIGINT NOT NULL DEFAULT 0,
    ast_edges_indexed BIGINT NOT NULL DEFAULT 0,
    doc_chunks_indexed BIGINT NOT NULL DEFAULT 0,
    s3_manifest_uri TEXT,
    manifest_sha256 TEXT,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT domain_sync_manifests_pkey PRIMARY KEY (tenant_id, domain_id),
    CONSTRAINT fk_domain_sync_manifests_tenant_domain FOREIGN KEY (tenant_id, domain_id)
        REFERENCES domain_packs(tenant_id, domain_id) ON DELETE CASCADE
);

-- -----------------------------------------------------------------------------
-- 4. Content-Addressed Git Files (Merkle Blob-SHA Tracking)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS code_files (
    file_id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
    repo_id UUID,
    repo_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    git_commit_sha TEXT NOT NULL,
    git_blob_sha TEXT NOT NULL,
    language TEXT NOT NULL,
    allowed_scopes TEXT[] NOT NULL,
    raw_content TEXT NOT NULL,
    line_count INTEGER NOT NULL CHECK (line_count >= 0),
    ranking_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    is_generated BOOLEAN NOT NULL DEFAULT FALSE,
    is_vendored BOOLEAN NOT NULL DEFAULT FALSE,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_code_files_tenant_domain_repo_path UNIQUE (tenant_id, domain_id, repo_name, file_path),
    CONSTRAINT fk_code_files_tenant_domain FOREIGN KEY (tenant_id, domain_id)
        REFERENCES domain_packs(tenant_id, domain_id) ON DELETE CASCADE,
    CONSTRAINT chk_code_files_tenant_nn CHECK (tenant_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_code_files_blob_sha ON code_files (domain_id, repo_name, git_blob_sha);
CREATE INDEX IF NOT EXISTS idx_code_files_scopes ON code_files USING GIN (allowed_scopes);
CREATE INDEX IF NOT EXISTS idx_code_files_trgm ON code_files USING GIN (raw_content gin_trgm_ops);

-- -----------------------------------------------------------------------------
-- 5. Tree-sitter AST Symbol Nodes (Polyglot Kinds + Dual-Space halfvec HNSW)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ast_symbols (
    symbol_id UUID PRIMARY KEY,
    file_id UUID NOT NULL REFERENCES code_files(file_id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
    repo_id UUID,
    repo_qualified_id TEXT,
    bare_name TEXT,
    qualified_name TEXT NOT NULL,
    symbol_kind TEXT NOT NULL CHECK (symbol_kind IN (
        'function', 'class', 'method', 'interface', 'type_alias',
        'hook', 'slice', 'selector', 'reducer', 'story',
        'tf_resource', 'tf_module', 'gcp_resource',
        'component', 'system', 'team', 'service', 'module', 'api', 'domain'
    )),
    start_line INTEGER NOT NULL CHECK (start_line >= 1),
    end_line INTEGER NOT NULL CHECK (end_line >= start_line),
    signature_skeleton TEXT NOT NULL,
    contextual_summary TEXT NOT NULL,
    allowed_scopes TEXT[] NOT NULL,
    fts_doc TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', qualified_name || ' ' || signature_skeleton || ' ' || contextual_summary)
    ) STORED,
    embedding vector(768) NOT NULL,
    embedding_space_id TEXT DEFAULT 'embeddinggemma-2-768',
    embedding_halfvec halfvec,
    shadow_embedding_space_id TEXT,
    shadow_embedding_halfvec halfvec,
    CONSTRAINT chk_ast_symbols_tenant_nn CHECK (tenant_id IS NOT NULL),
    CONSTRAINT chk_ast_symbols_nonzero_vec CHECK (
        vector_norm(embedding) > 0.0
        AND (embedding_halfvec IS NULL OR l2_norm(embedding_halfvec) > 0.0)
        AND (shadow_embedding_halfvec IS NULL OR l2_norm(shadow_embedding_halfvec) > 0.0)
    )
);

CREATE INDEX IF NOT EXISTS idx_ast_symbols_file_id ON ast_symbols (file_id);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_domain_qual ON ast_symbols (domain_id, qualified_name);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_repo_qual ON ast_symbols (domain_id, repo_qualified_id);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_bare_name ON ast_symbols (bare_name);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_tenant_domain ON ast_symbols (tenant_id, domain_id);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_fts ON ast_symbols USING GIN (fts_doc);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_trgm ON ast_symbols USING GIN (qualified_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_scopes ON ast_symbols USING GIN (allowed_scopes);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_vec ON ast_symbols USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_halfvec_768 ON ast_symbols USING hnsw ((embedding_halfvec::halfvec(768)) halfvec_cosine_ops) WHERE (vector_dims(embedding_halfvec) = 768);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_halfvec_1024 ON ast_symbols USING hnsw ((embedding_halfvec::halfvec(1024)) halfvec_cosine_ops) WHERE (vector_dims(embedding_halfvec) = 1024);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_shadow_halfvec_768 ON ast_symbols USING hnsw ((shadow_embedding_halfvec::halfvec(768)) halfvec_cosine_ops) WHERE (vector_dims(shadow_embedding_halfvec) = 768);
CREATE INDEX IF NOT EXISTS idx_ast_symbols_shadow_halfvec_1024 ON ast_symbols USING hnsw ((shadow_embedding_halfvec::halfvec(1024)) halfvec_cosine_ops) WHERE (vector_dims(shadow_embedding_halfvec) = 1024);

-- -----------------------------------------------------------------------------
-- 6. Directed AST Call & Dependency Graph (Forced RLS + Cross-Repo Edges)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ast_edges (
    edge_id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
    source_symbol_id UUID NOT NULL REFERENCES ast_symbols(symbol_id) ON DELETE CASCADE,
    target_qualified_name TEXT NOT NULL,
    target_symbol_id UUID REFERENCES ast_symbols(symbol_id) ON DELETE SET NULL,
    target_repo_qualified_id TEXT,
    is_cross_repo BOOLEAN NOT NULL DEFAULT FALSE,
    edge_kind TEXT NOT NULL CHECK (edge_kind IN (
        'CALLS', 'IMPORTS', 'USES_TYPE', 'INHERITS', 'TESTED_BY',
        'PROVISIONED_BY', 'DISPATCHES', 'SELECTS', 'DOCUMENTS_COMPONENT',
        'OWNED_BY_TEAM', 'PART_OF_SYSTEM', 'PUBLISHES_EVENT', 'SUBSCRIBES_EVENT',
        'READS_TABLE', 'WRITES_TABLE', 'RENDERS_TEMPLATE', 'IMPLEMENTS_SPEC',
        'IMPLEMENTS_ISSUE', 'BINDS_SERVICE_ACCOUNT', 'ACCESSES_SECRET'
    )),
    allowed_scopes TEXT[] NOT NULL,
    CONSTRAINT chk_ast_edges_tenant_nn CHECK (tenant_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_ast_edges_domain ON ast_edges (domain_id);
CREATE INDEX IF NOT EXISTS idx_ast_edges_tenant_domain ON ast_edges (tenant_id, domain_id);
CREATE INDEX IF NOT EXISTS idx_ast_edges_source ON ast_edges (source_symbol_id, edge_kind);
CREATE INDEX IF NOT EXISTS idx_ast_edges_target ON ast_edges (target_symbol_id, edge_kind);
CREATE INDEX IF NOT EXISTS idx_ast_edges_target_qual ON ast_edges (target_repo_qualified_id);
CREATE INDEX IF NOT EXISTS idx_ast_edges_domain_target_qname ON ast_edges (domain_id, target_qualified_name, edge_kind);
CREATE INDEX IF NOT EXISTS idx_ast_edges_domain_source_kind ON ast_edges (domain_id, source_symbol_id, edge_kind);
CREATE INDEX IF NOT EXISTS idx_ast_edges_scopes ON ast_edges USING GIN (allowed_scopes);

-- -----------------------------------------------------------------------------
-- 7. Contextual Enterprise Document & Research Chunks
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS doc_chunks (
    chunk_id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
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
        to_tsvector('english', contextual_header || ' ' || section_heading || ' ' || verbatim_content)
    ) STORED,
    embedding vector(768) NOT NULL,
    embedding_space_id TEXT DEFAULT 'embeddinggemma-2-768',
    embedding_halfvec halfvec,
    shadow_embedding_space_id TEXT,
    shadow_embedding_halfvec halfvec,
    CONSTRAINT fk_doc_chunks_tenant_domain FOREIGN KEY (tenant_id, domain_id)
        REFERENCES domain_packs(tenant_id, domain_id) ON DELETE CASCADE,
    CONSTRAINT chk_doc_chunks_tenant_nn CHECK (tenant_id IS NOT NULL),
    CONSTRAINT chk_doc_chunks_nonzero_vec CHECK (
        vector_norm(embedding) > 0.0
        AND (embedding_halfvec IS NULL OR l2_norm(embedding_halfvec) > 0.0)
        AND (shadow_embedding_halfvec IS NULL OR l2_norm(shadow_embedding_halfvec) > 0.0)
    )
);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_fts ON doc_chunks USING GIN (fts_doc);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_trgm ON doc_chunks USING GIN (verbatim_content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_scopes ON doc_chunks USING GIN (allowed_scopes);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_market ON doc_chunks (domain_id, market_scope);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_tenant_domain ON doc_chunks (tenant_id, domain_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_vec ON doc_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_halfvec_768 ON doc_chunks USING hnsw ((embedding_halfvec::halfvec(768)) halfvec_cosine_ops) WHERE (vector_dims(embedding_halfvec) = 768);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_halfvec_1024 ON doc_chunks USING hnsw ((embedding_halfvec::halfvec(1024)) halfvec_cosine_ops) WHERE (vector_dims(embedding_halfvec) = 1024);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_shadow_halfvec_768 ON doc_chunks USING hnsw ((shadow_embedding_halfvec::halfvec(768)) halfvec_cosine_ops) WHERE (vector_dims(shadow_embedding_halfvec) = 768);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_shadow_halfvec_1024 ON doc_chunks USING hnsw ((shadow_embedding_halfvec::halfvec(1024)) halfvec_cosine_ops) WHERE (vector_dims(shadow_embedding_halfvec) = 1024);

-- -----------------------------------------------------------------------------
-- 8. Enterprise ROI, Token Savings & Context-Miss Flywheel Ledger
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS execution_telemetry (
    trace_id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
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
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_execution_telemetry_tenant_nn CHECK (tenant_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_execution_telemetry_tenant_domain ON execution_telemetry (tenant_id, domain_id);

-- =============================================================================
-- Default halfvec Trigger Function
-- =============================================================================
CREATE OR REPLACE FUNCTION sync_default_embedding_halfvec() RETURNS trigger AS $$
BEGIN
    IF NEW.embedding_space_id IS NULL OR btrim(NEW.embedding_space_id) = '' THEN
        NEW.embedding_space_id := 'embeddinggemma-2-768';
    END IF;
    IF NEW.embedding IS NOT NULL AND (
        NEW.embedding_halfvec IS NULL
        OR (
            TG_OP = 'UPDATE'
            AND NEW.embedding IS DISTINCT FROM OLD.embedding
            AND NEW.embedding_halfvec IS NOT DISTINCT FROM OLD.embedding_halfvec
        )
    ) THEN
        NEW.embedding_halfvec := NEW.embedding::halfvec;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_ast_symbols_sync_halfvec ON ast_symbols;
CREATE TRIGGER trg_ast_symbols_sync_halfvec
    BEFORE INSERT OR UPDATE OF embedding, embedding_halfvec, embedding_space_id ON ast_symbols
    FOR EACH ROW EXECUTE FUNCTION sync_default_embedding_halfvec();

DROP TRIGGER IF EXISTS trg_doc_chunks_sync_halfvec ON doc_chunks;
CREATE TRIGGER trg_doc_chunks_sync_halfvec
    BEFORE INSERT OR UPDATE OF embedding, embedding_halfvec, embedding_space_id ON doc_chunks
    FOR EACH ROW EXECUTE FUNCTION sync_default_embedding_halfvec();

-- =============================================================================
-- SEC-01 & SEC-02: Enable AND Force Row-Level Security Across All Core Tables
-- =============================================================================
ALTER TABLE domain_packs ENABLE ROW LEVEL SECURITY;
ALTER TABLE domain_packs FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_domain_packs_tenant ON domain_packs;
CREATE POLICY rls_domain_packs_tenant ON domain_packs
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);

ALTER TABLE embedding_spaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE embedding_spaces FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_embedding_spaces_tenant ON embedding_spaces;
CREATE POLICY rls_embedding_spaces_tenant ON embedding_spaces
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);

ALTER TABLE domain_sync_manifests ENABLE ROW LEVEL SECURITY;
ALTER TABLE domain_sync_manifests FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_domain_sync_manifests_tenant ON domain_sync_manifests;
CREATE POLICY rls_domain_sync_manifests_tenant ON domain_sync_manifests
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);

ALTER TABLE code_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE code_files FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_code_files_scope ON code_files;
CREATE POLICY rls_code_files_scope ON code_files
    USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND (
            allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
            OR 'system:connector_ingest' = ANY (string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ','))
        )
    )
    WITH CHECK (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND (
            allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
            OR 'system:connector_ingest' = ANY (string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ','))
        )
    );

ALTER TABLE ast_symbols ENABLE ROW LEVEL SECURITY;
ALTER TABLE ast_symbols FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_ast_symbols_scope ON ast_symbols;
CREATE POLICY rls_ast_symbols_scope ON ast_symbols
    USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND (
            allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
            OR 'system:connector_ingest' = ANY (string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ','))
        )
    )
    WITH CHECK (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND (
            allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
            OR 'system:connector_ingest' = ANY (string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ','))
        )
    );

ALTER TABLE ast_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE ast_edges FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_ast_edges_scope ON ast_edges;
CREATE POLICY rls_ast_edges_scope ON ast_edges
    USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND (
            allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
            OR 'system:connector_ingest' = ANY (string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ','))
        )
    )
    WITH CHECK (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND (
            allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
            OR 'system:connector_ingest' = ANY (string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ','))
        )
    );

ALTER TABLE doc_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE doc_chunks FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_doc_chunks_scope ON doc_chunks;
CREATE POLICY rls_doc_chunks_scope ON doc_chunks
    USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND (
            allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
            OR 'system:connector_ingest' = ANY (string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ','))
        )
    )
    WITH CHECK (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND (
            allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
            OR 'system:connector_ingest' = ANY (string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ','))
        )
    );

ALTER TABLE execution_telemetry ENABLE ROW LEVEL SECURITY;
ALTER TABLE execution_telemetry FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_execution_telemetry_tenant ON execution_telemetry;
CREATE POLICY rls_execution_telemetry_tenant ON execution_telemetry
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);

-- Create non-superuser runtime role (SEC-01)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'scout_app_user') THEN
        CREATE ROLE scout_app_user LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOINHERIT;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO scout_app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON
    domain_packs,
    embedding_spaces,
    domain_sync_manifests,
    code_files,
    ast_symbols,
    ast_edges,
    doc_chunks,
    execution_telemetry
TO scout_app_user;
