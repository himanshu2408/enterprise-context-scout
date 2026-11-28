-- =============================================================================
-- Enterprise Context Scout: Production Enterprise Schema (Part 2 of 2)
-- Unified with Alembic head (0024_m5_scim_idx_conn_rls): All Remaining Tables + Forced RLS
-- =============================================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------------
-- 1. Repository Catalog (Multi-Tenant)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS repositories (
    repo_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('gitlab', 'github', 'local_git', 'airgap_bundle')),
    remote_url TEXT NOT NULL,
    default_branch TEXT NOT NULL DEFAULT 'main',
    head_commit_sha VARCHAR(40) NOT NULL,
    allowed_scopes TEXT[] NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_repositories_tenant_domain_repo UNIQUE (tenant_id, domain_id, repo_name),
    CONSTRAINT fk_repositories_tenant_domain FOREIGN KEY (tenant_id, domain_id)
        REFERENCES domain_packs(tenant_id, domain_id) ON DELETE CASCADE,
    CONSTRAINT chk_repositories_tenant_nn CHECK (tenant_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_repositories_domain ON repositories (domain_id);
CREATE INDEX IF NOT EXISTS idx_repositories_scopes ON repositories USING GIN (allowed_scopes);

-- Link core code_files & ast_symbols repo_id FKs cleanly when upgraded from 001
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'code_files_repo_id_fkey'
    ) THEN
        ALTER TABLE code_files
            ADD CONSTRAINT code_files_repo_id_fkey FOREIGN KEY (repo_id)
            REFERENCES repositories(repo_id) ON DELETE CASCADE;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'ast_symbols_repo_id_fkey'
    ) THEN
        ALTER TABLE ast_symbols
            ADD CONSTRAINT ast_symbols_repo_id_fkey FOREIGN KEY (repo_id)
            REFERENCES repositories(repo_id) ON DELETE CASCADE;
    END IF;
END
$$;

-- -----------------------------------------------------------------------------
-- 2. Repository Commits History (Merkle Tree Tracking)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS repo_commits (
    commit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    repo_id UUID NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    commit_sha VARCHAR(40) NOT NULL,
    parent_shas VARCHAR(40)[] NOT NULL DEFAULT '{}',
    author_name TEXT NOT NULL,
    author_email TEXT NOT NULL,
    commit_message TEXT NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT repo_commits_repo_id_commit_sha_key UNIQUE (repo_id, commit_sha),
    CONSTRAINT chk_repo_commits_tenant_nn CHECK (tenant_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_repo_commits_sha ON repo_commits (repo_id, commit_sha);
CREATE INDEX IF NOT EXISTS idx_repo_commits_tenant_repo ON repo_commits (tenant_id, repo_id);

-- -----------------------------------------------------------------------------
-- 3. Confluence Spaces & Pages
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS confluence_pages (
    page_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
    confluence_page_id TEXT NOT NULL,
    space_key TEXT NOT NULL,
    title TEXT NOT NULL,
    version_number INTEGER NOT NULL CHECK (version_number >= 1),
    author_username TEXT NOT NULL,
    source_url TEXT NOT NULL,
    allowed_scopes TEXT[] NOT NULL,
    status TEXT NOT NULL DEFAULT 'current',
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    last_modified_at TIMESTAMPTZ NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_confluence_tenant_domain_page UNIQUE (tenant_id, domain_id, confluence_page_id),
    CONSTRAINT fk_confluence_pages_tenant_domain FOREIGN KEY (tenant_id, domain_id)
        REFERENCES domain_packs(tenant_id, domain_id) ON DELETE CASCADE,
    CONSTRAINT chk_confluence_pages_tenant_nn CHECK (tenant_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_confluence_lookup ON confluence_pages (domain_id, space_key, confluence_page_id);
CREATE INDEX IF NOT EXISTS idx_confluence_scopes ON confluence_pages USING GIN (allowed_scopes);

-- -----------------------------------------------------------------------------
-- 4. Jira Projects & Issues (with Epic/Parent Hierarchy)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS jira_issues (
    issue_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
    issue_key TEXT NOT NULL,
    project_key TEXT NOT NULL,
    summary TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    status TEXT NOT NULL,
    resolution TEXT,
    assignee_username TEXT,
    source_url TEXT NOT NULL,
    allowed_scopes TEXT[] NOT NULL,
    epic_key TEXT,
    parent_issue_key TEXT,
    linked_issues_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    linked_mrs_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_jira_tenant_domain_issue UNIQUE (tenant_id, domain_id, issue_key),
    CONSTRAINT fk_jira_issues_tenant_domain FOREIGN KEY (tenant_id, domain_id)
        REFERENCES domain_packs(tenant_id, domain_id) ON DELETE CASCADE,
    CONSTRAINT chk_jira_issues_tenant_nn CHECK (tenant_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_jira_lookup ON jira_issues (domain_id, project_key, issue_key);
CREATE INDEX IF NOT EXISTS idx_jira_scopes ON jira_issues USING GIN (allowed_scopes);
CREATE INDEX IF NOT EXISTS idx_jira_issues_hierarchy ON jira_issues (tenant_id, domain_id, epic_key, parent_issue_key)
    WHERE (epic_key IS NOT NULL OR parent_issue_key IS NOT NULL);

-- -----------------------------------------------------------------------------
-- 5. Transactional Event Ingestion Queue & Dead Letter Queue
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS indexing_queue (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN (
        'gitlab_push', 'github_push', 'gitlab_mr', 'github_pr',
        'confluence_update', 'jira_update', 'cross_repo_relink',
        'watermark_sync', 'reconcile_sweep', 'purge_artifact'
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
    processed_at TIMESTAMPTZ,
    CONSTRAINT fk_indexing_queue_tenant_domain FOREIGN KEY (tenant_id, domain_id)
        REFERENCES domain_packs(tenant_id, domain_id) ON DELETE CASCADE,
    CONSTRAINT chk_indexing_queue_tenant_nn CHECK (tenant_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_indexing_queue_claim ON indexing_queue (scheduled_at, priority DESC, created_at ASC)
    WHERE (status = 'PENDING');
CREATE INDEX IF NOT EXISTS idx_indexing_queue_domain_status ON indexing_queue (domain_id, status);
CREATE INDEX IF NOT EXISTS idx_indexing_queue_tenant_claim ON indexing_queue (tenant_id, status, scheduled_at);

CREATE TABLE IF NOT EXISTS indexing_dead_letter (
    dlq_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    task_id UUID NOT NULL,
    domain_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    failure_reason TEXT NOT NULL,
    stack_trace TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    failed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_indexing_dead_letter_tenant_nn CHECK (tenant_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_indexing_dlq_domain ON indexing_dead_letter (domain_id, failed_at DESC);
CREATE INDEX IF NOT EXISTS idx_indexing_dlq_event ON indexing_dead_letter (event_type, failed_at DESC);
CREATE INDEX IF NOT EXISTS idx_indexing_dlq_tenant_domain ON indexing_dead_letter (tenant_id, domain_id);

-- -----------------------------------------------------------------------------
-- 6. Branch & PR/MR Overlay Snapshots + Deltas
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ref_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
    repo_id UUID,
    repo_name TEXT NOT NULL,
    ref_name TEXT NOT NULL,
    ref_kind TEXT NOT NULL CHECK (ref_kind IN ('base_branch', 'feature_branch', 'merge_request', 'pull_request')),
    mr_iid TEXT,
    base_snapshot_id UUID REFERENCES ref_snapshots(snapshot_id) ON DELETE SET NULL,
    base_commit_sha TEXT NOT NULL,
    head_commit_sha TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'open' CHECK (state IN ('active', 'open', 'merged', 'closed')),
    allowed_scopes TEXT[] NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_ref_snapshots_tenant_domain_repo_ref UNIQUE (tenant_id, domain_id, repo_name, ref_name),
    CONSTRAINT fk_ref_snapshots_tenant_domain FOREIGN KEY (tenant_id, domain_id)
        REFERENCES domain_packs(tenant_id, domain_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ref_snapshots_scopes ON ref_snapshots USING GIN (allowed_scopes);
CREATE INDEX IF NOT EXISTS idx_ref_snapshots_tenant_domain_mr ON ref_snapshots (tenant_id, domain_id, repo_name, mr_iid)
    WHERE (mr_iid IS NOT NULL);

CREATE TABLE IF NOT EXISTS overlay_file_deltas (
    delta_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    snapshot_id UUID NOT NULL REFERENCES ref_snapshots(snapshot_id) ON DELETE CASCADE,
    domain_id TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    previous_file_path TEXT,
    change_type TEXT NOT NULL CHECK (change_type IN ('added', 'modified', 'deleted', 'renamed')),
    base_blob_sha TEXT,
    overlay_blob_sha TEXT,
    language TEXT NOT NULL DEFAULT 'text',
    raw_content TEXT,
    line_count BIGINT NOT NULL DEFAULT 0 CHECK (line_count >= 0),
    ranking_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0 CHECK (ranking_weight >= 0.0 AND ranking_weight <= 2.0),
    is_generated BOOLEAN NOT NULL DEFAULT FALSE,
    is_vendored BOOLEAN NOT NULL DEFAULT FALSE,
    symbols_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    edges_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    allowed_scopes TEXT[] NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_overlay_deltas_tenant_snap_path UNIQUE (tenant_id, snapshot_id, file_path),
    CONSTRAINT fk_overlay_file_deltas_tenant_domain FOREIGN KEY (tenant_id, domain_id)
        REFERENCES domain_packs(tenant_id, domain_id) ON DELETE CASCADE,
    CONSTRAINT overlay_file_deltas_check CHECK (
        (change_type = 'deleted' AND raw_content IS NULL)
        OR (change_type <> 'deleted' AND raw_content IS NOT NULL AND overlay_blob_sha IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_overlay_deltas_scopes ON overlay_file_deltas USING GIN (allowed_scopes);
CREATE INDEX IF NOT EXISTS idx_overlay_deltas_tenant_domain_repo ON overlay_file_deltas (tenant_id, domain_id, repo_name, snapshot_id);
CREATE INDEX IF NOT EXISTS idx_overlay_deltas_trgm ON overlay_file_deltas USING GIN (raw_content gin_trgm_ops)
    WHERE (raw_content IS NOT NULL);

-- -----------------------------------------------------------------------------
-- 7. Runtime Side-Effect & External Callsite Inventory
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS side_effects (
    side_effect_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    file_id UUID REFERENCES code_files(file_id) ON DELETE CASCADE,
    symbol_id UUID REFERENCES ast_symbols(symbol_id) ON DELETE SET NULL,
    file_path TEXT NOT NULL,
    qualified_symbol TEXT NOT NULL,
    start_line BIGINT NOT NULL CHECK (start_line >= 1),
    end_line BIGINT NOT NULL CHECK (end_line >= start_line),
    effect_category TEXT NOT NULL CHECK (effect_category IN (
        'email', 'sms', 'push_notification', 'webhook',
        'pubsub_kafka_event', 'cron_job', 'feature_flag', 'db_write'
    )),
    target_identifier TEXT NOT NULL,
    provider_or_driver TEXT NOT NULL DEFAULT '',
    trigger_condition TEXT NOT NULL DEFAULT '',
    template_or_schema_ref TEXT,
    severity_tier TEXT NOT NULL DEFAULT 'external_io' CHECK (severity_tier IN (
        'external_comms', 'external_io', 'async_messaging',
        'scheduled_job', 'runtime_toggle', 'state_mutation'
    )),
    confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    allowed_scopes TEXT[] NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_side_effects_natural UNIQUE (
        tenant_id, domain_id, repo_name, file_path,
        qualified_symbol, effect_category, target_identifier, start_line
    ),
    CONSTRAINT fk_side_effects_tenant_domain FOREIGN KEY (tenant_id, domain_id)
        REFERENCES domain_packs(tenant_id, domain_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_side_effects_domain_cat ON side_effects (tenant_id, domain_id, effect_category);
CREATE INDEX IF NOT EXISTS idx_side_effects_scopes ON side_effects USING GIN (allowed_scopes);

-- -----------------------------------------------------------------------------
-- 8. 3-Tier Connector Sync Watermarks
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS connector_sync_watermarks (
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    container_key TEXT NOT NULL,
    watermark_cursor TEXT NOT NULL DEFAULT '',
    sync_tier TEXT NOT NULL DEFAULT 'watermark' CHECK (sync_tier IN ('webhook', 'watermark', 'reconcile', 'bundle')),
    items_synced BIGINT NOT NULL DEFAULT 0 CHECK (items_synced >= 0),
    last_webhook_at TIMESTAMPTZ,
    last_watermark_sync_at TIMESTAMPTZ,
    last_reconciliation_at TIMESTAMPTZ,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT connector_sync_watermarks_pkey PRIMARY KEY (tenant_id, domain_id, source_id, container_key),
    CONSTRAINT fk_connector_watermarks_tenant_domain FOREIGN KEY (tenant_id, domain_id)
        REFERENCES domain_packs(tenant_id, domain_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_connector_watermarks_lookup ON connector_sync_watermarks (tenant_id, domain_id, source_id);

-- -----------------------------------------------------------------------------
-- 9. Persistent Investigation Sessions & Compaction Checkpoints
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scout_sessions (
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    session_id TEXT NOT NULL,
    domain_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    allowed_scopes TEXT[] NOT NULL,
    turn_count BIGINT NOT NULL DEFAULT 0 CHECK (turn_count >= 0),
    compacted_summary TEXT NOT NULL DEFAULT '',
    recent_turns_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    pinned_file_blobs_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    pinned_symbols_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    total_tokens_compacted BIGINT NOT NULL DEFAULT 0 CHECK (total_tokens_compacted >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT scout_sessions_pkey PRIMARY KEY (tenant_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_scout_sessions_domain_user ON scout_sessions (tenant_id, domain_id, principal_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_scout_sessions_scopes ON scout_sessions USING GIN (allowed_scopes);

-- -----------------------------------------------------------------------------
-- 10. Signed Multi-LoRA Model Artifacts (Supply-Chain & Training Ceiling)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_artifacts (
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    domain_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    artifact_kind TEXT NOT NULL DEFAULT 'lora_adapter' CHECK (artifact_kind IN ('lora_adapter', 'bi_encoder', 'merged_checkpoint')),
    base_model_id TEXT NOT NULL,
    base_model_revision_sha256 TEXT NOT NULL CHECK (char_length(base_model_revision_sha256) = 64),
    storage_uri TEXT NOT NULL,
    weights_object_key TEXT NOT NULL,
    config_object_key TEXT NOT NULL,
    signature_object_key TEXT NOT NULL DEFAULT '',
    weights_sha256 TEXT NOT NULL CHECK (char_length(weights_sha256) = 64),
    cosign_signature_b64 TEXT NOT NULL DEFAULT '',
    cosign_key_id TEXT NOT NULL DEFAULT 'default',
    lora_rank BIGINT NOT NULL DEFAULT 16 CHECK (lora_rank IN (8, 16, 32)),
    lora_alpha BIGINT NOT NULL DEFAULT 32 CHECK (lora_alpha > 0),
    use_rslora BOOLEAN NOT NULL DEFAULT TRUE CHECK (use_rslora IS TRUE),
    use_dora BOOLEAN NOT NULL DEFAULT FALSE CHECK (use_dora IS FALSE),
    training_scope_ceiling TEXT[] NOT NULL CHECK (cardinality(training_scope_ceiling) >= 1),
    allowed_scopes TEXT[] NOT NULL,
    canary_leak_count BIGINT NOT NULL DEFAULT 0 CHECK (canary_leak_count = 0),
    status TEXT NOT NULL DEFAULT 'staged' CHECK (status IN ('staged', 'active', 'retired', 'revoked')),
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT model_artifacts_pkey PRIMARY KEY (tenant_id, domain_id, artifact_id)
);

CREATE INDEX IF NOT EXISTS idx_model_artifacts_active_lookup ON model_artifacts (tenant_id, is_active, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_artifacts_ceiling_gin ON model_artifacts USING GIN (training_scope_ceiling);
CREATE UNIQUE INDEX IF NOT EXISTS uq_idx_model_artifacts_one_active
    ON model_artifacts (tenant_id, domain_id, artifact_kind) WHERE (is_active = TRUE);

-- -----------------------------------------------------------------------------
-- 11. Tamper-Evident Chained Audit Event Log
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_events (
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    seq_no BIGINT NOT NULL,
    event_id UUID NOT NULL,
    domain_id TEXT NOT NULL DEFAULT 'default',
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'INFO',
    actor_id TEXT NOT NULL,
    actor_role TEXT NOT NULL,
    resource_uri TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT 'allow',
    trace_id TEXT NOT NULL DEFAULT '',
    span_id TEXT NOT NULL DEFAULT '',
    prev_hash_sha256 TEXT NOT NULL,
    event_hash_sha256 TEXT NOT NULL CHECK (char_length(event_hash_sha256) = 64),
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT audit_events_pkey PRIMARY KEY (tenant_id, seq_no)
);

CREATE INDEX IF NOT EXISTS idx_audit_events_domain_created ON audit_events (tenant_id, domain_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_idx_audit_events_event_id ON audit_events (tenant_id, event_id);

-- -----------------------------------------------------------------------------
-- 12. SCIM 2.0 Users & Groups
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scim_users (
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    scim_user_id TEXT NOT NULL,
    domain_id TEXT NOT NULL DEFAULT 'default',
    external_id TEXT NOT NULL DEFAULT '',
    user_name TEXT NOT NULL,
    email TEXT NOT NULL DEFAULT '',
    display_name TEXT NOT NULL DEFAULT '',
    active BOOLEAN NOT NULL DEFAULT TRUE,
    role_override TEXT,
    custom_scopes TEXT[] NOT NULL DEFAULT '{}',
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    emails_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT scim_users_pkey PRIMARY KEY (tenant_id, scim_user_id)
);

CREATE INDEX IF NOT EXISTS idx_scim_users_email_lower ON scim_users (tenant_id, lower(email));
CREATE INDEX IF NOT EXISTS idx_scim_users_emails_gin ON scim_users USING GIN (emails_json jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_scim_users_external_id ON scim_users (tenant_id, external_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_idx_scim_users_username ON scim_users (tenant_id, lower(user_name));

CREATE TABLE IF NOT EXISTS scim_groups (
    tenant_id UUID NOT NULL DEFAULT COALESCE(NULLIF(current_setting('app.tenant_id', true), '')::uuid, '00000000-0000-0000-0000-000000000001'::uuid),
    scim_group_id TEXT NOT NULL,
    domain_id TEXT NOT NULL DEFAULT 'default',
    external_id TEXT NOT NULL DEFAULT '',
    display_name TEXT NOT NULL,
    member_user_ids TEXT[] NOT NULL DEFAULT '{}',
    member_group_ids TEXT[] NOT NULL DEFAULT '{}',
    mapped_scopes TEXT[] NOT NULL DEFAULT '{}',
    mapped_role TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    members_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT scim_groups_pkey PRIMARY KEY (tenant_id, scim_group_id)
);

CREATE INDEX IF NOT EXISTS idx_scim_groups_member_groups_gin ON scim_groups USING GIN (member_group_ids);
CREATE INDEX IF NOT EXISTS idx_scim_groups_members_gin ON scim_groups USING GIN (member_user_ids);
CREATE INDEX IF NOT EXISTS idx_scim_groups_members_json_gin ON scim_groups USING GIN (members_json jsonb_path_ops);

-- -----------------------------------------------------------------------------
-- 13. FORCE Row-Level Security (RLS) on All Enterprise Tables
-- -----------------------------------------------------------------------------
ALTER TABLE repositories ENABLE ROW LEVEL SECURITY;
ALTER TABLE repositories FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_repositories_scope ON repositories;
CREATE POLICY rls_repositories_scope ON repositories
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

ALTER TABLE repo_commits ENABLE ROW LEVEL SECURITY;
ALTER TABLE repo_commits FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_repo_commits_scope ON repo_commits;
CREATE POLICY rls_repo_commits_scope ON repo_commits
    USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND EXISTS (
            SELECT 1 FROM repositories r
            WHERE r.repo_id = repo_commits.repo_id AND r.tenant_id = repo_commits.tenant_id
        )
    )
    WITH CHECK (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND EXISTS (
            SELECT 1 FROM repositories r
            WHERE r.repo_id = repo_commits.repo_id AND r.tenant_id = repo_commits.tenant_id
        )
    );

ALTER TABLE confluence_pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE confluence_pages FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_confluence_scope ON confluence_pages;
CREATE POLICY rls_confluence_scope ON confluence_pages
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

ALTER TABLE jira_issues ENABLE ROW LEVEL SECURITY;
ALTER TABLE jira_issues FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_jira_scope ON jira_issues;
CREATE POLICY rls_jira_scope ON jira_issues
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

ALTER TABLE indexing_queue ENABLE ROW LEVEL SECURITY;
ALTER TABLE indexing_queue FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_indexing_queue_tenant ON indexing_queue;
CREATE POLICY rls_indexing_queue_tenant ON indexing_queue
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);

ALTER TABLE indexing_dead_letter ENABLE ROW LEVEL SECURITY;
ALTER TABLE indexing_dead_letter FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_indexing_dead_letter_tenant ON indexing_dead_letter;
CREATE POLICY rls_indexing_dead_letter_tenant ON indexing_dead_letter
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);

ALTER TABLE ref_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE ref_snapshots FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_ref_snapshots_scope ON ref_snapshots;
CREATE POLICY rls_ref_snapshots_scope ON ref_snapshots
    USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
    )
    WITH CHECK (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
    );

ALTER TABLE overlay_file_deltas ENABLE ROW LEVEL SECURITY;
ALTER TABLE overlay_file_deltas FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_overlay_file_deltas_scope ON overlay_file_deltas;
CREATE POLICY rls_overlay_file_deltas_scope ON overlay_file_deltas
    USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
    )
    WITH CHECK (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
    );

ALTER TABLE side_effects ENABLE ROW LEVEL SECURITY;
ALTER TABLE side_effects FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_side_effects_scope ON side_effects;
CREATE POLICY rls_side_effects_scope ON side_effects
    USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
    )
    WITH CHECK (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
    );

ALTER TABLE connector_sync_watermarks ENABLE ROW LEVEL SECURITY;
ALTER TABLE connector_sync_watermarks FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_connector_sync_watermarks_tenant ON connector_sync_watermarks;
CREATE POLICY rls_connector_sync_watermarks_tenant ON connector_sync_watermarks
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);

ALTER TABLE scout_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE scout_sessions FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_scout_sessions_scope ON scout_sessions;
CREATE POLICY rls_scout_sessions_scope ON scout_sessions
    USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
    )
    WITH CHECK (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
    );

ALTER TABLE model_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_artifacts FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_model_artifacts_tenant ON model_artifacts;
CREATE POLICY rls_model_artifacts_tenant ON model_artifacts
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);

ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_events FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_audit_events_tenant ON audit_events;
CREATE POLICY rls_audit_events_tenant ON audit_events
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);

ALTER TABLE scim_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE scim_users FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_scim_users_tenant ON scim_users;
CREATE POLICY rls_scim_users_tenant ON scim_users
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);

ALTER TABLE scim_groups ENABLE ROW LEVEL SECURITY;
ALTER TABLE scim_groups FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_scim_groups_tenant ON scim_groups;
CREATE POLICY rls_scim_groups_tenant ON scim_groups
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);

-- -----------------------------------------------------------------------------
-- 14. Grant Permissions to scout_app_user Role
-- -----------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON
    repositories,
    repo_commits,
    confluence_pages,
    jira_issues,
    indexing_queue,
    indexing_dead_letter,
    ref_snapshots,
    overlay_file_deltas,
    side_effects,
    connector_sync_watermarks,
    scout_sessions,
    model_artifacts,
    audit_events,
    scim_users,
    scim_groups
TO scout_app_user;

COMMIT;
