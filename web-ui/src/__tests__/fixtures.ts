import { vi } from "vitest";
import { clearCachedAuthSession, setRuntimeAuthMode } from "../api/client";
import {
  AdminPurgeResponse,
  AuthSessionResponse,
  BackendDomainDetail,
  BranchOverlayOption,
  ContextPack,
  DocCitation,
  DomainInfo,
  HandoffApplyResponse,
  HandoffPreviewResponse,
  HealthResponse,
  InsufficientContextDiagnostic,
  PipelineStageSpec,
  PrincipalContext,
  ROIMetricsResponse,
  ScoutQueryResponse,
  ScoutSessionSummary,
  ScoutTokenSavingsBreakdown,
  SideEffectMatrixItem,
  SSEStageEvent,
  StageExecutionTrace,
  WorkspaceConfigResponse,
} from "../types/contracts";

export const TEST_ENGINEER_PRINCIPAL: PrincipalContext = {
  user_id: "test_engineer_01",
  role: "engineer",
  allowed_scopes: ["public_eng", "public_docs"],
  max_session_paid_tokens: 8000,
};

export const TEST_HEALTH_RESPONSE: HealthResponse = {
  status: "healthy",
  database_connected: true,
  database_status: "PostgreSQL 17.11 RLS Enforced",
  registered_domains: [
    "code_architecture_scout",
    "enterprise_docs_scout",
    "polyglot_unseen_v2",
  ],
  adapter_pool_status: {
    code_architecture_scout: {
      domain_id: "code_architecture_scout",
      is_valid: true,
      adapter_path: "adapters/code_architecture_scout",
      safetensors_bytes: 18432000,
      r: 8,
      lora_alpha: 16,
      is_moe: true,
      final_loss: 0.72,
      error_message: null,
      hot_swap_latency_ms: 0.02,
    },
    enterprise_docs_scout: {
      domain_id: "enterprise_docs_scout",
      is_valid: true,
      adapter_path: "adapters/enterprise_docs_scout",
      safetensors_bytes: 18432000,
      r: 8,
      lora_alpha: 16,
      is_moe: true,
      final_loss: 0.68,
      error_message: null,
      hot_swap_latency_ms: 0.02,
    },
  },
  hardware_profile: {
    device: "Apple Silicon MLX",
    unified_memory: "64GB",
    max_kv_tokens: 16384,
    quantization: "4-bit Affine",
    lora_slots_allocated: 2,
    lora_slots_capacity: 4,
  },
  timestamp: "2026-11-14T10:00:00Z",
};

export const TEST_BACKEND_DOMAINS: BackendDomainDetail[] = [
  {
    domain_id: "code_architecture_scout",
    display_name: "Code Architecture & AST",
    description: "AST-aware Python slicing & token budget contracts.",
    modalities: ["code"],
    sample_queries: [
      "Inspect DeterministicSlicer and TokenBudgetResult contracts for context packing.",
      "Find direct callers of compute_slice_bounds in slicer.",
    ],
    lora_enabled: true,
    default_scopes: ["public_eng"],
    doc_chunks_count: 5,
    documents_count: 2,
    code_files_count: 14,
    ast_symbols_count: 3443,
    manifest: {
      domain_id: "code_architecture_scout",
      display_name: "Code Architecture & AST",
      base_model_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
      lora_adapter_id: "code_architecture_scout_v1",
      lora_adapter_path: "adapters/code_architecture_scout",
      authority_rules: { primary_truth_source: "code" },
      output_contract: "ImplementationContextPack",
    },
    adapter_verification: TEST_HEALTH_RESPONSE.adapter_pool_status.code_architecture_scout,
    training_metrics: { final_loss: 0.72 },
  },
  {
    domain_id: "enterprise_docs_scout",
    display_name: "Enterprise Architecture Docs & ADRs",
    description: "System architecture, RLS policies, and design specifications.",
    modalities: ["document"],
    sample_queries: [
      "What does the RLS policy ADR say about tenant isolation and fail-closed defaults?",
    ],
    lora_enabled: false,
    default_scopes: ["public_docs"],
    doc_chunks_count: 14188,
    documents_count: 42,
    code_files_count: 0,
    ast_symbols_count: 0,
    manifest: {
      domain_id: "enterprise_docs_scout",
      display_name: "Enterprise Architecture Docs & ADRs",
      base_model_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
      lora_adapter_id: "enterprise_docs_scout_v1",
      lora_adapter_path: "adapters/enterprise_docs_scout",
      authority_rules: { primary_truth_source: "document" },
      output_contract: "DomainKnowledgeBrief",
    },
    adapter_verification: TEST_HEALTH_RESPONSE.adapter_pool_status.enterprise_docs_scout,
    training_metrics: { final_loss: 0.68 },
  },
  {
    domain_id: "polyglot_unseen_v2",
    display_name: "Zero-Shot Domain Verification",
    description: "Polyglot HTTPX & FastAPI codebase verification.",
    modalities: ["code"],
    sample_queries: [
      "How does HTTPX configure connection pool limits and keepalive timeouts?",
    ],
    lora_enabled: true,
    default_scopes: ["public_eng"],
    doc_chunks_count: 0,
    documents_count: 0,
    code_files_count: 180,
    ast_symbols_count: 50260,
    manifest: {
      domain_id: "polyglot_unseen_v2",
      display_name: "Zero-Shot Domain Verification",
      base_model_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
      lora_adapter_id: "polyglot_unseen_v2_base",
      lora_adapter_path: "",
      authority_rules: { primary_truth_source: "code" },
      output_contract: "ImplementationContextPack",
    },
    adapter_verification: {
      domain_id: "polyglot_unseen_v2",
      is_valid: true,
      adapter_path: "",
      safetensors_bytes: 0,
      r: 0,
      lora_alpha: 0,
      is_moe: false,
      final_loss: null,
      error_message: null,
      hot_swap_latency_ms: null,
    },
    training_metrics: null,
  },
];

export const TEST_PIPELINE_STAGES: PipelineStageSpec[] = [
  {
    stage_id: 1,
    number: "01",
    title: "Auth, Guardrail & Query Plan",
    summary: "Enforce principal RBAC scopes, ShieldGemma screen, and QueryPlan decomposition.",
    what_happens: "Binds PrincipalContext to PostgreSQL RLS scopes and decomposes concepts into facets.",
    gate_condition: "Fail closed on security violation or empty authorized scopes ($0).",
  },
  {
    stage_id: 2,
    number: "02",
    title: "Multi-Domain Hybrid Retrieval & AST Closure",
    summary: "Run calibrated PG17 RRF and 1-2 hop AST dependency expansion.",
    what_happens: "Probes authorized DomainPacks with cosine + BM25 + trigram floor filtering.",
    gate_condition: "Reject hits below relevance floor; rank domains by calibrated evidence.",
  },
  {
    stage_id: 3,
    number: "03",
    title: "Local Gemma-4 Synthesis & Facet Grounding",
    summary: "Synthesize full-chunk evidence with base Gemma-4 or domain QLoRA.",
    what_happens: "Feeds full retrieved chunks to local Gemma-4 and scores per-facet grounding.",
    gate_condition: "Verify sufficiency_verdict and cohesive same-chunk facet grounding.",
  },
  {
    stage_id: 4,
    number: "04",
    title: "Pre-Flight Sufficiency Gate",
    summary: "Deterministic verification gate: $0 abstention or verified handoff.",
    what_happens: "Audits symbol/file provenance, calibrated retrieval confidence, and facet coverage.",
    gate_condition: "Abstain with $0 paid_tokens_spent if insufficient or below relevance floor.",
  },
];

export const TEST_STAGE_TRACES: StageExecutionTrace[] = [
  {
    stage: 1,
    stage_name: "Auth, Guardrail & Query Plan",
    status: "completed",
    elapsed_ms: 12,
    summary: "Principal engineer authorized; 2 facets decomposed.",
    principal_id: "test_engineer_01",
    principal_scopes: ["public_eng", "public_docs"],
    guardrail_verdict: "SAFE",
    query_type: "code_architecture",
    core_subject: "DeterministicSlicer",
    required_facets: ["slice bounds", "TokenBudgetResult"],
    target_domains: ["code_architecture_scout"],
  },
  {
    stage: 2,
    stage_name: "Multi-Domain Hybrid Retrieval & AST Closure",
    status: "completed",
    elapsed_ms: 48,
    summary: "Selected code_architecture_scout (2 accepted hits, 1 AST hop).",
    ast_one_hop_count: 1,
    ast_two_hop_count: 0,
    domain_probes: [
      {
        domain_id: "code_architecture_scout",
        display_name: "Code Architecture & AST",
        authorized: true,
        probed_hits: 4,
        accepted_hits: 2,
        rejected_hits: 2,
        max_cosine_similarity: 0.89,
        max_calibrated_score: 0.91,
        selected: true,
        selection_reason: "Highest calibrated code + AST match",
        top_hits: [
          {
            item_id: "hit_opt_1",
            domain_id: "code_architecture_scout",
            title: "slicer/bounds.py",
            modality: "code",
            cosine_similarity: 0.89,
            bm25_score: 0.84,
            trigram_score: 0.78,
            calibrated_score: 0.91,
            accepted: true,
          },
        ],
      },
    ],
  },
  {
    stage: 3,
    stage_name: "Local Gemma-4 Synthesis & Facet Grounding",
    status: "completed",
    elapsed_ms: 140,
    summary: "Synthesized 2 chunks via code_architecture_scout_v1 QLoRA.",
    model_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
    adapter_mode: "QLoRA (code_architecture_scout_v1)",
    chunks_fed: 2,
    local_tokens_read: 42000,
    sufficiency_verdict: "sufficient",
    grounded_facets: 2,
    total_facets: 2,
  },
  {
    stage: 4,
    stage_name: "Pre-Flight Sufficiency Gate",
    status: "completed",
    elapsed_ms: 8,
    summary: "All deterministic pre-flight checks passed (handoff_ready).",
    final_decision: "handoff_ready",
    paid_tokens_spent: 0,
    checks: [
      {
        check_name: "symbol_provenance",
        passed: true,
        observed: "2/2 verified",
        threshold: "100%",
        details: "All AST symbols anchored to commit SHA",
      },
      {
        check_name: "calibrated_relevance_floor",
        passed: true,
        observed: "0.910",
        threshold: ">= 0.450",
        details: "Above minimum calibrated floor",
      },
    ],
  },
];

export const TEST_WORKSPACE_CONFIG: WorkspaceConfigResponse = {
  domains: TEST_BACKEND_DOMAINS,
  federated_option: {
    domain_id: "federated_all",
    display_name: "All Authorized Domains (Auto-Route)",
    description: "Dynamically probes all authorized DomainPacks in PostgreSQL 17.",
  },
  role_presets: [
    { role: "engineer", label: "Engineer", default_scopes: ["public_eng", "public_docs"] },
    { role: "domain_researcher", label: "Domain Researcher", default_scopes: ["public_docs"] },
    { role: "product_manager", label: "Product Manager", default_scopes: ["public_eng", "public_docs"] },
    { role: "compliance_officer", label: "Compliance Officer", default_scopes: ["public_eng", "public_docs"] },
  ],
  pipeline_stages: TEST_PIPELINE_STAGES,
  source_filters: ["all", "code", "wiki", "jira"],
  hardware_profile: TEST_HEALTH_RESPONSE.hardware_profile,
  database_connected: true,
  database_status: "PostgreSQL 17.11 RLS Enforced",
  runtime_mode: "offline_deterministic",
  timestamp: "2026-11-15T10:00:00Z",
};

export const TEST_DOMAINS: DomainInfo[] = [
  {
    domain_id: "code_architecture_scout",
    display_name: "Code Architecture & AST",
    description: "AST-aware Python slicing & token budget contracts.",
    modalities: ["code"],
    sample_queries: [
      "Inspect DeterministicSlicer and TokenBudgetResult contracts for context packing.",
    ],
    lora_enabled: true,
    default_scopes: ["public_eng"],
    doc_chunks_count: 5,
    documents_count: 2,
    code_files_count: 14,
    ast_symbols_count: 3443,
    base_model_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
    lora_adapter_id: "code_architecture_scout_v1",
    lora_adapter_path: "adapters/code_architecture_scout",
    primary_truth_source: "code",
    output_contract: "ImplementationContextPack",
    qlora_rank: 8,
    qlora_alpha: 16,
    hot_swap_latency_ms: 0.02,
    training_loss: 0.72,
    safetensors_bytes: 18432000,
    verified: true,
  },
  {
    domain_id: "enterprise_docs_scout",
    display_name: "Enterprise Architecture Docs & ADRs",
    description: "System architecture, RLS policies, and design specifications.",
    modalities: ["document"],
    sample_queries: [
      "What does the RLS policy ADR say about tenant isolation and fail-closed defaults?",
    ],
    lora_enabled: false,
    default_scopes: ["public_docs"],
    doc_chunks_count: 14188,
    documents_count: 42,
    code_files_count: 0,
    ast_symbols_count: 0,
    base_model_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
    lora_adapter_id: "enterprise_docs_scout_v1",
    lora_adapter_path: "adapters/enterprise_docs_scout",
    primary_truth_source: "document",
    output_contract: "DomainKnowledgeBrief",
    qlora_rank: 8,
    qlora_alpha: 16,
    hot_swap_latency_ms: 0.02,
    training_loss: 0.68,
    safetensors_bytes: 18432000,
    verified: true,
  },
];

export const TEST_TOKEN_SAVINGS: ScoutTokenSavingsBreakdown = {
  local_tokens_explored: 42000,
  pack_tokens_sent: 3800,
  tokens_saved: 38200,
  estimated_usd_saved: 0.1528,
  frontier_savings_ratio_pct: 90.95,
  qlora_overhead_eliminated_tokens: 420,
};

export const TEST_ROI_METRICS: ROIMetricsResponse = {
  summary: {
    total_queries: 249,
    total_local_tokens_explored: 10218000,
    total_pack_tokens_sent: 1021800,
    total_paid_tokens_saved: 9196200,
    total_estimated_usd_saved: 40.10565,
    frontier_savings_ratio_pct: 90.0,
    qlora_overhead_eliminated_per_query: 410,
    total_qlora_overhead_eliminated: 102090,
    telemetry_source: "postgresql_live",
  },
  m6_evaluation_report: null,
  m7_qlora_benchmark_report: null,
  timestamp: "2026-11-14T10:00:00Z",
};

export const TEST_DOC_CITATION: DocCitation = {
  chunk_id: "chunk_rls_01",
  doc_id: "doc_pg_rls_architecture",
  version: "v1.0",
  doc_title: "PostgreSQL 17 Forced Row-Level Security and Connection Pool Isolation",
  source_uri: "https://wiki.internal.example/pages/pg17-rls",
  section_heading: "3.2 Transaction-Scoped GUC Binding",
  verbatim_quote: "Every pooled query executes inside an explicit transaction setting app.tenant_id via set_config(..., true).",
  source_type: "confluence_wiki",
  published_date: "2026-10-01",
};

export const TEST_STALE_DOC_CITATION: DocCitation = {
  chunk_id: "chunk_stale_runbook_02",
  doc_id: "doc_legacy_runbook",
  version: "v0.9",
  doc_title: "Legacy Execution Timeout Runbook",
  source_uri: "https://wiki.internal.example/pages/1042",
  section_heading: "4. Manual Failover Procedure",
  verbatim_quote: "Restart the legacy single-slot worker manually if heartbeat stalls for 90 seconds.",
  page_id: "1042",
  source_type: "confluence_wiki",
  is_stale: true,
  stale_reason: "Superseded by automated pg_pool failover ADR-2026-11.",
  published_date: "2025-03-14",
};

export const TEST_JIRA_CITATION: DocCitation = {
  chunk_id: "chunk_jira_eng412",
  doc_id: "jira_eng_412",
  version: "1.0",
  doc_title: "ENG-412: Execution Timeout Circuit Breaker",
  source_uri: "https://jira.internal.example/browse/ENG-412",
  section_heading: "Incident Postmortem",
  verbatim_quote: "Connection pool saturation occurred when unindexed queries bypassed RLS timeout.",
  ticket_key: "ENG-412",
  source_type: "jira",
  published_date: "2026-11-02",
};

export const TEST_CODE_PACK: ContextPack = {
  trace_id: "tr_test_code_001",
  domain_id: "code_architecture_scout",
  user_task: "Inspect DeterministicSlicer slice bounds and 1-hop caller contracts.",
  architectural_execution_plan: [
    "Review DeterministicSlicer.compute_slice_bounds in slicer/bounds.py lines 14-32.",
    "Preserve token budget validation and TokenBudgetResult caller contract.",
  ],
  doc_vs_code_conflicts_detected: [],
  code_slices: [
    {
      repo_name: "context-core",
      file_path: "slicer/bounds.py",
      start_line: 14,
      end_line: 32,
      qualified_symbol: "DeterministicSlicer.compute_slice_bounds",
      selection_reason: "Core deterministic slice bounds calculation with positive budget check.",
      modification_intent: "modify_target",
      git_commit_sha: "abc1234def5678",
      git_blob_sha: "blob987654",
      verbatim_code_with_lines:
        "14: class DeterministicSlicer:\n" +
        "15:     def compute_slice_bounds(self, max_tokens: int, lines: int) -> tuple[int, int]:\n" +
        "16:         if max_tokens <= 0:\n" +
        "17:             raise ValueError('max_tokens must be positive')\n" +
        "18:         return (1, lines)",
      auto_expanded_by_ast: false,
    },
    {
      repo_name: "context-core",
      file_path: "slicer/budget.py",
      start_line: 5,
      end_line: 14,
      qualified_symbol: "TokenBudgetResult",
      selection_reason: "1-hop AST dependency supplying token budget estimate.",
      modification_intent: "imported_type",
      git_commit_sha: "abc1234def5678",
      git_blob_sha: "blob444333",
      verbatim_code_with_lines:
        "5: @dataclass(frozen=True)\n" +
        "6: class TokenBudgetResult:\n" +
        "7:     max_tokens: int\n" +
        "8:     reserved_tokens: int",
      auto_expanded_by_ast: true,
    },
  ],
  supporting_doc_citations: [TEST_DOC_CITATION],
  domain_knowledge_brief: {
    brief_title: "Deterministic Slicer & Token Budget Bounds",
    domain_id: "code_architecture_scout",
    executive_summary:
      "DeterministicSlicer computes AST-aligned slice bounds using TokenBudgetResult and enforces positive token guards before packing.",
    formal_expression_or_rule: "effective_tokens = max_tokens - reserved_tokens",
    core_findings_and_rules: [
      "Slicer consumes TokenBudgetResult.max_tokens directly.",
      "AST 1-hop dependency closure attached slicer/budget.py automatically.",
    ],
    domain_constraints_and_caveats: [
      "Rejects non-positive token budgets fail-closed before frontier handoff.",
    ],
    known_risks_and_failure_modes: [
      "Oversized single-function AST nodes require explicit line-range splitting.",
    ],
    structured_metadata: {
      ast_closure_slices: 1,
      truth_source: "code",
    },
    verified_citations: [TEST_DOC_CITATION],
  },
  preflight_report: {
    passed: true,
    unresolved_symbols: [],
    missing_reference_identifiers: [],
    novel_target_identifiers: [],
    normalized_rrf_score: 0.89,
    citations_verified_100_pct: true,
    auto_attached_ast_slices_count: 1,
    bounce_back_feedback: null,
  },
  local_tokens_consumed: 42000,
  pack_token_count: 3800,
};

export const TEST_DOCS_PACK: ContextPack = {
  ...TEST_CODE_PACK,
  trace_id: "tr_test_docs_002",
  domain_id: "enterprise_docs_scout",
  user_task: "Explain PostgreSQL 17 FORCE RLS tenant isolation and fail-closed defaults.",
  doc_vs_code_conflicts_detected: [
    "Legacy script bypasses RLS whereas ADR-2026-11 requires FORCE ROW LEVEL SECURITY on all tables.",
  ],
  code_slices: [],
  domain_knowledge_brief: {
    brief_title: "PostgreSQL 17 FORCE RLS & Tenant Isolation",
    domain_id: "enterprise_docs_scout",
    executive_summary:
      "All PostgreSQL 17 tables enforce tenant_id UUID NOT NULL with ENABLE and FORCE ROW LEVEL SECURITY under NOBYPASSRLS.",
    formal_expression_or_rule: "current_setting('app.tenant_id', true)::uuid = tenant_id",
    core_findings_and_rules: [
      "Runtime routes must never use db_admin_url.",
      "Missing or unset tenant_id fails closed with zero rows returned.",
    ],
    domain_constraints_and_caveats: [
      "Connection poolers must reset session variables between requests.",
    ],
    known_risks_and_failure_modes: [
      "Table owners bypass RLS unless FORCE ROW LEVEL SECURITY is enabled.",
    ],
    structured_metadata: {
      pg_version: "17.11",
      rls_enforced: true,
    },
    verified_citations: [TEST_DOC_CITATION],
    sufficiency_verdict: "sufficient",
  },
};

export const TEST_DIAGNOSTIC: InsufficientContextDiagnostic = {
  trace_id: "tr_test_abstain_003",
  domain_id: "code_architecture_scout",
  reason: "No verified symbols or documents matched query within caller RBAC scopes.",
  is_abstention: true,
  missing_symbols: ["NonExistentProprietaryEngine"],
  partial_candidate_files: [],
  missing_items: [
    {
      identifier: "NonExistentProprietaryEngine",
      reason: "Symbol absent from indexed AST and PostgreSQL doc chunks.",
    },
  ],
  failure_check: "symbol_existence_and_rrf_floor",
  paid_tokens_spent: 0,
};

export const TEST_SCOUT_QUERY_RESPONSE: ScoutQueryResponse = {
  trace_id: TEST_CODE_PACK.trace_id,
  domain_id: TEST_CODE_PACK.domain_id,
  passed: true,
  context_pack: TEST_CODE_PACK,
  diagnostic: null,
  citations: TEST_CODE_PACK.supporting_doc_citations,
  preflight_report: TEST_CODE_PACK.preflight_report,
  token_savings: TEST_TOKEN_SAVINGS,
  trace_attributes: { execution_mode: "chat_qa" },
};

export const TEST_HANDOFF_PREVIEW: HandoffPreviewResponse = {
  confirmation_token: "confirm_tok_test_001",
  patch_diff:
    "--- a/slicer/bounds.py\n" +
    "+++ b/slicer/bounds.py\n" +
    "@@ -16,2 +16,3 @@\n" +
    "+        if lines <= 0:\n" +
    "+            raise ValueError('lines must be positive')\n",
  target_files: ["slicer/bounds.py"],
  model_id: "gemini-4-argon",
  local_tokens_consumed: 42000,
  pack_token_count: 3800,
  estimated_paid_cost_usd: 0.014,
  estimated_paid_savings_usd: 0.1528,
  cost_low_usd: 0.011,
  cost_expected_usd: 0.014,
  cost_high_usd: 0.019,
  pricing_source: "Gemini 4 Argon verified list pricing",
  requires_human_confirmation: true,
  expires_in_seconds: 300,
  created_at: "2026-11-14T10:01:00Z",
};

export const TEST_HANDOFF_APPLY: HandoffApplyResponse = {
  success: true,
  confirmed: true,
  target_files: ["slicer/bounds.py"],
  lint_passed: true,
  tests_passed: true,
  verification_passed: true,
  details:
    "Confirmed diff structure & Python AST syntax verified for client-side git apply (host working tree unchanged; pytest disabled per OWASP ASI05).",
  execution_mode: "client_side_git_apply",
  server_files_mutated: false,
  confirmed_patch_diff: TEST_HANDOFF_PREVIEW.patch_diff,
  syntax_valid: true,
};

export const TEST_ADMIN_PURGE_RESPONSE: AdminPurgeResponse = {
  domain_id: "code_architecture_scout",
  tenant_id: "tenant_001",
  pg17_deleted: { doc_chunks: 4, code_chunks: 1 },
  zoekt_deleted: 1,
  object_store_deleted: ["s3://bucket/slicer/bounds.py"],
  permission_cache_invalidated: true,
  residue_found: 0,
  elapsed_seconds: 0.14,
  sla_seconds: 60,
  sla_met: true,
  warnings: [],
};

export const TEST_SESSIONS: ScoutSessionSummary[] = [
  {
    session_id: "sess_01",
    title: "DeterministicSlicer bounds & RLS trace",
    domain_id: "code_architecture_scout",
    turn_count: 3,
    compacted_tokens: 1850,
    updated_at: "2026-11-19T09:45:00Z",
    has_commit_drift: true,
    commit_drift_details: ["slicer/bounds.py modified at commit 9f3c21a"],
  },
  {
    session_id: "sess_02",
    title: "ADR-004 PostgreSQL 17 FORCE RLS",
    domain_id: "enterprise_docs_scout",
    turn_count: 2,
    compacted_tokens: 920,
    updated_at: "2026-11-19T08:15:00Z",
    has_commit_drift: false,
  },
];

export const TEST_BRANCHES: BranchOverlayOption[] = [
  {
    branch_id: "main",
    label: "main (HEAD)",
    kind: "head",
    commit_sha: "abc1234",
  },
  {
    branch_id: "mr-412",
    label: "MR !412 (overlay: 3 files)",
    kind: "mr_overlay",
    commit_sha: "def5678",
    mr_iid: "412",
    overlay_files_count: 3,
  },
];

export const TEST_SIDE_EFFECTS: SideEffectMatrixItem[] = [
  {
    side_effect_id: "se_01",
    effect_type: "DB_WRITE",
    qualified_symbol: "TenantAuditRepo.record_purge",
    file_path: "src/db/audit_repo.py",
    line_range: "L42-L68",
    target_resource: "pg17:audit_events",
    spec_status: "VERIFIED",
    verification_note: "Matches ADR-004 FORCE RLS write policy.",
  },
  {
    side_effect_id: "se_02",
    effect_type: "HTTP_EGRESS",
    qualified_symbol: "WebhookDispatcher.emit_alert",
    file_path: "src/webhooks/dispatcher.py",
    line_range: "L19-L44",
    target_resource: "https://internal-siem.corp.local/events",
    spec_status: "SPEC_DRIFT_WARNING",
    verification_note: "Undocumented egress header X-Trace-Signature not in ADR-011.",
  },
];

export function buildTestSession(
  principal: PrincipalContext = TEST_ENGINEER_PRINCIPAL
): AuthSessionResponse {
  return {
    access_token: `signed.hmac.${principal.role}.${principal.user_id}`,
    token_type: "Bearer",
    expires_in_seconds: 3600,
    principal,
  };
}

export function createMockSseStream(
  events: SSEStageEvent[]
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const evt of events) {
        const frame = `event: ${evt.event}\ndata: ${JSON.stringify(evt.data)}\n\n`;
        controller.enqueue(encoder.encode(frame));
      }
      controller.close();
    },
  });
}

export interface MockApiOverrides {
  authStatus?: number;
  authDetail?: string;
  mePrincipal?: PrincipalContext;
  meStatus?: number;
  runtimeMode?: string;
  healthStatus?: number;
  healthResponse?: HealthResponse;
  domainsStatus?: number;
  domainsDetail?: BackendDomainDetail[];
  workspaceConfigStatus?: number;
  workspaceConfigResponse?: WorkspaceConfigResponse;
  roiStatus?: number;
  roiResponse?: ROIMetricsResponse;
  queryResponse?: ScoutQueryResponse;
  queryStatus?: number;
  queryDetail?: string;
  sseEvents?: SSEStageEvent[];
  sseStatus?: number;
  sseDetail?: string;
  hangSseStream?: boolean;
  networkFailUrls?: string[];
  previewResponse?: HandoffPreviewResponse;
  previewStatus?: number;
  applyResponse?: HandoffApplyResponse;
  applyStatus?: number;
  purgeStatus?: number;
  purgeDetail?: string;
  purgeResponse?: AdminPurgeResponse;
}

export function installMockApiFetch(overrides: MockApiOverrides = {}) {
  clearCachedAuthSession();
  setRuntimeAuthMode("development");

  const mintedPrincipals = new Map<string, PrincipalContext>();

  const extractBearerToken = (init?: RequestInit): string | null => {
    const rawHeaders = init?.headers;
    if (!rawHeaders) return null;
    if (rawHeaders instanceof Headers) {
      const val = rawHeaders.get("Authorization") ?? rawHeaders.get("authorization");
      return val && val.startsWith("Bearer ") ? val.slice(7).trim() : null;
    }
    if (Array.isArray(rawHeaders)) {
      for (const [k, v] of rawHeaders) {
        if (k.toLowerCase() === "authorization" && v.startsWith("Bearer ")) {
          return v.slice(7).trim();
        }
      }
      return null;
    }
    for (const [k, v] of Object.entries(rawHeaders)) {
      if (k.toLowerCase() === "authorization" && typeof v === "string" && v.startsWith("Bearer ")) {
        return v.slice(7).trim();
      }
    }
    return null;
  };

  const hasValidAuth = (init?: RequestInit): boolean => {
    const token = extractBearerToken(init);
    if (token && token.length > 0) {
      return true;
    }
    if (overrides.runtimeMode === "production" && init?.credentials === "include") {
      return true;
    }
    return false;
  };

  const resolveRequestPrincipal = (init?: RequestInit): PrincipalContext | null => {
    const token = extractBearerToken(init);
    if (token && mintedPrincipals.has(token)) {
      return mintedPrincipals.get(token) ?? null;
    }
    if (overrides.runtimeMode === "production") {
      return overrides.mePrincipal ?? TEST_ENGINEER_PRINCIPAL;
    }
    return null;
  };

  const fetchMock = vi.fn(
    async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = typeof input === "string" ? input : input.toString();

      if (
        overrides.networkFailUrls &&
        overrides.networkFailUrls.some((pattern) => url.includes(pattern))
      ) {
        throw new TypeError("Failed to fetch");
      }

      if (url.endsWith("/api/v1/auth/session")) {
        if (overrides.runtimeMode === "production") {
          return new Response(
            JSON.stringify({ detail: "Dev session minting disabled in production mode" }),
            { status: 403, headers: { "Content-Type": "application/json" } }
          );
        }
        const status = overrides.authStatus ?? 200;
        if (status >= 400) {
          return new Response(
            JSON.stringify({ detail: overrides.authDetail ?? "Auth forbidden" }),
            { status, headers: { "Content-Type": "application/json" } }
          );
        }
        const parsedBody = init?.body
          ? (JSON.parse(String(init.body)) as Partial<PrincipalContext>)
          : {};
        const principal: PrincipalContext = {
          user_id: parsedBody.user_id ?? TEST_ENGINEER_PRINCIPAL.user_id,
          role: parsedBody.role ?? TEST_ENGINEER_PRINCIPAL.role,
          allowed_scopes:
            parsedBody.allowed_scopes ?? TEST_ENGINEER_PRINCIPAL.allowed_scopes,
          max_session_paid_tokens:
            parsedBody.max_session_paid_tokens ??
            TEST_ENGINEER_PRINCIPAL.max_session_paid_tokens,
        };
        const session = buildTestSession(principal);
        mintedPrincipals.set(session.access_token, principal);
        return new Response(JSON.stringify(session), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      if (url.endsWith("/api/v1/auth/me")) {
        const status = overrides.meStatus ?? overrides.authStatus ?? 200;
        if (status >= 400) {
          return new Response(
            JSON.stringify({ detail: overrides.authDetail ?? "Unauthorized" }),
            { status, headers: { "Content-Type": "application/json" } }
          );
        }
        if (!hasValidAuth(init)) {
          return new Response(
            JSON.stringify({ detail: "Missing authentication credentials" }),
            { status: 401, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response(
          JSON.stringify(
            overrides.mePrincipal ??
              resolveRequestPrincipal(init) ??
              TEST_ENGINEER_PRINCIPAL
          ),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      if (url.endsWith("/api/v1/health")) {
        const status = overrides.healthStatus ?? 200;
        if (status >= 400) {
          return new Response(JSON.stringify({ detail: "Health failed" }), {
            status,
            headers: { "Content-Type": "application/json" },
          });
        }
        const baseHealth = overrides.healthResponse ?? TEST_HEALTH_RESPONSE;
        const resolvedHealth: HealthResponse = overrides.runtimeMode
          ? {
              ...baseHealth,
              hardware_profile: {
                ...baseHealth.hardware_profile,
                runtime_mode: overrides.runtimeMode,
              },
            }
          : baseHealth;
        return new Response(JSON.stringify(resolvedHealth), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      if (url.endsWith("/api/v1/workspace-config")) {
        const status =
          overrides.workspaceConfigStatus ?? overrides.domainsStatus ?? 200;
        if (status >= 400) {
          return new Response(
            JSON.stringify({ detail: "Workspace config failed" }),
            { status, headers: { "Content-Type": "application/json" } }
          );
        }
        if (!hasValidAuth(init)) {
          return new Response(JSON.stringify({ detail: "Unauthorized" }), {
            status: 401,
            headers: { "Content-Type": "application/json" },
          });
        }
        const baseCfg =
          overrides.workspaceConfigResponse ?? TEST_WORKSPACE_CONFIG;
        const cfgWithDomains: WorkspaceConfigResponse = overrides.domainsDetail
          ? { ...baseCfg, domains: overrides.domainsDetail }
          : baseCfg;
        const cfg: WorkspaceConfigResponse = overrides.runtimeMode
          ? {
              ...cfgWithDomains,
              runtime_mode: overrides.runtimeMode,
              hardware_profile: {
                ...cfgWithDomains.hardware_profile,
                runtime_mode: overrides.runtimeMode,
              },
            }
          : cfgWithDomains;
        return new Response(JSON.stringify(cfg), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      if (url.endsWith("/api/v1/domains")) {
        const status = overrides.domainsStatus ?? 200;
        if (status >= 400) {
          return new Response(JSON.stringify({ detail: "Domains failed" }), {
            status,
            headers: { "Content-Type": "application/json" },
          });
        }
        if (!hasValidAuth(init)) {
          return new Response(JSON.stringify({ detail: "Unauthorized" }), {
            status: 401,
            headers: { "Content-Type": "application/json" },
          });
        }
        const details = overrides.domainsDetail ?? TEST_BACKEND_DOMAINS;
        return new Response(
          JSON.stringify({ domains: details, count: details.length }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      if (url.endsWith("/api/v1/telemetry/roi")) {
        const status = overrides.roiStatus ?? 200;
        if (status >= 400) {
          return new Response(JSON.stringify({ detail: "ROI failed" }), {
            status,
            headers: { "Content-Type": "application/json" },
          });
        }
        if (!hasValidAuth(init)) {
          return new Response(JSON.stringify({ detail: "Unauthorized" }), {
            status: 401,
            headers: { "Content-Type": "application/json" },
          });
        }
        return new Response(
          JSON.stringify(overrides.roiResponse ?? TEST_ROI_METRICS),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      if (url.endsWith("/api/v1/scout/query")) {
        const status = overrides.queryStatus ?? 200;
        if (status >= 400) {
          return new Response(
            JSON.stringify({ detail: overrides.queryDetail ?? "Query failed" }),
            { status, headers: { "Content-Type": "application/json" } }
          );
        }
        if (!hasValidAuth(init)) {
          return new Response(JSON.stringify({ detail: "Unauthorized" }), {
            status: 401,
            headers: { "Content-Type": "application/json" },
          });
        }
        return new Response(
          JSON.stringify(overrides.queryResponse ?? TEST_SCOUT_QUERY_RESPONSE),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      if (url.endsWith("/api/v1/scout/stream")) {
        const status = overrides.sseStatus ?? 200;
        if (status >= 400) {
          return new Response(
            JSON.stringify({ detail: overrides.sseDetail ?? "Blocked by policy" }),
            { status, headers: { "Content-Type": "application/json" } }
          );
        }
        if (!hasValidAuth(init)) {
          return new Response(JSON.stringify({ detail: "Unauthorized" }), {
            status: 401,
            headers: { "Content-Type": "application/json" },
          });
        }
        if (overrides.hangSseStream) {
          const encoder = new TextEncoder();
          const pendingStream = new ReadableStream<Uint8Array>({
            start(controller) {
              const firstEvent: SSEStageEvent = {
                event: "stage_update",
                data: {
                  stage: 2,
                  stage_name: "Federated Corpus Retrieval",
                  status: "running",
                  details: "Searching PostgreSQL 17 RLS shards...",
                  timestamp: "2026-11-14T10:00:01Z",
                },
              };
              controller.enqueue(
                encoder.encode(
                  `event: ${firstEvent.event}\ndata: ${JSON.stringify(firstEvent.data)}\n\n`
                )
              );
            },
          });
          return new Response(pendingStream, {
            status: 200,
            headers: { "Content-Type": "text/event-stream" },
          });
        }
        const events: SSEStageEvent[] = overrides.sseEvents ?? [
          {
            event: "stage_update",
            data: {
              stage: 1,
              stage_name: "Security & Privacy Check",
              status: "completed",
              details: "RBAC scope check and ShieldGemma prompt inspection passed.",
              timestamp: "2026-11-14T10:00:01Z",
            },
          },
          {
            event: "stage_update",
            data: {
              stage: 4,
              stage_name: "Verify Grounding & Build Pack",
              status: "completed",
              details: "All AST symbols and document quotes verified.",
              timestamp: "2026-11-14T10:00:04Z",
            },
          },
          {
            event: "context_pack_ready",
            data: {
              status: "ready",
              pack: TEST_CODE_PACK,
              token_savings: TEST_TOKEN_SAVINGS,
              timestamp: "2026-11-14T10:00:04Z",
            },
          },
        ];
        return new Response(createMockSseStream(events), {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }

      if (url.endsWith("/api/v1/handoff/preview")) {
        const status = overrides.previewStatus ?? 200;
        if (status >= 400) {
          return new Response(JSON.stringify({ detail: "Engineer role required" }), {
            status,
            headers: { "Content-Type": "application/json" },
          });
        }
        if (!hasValidAuth(init)) {
          return new Response(JSON.stringify({ detail: "Unauthorized" }), {
            status: 401,
            headers: { "Content-Type": "application/json" },
          });
        }
        const reqPrincipal = resolveRequestPrincipal(init);
        if (reqPrincipal && reqPrincipal.role !== "engineer") {
          return new Response(JSON.stringify({ detail: "Engineer role required" }), {
            status: 403,
            headers: { "Content-Type": "application/json" },
          });
        }
        return new Response(
          JSON.stringify(overrides.previewResponse ?? TEST_HANDOFF_PREVIEW),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      if (url.endsWith("/api/v1/handoff/apply")) {
        const status = overrides.applyStatus ?? 200;
        if (status >= 400) {
          return new Response(JSON.stringify({ detail: "Apply rejected" }), {
            status,
            headers: { "Content-Type": "application/json" },
          });
        }
        if (!hasValidAuth(init)) {
          return new Response(JSON.stringify({ detail: "Unauthorized" }), {
            status: 401,
            headers: { "Content-Type": "application/json" },
          });
        }
        const reqPrincipal = resolveRequestPrincipal(init);
        if (reqPrincipal && reqPrincipal.role !== "engineer") {
          return new Response(JSON.stringify({ detail: "Engineer role required" }), {
            status: 403,
            headers: { "Content-Type": "application/json" },
          });
        }
        return new Response(
          JSON.stringify(overrides.applyResponse ?? TEST_HANDOFF_APPLY),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      if (url.endsWith("/api/v1/admin/purge")) {
        const status = overrides.purgeStatus ?? 200;
        if (status >= 400) {
          return new Response(
            JSON.stringify({ detail: overrides.purgeDetail ?? "Purge forbidden" }),
            { status, headers: { "Content-Type": "application/json" } }
          );
        }
        if (!hasValidAuth(init)) {
          return new Response(JSON.stringify({ detail: "Unauthorized" }), {
            status: 401,
            headers: { "Content-Type": "application/json" },
          });
        }
        const reqPrincipal = resolveRequestPrincipal(init);
        if (reqPrincipal && reqPrincipal.role !== "compliance_officer") {
          return new Response(
            JSON.stringify({ detail: "Compliance officer role required" }),
            { status: 403, headers: { "Content-Type": "application/json" } }
          );
        }
        let parsed: Record<string, unknown> = {};
        if (typeof init?.body === "string" && init.body.trim()) {
          try {
            parsed = JSON.parse(init.body) as Record<string, unknown>;
          } catch {
            parsed = {};
          }
        }
        if (parsed.file_path && !String(parsed.repo_name ?? "").trim()) {
          return new Response(
            JSON.stringify({ detail: "file_path purge requires repo_name" }),
            { status: 400, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response(
          JSON.stringify(overrides.purgeResponse ?? TEST_ADMIN_PURGE_RESPONSE),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      return new Response(JSON.stringify({ detail: `Unhandled URL: ${url}` }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    }
  );

  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
