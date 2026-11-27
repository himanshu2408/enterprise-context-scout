/**
 * Strict TypeScript contracts mirroring enterprise-context-scout Pydantic v2 schemas.
 * Zero 'any' types allowed across the entire codebase.
 */

export type RoleType =
  | "engineer"
  | "domain_researcher"
  | "product_manager"
  | "compliance_officer";

export type SourceFilterType = "all" | "code" | "wiki" | "jira";

export type DomainModality = "code" | "document" | "issues";

export type ExecutionMode = "chat_qa" | "gemini_handoff";

export type ModificationIntent =
  | "modify_target"
  | "imported_type"
  | "direct_caller"
  | "test_fixture"
  | "reference_only";

export type ConfidenceBand = "high" | "medium" | "low";

export type SufficiencyVerdict = "sufficient" | "partial" | "insufficient";

export interface StageConfidence {
  stage: number;
  stage_name: string;
  confidence_score: number;
  confidence_band: ConfidenceBand;
  signals: Record<string, number | string | boolean>;
  summary: string;
}

export interface RetrievalHitTrace {
  item_id: string;
  domain_id: string;
  title: string;
  modality: DomainModality;
  cosine_similarity: number;
  bm25_score: number;
  trigram_score: number;
  calibrated_score: number;
  accepted: boolean;
  rejection_reason?: string | null;
}

export interface DomainProbeTrace {
  domain_id: string;
  display_name: string;
  authorized: boolean;
  probed_hits: number;
  accepted_hits: number;
  rejected_hits: number;
  max_cosine_similarity: number;
  max_calibrated_score: number;
  selected: boolean;
  selection_reason: string;
  top_hits: RetrievalHitTrace[];
}

export interface PreFlightCheckLedgerItem {
  check_name: string;
  passed: boolean;
  observed: string;
  threshold: string;
  details: string;
}

export interface StageExecutionTrace {
  stage: number;
  stage_name: string;
  status: "in_progress" | "completed" | "failed";
  elapsed_ms?: number | null;
  summary: string;
  principal_id?: string | null;
  principal_scopes?: string[];
  guardrail_verdict?: string | null;
  query_type?: string | null;
  core_subject?: string | null;
  required_facets?: string[];
  target_domains?: string[];
  domain_probes?: DomainProbeTrace[];
  ast_one_hop_count?: number | null;
  ast_two_hop_count?: number | null;
  model_id?: string | null;
  adapter_mode?: string | null;
  chunks_fed?: number | null;
  local_tokens_read?: number | null;
  sufficiency_verdict?: SufficiencyVerdict | null;
  grounded_facets?: number | null;
  total_facets?: number | null;
  checks?: PreFlightCheckLedgerItem[];
  final_decision?:
    | "handoff_ready"
    | "abstain_zero_paid"
    | "bounce_back"
    | "blocked"
    | null;
  paid_tokens_spent?: number | null;
}

export type FacetCoverageStatus = "grounded" | "partial" | "not_found";

export interface QueryFacetCoverage {
  facet_id: string;
  facet_query: string;
  status: FacetCoverageStatus;
  matched_sources: string[];
  confidence: number;
  explanation: string;
}

export interface PrincipalContext {
  user_id: string;
  role: RoleType;
  allowed_scopes: string[];
  max_session_paid_tokens: number;
}

export interface AuthSessionRequest {
  user_id: string;
  role: RoleType;
  allowed_scopes?: string[];
  max_session_paid_tokens?: number;
}

export interface AuthSessionResponse {
  access_token: string;
  token_type: "Bearer";
  expires_in_seconds: number;
  principal: PrincipalContext;
}

export interface HealthHardwareProfile {
  device: string;
  unified_memory: string;
  memory_gb?: number;
  arch?: string;
  os_name?: string;
  max_kv_tokens: number;
  quantization: string;
  lora_slots_allocated: number;
  lora_slots_capacity: number;
  postgres_version?: string | null;
  runtime_mode?: string;
}

export interface AdapterVerificationStatus {
  domain_id: string;
  is_valid: boolean;
  adapter_path: string;
  safetensors_bytes: number;
  r: number;
  lora_alpha: number;
  is_moe: boolean;
  final_loss: number | null;
  error_message: string | null;
  hot_swap_latency_ms?: number | null;
}

export interface HealthResponse {
  version?: string;
  status: "healthy" | "degraded" | "unhealthy";
  overall_status?: "healthy" | "degraded" | "unhealthy";
  database_connected: boolean;
  database_status: string;
  registered_domains: string[];
  adapter_pool_status: Record<string, AdapterVerificationStatus>;
  hardware_profile: HealthHardwareProfile;
  runtime_models?: Record<string, unknown>;
  timestamp: string;
}

export interface WorkspaceRolePreset {
  role: RoleType;
  label: string;
  default_scopes: string[];
}

export interface PipelineStageSpec {
  stage_id: number;
  number: string;
  title: string;
  summary: string;
  what_happens: string;
  gate_condition: string;
}

export interface FederatedOptionSpec {
  domain_id: string;
  display_name: string;
  description: string;
}

export interface CodeSlicePointer {
  repo_name: string;
  file_path: string;
  start_line: number;
  end_line: number;
  qualified_symbol: string;
  selection_reason: string;
  modification_intent: ModificationIntent;
}

export interface VerbatimCodeSlice extends CodeSlicePointer {
  git_commit_sha: string;
  git_blob_sha: string;
  verbatim_code_with_lines: string;
  auto_expanded_by_ast?: boolean;
}

export interface DocCitation {
  chunk_id: string;
  doc_id: string;
  version?: string;
  doc_title: string;
  source_uri: string;
  section_heading: string;
  verbatim_quote: string;
  full_section_content?: string;
  page_id?: string | null;
  ticket_key?: string | null;
  source_type?: string;
  is_stale?: boolean;
  stale_reason?: string | null;
  published_date?: string | null;
}

export interface DomainKnowledgeBrief {
  brief_title: string;
  domain_id: string;
  executive_summary: string;
  formal_expression_or_rule: string | null;
  core_findings_and_rules: string[];
  domain_constraints_and_caveats: string[];
  known_risks_and_failure_modes: string[];
  structured_metadata: Record<string, string | number | boolean>;
  verified_citations: DocCitation[];
  query_facets?: QueryFacetCoverage[];
  missing_or_unverified_aspects?: string[];
  completeness_ratio?: number;
  sufficiency_verdict?: SufficiencyVerdict;
  missing_aspects?: string[];
  synthesis_mode?:
    | "live_qlora_json"
    | "live_base_gemma4_json"
    | "deterministic_extractive_fallback"
    | string;
}

export interface PreFlightValidationReport {
  passed: boolean;
  unresolved_symbols: string[];
  missing_reference_identifiers: string[];
  novel_target_identifiers: string[];
  normalized_rrf_score: number;
  citations_verified_100_pct: boolean;
  auto_attached_ast_slices_count: number;
  bounce_back_feedback: string | null;
}

export interface ContextPack {
  trace_id: string;
  domain_id: string;
  user_task: string;
  architectural_execution_plan: string[];
  doc_vs_code_conflicts_detected: string[];
  code_slices: VerbatimCodeSlice[];
  supporting_doc_citations: DocCitation[];
  domain_knowledge_brief?: DomainKnowledgeBrief | null;
  preflight_report: PreFlightValidationReport;
  local_tokens_consumed: number;
  pack_token_count: number;
  query_facets?: QueryFacetCoverage[];
  missing_or_unverified_aspects?: string[];
  completeness_ratio?: number;
  synthesis_mode?: string;
  stage_confidences?: StageConfidence[];
  stage_traces?: StageExecutionTrace[];
  session_id?: string | null;
  branch_id?: string | null;
  commit_drift_warnings?: string[];
  spec_drift_warnings?: string[];
  side_effects?: SideEffectMatrixItem[];
  architecture_mermaid?: string | null;
}

export interface InsufficientContextDiagnostic {
  trace_id: string;
  domain_id: string;
  reason: string;
  is_abstention: boolean;
  missing_symbols: string[];
  partial_candidate_files: string[];
  missing_items: Array<string | { identifier: string; reason?: string }>;
  failure_check: string | null;
  paid_tokens_spent: number;
  query_facets?: QueryFacetCoverage[];
  missing_or_unverified_aspects?: string[];
  stage_confidences?: StageConfidence[];
  stage_traces?: StageExecutionTrace[];
}

export interface ScoutTokenSavingsBreakdown {
  local_tokens_explored: number;
  pack_tokens_sent: number;
  tokens_saved: number;
  estimated_usd_saved: number;
  frontier_savings_ratio_pct: number;
  qlora_overhead_eliminated_tokens: number;
}

export type SSEEventType =
  | "stage_update"
  | "bounce_back"
  | "context_pack_ready"
  | "error";

export interface SSEStageUpdateData {
  event_type?: "stage_update";
  stage: number;
  stage_name: string;
  status: "in_progress" | "running" | "completed" | "failed";
  details: string;
  message?: string;
  confidence?: StageConfidence | null;
  stage_trace?: StageExecutionTrace | null;
  elapsed_ms?: number | null;
  timestamp: string;
}

export interface SSEBounceBackData {
  event_type?: "bounce_back";
  stage?: number;
  status?: string;
  bounce_count?: number;
  feedback: string;
  remediation_action?: string;
  details: string;
  paid_tokens_spent: number;
  cost_usd: number;
  timestamp: string;
}

export interface SSEContextPackReadyData {
  event_type?: "context_pack_ready";
  status: "ready";
  pack: ContextPack | null;
  token_savings: ScoutTokenSavingsBreakdown;
  stage_traces?: StageExecutionTrace[];
  timestamp: string;
}

export interface SSEErrorData {
  event_type?: "error";
  stage: string;
  status: "blocked" | "error";
  category: string;
  rationale: string;
  diagnostic?: InsufficientContextDiagnostic | null;
  timestamp: string;
}

export type SSEStageEvent =
  | { event: "stage_update"; data: SSEStageUpdateData }
  | { event: "bounce_back"; data: SSEBounceBackData }
  | { event: "context_pack_ready"; data: SSEContextPackReadyData }
  | { event: "error"; data: SSEErrorData };

export interface HandoffPreviewResponse {
  confirmation_token: string;
  patch_diff: string;
  target_files: string[];
  model_id: "gemini-3.8-flash" | "gemini-4-argon" | string;
  local_tokens_consumed: number;
  pack_token_count: number;
  estimated_paid_cost_usd: number;
  estimated_paid_savings_usd: number;
  cost_low_usd?: number | null;
  cost_expected_usd?: number | null;
  cost_high_usd?: number | null;
  pricing_source?: string | null;
  requires_human_confirmation: boolean;
  expires_in_seconds: number;
  created_at: string;
}

export interface HandoffApplyRequest {
  confirmation_token: string;
  confirmed: boolean;
}

export interface HandoffApplyResponse {
  success: boolean;
  confirmed: boolean;
  target_files: string[];
  lint_passed: boolean;
  tests_passed: boolean;
  verification_passed: boolean;
  details: string;
  execution_mode?: "client_side_git_apply" | string;
  server_files_mutated?: boolean;
  confirmed_patch_diff?: string;
  syntax_valid?: boolean;
}

export type TelemetrySource =
  | "postgresql_live"
  | "in_memory_session"
  | "m6_eval_baseline";

export interface AggregateROISummary {
  total_queries: number;
  total_local_tokens_explored: number;
  total_pack_tokens_sent: number;
  total_paid_tokens_saved: number;
  total_estimated_usd_saved: number;
  frontier_savings_ratio_pct: number;
  qlora_overhead_eliminated_per_query: number;
  total_qlora_overhead_eliminated: number;
  telemetry_source: TelemetrySource;
}

export interface ROIMetricsResponse {
  summary: AggregateROISummary;
  m6_evaluation_report?: Record<string, unknown> | null;
  m7_qlora_benchmark_report?: Record<string, unknown> | null;
  timestamp: string;
}

export interface DomainInfo {
  domain_id: string;
  display_name: string;
  description: string;
  modalities: DomainModality[];
  sample_queries: string[];
  lora_enabled: boolean;
  default_scopes: string[];
  doc_chunks_count: number;
  documents_count: number;
  code_files_count: number;
  ast_symbols_count: number;
  base_model_id: string;
  lora_adapter_id: string;
  lora_adapter_path: string;
  primary_truth_source: "code" | "document";
  output_contract: string;
  qlora_rank: number | null;
  qlora_alpha: number | null;
  hot_swap_latency_ms: number | null;
  training_loss: number | null;
  safetensors_bytes: number | null;
  verified: boolean;
}

export interface BackendDomainManifest {
  domain_id: string;
  display_name: string;
  base_model_id: string;
  lora_adapter_id: string;
  lora_adapter_path: string;
  description?: string;
  modalities?: DomainModality[];
  sample_queries?: string[];
  lora_enabled?: boolean;
  default_scopes?: string[];
  authority_rules?: {
    primary_truth_source?: "code" | "document";
  };
  output_contract?: string;
}

export interface BackendDomainDetail {
  domain_id: string;
  display_name: string;
  description?: string;
  modalities?: DomainModality[];
  sample_queries?: string[];
  lora_enabled?: boolean;
  default_scopes?: string[];
  doc_chunks_count?: number;
  documents_count?: number;
  code_files_count?: number;
  ast_symbols_count?: number;
  manifest: BackendDomainManifest;
  adapter_verification: AdapterVerificationStatus;
  training_metrics: Record<string, unknown> | null;
}

export interface DomainsResponse {
  domains: BackendDomainDetail[];
  count: number;
}

export interface WorkspaceConfigResponse {
  domains: BackendDomainDetail[];
  federated_option: FederatedOptionSpec;
  role_presets: WorkspaceRolePreset[];
  pipeline_stages: PipelineStageSpec[];
  source_filters: SourceFilterType[];
  hardware_profile: HealthHardwareProfile;
  database_connected: boolean;
  database_status: string;
  runtime_mode: string;
  timestamp: string;
}

export interface ScoutQueryRequest {
  domain_id: string;
  query: string;
  trace_id?: string;
  max_bounce_backs?: number;
  session_id?: string;
  branch_id?: string;
}

export interface ScoutQueryResponse {
  trace_id: string;
  domain_id: string;
  passed: boolean;
  context_pack: ContextPack | null;
  diagnostic: InsufficientContextDiagnostic | null;
  citations: DocCitation[];
  preflight_report: PreFlightValidationReport | null;
  token_savings: ScoutTokenSavingsBreakdown;
  trace_attributes: Record<string, unknown>;
  stage_traces?: StageExecutionTrace[];
}


export type HandoffExportFormat =
  | "claude_xml"
  | "cursor_antigravity_md"
  | "gemini_json"
  | "codex_patch"
  | "local_llm_pack";

export type SliceFeedbackStatus = "pinned" | "included" | "excluded";
export type SliceFeedbackMap = Record<string, SliceFeedbackStatus>;

export interface ScoutSessionSummary {
  session_id: string;
  title: string;
  domain_id: string;
  turn_count: number;
  compacted_tokens: number;
  updated_at: string;
  has_commit_drift: boolean;
  commit_drift_details?: string[];
}

export interface BranchOverlayOption {
  branch_id: string;
  label: string;
  kind: "head" | "mr_overlay";
  commit_sha: string;
  mr_iid?: string | null;
  overlay_files_count?: number;
}

export type SideEffectCategory =
  | "DB_WRITE"
  | "HTTP_EGRESS"
  | "AUTH_GUARD"
  | "QUEUE_PUBLISH"
  | "ENV_CONTRACT";

export interface SideEffectMatrixItem {
  side_effect_id: string;
  effect_type: SideEffectCategory;
  qualified_symbol: string;
  file_path: string;
  line_range: string;
  target_resource: string;
  spec_status: "VERIFIED" | "SPEC_DRIFT_WARNING";
  verification_note: string;
}

export interface AdminPurgeRequest {
  domain_id: string;
  repo_name?: string;
  file_path?: string;
  doc_id?: string;
  include_derived_wiki?: boolean;
}

export interface AdminPurgeResponse {
  domain_id: string;
  tenant_id: string;
  pg17_deleted: Record<string, number>;
  zoekt_deleted: number;
  object_store_deleted: string[];
  permission_cache_invalidated: boolean;
  residue_found: number;
  elapsed_seconds: number;
  sla_seconds: number;
  sla_met: boolean;
  warnings: string[];
}
