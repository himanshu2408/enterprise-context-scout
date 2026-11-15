/**
 * Strict TypeScript contracts mirroring enterprise-context-scout Pydantic v2 schemas.
 * Zero 'any' types allowed across the entire codebase.
 */

export type RoleType =
  | "engineer"
  | "quant_researcher"
  | "product_manager"
  | "compliance_officer";

export type SourceFilterType = "all" | "code" | "wiki" | "jira";

export type MarketRegime =
  | "IN_NSE"
  | "DE_XETRA"
  | "US_EQUITY"
  | "GLOBAL"
  | "INTERNAL_ADR";

export type ExecutionMode = "chat_qa" | "gemini_handoff";

export type ModificationIntent =
  | "modify_target"
  | "imported_type"
  | "direct_caller"
  | "test_fixture"
  | "reference_only";

export type FrictionVerdict =
  | "REALISTIC_FRICTION"
  | "UNVERIFIED_FRICTION_WARNING"
  | "FAILS_AFTER_COSTS";

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
  max_kv_tokens: number;
  quantization: string;
  lora_slots_allocated: number;
  lora_slots_capacity: number;
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
  status: "healthy" | "degraded" | "unhealthy";
  database_connected: boolean;
  database_status: string;
  registered_domains: string[];
  adapter_pool_status: Record<string, AdapterVerificationStatus>;
  hardware_profile: HealthHardwareProfile;
  timestamp: string;
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
}

export interface QuantStrategyBrief {
  strategy_name: string;
  target_markets: MarketRegime[];
  is_intraday_futures: boolean;
  mathematical_signal_latex: string;
  entry_exit_rules: string[];
  sample_period_years: string;
  reported_sharpe_ratio: number | null;
  reported_max_drawdown_pct: number | null;
  transaction_cost_bps_assumed: number | null;
  friction_verdict: FrictionVerdict;
  market_specific_caveats: string[];
  known_failure_regimes: string[];
  verified_citations: DocCitation[];
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
  quant_strategy_spec: QuantStrategyBrief | null;
  domain_knowledge_brief?: DomainKnowledgeBrief | null;
  preflight_report: PreFlightValidationReport;
  local_tokens_consumed: number;
  pack_token_count: number;
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
  quant_strategy_spec?: QuantStrategyBrief | null;
  token_savings: ScoutTokenSavingsBreakdown;
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
  custom_patch?: string;
}

export interface HandoffApplyResponse {
  success: boolean;
  confirmed: boolean;
  target_files: string[];
  lint_passed: boolean;
  tests_passed: boolean;
  verification_passed: boolean;
  details: string;
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
  authority_rules?: {
    primary_truth_source?: "code" | "document";
  };
  output_contract?: string;
}

export interface BackendDomainDetail {
  domain_id: string;
  display_name: string;
  manifest: BackendDomainManifest;
  adapter_verification: AdapterVerificationStatus;
  training_metrics: Record<string, unknown> | null;
}

export interface DomainsResponse {
  domains: BackendDomainDetail[];
  count: number;
}

export interface ScoutQueryRequest {
  domain_id: string;
  query: string;
  trace_id?: string;
  normalized_rrf_score?: number;
  max_bounce_backs?: number;
}

export interface ScoutQueryResponse {
  trace_id: string;
  domain_id: string;
  passed: boolean;
  context_pack: ContextPack | null;
  diagnostic: InsufficientContextDiagnostic | null;
  citations: DocCitation[];
  preflight_report: PreFlightValidationReport | null;
  quant_strategy_spec: QuantStrategyBrief | null;
  token_savings: ScoutTokenSavingsBreakdown;
  trace_attributes: Record<string, unknown>;
}
