/**
 * Strict TypeScript contracts mirroring enterprise-context-scout Pydantic v2 schemas.
 * Zero 'any' types allowed across the entire codebase.
 */

export type RoleType =
  | "engineer"
  | "quant_researcher"
  | "product_manager"
  | "compliance_officer";

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
  doc_title: string;
  source_uri: string;
  section_heading: string;
  verbatim_quote: string;
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
  preflight_report: PreFlightValidationReport;
  local_tokens_consumed: number;
  pack_token_count: number;
}

export type SSEEventType =
  | "stage_update"
  | "bounce_back"
  | "context_pack_ready"
  | "error";

export interface SSEStageUpdateData {
  stage: number;
  stage_name: string;
  status: "pending" | "running" | "completed" | "failed";
  message: string;
  details?: Record<string, unknown>;
  timestamp: string;
}

export interface SSEBounceBackData {
  bounce_count: number;
  feedback: string;
  unresolved_symbols?: string[];
  remediation_action: string;
  cost_usd: number;
  timestamp: string;
}

export interface ScoutTokenSavingsBreakdown {
  local_tokens_explored: number;
  pack_tokens_sent: number;
  tokens_saved: number;
  estimated_usd_saved: number;
  frontier_savings_ratio_pct: number;
  qlora_overhead_eliminated_tokens?: number;
}

export interface SSEContextPackReadyData {
  trace_id?: string;
  domain_id?: string;
  status?: string;
  pack?: ContextPack;
  context_pack?: ContextPack;
  quant_strategy_spec?: QuantStrategyBrief;
  quant_strategy_brief?: QuantStrategyBrief;
  token_savings?: ScoutTokenSavingsBreakdown;
  telemetry?: {
    local_tokens_consumed: number;
    frontier_tokens_sent: number;
    frontier_tokens_saved: number;
    estimated_usd_saved: number;
    savings_ratio_pct: number;
  };
  timestamp: string;
}

export interface SSEErrorData {
  error_code: string;
  message: string;
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
  estimated_paid_cost_usd: number;
  pack_token_count: number;
  estimated_cost_usd?: number;
  tokens_used?: number;
  local_tokens_consumed?: number;
  estimated_paid_savings_usd?: number;
  model_id: "gemini-3.8-flash" | "gemini-4-argon" | string;
  requires_human_confirmation: boolean;
  expires_in_seconds?: number;
  created_at?: string;
}

export interface HandoffApplyRequest {
  confirmation_token: string;
  confirmed: boolean;
  patch_diff?: string;
}

export interface HandoffApplyResponse {
  success: boolean;
  patch_diff: string;
  target_files: string[];
  lint_passed: boolean;
  tests_passed: boolean;
  verification_passed: boolean;
  confirmed: boolean;
  details: string;
}

export interface AggregatedTelemetry {
  total_requests?: number;
  total_queries?: number;
  total_local_tokens_consumed?: number;
  total_local_tokens_explored?: number;
  total_frontier_tokens_saved?: number;
  total_paid_tokens_saved?: number;
  total_usd_saved?: number;
  total_estimated_usd_saved?: number;
  frontier_token_savings_ratio_pct?: number;
  frontier_savings_ratio_pct?: number;
  qlora_few_shot_tokens_eliminated_avg?: number;
  qlora_overhead_eliminated_per_query?: number;
  total_qlora_overhead_eliminated?: number;
  hot_swap_latency_ms?: number;
}

export interface ROIMetricsResponse {
  summary?: AggregatedTelemetry;
  aggregated_telemetry?: AggregatedTelemetry;
  m6_eval_report?: Record<string, unknown>;
  m6_evaluation_report?: Record<string, unknown>;
  m7_qlora_benchmark_report?: Record<string, unknown>;
  timestamp?: string;
}

export interface DomainInfo {
  domain_id: string;
  display_name: string;
  base_model_id: string;
  lora_adapter_id: string;
  lora_adapter_path: string;
  qlora_rank: number;
  qlora_alpha: number;
  hot_swap_latency_ms: number;
  training_loss: number;
  verified: boolean;
}

export interface DomainsResponse {
  domains: DomainInfo[];
  count: number;
}

export interface ScoutQueryRequest {
  domain_id: string;
  query: string;
  execution_mode: ExecutionMode;
}

export interface ScoutQueryResponse {
  trace_id: string;
  domain_id: string;
  context_pack: ContextPack;
  telemetry: {
    local_tokens_consumed: number;
    frontier_tokens_sent: number;
    frontier_tokens_saved: number;
    estimated_usd_saved: number;
    savings_ratio_pct: number;
  };
}
