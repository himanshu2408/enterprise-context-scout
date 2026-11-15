import { vi } from "vitest";
import {
  AuthSessionResponse,
  BackendDomainDetail,
  ContextPack,
  DocCitation,
  DomainInfo,
  HandoffApplyResponse,
  HandoffPreviewResponse,
  HealthResponse,
  InsufficientContextDiagnostic,
  PrincipalContext,
  ROIMetricsResponse,
  ScoutQueryResponse,
  ScoutTokenSavingsBreakdown,
  SSEStageEvent,
} from "../types/contracts";

export const TEST_ENGINEER_PRINCIPAL: PrincipalContext = {
  user_id: "test_engineer_01",
  role: "engineer",
  allowed_scopes: ["public_eng", "public_quant"],
  max_session_paid_tokens: 8000,
};

export const TEST_HEALTH_RESPONSE: HealthResponse = {
  status: "healthy",
  database_connected: true,
  database_status: "PostgreSQL 17.11 RLS Enforced",
  registered_domains: [
    "trading_code_scout",
    "quant_research_scout",
    "polyglot_unseen_v2",
  ],
  adapter_pool_status: {
    trading_code_scout: {
      domain_id: "trading_code_scout",
      is_valid: true,
      adapter_path: "adapters/trading_code_scout",
      safetensors_bytes: 18432000,
      r: 8,
      lora_alpha: 16,
      is_moe: true,
      final_loss: 0.72,
      error_message: null,
      hot_swap_latency_ms: 0.02,
    },
    quant_research_scout: {
      domain_id: "quant_research_scout",
      is_valid: true,
      adapter_path: "adapters/quant_research_scout",
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
    domain_id: "trading_code_scout",
    display_name: "Trading Codebase & Execution AST",
    manifest: {
      domain_id: "trading_code_scout",
      display_name: "Trading Codebase & Execution AST",
      base_model_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
      lora_adapter_id: "trading_code_scout_v1",
      lora_adapter_path: "adapters/trading_code_scout",
      authority_rules: { primary_truth_source: "code" },
      output_contract: "ImplementationContextPack",
    },
    adapter_verification: TEST_HEALTH_RESPONSE.adapter_pool_status.trading_code_scout,
    training_metrics: { final_loss: 0.72 },
  },
  {
    domain_id: "quant_research_scout",
    display_name: "Quantitative Research & Market Papers",
    manifest: {
      domain_id: "quant_research_scout",
      display_name: "Quantitative Research & Market Papers",
      base_model_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
      lora_adapter_id: "quant_research_scout_v1",
      lora_adapter_path: "adapters/quant_research_scout",
      authority_rules: { primary_truth_source: "document" },
      output_contract: "QuantStrategyBrief",
    },
    adapter_verification: TEST_HEALTH_RESPONSE.adapter_pool_status.quant_research_scout,
    training_metrics: { final_loss: 0.68 },
  },
  {
    domain_id: "polyglot_unseen_v2",
    display_name: "Zero-Shot Domain Verification",
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

export const TEST_DOMAINS: DomainInfo[] = [
  {
    domain_id: "trading_code_scout",
    display_name: "Trading Codebase & Execution AST",
    base_model_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
    lora_adapter_id: "trading_code_scout_v1",
    lora_adapter_path: "adapters/trading_code_scout",
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
    domain_id: "quant_research_scout",
    display_name: "Quantitative Research & Market Papers",
    base_model_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
    lora_adapter_id: "quant_research_scout_v1",
    lora_adapter_path: "adapters/quant_research_scout",
    primary_truth_source: "document",
    output_contract: "QuantStrategyBrief",
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
  chunk_id: "chunk_hawkes_01",
  doc_id: "doc_hawkes_orderbook",
  version: "v1.0",
  doc_title: "The role of volume in order book dynamics: a multivariate Hawkes process analysis",
  source_uri: "https://arxiv.org/abs/example-hawkes",
  section_heading: "3.2 Excitation Kernel Stability",
  verbatim_quote: "The spectral radius of the branching matrix remains strictly below 1.0 under stationary volume regimes.",
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
  domain_id: "trading_code_scout",
  user_task: "Inspect AlmgrenChrissExecutionOptimizer liquidation schedule and 1-hop caller contracts.",
  architectural_execution_plan: [
    "Review AlmgrenChrissExecutionOptimizer.compute_trajectory in execution/optimizer.py lines 14-48.",
    "Preserve risk aversion validation and VolatilityForecastResult caller contract.",
  ],
  doc_vs_code_conflicts_detected: [],
  code_slices: [
    {
      repo_name: "quant-exec-core",
      file_path: "execution/optimizer.py",
      start_line: 14,
      end_line: 32,
      qualified_symbol: "AlmgrenChrissExecutionOptimizer.compute_trajectory",
      selection_reason: "Core optimal execution trajectory calculation with risk aversion check.",
      modification_intent: "modify_target",
      git_commit_sha: "abc1234def5678",
      git_blob_sha: "blob987654",
      verbatim_code_with_lines:
        "14: class AlmgrenChrissExecutionOptimizer:\n" +
        "15:     def compute_trajectory(self, shares: float, volatility: float) -> list[float]:\n" +
        "16:         if shares <= 0:\n" +
        "17:             raise ValueError('shares must be positive')\n" +
        "18:         return [shares * 0.5, shares * 0.5]",
      auto_expanded_by_ast: false,
    },
    {
      repo_name: "quant-exec-core",
      file_path: "execution/volatility.py",
      start_line: 5,
      end_line: 14,
      qualified_symbol: "VolatilityForecastResult",
      selection_reason: "1-hop AST dependency supplying volatility estimate.",
      modification_intent: "imported_type",
      git_commit_sha: "abc1234def5678",
      git_blob_sha: "blob444333",
      verbatim_code_with_lines:
        "5: @dataclass(frozen=True)\n" +
        "6: class VolatilityForecastResult:\n" +
        "7:     sigma_annualized: float\n" +
        "8:     horizon_minutes: int",
      auto_expanded_by_ast: true,
    },
  ],
  supporting_doc_citations: [TEST_DOC_CITATION],
  quant_strategy_spec: null,
  domain_knowledge_brief: {
    brief_title: "Almgren-Chriss Optimal Execution Schedule & Risk Bounds",
    domain_id: "trading_code_scout",
    executive_summary:
      "AlmgrenChrissExecutionOptimizer splits parent orders using VolatilityForecastResult and enforces positive share guards before scheduling child slices.",
    formal_expression_or_rule: "x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T)",
    core_findings_and_rules: [
      "Execution optimizer consumes VolatilityForecastResult.sigma_annualized directly.",
      "AST 1-hop dependency closure attached execution/volatility.py automatically.",
    ],
    domain_constraints_and_caveats: [
      "Rejects non-positive share quantities fail-closed before frontier handoff.",
    ],
    known_risks_and_failure_modes: [
      "Temporary impact assumption degrades under flash illiquidity spikes.",
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

export const TEST_QUANT_PACK: ContextPack = {
  ...TEST_CODE_PACK,
  trace_id: "tr_test_quant_002",
  domain_id: "quant_research_scout",
  user_task: "Explain Hawkes process excitation kernel stability and slippage friction assumptions.",
  doc_vs_code_conflicts_detected: [
    "Code uses static 2-bps impact whereas Hawkes kernel research recommends regime-scaled impact.",
  ],
  code_slices: [],
  quant_strategy_spec: {
    strategy_name: "Multivariate Hawkes Order-Book Imbalance",
    target_markets: ["DE_XETRA", "US_EQUITY"],
    is_intraday_futures: true,
    mathematical_signal_latex: "\\lambda_t = \\mu + \\int_0^t \\alpha e^{-\\beta (t - s)} dN_s",
    entry_exit_rules: [
      "Enter when buy-side intensity exceeds sell-side intensity by 2.5x baseline.",
      "Flatten inventory if branching spectral radius approaches 0.95.",
    ],
    sample_period_years: "2018-2024",
    reported_sharpe_ratio: 1.42,
    reported_max_drawdown_pct: 6.4,
    transaction_cost_bps_assumed: 1.5,
    friction_verdict: "UNVERIFIED_FRICTION_WARNING",
    market_specific_caveats: [
      "Queue priority modeling ignores hidden iceberg replenishment.",
    ],
    known_failure_regimes: ["Opening auction imbalances", "Central bank announcement windows"],
    verified_citations: [TEST_DOC_CITATION],
  },
};

export const TEST_DIAGNOSTIC: InsufficientContextDiagnostic = {
  trace_id: "tr_test_abstain_003",
  domain_id: "trading_code_scout",
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
  quant_strategy_spec: TEST_CODE_PACK.quant_strategy_spec,
  token_savings: TEST_TOKEN_SAVINGS,
  trace_attributes: { execution_mode: "chat_qa" },
};

export const TEST_HANDOFF_PREVIEW: HandoffPreviewResponse = {
  confirmation_token: "confirm_tok_test_001",
  patch_diff:
    "--- a/execution/optimizer.py\n" +
    "+++ b/execution/optimizer.py\n" +
    "@@ -16,2 +16,3 @@\n" +
    "+        if volatility <= 0:\n" +
    "+            raise ValueError('volatility must be positive')\n",
  target_files: ["execution/optimizer.py"],
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
  target_files: ["execution/optimizer.py"],
  lint_passed: true,
  tests_passed: true,
  verification_passed: true,
  details: "Applied patch in sandbox worktree and passed ruff + pytest.",
};

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
  healthStatus?: number;
  healthResponse?: HealthResponse;
  domainsStatus?: number;
  domainsDetail?: BackendDomainDetail[];
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
}

export function installMockApiFetch(overrides: MockApiOverrides = {}) {
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
        return new Response(JSON.stringify(buildTestSession(principal)), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      if (url.endsWith("/api/v1/health")) {
        const status = overrides.healthStatus ?? 200;
        if (status >= 400) {
          return new Response(JSON.stringify({ detail: "Health failed" }), {
            status,
            headers: { "Content-Type": "application/json" },
          });
        }
        return new Response(
          JSON.stringify(overrides.healthResponse ?? TEST_HEALTH_RESPONSE),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      if (url.endsWith("/api/v1/domains")) {
        const status = overrides.domainsStatus ?? 200;
        if (status >= 400) {
          return new Response(JSON.stringify({ detail: "Domains failed" }), {
            status,
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
              quant_strategy_spec: null,
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
        return new Response(
          JSON.stringify(overrides.applyResponse ?? TEST_HANDOFF_APPLY),
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
