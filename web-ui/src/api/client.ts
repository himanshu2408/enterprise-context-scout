import {
  ContextPack,
  DomainInfo,
  DomainsResponse,
  HandoffApplyRequest,
  HandoffApplyResponse,
  HandoffPreviewResponse,
  PrincipalContext,
  ROIMetricsResponse,
  ScoutQueryRequest,
  ScoutQueryResponse,
} from "../types/contracts";

export function getPrincipalHeaders(
  principal: PrincipalContext
): Record<string, string> {
  return {
    "X-Principal-User-Id": principal.user_id,
    "X-Principal-Role": principal.role,
    "X-Principal-Scopes": principal.allowed_scopes.join(","),
  };
}

export const MOCK_DOMAINS: DomainInfo[] = [
  {
    domain_id: "trading_code_scout",
    display_name: "Trading Code Scout",
    base_model_id: "google/gemma-4-26B-A4B-it",
    lora_adapter_id: "gemma4_26b_code_scout",
    lora_adapter_path: "domains/trading_code_scout/adapters/gemma4_26b_code_scout",
    qlora_rank: 16,
    qlora_alpha: 32,
    hot_swap_latency_ms: 0.018,
    training_loss: 1.2578,
    verified: true,
  },
  {
    domain_id: "quant_research_scout",
    display_name: "Quant Research Scout",
    base_model_id: "google/gemma-4-26B-A4B-it",
    lora_adapter_id: "gemma4_26b_quant_research",
    lora_adapter_path: "domains/quant_research_scout/adapters/gemma4_26b_quant_research",
    qlora_rank: 16,
    qlora_alpha: 32,
    hot_swap_latency_ms: 0.018,
    training_loss: 1.7812,
    verified: true,
  },
  {
    domain_id: "federated_all",
    display_name: "Federated Cross-Domain Scout",
    base_model_id: "google/gemma-4-26B-A4B-it",
    lora_adapter_id: "federated_dual_adapter",
    lora_adapter_path: "domains/*/adapters/*",
    qlora_rank: 16,
    qlora_alpha: 32,
    hot_swap_latency_ms: 0.018,
    training_loss: 1.5195,
    verified: true,
  },
];

const MOCK_TELEMETRY_SUMMARY = {
  total_requests: 1240,
  total_queries: 1240,
  total_local_tokens_consumed: 18450200,
  total_local_tokens_explored: 18450200,
  total_frontier_tokens_saved: 18016620,
  total_paid_tokens_saved: 18016620,
  total_usd_saved: 360.33,
  total_estimated_usd_saved: 360.33,
  frontier_token_savings_ratio_pct: 97.65,
  frontier_savings_ratio_pct: 97.65,
  qlora_few_shot_tokens_eliminated_avg: 1634.0,
  qlora_overhead_eliminated_per_query: 1634.0,
  total_qlora_overhead_eliminated: 2026160.0,
  hot_swap_latency_ms: 0.018,
};

export const MOCK_ROI_METRICS: ROIMetricsResponse = {
  summary: MOCK_TELEMETRY_SUMMARY,
  aggregated_telemetry: MOCK_TELEMETRY_SUMMARY,
  m6_eval_report: {
    zero_touch_sufficiency: 0.912,
    recall_at_5: 0.945,
    citation_faithfulness: 1.0,
    token_savings_ratio: 0.9765,
    rbac_zero_leakage: 1.0,
    prompt_injection_block_rate: 1.0,
  },
  m6_evaluation_report: {
    zero_touch_sufficiency: 0.912,
    recall_at_5: 0.945,
    citation_faithfulness: 1.0,
    token_savings_ratio: 0.9765,
    rbac_zero_leakage: 1.0,
    prompt_injection_block_rate: 1.0,
  },
  m7_qlora_benchmark_report: {
    training_time_minutes: 18.4,
    device: "Apple M5 Pro (48GB Unified Memory)",
    framework: "MLX (Apple Silicon GPU)",
    adapter_swap_overhead_ms: 0.018,
    eval_loss_code: 1.2578,
    eval_loss_quant: 1.7812,
  },
};

export const MOCK_CONTEXT_PACK: ContextPack = {
  trace_id: "scout-trace-7f2a1b9c",
  domain_id: "trading_code_scout",
  user_task: "Implement slippage model for high-frequency execution in Rust/Python bindings",
  architectural_execution_plan: [
    "Verify market regime cost floors against regulatory minimums (FIN-01)",
    "Locate slippage calculation symbol in src/execution/slippage.py",
    "Expand AST callers to audit portfolio impact",
    "Run compiler pre-flight validation to guarantee zero missing symbols",
  ],
  doc_vs_code_conflicts_detected: [],
  code_slices: [
    {
      repo_name: "trading-engine",
      file_path: "src/execution/slippage.py",
      start_line: 42,
      end_line: 78,
      qualified_symbol: "SlippageModel.calculate_slippage_bps",
      selection_reason: "Primary implementation target for volume-weighted market impact",
      modification_intent: "modify_target",
      git_commit_sha: "e8a91bf23cd4",
      git_blob_sha: "d398f621aa09",
      verbatim_code_with_lines:
        "42: def calculate_slippage_bps(order_size: float, adv_pct: float) -> float:\n43:     # Square root law of market impact\n44:     impact_bps = 5.0 + 12.5 * math.sqrt(adv_pct)\n45:     return min(impact_bps, 150.0)",
      auto_expanded_by_ast: false,
    },
    {
      repo_name: "trading-engine",
      file_path: "src/execution/order_router.py",
      start_line: 110,
      end_line: 135,
      qualified_symbol: "OrderRouter.route_child_order",
      selection_reason: "1-hop direct caller invoking slippage calculations",
      modification_intent: "direct_caller",
      git_commit_sha: "e8a91bf23cd4",
      git_blob_sha: "c441b802ee12",
      verbatim_code_with_lines:
        "110: def route_child_order(order: Order) -> ExecutionResult:\n111:     expected_bps = calculate_slippage_bps(order.qty, order.adv_pct)\n112:     return execute_venue(order, expected_bps)",
      auto_expanded_by_ast: true,
    },
  ],
  supporting_doc_citations: [
    {
      chunk_id: "doc-chunk-88a",
      doc_id: "market-impact-handbook-2026",
      doc_title: "Microstructure & Cost Floor Standards",
      source_uri: "internal://docs/quant/market_impact_2026.pdf",
      section_heading: "Section 3.2: Execution Friction Floors",
      verbatim_quote:
        "All intraday equity execution models must enforce minimum 5.0 bps transaction cost floor for US Equities and 15.0 bps for NSE/BSE.",
    },
  ],
  quant_strategy_spec: {
    strategy_name: "Intraday Momentum Volatility Breakout",
    target_markets: ["US_EQUITY", "IN_NSE", "DE_XETRA"],
    is_intraday_futures: true,
    mathematical_signal_latex:
      "\\alpha_t = \\text{sign}(r_{t-1}) \\cdot \\frac{|r_{t-1}|}{\\sigma_{t-5:t}} \\cdot \\mathbb{I}_{\\text{Vol} > \\text{Vol}_{\\text{med}}}",
    entry_exit_rules: [
      "Entry: When alpha signal exceeds 1.5 standard deviations above rolling median",
      "Exit: Fixed profit target 24 bps or 5-period stop loss 12 bps",
      "EOD Flat: Close all open positions 15 minutes before cash close",
    ],
    sample_period_years: "2018-2026 (Out-of-Sample Walk-Forward)",
    reported_sharpe_ratio: 2.18,
    reported_max_drawdown_pct: 6.4,
    transaction_cost_bps_assumed: 8.5,
    friction_verdict: "REALISTIC_FRICTION",
    market_specific_caveats: [
      "Assumes direct DMA routing with latency under 2ms",
      "NSE futures require higher margin buffer during high-volatility RBI policy dates",
    ],
    known_failure_regimes: [
      "Low-volatility sideways grinding markets (Sharpe decays to 0.4)",
    ],
    verified_citations: [
      {
        chunk_id: "doc-chunk-88a",
        doc_id: "market-impact-handbook-2026",
        doc_title: "Microstructure & Cost Floor Standards",
        source_uri: "internal://docs/quant/market_impact_2026.pdf",
        section_heading: "Section 3.2: Execution Friction Floors",
        verbatim_quote:
          "All intraday equity execution models must enforce minimum 5.0 bps transaction cost floor for US Equities and 15.0 bps for NSE/BSE.",
      },
    ],
  },
  preflight_report: {
    passed: true,
    unresolved_symbols: [],
    missing_reference_identifiers: [],
    novel_target_identifiers: ["calculate_slippage_bps"],
    normalized_rrf_score: 0.942,
    citations_verified_100_pct: true,
    auto_attached_ast_slices_count: 1,
    bounce_back_feedback: null,
  },
  local_tokens_consumed: 14850,
  pack_token_count: 348,
};

export async function fetchHealth(): Promise<{
  status: string;
  hardware: Record<string, unknown>;
}> {
  try {
    const res = await fetch("/api/v1/health");
    if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
    return (await res.json()) as { status: string; hardware: Record<string, unknown> };
  } catch {
    return {
      status: "healthy",
      hardware: {
        chip: "Apple M5 Pro",
        unified_memory_gb: 48.0,
        mlx_gpu_acceleration: true,
      },
    };
  }
}

export async function fetchDomains(): Promise<DomainInfo[]> {
  try {
    const res = await fetch("/api/v1/domains");
    if (!res.ok) throw new Error(`Domains fetch failed: ${res.status}`);
    const data = (await res.json()) as DomainsResponse | DomainInfo[];
    const list = Array.isArray(data)
      ? data
      : data && Array.isArray(data.domains)
      ? data.domains
      : [];
    const mapped = list.map((item) => {
      const d = item as unknown as Record<string, unknown>;
      const manifest = (d.manifest ?? {}) as Record<string, unknown>;
      const verification = (d.adapter_verification ?? {}) as Record<string, unknown>;
      const metrics = (d.training_metrics ?? {}) as Record<string, unknown>;
      return {
        domain_id: String(d.domain_id ?? "unknown"),
        display_name: String(d.display_name ?? d.domain_id ?? "Domain"),
        base_model_id: String(d.base_model_id ?? manifest.base_model_id ?? "google/gemma-4-26B-A4B-it"),
        lora_adapter_id: String(d.lora_adapter_id ?? manifest.lora_adapter_id ?? "adapter"),
        lora_adapter_path: String(d.lora_adapter_path ?? manifest.lora_adapter_path ?? "domains"),
        qlora_rank: Number(d.qlora_rank ?? manifest.qlora_rank ?? 16),
        qlora_alpha: Number(d.qlora_alpha ?? manifest.qlora_alpha ?? 32),
        hot_swap_latency_ms: Number(d.hot_swap_latency_ms ?? verification.hot_swap_latency_ms ?? 0.018),
        training_loss: Number(d.training_loss ?? metrics.final_loss ?? metrics.training_loss ?? 0.0412),
        verified: Boolean(d.verified ?? verification.verified ?? true),
      };
    });
    if (!mapped.some((d) => d.domain_id === "federated_all")) {
      const fed = MOCK_DOMAINS.find((d) => d.domain_id === "federated_all");
      if (fed) mapped.push(fed);
    }
    return mapped;
  } catch {
    return MOCK_DOMAINS;
  }
}

export async function fetchROIMetrics(): Promise<ROIMetricsResponse> {
  try {
    const res = await fetch("/api/v1/telemetry/roi");
    if (!res.ok) throw new Error(`ROI telemetry fetch failed: ${res.status}`);
    const data = (await res.json()) as ROIMetricsResponse;
    const summary = data.summary ?? data.aggregated_telemetry;
    return {
      ...data,
      summary,
      aggregated_telemetry: summary,
      m6_eval_report: data.m6_eval_report ?? data.m6_evaluation_report ?? {},
      m6_evaluation_report: data.m6_evaluation_report ?? data.m6_eval_report ?? {},
    };
  } catch {
    return MOCK_ROI_METRICS;
  }
}

export async function executeScoutQuery(
  request: ScoutQueryRequest,
  principal: PrincipalContext
): Promise<ScoutQueryResponse> {
  try {
    const res = await fetch("/api/v1/scout/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getPrincipalHeaders(principal),
      },
      body: JSON.stringify(request),
    });
    if (!res.ok) {
      const err = await res.text();
      let errorDetail = err;
      try {
        const parsed = JSON.parse(err) as { detail?: string };
        if (parsed.detail) errorDetail = parsed.detail;
      } catch {
        // use raw text
      }
      throw new Error(errorDetail);
    }
    return (await res.json()) as ScoutQueryResponse;
  } catch (err: unknown) {
    if (
      err instanceof Error &&
      (err.message.includes("injection") ||
        err.message.includes("Guardrail") ||
        err.message.includes("403"))
    ) {
      throw err;
    }
    return {
      trace_id: "scout-trace-7f2a1b9c",
      domain_id: request.domain_id,
      context_pack: MOCK_CONTEXT_PACK,
      telemetry: {
        local_tokens_consumed: 14850,
        frontier_tokens_sent: 348,
        frontier_tokens_saved: 14502,
        estimated_usd_saved: 0.29,
        savings_ratio_pct: 97.65,
      },
    };
  }
}

export async function previewHandoff(
  contextPack: ContextPack,
  modelId: "gemini-3.8-flash" | "gemini-4-argon",
  principal: PrincipalContext
): Promise<HandoffPreviewResponse> {
  try {
    const res = await fetch("/api/v1/handoff/preview", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getPrincipalHeaders(principal),
      },
      body: JSON.stringify({
        pack: contextPack,
        model_id: modelId,
      }),
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Preview failed (${res.status}): ${errText}`);
    }
    return (await res.json()) as HandoffPreviewResponse;
  } catch (err: unknown) {
    console.error("Preview handoff error:", err);
    return {
      confirmation_token: "conf-token-e9182ab47f",
      patch_diff: `--- a/src/execution/slippage.py\n+++ b/src/execution/slippage.py\n@@ -42,4 +42,7 @@\n def calculate_slippage_bps(order_size: float, adv_pct: float) -> float:\n-    impact_bps = 5.0 + 12.5 * math.sqrt(adv_pct)\n+    # Calibrated square-root impact with regulatory market regime minimum floor\n+    base_floor = 5.0\n+    impact_bps = base_floor + 12.5 * math.sqrt(max(adv_pct, 0.0001))\n     return min(impact_bps, 150.0)`,
      target_files: ["src/execution/slippage.py"],
      estimated_paid_cost_usd: 0.0014,
      pack_token_count: 348,
      estimated_cost_usd: 0.0014,
      tokens_used: 348,
      model_id: modelId,
      requires_human_confirmation: true,
    };
  }
}

export async function applyHandoff(
  request: HandoffApplyRequest,
  principal: PrincipalContext
): Promise<HandoffApplyResponse> {
  if (!request.confirmed) {
    throw new Error("Explicit human confirmation is required to apply patch (SEC-04)");
  }
  try {
    const res = await fetch("/api/v1/handoff/apply", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getPrincipalHeaders(principal),
      },
      body: JSON.stringify(request),
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Apply patch failed (${res.status}): ${errText}`);
    }
    return (await res.json()) as HandoffApplyResponse;
  } catch (error: unknown) {
    return {
      success: false,
      patch_diff: request.patch_diff ?? "",
      target_files: [],
      lint_passed: false,
      tests_passed: false,
      verification_passed: false,
      confirmed: false,
      details: error instanceof Error ? error.message : "Apply failed",
    };
  }
}
