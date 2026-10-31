import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { CitationInspector } from "../components/CitationInspector";
import { TokenSavingsMeter } from "../components/TokenSavingsMeter";
import { GeminiDiffModal } from "../components/GeminiDiffModal";
import { RoleSwitcher, ROLE_DEFAULT_SCOPES } from "../components/RoleSwitcher";
import { DomainSelector } from "../components/DomainSelector";
import { MOCK_DOMAINS } from "../api/client";
import {
  ContextPack,
  PrincipalContext,
  RoleType,
} from "../types/contracts";

describe("Adversarial Stress Test: CitationInspector Resilience", () => {
  it("safely handles completely empty code_slices array without crashing", () => {
    const emptyPack: ContextPack = {
      trace_id: "stress-trace-001",
      domain_id: "trading_code_scout",
      user_task: "Stress test empty code slices",
      architectural_execution_plan: ["Step 1: Inspect empty slice list"],
      doc_vs_code_conflicts_detected: [],
      code_slices: [],
      supporting_doc_citations: [],
      quant_strategy_spec: null,
      preflight_report: {
        passed: false,
        unresolved_symbols: [],
        missing_reference_identifiers: [],
        novel_target_identifiers: [],
        normalized_rrf_score: 0.0,
        citations_verified_100_pct: false,
        auto_attached_ast_slices_count: 0,
        bounce_back_feedback: "Empty slice set",
      },
      local_tokens_consumed: 0,
      pack_token_count: 0,
    };

    render(<CitationInspector contextPack={emptyPack} />);

    expect(screen.getByTestId("code-slices-view")).toBeInTheDocument();
    expect(screen.getByText("No code slices in this pack.")).toBeInTheDocument();
    expect(screen.getByText("Code Slices (0)")).toBeInTheDocument();
  });

  it("handles code slice with undefined auto_expanded_by_ast and arbitrary modification intent", () => {
    const slicePack: ContextPack = {
      trace_id: "stress-trace-002",
      domain_id: "trading_code_scout",
      user_task: "Stress test optional fields in code slice",
      architectural_execution_plan: ["Validate auto_expanded_by_ast undefined behavior"],
      doc_vs_code_conflicts_detected: [],
      code_slices: [
        {
          repo_name: "quant-core",
          file_path: "src/models/signal.py",
          start_line: 1,
          end_line: 10,
          qualified_symbol: "signal_fn",
          selection_reason: "Primary entry point",
          modification_intent: "reference_only",
          git_commit_sha: "abc1234",
          git_blob_sha: "def5678",
          verbatim_code_with_lines: "1: def signal_fn(): pass",
          auto_expanded_by_ast: undefined,
        },
      ],
      supporting_doc_citations: [],
      quant_strategy_spec: null,
      preflight_report: {
        passed: true,
        unresolved_symbols: [],
        missing_reference_identifiers: [],
        novel_target_identifiers: [],
        normalized_rrf_score: 1.0,
        citations_verified_100_pct: true,
        auto_attached_ast_slices_count: 0,
        bounce_back_feedback: null,
      },
      local_tokens_consumed: 100,
      pack_token_count: 50,
    };

    render(<CitationInspector contextPack={slicePack} />);

    expect(screen.getByText("REFERENCE ONLY")).toBeInTheDocument();
    expect(screen.queryByTestId("ast-closure-badge")).not.toBeInTheDocument();
  });

  it("handles quant strategy with all null metrics and unformatted LaTeX without throwing", () => {
    const quantPack: ContextPack = {
      trace_id: "stress-trace-003",
      domain_id: "quant_research_scout",
      user_task: "Stress test quant strategy with null metrics",
      architectural_execution_plan: ["Test null numerical properties"],
      doc_vs_code_conflicts_detected: [],
      code_slices: [],
      supporting_doc_citations: [],
      quant_strategy_spec: {
        strategy_name: "Zero Metrics Edge Case Strategy",
        target_markets: ["GLOBAL"],
        is_intraday_futures: false,
        mathematical_signal_latex: "%%UNFORMATTED_RAW_LATEX_$$ \\frac{d}{dx} \\sum_{i=0}^\\infty &&",
        entry_exit_rules: ["Rule 1: Always check limits"],
        sample_period_years: "Unknown Sample Period",
        reported_sharpe_ratio: null,
        reported_max_drawdown_pct: null,
        transaction_cost_bps_assumed: null,
        friction_verdict: "UNVERIFIED_FRICTION_WARNING",
        market_specific_caveats: ["Untested liquidity regime"],
        known_failure_regimes: ["High volatility spikes"],
        verified_citations: [],
      },
      preflight_report: {
        passed: false,
        unresolved_symbols: ["unresolved_kernel"],
        missing_reference_identifiers: ["id_x"],
        novel_target_identifiers: [],
        normalized_rrf_score: 0.42,
        citations_verified_100_pct: false,
        auto_attached_ast_slices_count: 0,
        bounce_back_feedback: "Missing verified citations",
      },
      local_tokens_consumed: 500,
      pack_token_count: 120,
    };

    render(<CitationInspector contextPack={quantPack} />);

    const quantTab = screen.getByTestId("tab-quant-strategy");
    fireEvent.click(quantTab);

    expect(screen.getByText("Zero Metrics Edge Case Strategy")).toBeInTheDocument();
    expect(screen.getByTestId("latex-mathematical-signal")).toHaveTextContent(
      "%%UNFORMATTED_RAW_LATEX_$$ \\frac{d}{dx} \\sum_{i=0}^\\infty &&"
    );
    expect(screen.getByTestId("friction-verdict-badge")).toHaveTextContent(
      "UNVERIFIED FRICTION WARNING"
    );
    expect(screen.getByText("N/A")).toBeInTheDocument();
    expect(screen.getByText("N/A%")).toBeInTheDocument();
    expect(screen.getByText("0 bps")).toBeInTheDocument();

    const preflightTab = screen.getByTestId("tab-preflight-report");
    fireEvent.click(preflightTab);

    expect(screen.getByTestId("preflight-passed-badge")).toHaveTextContent("FAILED");
    expect(screen.getByText("Normalized RRF: 42.0%")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("NO")).toBeInTheDocument();
  });
});

describe("Adversarial Stress Test: TokenSavingsMeter Edge Cases", () => {
  it("safely handles zero tokens and division-by-zero boundary conditions", () => {
    const zeroTelemetry = {
      local_tokens_consumed: 0,
      frontier_tokens_sent: 0,
      frontier_tokens_saved: 0,
      estimated_usd_saved: 0,
      savings_ratio_pct: 0,
    };

    render(<TokenSavingsMeter telemetry={zeroTelemetry} />);

    expect(screen.getByTestId("frontier-savings-ratio")).toHaveTextContent("0.00%");
    expect(screen.getByTestId("local-tokens-explored")).toHaveTextContent("0");
    expect(screen.getByTestId("frontier-tokens-sent")).toHaveTextContent("0");
    expect(screen.getByTestId("usd-saved")).toHaveTextContent("$0.00");
  });

  it("handles negative delta overhead and ensures UI does not break", () => {
    const negativeTelemetry = {
      local_tokens_consumed: 100,
      frontier_tokens_sent: 250,
      frontier_tokens_saved: -150,
      estimated_usd_saved: -0.05,
      savings_ratio_pct: -50.0,
    };

    render(<TokenSavingsMeter telemetry={negativeTelemetry} />);

    expect(screen.getByTestId("frontier-savings-ratio")).toHaveTextContent("-50.00%");
    expect(screen.getByTestId("usd-saved")).toHaveTextContent("$-0.05");
  });

  it("clamps progress bar fill width safely when ratio exceeds 100%", () => {
    const overflowTelemetry = {
      local_tokens_consumed: 50000,
      frontier_tokens_sent: 100,
      frontier_tokens_saved: 49900,
      estimated_usd_saved: 1.25,
      savings_ratio_pct: 120.5,
    };

    const { container } = render(<TokenSavingsMeter telemetry={overflowTelemetry} />);

    expect(screen.getByTestId("frontier-savings-ratio")).toHaveTextContent("120.50%");
    const fill = container.querySelector(".progress-bar-fill") as HTMLElement;
    expect(fill.style.width).toBe("100%");
  });

  it("safely defaults when telemetry and aggregatedMetrics are undefined", () => {
    render(<TokenSavingsMeter telemetry={undefined} aggregatedMetrics={undefined} />);

    expect(screen.getByTestId("frontier-savings-ratio")).toHaveTextContent("97.65%");
    expect(screen.getByTestId("local-tokens-explored")).toHaveTextContent("14,850");
    expect(screen.getByTestId("frontier-tokens-sent")).toHaveTextContent("348");
  });
});

describe("Adversarial Stress Test: GeminiDiffModal Human Confirmation Enclosure", () => {
  const principal: PrincipalContext = {
    user_id: "sec_tester",
    role: "engineer",
    allowed_scopes: ["public_eng"],
    max_session_paid_tokens: 16000,
  };

  const dummyPack: ContextPack = {
    trace_id: "stress-trace-diff",
    domain_id: "trading_code_scout",
    user_task: "Verify diff confirmation enclosure",
    architectural_execution_plan: ["Apply diff verification gate"],
    doc_vs_code_conflicts_detected: [],
    code_slices: [],
    supporting_doc_citations: [],
    quant_strategy_spec: null,
    preflight_report: {
      passed: true,
      unresolved_symbols: [],
      missing_reference_identifiers: [],
      novel_target_identifiers: [],
      normalized_rrf_score: 1.0,
      citations_verified_100_pct: true,
      auto_attached_ast_slices_count: 0,
      bounce_back_feedback: null,
    },
    local_tokens_consumed: 1000,
    pack_token_count: 100,
  };

  it("strictly disables the Approve button until checkbox is explicitly clicked", async () => {
    const onClose = vi.fn();
    render(
      <GeminiDiffModal
        isOpen={true}
        onClose={onClose}
        contextPack={dummyPack}
        principal={principal}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("unified-diff-view")).toBeInTheDocument();
    });

    const approveBtn = screen.getByTestId("approve-apply-btn");
    expect(approveBtn).toBeDisabled();

    // Directly attempting to click disabled button must not cause apply or state transition
    fireEvent.click(approveBtn);
    expect(screen.queryByTestId("apply-success-view")).not.toBeInTheDocument();

    // Check confirmation
    const checkbox = screen.getByTestId("confirm-apply-checkbox");
    fireEvent.click(checkbox);
    expect(approveBtn).not.toBeDisabled();

    // Uncheck confirmation
    fireEvent.click(checkbox);
    expect(approveBtn).toBeDisabled();
  });

  it("handles modal close/reject cleanly without triggering sandbox apply", async () => {
    const onClose = vi.fn();
    const onApplyComplete = vi.fn();

    render(
      <GeminiDiffModal
        isOpen={true}
        onClose={onClose}
        contextPack={dummyPack}
        principal={principal}
        onApplyComplete={onApplyComplete}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("unified-diff-view")).toBeInTheDocument();
    });

    const rejectBtn = screen.getByTestId("reject-patch-btn");
    fireEvent.click(rejectBtn);

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onApplyComplete).not.toHaveBeenCalled();
  });

  it("handles model switching between gemini-3.8-flash and gemini-4-argon", async () => {
    const onClose = vi.fn();
    render(
      <GeminiDiffModal
        isOpen={true}
        onClose={onClose}
        contextPack={dummyPack}
        principal={principal}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("unified-diff-view")).toBeInTheDocument();
    });

    const argonBtn = screen.getByTestId("select-model-argon");
    fireEvent.click(argonBtn);

    await waitFor(() => {
      expect(argonBtn).toHaveClass("btn-pill-active");
    });
  });
});

describe("Adversarial Stress Test: RoleSwitcher Coverage & Scope Boundaries", () => {
  const allRoles: RoleType[] = [
    "engineer",
    "quant_researcher",
    "product_manager",
    "compliance_officer",
  ];

  it("supports all 4 enterprise roles with their respective RBAC scopes", () => {
    const onPrincipalChange = vi.fn();
    const principal: PrincipalContext = {
      user_id: "test_user",
      role: "engineer",
      allowed_scopes: ROLE_DEFAULT_SCOPES.engineer,
      max_session_paid_tokens: 16000,
    };

    const { rerender } = render(
      <RoleSwitcher principal={principal} onPrincipalChange={onPrincipalChange} />
    );

    for (const role of allRoles) {
      const btn = screen.getByTestId(`role-btn-${role}`);
      expect(btn).toBeInTheDocument();
      fireEvent.click(btn);

      const expectedScopes = ROLE_DEFAULT_SCOPES[role];
      expect(onPrincipalChange).toHaveBeenCalledWith(
        expect.objectContaining({
          role,
          allowed_scopes: expectedScopes,
        })
      );

      // Re-render with new principal to verify scope chip display
      rerender(
        <RoleSwitcher
          principal={{
            ...principal,
            role,
            allowed_scopes: expectedScopes,
          }}
          onPrincipalChange={onPrincipalChange}
        />
      );

      for (const scope of expectedScopes) {
        expect(screen.getByTestId(`scope-chip-${scope}`)).toBeInTheDocument();
      }
    }
  });

  it("handles empty and special characters in user_id without breaking", () => {
    const onPrincipalChange = vi.fn();
    const principal: PrincipalContext = {
      user_id: "test_user",
      role: "engineer",
      allowed_scopes: ROLE_DEFAULT_SCOPES.engineer,
      max_session_paid_tokens: 16000,
    };

    render(
      <RoleSwitcher principal={principal} onPrincipalChange={onPrincipalChange} />
    );

    const input = screen.getByTestId("principal-user-id-input");
    fireEvent.change(input, { target: { value: "<script>alert(1)</script>" } });

    expect(onPrincipalChange).toHaveBeenCalledWith(
      expect.objectContaining({
        user_id: "<script>alert(1)</script>",
      })
    );
  });
});

describe("Adversarial Stress Test: DomainSelector Full Domain Matrix & QLoRA Latency", () => {
  it("reflects all 3 domain options and displays 0.018 ms latency and r=16, alpha=32 specs", () => {
    const onDomainChange = vi.fn();
    render(
      <DomainSelector
        selectedDomainId="trading_code_scout"
        onDomainChange={onDomainChange}
        domains={MOCK_DOMAINS}
      />
    );

    expect(screen.getByTestId("domain-card-trading_code_scout")).toBeInTheDocument();
    expect(screen.getByTestId("domain-card-quant_research_scout")).toBeInTheDocument();
    expect(screen.getByTestId("domain-card-federated_all")).toBeInTheDocument();

    // Verify hot-swap latency banner
    expect(screen.getByText(/Hot-Swap: 0.018 ms/i)).toBeInTheDocument();

    // Verify each card contains r=16, alpha=32 and 0.018 ms swap
    const rankSpecs = screen.getAllByText("r=16, α=32");
    expect(rankSpecs.length).toBe(3);

    const latencySpecs = screen.getAllByText("0.018 ms swap");
    expect(latencySpecs.length).toBe(3);
  });
});
