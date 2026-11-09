import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { CitationInspector } from "../components/CitationInspector";
import { MOCK_CONTEXT_PACK } from "../api/client";

describe("CitationInspector Component", () => {
  it("renders empty state message when contextPack is null", () => {
    render(<CitationInspector contextPack={null} />);
    expect(screen.getByText("No Active ContextPack Emitted")).toBeInTheDocument();
  });

  it("renders Code Slices tab with verbatim code and AST closure badges", () => {
    render(<CitationInspector contextPack={MOCK_CONTEXT_PACK} />);

    expect(screen.getByTestId("code-slices-view")).toBeInTheDocument();
    expect(screen.getByText("src/execution/slippage.py")).toBeInTheDocument();
    expect(screen.getByText("src/execution/order_router.py")).toBeInTheDocument();

    // Verify AST 1-Hop Closure badge on the second slice
    expect(screen.getByTestId("ast-closure-badge")).toBeInTheDocument();
    expect(screen.getByText("TARGET TO MODIFY")).toBeInTheDocument();
    expect(screen.getByText("DIRECT CALLER (AST)")).toBeInTheDocument();

    // Verify code block
    expect(screen.getByTestId("code-block-0")).toHaveTextContent(
      "calculate_slippage_bps"
    );
  });

  it("switches to Quant Strategy tab and renders LaTeX signal and friction verdict", () => {
    render(<CitationInspector contextPack={MOCK_CONTEXT_PACK} />);

    const quantTab = screen.getByTestId("tab-quant-strategy");
    fireEvent.click(quantTab);

    expect(screen.getByTestId("quant-strategy-view")).toBeInTheDocument();
    expect(
      screen.getByText("Intraday Momentum Volatility Breakout")
    ).toBeInTheDocument();

    // LaTeX formula
    expect(screen.getByTestId("latex-mathematical-signal")).toHaveTextContent(
      "\\alpha_t = \\text{sign}(r_{t-1})"
    );

    // Friction verdict
    expect(screen.getByTestId("friction-verdict-badge")).toHaveTextContent(
      "REALISTIC FRICTION"
    );

    // Market regime tags
    expect(screen.getByTestId("market-regimes-list")).toBeInTheDocument();
    expect(screen.getByText("US_EQUITY")).toBeInTheDocument();
    expect(screen.getByText("IN_NSE")).toBeInTheDocument();
    expect(screen.getByText("DE_XETRA")).toBeInTheDocument();

    // Verified citations
    expect(
      screen.getByText("Microstructure & Cost Floor Standards")
    ).toBeInTheDocument();
    expect(
      screen.getByText(/All intraday equity execution models must enforce minimum 5.0 bps/i)
    ).toBeInTheDocument();
  });

  it("switches to Pre-Flight Gate tab and renders validation summary", () => {
    render(<CitationInspector contextPack={MOCK_CONTEXT_PACK} />);

    const preflightTab = screen.getByTestId("tab-preflight-report");
    fireEvent.click(preflightTab);

    expect(screen.getByTestId("preflight-report-view")).toBeInTheDocument();
    expect(screen.getByTestId("preflight-passed-badge")).toHaveTextContent(
      "PRE-FLIGHT VALIDATION PASSED"
    );
    expect(screen.getByText("Normalized RRF: 94.2%")).toBeInTheDocument();
    expect(screen.getByText("YES (100%)")).toBeInTheDocument();
  });

  it("auto-switches to Quant Strategy tab when contextPack has 0 code_slices and a QuantStrategyBrief", () => {
    const { rerender } = render(<CitationInspector contextPack={MOCK_CONTEXT_PACK} />);
    expect(screen.getByTestId("code-slices-view")).toBeInTheDocument();

    const quantOnlyPack = {
      ...MOCK_CONTEXT_PACK,
      domain_id: "quant_research_scout",
      user_task: "best way to implement momentum strategy in indian stock market?",
      code_slices: [],
    };
    rerender(<CitationInspector contextPack={quantOnlyPack} />);

    expect(screen.getByTestId("quant-strategy-view")).toBeInTheDocument();
    expect(
      screen.getByText("Intraday Momentum Volatility Breakout")
    ).toBeInTheDocument();
  });

  it("renders domain-agnostic Domain Knowledge labels and metrics for non-quant custom domains", () => {
    const customDomainPack = {
      ...MOCK_CONTEXT_PACK,
      domain_id: "crispr_biology_scout",
      user_task: "How does Cas9 guide RNA mismatch affect off-target cleavage kinetics?",
      code_slices: [],
      quant_strategy_spec: {
        ...MOCK_CONTEXT_PACK.quant_strategy_spec!,
        strategy_name: "CRISPR-Cas9 Guide RNA Specificity",
        reported_sharpe_ratio: null,
        reported_max_drawdown_pct: null,
        transaction_cost_bps_assumed: null,
        entry_exit_rules: ["Cas9 endonuclease cleavage rate decreases exponentially."],
        market_specific_caveats: ["Experimental assays require 10 mM Mg2+ cofactor concentration."],
        known_failure_regimes: ["Severe off-target cleavage failure occurs at positions 18-20."],
      },
    };

    render(<CitationInspector contextPack={customDomainPack} />);

    expect(screen.getByTestId("tab-quant-strategy")).toHaveTextContent(
      "Domain Knowledge (Active)"
    );
    expect(screen.getByText("Citation Verification")).toBeInTheDocument();
    expect(screen.getByText("NFKC 100%")).toBeInTheDocument();
    expect(screen.getByText("N/A (Non-Fin)")).toBeInTheDocument();
    expect(
      screen.getByText("Core Empirical Findings & Domain Rules:")
    ).toBeInTheDocument();
    expect(
      screen.getByText("Domain Constraints & Operational Caveats:")
    ).toBeInTheDocument();
    expect(
      screen.getByText("Known Risks & Failure Modes:")
    ).toBeInTheDocument();
  });
});
