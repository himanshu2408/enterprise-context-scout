import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CitationInspector } from "../components/CitationInspector";
import {
  TEST_CODE_PACK,
  TEST_DIAGNOSTIC,
  TEST_JIRA_CITATION,
  TEST_QUANT_PACK,
  TEST_STALE_DOC_CITATION,
  TEST_TOKEN_SAVINGS,
} from "./fixtures";

afterEach(() => {
  cleanup();
});

describe("CitationInspector (Answer-First Workspace)", () => {
  it("renders empty state when contextPack is null and abstention message when diagnostic is set", () => {
    const { rerender } = render(
      <CitationInspector
        contextPack={null}
        quantSpec={null}
        tokenSavings={null}
        principalRole="engineer"
        onOpenDiffModal={vi.fn()}
      />
    );

    expect(screen.getByTestId("citation-inspector")).toHaveTextContent(
      "Grounded Answer & Context Workbench"
    );

    rerender(
      <CitationInspector
        contextPack={null}
        quantSpec={null}
        tokenSavings={null}
        principalRole="engineer"
        onOpenDiffModal={vi.fn()}
        abstainedDiagnostic={TEST_DIAGNOSTIC}
      />
    );

    expect(screen.getByTestId("citation-inspector")).toHaveTextContent(
      "local scout abstained safely"
    );
  });

  it("renders Answer-First Hero Brief, formatted inline [repo@sha:file:Lstart-Lend] & doc chips that jump and highlight", () => {
    const onOpenModal = vi.fn();
    render(
      <CitationInspector
        contextPack={TEST_CODE_PACK}
        quantSpec={null}
        tokenSavings={TEST_TOKEN_SAVINGS}
        principalRole="engineer"
        onOpenDiffModal={onOpenModal}
      />
    );

    const hero = screen.getByTestId("answer-first-hero");
    expect(hero).toHaveTextContent(
      "Almgren-Chriss Optimal Execution Schedule & Risk Bounds"
    );
    expect(hero).toHaveTextContent(
      "AlmgrenChrissExecutionOptimizer splits parent orders"
    );
    expect(screen.getByTestId("preflight-passed-badge")).toHaveTextContent(
      "100% Grounded & Verified"
    );

    // Inline code citation chip matches [repo@sha:file:Lstart-Lend]
    expect(screen.getByTestId("inline-cite-code-0")).toHaveTextContent(
      "[quant-exec-core@abc1234:execution/optimizer.py:L14-L32]"
    );
    expect(
      screen.getByTestId("inline-cite-doc-chunk_hawkes_01")
    ).toHaveTextContent(
      "[1] The role of volume in order book dynamics: a multivariate Hawkes process analysis: 3.2 Excitation Kernel Stability"
    );

    // Click inline doc citation chip -> switches to Document Sections tab and highlights card
    fireEvent.click(screen.getByTestId("inline-cite-doc-chunk_hawkes_01"));
    expect(screen.getByTestId("quant-strategy-view")).toBeInTheDocument();
    expect(screen.getByTestId("citation-card-chunk_hawkes_01")).toHaveClass(
      "evidence-item-highlighted"
    );

    // Click inline code slice chip #1 -> switches back to Code & Dependencies tab and highlights slice 1
    fireEvent.click(screen.getByTestId("inline-cite-code-1"));
    expect(screen.getByTestId("code-slices-view")).toBeInTheDocument();
    expect(screen.getByTestId("code-slice-card-1")).toHaveClass(
      "evidence-item-highlighted"
    );
    expect(screen.getByTestId("ast-closure-badge")).toBeInTheDocument();

    // Engineer role has patch handoff button enabled
    fireEvent.click(screen.getByTestId("open-diff-modal-btn"));
    expect(onOpenModal).toHaveBeenCalledTimes(1);
  });

  it("renders prominent doc_vs_code_conflicts_detected banner, Quant brief, and hides patch CTA for non-engineers", () => {
    render(
      <CitationInspector
        contextPack={{
          ...TEST_QUANT_PACK,
          code_slices: TEST_CODE_PACK.code_slices,
        }}
        quantSpec={TEST_QUANT_PACK.quant_strategy_spec}
        tokenSavings={TEST_TOKEN_SAVINGS}
        principalRole="product_manager"
        onOpenDiffModal={vi.fn()}
      />
    );

    expect(screen.getByTestId("conflicts-banner")).toHaveTextContent(
      "Code uses static 2-bps impact"
    );
    expect(screen.queryByTestId("open-diff-modal-btn")).not.toBeInTheDocument();
    expect(screen.getByTestId("handoff-readonly-notice")).toHaveTextContent(
      "Read-Only Role (product_manager)"
    );

    fireEvent.click(screen.getByTestId("tab-quant-strategy"));
    expect(screen.getByTestId("quant-executive-answer")).toBeInTheDocument();
    expect(screen.getByTestId("entry-exit-rules-list")).toHaveTextContent(
      "2.5x baseline"
    );
    expect(screen.getByTestId("latex-mathematical-signal")).toHaveTextContent(
      "\\lambda_t"
    );
    expect(screen.getByTestId("friction-verdict-badge")).toHaveTextContent(
      "UNVERIFIED_FRICTION_WARNING"
    );
    expect(screen.getByTestId("market-regimes-list")).toHaveTextContent(
      "DE_XETRA, US_EQUITY"
    );
  });

  it("renders Outdated / Stale Document warning banner and stale-citation-badge for is_stale citations", () => {
    render(
      <CitationInspector
        contextPack={{
          ...TEST_CODE_PACK,
          supporting_doc_citations: [
            TEST_CODE_PACK.supporting_doc_citations[0],
            TEST_STALE_DOC_CITATION,
          ],
        }}
        quantSpec={null}
        tokenSavings={TEST_TOKEN_SAVINGS}
        principalRole="engineer"
        onOpenDiffModal={vi.fn()}
      />
    );

    const staleBanner = screen.getByTestId("stale-docs-warning-banner");
    expect(staleBanner).toHaveTextContent("Outdated / Stale Document Warning (1)");
    expect(staleBanner).toHaveTextContent("Legacy Execution Timeout Runbook");
    expect(staleBanner).toHaveTextContent(
      "Superseded by automated pg_pool failover ADR-2026-11."
    );

    fireEvent.click(screen.getByTestId("tab-quant-strategy"));
    const staleCard = screen.getByTestId(
      `citation-card-${TEST_STALE_DOC_CITATION.chunk_id}`
    );
    expect(staleCard).toHaveTextContent("Outdated / Stale");
    expect(screen.getByTestId("stale-citation-badge")).toHaveTextContent(
      "Outdated / Stale"
    );
  });

  it("auto-switches to Quant Strategy view when code_slices is empty and QuantStrategyBrief is present", () => {
    const { rerender } = render(
      <CitationInspector
        contextPack={TEST_CODE_PACK}
        quantSpec={null}
        tokenSavings={TEST_TOKEN_SAVINGS}
        principalRole="engineer"
        onOpenDiffModal={vi.fn()}
      />
    );
    expect(screen.getByTestId("code-slices-view")).toBeInTheDocument();

    rerender(
      <CitationInspector
        contextPack={TEST_QUANT_PACK}
        quantSpec={TEST_QUANT_PACK.quant_strategy_spec}
        tokenSavings={TEST_TOKEN_SAVINGS}
        principalRole="quant_researcher"
        onOpenDiffModal={vi.fn()}
      />
    );

    expect(screen.getByTestId("quant-strategy-view")).toBeInTheDocument();
    expect(screen.getByTestId("quant-executive-answer")).toHaveTextContent(
      "Multivariate Hawkes Order-Book Imbalance"
    );
  });

  it("renders domain-neutral Citations & Docs labeling without fake quant metrics when quantSpec is null", () => {
    render(
      <CitationInspector
        contextPack={{
          ...TEST_CODE_PACK,
          domain_id: "crispr_biology_scout",
          code_slices: [],
          quant_strategy_spec: null,
          domain_knowledge_brief: {
            brief_title: "CRISPR-Cas9 Guide RNA Specificity",
            domain_id: "crispr_biology_scout",
            executive_summary:
              "Cas9 cleavage specificity drops at PAM-distal mismatches.",
            formal_expression_or_rule: "k_cleave = k_0 * exp(-Delta_G / RT)",
            core_findings_and_rules: [
              "Seed region mismatches completely abolish cleavage.",
            ],
            domain_constraints_and_caveats: [
              "Assays require 10 mM Mg2+ cofactor.",
            ],
            known_risks_and_failure_modes: [
              "Off-target cleavage at positions 18-20.",
            ],
            structured_metadata: { truth_source: "document" },
            verified_citations: TEST_CODE_PACK.supporting_doc_citations,
          },
        }}
        quantSpec={null}
        tokenSavings={TEST_TOKEN_SAVINGS}
        principalRole="engineer"
        onOpenDiffModal={vi.fn()}
      />
    );

    expect(screen.getByTestId("tab-quant-strategy")).toHaveTextContent(
      "Document Sections (1)"
    );
    expect(screen.getByTestId("answer-first-hero")).toHaveTextContent(
      "k_cleave = k_0 * exp(-Delta_G / RT)"
    );
    expect(screen.getByTestId("answer-first-hero")).toHaveTextContent(
      "Constraints & Known Risks"
    );
    expect(
      screen.queryByTestId("quant-executive-answer")
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("friction-verdict-badge")
    ).not.toBeInTheDocument();
    // Engineer still sees read-only notice when code_slices is empty
    expect(screen.getByTestId("handoff-readonly-notice")).toHaveTextContent(
      "No mutable code slices in this pack"
    );
  });

  it("filters code slices and document citations according to sourceFilter", () => {
    const packWithMixedCitations = {
      ...TEST_CODE_PACK,
      supporting_doc_citations: [
        TEST_CODE_PACK.supporting_doc_citations[0],
        TEST_JIRA_CITATION,
      ],
    };

    const { rerender } = render(
      <CitationInspector
        contextPack={packWithMixedCitations}
        quantSpec={null}
        tokenSavings={TEST_TOKEN_SAVINGS}
        principalRole="engineer"
        sourceFilter="jira"
        onOpenDiffModal={vi.fn()}
      />
    );

    expect(screen.queryByTestId("inline-cite-code-0")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("inline-cite-doc-chunk_hawkes_01")
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("inline-cite-doc-chunk_jira_eng412")
    ).toBeInTheDocument();
    expect(screen.getByTestId("tab-code-slices")).toHaveTextContent(
      "Code & Dependencies (0)"
    );

    rerender(
      <CitationInspector
        contextPack={packWithMixedCitations}
        quantSpec={null}
        tokenSavings={TEST_TOKEN_SAVINGS}
        principalRole="engineer"
        sourceFilter="code"
        onOpenDiffModal={vi.fn()}
      />
    );

    expect(screen.getByTestId("inline-cite-code-0")).toBeInTheDocument();
    expect(
      screen.queryByTestId("inline-cite-doc-chunk_jira_eng412")
    ).not.toBeInTheDocument();
  });

  it("switches to Pre-Flight Gate and Telemetry & Pack Tokens tabs and renders verified metrics", () => {
    render(
      <CitationInspector
        contextPack={TEST_CODE_PACK}
        quantSpec={null}
        tokenSavings={TEST_TOKEN_SAVINGS}
        principalRole="engineer"
        onOpenDiffModal={vi.fn()}
      />
    );

    fireEvent.click(screen.getByTestId("tab-preflight-report"));
    const preflightView = screen.getByTestId("preflight-report-view");
    expect(preflightView).toHaveTextContent("PASSED");
    expect(preflightView).toHaveTextContent("0.890");
    expect(preflightView).toHaveTextContent("Verified (Zero Unmatched)");

    fireEvent.click(screen.getByTestId("tab-telemetry-roi"));
    const telemetryView = screen.getByTestId("telemetry-roi-view");
    expect(telemetryView).toHaveTextContent("42,000");
    expect(telemetryView).toHaveTextContent("3,800");
    expect(telemetryView).toHaveTextContent("38,200");
    expect(telemetryView).toHaveTextContent("$0.1528");
  });

  it("renders unmistakable amber Partial Coverage callout listing missing_or_unverified_aspects and per-facet pills", () => {
    render(
      <CitationInspector
        contextPack={{
          ...TEST_CODE_PACK,
          domain_id: "data_engineering_scout",
          code_slices: [],
          query_facets: [
            {
              facet_id: "f1",
              facet_query: "Apache Iceberg CoW vs MoR",
              status: "grounded",
              matched_sources: ["iceberg_spec.md"],
              confidence: 0.92,
              explanation: "Grounded in 2 chunks.",
            },
            {
              facet_id: "f2",
              facet_query: "Spark AQE skew joins",
              status: "grounded",
              matched_sources: ["spark_aqe.md"],
              confidence: 0.88,
              explanation: "Grounded in 1 chunk.",
            },
            {
              facet_id: "f3",
              facet_query: "Oracle Exadata Smart Scan HCC offload",
              status: "not_found",
              matched_sources: [],
              confidence: 0.0,
              explanation: "No chunks above RRF floor 0.55.",
            },
          ],
          missing_or_unverified_aspects: [
            "Oracle Exadata Smart Scan HCC offload (no grounded documents above RRF floor)",
          ],
          completeness_ratio: 0.67,
        }}
        quantSpec={null}
        tokenSavings={TEST_TOKEN_SAVINGS}
        principalRole="engineer"
        onOpenDiffModal={vi.fn()}
      />
    );

    const callout = screen.getByTestId("partial-coverage-callout");
    expect(callout).toHaveTextContent("Partial corpus coverage (2 of 3 facets grounded)");
    expect(callout).toHaveTextContent("Not found in indexed corpus:");
    expect(callout).toHaveTextContent("Oracle Exadata Smart Scan HCC offload");
    expect(screen.getByTestId("preflight-passed-badge")).toHaveTextContent(
      "Partial Coverage (2/3 facets)"
    );
    expect(screen.getByTestId("facet-pill-f1")).toHaveTextContent("Grounded");
    expect(screen.getByTestId("facet-pill-f3")).toHaveTextContent("Not found");
  });

  it("renders grouped Code & AST Dependency inspector and full un-truncated full_section_content with highlighted quote", () => {
    const richPack = {
      ...TEST_CODE_PACK,
      code_slices: [
        TEST_CODE_PACK.code_slices[0], // modify_target
        TEST_CODE_PACK.code_slices[1], // imported_type
        {
          ...TEST_CODE_PACK.code_slices[0],
          file_path: "execution/router.py",
          qualified_symbol: "OrderRouter.dispatch",
          modification_intent: "direct_caller" as const,
          auto_expanded_by_ast: true,
        },
        {
          ...TEST_CODE_PACK.code_slices[0],
          file_path: "tests/test_optimizer.py",
          qualified_symbol: "test_almgren_chriss_slippage",
          modification_intent: "test_fixture" as const,
          auto_expanded_by_ast: true,
        },
      ],
      supporting_doc_citations: [
        {
          ...TEST_CODE_PACK.supporting_doc_citations[0],
          full_section_content:
            "Section intro paragraph. " +
            TEST_CODE_PACK.supporting_doc_citations[0].verbatim_quote +
            " Trailing analytical derivation continues in full without truncation.",
        },
      ],
    };

    render(
      <CitationInspector
        contextPack={richPack}
        quantSpec={null}
        tokenSavings={TEST_TOKEN_SAVINGS}
        principalRole="engineer"
        onOpenDiffModal={vi.fn()}
      />
    );

    expect(screen.getByTestId("slice-group-modify_target")).toHaveTextContent(
      "Target Implementation Slices (1)"
    );
    expect(screen.getByTestId("slice-group-imported_type")).toHaveTextContent(
      "Types & Schemas (AST Closure) (1)"
    );
    expect(screen.getByTestId("slice-group-direct_caller")).toHaveTextContent(
      "Direct Callers (AST Closure) (1)"
    );
    expect(screen.getByTestId("slice-group-test_fixture")).toHaveTextContent(
      "Unit Tests (AST Closure) (1)"
    );

    // Switch to Document Sections tab and expand full section context
    fireEvent.click(screen.getByTestId("tab-quant-strategy"));
    fireEvent.click(screen.getByTestId("full-section-toggle-chunk_hawkes_01"));
    const fullSection = screen.getByTestId(
      "full-section-content-chunk_hawkes_01"
    );
    expect(fullSection).toHaveTextContent("Section intro paragraph.");
    expect(fullSection).toHaveTextContent(
      "Trailing analytical derivation continues in full without truncation."
    );
  });
});
