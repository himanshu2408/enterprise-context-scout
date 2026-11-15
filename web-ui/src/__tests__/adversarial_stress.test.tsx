import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { applyHandoff, clearCachedAuthSession } from "../api/client";
import { CitationInspector } from "../components/CitationInspector";
import { ScoutConsole } from "../components/ScoutConsole";
import {
  installMockApiFetch,
  TEST_CODE_PACK,
  TEST_ENGINEER_PRINCIPAL,
  TEST_QUANT_PACK,
  TEST_TOKEN_SAVINGS,
} from "./fixtures";

beforeEach(() => {
  clearCachedAuthSession();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("Adversarial & Regression Security Tests", () => {
  it("escapes XSS script and onerror payloads in code slices, citations, conflicts, and diagnostics", () => {
    const xssPayload = `<img src=x onerror="window.__xss=true" /><script>alert(1)</script>`;

    render(
      <>
        <CitationInspector
          contextPack={{
            ...TEST_CODE_PACK,
            doc_vs_code_conflicts_detected: [xssPayload],
            code_slices: [
              {
                ...TEST_CODE_PACK.code_slices[0],
                verbatim_code_with_lines: xssPayload,
              },
            ],
            supporting_doc_citations: [
              {
                ...TEST_CODE_PACK.supporting_doc_citations[0],
                verbatim_quote: xssPayload,
                stale_reason: xssPayload,
              },
            ],
          }}
          quantSpec={null}
          tokenSavings={TEST_TOKEN_SAVINGS}
          principalRole="engineer"
          onOpenDiffModal={vi.fn()}
        />
      </>
    );

    const codeBlock = screen.getByTestId("code-block-0");
    expect(codeBlock.textContent).toContain("<script>alert(1)</script>");
    expect(codeBlock.querySelector("script")).toBeNull();
    expect(codeBlock.querySelector("img")).toBeNull();

    const conflictsBanner = screen.getByTestId("conflicts-banner");
    expect(conflictsBanner.textContent).toContain("<script>alert(1)</script>");
    expect(conflictsBanner.querySelector("script")).toBeNull();
    expect(conflictsBanner.querySelector("img")).toBeNull();

    const staleBanner = screen.getByTestId("stale-docs-warning-banner");
    expect(staleBanner.querySelector("script")).toBeNull();
    expect(staleBanner.querySelector("img")).toBeNull();

    fireEvent.click(screen.getByTestId("tab-quant-strategy"));
    const citeCard = screen.getByTestId("citation-card-chunk_hawkes_01");
    expect(citeCard.textContent).toContain("<script>alert(1)</script>");
    expect(citeCard.querySelector("script")).toBeNull();
    expect(citeCard.querySelector("img")).toBeNull();
  });

  it("clears cached Bearer token when 401 occurs during stream and surfaces error without fallback", async () => {
    const fetchMock = installMockApiFetch({
      sseStatus: 401,
      sseDetail: "Expired or invalid Bearer token",
    });

    render(
      <ScoutConsole
        selectedDomain="trading_code_scout"
        principal={TEST_ENGINEER_PRINCIPAL}
        onContextPackReady={vi.fn()}
      />
    );

    fireEvent.click(screen.getByTestId("run-scout-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("api-error-banner")).toHaveTextContent(
        "HTTP 401: Expired or invalid Bearer token"
      );
    });

    // Second run must re-mint a fresh token because 401 cleared cache
    fireEvent.click(screen.getByTestId("run-scout-btn"));
    await waitFor(() => {
      const authCalls = fetchMock.mock.calls.filter(([url]) =>
        String(url).endsWith("/api/v1/auth/session")
      );
      expect(authCalls.length).toBe(2);
    });
  });

  it("clears cached Bearer token when 401 occurs during REST query and fails closed", async () => {
    const fetchMock = installMockApiFetch({
      queryStatus: 401,
      queryDetail: "Invalid HMAC signature",
    });

    render(
      <ScoutConsole
        selectedDomain="trading_code_scout"
        principal={TEST_ENGINEER_PRINCIPAL}
        onContextPackReady={vi.fn()}
      />
    );

    fireEvent.click(screen.getByTestId("mode-rest-btn"));
    fireEvent.click(screen.getByTestId("run-scout-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("api-error-banner")).toHaveTextContent(
        "Invalid HMAC signature"
      );
    });

    fireEvent.click(screen.getByTestId("run-scout-btn"));
    await waitFor(() => {
      const authCalls = fetchMock.mock.calls.filter(([url]) =>
        String(url).endsWith("/api/v1/auth/session")
      );
      expect(authCalls.length).toBe(2);
    });
  });

  it("wipes prior ContextPack on unscoped_probe scope reduction (SEC-01 client hygiene)", async () => {
    installMockApiFetch();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("header-auth-badge")).toHaveTextContent(
        "Bearer: engineer"
      );
    });

    fireEvent.click(screen.getByTestId("run-scout-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("answer-first-hero")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("unscoped-probe-btn"));

    await waitFor(() => {
      expect(screen.queryByTestId("answer-first-hero")).not.toBeInTheDocument();
      expect(screen.getByTestId("active-scopes-list")).toHaveTextContent(
        "unscoped_probe"
      );
    });
  });

  it("renders zero mock fallback data on complete backend network outage", async () => {
    installMockApiFetch({
      networkFailUrls: [
        "/api/v1/auth/session",
        "/api/v1/health",
        "/api/v1/domains",
        "/api/v1/telemetry/roi",
        "/api/v1/scout/stream",
      ],
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("auth-error-banner")).toHaveTextContent(
        "Authentication service unreachable"
      );
      expect(screen.getByTestId("domains-error-banner")).toHaveTextContent(
        "Domains endpoint unreachable"
      );
      expect(screen.getByTestId("roi-error-banner")).toHaveTextContent(
        "Telemetry endpoint unreachable"
      );
    });

    expect(
      screen.queryByTestId("hardware-profile-chip")
    ).not.toBeInTheDocument();
    expect(document.body.textContent ?? "").not.toContain("Apple M5 Pro");

    fireEvent.click(screen.getByTestId("run-scout-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("api-error-banner")).toHaveTextContent(
        "Authentication service unreachable"
      );
    });
    expect(screen.queryByTestId("answer-first-hero")).not.toBeInTheDocument();
  });

  it("handles CitationInspector resilience with empty code_slices, null quant metrics, and missing optional fields", () => {
    render(
      <CitationInspector
        contextPack={{
          ...TEST_QUANT_PACK,
          code_slices: [],
          supporting_doc_citations: [
            {
              chunk_id: "minimal_cite",
              doc_id: "minimal_doc",
              doc_title: "Minimal Citation",
              source_uri: "https://example.org/doc",
              section_heading: "Summary",
              verbatim_quote: "Exact minimal quote.",
            },
          ],
          domain_knowledge_brief: null,
        }}
        quantSpec={{
          ...TEST_QUANT_PACK.quant_strategy_spec!,
          reported_sharpe_ratio: null,
          reported_max_drawdown_pct: null,
          transaction_cost_bps_assumed: null,
          verified_citations: [],
        }}
        tokenSavings={null}
        principalRole="compliance_officer"
        onOpenDiffModal={vi.fn()}
      />
    );

    expect(screen.getByTestId("quant-strategy-view")).toBeInTheDocument();
    expect(screen.getByTestId("quant-executive-answer")).toHaveTextContent(
      "N/A"
    );
    expect(
      screen.getByTestId("inline-cite-doc-minimal_cite")
    ).toHaveTextContent("[1] Minimal Citation: Summary (v1.0)");
    expect(screen.queryByTestId("open-diff-modal-btn")).not.toBeInTheDocument();
  });

  it("rejects unconfirmed applyHandoff client-side without touching /api/v1/handoff/apply (SEC-04)", async () => {
    const fetchMock = installMockApiFetch();

    await expect(
      applyHandoff(
        {
          confirmation_token: "confirm_tok_test_001",
          confirmed: false,
        },
        TEST_ENGINEER_PRINCIPAL
      )
    ).rejects.toThrow("Explicit human confirmation is required");

    const applyCalls = fetchMock.mock.calls.filter(([url]) =>
      String(url).endsWith("/api/v1/handoff/apply")
    );
    expect(applyCalls).toHaveLength(0);
  });
});
