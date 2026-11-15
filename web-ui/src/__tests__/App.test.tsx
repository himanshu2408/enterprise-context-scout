import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { clearCachedAuthSession } from "../api/client";
import {
  buildTestSession,
  installMockApiFetch,
  TEST_CODE_PACK,
  TEST_ENGINEER_PRINCIPAL,
  TEST_JIRA_CITATION,
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

describe("App Integration", () => {
  it("loads dynamic health & domains, mints Bearer token on persona switch, and gates Engineer patch CTA", async () => {
    const fetchMock = installMockApiFetch();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("hardware-profile-chip")).toHaveTextContent(
        "Apple Silicon MLX"
      );
      expect(screen.getByTestId("hardware-profile-chip")).toHaveTextContent(
        "2/2 Adapters"
      );
      expect(screen.getByTestId("health-db-badge")).toHaveTextContent(
        "PG17 RLS Connected"
      );
      expect(screen.getByTestId("header-auth-badge")).toHaveTextContent(
        "Bearer: engineer"
      );
    });

    // Run SSE workflow as Engineer
    fireEvent.click(screen.getByTestId("run-scout-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("answer-first-hero")).toBeInTheDocument();
      expect(screen.getByTestId("open-diff-modal-btn")).toBeInTheDocument();
    });

    // Switch persona to product_manager -> mints new session & hides write CTA immediately
    fireEvent.click(screen.getByTestId("role-btn-product_manager"));

    await waitFor(() => {
      expect(screen.getByTestId("header-auth-badge")).toHaveTextContent(
        "Bearer: product_manager"
      );
      expect(
        screen.queryByTestId("open-diff-modal-btn")
      ).not.toBeInTheDocument();
      expect(screen.getByTestId("handoff-readonly-notice")).toHaveTextContent(
        "Read-Only Role (product_manager)"
      );
    });

    const authCalls = fetchMock.mock.calls.filter(([url]) =>
      String(url).endsWith("/api/v1/auth/session")
    );
    expect(authCalls.length).toBeGreaterThanOrEqual(2);
  });

  it("fails closed with explicit errors and zero fake hardware/domain fallbacks when backend returns 401/500", async () => {
    installMockApiFetch({
      authStatus: 401,
      authDetail: "Invalid dev auth configuration",
      healthStatus: 500,
      domainsStatus: 500,
      roiStatus: 500,
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("auth-error-banner")).toHaveTextContent(
        "Invalid dev auth configuration"
      );
      expect(screen.getByTestId("domains-error-banner")).toBeInTheDocument();
      expect(screen.getByTestId("roi-error-banner")).toBeInTheDocument();
    });

    expect(
      screen.queryByTestId("hardware-profile-chip")
    ).not.toBeInTheDocument();
    expect(document.body.textContent ?? "").not.toContain("Apple M5 Pro");
    expect(
      screen.queryByTestId("domain-card-trading_code_scout")
    ).not.toBeInTheDocument();
  });

  it("retries failed auth session and domains/health fetches when user clicks retry controls", async () => {
    const baseFetch = installMockApiFetch();
    let failInitial = true;

    const flakyFetch = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = typeof input === "string" ? input : input.toString();
        if (
          failInitial &&
          (url.endsWith("/api/v1/auth/session") ||
            url.endsWith("/api/v1/domains") ||
            url.endsWith("/api/v1/health"))
        ) {
          return new Response(
            JSON.stringify({ detail: "Temporary gateway timeout" }),
            { status: 503, headers: { "Content-Type": "application/json" } }
          );
        }
        return baseFetch(input, init);
      }
    );
    vi.stubGlobal("fetch", flakyFetch);

    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("auth-error-banner")).toHaveTextContent(
        "Temporary gateway timeout"
      );
      expect(screen.getByTestId("domains-error-banner")).toHaveTextContent(
        "Temporary gateway timeout"
      );
    });

    failInitial = false;
    fireEvent.click(screen.getByTestId("retry-auth-btn"));
    fireEvent.click(screen.getByTestId("domains-retry-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("header-auth-badge")).toHaveTextContent(
        "Bearer: engineer"
      );
      expect(screen.getByTestId("hardware-profile-chip")).toHaveTextContent(
        "Apple Silicon MLX"
      );
      expect(
        screen.getByTestId("domain-card-trading_code_scout")
      ).toBeInTheDocument();
    });

    expect(buildTestSession(TEST_ENGINEER_PRINCIPAL).token_type).toBe("Bearer");
  });

  it("filters domains, console badge, and inspector evidence via Source Filter bar", async () => {
    installMockApiFetch({
      sseEvents: [
        {
          event: "context_pack_ready",
          data: {
            status: "ready",
            pack: {
              ...TEST_CODE_PACK,
              supporting_doc_citations: [
                ...TEST_CODE_PACK.supporting_doc_citations,
                TEST_JIRA_CITATION,
              ],
            },
            token_savings: TEST_TOKEN_SAVINGS,
            timestamp: "2026-11-14T10:00:02Z",
          },
        },
      ],
    });

    render(<App />);

    await waitFor(() => {
      expect(
        screen.getByTestId("domain-card-trading_code_scout")
      ).toBeInTheDocument();
    });

    // Select a code domain, then switch source filter to wiki -> auto-switches domain to quant_research_scout
    fireEvent.click(screen.getByTestId("domain-card-trading_code_scout"));
    expect(
      screen.getByTestId("domain-card-trading_code_scout")
    ).toHaveClass("domain-row-active");

    fireEvent.click(screen.getByTestId("source-filter-wiki"));
    expect(screen.getByTestId("console-source-filter-pill")).toHaveTextContent(
      "Confluence & Docs"
    );
    expect(
      screen.queryByTestId("domain-card-trading_code_scout")
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("domain-card-quant_research_scout")
    ).toHaveClass("domain-row-active");

    // Switch back to all and run a query with both code + wiki + jira citations
    fireEvent.click(screen.getByTestId("source-filter-all"));
    fireEvent.click(screen.getByTestId("run-scout-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("inline-cite-code-0")).toBeInTheDocument();
      expect(
        screen.getByTestId("inline-cite-doc-chunk_jira_eng412")
      ).toBeInTheDocument();
    });

    // Jira filter hides code slices & wiki citations, leaving only Jira ticket citation
    fireEvent.click(screen.getByTestId("source-filter-jira"));
    expect(screen.queryByTestId("inline-cite-code-0")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("inline-cite-doc-chunk_hawkes_01")
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("inline-cite-doc-chunk_jira_eng412")
    ).toBeInTheDocument();
  });

  it("renders collapsed Cost & Telemetry Strip by default and expands Frontier Rates on toggle", async () => {
    installMockApiFetch();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("cumulative-roi-card")).toHaveTextContent(
        "249 queries"
      );
    });

    expect(
      screen.queryByTestId("frontier-rates-drawer")
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("toggle-telemetry-strip-btn"));
    expect(screen.getByTestId("frontier-rates-drawer")).toBeInTheDocument();
    expect(screen.getByTestId("frontier-rates-drawer")).toHaveTextContent(
      "Gemini 4 Argon"
    );
  });

  it("wipes prior ContextPack and closes diff modal when allowed_scopes shrink (SEC-01 hygiene)", async () => {
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
      expect(screen.getByTestId("frontier-savings-ratio")).toHaveTextContent(
        "91.0%"
      );
    });

    // Open diff modal as engineer
    fireEvent.click(screen.getByTestId("open-diff-modal-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("gemini-diff-modal")).toBeInTheDocument();
    });

    // Shrink scope by removing public_eng -> must wipe pack & savings immediately
    fireEvent.click(screen.getByTestId("scope-chip-public_eng"));

    await waitFor(() => {
      expect(screen.queryByTestId("answer-first-hero")).not.toBeInTheDocument();
      expect(screen.getByTestId("empty-query-savings")).toBeInTheDocument();
      expect(screen.queryByTestId("gemini-diff-modal")).not.toBeInTheDocument();
      expect(screen.queryByTestId("shield-gemma-safe-badge")).not.toBeInTheDocument();
      expect(screen.queryByTestId("check-stage-1")).not.toBeInTheDocument();
    });
  });
});
