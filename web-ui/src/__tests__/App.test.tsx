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
  TEST_WORKSPACE_CONFIG,
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
      expect(
        screen.getByTestId("domain-card-code_architecture_scout")
      ).toBeInTheDocument();
      expect(screen.getByTestId("run-scout-btn")).toBeEnabled();
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
      screen.queryByTestId("domain-card-code_architecture_scout")
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
            url.endsWith("/api/v1/workspace-config") ||
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
      expect(screen.getByTestId("domains-error-banner")).toBeInTheDocument();
    });

    failInitial = false;
    fireEvent.click(screen.getByTestId("retry-auth-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("header-auth-badge")).toHaveTextContent(
        "Bearer: engineer"
      );
      expect(screen.getByTestId("hardware-profile-chip")).toHaveTextContent(
        "Apple Silicon MLX"
      );
      expect(
        screen.getByTestId("domain-card-code_architecture_scout")
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
        screen.getByTestId("domain-card-code_architecture_scout")
      ).toBeInTheDocument();
    });

    // Select a code domain, then switch source filter to wiki -> auto-switches domain to enterprise_docs_scout
    fireEvent.click(screen.getByTestId("domain-card-code_architecture_scout"));
    expect(
      screen.getByTestId("domain-card-code_architecture_scout")
    ).toHaveClass("domain-row-active");

    fireEvent.click(screen.getByTestId("source-filter-wiki"));
    expect(screen.getByTestId("console-source-filter-pill")).toHaveTextContent(
      "Confluence & Docs"
    );
    expect(
      screen.queryByTestId("domain-card-code_architecture_scout")
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("domain-card-enterprise_docs_scout")
    ).toHaveClass("domain-row-active");

    // Switch back to all and run a query with both code + wiki + jira citations
    fireEvent.click(screen.getByTestId("source-filter-all"));
    await waitFor(() => {
      expect(screen.getByTestId("run-scout-btn")).toBeEnabled();
    });
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
      screen.queryByTestId("inline-cite-doc-chunk_rls_01")
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
      "Gemini 4 Carbon"
    );
  });

  it("wipes prior ContextPack and closes diff modal when allowed_scopes shrink (SEC-01 hygiene)", async () => {
    installMockApiFetch();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("header-auth-badge")).toHaveTextContent(
        "Bearer: engineer"
      );
      expect(
        screen.getByTestId("domain-card-code_architecture_scout")
      ).toBeInTheDocument();
      expect(screen.getByTestId("run-scout-btn")).toBeEnabled();
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

  it("renders a custom 5th domain (biotech_scout) dynamically from /api/v1/workspace-config with zero hardcoded config", async () => {
    installMockApiFetch({
      workspaceConfigResponse: {
        domains: [
          {
            domain_id: "biotech_scout",
            display_name: "Biotech & Clinical Trials",
            description: "CRISPR & mRNA clinical assay protocols.",
            modalities: ["document"],
            sample_queries: [
              "Summarize Phase III CRISPR off-target assay thresholds.",
            ],
            lora_enabled: false,
            default_scopes: ["public_bio"],
            doc_chunks_count: 1240,
            documents_count: 18,
            code_files_count: 0,
            ast_symbols_count: 0,
            manifest: {
              domain_id: "biotech_scout",
              display_name: "Biotech & Clinical Trials",
              base_model_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
              lora_adapter_id: "biotech_base",
              lora_adapter_path: "",
              authority_rules: { primary_truth_source: "document" },
              output_contract: "DomainKnowledgeBrief",
            },
            adapter_verification: {
              domain_id: "biotech_scout",
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
        ],
        federated_option: {
          domain_id: "federated_all",
          display_name: "All Authorized Domains (Auto-Route)",
          description: "Dynamic multi-domain routing.",
        },
        role_presets: [
          {
            role: "engineer",
            label: "Engineer",
            default_scopes: ["public_bio"],
          },
        ],
        pipeline_stages: [
          {
            stage_id: 1,
            number: "01",
            title: "Auth, Guardrail & Query Plan",
            summary: "RBAC & QueryPlan",
            what_happens: "Binds PrincipalContext",
            gate_condition: "Fail closed",
          },
          {
            stage_id: 2,
            number: "02",
            title: "Retrieval",
            summary: "Hybrid RRF",
            what_happens: "Probes PG17",
            gate_condition: "Relevance floor",
          },
          {
            stage_id: 3,
            number: "03",
            title: "Synthesis",
            summary: "Local Gemma-4",
            what_happens: "Full chunks",
            gate_condition: "Sufficiency",
          },
          {
            stage_id: 4,
            number: "04",
            title: "Pre-Flight Gate",
            summary: "$0 Gate",
            what_happens: "Checks ledger",
            gate_condition: "Abstain on miss",
          },
        ],
        source_filters: ["all", "code", "wiki", "jira"],
        hardware_profile: {
          device: "Apple Silicon M5 Pro",
          unified_memory: "48.0 GB",
          max_kv_tokens: 32768,
          quantization: "4-bit (MLX)",
          lora_slots_allocated: 0,
          lora_slots_capacity: 4,
          postgres_version: "17.11",
          runtime_mode: "offline_deterministic",
        },
        database_connected: true,
        database_status: "PostgreSQL 17.11 RLS Enforced",
        runtime_mode: "offline_deterministic",
        timestamp: "2026-11-15T10:00:00Z",
      },
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("domain-card-biotech_scout")).toBeInTheDocument();
    });

    expect(screen.getByTestId("domain-card-biotech_scout")).toHaveTextContent(
      "Biotech & Clinical Trials"
    );
    expect(screen.getByTestId("domain-counts-biotech_scout")).toHaveTextContent(
      "1,240 chunks"
    );
    expect(screen.getByTestId("scope-chip-public_bio")).toBeInTheDocument();
    expect(screen.getByTestId("quick-pick-biotech_scout-0")).toHaveTextContent(
      "Biotech & Clinical Trials"
    );
  });

  it("hydrates extended engineer default_scopes from /api/v1/workspace-config on initial mount without manual click", async () => {
    const extendedScopes = [
      "public_eng",
      "public_docs",
      "public_de",
      "repo:core_services",
    ];
    installMockApiFetch({
      workspaceConfigResponse: {
        ...TEST_WORKSPACE_CONFIG,
        role_presets: [
          {
            role: "engineer",
            label: "Engineer",
            default_scopes: extendedScopes,
          },
          ...TEST_WORKSPACE_CONFIG.role_presets.filter(
            (p) => p.role !== "engineer"
          ),
        ],
      },
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("active-scopes-list")).toHaveTextContent(
        "public_de"
      );
      expect(screen.getByTestId("active-scopes-list")).toHaveTextContent(
        "repo:core_services"
      );
      expect(screen.getByTestId("header-auth-badge")).toHaveTextContent(
        "Bearer: engineer"
      );
    });
  });
});
