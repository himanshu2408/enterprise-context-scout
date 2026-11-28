import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import indexHtmlRaw from "../../index.html?raw";
import viteConfigRaw from "../../vite.config.ts?raw";
import App from "../App";
import { clearCachedAuthSession } from "../api/client";
import { ArchitectureMapViewer } from "../components/ArchitectureMapViewer";
import { ProductTruthView } from "../components/ProductTruthView";
import { SessionBranchBar } from "../components/SessionBranchBar";
import { SpecializationAdminView } from "../components/SpecializationAdminView";
import {
  installMockApiFetch,
  TEST_BRANCHES,
  TEST_CODE_PACK,
  TEST_DOMAINS,
  TEST_ENGINEER_PRINCIPAL,
  TEST_HEALTH_RESPONSE,
  TEST_SESSIONS,
  TEST_SIDE_EFFECTS,
} from "./fixtures";

beforeEach(() => {
  clearCachedAuthSession();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("M5-03 Persona, Admin, Production OIDC & Air-Gap Suite", () => {
  it("SessionBranchBar renders session drawer, commit drift alerts, and Branch/MR overlay picker", () => {
    const onSelectSession = vi.fn();
    const onNewSession = vi.fn();
    const onSelectBranch = vi.fn();

    render(
      <SessionBranchBar
        sessions={TEST_SESSIONS}
        activeSessionId="sess_01"
        onSelectSession={onSelectSession}
        onNewSession={onNewSession}
        branches={TEST_BRANCHES}
        selectedBranchId="main"
        onSelectBranch={onSelectBranch}
        commitDriftWarnings={["slicer/bounds.py modified since last turn"]}
      />
    );

    expect(screen.getByTestId("session-branch-bar")).toBeInTheDocument();
    expect(screen.getByTestId("commit-drift-badge")).toHaveTextContent(
      "Commit Drift"
    );
    expect(screen.getByTestId("active-branch-badge")).toHaveTextContent("HEAD");

    fireEvent.click(screen.getByTestId("toggle-session-drawer-btn"));
    expect(screen.getByTestId("session-history-drawer")).toBeInTheDocument();
    expect(
      screen.getByTestId("session-drift-pill-sess_01")
    ).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("select-session-sess_02"));
    expect(onSelectSession).toHaveBeenCalledWith("sess_02");

    fireEvent.click(screen.getByTestId("new-session-btn"));
    expect(onNewSession).toHaveBeenCalledTimes(1);

    fireEvent.change(screen.getByTestId("branch-mr-picker"), {
      target: { value: "mr-412" },
    });
    expect(onSelectBranch).toHaveBeenCalledWith("mr-412");
  });

  it("ProductTruthView renders side-effect matrix, SPEC_DRIFT_WARNING banner, filters, and CSV/MD export", () => {
    const { rerender } = render(
      <ProductTruthView
        contextPack={null}
        principalRole="product_manager"
        selectedDomain="code_architecture_scout"
      />
    );
    expect(screen.getByTestId("product-truth-empty")).toBeInTheDocument();

    const createObjectUrlMock = vi.fn(() => "blob:mock-truth");
    const revokeObjectUrlMock = vi.fn();
    Object.defineProperty(URL, "createObjectURL", {
      value: createObjectUrlMock,
      configurable: true,
      writable: true,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      value: revokeObjectUrlMock,
      configurable: true,
      writable: true,
    });

    rerender(
      <ProductTruthView
        contextPack={{
          ...TEST_CODE_PACK,
          side_effects: TEST_SIDE_EFFECTS,
        }}
        principalRole="product_manager"
        selectedDomain="code_architecture_scout"
      />
    );

    expect(
      screen.getByTestId("spec-drift-warning-banner")
    ).toHaveTextContent("SPEC_DRIFT_WARNING");
    expect(screen.getByTestId("side-effect-matrix-table")).toBeInTheDocument();
    expect(screen.getByTestId("side-effect-row-0")).toHaveTextContent(
      "TenantAuditRepo.record_purge"
    );
    expect(screen.getByTestId("side-effect-row-1")).toHaveTextContent(
      "WebhookDispatcher.emit_alert"
    );

    // Filter to Drift Only
    fireEvent.click(screen.getByTestId("side-effect-filter-drift"));
    expect(screen.getByTestId("side-effect-row-0")).toHaveTextContent(
      "WebhookDispatcher.emit_alert"
    );
    expect(screen.queryByTestId("side-effect-row-1")).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("side-effect-filter-all"));
    expect(screen.getByTestId("side-effect-row-1")).toBeInTheDocument();

    // CSV & Markdown Export preview
    fireEvent.click(screen.getByTestId("export-truth-csv-btn"));
    expect(screen.getByTestId("truth-export-preview")).toHaveTextContent(
      "TenantAuditRepo.record_purge"
    );
    fireEvent.click(screen.getByTestId("export-truth-md-btn"));
    expect(screen.getByTestId("truth-export-preview")).toHaveTextContent(
      "| Category |"
    );

    expect(createObjectUrlMock).toHaveBeenCalled();
    expect(revokeObjectUrlMock).toHaveBeenCalled();
  });

  it("ArchitectureMapViewer renders interactive SVG nodes, layout toggle, and Mermaid source", () => {
    const { rerender } = render(
      <ArchitectureMapViewer
        contextPack={null}
        selectedDomain="code_architecture_scout"
      />
    );
    expect(screen.getByTestId("architecture-map-empty")).toBeInTheDocument();

    rerender(
      <ArchitectureMapViewer
        contextPack={TEST_CODE_PACK}
        selectedDomain="code_architecture_scout"
      />
    );

    expect(screen.getByTestId("architecture-map-svg")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("arch-node-0"));
    expect(
      screen.getByTestId("arch-selected-node-detail")
    ).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("toggle-mermaid-source-btn"));
    const mermaidPre = screen.getByTestId("mermaid-source-code");
    expect(mermaidPre).toHaveTextContent("flowchart LR");

    fireEvent.click(screen.getByTestId("arch-layout-toggle-btn"));
    expect(mermaidPre).toHaveTextContent("flowchart TD");
  });

  it("ProductTruthView and ArchitectureMapViewer render server-backed side_effects and architecture_mermaid directly", () => {
    const serverPack = {
      ...TEST_CODE_PACK,
      // Intentionally stripped of regex keywords so client fallback cannot invent rows
      code_slices: TEST_CODE_PACK.code_slices.map((sl) => ({
        ...sl,
        verbatim_code_with_lines: "1 | def pure_helper(x: int) -> int:\n2 |     return x + 1",
      })),
      side_effects: [
        {
          side_effect_id: "se-server-99",
          effect_type: "QUEUE_PUBLISH" as const,
          qualified_symbol: "context_scout.events.Publisher.dispatch",
          file_path: "src/context_scout/events/publisher.py",
          line_range: "L42-L58",
          target_resource: "pubsub_kafka_event: audit.events.v1",
          spec_status: "VERIFIED" as const,
          verification_note: "Tree-sitter Pillar-5 pubsub_kafka_event in publisher.py",
        },
      ],
      architecture_mermaid:
        'flowchart LR\n  server_node_a["Publisher.dispatch"]\n  server_node_b["audit.events.v1"]\n  server_node_a -->|QUEUE_PUBLISH| server_node_b',
    };

    render(
      <div>
        <ProductTruthView
          contextPack={serverPack}
          principalRole="product_manager"
          selectedDomain="code_architecture_scout"
        />
        <ArchitectureMapViewer
          contextPack={serverPack}
          selectedDomain="code_architecture_scout"
        />
      </div>
    );

    // ProductTruthView renders the exact server side_effects row (not fallback pure slice)
    expect(screen.getByTestId("side-effect-row-0")).toHaveTextContent(
      "context_scout.events.Publisher.dispatch"
    );
    expect(screen.getByTestId("side-effect-row-0")).toHaveTextContent(
      "pubsub_kafka_event: audit.events.v1"
    );
    expect(screen.queryByTestId("side-effect-row-1")).not.toBeInTheDocument();

    // ArchitectureMapViewer uses server architecture_mermaid directly and rewrites direction on toggle
    fireEvent.click(screen.getByTestId("toggle-mermaid-source-btn"));
    const mermaidPre = screen.getByTestId("mermaid-source-code");
    expect(mermaidPre).toHaveTextContent('server_node_a["Publisher.dispatch"]');
    expect(mermaidPre).toHaveTextContent("server_node_a -->|QUEUE_PUBLISH| server_node_b");
    expect(mermaidPre).toHaveTextContent("flowchart LR");

    fireEvent.click(screen.getByTestId("arch-layout-toggle-btn"));
    expect(mermaidPre).toHaveTextContent("flowchart TD");
    expect(mermaidPre).toHaveTextContent("server_node_a -->|QUEUE_PUBLISH| server_node_b");
  });

  it("SpecializationAdminView renders governance cards, dry-run probe, sidecar table, and RBAC-gated purge", async () => {
    const fetchMock = installMockApiFetch();
    const { rerender } = render(
      <SpecializationAdminView
        domains={[]}
        principal={TEST_ENGINEER_PRINCIPAL}
        hardwareProfile={TEST_HEALTH_RESPONSE.hardware_profile}
        adapterPoolStatus={TEST_HEALTH_RESPONSE.adapter_pool_status}
      />
    );

    expect(screen.getByTestId("specialization-admin-view")).toBeInTheDocument();
    expect(screen.getByTestId("admin-domain-select")).toBeDisabled();

    // Non-compliance engineer sees RBAC restriction on purge
    rerender(
      <SpecializationAdminView
        domains={TEST_DOMAINS}
        principal={TEST_ENGINEER_PRINCIPAL}
        hardwareProfile={TEST_HEALTH_RESPONSE.hardware_profile}
        adapterPoolStatus={TEST_HEALTH_RESPONSE.adapter_pool_status}
      />
    );

    expect(screen.getByTestId("admin-archetype-badge")).toHaveTextContent(
      "Code + AST QLoRA"
    );
    expect(
      screen.getByTestId("training-scope-ceiling-panel")
    ).toHaveTextContent("OWASP ASI06");
    expect(screen.getByTestId("rslora-policy-card")).toHaveTextContent(
      "use_rslora: true"
    );
    expect(screen.getByTestId("rslora-policy-card")).toHaveTextContent(
      "80/10/10"
    );
    expect(screen.getByTestId("canary-gate-card")).toHaveTextContent(
      "CanaryMemorizationGate: 0/10 leak"
    );
    expect(screen.getByTestId("adapter-sidecar-table")).toHaveTextContent(
      "code_architecture_scout"
    );

    fireEvent.click(screen.getByTestId("run-dryrun-probe-btn"));
    expect(screen.getByTestId("dryrun-probe-result")).toHaveTextContent(
      "PASS (Simulated UI Metadata Check): 1-step vLLM dry-run schema valid"
    );

    expect(screen.getByTestId("admin-purge-rbac-notice")).toBeInTheDocument();
    expect(screen.getByTestId("admin-purge-submit-btn")).toBeDisabled();

    // Compliance officer can execute <60s purge
    rerender(
      <SpecializationAdminView
        domains={TEST_DOMAINS}
        principal={{
          ...TEST_ENGINEER_PRINCIPAL,
          role: "compliance_officer",
          allowed_scopes: ["public_eng", "public_docs", "audit:compliance"],
        }}
        hardwareProfile={TEST_HEALTH_RESPONSE.hardware_profile}
        adapterPoolStatus={TEST_HEALTH_RESPONSE.adapter_pool_status}
      />
    );

    expect(
      screen.queryByTestId("admin-purge-rbac-notice")
    ).not.toBeInTheDocument();

    // (a) Fail-closed file_path without repo_name: button disabled, validation banner visible, programmatic submit blocked
    expect(screen.getByTestId("admin-purge-target-type")).toHaveValue(
      "file_path"
    );
    fireEvent.change(screen.getByTestId("admin-purge-target-input"), {
      target: { value: "slicer/bounds.py" },
    });
    expect(screen.getByTestId("admin-purge-submit-btn")).toBeDisabled();
    expect(
      screen.getByTestId("admin-purge-validation-error")
    ).toHaveTextContent("Repository Name is required for file path purges.");
    expect(screen.getByTestId("admin-purge-repo-input")).toHaveAttribute(
      "aria-invalid",
      "true"
    );

    fireEvent.submit(screen.getByTestId("admin-purge-form"));
    expect(screen.getByTestId("admin-purge-error")).toHaveTextContent(
      "Repository Name is required for file path purges."
    );
    expect(screen.queryByTestId("admin-purge-result")).not.toBeInTheDocument();
    expect(
      fetchMock.mock.calls.filter(([url]) =>
        String(url).endsWith("/api/v1/admin/purge")
      ).length
    ).toBe(0);

    // (b) Valid file_path + genuine repo_name sends exact repo_name
    fireEvent.change(screen.getByTestId("admin-purge-repo-input"), {
      target: { value: "acme/leaked-repo" },
    });
    fireEvent.change(screen.getByTestId("admin-purge-target-input"), {
      target: { value: "src/secrets/leaked.py" },
    });
    expect(screen.getByTestId("admin-purge-submit-btn")).toBeEnabled();
    expect(
      screen.queryByTestId("admin-purge-validation-error")
    ).not.toBeInTheDocument();
    fireEvent.click(screen.getByTestId("admin-purge-submit-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("admin-purge-result")).toHaveTextContent(
        "SLA MET"
      );
      const purgeCalls = fetchMock.mock.calls.filter(([url]) =>
        String(url).endsWith("/api/v1/admin/purge")
      );
      expect(purgeCalls.length).toBe(1);
      const fileBody = JSON.parse(
        String(purgeCalls[0]?.[1]?.body ?? "{}")
      ) as Record<string, unknown>;
      expect(fileBody).toEqual({
        domain_id: "code_architecture_scout",
        repo_name: "acme/leaked-repo",
        file_path: "src/secrets/leaked.py",
        include_derived_wiki: true,
      });
    });

    // (c) Dotted doc_id in doc_id mode disables repo input and omits repo_name and file_path
    fireEvent.change(screen.getByTestId("admin-purge-target-type"), {
      target: { value: "doc_id" },
    });
    expect(screen.getByTestId("admin-purge-repo-input")).toBeDisabled();
    fireEvent.change(screen.getByTestId("admin-purge-target-input"), {
      target: { value: "doc.v1.2026" },
    });
    expect(screen.getByTestId("admin-purge-submit-btn")).toBeEnabled();
    fireEvent.click(screen.getByTestId("admin-purge-submit-btn"));

    await waitFor(() => {
      const purgeCalls = fetchMock.mock.calls.filter(([url]) =>
        String(url).endsWith("/api/v1/admin/purge")
      );
      expect(purgeCalls.length).toBe(2);
      const docBody = JSON.parse(
        String(purgeCalls[1]?.[1]?.body ?? "{}")
      ) as Record<string, unknown>;
      expect(docBody).toEqual({
        domain_id: "code_architecture_scout",
        doc_id: "doc.v1.2026",
        include_derived_wiki: true,
      });
      expect(docBody.repo_name).toBeUndefined();
      expect(docBody.file_path).toBeUndefined();
    });

    // (d) Entire repository purge in repo mode disables target input and omits file_path and doc_id
    fireEvent.change(screen.getByTestId("admin-purge-target-type"), {
      target: { value: "repo" },
    });
    expect(screen.getByTestId("admin-purge-target-input")).toBeDisabled();
    fireEvent.change(screen.getByTestId("admin-purge-repo-input"), {
      target: { value: "acme/retired-repo" },
    });
    expect(screen.getByTestId("admin-purge-submit-btn")).toBeEnabled();
    fireEvent.click(screen.getByTestId("admin-purge-submit-btn"));

    await waitFor(() => {
      const purgeCalls = fetchMock.mock.calls.filter(([url]) =>
        String(url).endsWith("/api/v1/admin/purge")
      );
      expect(purgeCalls.length).toBe(3);
      const repoBody = JSON.parse(
        String(purgeCalls[2]?.[1]?.body ?? "{}")
      ) as Record<string, unknown>;
      expect(repoBody).toEqual({
        domain_id: "code_architecture_scout",
        repo_name: "acme/retired-repo",
        include_derived_wiki: true,
      });
      expect(repoBody.file_path).toBeUndefined();
      expect(repoBody.doc_id).toBeUndefined();
    });
  });

  it("SpecializationAdminView surfaces backend HTTP 400 purge rejections in admin-purge-error", async () => {
    installMockApiFetch({
      purgeStatus: 400,
      purgeDetail: "file_path purge requires repo_name",
    });

    render(
      <SpecializationAdminView
        domains={TEST_DOMAINS}
        principal={{
          ...TEST_ENGINEER_PRINCIPAL,
          role: "compliance_officer",
          allowed_scopes: ["public_eng", "public_docs", "audit:compliance"],
        }}
        hardwareProfile={TEST_HEALTH_RESPONSE.hardware_profile}
        adapterPoolStatus={TEST_HEALTH_RESPONSE.adapter_pool_status}
      />
    );

    fireEvent.change(screen.getByTestId("admin-purge-repo-input"), {
      target: { value: "acme/leaked-repo" },
    });
    fireEvent.change(screen.getByTestId("admin-purge-target-input"), {
      target: { value: "src/secrets/leaked.py" },
    });
    fireEvent.click(screen.getByTestId("admin-purge-submit-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("admin-purge-error")).toHaveTextContent(
        "file_path purge requires repo_name"
      );
    });
    expect(screen.queryByTestId("admin-purge-result")).not.toBeInTheDocument();
  });

  it("hides RoleSwitcher and hydrates principal via /api/v1/auth/me in production OIDC mode", async () => {
    const fetchMock = installMockApiFetch({
      runtimeMode: "production",
      mePrincipal: {
        user_id: "oidc_alice",
        role: "compliance_officer",
        allowed_scopes: ["public_eng", "audit:compliance"],
        max_session_paid_tokens: 12000,
      },
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("production-oidc-badge")).toBeInTheDocument();
      expect(screen.getByTestId("production-oidc-badge")).toHaveTextContent(
        "compliance_officer"
      );
    });

    expect(screen.queryByTestId("role-switcher")).not.toBeInTheDocument();
    const meCalls = fetchMock.mock.calls.filter(([url]) =>
      String(url).endsWith("/api/v1/auth/me")
    );
    expect(meCalls.length).toBeGreaterThanOrEqual(1);
  });

  it("enforces air-gapped index.html CSP, 127.0.0.1 Vite binding, and valid wOF2 self-hosted fonts", async () => {
    expect(indexHtmlRaw).not.toContain("fonts.googleapis.com");
    expect(indexHtmlRaw).not.toContain("fonts.gstatic.com");
    expect(indexHtmlRaw).toContain('http-equiv="Content-Security-Policy"');
    expect(indexHtmlRaw).toContain("default-src 'self'");
    expect(indexHtmlRaw).not.toContain("frame-ancestors");

    expect(viteConfigRaw).toContain("127.0.0.1");
    expect(viteConfigRaw).not.toMatch(/allowedHosts:\s*true/);

    const fsModuleName = "node:" + "fs";
    const pathModuleName = "node:" + "path";
    const procModuleName = "node:" + "process";
    const nodeFs = (await import(/* @vite-ignore */ fsModuleName)) as {
      existsSync: (p: string) => boolean;
      readFileSync: (p: string) => Uint8Array;
    };
    const nodePath = (await import(/* @vite-ignore */ pathModuleName)) as {
      join: (...parts: string[]) => string;
    };
    const nodeProc = (await import(/* @vite-ignore */ procModuleName)) as {
      cwd: () => string;
    };
    const webUiRoot = nodeProc.cwd();
    const fontFiles = [
      "PlusJakartaSans-Regular.woff2",
      "PlusJakartaSans-SemiBold.woff2",
      "PlusJakartaSans-Bold.woff2",
      "JetBrainsMono-Regular.woff2",
      "JetBrainsMono-Medium.woff2",
    ];
    for (const fontFile of fontFiles) {
      const fullPath = nodePath.join(webUiRoot, "public", "fonts", fontFile);
      expect(nodeFs.existsSync(fullPath)).toBe(true);
      const bytes = nodeFs.readFileSync(fullPath);
      const header = String.fromCharCode(
        bytes[0] ?? 0,
        bytes[1] ?? 0,
        bytes[2] ?? 0,
        bytes[3] ?? 0
      );
      expect(header).toBe("wOF2");
    }
  });

  it("fails Dry-Run Probe + sanitizes host paths on invalid adapters", () => {
    render(
      <SpecializationAdminView
        domains={[
          {
            ...TEST_DOMAINS[1],
            domain_id: "data_engineering_scout",
            display_name: "Data Engineering Scout",
            lora_enabled: false,
            verified: false,
            qlora_rank: null,
            qlora_alpha: null,
            safetensors_bytes: 0,
          },
        ]}
        principal={{
          ...TEST_ENGINEER_PRINCIPAL,
          role: "compliance_officer",
          allowed_scopes: ["public_eng", "audit:compliance"],
        }}
        adapterPoolStatus={{
          data_engineering_scout: {
            domain_id: "data_engineering_scout",
            adapter_path: "/opt/workspace/enterprise-context-scout/domains/data_engineering_scout/adapter",
            is_valid: false,
            r: 0,
            lora_alpha: 0,
            is_moe: false,
            final_loss: null,
            safetensors_bytes: 0,
            error_message:
              "Adapter directory not found: /opt/workspace/enterprise-context-scout/domains/data_engineering_scout/adapter",
          },
        }}
      />
    );

    fireEvent.click(screen.getByTestId("run-dryrun-probe-btn"));
    const probeResult = screen.getByTestId("dryrun-probe-result");
    expect(probeResult).toHaveTextContent(
      "FAIL (Simulated UI Metadata Check):"
    );
    expect(probeResult).toHaveTextContent("safetensors_artifact: 0 B (MISSING)");
    expect(probeResult).toHaveTextContent("rank_bound (r <= 32): 0 (MISSING)");

    const sidecarTable = screen.getByTestId("adapter-sidecar-table");
    expect(sidecarTable).not.toHaveTextContent("/opt/workspace/");
    expect(sidecarTable).toHaveTextContent(
      "domains/data_engineering_scout/adapter"
    );
  });
});
