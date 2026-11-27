import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import {
  applyHandoff,
  clearCachedAuthSession,
  fetchDomains,
  fetchTelemetryRoi,
  previewHandoff,
  setRuntimeAuthMode,
} from "../api/client";
import { streamScoutWorkflow } from "../api/sse";
import {
  ArchitectureMapViewer,
  MERMAID_SECURITY_CONFIG,
} from "../components/ArchitectureMapViewer";
import { SSEStageEvent } from "../types/contracts";
import {
  installMockApiFetch,
  TEST_CODE_PACK,
  TEST_ENGINEER_PRINCIPAL,
} from "./fixtures";

afterEach(() => {
  clearCachedAuthSession();
  setRuntimeAuthMode("development");
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("PR-2 Web-UI auth, SSE buffering/abort, strict Mermaid, and mock 401/403 fidelity", () => {
  it("never calls /api/v1/auth/session in production OIDC/IAP mode and passes credentials: include across endpoints", async () => {
    const fetchMock = installMockApiFetch({
      runtimeMode: "production",
      mePrincipal: {
        user_id: "oidc_bob",
        role: "engineer",
        allowed_scopes: ["public_eng"],
        max_session_paid_tokens: 20000,
      },
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("production-oidc-badge")).toBeInTheDocument();
      expect(
        screen.getByTestId("domain-card-code_architecture_scout")
      ).toBeInTheDocument();
      expect(screen.getByTestId("run-scout-btn")).not.toBeDisabled();
    });

    fireEvent.click(screen.getByTestId("run-scout-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("answer-first-hero")).toBeInTheDocument();
    });

    const sessionCalls = fetchMock.mock.calls.filter(([url]) =>
      String(url).endsWith("/api/v1/auth/session")
    );
    expect(sessionCalls).toHaveLength(0);

    for (const ep of [
      "/api/v1/health",
      "/api/v1/auth/me",
      "/api/v1/domains",
      "/api/v1/telemetry/roi",
      "/api/v1/scout/stream",
    ]) {
      const matching = fetchMock.mock.calls.filter(([url]) =>
        String(url).endsWith(ep)
      );
      expect(matching.length).toBeGreaterThanOrEqual(1);
      for (const [, init] of matching) {
        expect(init?.credentials).toBe("include");
      }
    }
  });

  it("attaches Authorization + credentials across domains/ROI/handoff and clears cache on 401", async () => {
    let previewCalls = 0;
    const baseFetch = installMockApiFetch();
    const wrappedFetch = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.endsWith("/api/v1/handoff/preview")) {
          previewCalls += 1;
          if (previewCalls === 1) {
            return new Response(JSON.stringify({ detail: "Expired token" }), {
              status: 401,
              headers: { "Content-Type": "application/json" },
            });
          }
        }
        return baseFetch(input, init);
      }
    );
    vi.stubGlobal("fetch", wrappedFetch);

    await fetchDomains(TEST_ENGINEER_PRINCIPAL);
    await fetchTelemetryRoi(TEST_ENGINEER_PRINCIPAL);

    await expect(
      previewHandoff(TEST_CODE_PACK, "gemini-3.8-flash", TEST_ENGINEER_PRINCIPAL)
    ).rejects.toThrow(/401/);

    const preview = await previewHandoff(
      TEST_CODE_PACK,
      "gemini-3.8-flash",
      TEST_ENGINEER_PRINCIPAL
    );
    await applyHandoff(
      { confirmation_token: preview.confirmation_token, confirmed: true },
      TEST_ENGINEER_PRINCIPAL
    );

    const sessionCalls = wrappedFetch.mock.calls.filter(([url]) =>
      String(url).endsWith("/api/v1/auth/session")
    );
    expect(sessionCalls.length).toBe(2);

    for (const ep of [
      "/api/v1/domains",
      "/api/v1/telemetry/roi",
      "/api/v1/handoff/preview",
      "/api/v1/handoff/apply",
    ]) {
      const call = wrappedFetch.mock.calls.find(([url]) =>
        String(url).endsWith(ep)
      );
      expect(call?.[1]?.credentials).toBe("include");
      const hdrs = call?.[1]?.headers as Record<string, string>;
      expect(hdrs.Authorization).toMatch(/^Bearer /);
    }
  });

  it("enforces 401/403 fidelity in installMockApiFetch for unauthenticated or unauthorized callers", async () => {
    installMockApiFetch();

    const unauthDomains = await fetch("/api/v1/domains");
    expect(unauthDomains.status).toBe(401);

    const nonEngPrincipal = {
      ...TEST_ENGINEER_PRINCIPAL,
      role: "product_manager" as const,
    };
    await expect(
      previewHandoff(TEST_CODE_PACK, "gemini-3.8-flash", nonEngPrincipal)
    ).rejects.toThrow(/403/);
  });

  it("handles split UTF-8 and CRLF SSE frames plus AbortController cancellation cleanly", async () => {
    installMockApiFetch();
    const encoder = new TextEncoder();
    const frameBytes = encoder.encode(
      'event: stage_update\r\ndata: {"stage":1,"stage_name":"Check — UTF8 🚀","status":"completed","details":"ok"}\r\n\r\n'
    );
    const crlfSplitPos = frameBytes.length - 3; // right between \r and \n\r\n
    const utf8SplitPos = frameBytes.indexOf(0xf0) + 2; // mid 4-byte emoji

    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(frameBytes.slice(0, utf8SplitPos));
        controller.enqueue(frameBytes.slice(utf8SplitPos, crlfSplitPos));
        controller.enqueue(frameBytes.slice(crlfSplitPos));
        controller.close();
      },
    });

    const baseFetch = installMockApiFetch();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.endsWith("/api/v1/scout/stream")) {
          return new Response(stream, {
            status: 200,
            headers: { "Content-Type": "text/event-stream" },
          });
        }
        return baseFetch(input, init);
      })
    );

    const received: SSEStageEvent[] = [];
    let completed = false;
    let errored: Error | null = null;

    await new Promise<void>((resolve) => {
      streamScoutWorkflow({
        request: { domain_id: "code_architecture_scout", query: "Check SSE" },
        principal: TEST_ENGINEER_PRINCIPAL,
        onEvent: (ev) => received.push(ev),
        onError: (err) => {
          errored = err;
          resolve();
        },
        onComplete: () => {
          completed = true;
          resolve();
        },
      });
    });

    expect(errored).toBeNull();
    expect(completed).toBe(true);
    expect(received).toHaveLength(1);
    expect(received[0]?.event).toBe("stage_update");
    if (received[0]?.event === "stage_update") {
      expect(received[0].data.stage_name).toBe("Check — UTF8 🚀");
    }

    // Abort mid-stream cancels without triggering onError
    installMockApiFetch({ hangSseStream: true });
    let abortErrored = false;
    const cancel = streamScoutWorkflow({
      request: { domain_id: "code_architecture_scout", query: "Hang SSE" },
      principal: TEST_ENGINEER_PRINCIPAL,
      onEvent: () => {},
      onError: () => {
        abortErrored = true;
      },
      onComplete: () => {},
    });
    cancel();
    await new Promise((r) => setTimeout(r, 20));
    expect(abortErrored).toBe(false);
  });

  it("enforces Mermaid securityLevel: strict and strips click/script/init directives", () => {
    expect(MERMAID_SECURITY_CONFIG.securityLevel).toBe("strict");
    expect(MERMAID_SECURITY_CONFIG.htmlLabels).toBe(false);

    render(
      <ArchitectureMapViewer
        selectedDomain="code_architecture_scout"
        contextPack={{
          ...TEST_CODE_PACK,
          architecture_mermaid: [
            "%%{init: {'securityLevel': 'loose'}}%%",
            "flowchart LR",
            '  A["Safe Node"] --> B["Next Node"]',
            '  click A "javascript:alert(1)"',
            "  <script>alert(1)</script>",
          ].join("\n"),
        }}
      />
    );

    fireEvent.click(screen.getByTestId("toggle-mermaid-source-btn"));
    const code = screen.getByTestId("mermaid-source-code").textContent ?? "";
    expect(code).toContain("%%{init: {'securityLevel': 'strict'}}%%");
    expect(code).toContain("flowchart LR");
    expect(code).not.toContain("'loose'");
    expect(code).not.toContain("javascript:");
    expect(code).not.toContain("<script>");
    expect(code).not.toContain("click A");
  });
});
