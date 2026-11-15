import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearCachedAuthSession } from "../api/client";
import { ScoutConsole } from "../components/ScoutConsole";
import {
  installMockApiFetch,
  TEST_CODE_PACK,
  TEST_DIAGNOSTIC,
  TEST_ENGINEER_PRINCIPAL,
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

describe("ScoutConsole", () => {
  it("renders console input, 6 quick-picks, source filter pill, and 4-stage stepper in idle state", () => {
    render(
      <ScoutConsole
        selectedDomain="trading_code_scout"
        principal={TEST_ENGINEER_PRINCIPAL}
        sourceFilter="wiki"
        onContextPackReady={vi.fn()}
      />
    );

    expect(screen.getByTestId("scout-query-input")).toBeInTheDocument();
    expect(screen.getByTestId("run-scout-btn")).toBeInTheDocument();
    expect(screen.getByTestId("console-source-filter-pill")).toHaveTextContent(
      "Confluence & Docs"
    );

    expect(screen.getByTestId("stage-card-1")).toBeInTheDocument();
    expect(screen.getByTestId("stage-card-2")).toBeInTheDocument();
    expect(screen.getByTestId("stage-card-3")).toBeInTheDocument();
    expect(screen.getByTestId("stage-card-4")).toBeInTheDocument();

    expect(screen.getByTestId("quick-pick-code_ast")).toBeInTheDocument();
    expect(screen.getByTestId("quick-pick-runbook_issue")).toBeInTheDocument();
    expect(screen.getByTestId("quick-pick-orderbook_docs")).toBeInTheDocument();
    expect(screen.getByTestId("quick-pick-cross_domain")).toBeInTheDocument();
    expect(screen.getByTestId("quick-pick-guardrail_block")).toBeInTheDocument();
    expect(
      screen.getByTestId("quick-pick-zero_dollar_abstain")
    ).toBeInTheDocument();
  });

  it("toggles execution mode and updates query + domain when clicking a quick-pick", () => {
    const onSelectDomain = vi.fn();
    render(
      <ScoutConsole
        selectedDomain="trading_code_scout"
        principal={TEST_ENGINEER_PRINCIPAL}
        onSelectDomain={onSelectDomain}
        onContextPackReady={vi.fn()}
      />
    );

    const streamBtn = screen.getByTestId("mode-stream-btn");
    const restBtn = screen.getByTestId("mode-rest-btn");
    expect(streamBtn).toHaveClass("mode-btn-active");

    fireEvent.click(restBtn);
    expect(restBtn).toHaveClass("mode-btn-active");
    expect(streamBtn).not.toHaveClass("mode-btn-active");

    fireEvent.click(screen.getByTestId("quick-pick-runbook_issue"));
    const textarea = screen.getByTestId(
      "scout-query-input"
    ) as HTMLTextAreaElement;
    expect(textarea.value).toContain("production rollback runbook");
    expect(onSelectDomain).toHaveBeenCalledWith("federated_all");
  });

  it("streams live SSE stages [1..4] with signed Bearer token, bounce-back remediation, and elapsed ms", async () => {
    const fetchMock = installMockApiFetch({
      sseEvents: [
        {
          event: "stage_update",
          data: {
            stage: 1,
            stage_name: "Identity, RBAC & Prompt Guard",
            status: "completed",
            details: "Passed ShieldGemma & PG17 RLS.",
            timestamp: "2026-11-14T10:00:01Z",
          },
        },
        {
          event: "stage_update",
          data: {
            stage: 2,
            stage_name: "Federated Corpus Retrieval",
            status: "completed",
            details: "Searched 2 authorized domains under RLS.",
            timestamp: "2026-11-14T10:00:01Z",
          },
        },
        {
          event: "stage_update",
          data: {
            stage: 3,
            stage_name: "Local Synthesis & AST Closure",
            status: "completed",
            details: "Expanded 1-hop AST dependency closure.",
            timestamp: "2026-11-14T10:00:02Z",
          },
        },
        {
          event: "bounce_back",
          data: {
            stage: 4,
            status: "retrying",
            bounce_count: 1,
            feedback: "Missing 1-hop callee VolatilityForecastResult.",
            remediation_action: "expand_ast_1hop",
            details: "Expanding AST slice locally.",
            paid_tokens_spent: 0,
            cost_usd: 0,
            timestamp: "2026-11-14T10:00:02Z",
          },
        },
        {
          event: "stage_update",
          data: {
            stage: 4,
            stage_name: "Pre-Flight Grounding & Audit",
            status: "completed",
            details: "All AST slices verified.",
            timestamp: "2026-11-14T10:00:03Z",
          },
        },
        {
          event: "context_pack_ready",
          data: {
            status: "ready",
            pack: TEST_CODE_PACK,
            token_savings: TEST_TOKEN_SAVINGS,
            timestamp: "2026-11-14T10:00:03Z",
          },
        },
      ],
    });

    const onPackReady = vi.fn();
    const onComplete = vi.fn();

    render(
      <ScoutConsole
        selectedDomain="trading_code_scout"
        principal={TEST_ENGINEER_PRINCIPAL}
        onContextPackReady={onPackReady}
        onQueryComplete={onComplete}
      />
    );

    fireEvent.click(screen.getByTestId("run-scout-btn"));

    await waitFor(() => {
      expect(onPackReady).toHaveBeenCalledWith(
        TEST_CODE_PACK,
        TEST_TOKEN_SAVINGS,
        null,
        null
      );
    });

    expect(screen.getByTestId("bounce-back-badge")).toHaveTextContent(
      "Missing 1-hop callee VolatilityForecastResult."
    );
    expect(screen.getByTestId("check-stage-1")).toBeInTheDocument();
    expect(screen.getByTestId("check-stage-2")).toBeInTheDocument();
    expect(screen.getByTestId("check-stage-3")).toBeInTheDocument();
    expect(screen.getByTestId("check-stage-4")).toBeInTheDocument();
    expect(screen.getByTestId("timeline-elapsed-ms")).toBeInTheDocument();
    expect(screen.getByTestId("shield-gemma-safe-badge")).toBeInTheDocument();
    expect(onComplete).toHaveBeenCalledTimes(1);

    const streamCall = fetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/api/v1/scout/stream")
    );
    expect(streamCall).toBeDefined();
    const headers = (streamCall?.[1]?.headers ?? {}) as Record<string, string>;
    expect(headers.Authorization).toMatch(/^Bearer signed\.hmac\./);
  });

  it("displays ShieldGemma alert on HTTP 403 prompt injection block and clears pack", async () => {
    installMockApiFetch({
      sseStatus: 403,
      sseDetail: "Blocked prompt_injection pattern detected by ShieldGemma 2.",
    });

    const onPackReady = vi.fn();
    const onClearPack = vi.fn();
    render(
      <ScoutConsole
        selectedDomain="trading_code_scout"
        principal={TEST_ENGINEER_PRINCIPAL}
        onContextPackReady={onPackReady}
        onClearContextPack={onClearPack}
      />
    );

    fireEvent.click(screen.getByTestId("quick-pick-guardrail_block"));
    fireEvent.click(screen.getByTestId("run-scout-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("guardrail-alert-banner")).toHaveTextContent(
        "ShieldGemma 2"
      );
    });
    expect(screen.getByTestId("fail-stage-1")).toBeInTheDocument();
    expect(onClearPack).toHaveBeenCalledTimes(1);
    expect(onPackReady).toHaveBeenCalledWith(null, null, null, null);
  });

  it("executes synchronous REST mode and displays $0 abstention diagnostic when insufficient", async () => {
    installMockApiFetch({
      queryResponse: {
        trace_id: TEST_DIAGNOSTIC.trace_id,
        domain_id: "trading_code_scout",
        passed: false,
        context_pack: null,
        diagnostic: TEST_DIAGNOSTIC,
        citations: [],
        preflight_report: null,
        quant_strategy_spec: null,
        token_savings: {
          local_tokens_explored: 5000,
          pack_tokens_sent: 0,
          tokens_saved: 5000,
          estimated_usd_saved: 0,
          frontier_savings_ratio_pct: 100,
          qlora_overhead_eliminated_tokens: 0,
        },
        trace_attributes: { execution_mode: "chat_qa" },
      },
    });

    const onPackReady = vi.fn();
    const onClearPack = vi.fn();
    render(
      <ScoutConsole
        selectedDomain="trading_code_scout"
        principal={TEST_ENGINEER_PRINCIPAL}
        onContextPackReady={onPackReady}
        onClearContextPack={onClearPack}
      />
    );

    fireEvent.click(screen.getByTestId("mode-rest-btn"));
    fireEvent.click(screen.getByTestId("quick-pick-zero_dollar_abstain"));
    fireEvent.click(screen.getByTestId("run-scout-btn"));

    await waitFor(() => {
      expect(
        screen.getByTestId("insufficient-context-banner")
      ).toHaveTextContent("$0 Paid Tokens Spent");
    });
    expect(
      screen.getByTestId("insufficient-context-banner")
    ).toHaveTextContent("NonExistentProprietaryEngine");
    expect(screen.getByTestId("fail-stage-4")).toBeInTheDocument();
    expect(onClearPack).toHaveBeenCalledTimes(1);
    expect(onPackReady).toHaveBeenCalledWith(
      null,
      expect.objectContaining({ pack_tokens_sent: 0 }),
      null,
      TEST_DIAGNOSTIC
    );
  });

  it("handles SSE insufficient_context error event with $0 abstention diagnostic", async () => {
    installMockApiFetch({
      sseEvents: [
        {
          event: "stage_update",
          data: {
            stage: 1,
            stage_name: "Identity, RBAC & Prompt Guard",
            status: "completed",
            details: "Passed ShieldGemma & RBAC.",
            timestamp: "2026-11-14T10:00:01Z",
          },
        },
        {
          event: "error",
          data: {
            stage: "preflight",
            status: "error",
            category: "insufficient_context",
            rationale: TEST_DIAGNOSTIC.reason,
            diagnostic: TEST_DIAGNOSTIC,
            timestamp: "2026-11-14T10:00:02Z",
          },
        },
      ],
    });

    const onPackReady = vi.fn();
    render(
      <ScoutConsole
        selectedDomain="trading_code_scout"
        principal={TEST_ENGINEER_PRINCIPAL}
        onContextPackReady={onPackReady}
      />
    );

    fireEvent.click(screen.getByTestId("run-scout-btn"));

    await waitFor(() => {
      expect(
        screen.getByTestId("insufficient-context-banner")
      ).toHaveTextContent("NonExistentProprietaryEngine");
    });
    expect(screen.getByTestId("fail-stage-4")).toBeInTheDocument();
    expect(onPackReady).toHaveBeenCalledWith(null, null, null, TEST_DIAGNOSTIC);
  });

  it("fails closed with explicit API error banner when stream returns HTTP 500", async () => {
    installMockApiFetch({
      sseStatus: 500,
      sseDetail: "Database pool exhausted",
    });

    const onPackReady = vi.fn();
    render(
      <ScoutConsole
        selectedDomain="trading_code_scout"
        principal={TEST_ENGINEER_PRINCIPAL}
        onContextPackReady={onPackReady}
      />
    );

    fireEvent.click(screen.getByTestId("run-scout-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("api-error-banner")).toHaveTextContent(
        "HTTP 500: Database pool exhausted"
      );
    });
    expect(onPackReady).not.toHaveBeenCalled();
  });

  it("supports stream cancellation and preserves custom user query when switching domains", async () => {
    installMockApiFetch({ hangSseStream: true });
    const onPackReady = vi.fn();

    const { rerender } = render(
      <ScoutConsole
        selectedDomain="trading_code_scout"
        principal={TEST_ENGINEER_PRINCIPAL}
        onContextPackReady={onPackReady}
      />
    );

    // Custom prompt must be preserved when selectedDomain changes
    const input = screen.getByTestId("scout-query-input") as HTMLTextAreaElement;
    fireEvent.change(input, {
      target: { value: "Custom user audit query across Kalman filters" },
    });

    rerender(
      <ScoutConsole
        selectedDomain="quant_research_scout"
        principal={TEST_ENGINEER_PRINCIPAL}
        onContextPackReady={onPackReady}
      />
    );
    expect(
      (screen.getByTestId("scout-query-input") as HTMLTextAreaElement).value
    ).toBe("Custom user audit query across Kalman filters");

    // Start streaming and cancel via Stop button
    fireEvent.click(screen.getByTestId("run-scout-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("cancel-stream-btn")).toBeInTheDocument();
      expect(screen.getByTestId("stage-card-2")).toHaveTextContent(
        "Searching PostgreSQL 17 RLS shards..."
      );
    });

    fireEvent.click(screen.getByTestId("cancel-stream-btn"));
    expect(screen.getByTestId("stage-card-2")).toHaveTextContent(
      "Cancelled by user."
    );
    expect(screen.getByTestId("fail-stage-2")).toBeInTheDocument();
    expect(onPackReady).not.toHaveBeenCalled();
  });
});
