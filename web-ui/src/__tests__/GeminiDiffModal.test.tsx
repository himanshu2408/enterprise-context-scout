import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearCachedAuthSession } from "../api/client";
import { FrontierHandoffModal } from "../components/FrontierHandoffModal";
import { GeminiDiffModal } from "../components/GeminiDiffModal";
import {
  installMockApiFetch,
  TEST_CODE_PACK,
  TEST_ENGINEER_PRINCIPAL,
} from "./fixtures";

beforeEach(() => {
  clearCachedAuthSession();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("FrontierHandoffModal (and GeminiDiffModal alias)", () => {
  it("does not render when isOpen is false", () => {
    const fetchMock = installMockApiFetch();
    render(
      <FrontierHandoffModal
        isOpen={false}
        onClose={vi.fn()}
        contextPack={TEST_CODE_PACK}
        principal={TEST_ENGINEER_PRINCIPAL}
      />
    );

    expect(screen.queryByTestId("frontier-handoff-modal")).not.toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("loads live preview diff with TTL, requires human confirmation, and submits applyHandoff without custom_patch", async () => {
    const fetchMock = installMockApiFetch();

    render(
      <GeminiDiffModal
        isOpen={true}
        onClose={vi.fn()}
        contextPack={TEST_CODE_PACK}
        principal={TEST_ENGINEER_PRINCIPAL}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("unified-diff-view")).toHaveTextContent(
        "if lines <= 0:"
      );
    });

    expect(screen.getByTestId("gemini-diff-modal")).toHaveTextContent(
      "TTL 300s"
    );
    expect(screen.getByTestId("gemini-diff-modal")).toHaveTextContent(
      "slicer/bounds.py"
    );

    // Apply button disabled until human confirmation checkbox checked
    const applyBtn = screen.getByTestId("approve-apply-btn");
    expect(applyBtn).toBeDisabled();

    fireEvent.click(screen.getByTestId("confirm-apply-checkbox"));
    expect(applyBtn).toBeEnabled();
    fireEvent.click(applyBtn);

    await waitFor(() => {
      expect(screen.getByTestId("apply-success-view")).toHaveTextContent(
        "Patch Applied & Verified in Sandbox"
      );
    });

    // Apply button locks after success to prevent duplicate execution
    expect(applyBtn).toBeDisabled();

    const applyCall = fetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/api/v1/handoff/apply")
    );
    expect(applyCall).toBeDefined();
    const sentBody = JSON.parse(String(applyCall?.[1]?.body)) as {
      confirmation_token: string;
      confirmed: boolean;
      custom_patch?: string;
    };
    expect(sentBody.confirmation_token).toBe("confirm_tok_test_001");
    expect(sentBody.confirmed).toBe(true);
    expect(sentBody.custom_patch).toBeUndefined();
    expect("custom_patch" in sentBody).toBe(false);
  });

  it("supports all 5 export formats and slice Pin/Exclude feedback filtering", async () => {
    installMockApiFetch();

    render(
      <FrontierHandoffModal
        isOpen={true}
        onClose={vi.fn()}
        contextPack={TEST_CODE_PACK}
        principal={TEST_ENGINEER_PRINCIPAL}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("unified-diff-view")).toBeInTheDocument();
    });

    const preview = screen.getByTestId("handoff-export-preview");
    expect(preview).toHaveTextContent("<context_pack");

    fireEvent.click(screen.getByTestId("handoff-format-cursor_antigravity_md"));
    expect(preview).toHaveTextContent("@slicer/bounds.py:L14-L32");

    fireEvent.click(screen.getByTestId("handoff-format-gemini_json"));
    expect(preview).toHaveTextContent('"trace_id"');

    fireEvent.click(screen.getByTestId("handoff-format-codex_patch"));
    expect(preview).toHaveTextContent("# Codex Unified Bundle");

    fireEvent.click(screen.getByTestId("handoff-format-local_llm_pack"));
    expect(preview).toHaveTextContent("### Open-Weights Prompt Pack");

    // Pin slice 0 and Exclude slice 1
    fireEvent.click(screen.getByTestId("pin-slice-btn-0"));
    fireEvent.click(screen.getByTestId("exclude-slice-btn-1"));

    expect(screen.getByTestId("slice-feedback-summary")).toHaveTextContent(
      "Pinned: 1"
    );
    expect(screen.getByTestId("slice-feedback-summary")).toHaveTextContent(
      "Excluded: 1"
    );
    expect(preview).toHaveTextContent("slicer/bounds.py");
    expect(preview).not.toHaveTextContent("slicer/budget.py");
  });

  it("resets confirmation checkbox and triggers fresh preview when switching target model", async () => {
    const fetchMock = installMockApiFetch();

    render(
      <FrontierHandoffModal
        isOpen={true}
        onClose={vi.fn()}
        contextPack={TEST_CODE_PACK}
        principal={TEST_ENGINEER_PRINCIPAL}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("unified-diff-view")).toBeInTheDocument();
    });

    const checkbox = screen.getByTestId(
      "confirm-apply-checkbox"
    ) as HTMLInputElement;
    fireEvent.click(checkbox);
    expect(checkbox.checked).toBe(true);
    expect(screen.getByTestId("approve-apply-btn")).toBeEnabled();

    fireEvent.click(screen.getByTestId("select-model-flash"));

    await waitFor(() => {
      const previewCalls = fetchMock.mock.calls.filter(([url]) =>
        String(url).endsWith("/api/v1/handoff/preview")
      );
      expect(previewCalls.length).toBe(2);
    });

    expect(
      (screen.getByTestId("confirm-apply-checkbox") as HTMLInputElement).checked
    ).toBe(false);
    expect(screen.getByTestId("approve-apply-btn")).toBeDisabled();
  });

  it("displays explicit error banner when preview fails with HTTP 403 RBAC rejection", async () => {
    installMockApiFetch({ previewStatus: 403 });

    render(
      <FrontierHandoffModal
        isOpen={true}
        onClose={vi.fn()}
        contextPack={TEST_CODE_PACK}
        principal={{ ...TEST_ENGINEER_PRINCIPAL, role: "product_manager" }}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("preview-error-banner")).toHaveTextContent(
        "403"
      );
    });
    expect(screen.getByTestId("approve-apply-btn")).toBeDisabled();
  });

  it("displays explicit error banner when preview fails with HTTP 500", async () => {
    installMockApiFetch({ previewStatus: 500 });

    render(
      <FrontierHandoffModal
        isOpen={true}
        onClose={vi.fn()}
        contextPack={TEST_CODE_PACK}
        principal={TEST_ENGINEER_PRINCIPAL}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("preview-error-banner")).toHaveTextContent(
        "Preview failed (500)"
      );
    });
    expect(screen.getByTestId("approve-apply-btn")).toBeDisabled();
  });

  it("displays apply-failure-view and never shows success view when /api/v1/handoff/apply rejects", async () => {
    installMockApiFetch({ applyStatus: 500 });

    render(
      <FrontierHandoffModal
        isOpen={true}
        onClose={vi.fn()}
        contextPack={TEST_CODE_PACK}
        principal={TEST_ENGINEER_PRINCIPAL}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("unified-diff-view")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("confirm-apply-checkbox"));
    fireEvent.click(screen.getByTestId("approve-apply-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("apply-failure-view")).toHaveTextContent(
        "Apply patch failed (500)"
      );
    });
    expect(screen.queryByTestId("apply-success-view")).not.toBeInTheDocument();
  });

  it("calls onClose when clicking Reject or Close without triggering sandbox apply", async () => {
    const fetchMock = installMockApiFetch();
    const onClose = vi.fn();

    render(
      <FrontierHandoffModal
        isOpen={true}
        onClose={onClose}
        contextPack={TEST_CODE_PACK}
        principal={TEST_ENGINEER_PRINCIPAL}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("unified-diff-view")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("reject-patch-btn"));
    fireEvent.click(screen.getByTestId("modal-close-btn"));

    expect(onClose).toHaveBeenCalledTimes(2);
    const applyCalls = fetchMock.mock.calls.filter(([url]) =>
      String(url).endsWith("/api/v1/handoff/apply")
    );
    expect(applyCalls).toHaveLength(0);
  });
});
