import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { GeminiDiffModal } from "../components/GeminiDiffModal";
import { MOCK_CONTEXT_PACK } from "../api/client";
import { PrincipalContext } from "../types/contracts";
import { ROLE_DEFAULT_SCOPES } from "../components/RoleSwitcher";

describe("GeminiDiffModal Component", () => {
  const principal: PrincipalContext = {
    user_id: "demo_engineer",
    role: "engineer",
    allowed_scopes: ROLE_DEFAULT_SCOPES.engineer,
    max_session_paid_tokens: 16000,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does not render when isOpen is false", () => {
    const onClose = vi.fn();
    render(
      <GeminiDiffModal
        isOpen={false}
        onClose={onClose}
        contextPack={MOCK_CONTEXT_PACK}
        principal={principal}
      />
    );

    expect(screen.queryByTestId("gemini-diff-modal")).not.toBeInTheDocument();
  });

  it("renders preview diff and keeps Approve button disabled until human confirms", async () => {
    const onClose = vi.fn();
    render(
      <GeminiDiffModal
        isOpen={true}
        onClose={onClose}
        contextPack={MOCK_CONTEXT_PACK}
        principal={principal}
      />
    );

    expect(screen.getByTestId("gemini-diff-modal")).toBeInTheDocument();

    // Wait for preview diff to load
    await waitFor(() => {
      expect(screen.getByTestId("unified-diff-view")).toBeInTheDocument();
    });

    expect(screen.getByText("Target Files: src/execution/slippage.py")).toBeInTheDocument();

    // Verify Approve button is disabled before checking checkbox
    const approveBtn = screen.getByTestId("approve-apply-btn");
    expect(approveBtn).toBeDisabled();

    // Check confirmation checkbox
    const confirmCheckbox = screen.getByTestId("confirm-apply-checkbox");
    fireEvent.click(confirmCheckbox);

    // Verify Approve button is now enabled
    expect(approveBtn).not.toBeDisabled();

    // Uncheck and verify it is disabled again
    fireEvent.click(confirmCheckbox);
    expect(approveBtn).toBeDisabled();
  });

  it("applies patch to sandbox after human confirmation and displays verification success", async () => {
    const onClose = vi.fn();
    const onApplyComplete = vi.fn();

    vi.spyOn(globalThis, "fetch").mockImplementation(async (url: string | URL | Request) => {
      const urlStr = typeof url === "string" ? url : url.toString();
      if (urlStr.includes("/api/v1/handoff/apply")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            success: true,
            confirmed: true,
            target_files: ["src/execution/slippage.py"],
            lint_passed: true,
            tests_passed: true,
            verification_passed: true,
            details: "Patch successfully verified and applied in isolated GitWorktreeSandbox.",
            telemetry: {
              execution_id: "exec-123",
              duration_ms: 45,
              cpu_pct: 10.0,
              memory_mb: 128,
              exit_code: 0,
            },
          }),
        } as Response;
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({
          confirmation_token: "conf-token-e9182ab47f",
          patch_diff: `--- a/src/execution/slippage.py\n+++ b/src/execution/slippage.py\n@@ -42,4 +42,7 @@\n def calculate_slippage_bps(order_size: float, adv_pct: float) -> float:\n-    impact_bps = 5.0 + 12.5 * math.sqrt(adv_pct)\n+    # Calibrated square-root impact with regulatory market regime minimum floor\n+    base_floor = 5.0\n+    impact_bps = base_floor + 12.5 * math.sqrt(max(adv_pct, 0.0001))\n     return min(impact_bps, 150.0)`,
          target_files: ["src/execution/slippage.py"],
          estimated_paid_cost_usd: 0.0014,
          pack_token_count: 348,
          estimated_cost_usd: 0.0014,
          tokens_used: 348,
          model_id: "gemini-3.8-flash",
          requires_human_confirmation: true,
        }),
      } as Response;
    });

    render(
      <GeminiDiffModal
        isOpen={true}
        onClose={onClose}
        contextPack={MOCK_CONTEXT_PACK}
        principal={principal}
        onApplyComplete={onApplyComplete}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("unified-diff-view")).toBeInTheDocument();
    });

    // Check confirmation checkbox
    const confirmCheckbox = screen.getByTestId("confirm-apply-checkbox");
    fireEvent.click(confirmCheckbox);

    const approveBtn = screen.getByTestId("approve-apply-btn");
    fireEvent.click(approveBtn);

    await waitFor(() => {
      expect(screen.getByTestId("apply-success-view")).toBeInTheDocument();
      expect(
        screen.getByText("Patch Verified and Applied in Git Worktree Sandbox!")
      ).toBeInTheDocument();
      expect(screen.getByText("Sandbox Verification Passed")).toBeInTheDocument();
      expect(screen.getByText("Ruff / Linter Passed")).toBeInTheDocument();
      expect(screen.getByText("Pytest Suite Passed")).toBeInTheDocument();
    });

    expect(onApplyComplete).toHaveBeenCalledTimes(1);
  });

  it("displays error message and never shows success view when backend rejects apply", async () => {
    const onClose = vi.fn();
    const onApplyComplete = vi.fn();

    vi.spyOn(globalThis, "fetch").mockImplementation(async (url: string | URL | Request) => {
      const urlStr = typeof url === "string" ? url : url.toString();
      if (urlStr.includes("/api/v1/handoff/apply")) {
        return {
          ok: false,
          status: 400,
          text: async () => "Invalid confirmation token or expired token",
        } as Response;
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({
          confirmation_token: "conf-token-e9182ab47f",
          patch_diff: "--- a/file.py\n+++ b/file.py",
          target_files: ["file.py"],
          estimated_paid_cost_usd: 0.001,
          pack_token_count: 100,
          model_id: "gemini-3.8-flash",
          requires_human_confirmation: true,
        }),
      } as Response;
    });

    render(
      <GeminiDiffModal
        isOpen={true}
        onClose={onClose}
        contextPack={MOCK_CONTEXT_PACK}
        principal={principal}
        onApplyComplete={onApplyComplete}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("unified-diff-view")).toBeInTheDocument();
    });

    const confirmCheckbox = screen.getByTestId("confirm-apply-checkbox");
    fireEvent.click(confirmCheckbox);

    const approveBtn = screen.getByTestId("approve-apply-btn");
    fireEvent.click(approveBtn);

    await waitFor(() => {
      expect(screen.getByTestId("apply-failure-view")).toBeInTheDocument();
      expect(screen.queryByTestId("apply-success-view")).not.toBeInTheDocument();
      expect(
        screen.getByText("Apply patch failed (400): Invalid confirmation token or expired token")
      ).toBeInTheDocument();
    });
  });

  it("resets confirmation checkbox when switching target model", async () => {
    const onClose = vi.fn();
    render(
      <GeminiDiffModal
        isOpen={true}
        onClose={onClose}
        contextPack={MOCK_CONTEXT_PACK}
        principal={principal}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("unified-diff-view")).toBeInTheDocument();
    });

    const confirmCheckbox = screen.getByTestId("confirm-apply-checkbox");
    fireEvent.click(confirmCheckbox);
    expect(confirmCheckbox).toBeChecked();
    expect(screen.getByTestId("approve-apply-btn")).not.toBeDisabled();

    // Switch model to Argon
    const argonBtn = screen.getByTestId("select-model-argon");
    fireEvent.click(argonBtn);

    // Expect confirmation to be reset to unchecked and approve button to be disabled
    await waitFor(() => {
      expect(confirmCheckbox).not.toBeChecked();
      expect(screen.getByTestId("approve-apply-btn")).toBeDisabled();
    });
  });

  it("calls onClose when clicking Reject or Close button", async () => {
    const onClose = vi.fn();

    render(
      <GeminiDiffModal
        isOpen={true}
        onClose={onClose}
        contextPack={MOCK_CONTEXT_PACK}
        principal={principal}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("unified-diff-view")).toBeInTheDocument();
    });

    const rejectBtn = screen.getByTestId("reject-patch-btn");
    fireEvent.click(rejectBtn);

    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
