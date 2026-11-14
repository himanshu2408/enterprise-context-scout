import React, { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Edit3,
  GitBranch,
  Loader2,
  ShieldCheck,
  X,
  XCircle,
} from "lucide-react";
import { applyHandoff, previewHandoff } from "../api/client";
import {
  ContextPack,
  HandoffApplyResponse,
  HandoffPreviewResponse,
  PrincipalContext,
} from "../types/contracts";

export interface GeminiDiffModalProps {
  isOpen: boolean;
  onClose: () => void;
  contextPack: ContextPack;
  principal: PrincipalContext;
}

type FrontierModelChoice = "gemini-3.8-flash" | "gemini-4-argon";

export const GeminiDiffModal: React.FC<GeminiDiffModalProps> = ({
  isOpen,
  onClose,
  contextPack,
  principal,
}) => {
  const [selectedModel, setSelectedModel] =
    useState<FrontierModelChoice>("gemini-4-argon");
  const [preview, setPreview] = useState<HandoffPreviewResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState<boolean>(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [confirmed, setConfirmed] = useState<boolean>(false);
  const [editMode, setEditMode] = useState<boolean>(false);
  const [editedDiff, setEditedDiff] = useState<string>("");
  const [applying, setApplying] = useState<boolean>(false);
  const [applyResult, setApplyResult] = useState<HandoffApplyResponse | null>(
    null
  );
  const [applyError, setApplyError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;

    let cancelled = false;
    setPreviewLoading(true);
    setPreviewError(null);
    setApplyResult(null);
    setApplyError(null);
    setConfirmed(false);
    setEditMode(false);

    void previewHandoff(contextPack, selectedModel, principal)
      .then((res) => {
        if (cancelled) return;
        setPreview(res);
        setEditedDiff(res.patch_diff);
        setPreviewLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setPreview(null);
        setPreviewLoading(false);
        setPreviewError(
          err instanceof Error ? err.message : "Failed to load diff preview."
        );
      });

    return () => {
      cancelled = true;
    };
  }, [isOpen, contextPack, selectedModel, principal]);

  if (!isOpen) return null;

  const handleApply = async (): Promise<void> => {
    if (!preview || !confirmed || applying) return;

    setApplying(true);
    setApplyError(null);

    try {
      const customPatch =
        editMode && editedDiff !== preview.patch_diff ? editedDiff : undefined;
      const result = await applyHandoff(
        {
          confirmation_token: preview.confirmation_token,
          confirmed: true,
          custom_patch: customPatch,
        },
        principal
      );
      setApplyResult(result);
    } catch (err: unknown) {
      setApplyError(
        err instanceof Error ? err.message : "Patch application failed."
      );
    } finally {
      setApplying(false);
    }
  };

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-labelledby="diff-modal-title"
      data-testid="gemini-diff-modal"
    >
      <div className="modal-container">
        <div className="modal-header">
          <div className="panel-title-row">
            <GitBranch size={18} className="icon-accent" />
            <h2 id="diff-modal-title" className="modal-title">
              Sandboxed Frontier Patch Handoff (HITL)
            </h2>
          </div>
          <button
            type="button"
            className="modal-close-btn"
            onClick={onClose}
            aria-label="Close modal"
            data-testid="modal-close-btn"
          >
            <X size={18} />
          </button>
        </div>

        <div className="modal-model-bar">
          <span className="field-label">Frontier Executor Model:</span>
          <div className="model-btn-group">
            <button
              type="button"
              className={`model-choice-btn ${
                selectedModel === "gemini-3.8-flash" ? "model-choice-active" : ""
              }`}
              onClick={() => setSelectedModel("gemini-3.8-flash")}
              data-testid="select-model-flash"
              disabled={applying}
            >
              Gemini 3.8 Flash
            </button>
            <button
              type="button"
              className={`model-choice-btn ${
                selectedModel === "gemini-4-argon" ? "model-choice-active" : ""
              }`}
              onClick={() => setSelectedModel("gemini-4-argon")}
              data-testid="select-model-argon"
              disabled={applying}
            >
              Gemini 4 Argon
            </button>
          </div>
        </div>

        <div className="modal-body">
          {previewLoading && (
            <div className="modal-loading-state" data-testid="preview-loading">
              <Loader2 size={18} className="spin" />
              <span>
                Generating grounded patch preview with {selectedModel}...
              </span>
            </div>
          )}

          {previewError && (
            <div
              className="alert-banner alert-banner-danger"
              data-testid="preview-error-banner"
              role="alert"
            >
              <AlertTriangle size={16} />
              <span>{previewError}</span>
            </div>
          )}

          {!previewLoading && preview && (
            <>
              <div className="handoff-cost-strip">
                <div className="stat-box">
                  <span className="stat-label">Est. Frontier Cost</span>
                  <span className="stat-value">
                    ${preview.estimated_paid_cost_usd.toFixed(4)}
                  </span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">Cost Range (L/E/H)</span>
                  <span className="stat-value">
                    {preview.cost_low_usd !== undefined &&
                    preview.cost_low_usd !== null &&
                    preview.cost_high_usd !== undefined &&
                    preview.cost_high_usd !== null
                      ? `$${preview.cost_low_usd.toFixed(4)}–$${preview.cost_high_usd.toFixed(4)}`
                      : `$${preview.estimated_paid_cost_usd.toFixed(4)}`}
                  </span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">Est. Local Savings</span>
                  <span className="stat-value stat-value-accent">
                    ${preview.estimated_paid_savings_usd.toFixed(4)}
                  </span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">Target Files</span>
                  <span className="stat-value">
                    {preview.target_files.join(", ") || "None"}
                  </span>
                </div>
              </div>

              <div className="diff-toolbar">
                <span>
                  Confirmation Token:{" "}
                  <code>{preview.confirmation_token.slice(0, 14)}...</code> ·
                  TTL {preview.expires_in_seconds}s
                </span>
                <button
                  type="button"
                  className="link-action"
                  onClick={() => setEditMode((v) => !v)}
                  data-testid="toggle-edit-patch-btn"
                >
                  <Edit3 size={12} />{" "}
                  {editMode ? "View Read-Only Diff" : "Edit Patch Before Apply"}
                </button>
              </div>

              {editMode ? (
                <textarea
                  className="diff-editor-textarea"
                  rows={12}
                  value={editedDiff}
                  onChange={(e) => setEditedDiff(e.target.value)}
                  data-testid="custom-patch-textarea"
                />
              ) : (
                <pre className="unified-diff-pre" data-testid="unified-diff-view">
                  {preview.patch_diff.split("\n").map((line, i) => {
                    const cls = line.startsWith("+")
                      ? "diff-line-add"
                      : line.startsWith("-")
                        ? "diff-line-del"
                        : line.startsWith("@@")
                          ? "diff-line-hunk"
                          : "diff-line-ctx";
                    return (
                      <div key={i} className={cls}>
                        {line || " "}
                      </div>
                    );
                  })}
                </pre>
              )}

              {applyResult && applyResult.success && (
                <div
                  className="alert-banner alert-banner-safe"
                  data-testid="apply-success-view"
                >
                  <div className="alert-banner-header">
                    <CheckCircle2 size={17} />
                    <strong>Patch Applied &amp; Verified in Sandbox</strong>
                  </div>
                  <p className="alert-banner-body">{applyResult.details}</p>
                  <div className="apply-checks-row">
                    <span>
                      Lint: {applyResult.lint_passed ? "Passed" : "Failed"}
                    </span>
                    <span>
                      Tests: {applyResult.tests_passed ? "Passed" : "Failed"}
                    </span>
                    <span>
                      Verification:{" "}
                      {applyResult.verification_passed ? "Passed" : "Failed"}
                    </span>
                  </div>
                </div>
              )}

              {(applyError || (applyResult && !applyResult.success)) && (
                <div
                  className="alert-banner alert-banner-danger"
                  data-testid="apply-failure-view"
                  role="alert"
                >
                  <div className="alert-banner-header">
                    <XCircle size={17} />
                    <strong>Sandbox Patch Rejected / Failed</strong>
                  </div>
                  <p className="alert-banner-body">
                    {applyError ?? applyResult?.details}
                  </p>
                </div>
              )}
            </>
          )}
        </div>

        <div className="modal-footer">
          <label className="confirm-checkbox-label">
            <input
              type="checkbox"
              checked={confirmed}
              onChange={(e) => setConfirmed(e.target.checked)}
              disabled={!preview || applying || Boolean(applyResult?.success)}
              data-testid="confirm-apply-checkbox"
            />
            <span>
              I reviewed the unified diff &amp; target files and explicitly
              approve sandbox execution.
            </span>
          </label>

          <div className="modal-actions">
            <button
              type="button"
              className="btn-secondary"
              onClick={onClose}
              data-testid="reject-patch-btn"
            >
              Reject / Cancel
            </button>
            <button
              type="button"
              className="btn-primary"
              disabled={
                !preview ||
                !confirmed ||
                applying ||
                Boolean(applyResult?.success)
              }
              onClick={() => void handleApply()}
              data-testid="approve-apply-btn"
            >
              {applying ? (
                <>
                  <Loader2 size={15} className="spin" /> Applying in Sandbox...
                </>
              ) : (
                <>
                  <ShieldCheck size={15} /> Approve &amp; Apply Patch
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
