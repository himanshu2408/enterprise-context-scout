import React, { useState, useEffect } from "react";
import {
  X,
  Sparkles,
  CheckCircle2,
  FileCode,
  ShieldCheck,
  AlertTriangle,
  RotateCw,
  GitBranch,
} from "lucide-react";
import {
  ContextPack,
  HandoffApplyResponse,
  HandoffPreviewResponse,
  PrincipalContext,
} from "../types/contracts";
import { applyHandoff, previewHandoff } from "../api/client";

export interface GeminiDiffModalProps {
  isOpen: boolean;
  onClose: () => void;
  contextPack: ContextPack | null;
  principal: PrincipalContext;
  onApplyComplete?: (response: HandoffApplyResponse) => void;
}

export const GeminiDiffModal: React.FC<GeminiDiffModalProps> = ({
  isOpen,
  onClose,
  contextPack,
  principal,
  onApplyComplete,
}) => {
  const [modelId, setModelId] = useState<"gemini-3.8-flash" | "gemini-4-argon">(
    "gemini-3.8-flash"
  );
  const [preview, setPreview] = useState<HandoffPreviewResponse | null>(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const [isApplying, setIsApplying] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [applyResult, setApplyResult] = useState<HandoffApplyResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Fetch preview when modal opens or model changes
  useEffect(() => {
    setConfirmed(false);
    if (!isOpen || !contextPack) {
      setPreview(null);
      setApplyResult(null);
      setErrorMessage(null);
      return;
    }

    let active = true;
    setIsLoadingPreview(true);
    setErrorMessage(null);

    previewHandoff(contextPack, modelId, principal)
      .then((res) => {
        if (active) {
          setPreview(res);
          setIsLoadingPreview(false);
        }
      })
      .catch((err: unknown) => {
        if (active) {
          setErrorMessage(err instanceof Error ? err.message : String(err));
          setIsLoadingPreview(false);
        }
      });

    return () => {
      active = false;
    };
  }, [isOpen, contextPack, modelId, principal]);

  if (!isOpen) return null;

  const handleApply = async (): Promise<void> => {
    if (!confirmed || !preview || isApplying) return;
    setIsApplying(true);
    setErrorMessage(null);

    try {
      const res = await applyHandoff(
        {
          confirmation_token: preview.confirmation_token,
          confirmed: true,
          patch_diff: preview.patch_diff,
        },
        principal
      );
      setApplyResult(res);
      if (onApplyComplete) {
        onApplyComplete(res);
      }
    } catch (err: unknown) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setIsApplying(false);
    }
  };

  const renderDiffLines = (diffText: string): JSX.Element[] => {
    return diffText.split("\n").map((line, idx) => {
      let lineClass = "diff-line-normal";
      if (line.startsWith("+") && !line.startsWith("+++")) {
        lineClass = "diff-line-added";
      } else if (line.startsWith("-") && !line.startsWith("---")) {
        lineClass = "diff-line-deleted";
      } else if (line.startsWith("@@")) {
        lineClass = "diff-line-header";
      }

      return (
        <div key={idx} className={`diff-line ${lineClass}`}>
          <span className="diff-line-num">{idx + 1}</span>
          <span className="diff-line-content">{line}</span>
        </div>
      );
    });
  };

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" data-testid="gemini-diff-modal">
      <div className="modal-container">
        <div className="modal-header">
          <div className="modal-title-group">
            <Sparkles className="icon icon-emerald" size={20} />
            <h3 className="modal-title font-semibold text-lg">
              1-Click Gemini Implementation Handoff
            </h3>
            <span className="badge badge-sky text-xs">Sandbox Isolation</span>
          </div>
          <button
            type="button"
            className="btn-icon"
            onClick={onClose}
            aria-label="Close modal"
            data-testid="modal-close-btn"
          >
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          {errorMessage && (
            <div className="alert-banner alert-banner-rose mb-3" role="alert">
              <AlertTriangle size={16} className="mr-2 flex-shrink-0" />
              <div className="text-sm">{errorMessage}</div>
            </div>
          )}

          {applyResult && applyResult.success ? (
            <div className="apply-success-card" data-testid="apply-success-view">
              <div className="flex-row-gap text-emerald font-semibold text-base mb-2">
                <CheckCircle2 size={20} />
                <span>Patch Verified and Applied in Git Worktree Sandbox!</span>
              </div>
              <div className="text-xs text-muted mb-3">{applyResult.details}</div>

              <div className="verification-badges-grid">
                <div className="badge badge-emerald">
                  <CheckCircle2 size={12} className="mr-1" />
                  Sandbox Verification Passed
                </div>
                <div className="badge badge-emerald">
                  <CheckCircle2 size={12} className="mr-1" />
                  Ruff / Linter Passed
                </div>
                <div className="badge badge-emerald">
                  <CheckCircle2 size={12} className="mr-1" />
                  Pytest Suite Passed
                </div>
                <div className="badge badge-sky">
                  <GitBranch size={12} className="mr-1" />
                  Worktree Isolated
                </div>
              </div>
            </div>
          ) : applyResult && !applyResult.success ? (
            <div className="apply-error-card alert-banner alert-banner-rose mb-3" data-testid="apply-failure-view">
              <AlertTriangle size={18} className="mr-2 flex-shrink-0" />
              <div>
                <div className="font-semibold">Patch Application Failed</div>
                <div className="text-xs">{applyResult.details}</div>
              </div>
            </div>
          ) : (
            <>
              <div className="model-selection-row">
                <label className="text-xs font-semibold text-muted">Frontier Paid Target Model:</label>
                <div className="flex-row-gap">
                  <button
                    type="button"
                    data-testid="select-model-flash"
                    className={`btn-pill ${modelId === "gemini-3.8-flash" ? "btn-pill-active" : ""}`}
                    onClick={() => {
                      setModelId("gemini-3.8-flash");
                      setConfirmed(false);
                    }}
                  >
                    gemini-3.8-flash ($0.75/1M)
                  </button>
                  <button
                    type="button"
                    data-testid="select-model-argon"
                    className={`btn-pill ${modelId === "gemini-4-argon" ? "btn-pill-active" : ""}`}
                    onClick={() => {
                      setModelId("gemini-4-argon");
                      setConfirmed(false);
                    }}
                  >
                    gemini-4-argon ($2.00/1M)
                  </button>
                </div>
              </div>

              {isLoadingPreview ? (
                <div className="loading-box" data-testid="preview-loading">
                  <RotateCw size={24} className="animate-spin icon-sky mb-2" />
                  <div className="text-sm text-muted">
                    Generating unified git diff via {modelId}...
                  </div>
                </div>
              ) : preview ? (
                <div className="diff-view-section">
                  <div className="diff-metadata-bar">
                    <div className="flex-row-gap text-xs">
                      <FileCode size={14} className="icon-sky" />
                      <span className="font-semibold text-zinc-300">
                        Target Files: {preview.target_files.join(", ")}
                      </span>
                    </div>

                    <div className="flex-row-gap text-xs">
                      <span className="badge badge-zinc">
                        Tokens Used: {preview.pack_token_count ?? preview.tokens_used ?? 0}
                      </span>
                      <span className="badge badge-emerald">
                        Cost: ${(preview.estimated_paid_cost_usd ?? preview.estimated_cost_usd ?? 0).toFixed(4)} USD
                      </span>
                      <span className="badge badge-purple font-mono">
                        {preview.confirmation_token.substring(0, 16)}...
                      </span>
                    </div>
                  </div>

                  <div className="diff-container" data-testid="unified-diff-view">
                    {renderDiffLines(preview.patch_diff)}
                  </div>

                  <div className="confirmation-gate-card">
                    <div className="flex-row-gap">
                      <ShieldCheck size={18} className="icon-sky flex-shrink-0" />
                      <div>
                        <div className="font-semibold text-sm text-zinc-200">
                          Human-in-the-Loop Confirmation Gate (SEC-04)
                        </div>
                        <div className="text-xs text-muted">
                          Explicit confirmation is required before any patch can be written to the sandbox repository.
                        </div>
                      </div>
                    </div>

                    <label className="checkbox-label mt-2" htmlFor="human-confirm-checkbox">
                      <input
                        id="human-confirm-checkbox"
                        data-testid="confirm-apply-checkbox"
                        type="checkbox"
                        className="checkbox-input"
                        checked={confirmed}
                        onChange={(e) => setConfirmed(e.target.checked)}
                      />
                      <span className="text-xs font-semibold text-zinc-300">
                        I confirm and authorize applying this patch in GitWorktreeSandbox
                      </span>
                    </label>
                  </div>
                </div>
              ) : null}
            </>
          )}
        </div>

        <div className="modal-footer">
          <button
            type="button"
            data-testid="reject-patch-btn"
            className="btn btn-secondary"
            onClick={onClose}
          >
            {applyResult ? "Close" : "Reject"}
          </button>

          {!applyResult && (
            <button
              type="button"
              data-testid="approve-apply-btn"
              className="btn btn-emerald"
              disabled={!confirmed || isApplying || isLoadingPreview || !preview}
              onClick={handleApply}
            >
              {isApplying ? (
                <>
                  <RotateCw size={14} className="animate-spin mr-1" />
                  Applying in Sandbox...
                </>
              ) : (
                <>
                  <CheckCircle2 size={14} className="mr-1" />
                  Approve & Apply in Git Worktree Sandbox
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
