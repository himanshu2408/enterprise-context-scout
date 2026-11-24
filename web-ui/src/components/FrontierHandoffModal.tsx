import React, { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  Copy,
  EyeOff,
  GitBranch,
  Loader2,
  Pin,
  ShieldCheck,
  X,
  XCircle,
} from "lucide-react";
import { applyHandoff, previewHandoff } from "../api/client";
import {
  ContextPack,
  HandoffApplyResponse,
  HandoffExportFormat,
  HandoffPreviewResponse,
  PrincipalContext,
  SliceFeedbackMap,
  SliceFeedbackStatus,
  VerbatimCodeSlice,
} from "../types/contracts";

export interface FrontierHandoffModalProps {
  isOpen: boolean;
  onClose: () => void;
  contextPack: ContextPack;
  principal: PrincipalContext;
}

type FrontierModelChoice = "gemini-3.8-flash" | "gemini-4-argon";

const EXPORT_FORMAT_OPTIONS: Array<{
  id: HandoffExportFormat;
  label: string;
  testId: string;
}> = [
  {
    id: "claude_xml",
    label: "Claude XML",
    testId: "handoff-format-claude_xml",
  },
  {
    id: "cursor_antigravity_md",
    label: "Cursor / Antigravity @-MD",
    testId: "handoff-format-cursor_antigravity_md",
  },
  {
    id: "gemini_json",
    label: "Gemini JSON",
    testId: "handoff-format-gemini_json",
  },
  {
    id: "codex_patch",
    label: "Codex Bundle",
    testId: "handoff-format-codex_patch",
  },
  {
    id: "local_llm_pack",
    label: "Local LLM Pack",
    testId: "handoff-format-local_llm_pack",
  },
];

function getSliceKey(slice: VerbatimCodeSlice, index: number): string {
  return `${slice.repo_name}:${slice.file_path}:${slice.start_line}-${slice.end_line}:${index}`;
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function buildOrderedSlices(
  slices: VerbatimCodeSlice[],
  feedback: SliceFeedbackMap
): Array<{ slice: VerbatimCodeSlice; status: SliceFeedbackStatus }> {
  const pinned: Array<{ slice: VerbatimCodeSlice; status: SliceFeedbackStatus }> = [];
  const included: Array<{ slice: VerbatimCodeSlice; status: SliceFeedbackStatus }> = [];

  slices.forEach((slice, idx) => {
    const key = getSliceKey(slice, idx);
    const status: SliceFeedbackStatus = feedback[key] ?? "included";
    if (status === "excluded") return;
    if (status === "pinned") {
      pinned.push({ slice, status });
    } else {
      included.push({ slice, status });
    }
  });

  return [...pinned, ...included];
}

function formatHandoffExport(
  format: HandoffExportFormat,
  contextPack: ContextPack,
  orderedSlices: Array<{ slice: VerbatimCodeSlice; status: SliceFeedbackStatus }>,
  patchDiff: string | null
): string {
  const slices = orderedSlices.map((item) => item.slice);
  const planLines = contextPack.architectural_execution_plan ?? [];
  const citations = contextPack.supporting_doc_citations ?? [];

  if (format === "claude_xml") {
    const sliceXml = orderedSlices
      .map(
        ({ slice, status }) =>
          `  <code_slice priority="${status}" file="${escapeXml(slice.file_path)}" lines="${slice.start_line}-${slice.end_line}" symbol="${escapeXml(slice.qualified_symbol)}" commit="${escapeXml(slice.git_commit_sha)}">\n${escapeXml(slice.verbatim_code_with_lines)}\n  </code_slice>`
      )
      .join("\n");
    const citationXml = citations
      .map(
        (c) =>
          `  <citation id="${escapeXml(c.chunk_id)}" title="${escapeXml(c.doc_title)}">${escapeXml(c.verbatim_quote)}</citation>`
      )
      .join("\n");
    return [
      `<context_pack trace_id="${escapeXml(contextPack.trace_id)}" domain="${escapeXml(contextPack.domain_id)}">`,
      `  <task>${escapeXml(contextPack.user_task)}</task>`,
      `  <execution_plan>\n${planLines.map((step) => `    <step>${escapeXml(step)}</step>`).join("\n")}\n  </execution_plan>`,
      `  <code_slices>\n${sliceXml}\n  </code_slices>`,
      `  <citations>\n${citationXml}\n  </citations>`,
      `</context_pack>`,
    ].join("\n");
  }

  if (format === "cursor_antigravity_md") {
    const refs = orderedSlices
      .map(
        ({ slice, status }) =>
          `- ${status === "pinned" ? "[PINNED] " : ""}@${slice.file_path}:L${slice.start_line}-L${slice.end_line} (\`${slice.qualified_symbol}\`)`
      )
      .join("\n");
    const codeBlocks = slices
      .map(
        (slice) =>
          `### @${slice.file_path}:L${slice.start_line}-L${slice.end_line}\n\`\`\`\n${slice.verbatim_code_with_lines}\n\`\`\``
      )
      .join("\n\n");
    return [
      `# Grounded Context Pack (${contextPack.domain_id})`,
      `**Task:** ${contextPack.user_task}`,
      `## Grounded References\n${refs || "- (no slices selected)"}`,
      `## Execution Plan\n${planLines.map((s, i) => `${i + 1}. ${s}`).join("\n")}`,
      codeBlocks,
    ]
      .filter(Boolean)
      .join("\n\n");
  }

  if (format === "gemini_json") {
    return JSON.stringify(
      {
        trace_id: contextPack.trace_id,
        domain_id: contextPack.domain_id,
        user_task: contextPack.user_task,
        architectural_execution_plan: planLines,
        code_slices: orderedSlices.map(({ slice, status }) => ({
          priority: status,
          repo_name: slice.repo_name,
          file_path: slice.file_path,
          start_line: slice.start_line,
          end_line: slice.end_line,
          qualified_symbol: slice.qualified_symbol,
          git_commit_sha: slice.git_commit_sha,
          verbatim_code_with_lines: slice.verbatim_code_with_lines,
        })),
        supporting_doc_citations: citations.map((c) => ({
          chunk_id: c.chunk_id,
          doc_title: c.doc_title,
          verbatim_quote: c.verbatim_quote,
        })),
      },
      null,
      2
    );
  }

  if (format === "codex_patch") {
    const manifest = orderedSlices
      .map(
        ({ slice, status }) =>
          `# [${status.toUpperCase()}] ${slice.file_path}:${slice.start_line}-${slice.end_line} (${slice.qualified_symbol})`
      )
      .join("\n");
    return [
      `# Codex Unified Bundle | trace=${contextPack.trace_id}`,
      `# Task: ${contextPack.user_task}`,
      manifest,
      patchDiff ? `\n${patchDiff}` : "# (Sandbox diff preview pending)",
    ].join("\n");
  }

  // local_llm_pack
  const localSlices = orderedSlices
    .map(
      ({ slice, status }) =>
        `[${status.toUpperCase()}] ${slice.file_path}:${slice.start_line}-${slice.end_line} (${slice.qualified_symbol})\n${slice.verbatim_code_with_lines}`
    )
    .join("\n---\n");
  return [
    `### Open-Weights Prompt Pack`,
    `Domain: ${contextPack.domain_id} | Trace: ${contextPack.trace_id}`,
    `Task: ${contextPack.user_task}`,
    `Plan:\n${planLines.map((s, i) => `${i + 1}. ${s}`).join("\n")}`,
    `Grounded Code Slices:\n${localSlices || "(none)"}`,
  ].join("\n\n");
}

export const FrontierHandoffModal: React.FC<FrontierHandoffModalProps> = ({
  isOpen,
  onClose,
  contextPack,
  principal,
}) => {
  const [selectedModel, setSelectedModel] =
    useState<FrontierModelChoice>("gemini-4-argon");
  const [exportFormat, setExportFormat] =
    useState<HandoffExportFormat>("claude_xml");
  const [sliceFeedback, setSliceFeedback] = useState<SliceFeedbackMap>({});
  const [copiedPack, setCopiedPack] = useState<boolean>(false);

  const [preview, setPreview] = useState<HandoffPreviewResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState<boolean>(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [confirmed, setConfirmed] = useState<boolean>(false);
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

    void previewHandoff(contextPack, selectedModel, principal)
      .then((res) => {
        if (cancelled) return;
        setPreview(res);
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

  useEffect(() => {
    setSliceFeedback({});
    setCopiedPack(false);
  }, [contextPack.trace_id]);

  const orderedSlices = useMemo(
    () => buildOrderedSlices(contextPack.code_slices, sliceFeedback),
    [contextPack.code_slices, sliceFeedback]
  );

  const feedbackCounts = useMemo(() => {
    let pinned = 0;
    let included = 0;
    let excluded = 0;
    contextPack.code_slices.forEach((slice, idx) => {
      const status = sliceFeedback[getSliceKey(slice, idx)] ?? "included";
      if (status === "pinned") pinned += 1;
      else if (status === "excluded") excluded += 1;
      else included += 1;
    });
    return { pinned, included, excluded };
  }, [contextPack.code_slices, sliceFeedback]);

  const exportText = useMemo(
    () =>
      formatHandoffExport(
        exportFormat,
        contextPack,
        orderedSlices,
        preview?.patch_diff ?? null
      ),
    [exportFormat, contextPack, orderedSlices, preview?.patch_diff]
  );

  if (!isOpen) return null;

  const toggleSliceFeedback = (
    slice: VerbatimCodeSlice,
    idx: number,
    targetStatus: "pinned" | "excluded"
  ): void => {
    const key = getSliceKey(slice, idx);
    setSliceFeedback((prev) => {
      const current = prev[key] ?? "included";
      return {
        ...prev,
        [key]: current === targetStatus ? "included" : targetStatus,
      };
    });
  };

  const handleCopyExport = async (): Promise<void> => {
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        await navigator.clipboard.writeText(exportText);
      }
      setCopiedPack(true);
      setTimeout(() => setCopiedPack(false), 1800);
    } catch {
      setCopiedPack(false);
    }
  };

  const handleApply = async (): Promise<void> => {
    if (!preview || !confirmed || applying) return;

    setApplying(true);
    setApplyError(null);

    try {
      const result = await applyHandoff(
        {
          confirmation_token: preview.confirmation_token,
          confirmed: true,
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
      data-testid="frontier-handoff-modal"
    >
      <div className="modal-container" data-testid="gemini-diff-modal">
        <div className="modal-header">
          <div className="panel-title-row">
            <GitBranch size={18} className="icon-accent" aria-hidden="true" />
            <h2 id="diff-modal-title" className="modal-title">
              Frontier Handoff &amp; Export Pack (HITL)
            </h2>
          </div>
          <button
            type="button"
            className="modal-close-btn"
            onClick={onClose}
            aria-label="Close modal"
            data-testid="modal-close-btn"
          >
            <X size={18} aria-hidden="true" />
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
          <section
            className="handoff-export-section"
            aria-labelledby="handoff-export-heading"
          >
            <div className="handoff-export-header">
              <h3 id="handoff-export-heading" className="handoff-subsection-title">
                Vendor-Neutral Context Export
              </h3>
              <button
                type="button"
                className="btn-secondary btn-compact"
                onClick={() => void handleCopyExport()}
                data-testid="copy-handoff-pack-btn"
              >
                {copiedPack ? (
                  <>
                    <Check size={13} aria-hidden="true" /> Copied Pack
                  </>
                ) : (
                  <>
                    <Copy size={13} aria-hidden="true" /> Copy Pack
                  </>
                )}
              </button>
            </div>

            <div
              className="handoff-format-tabs"
              role="tablist"
              aria-label="Handoff export formats"
            >
              {EXPORT_FORMAT_OPTIONS.map((fmt) => {
                const active = exportFormat === fmt.id;
                return (
                  <button
                    key={fmt.id}
                    type="button"
                    role="tab"
                    aria-selected={active}
                    className={`handoff-format-tab ${
                      active ? "handoff-format-tab-active" : ""
                    }`}
                    onClick={() => setExportFormat(fmt.id)}
                    data-testid={fmt.testId}
                  >
                    {fmt.label}
                  </button>
                );
              })}
            </div>

            <div className="handoff-slice-feedback-header">
              <span
                className="slice-feedback-summary"
                data-testid="slice-feedback-summary"
              >
                Pinned: {feedbackCounts.pinned} · Included:{" "}
                {feedbackCounts.included} · Excluded: {feedbackCounts.excluded}
              </span>
            </div>

            {contextPack.code_slices.length > 0 && (
              <ul
                className="handoff-slice-feedback-list"
                data-testid="handoff-slice-feedback-list"
              >
                {contextPack.code_slices.map((slice, idx) => {
                  const key = getSliceKey(slice, idx);
                  const status = sliceFeedback[key] ?? "included";
                  return (
                    <li
                      key={key}
                      className={`handoff-slice-feedback-item handoff-slice-status-${status}`}
                    >
                      <div className="handoff-slice-meta">
                        <code>
                          {slice.file_path}:{slice.start_line}-{slice.end_line}
                        </code>
                        <span className="handoff-slice-symbol">
                          {slice.qualified_symbol}
                        </span>
                      </div>
                      <div className="handoff-slice-actions">
                        <button
                          type="button"
                          className={`btn-slice-toggle ${
                            status === "pinned" ? "btn-slice-toggle-pin" : ""
                          }`}
                          aria-pressed={status === "pinned"}
                          onClick={() =>
                            toggleSliceFeedback(slice, idx, "pinned")
                          }
                          data-testid={`pin-slice-btn-${idx}`}
                        >
                          <Pin size={12} aria-hidden="true" />{" "}
                          {status === "pinned" ? "Pinned" : "Pin"}
                        </button>
                        <button
                          type="button"
                          className={`btn-slice-toggle ${
                            status === "excluded"
                              ? "btn-slice-toggle-exclude"
                              : ""
                          }`}
                          aria-pressed={status === "excluded"}
                          onClick={() =>
                            toggleSliceFeedback(slice, idx, "excluded")
                          }
                          data-testid={`exclude-slice-btn-${idx}`}
                        >
                          <EyeOff size={12} aria-hidden="true" />{" "}
                          {status === "excluded" ? "Excluded" : "Exclude"}
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}

            <pre
              className="handoff-export-preview"
              data-testid="handoff-export-preview"
            >
              {exportText}
            </pre>
          </section>

          <section
            className="handoff-sandbox-section"
            aria-labelledby="handoff-sandbox-heading"
          >
            <h3 id="handoff-sandbox-heading" className="handoff-subsection-title">
              Sandboxed Patch Preview (Read-Only Diff)
            </h3>

            {previewLoading && (
              <div className="modal-loading-state" data-testid="preview-loading">
                <Loader2 size={18} className="spin" aria-hidden="true" />
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
                <AlertTriangle size={16} aria-hidden="true" />
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
                </div>

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

                {applyResult && applyResult.success && (
                  <div
                    className="alert-banner alert-banner-safe"
                    data-testid="apply-success-view"
                  >
                    <div className="alert-banner-header">
                      <CheckCircle2 size={17} aria-hidden="true" />
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
                      <XCircle size={17} aria-hidden="true" />
                      <strong>Sandbox Patch Rejected / Failed</strong>
                    </div>
                    <p className="alert-banner-body">
                      {applyError ?? applyResult?.details}
                    </p>
                  </div>
                )}
              </>
            )}
          </section>
        </div>

        <div className="modal-footer">
          <label className="confirm-checkbox-label">
            <input
              id="confirm-apply-checkbox"
              name="confirm_apply"
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
                  <Loader2 size={15} className="spin" aria-hidden="true" /> Applying in Sandbox...
                </>
              ) : (
                <>
                  <ShieldCheck size={15} aria-hidden="true" /> Approve &amp; Apply Patch
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
