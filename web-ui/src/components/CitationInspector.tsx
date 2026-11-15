import React, { useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  Code,
  FileCode,
  FileText,
  GitBranch,
  Lock,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import {
  ContextPack,
  DocCitation,
  InsufficientContextDiagnostic,
  QuantStrategyBrief,
  RoleType,
  ScoutTokenSavingsBreakdown,
  SourceFilterType,
} from "../types/contracts";

export interface CitationInspectorProps {
  contextPack: ContextPack | null;
  quantSpec: QuantStrategyBrief | null;
  tokenSavings: ScoutTokenSavingsBreakdown | null;
  principalRole: RoleType;
  sourceFilter?: SourceFilterType;
  onOpenDiffModal: () => void;
  abstainedDiagnostic?: InsufficientContextDiagnostic | null;
}

type InspectorTab =
  | "code-slices"
  | "quant"
  | "preflight-report"
  | "telemetry-roi";

function matchesCitationSourceFilter(
  cite: DocCitation,
  sourceFilter: SourceFilterType
): boolean {
  if (sourceFilter === "all") return true;
  const srcType = (cite.source_type ?? "").toLowerCase();
  const isJira =
    srcType === "jira" ||
    /jira/i.test(cite.source_uri) ||
    Boolean(cite.ticket_key);
  if (sourceFilter === "jira") {
    return isJira;
  }
  if (sourceFilter === "wiki") {
    return !isJira;
  }
  if (sourceFilter === "code") {
    return (
      srcType === "code" ||
      srcType === "github" ||
      srcType === "gitlab" ||
      srcType === "local_git"
    );
  }
  return true;
}

function formatCitationVersionOrDate(cite: DocCitation): string {
  if (cite.published_date && cite.version) {
    return `${cite.version} · ${cite.published_date}`;
  }
  if (cite.published_date) {
    return `Updated ${cite.published_date}`;
  }
  return cite.version || "v1.0";
}

export const CitationInspector: React.FC<CitationInspectorProps> = ({
  contextPack,
  quantSpec,
  tokenSavings,
  principalRole,
  sourceFilter = "all",
  onOpenDiffModal,
  abstainedDiagnostic = null,
}) => {
  const [activeTab, setActiveTab] = useState<InspectorTab>("code-slices");
  const [highlightedSliceIdx, setHighlightedSliceIdx] = useState<number | null>(
    null
  );
  const [highlightedChunkId, setHighlightedChunkId] = useState<string | null>(
    null
  );

  const sliceRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const citationRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const mergedCitations = useMemo<DocCitation[]>(() => {
    if (!contextPack) return [];
    const seen = new Set<string>();
    const combined: DocCitation[] = [];
    const sources = [
      ...contextPack.supporting_doc_citations,
      ...(contextPack.domain_knowledge_brief?.verified_citations ?? []),
      ...(quantSpec?.verified_citations ?? []),
    ];
    for (const cite of sources) {
      if (!seen.has(cite.chunk_id)) {
        seen.add(cite.chunk_id);
        combined.push(cite);
      }
    }
    return combined;
  }, [contextPack, quantSpec]);

  const filteredSlices = useMemo(() => {
    if (!contextPack) return [];
    if (sourceFilter === "wiki" || sourceFilter === "jira") {
      return [];
    }
    return contextPack.code_slices;
  }, [contextPack, sourceFilter]);

  const filteredCitations = useMemo(
    () =>
      mergedCitations.filter((cite) =>
        matchesCitationSourceFilter(cite, sourceFilter)
      ),
    [mergedCitations, sourceFilter]
  );

  const effectiveTab: InspectorTab = useMemo(() => {
    if (!contextPack) return activeTab;
    if (
      activeTab === "code-slices" &&
      filteredSlices.length === 0 &&
      (quantSpec !== null || filteredCitations.length > 0 || sourceFilter === "wiki" || sourceFilter === "jira")
    ) {
      return "quant";
    }
    if (
      activeTab === "quant" &&
      sourceFilter === "code" &&
      filteredCitations.length === 0 &&
      filteredSlices.length > 0
    ) {
      return "code-slices";
    }
    return activeTab;
  }, [
    activeTab,
    contextPack,
    filteredSlices.length,
    filteredCitations.length,
    quantSpec,
    sourceFilter,
  ]);

  const handleJumpToCodeSlice = (idx: number): void => {
    setActiveTab("code-slices");
    setHighlightedSliceIdx(idx);
    setHighlightedChunkId(null);
    window.setTimeout(() => {
      const target = sliceRefs.current[idx];
      if (target && typeof target.scrollIntoView === "function") {
        target.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }, 0);
  };

  const handleJumpToCitation = (chunkId: string): void => {
    setActiveTab("quant");
    setHighlightedChunkId(chunkId);
    setHighlightedSliceIdx(null);
    window.setTimeout(() => {
      const target = citationRefs.current[chunkId];
      if (target && typeof target.scrollIntoView === "function") {
        target.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }, 0);
  };

  if (!contextPack) {
    return (
      <section
        className="panel inspector-empty-panel"
        data-testid="citation-inspector"
      >
        <div className="empty-state-content">
          <Sparkles size={28} className="icon-accent" />
          <h3>Grounded Answer Workspace</h3>
          <p>
            {abstainedDiagnostic
              ? "The local scout abstained safely because verified evidence was insufficient inside your authorized scopes."
              : "Select your persona, RBAC scopes, and domain, then ask a question above. Local Gemma 4 + EmbeddingGemma 2 will synthesize an Answer-First brief backed by verbatim AST code slices and research citations."}
          </p>
        </div>
      </section>
    );
  }

  const brief = contextPack.domain_knowledge_brief ?? null;
  const resolvedQuant = quantSpec ?? contextPack.quant_strategy_spec ?? null;
  const conflicts = contextPack.doc_vs_code_conflicts_detected;
  const staleCitations = filteredCitations.filter(
    (cite) => cite.is_stale === true || Boolean(cite.stale_reason)
  );
  const canOpenPatchModal =
    principalRole === "engineer" && contextPack.code_slices.length > 0;
  const citationsTabLabel = resolvedQuant
    ? `Strategy & Citations (${filteredCitations.length})`
    : `Citations & Docs (${filteredCitations.length})`;

  const answerHeadline =
    brief?.brief_title ??
    resolvedQuant?.strategy_name ??
    "Grounded Engineering & Knowledge Answer";

  const directSummary =
    brief?.executive_summary ??
    (contextPack.architectural_execution_plan.length > 0
      ? contextPack.architectural_execution_plan.join(" ")
      : `Synthesized ${contextPack.code_slices.length} AST code slices and ${mergedCitations.length} verified document citations for "${contextPack.user_task}".`);

  return (
    <section
      className="panel inspector-panel"
      data-testid="citation-inspector"
    >
      {conflicts.length > 0 && (
        <div
          className="conflicts-alert-banner"
          data-testid="conflicts-banner"
          role="alert"
        >
          <div className="conflicts-header">
            <AlertTriangle size={17} />
            <strong>
              Code vs. Documentation Conflict Detected ({conflicts.length})
            </strong>
          </div>
          <ul className="conflicts-list">
            {conflicts.map((conflict, idx) => (
              <li key={idx}>{conflict}</li>
            ))}
          </ul>
        </div>
      )}

      {staleCitations.length > 0 && (
        <div
          className="stale-docs-alert-banner"
          data-testid="stale-docs-warning-banner"
          role="alert"
        >
          <div className="conflicts-header">
            <AlertTriangle size={16} />
            <strong>
              Outdated / Stale Document Warning ({staleCitations.length})
            </strong>
          </div>
          <ul className="conflicts-list">
            {staleCitations.map((cite) => (
              <li key={`stale-${cite.chunk_id}`}>
                <strong>{cite.doc_title}</strong>
                {cite.published_date ? ` (${cite.published_date})` : ""}:{" "}
                {cite.stale_reason ??
                  "Flagged as stale or superseded by newer documentation."}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="answer-first-hero" data-testid="answer-first-hero">
        <div className="hero-top-meta">
          <div className="hero-badges-left">
            <span
              className={`badge ${
                contextPack.preflight_report.passed
                  ? "badge-success"
                  : "badge-warning"
              }`}
              data-testid="preflight-passed-badge"
            >
              <CheckCircle2 size={12} />
              {contextPack.preflight_report.passed
                ? "100% Grounded & Verified"
                : "Preflight Review Flagged"}
            </span>
            <span className="badge badge-neutral">
              Domain: {contextPack.domain_id}
            </span>
            <span className="badge badge-neutral">
              Trace: <code>{contextPack.trace_id}</code>
            </span>
          </div>
          <span className="token-pack-pill">
            {contextPack.pack_token_count.toLocaleString()} curated pack tokens
          </span>
        </div>

        <h2 className="hero-title">{answerHeadline}</h2>
        <p className="hero-summary">{directSummary}</p>

        {Boolean(brief?.formal_expression_or_rule) && (
          <div className="formal-rule-box">
            <span className="formal-label">Formal Rule / Expression:</span>
            <code>{brief?.formal_expression_or_rule}</code>
          </div>
        )}

        {((brief?.core_findings_and_rules.length ?? 0) > 0 ||
          contextPack.architectural_execution_plan.length > 0) && (
          <div className="hero-columns">
            <div className="hero-block">
              <h4>Key Findings &amp; Execution Plan</h4>
              <ul>
                {((brief?.core_findings_and_rules.length ?? 0) > 0
                  ? brief?.core_findings_and_rules ?? []
                  : contextPack.architectural_execution_plan
                ).map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>

            {brief !== null &&
              (brief.domain_constraints_and_caveats.length > 0 ||
                brief.known_risks_and_failure_modes.length > 0) && (
                <div className="hero-block">
                  <h4>Constraints &amp; Known Risks</h4>
                  <ul>
                    {brief.domain_constraints_and_caveats.map((item, idx) => (
                      <li key={`c-${idx}`}>{item}</li>
                    ))}
                    {brief.known_risks_and_failure_modes.map((item, idx) => (
                      <li key={`r-${idx}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
          </div>
        )}

        {(filteredSlices.length > 0 || filteredCitations.length > 0) && (
          <div className="inline-citations-bar">
            <span className="inline-citations-label">Verified Citations:</span>
            <div className="inline-citations-chips">
              {filteredSlices.map((slice, idx) => (
                <button
                  key={`${slice.file_path}-${idx}`}
                  type="button"
                  className="cite-chip cite-chip-code"
                  onClick={() => handleJumpToCodeSlice(idx)}
                  data-testid={`inline-cite-code-${idx}`}
                  title={slice.qualified_symbol}
                >
                  <FileCode size={12} />
                  <code>
                    [{slice.repo_name}@{slice.git_commit_sha.slice(0, 7)}:
                    {slice.file_path}:L{slice.start_line}-L{slice.end_line}]
                  </code>
                </button>
              ))}
              {filteredCitations.map((cite) => {
                const isStale =
                  cite.is_stale === true || Boolean(cite.stale_reason);
                return (
                  <button
                    key={cite.chunk_id}
                    type="button"
                    className={`cite-chip cite-chip-doc ${
                      isStale ? "cite-chip-stale" : ""
                    }`}
                    onClick={() => handleJumpToCitation(cite.chunk_id)}
                    data-testid={`inline-cite-doc-${cite.chunk_id}`}
                    title={cite.stale_reason ?? cite.source_uri}
                  >
                    {isStale ? (
                      <AlertTriangle size={12} />
                    ) : (
                      <FileText size={12} />
                    )}
                    <span>
                      [{cite.doc_title} · {cite.section_heading} ·{" "}
                      {formatCitationVersionOrDate(cite)}]
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <div className="hero-action-footer">
          {canOpenPatchModal ? (
            <button
              type="button"
              className="btn-primary"
              onClick={onOpenDiffModal}
              data-testid="open-diff-modal-btn"
            >
              <GitBranch size={15} /> Review &amp; Apply Patch in Sandbox
            </button>
          ) : (
            <div
              className="readonly-handoff-note"
              data-testid="handoff-readonly-notice"
            >
              <Lock size={14} />
              <span>
                {principalRole !== "engineer"
                  ? `Read-Only Role (${principalRole}): Frontier Patch Handoff requires Engineer role.`
                  : "No mutable code slices in this pack for sandbox patching."}
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="inspector-tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={effectiveTab === "code-slices"}
          className={`tab-btn ${
            effectiveTab === "code-slices" ? "tab-btn-active" : ""
          }`}
          onClick={() => setActiveTab("code-slices")}
          data-testid="tab-code-slices"
        >
          <Code size={14} /> Code Slices ({filteredSlices.length})
        </button>

        <button
          type="button"
          role="tab"
          aria-selected={effectiveTab === "quant"}
          className={`tab-btn ${
            effectiveTab === "quant" ? "tab-btn-active" : ""
          }`}
          onClick={() => setActiveTab("quant")}
          data-testid="tab-quant-strategy"
        >
          <BookOpen size={14} /> {citationsTabLabel}
        </button>

        <button
          type="button"
          role="tab"
          aria-selected={effectiveTab === "preflight-report"}
          className={`tab-btn ${
            effectiveTab === "preflight-report" ? "tab-btn-active" : ""
          }`}
          onClick={() => setActiveTab("preflight-report")}
          data-testid="tab-preflight-report"
        >
          <ShieldCheck size={14} /> Pre-Flight Gate
        </button>

        <button
          type="button"
          role="tab"
          aria-selected={effectiveTab === "telemetry-roi"}
          className={`tab-btn ${
            effectiveTab === "telemetry-roi" ? "tab-btn-active" : ""
          }`}
          onClick={() => setActiveTab("telemetry-roi")}
          data-testid="tab-telemetry-roi"
        >
          Telemetry &amp; Pack Tokens
        </button>
      </div>

      <div className="inspector-tab-body">
        {effectiveTab === "code-slices" && (
          <div className="evidence-list" data-testid="code-slices-view">
            {filteredSlices.length === 0 ? (
              <p className="empty-tab-note">
                {sourceFilter === "wiki" || sourceFilter === "jira"
                  ? `Code slices hidden under active "${sourceFilter}" source filter. Switch to Citations & Docs or select All Federated.`
                  : "No source code slices returned for this query. Switch to Citations & Docs to inspect grounded document quotes."}
              </p>
            ) : (
              filteredSlices.map((slice, idx) => (
                <div
                  key={`${slice.file_path}-${idx}`}
                  ref={(el) => {
                    sliceRefs.current[idx] = el;
                  }}
                  className={`evidence-card ${
                    highlightedSliceIdx === idx
                      ? "evidence-card-highlighted"
                      : ""
                  }`}
                  data-testid={`code-slice-card-${idx}`}
                >
                  <div className="evidence-card-head">
                    <div>
                      <strong className="symbol-title">
                        {slice.qualified_symbol}
                      </strong>
                      <div className="file-coords">
                        <code>
                          {slice.repo_name}/{slice.file_path}:
                          {slice.start_line}-{slice.end_line}
                        </code>
                      </div>
                    </div>
                    <div className="evidence-badges">
                      <span className="badge badge-neutral">
                        {slice.modification_intent}
                      </span>
                      <span className="badge badge-neutral">
                        sha:{slice.git_commit_sha.slice(0, 7)}
                      </span>
                      {slice.auto_expanded_by_ast && (
                        <span
                          className="badge badge-info"
                          data-testid="ast-closure-badge"
                        >
                          AST 1-Hop Closure
                        </span>
                      )}
                    </div>
                  </div>
                  <p className="evidence-reason">{slice.selection_reason}</p>
                  <pre
                    className="code-verbatim"
                    data-testid={`code-block-${idx}`}
                  >
                    <code>{slice.verbatim_code_with_lines}</code>
                  </pre>
                </div>
              ))
            )}
          </div>
        )}

        {effectiveTab === "quant" && (
          <div className="evidence-list" data-testid="quant-strategy-view">
            {resolvedQuant && (
              <div
                className="quant-spec-card"
                data-testid="quant-executive-answer"
              >
                <div className="evidence-card-head">
                  <div>
                    <strong className="symbol-title">
                      {resolvedQuant.strategy_name}
                    </strong>
                    <div
                      className="markets-list"
                      data-testid="market-regimes-list"
                    >
                      Target Markets: {resolvedQuant.target_markets.join(", ")} ·
                      Sample: {resolvedQuant.sample_period_years}
                    </div>
                  </div>
                  <span
                    className={`badge ${
                      resolvedQuant.friction_verdict === "REALISTIC_FRICTION"
                        ? "badge-success"
                        : "badge-warning"
                    }`}
                    data-testid="friction-verdict-badge"
                  >
                    {resolvedQuant.friction_verdict}
                  </span>
                </div>

                {resolvedQuant.mathematical_signal_latex && (
                  <div
                    className="math-signal-box"
                    data-testid="latex-mathematical-signal"
                  >
                    <code>{resolvedQuant.mathematical_signal_latex}</code>
                  </div>
                )}

                <div className="quant-stats-strip">
                  <span>
                    Sharpe:{" "}
                    <strong>
                      {resolvedQuant.reported_sharpe_ratio ?? "N/A"}
                    </strong>
                  </span>
                  <span>
                    Max Drawdown:{" "}
                    <strong>
                      {resolvedQuant.reported_max_drawdown_pct !== null
                        ? `${resolvedQuant.reported_max_drawdown_pct}%`
                        : "N/A"}
                    </strong>
                  </span>
                  <span>
                    Assumed Friction:{" "}
                    <strong>
                      {resolvedQuant.transaction_cost_bps_assumed !== null
                        ? `${resolvedQuant.transaction_cost_bps_assumed} bps`
                        : "N/A"}
                    </strong>
                  </span>
                </div>

                {resolvedQuant.entry_exit_rules.length > 0 && (
                  <div className="quant-rules-block">
                    <h5>Entry / Exit Rules</h5>
                    <ul data-testid="entry-exit-rules-list">
                      {resolvedQuant.entry_exit_rules.map((rule, rIdx) => (
                        <li key={rIdx}>{rule}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {filteredCitations.length === 0 ? (
              <p className="empty-tab-note">
                {sourceFilter !== "all" && mergedCitations.length > 0
                  ? `No document citations match the "${sourceFilter}" source filter (${mergedCitations.length} total in pack).`
                  : "No supporting document citations attached to this pack."}
              </p>
            ) : (
              filteredCitations.map((cite) => {
                const isStale =
                  cite.is_stale === true || Boolean(cite.stale_reason);
                return (
                  <div
                    key={cite.chunk_id}
                    ref={(el) => {
                      citationRefs.current[cite.chunk_id] = el;
                    }}
                    className={`evidence-card ${
                      highlightedChunkId === cite.chunk_id
                        ? "evidence-card-highlighted"
                        : ""
                    }`}
                    data-testid={`citation-card-${cite.chunk_id}`}
                  >
                    <div className="evidence-card-head">
                      <div>
                        <strong className="symbol-title">{cite.doc_title}</strong>
                        <div className="file-coords">
                          Section: {cite.section_heading} ·{" "}
                          <code>{cite.doc_id}</code>
                          {cite.version ? ` · ${cite.version}` : ""}
                          {cite.published_date ? ` · Updated ${cite.published_date}` : ""}
                          {cite.source_type ? ` · ${cite.source_type}` : ""}
                        </div>
                      </div>
                      <div className="evidence-badges">
                        {isStale ? (
                          <span
                            className="badge badge-warning"
                            data-testid="stale-citation-badge"
                          >
                            <AlertTriangle size={11} /> Outdated / Stale
                          </span>
                        ) : (
                          <span className="badge badge-success">
                            Verified Quote
                          </span>
                        )}
                      </div>
                    </div>
                    {isStale && cite.stale_reason && (
                      <div className="inline-alert inline-alert-warning stale-reason-note">
                        <AlertTriangle size={13} />
                        <span>{cite.stale_reason}</span>
                      </div>
                    )}
                    <blockquote className="verbatim-quote">
                      &ldquo;{cite.verbatim_quote}&rdquo;
                    </blockquote>
                    <div className="citation-source-uri">
                      Source: <code>{cite.source_uri}</code>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}

        {effectiveTab === "preflight-report" && (
          <div className="preflight-panel" data-testid="preflight-report-view">
            <div className="preflight-metrics-grid">
              <div className="stat-box">
                <span className="stat-label">Pre-Flight Result</span>
                <span className="stat-value">
                  {contextPack.preflight_report.passed ? "PASSED" : "FAILED"}
                </span>
              </div>
              <div className="stat-box">
                <span className="stat-label">Normalized RRF Score</span>
                <span className="stat-value">
                  {contextPack.preflight_report.normalized_rrf_score.toFixed(3)}
                </span>
              </div>
              <div className="stat-box">
                <span className="stat-label">Citations Grounded</span>
                <span className="stat-value">
                  {contextPack.preflight_report.citations_verified_100_pct
                    ? "Verified (Zero Unmatched)"
                    : "Unverified Quotes Present"}
                </span>
              </div>
              <div className="stat-box">
                <span className="stat-label">AST 1-Hop Attached</span>
                <span className="stat-value">
                  {contextPack.preflight_report.auto_attached_ast_slices_count}
                </span>
              </div>
            </div>

            {contextPack.preflight_report.bounce_back_feedback && (
              <div className="inline-alert inline-alert-info">
                Bounce-back guidance:{" "}
                {contextPack.preflight_report.bounce_back_feedback}
              </div>
            )}
          </div>
        )}

        {effectiveTab === "telemetry-roi" && (
          <div className="preflight-panel" data-testid="telemetry-roi-view">
            <div className="preflight-metrics-grid">
              <div className="stat-box">
                <span className="stat-label">Local Tokens Explored</span>
                <span className="stat-value">
                  {(
                    tokenSavings?.local_tokens_explored ??
                    contextPack.local_tokens_consumed
                  ).toLocaleString()}
                </span>
              </div>
              <div className="stat-box">
                <span className="stat-label">Curated Context Pack</span>
                <span className="stat-value">
                  {(
                    tokenSavings?.pack_tokens_sent ??
                    contextPack.pack_token_count
                  ).toLocaleString()}
                </span>
              </div>
              <div className="stat-box">
                <span className="stat-label">Paid Tokens Saved</span>
                <span className="stat-value">
                  {(
                    tokenSavings?.tokens_saved ??
                    Math.max(
                      0,
                      contextPack.local_tokens_consumed -
                        contextPack.pack_token_count
                    )
                  ).toLocaleString()}
                </span>
              </div>
              <div className="stat-box">
                <span className="stat-label">Estimated USD Avoided</span>
                <span className="stat-value">
                  ${(tokenSavings?.estimated_usd_saved ?? 0).toFixed(4)}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};
