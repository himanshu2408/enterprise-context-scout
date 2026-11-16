import React, { useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
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
  ModificationIntent,
  QuantStrategyBrief,
  QueryFacetCoverage,
  RoleType,
  ScoutTokenSavingsBreakdown,
  SourceFilterType,
  VerbatimCodeSlice,
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

const SLICE_GROUP_ORDER: Array<{
  intent: ModificationIntent;
  heading: string;
  testId: string;
}> = [
  {
    intent: "modify_target",
    heading: "Target Implementation Slices",
    testId: "slice-group-modify_target",
  },
  {
    intent: "imported_type",
    heading: "Types & Schemas (AST Closure)",
    testId: "slice-group-imported_type",
  },
  {
    intent: "direct_caller",
    heading: "Direct Callers (AST Closure)",
    testId: "slice-group-direct_caller",
  },
  {
    intent: "test_fixture",
    heading: "Unit Tests (AST Closure)",
    testId: "slice-group-test_fixture",
  },
  {
    intent: "reference_only",
    heading: "Reference & Contracts",
    testId: "slice-group-reference_only",
  },
];

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
    return `${cite.version}, ${cite.published_date}`;
  }
  if (cite.published_date) {
    return `Updated ${cite.published_date}`;
  }
  return cite.version || "v1.0";
}

function formatCleanCitationChipLabel(
  cite: DocCitation,
  index: number
): string {
  const title = cite.doc_title.trim();
  let section = cite.section_heading.trim();
  for (const sep of [" > ", " — ", " - ", ": "]) {
    const prefix = `${title}${sep}`;
    if (section.toLowerCase().startsWith(prefix.toLowerCase())) {
      section = section.slice(prefix.length).trim();
      break;
    }
  }
  const versionOrDate = formatCitationVersionOrDate(cite);
  if (!section || section.toLowerCase() === title.toLowerCase()) {
    return `[${index + 1}] ${title} (${versionOrDate})`;
  }
  return `[${index + 1}] ${title}: ${section} (${versionOrDate})`;
}

function formatSynthesisMode(mode: string | undefined): string | null {
  if (!mode) return null;
  if (mode === "live_qlora_json") return "Live QLoRA Synthesis";
  if (mode === "deterministic_extractive_fallback") {
    return "Deterministic Extractive Fallback";
  }
  return mode;
}

function renderHighlightedSection(
  fullText: string,
  quote: string
): React.ReactNode {
  const trimmedQuote = quote.trim();
  if (!trimmedQuote) return fullText;
  const idx = fullText.indexOf(trimmedQuote);
  if (idx === -1) return fullText;

  const before = fullText.slice(0, idx);
  const match = fullText.slice(idx, idx + trimmedQuote.length);
  const after = fullText.slice(idx + trimmedQuote.length);

  return (
    <>
      {before}
      <mark className="quote-highlight-mark">{match}</mark>
      {after}
    </>
  );
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
  const [collapsedSlices, setCollapsedSlices] = useState<
    Record<number, boolean>
  >({});
  const [expandedSections, setExpandedSections] = useState<
    Record<string, boolean>
  >({});

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

  const indexedSlices = useMemo(() => {
    return filteredSlices.map((slice, idx) => ({ slice, idx }));
  }, [filteredSlices]);

  const groupedSlices = useMemo(() => {
    const groups: Array<{
      intent: ModificationIntent;
      heading: string;
      testId: string;
      items: Array<{ slice: VerbatimCodeSlice; idx: number }>;
    }> = [];

    for (const def of SLICE_GROUP_ORDER) {
      const items = indexedSlices.filter(
        (entry) => entry.slice.modification_intent === def.intent
      );
      if (items.length > 0) {
        groups.push({ ...def, items });
      }
    }
    return groups;
  }, [indexedSlices]);

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
      (quantSpec !== null ||
        filteredCitations.length > 0 ||
        sourceFilter === "wiki" ||
        sourceFilter === "jira")
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
    setCollapsedSlices((prev) => ({ ...prev, [idx]: false }));
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
    setExpandedSections((prev) => ({ ...prev, [chunkId]: true }));
    window.setTimeout(() => {
      const target = citationRefs.current[chunkId];
      if (target && typeof target.scrollIntoView === "function") {
        target.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }, 0);
  };

  if (!contextPack) {
    if (abstainedDiagnostic) {
      const stage2Trace = abstainedDiagnostic.stage_traces?.find(
        (t) => t.stage === 2
      );
      const probedDomains = stage2Trace?.domain_probes ?? [];
      const missingAspectsList = [
        ...abstainedDiagnostic.missing_symbols,
        ...(abstainedDiagnostic.missing_or_unverified_aspects ?? []),
        ...abstainedDiagnostic.missing_items.map((item) =>
          typeof item === "string"
            ? item
            : `${item.identifier}${item.reason ? ` — ${item.reason}` : ""}`
        ),
      ];
      const uniqueMissing = Array.from(new Set(missingAspectsList));

      return (
        <section
          className="abstention-protection-panel"
          data-testid="citation-inspector"
        >
          <div
            className="abstention-panel-inner"
            data-testid="abstention-protection-panel"
          >
            <div className="abstention-panel-header">
              <div className="abstention-title-group">
                <ShieldCheck size={20} className="icon-emerald" />
                <div>
                  <h3 className="abstention-title">
                    Insufficient Grounded Evidence in Indexed Corpus — Paid
                    Cloud Handoff Blocked ($0.00 spent)
                  </h3>
                  <p className="abstention-subtitle">
                    Pre-flight gate abstained safely before invoking any paid
                    frontier model.
                  </p>
                </div>
              </div>
              <span className="status-tag status-tag-ok tabular-num">
                ${abstainedDiagnostic.paid_tokens_spent.toFixed(2)} Paid Spend
              </span>
            </div>

            <div className="abstention-reason-box">
              <strong>Diagnostic:</strong> {abstainedDiagnostic.reason}
              {abstainedDiagnostic.failure_check && (
                <div className="abstention-gate-line">
                  Gate check:{" "}
                  <code>{abstainedDiagnostic.failure_check}</code>
                </div>
              )}
            </div>

            <div className="abstention-grid">
              <div className="abstention-card">
                <h4>Domains Searched</h4>
                {probedDomains.length > 0 ? (
                  <ul className="abstention-list">
                    {probedDomains.map((p) => (
                      <li key={p.domain_id}>
                        <code>{p.domain_id}</code>:{" "}
                        <span className="tabular-num">
                          {p.probed_hits} probed, {p.accepted_hits} accepted,{" "}
                          {p.rejected_hits} rejected (max cal{" "}
                          {p.max_calibrated_score.toFixed(2)})
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="abstention-meta">
                    Probed scope: <code>{abstainedDiagnostic.domain_id}</code>
                  </p>
                )}
              </div>

              <div className="abstention-card">
                <h4>Missing Concepts &amp; Unverified Facets</h4>
                {uniqueMissing.length > 0 ? (
                  <ul className="abstention-list">
                    {uniqueMissing.map((item, idx) => (
                      <li key={idx}>
                        <code>{item}</code>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="abstention-meta">
                    No retrieved chunks met the calibrated relevance floor inside
                    your authorized scopes.
                  </p>
                )}
              </div>
            </div>

            <div className="abstention-resolution">
              <strong>How to resolve:</strong> Index the target repository or
              documentation into PostgreSQL 17, verify your active RBAC scopes
              cover the target corpus, or narrow your query to indexed symbols
              and concepts.
            </div>
          </div>
        </section>
      );
    }

    return (
      <section className="workbench-empty" data-testid="citation-inspector">
        <div className="empty-state-content">
          <Sparkles size={24} className="icon-accent" />
          <h3>Grounded Answer &amp; Context Workbench</h3>
          <p>
            Select your persona, RBAC scopes, and domain, then run a query
            above. Local Gemma 4 + EmbeddingGemma 2 will assemble an
            un-truncated Answer-First brief backed by verbatim AST code slices
            and full document sections.
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
    ? `Strategy & Sections (${filteredCitations.length})`
    : `Document Sections (${filteredCitations.length})`;

  const queryFacets: QueryFacetCoverage[] =
    contextPack.query_facets ?? brief?.query_facets ?? [];
  const missingAspects: string[] = Array.from(
    new Set([
      ...(contextPack.missing_or_unverified_aspects ?? []),
      ...(brief?.missing_or_unverified_aspects ?? []),
      ...(brief?.missing_aspects ?? []),
      ...(resolvedQuant?.missing_aspects ?? []),
    ])
  );
  const sufficiencyVerdict =
    brief?.sufficiency_verdict ?? resolvedQuant?.sufficiency_verdict;
  const groundedFacetCount = queryFacets.filter(
    (f) => f.status === "grounded"
  ).length;
  const hasPartialOrMissing =
    sufficiencyVerdict === "partial" ||
    missingAspects.length > 0 ||
    queryFacets.some((f) => f.status === "not_found" || f.status === "partial");
  const completenessRatio =
    contextPack.completeness_ratio ?? brief?.completeness_ratio ?? 1.0;
  const isPartialCoverage = completenessRatio < 1.0 || hasPartialOrMissing;

  const synthesisModeLabel = formatSynthesisMode(
    contextPack.synthesis_mode ?? brief?.synthesis_mode
  );

  const answerHeadline =
    brief?.brief_title ??
    resolvedQuant?.strategy_name ??
    "Grounded Engineering & Knowledge Answer";

  const directSummary =
    brief?.executive_summary ??
    (contextPack.architectural_execution_plan.length > 0
      ? contextPack.architectural_execution_plan.join(" ")
      : `Synthesized ${contextPack.code_slices.length} AST code slices and ${mergedCitations.length} verified document citations for "${contextPack.user_task}".`);

  const rawFindings =
    (brief?.core_findings_and_rules.length ?? 0) > 0
      ? brief?.core_findings_and_rules ?? []
      : contextPack.architectural_execution_plan;
  const normalizedSummary = directSummary.trim().toLowerCase();
  const coreFindings = rawFindings.filter(
    (item) => item.trim().toLowerCase() !== normalizedSummary
  );

  return (
    <section className="workbench-split" data-testid="citation-inspector">
      {/* Left Column: Primary Answer (<72ch editorial measure) */}
      <div className="workbench-answer-col">
        {hasPartialOrMissing ? (
          <div
            className="coverage-callout coverage-callout-amber"
            data-testid="partial-coverage-callout"
            role="status"
          >
            <div className="coverage-callout-head">
              <AlertTriangle size={16} />
              <strong>
                Partial corpus coverage
                {queryFacets.length > 0
                  ? ` (${groundedFacetCount} of ${queryFacets.length} facets grounded)`
                  : ""}
                {" — "}Not found in indexed corpus:
              </strong>
            </div>
            {missingAspects.length > 0 && (
              <ul className="coverage-missing-list">
                {missingAspects.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            )}
            {queryFacets.length > 0 && (
              <div className="facet-pills-row">
                {queryFacets.map((f) => (
                  <span
                    key={f.facet_id}
                    className={`facet-pill facet-pill-${f.status}`}
                    title={f.explanation}
                    data-testid={`facet-pill-${f.facet_id}`}
                  >
                    {f.facet_query}:{" "}
                    {f.status === "grounded"
                      ? "Grounded"
                      : f.status === "partial"
                        ? "Partial"
                        : "Not found"}
                  </span>
                ))}
              </div>
            )}
          </div>
        ) : queryFacets.length > 0 ? (
          <div
            className="coverage-callout coverage-callout-ok"
            data-testid="coverage-honesty-banner"
          >
            <CheckCircle2 size={15} />
            <span>
              {groundedFacetCount} of {queryFacets.length} query facets grounded
              {synthesisModeLabel ? ` (${synthesisModeLabel})` : ""}
            </span>
          </div>
        ) : null}

        {conflicts.length > 0 && (
          <div
            className="conflicts-alert-banner"
            data-testid="conflicts-banner"
            role="alert"
          >
            <div className="conflicts-header">
              <AlertTriangle size={16} />
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

        <article className="answer-first-hero" data-testid="answer-first-hero">
          <div className="hero-top-meta">
            <div className="hero-badges-left">
              <span
                className={`status-tag ${
                  isPartialCoverage
                    ? "status-tag-warn"
                    : contextPack.preflight_report.passed
                      ? "status-tag-ok"
                      : "status-tag-warn"
                }`}
                data-testid="preflight-passed-badge"
              >
                {isPartialCoverage ? (
                  <AlertTriangle size={12} />
                ) : (
                  <CheckCircle2 size={12} />
                )}
                {isPartialCoverage
                  ? queryFacets.length > 0
                    ? `Partial Coverage (${groundedFacetCount}/${queryFacets.length} facets)`
                    : "Partial Coverage"
                  : contextPack.preflight_report.passed
                    ? "100% Grounded & Verified"
                    : "Preflight Review Flagged"}
              </span>
              <span className="meta-inline">
                Domain: <code>{contextPack.domain_id}</code>
              </span>
              <span className="meta-inline">
                Trace: <code>{contextPack.trace_id}</code>
              </span>
              {synthesisModeLabel && !hasPartialOrMissing && queryFacets.length === 0 && (
                <span className="meta-inline">{synthesisModeLabel}</span>
              )}
            </div>
            <span className="token-pack-pill tabular-num">
              {contextPack.pack_token_count.toLocaleString()} curated pack tokens
            </span>
          </div>

          <h2 className="hero-title">{answerHeadline}</h2>
          <p className="hero-summary">{directSummary}</p>

          {Boolean(brief?.formal_expression_or_rule) && (
            <div className="formal-rule-box">
              <span className="formal-label">Formal Rule / Expression</span>
              <code>{brief?.formal_expression_or_rule}</code>
            </div>
          )}

          {coreFindings.length > 0 && (
            <div className="answer-section-block">
              <h3 className="answer-subheading">
                Key Findings &amp; Execution Plan
              </h3>
              <ol className="answer-ordered-list">
                {coreFindings.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ol>
            </div>
          )}

          {brief !== null &&
            (brief.domain_constraints_and_caveats.length > 0 ||
              brief.known_risks_and_failure_modes.length > 0) && (
              <div className="answer-section-block">
                <h3 className="answer-subheading">
                  Constraints &amp; Known Risks
                </h3>
                <ul className="answer-bullet-list">
                  {brief.domain_constraints_and_caveats.map((item, idx) => (
                    <li key={`c-${idx}`}>{item}</li>
                  ))}
                  {brief.known_risks_and_failure_modes.map((item, idx) => (
                    <li key={`r-${idx}`}>{item}</li>
                  ))}
                </ul>
              </div>
            )}

          {(filteredSlices.length > 0 || filteredCitations.length > 0) && (
            <div className="inline-citations-bar">
              <span className="inline-citations-label">Verified Sources</span>
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
                {filteredCitations.map((cite, idx) => {
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
                      <span>{formatCleanCitationChipLabel(cite, idx)}</span>
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
        </article>
      </div>

      {/* Right Column: Complete Un-Truncated Evidence & Frontier Pack */}
      <div className="workbench-evidence-col">
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
            <Code size={14} /> Code &amp; Dependencies ({filteredSlices.length})
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
            Frontier Pack ({contextPack.pack_token_count.toLocaleString()} tok)
          </button>
        </div>

        <div className="inspector-tab-body">
          {effectiveTab === "code-slices" && (
            <div className="evidence-list" data-testid="code-slices-view">
              {filteredSlices.length === 0 ? (
                <p className="empty-tab-note">
                  {sourceFilter === "wiki" || sourceFilter === "jira"
                    ? `Code slices hidden under active "${sourceFilter}" source filter. Switch to Document Sections or select All Federated.`
                    : "No source code slices returned for this query. Switch to Document Sections to inspect grounded document quotes."}
                </p>
              ) : (
                groupedSlices.map((group) => (
                  <div
                    key={group.intent}
                    className="evidence-group"
                    data-testid={group.testId}
                  >
                    <h3 className="evidence-group-heading">
                      {group.heading} ({group.items.length})
                    </h3>
                    {group.items.map(({ slice, idx }) => {
                      const isCollapsed = Boolean(collapsedSlices[idx]);
                      return (
                        <div
                          key={`${slice.file_path}-${idx}`}
                          ref={(el) => {
                            sliceRefs.current[idx] = el;
                          }}
                          className={`evidence-item ${
                            highlightedSliceIdx === idx
                              ? "evidence-item-highlighted"
                              : ""
                          }`}
                          data-testid={`code-slice-card-${idx}`}
                        >
                          <div className="evidence-item-head">
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
                              <span className="status-tag status-tag-neutral">
                                {slice.modification_intent}
                              </span>
                              <span className="status-tag status-tag-neutral tabular-num">
                                sha:{slice.git_commit_sha.slice(0, 7)}
                              </span>
                              {slice.auto_expanded_by_ast && (
                                <span
                                  className="status-tag status-tag-info"
                                  data-testid="ast-closure-badge"
                                >
                                  AST Closure
                                </span>
                              )}
                              <button
                                type="button"
                                className="btn-quiet-toggle"
                                onClick={() =>
                                  setCollapsedSlices((prev) => ({
                                    ...prev,
                                    [idx]: !prev[idx],
                                  }))
                                }
                                data-testid={`toggle-slice-${idx}`}
                              >
                                {isCollapsed ? (
                                  <>
                                    <ChevronRight size={12} /> Expand
                                  </>
                                ) : (
                                  <>
                                    <ChevronDown size={12} /> Collapse
                                  </>
                                )}
                              </button>
                            </div>
                          </div>
                          <p className="evidence-reason">
                            {slice.selection_reason}
                          </p>
                          {!isCollapsed && (
                            <pre
                              className="code-verbatim"
                              data-testid={`code-block-${idx}`}
                            >
                              <code>{slice.verbatim_code_with_lines}</code>
                            </pre>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ))
              )}
            </div>
          )}

          {effectiveTab === "quant" && (
            <div className="evidence-list" data-testid="quant-strategy-view">
              {resolvedQuant && (
                <div
                  className="quant-spec-block"
                  data-testid="quant-executive-answer"
                >
                  <div className="evidence-item-head">
                    <div>
                      <strong className="symbol-title">
                        {resolvedQuant.strategy_name}
                      </strong>
                      <div
                        className="markets-list"
                        data-testid="market-regimes-list"
                      >
                        Target Markets: {resolvedQuant.target_markets.join(", ")}
                        {" | "}Sample: {resolvedQuant.sample_period_years}
                      </div>
                    </div>
                    <span
                      className={`status-tag ${
                        resolvedQuant.friction_verdict === "REALISTIC_FRICTION"
                          ? "status-tag-ok"
                          : "status-tag-warn"
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

                  <div className="quant-stats-strip tabular-num">
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
                  const hasFullSection = Boolean(
                    cite.full_section_content &&
                      cite.full_section_content.trim().length > 0
                  );
                  const isSectionOpen = Boolean(
                    expandedSections[cite.chunk_id]
                  );

                  return (
                    <div
                      key={cite.chunk_id}
                      ref={(el) => {
                        citationRefs.current[cite.chunk_id] = el;
                      }}
                      className={`evidence-item ${
                        highlightedChunkId === cite.chunk_id
                          ? "evidence-item-highlighted"
                          : ""
                      }`}
                      data-testid={`citation-card-${cite.chunk_id}`}
                    >
                      <div className="evidence-item-head">
                        <div>
                          <strong className="symbol-title">
                            {cite.doc_title}
                          </strong>
                          <div className="file-coords">
                            Section: {cite.section_heading} |{" "}
                            <code>{cite.doc_id}</code>
                            {cite.version ? ` | ${cite.version}` : ""}
                            {cite.published_date
                              ? ` | Updated ${cite.published_date}`
                              : ""}
                            {cite.source_type ? ` | ${cite.source_type}` : ""}
                          </div>
                        </div>
                        <div className="evidence-badges">
                          {isStale ? (
                            <span
                              className="status-tag status-tag-warn"
                              data-testid="stale-citation-badge"
                            >
                              <AlertTriangle size={11} /> Outdated / Stale
                            </span>
                          ) : (
                            <span className="status-tag status-tag-ok">
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

                      {hasFullSection && (
                        <div className="full-section-wrap">
                          <button
                            type="button"
                            className="btn-quiet-toggle"
                            onClick={() =>
                              setExpandedSections((prev) => ({
                                ...prev,
                                [cite.chunk_id]: !prev[cite.chunk_id],
                              }))
                            }
                            aria-expanded={isSectionOpen}
                            data-testid={`full-section-toggle-${cite.chunk_id}`}
                          >
                            {isSectionOpen ? (
                              <ChevronDown size={12} />
                            ) : (
                              <ChevronRight size={12} />
                            )}
                            <span>
                              {isSectionOpen
                                ? "Hide Full Section Context"
                                : "Read Full Section Context"}
                            </span>
                          </button>

                          {isSectionOpen && (
                            <div
                              className="full-section-body"
                              data-testid={`full-section-content-${cite.chunk_id}`}
                            >
                              {renderHighlightedSection(
                                cite.full_section_content ?? "",
                                cite.verbatim_quote
                              )}
                            </div>
                          )}
                        </div>
                      )}

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
                  <span className="stat-value tabular-num">
                    {contextPack.preflight_report.normalized_rrf_score.toFixed(
                      3
                    )}
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
                  <span className="stat-label">AST Closure Attached</span>
                  <span className="stat-value tabular-num">
                    {
                      contextPack.preflight_report
                        .auto_attached_ast_slices_count
                    }
                  </span>
                </div>
              </div>

              {contextPack.preflight_report.bounce_back_feedback && (
                <div className="inline-alert inline-alert-info">
                  Bounce-back guidance:{" "}
                  {contextPack.preflight_report.bounce_back_feedback}
                </div>
              )}

              {(() => {
                const s4 = contextPack.stage_traces?.find(
                  (t) => t.stage === 4
                );
                if (!s4 || !s4.checks || s4.checks.length === 0) return null;
                return (
                  <div
                    className="checks-ledger-table"
                    data-testid="inspector-preflight-ledger"
                  >
                    {s4.checks.map((chk) => (
                      <div key={chk.check_name} className="check-ledger-row">
                        <span
                          className={`status-tag ${
                            chk.passed ? "status-tag-ok" : "status-tag-danger"
                          }`}
                        >
                          {chk.passed ? "PASS" : "FAIL"}
                        </span>
                        <span className="check-ledger-name">
                          {chk.check_name}
                        </span>
                        <span className="check-ledger-obs tabular-num">
                          {chk.observed} (req: {chk.threshold})
                        </span>
                      </div>
                    ))}
                  </div>
                );
              })()}
            </div>
          )}

          {effectiveTab === "telemetry-roi" && (
            <div className="preflight-panel" data-testid="telemetry-roi-view">
              <div className="preflight-metrics-grid">
                <div className="stat-box">
                  <span className="stat-label">Local Tokens Explored</span>
                  <span className="stat-value tabular-num">
                    {(
                      tokenSavings?.local_tokens_explored ??
                      contextPack.local_tokens_consumed
                    ).toLocaleString()}
                  </span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">Curated Context Pack</span>
                  <span className="stat-value tabular-num">
                    {(
                      tokenSavings?.pack_tokens_sent ??
                      contextPack.pack_token_count
                    ).toLocaleString()}
                  </span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">Paid Tokens Saved</span>
                  <span className="stat-value tabular-num">
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
                  <span className="stat-value tabular-num">
                    ${(tokenSavings?.estimated_usd_saved ?? 0).toFixed(4)}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
