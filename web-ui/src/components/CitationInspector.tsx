import React, { useState } from "react";
import {
  FileCode,
  Sigma,
  CheckCircle2,
  AlertCircle,
  GitCommit,
  GitBranch,
  ShieldCheck,
  TrendingUp,
  FileCheck2,
} from "lucide-react";
import {
  ContextPack,
  FrictionVerdict,
  MarketRegime,
  ModificationIntent,
} from "../types/contracts";

export interface CitationInspectorProps {
  contextPack: ContextPack | null;
}

type TabType = "code" | "quant" | "preflight";

export const CitationInspector: React.FC<CitationInspectorProps> = ({
  contextPack,
}) => {
  const [activeTab, setActiveTab] = useState<TabType>("code");

  if (!contextPack) {
    return (
      <div className="card citation-inspector-card empty-card" data-testid="citation-inspector">
        <div className="empty-state">
          <FileCode size={32} className="icon-muted mb-2" />
          <div className="font-semibold text-muted">No Active ContextPack Emitted</div>
          <div className="text-xs text-muted mt-1">
            Execute a Scout workflow above to inspect AST code closures, quant friction audits, and compiler pre-flight validation.
          </div>
        </div>
      </div>
    );
  }

  const {
    code_slices,
    quant_strategy_spec,
    preflight_report,
    architectural_execution_plan,
  } = contextPack;

  const renderModificationIntentBadge = (intent: ModificationIntent): JSX.Element => {
    switch (intent) {
      case "modify_target":
        return <span className="badge badge-sky text-xs">TARGET TO MODIFY</span>;
      case "direct_caller":
        return <span className="badge badge-purple text-xs">DIRECT CALLER (AST)</span>;
      case "imported_type":
        return <span className="badge badge-emerald text-xs">IMPORTED TYPE</span>;
      case "test_fixture":
        return <span className="badge badge-amber text-xs">TEST FIXTURE</span>;
      default:
        return <span className="badge badge-zinc text-xs">REFERENCE ONLY</span>;
    }
  };

  const renderFrictionVerdictBadge = (verdict: FrictionVerdict): JSX.Element => {
    switch (verdict) {
      case "REALISTIC_FRICTION":
        return (
          <span
            className="badge badge-emerald"
            data-testid="friction-verdict-badge"
          >
            <CheckCircle2 size={12} className="mr-1" /> REALISTIC FRICTION
          </span>
        );
      case "UNVERIFIED_FRICTION_WARNING":
        return (
          <span
            className="badge badge-amber"
            data-testid="friction-verdict-badge"
          >
            <AlertCircle size={12} className="mr-1" /> UNVERIFIED FRICTION WARNING
          </span>
        );
      case "FAILS_AFTER_COSTS":
        return (
          <span
            className="badge badge-rose"
            data-testid="friction-verdict-badge"
          >
            <AlertCircle size={12} className="mr-1" /> FAILS AFTER COSTS
          </span>
        );
    }
  };

  return (
    <div className="card citation-inspector-card" data-testid="citation-inspector">
      <div className="card-header">
        <div className="card-title">
          <FileCheck2 className="icon icon-sky" size={18} />
          <span>Citation & AST Closure Inspector (ARC-03)</span>
        </div>

        <div className="tab-buttons" role="tablist" aria-label="Inspection tabs">
          <button
            type="button"
            data-testid="tab-code-slices"
            role="tab"
            aria-selected={activeTab === "code"}
            className={`tab-btn ${activeTab === "code" ? "tab-btn-active" : ""}`}
            onClick={() => setActiveTab("code")}
          >
            <FileCode size={14} className="mr-1" />
            Code Slices ({code_slices.length})
          </button>

          <button
            type="button"
            data-testid="tab-quant-strategy"
            role="tab"
            aria-selected={activeTab === "quant"}
            className={`tab-btn ${activeTab === "quant" ? "tab-btn-active" : ""}`}
            onClick={() => setActiveTab("quant")}
          >
            <Sigma size={14} className="mr-1" />
            Quant Strategy {quant_strategy_spec ? "(Active)" : "(N/A)"}
          </button>

          <button
            type="button"
            data-testid="tab-preflight-report"
            role="tab"
            aria-selected={activeTab === "preflight"}
            className={`tab-btn ${activeTab === "preflight" ? "tab-btn-active" : ""}`}
            onClick={() => setActiveTab("preflight")}
          >
            <ShieldCheck size={14} className="mr-1" />
            Pre-Flight Gate ({preflight_report.passed ? "PASSED" : "FAILED"})
          </button>
        </div>
      </div>

      <div className="tab-content">
        {/* Tab 1: Code Slices */}
        {activeTab === "code" && (
          <div className="code-slices-container" data-testid="code-slices-view">
            {code_slices.length === 0 ? (
              <div className="text-muted text-sm">No code slices in this pack.</div>
            ) : (
              code_slices.map((slice, idx) => (
                <div
                  key={`${slice.file_path}-${slice.start_line}-${idx}`}
                  data-testid={`code-slice-card-${idx}`}
                  className="code-slice-card"
                >
                  <div className="code-slice-header">
                    <div className="flex-row-gap">
                      <span className="font-mono text-sm font-semibold text-sky">
                        {slice.file_path}
                      </span>
                      <span className="badge badge-zinc text-xs">
                        lines {slice.start_line}-{slice.end_line}
                      </span>
                      {renderModificationIntentBadge(slice.modification_intent)}
                      {slice.auto_expanded_by_ast && (
                        <span
                          className="badge badge-emerald text-xs"
                          data-testid="ast-closure-badge"
                        >
                          <GitBranch size={10} className="mr-1" /> AST 1-Hop Closure
                        </span>
                      )}
                    </div>

                    <div className="flex-row-gap text-xs text-muted">
                      <span className="flex-row-gap">
                        <GitCommit size={12} /> Blob: {slice.git_blob_sha}
                      </span>
                    </div>
                  </div>

                  <div className="slice-symbol-reason">
                    <span className="font-semibold text-xs text-zinc-400">
                      Symbol: {slice.qualified_symbol}
                    </span>
                    <span className="text-xs text-muted ml-2">
                      ({slice.selection_reason})
                    </span>
                  </div>

                  <pre className="code-block" data-testid={`code-block-${idx}`}>
                    <code>{slice.verbatim_code_with_lines}</code>
                  </pre>
                </div>
              ))
            )}
          </div>
        )}

        {/* Tab 2: Quant Strategy Brief */}
        {activeTab === "quant" && (
          <div className="quant-strategy-container" data-testid="quant-strategy-view">
            {!quant_strategy_spec ? (
              <div className="text-muted text-sm">
                No QuantStrategyBrief attached. Active DomainPack focuses exclusively on software code AST closure.
              </div>
            ) : (
              <div className="quant-strategy-card">
                <div className="quant-header-row">
                  <div>
                    <h4 className="quant-title font-semibold text-lg text-zinc-100">
                      {quant_strategy_spec.strategy_name}
                    </h4>
                    <div className="text-xs text-muted mt-1">
                      Sample Period: {quant_strategy_spec.sample_period_years}
                    </div>
                  </div>
                  <div>
                    {renderFrictionVerdictBadge(quant_strategy_spec.friction_verdict)}
                  </div>
                </div>

                <div className="latex-signal-box">
                  <div className="text-xs font-semibold text-muted mb-1">
                    Mathematical Signal Formulation (LaTeX):
                  </div>
                  <pre
                    className="latex-display font-mono"
                    data-testid="latex-mathematical-signal"
                  >
                    {quant_strategy_spec.mathematical_signal_latex}
                  </pre>
                </div>

                <div className="quant-metrics-grid">
                  <div className="metric-box">
                    <div className="metric-label">Reported Sharpe Ratio</div>
                    <div className="metric-value text-emerald">
                      {quant_strategy_spec.reported_sharpe_ratio?.toFixed(2) ?? "N/A"}
                    </div>
                  </div>
                  <div className="metric-box">
                    <div className="metric-label">Reported Max Drawdown</div>
                    <div className="metric-value text-rose">
                      {quant_strategy_spec.reported_max_drawdown_pct?.toFixed(1) ?? "N/A"}%
                    </div>
                  </div>
                  <div className="metric-box">
                    <div className="metric-label">Assumed Cost Friction</div>
                    <div className="metric-value text-sky">
                      {quant_strategy_spec.transaction_cost_bps_assumed ?? 0} bps
                    </div>
                  </div>
                  <div className="metric-box">
                    <div className="metric-label">Target Markets</div>
                    <div className="market-regimes-list" data-testid="market-regimes-list">
                      {quant_strategy_spec.target_markets.map((mkt: MarketRegime) => (
                        <span key={mkt} className="badge badge-purple text-xs">
                          {mkt}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="citations-table-section mt-4">
                  <div className="text-xs font-semibold text-muted mb-2">
                    Verified Citations & Verbatim Quotes:
                  </div>
                  <div className="citations-list">
                    {quant_strategy_spec.verified_citations.map((cite) => (
                      <div
                        key={cite.chunk_id}
                        className="citation-card"
                        data-testid={`citation-card-${cite.chunk_id}`}
                      >
                        <div className="citation-header">
                          <span className="font-semibold text-xs text-sky">
                            {cite.doc_title}
                          </span>
                          <span className="badge badge-zinc text-xs">
                            {cite.section_heading}
                          </span>
                        </div>
                        <div className="citation-quote font-mono text-xs">
                          &ldquo;{cite.verbatim_quote}&rdquo;
                        </div>
                        <div className="citation-uri text-xs text-muted">
                          Source: {cite.source_uri}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Pre-Flight Report */}
        {activeTab === "preflight" && (
          <div className="preflight-report-container" data-testid="preflight-report-view">
            <div className="preflight-summary-card">
              <div className="flex-between">
                <div className="flex-row-gap">
                  <span
                    className={`badge ${preflight_report.passed ? "badge-emerald" : "badge-rose"} text-sm font-semibold`}
                    data-testid="preflight-passed-badge"
                  >
                    {preflight_report.passed ? "PRE-FLIGHT VALIDATION PASSED" : "FAILED"}
                  </span>
                  <span className="text-xs text-muted">
                    Compiler completeness check before paid token handoff
                  </span>
                </div>
                <div className="font-mono text-sm text-sky">
                  Normalized RRF: {(preflight_report.normalized_rrf_score * 100).toFixed(1)}%
                </div>
              </div>

              <div className="preflight-stats-grid mt-4">
                <div className="preflight-stat-box">
                  <div className="text-xs text-muted">Unresolved Symbols</div>
                  <div className="stat-val font-semibold text-emerald">
                    {preflight_report.unresolved_symbols.length}
                  </div>
                </div>
                <div className="preflight-stat-box">
                  <div className="text-xs text-muted">Auto-Attached AST Slices</div>
                  <div className="stat-val font-semibold text-sky">
                    {preflight_report.auto_attached_ast_slices_count}
                  </div>
                </div>
                <div className="preflight-stat-box">
                  <div className="text-xs text-muted">Citations 100% Verified</div>
                  <div className="stat-val font-semibold text-purple">
                    {preflight_report.citations_verified_100_pct ? "YES (100%)" : "NO"}
                  </div>
                </div>
              </div>

              <div className="execution-plan-section mt-4">
                <div className="text-xs font-semibold text-muted mb-2">
                  Architectural Execution Plan:
                </div>
                <ol className="plan-list">
                  {architectural_execution_plan.map((step, idx) => (
                    <li key={idx} className="plan-item text-xs">
                      <TrendingUp size={12} className="mr-1 text-sky flex-shrink-0" />
                      <span>{step}</span>
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
