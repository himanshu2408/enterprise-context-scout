import React, { useState } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Coins,
  Database,
  Sparkles,
} from "lucide-react";
import {
  ROIMetricsResponse,
  ScoutTokenSavingsBreakdown,
} from "../types/contracts";

export interface TokenSavingsMeterProps {
  currentQuerySavings: ScoutTokenSavingsBreakdown | null;
  roiMetrics: ROIMetricsResponse | null;
  roiLoading?: boolean;
  roiError?: string | null;
  defaultExpanded?: boolean;
}

interface FrontierRateCard {
  name: string;
  inputUsdPer1M: number;
}

const OCT_2026_FRONTIER_RATES: FrontierRateCard[] = [
  { name: "Gemini 3.8 Flash", inputUsdPer1M: 0.75 },
  { name: "Gemini 4 Argon", inputUsdPer1M: 2.0 },
  { name: "Claude Opus 5.5", inputUsdPer1M: 4.0 },
  { name: "GPT-5.6 Sol", inputUsdPer1M: 4.0 },
];

export const TokenSavingsMeter: React.FC<TokenSavingsMeterProps> = ({
  currentQuerySavings,
  roiMetrics,
  roiLoading = false,
  roiError = null,
  defaultExpanded = true,
}) => {
  const [expanded, setExpanded] = useState<boolean>(defaultExpanded);

  const ratioPct = currentQuerySavings
    ? Math.min(100, Math.max(0, currentQuerySavings.frontier_savings_ratio_pct))
    : 0;

  const summary = roiMetrics?.summary ?? null;

  return (
    <section
      className="telemetry-strip-panel"
      data-testid="token-savings-meter"
    >
      <div className="telemetry-strip-row">
        <div className="telemetry-strip-left">
          <div className="scope-section-title">
            <Coins size={15} className="icon-accent" />
            <h2 className="section-heading">Pack Size &amp; Cost</h2>
          </div>

          {summary && (
            <span
              className="status-tag status-tag-info"
              data-testid="telemetry-source-badge"
            >
              <Database size={11} />
              {summary.telemetry_source === "postgresql_live"
                ? "PG17 Live Ledger"
                : summary.telemetry_source}
            </span>
          )}

          {roiError ? (
            <div
              className="inline-alert inline-alert-danger inline-alert-compact"
              data-testid="roi-error-banner"
            >
              <AlertTriangle size={13} />
              <span>Cumulative ROI unavailable: {roiError}</span>
            </div>
          ) : roiLoading && !summary ? (
            <span className="roi-loading-inline">
              Loading cumulative PostgreSQL ROI...
            </span>
          ) : summary ? (
            <div
              className="cumulative-roi-pill"
              data-testid="cumulative-roi-card"
            >
              <span>
                Cumulative ({summary.total_queries} queries):{" "}
                <strong>
                  ${summary.total_estimated_usd_saved.toFixed(2)} saved
                </strong>
                {", "}
                {summary.total_paid_tokens_saved.toLocaleString()} paid tokens
                avoided ({summary.frontier_savings_ratio_pct.toFixed(1)}% avg)
              </span>
            </div>
          ) : null}
        </div>

        <button
          type="button"
          className="btn-quiet-toggle"
          onClick={() => setExpanded((prev) => !prev)}
          aria-expanded={expanded}
          data-testid="toggle-telemetry-strip-btn"
        >
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <span>
            {expanded ? "Hide Frontier Rates" : "Compare Frontier Rates"}
          </span>
        </button>
      </div>

      <div className="telemetry-strip-metrics">
        {currentQuerySavings ? (
          <div className="savings-strip-summary">
            <div className="savings-strip-ratio-wrap">
              <span className="savings-label">Current Query Reduction</span>
              <span
                className="savings-pct-value tabular-num"
                data-testid="frontier-savings-ratio"
              >
                {currentQuerySavings.frontier_savings_ratio_pct.toFixed(1)}%
              </span>
              <div
                className="progress-track"
                aria-label="Frontier token reduction"
              >
                <div
                  className="progress-bar-fill"
                  style={{ width: `${ratioPct}%` }}
                />
              </div>
            </div>

            <div className="savings-strip-stats">
              <div className="stat-inline">
                <span className="stat-label">Local Explored</span>
                <span
                  className="stat-value tabular-num"
                  data-testid="local-tokens-explored"
                >
                  {currentQuerySavings.local_tokens_explored.toLocaleString()}
                </span>
              </div>
              <div className="stat-inline">
                <span className="stat-label">Pack Tokens</span>
                <span
                  className="stat-value tabular-num"
                  data-testid="frontier-tokens-sent"
                >
                  {currentQuerySavings.pack_tokens_sent.toLocaleString()}
                </span>
              </div>
              <div className="stat-inline">
                <span className="stat-label">QLoRA Overhead Saved</span>
                <span
                  className="stat-value tabular-num"
                  data-testid="qlora-overhead-eliminated"
                >
                  {currentQuerySavings.qlora_overhead_eliminated_tokens.toLocaleString()}{" "}
                  tok
                </span>
              </div>
              <div className="stat-inline">
                <span className="stat-label">Est. USD Avoided</span>
                <span
                  className="stat-value stat-value-accent tabular-num"
                  data-testid="usd-saved"
                >
                  ${currentQuerySavings.estimated_usd_saved.toFixed(4)}
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div
            className="empty-telemetry-inline"
            data-testid="empty-query-savings"
          >
            <Sparkles size={13} className="icon-muted" />
            <span>
              Run a query to measure live local corpus tokens filtered vs.
              frontier context pack size.
            </span>
          </div>
        )}
      </div>

      {expanded && (
        <div
          className="frontier-rates-drawer"
          data-testid="frontier-rates-drawer"
        >
          <span className="field-label">
            Frontier Input Comparison (Oct 2026 Verified List Rates)
          </span>
          <div className="rates-table-horizontal">
            {OCT_2026_FRONTIER_RATES.map((model) => {
              const naiveUsd = currentQuerySavings
                ? (currentQuerySavings.local_tokens_explored / 1_000_000) *
                  model.inputUsdPer1M
                : null;
              const scoutUsd = currentQuerySavings
                ? (currentQuerySavings.pack_tokens_sent / 1_000_000) *
                  model.inputUsdPer1M
                : null;
              return (
                <div key={model.name} className="rate-row">
                  <span className="rate-model">{model.name}</span>
                  <span className="rate-price tabular-num">
                    ${model.inputUsdPer1M.toFixed(2)}/1M
                  </span>
                  <span className="rate-delta tabular-num">
                    {naiveUsd !== null && scoutUsd !== null
                      ? `$${naiveUsd.toFixed(4)} vs $${scoutUsd.toFixed(4)}`
                      : "Run query"}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
};
