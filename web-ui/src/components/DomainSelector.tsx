import React, { useMemo, useState } from "react";
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Code,
  Cpu,
  Layers,
  RefreshCw,
} from "lucide-react";
import { FEDERATED_ALL_DOMAIN_ID } from "../api/client";
import { DomainInfo, SourceFilterType } from "../types/contracts";

export interface DomainSelectorProps {
  domains: DomainInfo[];
  selectedDomain: string;
  onSelectDomain: (domainId: string) => void;
  sourceFilter?: SourceFilterType;
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
}

export function matchesDomainSourceFilter(
  domain: DomainInfo,
  sourceFilter: SourceFilterType
): boolean {
  if (sourceFilter === "all") return true;
  const descriptor =
    `${domain.domain_id} ${domain.display_name} ${domain.output_contract}`.toLowerCase();
  if (sourceFilter === "code") {
    return (
      domain.primary_truth_source === "code" ||
      /code|repo|ast|git|polyglot/.test(descriptor)
    );
  }
  if (sourceFilter === "wiki") {
    return (
      domain.primary_truth_source === "document" ||
      /wiki|confluence|doc|quant|research|handbook|runbook/.test(descriptor)
    );
  }
  if (sourceFilter === "jira") {
    return /jira|issue|ticket/.test(descriptor);
  }
  return true;
}

function describeDomain(domain: DomainInfo): string {
  if (domain.primary_truth_source === "document") {
    return `Verified document grounding (${domain.output_contract})`;
  }
  return `AST dependency closure & code slices (${domain.output_contract})`;
}

export const DomainSelector: React.FC<DomainSelectorProps> = ({
  domains,
  selectedDomain,
  onSelectDomain,
  sourceFilter = "all",
  loading = false,
  error = null,
  onRetry,
}) => {
  const [expandedMetrics, setExpandedMetrics] = useState<
    Record<string, boolean>
  >({});

  const visibleDomains = useMemo(
    () => domains.filter((d) => matchesDomainSourceFilter(d, sourceFilter)),
    [domains, sourceFilter]
  );

  const toggleMetrics = (domainId: string, e: React.MouseEvent): void => {
    e.stopPropagation();
    setExpandedMetrics((prev) => ({
      ...prev,
      [domainId]: !prev[domainId],
    }));
  };

  return (
    <section
      className="scope-section domain-scope-section"
      data-testid="domain-selector"
    >
      <div className="scope-section-head">
        <div className="scope-section-title">
          <Layers size={15} className="icon-accent" />
          <h2 className="section-heading">Knowledge Domains</h2>
        </div>
        <span className="meta-count">
          {loading
            ? "Syncing..."
            : sourceFilter === "all"
              ? `${domains.length} registered`
              : `${visibleDomains.length} of ${domains.length} matching`}
        </span>
      </div>

      {error && (
        <div
          className="inline-alert inline-alert-danger"
          data-testid="domains-error-banner"
        >
          <div className="alert-row">
            <AlertTriangle size={14} />
            <span>Failed to load domains: {error}</span>
          </div>
          {onRetry && (
            <button
              type="button"
              className="btn-inline-retry"
              onClick={onRetry}
              data-testid="domains-retry-btn"
            >
              <RefreshCw size={11} /> Retry
            </button>
          )}
        </div>
      )}

      <div className="domain-list">
        <button
          type="button"
          className={`domain-row ${
            selectedDomain === FEDERATED_ALL_DOMAIN_ID ? "domain-row-active" : ""
          }`}
          onClick={() => onSelectDomain(FEDERATED_ALL_DOMAIN_ID)}
          data-testid="domain-card-federated_all"
        >
          <div className="domain-row-main">
            <div className="domain-name-row">
              <Layers size={14} className="icon-accent" />
              <span className="domain-title">All Authorized Domains</span>
            </div>
            <span className="status-tag status-tag-info">Auto-route</span>
          </div>
          <p className="domain-desc">
            Federated search across allowed repositories, Confluence docs, and issues.
          </p>
        </button>

        {loading && domains.length === 0 && (
          <div className="domain-skeleton" data-testid="domains-loading">
            Querying live domain adapter registry...
          </div>
        )}

        {!loading &&
          !error &&
          domains.length > 0 &&
          visibleDomains.length === 0 && (
            <div
              className="domain-filter-empty"
              data-testid="domains-filter-empty"
            >
              No standalone domain registered for filter{" "}
              <code>{sourceFilter}</code>; use{" "}
              <strong>All Authorized Domains</strong> for federated retrieval.
            </div>
          )}

        {visibleDomains.map((domain) => {
          const active = selectedDomain === domain.domain_id;
          const isOpen = Boolean(expandedMetrics[domain.domain_id]);
          const isDoc = domain.primary_truth_source === "document";

          return (
            <div
              key={domain.domain_id}
              className={`domain-row ${active ? "domain-row-active" : ""}`}
              data-testid={`domain-card-${domain.domain_id}`}
              role="button"
              tabIndex={0}
              onClick={() => onSelectDomain(domain.domain_id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelectDomain(domain.domain_id);
                }
              }}
            >
              <div className="domain-row-main">
                <div className="domain-name-row">
                  {isDoc ? (
                    <BookOpen size={14} className="icon-emerald" />
                  ) : (
                    <Code size={14} className="icon-accent" />
                  )}
                  <span className="domain-title">{domain.display_name}</span>
                </div>
                <span
                  className={`status-tag ${
                    domain.verified ? "status-tag-ok" : "status-tag-warn"
                  }`}
                >
                  <CheckCircle2 size={11} />
                  {domain.verified ? "Verified" : "Unverified"}
                </span>
              </div>

              <p className="domain-desc">{describeDomain(domain)}</p>

              <div className="domain-meta-row">
                <code className="domain-id-code">{domain.domain_id}</code>
                <button
                  type="button"
                  className="btn-quiet-toggle"
                  onClick={(e) => toggleMetrics(domain.domain_id, e)}
                  data-testid={`toggle-adapter-metrics-${domain.domain_id}`}
                >
                  {isOpen ? (
                    <ChevronDown size={12} />
                  ) : (
                    <ChevronRight size={12} />
                  )}
                  <Cpu size={11} />
                  <span>Adapter telemetry</span>
                </button>
              </div>

              {isOpen && (
                <div
                  className="adapter-metrics-drawer"
                  data-testid={`adapter-metrics-${domain.domain_id}`}
                >
                  <div className="metric-kv">
                    <span>Adapter</span>
                    <code>{domain.lora_adapter_id}</code>
                  </div>
                  <div className="metric-kv">
                    <span>Rank / Alpha</span>
                    <code>
                      {domain.qlora_rank !== null &&
                      domain.qlora_alpha !== null
                        ? `r=${domain.qlora_rank}, α=${domain.qlora_alpha}`
                        : "N/A (base)"}
                    </code>
                  </div>
                  <div className="metric-kv">
                    <span>Hot-Swap Latency</span>
                    <code>
                      {domain.hot_swap_latency_ms !== null
                        ? `${domain.hot_swap_latency_ms} ms`
                        : "N/A"}
                    </code>
                  </div>
                  <div className="metric-kv">
                    <span>Training Loss</span>
                    <code>
                      {domain.training_loss !== null
                        ? domain.training_loss.toFixed(4)
                        : "N/A"}
                    </code>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
};
