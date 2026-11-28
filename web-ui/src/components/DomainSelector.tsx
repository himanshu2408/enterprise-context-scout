import React, { useMemo, useState } from "react";
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Code,
  Cpu,
  Database,
  Layers,
  RefreshCw,
  Ticket,
} from "lucide-react";
import { FEDERATED_ALL_DOMAIN_ID } from "../api/client";
import {
  DomainInfo,
  FederatedOptionSpec,
  SourceFilterType,
} from "../types/contracts";

export interface DomainSelectorProps {
  domains: DomainInfo[];
  selectedDomain: string;
  onSelectDomain: (domainId: string) => void;
  federatedOption?: FederatedOptionSpec | null;
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
  if (domain.modalities && domain.modalities.length > 0) {
    if (sourceFilter === "code") return domain.modalities.includes("code");
    if (sourceFilter === "wiki") return domain.modalities.includes("document");
    if (sourceFilter === "jira") return domain.modalities.includes("issues");
  }
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
      /wiki|confluence|doc|research|handbook|runbook/.test(descriptor)
    );
  }
  if (sourceFilter === "jira") {
    return /jira|issue|ticket/.test(descriptor);
  }
  return true;
}

function describeDomain(domain: DomainInfo): string {
  if (domain.description && domain.description.trim()) {
    return domain.description;
  }
  if (domain.primary_truth_source === "document") {
    return `Verified document grounding (${domain.output_contract})`;
  }
  return `AST dependency closure & code slices (${domain.output_contract})`;
}

export const DomainSelector: React.FC<DomainSelectorProps> = ({
  domains,
  selectedDomain,
  onSelectDomain,
  federatedOption,
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

  const federatedLabel =
    federatedOption?.display_name ?? "All Authorized Domains (Auto-Route)";
  const federatedDesc =
    federatedOption?.description ??
    "Dynamically probes all authorized DomainPacks in PostgreSQL 17, routes to the best domain(s), or abstains at $0.";

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
        <span className="meta-count tabular-num">
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
          className={`domain-row domain-row-federated ${
            selectedDomain === FEDERATED_ALL_DOMAIN_ID ? "domain-row-active" : ""
          }`}
          onClick={() => onSelectDomain(FEDERATED_ALL_DOMAIN_ID)}
          data-testid="domain-card-federated_all"
        >
          <div className="domain-row-main">
            <div className="domain-name-row">
              <Layers size={14} className="icon-accent" />
              <span className="domain-title">{federatedLabel}</span>
            </div>
            <span className="status-tag status-tag-info">Dynamic Route</span>
          </div>
          <p className="domain-desc">{federatedDesc}</p>
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
              <strong>{federatedLabel}</strong> for federated retrieval.
            </div>
          )}

        {visibleDomains.map((domain) => {
          const active = selectedDomain === domain.domain_id;
          const isOpen = Boolean(expandedMetrics[domain.domain_id]);
          const isDoc = domain.primary_truth_source === "document";
          const cartridgeLabel = domain.lora_enabled
            ? domain.lora_adapter_id
            : "Base Gemma-4-26B";

          return (
            <div
              key={domain.domain_id}
              className={`domain-row ${active ? "domain-row-active" : ""}`}
              data-testid={`domain-card-${domain.domain_id}`}
              onKeyDown={(e) => {
                if (e.target !== e.currentTarget) {
                  return;
                }
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelectDomain(domain.domain_id);
                }
              }}
              onClick={(e) => {
                const target = e.target as HTMLElement | null;
                if (target?.closest("button, a, input, select, textarea")) {
                  return;
                }
                onSelectDomain(domain.domain_id);
              }}
            >
              <button
                type="button"
                className="domain-select-btn"
                aria-pressed={active}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectDomain(domain.domain_id);
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
                    {domain.verified ? "Verified" : "Base"}
                  </span>
                </div>
              </button>

              <p className="domain-desc">{describeDomain(domain)}</p>

              <div className="domain-telemetry-badges">
                {domain.modalities.map((mod) => (
                  <span
                    key={mod}
                    className="modality-pill"
                    data-testid={`domain-modality-${domain.domain_id}-${mod}`}
                  >
                    {mod === "code" && <Code size={10} />}
                    {mod === "document" && <BookOpen size={10} />}
                    {mod === "issues" && <Ticket size={10} />}
                    <span>{mod}</span>
                  </span>
                ))}
                {(domain.doc_chunks_count > 0 || domain.ast_symbols_count > 0) && (
                  <span
                    className="pg-count-pill tabular-num"
                    data-testid={`domain-counts-${domain.domain_id}`}
                  >
                    <Database size={10} />
                    {domain.doc_chunks_count > 0 &&
                      `${domain.doc_chunks_count.toLocaleString()} chunks`}
                    {domain.doc_chunks_count > 0 &&
                      domain.ast_symbols_count > 0 &&
                      " · "}
                    {domain.ast_symbols_count > 0 &&
                      `${domain.ast_symbols_count.toLocaleString()} AST`}
                  </span>
                )}
              </div>

              <div className="domain-meta-row">
                <code className="domain-id-code" title={cartridgeLabel}>
                  {domain.domain_id} · {cartridgeLabel}
                </code>
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
                  <span>Telemetry</span>
                </button>
              </div>

              {isOpen && (
                <div
                  className="adapter-metrics-drawer"
                  data-testid={`adapter-metrics-${domain.domain_id}`}
                >
                  <div className="metric-kv">
                    <span>Cartridge</span>
                    <code title={cartridgeLabel}>{cartridgeLabel}</code>
                  </div>
                  <div className="metric-kv">
                    <span>Rank / Alpha</span>
                    <code>
                      {domain.lora_enabled &&
                      domain.qlora_rank !== null &&
                      domain.qlora_alpha !== null
                        ? `r=${domain.qlora_rank}, α=${domain.qlora_alpha}`
                        : "Base Gemma-4-26B"}
                    </code>
                  </div>
                  <div className="metric-kv">
                    <span>PG17 Corpus</span>
                    <code className="tabular-num">
                      {domain.documents_count} docs · {domain.code_files_count} files
                    </code>
                  </div>
                  <div className="metric-kv">
                    <span>Training Loss</span>
                    <code className="tabular-num">
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
