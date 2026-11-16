import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Code,
  Cpu,
  Database,
  KeyRound,
  Layers,
  RefreshCw,
  Shield,
  SlidersHorizontal,
  Ticket,
} from "lucide-react";
import {
  exchangeAuthSession,
  FEDERATED_ALL_DOMAIN_ID,
  fetchHealth,
  fetchROIMetrics,
  fetchWorkspaceConfig,
  mapDomainDetailToInfo,
} from "./api/client";
import { CitationInspector } from "./components/CitationInspector";
import {
  DomainSelector,
  matchesDomainSourceFilter,
} from "./components/DomainSelector";
import { GeminiDiffModal } from "./components/GeminiDiffModal";
import { ROLE_DEFAULT_SCOPES, RoleSwitcher } from "./components/RoleSwitcher";
import { ScoutConsole } from "./components/ScoutConsole";
import { TokenSavingsMeter } from "./components/TokenSavingsMeter";
import {
  ContextPack,
  DomainInfo,
  HealthResponse,
  InsufficientContextDiagnostic,
  PrincipalContext,
  QuantStrategyBrief,
  ROIMetricsResponse,
  ScoutTokenSavingsBreakdown,
  SourceFilterType,
  WorkspaceConfigResponse,
} from "./types/contracts";

const SOURCE_FILTERS: Array<{
  id: SourceFilterType;
  label: string;
  testId: string;
}> = [
  { id: "all", label: "All Federated", testId: "source-filter-all" },
  { id: "code", label: "Code Repos", testId: "source-filter-code" },
  { id: "wiki", label: "Confluence & Docs", testId: "source-filter-wiki" },
  { id: "jira", label: "Jira & Issues", testId: "source-filter-jira" },
];

export function App(): React.ReactElement {
  const [principal, setPrincipal] = useState<PrincipalContext>({
    user_id: "demo_engineer",
    role: "engineer",
    allowed_scopes: [...ROLE_DEFAULT_SCOPES.engineer],
    max_session_paid_tokens: 64000,
  });
  const [selectedDomain, setSelectedDomain] = useState<string>(
    FEDERATED_ALL_DOMAIN_ID
  );
  const [sourceFilter, setSourceFilter] = useState<SourceFilterType>("all");
  const [scopePanelOpen, setScopePanelOpen] = useState<boolean>(true);
  const [scopeResetVersion, setScopeResetVersion] = useState<number>(0);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [workspaceConfig, setWorkspaceConfig] =
    useState<WorkspaceConfigResponse | null>(null);
  const [domains, setDomains] = useState<DomainInfo[]>([]);
  const [domainsLoading, setDomainsLoading] = useState<boolean>(true);
  const [domainsError, setDomainsError] = useState<string | null>(null);
  const [configAttempt, setConfigAttempt] = useState<number>(0);

  const [roiMetrics, setRoiMetrics] = useState<ROIMetricsResponse | null>(null);
  const [roiLoading, setRoiLoading] = useState<boolean>(true);
  const [roiError, setRoiError] = useState<string | null>(null);

  const [authStatus, setAuthStatus] = useState<
    "authenticating" | "ready" | "error"
  >("authenticating");
  const [authError, setAuthError] = useState<string | null>(null);
  const [authAttempt, setAuthAttempt] = useState<number>(0);

  const [contextPack, setContextPack] = useState<ContextPack | null>(null);
  const [quantSpec, setQuantSpec] = useState<QuantStrategyBrief | null>(null);
  const [tokenSavings, setTokenSavings] =
    useState<ScoutTokenSavingsBreakdown | null>(null);
  const [abstainedDiagnostic, setAbstainedDiagnostic] =
    useState<InsufficientContextDiagnostic | null>(null);
  const [diffModalOpen, setDiffModalOpen] = useState<boolean>(false);
  const initialScopesHydratedRef = useRef<boolean>(false);

  const loadHealth = useCallback(async (): Promise<void> => {
    setHealthError(null);
    try {
      const h = await fetchHealth();
      setHealth(h);
    } catch (err: unknown) {
      setHealth(null);
      setHealthError(
        err instanceof Error ? err.message : "Health endpoint unavailable"
      );
    }
  }, []);

  const loadRoi = useCallback(async (): Promise<void> => {
    setRoiLoading(true);
    setRoiError(null);
    try {
      const data = await fetchROIMetrics();
      setRoiMetrics(data);
    } catch (err: unknown) {
      setRoiError(
        err instanceof Error ? err.message : "Failed to load ROI metrics"
      );
    } finally {
      setRoiLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHealth();
    void loadRoi();
  }, [loadHealth, loadRoi]);

  useEffect(() => {
    let cancelled = false;
    setAuthStatus("authenticating");
    setAuthError(null);
    setDomainsLoading(true);
    setDomainsError(null);

    void exchangeAuthSession(principal)
      .then(async () => {
        if (cancelled) return;
        setAuthStatus("ready");
        try {
          const cfg = await fetchWorkspaceConfig(principal);
          if (cancelled) return;
          setWorkspaceConfig(cfg);
          setDomains(cfg.domains.map(mapDomainDetailToInfo));
          setDomainsError(null);
          if (!initialScopesHydratedRef.current) {
            initialScopesHydratedRef.current = true;
            const matchedPreset = cfg.role_presets.find(
              (r) => r.role === principal.role
            );
            if (matchedPreset && matchedPreset.default_scopes.length > 0) {
              const sameLength =
                matchedPreset.default_scopes.length ===
                principal.allowed_scopes.length;
              const sameSet =
                sameLength &&
                matchedPreset.default_scopes.every((s) =>
                  principal.allowed_scopes.includes(s)
                );
              if (!sameSet) {
                setPrincipal((prev) => ({
                  ...prev,
                  allowed_scopes: [...matchedPreset.default_scopes],
                }));
              }
            }
          }
        } catch (cfgErr: unknown) {
          if (cancelled) return;
          setWorkspaceConfig(null);
          setDomains([]);
          setDomainsError(
            cfgErr instanceof Error
              ? cfgErr.message
              : "Workspace config unavailable"
          );
        } finally {
          if (!cancelled) {
            setDomainsLoading(false);
          }
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setAuthStatus("error");
        setAuthError(
          err instanceof Error
            ? err.message
            : "Signed Bearer authentication failed."
        );
        setDomains([]);
        setDomainsError("Authentication required to load workspace domains.");
        setDomainsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [principal, authAttempt, configAttempt]);

  const handleChangePrincipal = (next: PrincipalContext): void => {
    initialScopesHydratedRef.current = true;
    if (next.role !== "engineer") {
      setDiffModalOpen(false);
    }
    const scopesShrank = principal.allowed_scopes.some(
      (s) => !next.allowed_scopes.includes(s)
    );
    if (scopesShrank) {
      setContextPack(null);
      setQuantSpec(null);
      setTokenSavings(null);
      setAbstainedDiagnostic(null);
      setScopeResetVersion((v) => v + 1);
    }
    setPrincipal(next);
  };

  const handleContextPackReady = (
    pack: ContextPack | null,
    savings: ScoutTokenSavingsBreakdown | null,
    spec: QuantStrategyBrief | null,
    diagnostic?: InsufficientContextDiagnostic | null
  ): void => {
    setContextPack(pack);
    setTokenSavings(savings);
    setQuantSpec(spec);
    setAbstainedDiagnostic(diagnostic ?? null);
  };

  const handleClearContextPack = useCallback((): void => {
    setContextPack(null);
    setQuantSpec(null);
    setTokenSavings(null);
  }, []);

  const handleQueryComplete = (): void => {
    void loadRoi();
  };

  const handleSelectSourceFilter = (nextFilter: SourceFilterType): void => {
    setSourceFilter(nextFilter);
    if (nextFilter === "all" || selectedDomain === FEDERATED_ALL_DOMAIN_ID) {
      return;
    }
    const current = domains.find((d) => d.domain_id === selectedDomain);
    if (current && matchesDomainSourceFilter(current, nextFilter)) {
      return;
    }
    const firstMatch = domains.find((d) =>
      matchesDomainSourceFilter(d, nextFilter)
    );
    setSelectedDomain(
      firstMatch ? firstMatch.domain_id : FEDERATED_ALL_DOMAIN_ID
    );
  };

  const availableDomainScopes = useMemo(() => {
    const scopes = new Set<string>();
    for (const d of domains) {
      for (const sc of d.default_scopes) {
        if (sc) scopes.add(sc);
      }
    }
    return Array.from(scopes);
  }, [domains]);

  const hw = health?.hardware_profile ?? workspaceConfig?.hardware_profile;
  const adapterEntries = Object.values(health?.adapter_pool_status ?? {});
  const verifiedAdapterCount = adapterEntries.filter((a) => a.is_valid).length;
  const dbConnected =
    health?.database_connected ?? workspaceConfig?.database_connected ?? false;
  const dbStatus =
    health?.database_status ?? workspaceConfig?.database_status ?? "";
  const pgVersionLabel = hw?.postgres_version
    ? `PG ${hw.postgres_version} RLS`
    : "PG17 RLS Connected";
  const runtimeMode = hw?.runtime_mode ?? workspaceConfig?.runtime_mode;

  return (
    <div className="app-shell">
      <header className="top-bar" data-testid="app-header">
        <div className="brand-block">
          <div className="brand-icon-wrap">
            <Shield size={18} />
          </div>
          <div>
            <h1 className="brand-title">Enterprise Context Scout</h1>
            <p className="brand-subtitle">
              Local-First Grounded Engineering &amp; Research Workbench
            </p>
          </div>
        </div>

        <div className="header-telemetry-strip">
          {hw ? (
            <div
              className="telemetry-chip tabular-num"
              data-testid="hardware-profile-chip"
            >
              <Cpu size={14} className="icon-accent" />
              <span>
                {hw.device}, {hw.unified_memory}, {hw.quantization}
                {adapterEntries.length > 0
                  ? `, ${verifiedAdapterCount}/${adapterEntries.length} Adapters`
                  : ""}
                {`, LoRA ${hw.lora_slots_allocated}/${hw.lora_slots_capacity}`}
                {runtimeMode ? ` · ${runtimeMode}` : ""}
              </span>
            </div>
          ) : (
            <div className="telemetry-chip telemetry-chip-warn">
              <Activity size={14} />
              <span>{healthError ?? "Connecting to runtime..."}</span>
            </div>
          )}

          <div
            className={`telemetry-chip ${
              dbConnected ? "telemetry-chip-ok" : "telemetry-chip-danger"
            }`}
            data-testid="health-db-badge"
          >
            <Database size={14} />
            <span>
              {health || workspaceConfig
                ? dbConnected
                  ? pgVersionLabel
                  : `DB Disconnected (${dbStatus})`
                : "PG17 Status Pending"}
            </span>
          </div>

          <div
            className={`telemetry-chip ${
              authStatus === "ready"
                ? "telemetry-chip-ok"
                : authStatus === "error"
                  ? "telemetry-chip-danger"
                  : "telemetry-chip-info"
            }`}
            data-testid="header-auth-badge"
          >
            <KeyRound size={14} />
            <span>
              {authStatus === "ready"
                ? `Bearer: ${principal.role}`
                : authStatus === "authenticating"
                  ? "Signing Session..."
                  : "Bearer Error"}
            </span>
            {authStatus === "error" && (
              <button
                type="button"
                className="btn-header-retry"
                onClick={() => setAuthAttempt((n) => n + 1)}
              >
                <RefreshCw size={11} />
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="workspace-stack">
        <section className="top-scope-bar" data-testid="top-scope-bar">
          <div className="scope-summary-bar">
            <div className="scope-summary-text">
              <SlidersHorizontal size={14} className="icon-accent" />
              <span>
                Target: <code>{selectedDomain}</code> | Role:{" "}
                <strong>{principal.role}</strong> (
                <code>{principal.allowed_scopes.join(", ")}</code>)
              </span>
            </div>
            <button
              type="button"
              className="btn-quiet-toggle"
              onClick={() => setScopePanelOpen((v) => !v)}
              aria-expanded={scopePanelOpen}
              data-testid="toggle-scope-bar-btn"
            >
              {scopePanelOpen ? (
                <>
                  <ChevronDown size={13} /> Hide Scope
                </>
              ) : (
                <>
                  <ChevronRight size={13} /> Configure Scope
                </>
              )}
            </button>
          </div>

          {scopePanelOpen && (
            <>
              <div className="source-filter-bar-row">
                <span className="field-label">Source Scope</span>
                <div
                  className="source-filter-bar"
                  role="group"
                  aria-label="Source filter"
                  data-testid="source-filter-bar"
                >
                  {SOURCE_FILTERS.map((f) => {
                    const active = sourceFilter === f.id;
                    return (
                      <button
                        key={f.id}
                        type="button"
                        className={`source-filter-btn ${
                          active ? "source-filter-btn-active" : ""
                        }`}
                        onClick={() => handleSelectSourceFilter(f.id)}
                        aria-pressed={active}
                        data-testid={f.testId}
                      >
                        {f.id === "all" && <Layers size={13} />}
                        {f.id === "code" && <Code size={13} />}
                        {f.id === "wiki" && <BookOpen size={13} />}
                        {f.id === "jira" && <Ticket size={13} />}
                        <span>{f.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="scope-controls-grid">
                <RoleSwitcher
                  principal={principal}
                  onChangePrincipal={handleChangePrincipal}
                  rolePresets={workspaceConfig?.role_presets}
                  availableScopes={availableDomainScopes}
                  authStatus={authStatus}
                  authError={authError}
                  onRetryAuth={() => {
                    if (!health) void loadHealth();
                    setAuthAttempt((n) => n + 1);
                  }}
                />

                <DomainSelector
                  domains={domains}
                  selectedDomain={selectedDomain}
                  onSelectDomain={setSelectedDomain}
                  federatedOption={workspaceConfig?.federated_option}
                  sourceFilter={sourceFilter}
                  loading={domainsLoading}
                  error={domainsError}
                  onRetry={() => {
                    void loadHealth();
                    setConfigAttempt((n) => n + 1);
                  }}
                />
              </div>
            </>
          )}
        </section>

        <section className="workspace-main">
          <ScoutConsole
            selectedDomain={selectedDomain}
            principal={principal}
            domains={domains}
            pipelineStages={workspaceConfig?.pipeline_stages}
            sourceFilter={sourceFilter}
            scopeResetVersion={scopeResetVersion}
            onSelectDomain={setSelectedDomain}
            onContextPackReady={handleContextPackReady}
            onClearContextPack={handleClearContextPack}
            onQueryComplete={handleQueryComplete}
          />

          <CitationInspector
            contextPack={contextPack}
            quantSpec={quantSpec}
            tokenSavings={tokenSavings}
            principalRole={principal.role}
            sourceFilter={sourceFilter}
            onOpenDiffModal={() => setDiffModalOpen(true)}
            abstainedDiagnostic={abstainedDiagnostic}
          />

          <TokenSavingsMeter
            currentQuerySavings={tokenSavings}
            roiMetrics={roiMetrics}
            roiLoading={roiLoading}
            roiError={roiError}
            defaultExpanded={false}
          />
        </section>
      </main>

      {contextPack && (
        <GeminiDiffModal
          isOpen={diffModalOpen}
          onClose={() => setDiffModalOpen(false)}
          contextPack={contextPack}
          principal={principal}
        />
      )}
    </div>
  );
}

export default App;
