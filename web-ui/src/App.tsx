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
  Map as MapIcon,
  RefreshCw,
  Shield,
  ShieldCheck,
  SlidersHorizontal,
  Ticket,
  Wrench,
} from "lucide-react";
import {
  ApiClientError,
  exchangeAuthSession,
  FEDERATED_ALL_DOMAIN_ID,
  fetchCurrentPrincipal,
  fetchDomains,
  fetchHealth,
  fetchROIMetrics,
  fetchWorkspaceConfig,
  mapDomainDetailToInfo,
} from "./api/client";
import { ArchitectureMapViewer } from "./components/ArchitectureMapViewer";
import { CitationInspector } from "./components/CitationInspector";
import {
  DomainSelector,
  matchesDomainSourceFilter,
} from "./components/DomainSelector";
import { FrontierHandoffModal } from "./components/FrontierHandoffModal";
import { ProductTruthView } from "./components/ProductTruthView";
import { ROLE_DEFAULT_SCOPES, RoleSwitcher } from "./components/RoleSwitcher";
import { ScoutConsole } from "./components/ScoutConsole";
import { SessionBranchBar } from "./components/SessionBranchBar";
import { SpecializationAdminView } from "./components/SpecializationAdminView";
import { TokenSavingsMeter } from "./components/TokenSavingsMeter";
import {
  BranchOverlayOption,
  ContextPack,
  DomainInfo,
  HealthResponse,
  InsufficientContextDiagnostic,
  PrincipalContext,
  ROIMetricsResponse,
  ScoutSessionSummary,
  ScoutTokenSavingsBreakdown,
  SourceFilterType,
  WorkspaceConfigResponse,
} from "./types/contracts";

type WorkbenchView =
  | "workbench"
  | "product_truth"
  | "architecture_map"
  | "specialization_admin";

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

const WORKBENCH_TABS: Array<{
  id: WorkbenchView;
  label: string;
  testId: string;
}> = [
  {
    id: "workbench",
    label: "Scout Workbench",
    testId: "view-tab-workbench",
  },
  {
    id: "product_truth",
    label: "Product Truth",
    testId: "view-tab-product-truth",
  },
  {
    id: "architecture_map",
    label: "Architecture Map",
    testId: "view-tab-architecture-map",
  },
  {
    id: "specialization_admin",
    label: "Specialization Admin",
    testId: "view-tab-specialization-admin",
  },
];

const DEFAULT_BRANCHES: BranchOverlayOption[] = [
  {
    branch_id: "main",
    label: "HEAD (main)",
    kind: "head",
    commit_sha: "HEAD",
  },
  {
    branch_id: "mr-overlay-active",
    label: "MR Overlay (CoW Working Tree)",
    kind: "mr_overlay",
    commit_sha: "overlay",
    mr_iid: "MR-active",
  },
];

function createSessionId(): string {
  return `sess-${Date.now().toString(36)}`;
}

function principalsEqual(a: PrincipalContext, b: PrincipalContext): boolean {
  if (
    a.user_id !== b.user_id ||
    a.role !== b.role ||
    a.max_session_paid_tokens !== b.max_session_paid_tokens ||
    a.allowed_scopes.length !== b.allowed_scopes.length
  ) {
    return false;
  }
  return a.allowed_scopes.every((s, idx) => s === b.allowed_scopes[idx]);
}

export function App(): React.ReactElement {
  const [principal, setPrincipal] = useState<PrincipalContext>({
    user_id: "demo_engineer",
    role: "engineer",
    allowed_scopes: [...ROLE_DEFAULT_SCOPES.engineer],
    max_session_paid_tokens: 64000,
  });
  const [isProductionMode, setIsProductionMode] = useState<boolean>(false);
  const [activeView, setActiveView] = useState<WorkbenchView>("workbench");

  const [selectedDomain, setSelectedDomain] = useState<string>(
    FEDERATED_ALL_DOMAIN_ID
  );
  const [sourceFilter, setSourceFilter] = useState<SourceFilterType>("all");
  const [scopePanelOpen, setScopePanelOpen] = useState<boolean>(true);
  const [scopeResetVersion, setScopeResetVersion] = useState<number>(0);

  const [initialSessionId] = useState<string>(() => createSessionId());
  const [sessions, setSessions] = useState<ScoutSessionSummary[]>(() => [
    {
      session_id: initialSessionId,
      title: "Current Workspace Session",
      domain_id: FEDERATED_ALL_DOMAIN_ID,
      turn_count: 0,
      compacted_tokens: 0,
      updated_at: new Date().toISOString(),
      has_commit_drift: false,
    },
  ]);
  const [activeSessionId, setActiveSessionId] =
    useState<string>(initialSessionId);
  const [selectedBranchId, setSelectedBranchId] = useState<string>("main");

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
  const [tokenSavings, setTokenSavings] =
    useState<ScoutTokenSavingsBreakdown | null>(null);
  const [abstainedDiagnostic, setAbstainedDiagnostic] =
    useState<InsufficientContextDiagnostic | null>(null);
  const [diffModalOpen, setDiffModalOpen] = useState<boolean>(false);
  const initialScopesHydratedRef = useRef<boolean>(false);

  const loadHealth = useCallback(async (): Promise<HealthResponse | null> => {
    setHealthError(null);
    try {
      const h = await fetchHealth();
      setHealth(h);
      if (h.hardware_profile?.runtime_mode === "production") {
        setIsProductionMode(true);
      }
      return h;
    } catch (err: unknown) {
      setHealth(null);
      setHealthError(
        err instanceof Error ? err.message : "Health endpoint unavailable"
      );
      return null;
    }
  }, []);

  const loadRoi = useCallback(
    async (activePrincipal?: PrincipalContext): Promise<void> => {
      setRoiLoading(true);
      setRoiError(null);
      try {
        const data = await fetchROIMetrics(activePrincipal);
        setRoiMetrics(data);
      } catch (err: unknown) {
        setRoiError(
          err instanceof Error ? err.message : "Failed to load ROI metrics"
        );
      } finally {
        setRoiLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    void loadHealth();
  }, [loadHealth]);

  useEffect(() => {
    if (authStatus === "ready") {
      void loadRoi(isProductionMode ? undefined : principal);
    } else if (authStatus === "error") {
      void loadRoi();
    }
  }, [authStatus, isProductionMode, principal, loadRoi]);

  useEffect(() => {
    let cancelled = false;
    setAuthStatus("authenticating");
    setAuthError(null);
    setDomainsLoading(true);
    setDomainsError(null);

    const hydrateProductionIdentity = async (): Promise<void> => {
      setIsProductionMode(true);
      try {
        const me = await fetchCurrentPrincipal();
        if (cancelled) return;
        setPrincipal((prev) => (principalsEqual(prev, me) ? prev : me));
        setAuthStatus("ready");
        try {
          const domainList = await fetchDomains();
          if (cancelled) return;
          setDomains(domainList);
          setDomainsError(null);
        } catch (domErr: unknown) {
          if (cancelled) return;
          setDomains([]);
          setDomainsError(
            domErr instanceof Error
              ? domErr.message
              : "Domains endpoint unavailable"
          );
        }
      } catch (meErr: unknown) {
        if (cancelled) return;
        setAuthStatus("error");
        setAuthError(
          meErr instanceof Error
            ? meErr.message
            : "Production OIDC / IAP identity verification failed."
        );
        setDomains([]);
        setDomainsError("Authentication required to load workspace domains.");
      } finally {
        if (!cancelled) {
          setDomainsLoading(false);
        }
      }
    };

    if (
      isProductionMode ||
      health?.hardware_profile?.runtime_mode === "production"
    ) {
      void hydrateProductionIdentity();
      return () => {
        cancelled = true;
      };
    }

    void exchangeAuthSession(principal)
      .then(async (session) => {
        if (cancelled) return;

        // Verify /api/v1/auth/me in dev mode (gracefully fall back to session.principal if 404)
        let activePrincipal = session.principal ?? principal;
        try {
          const verifiedMe = await fetchCurrentPrincipal(principal);
          if (cancelled) return;
          activePrincipal = verifiedMe;
        } catch (meErr: unknown) {
          if (
            !(meErr instanceof ApiClientError && meErr.status === 404)
          ) {
            // Keep session.principal when /auth/me is unavailable in legacy mocks
          }
        }

        setAuthStatus("ready");
        try {
          const cfg = await fetchWorkspaceConfig(activePrincipal);
          if (cancelled) return;
          setWorkspaceConfig(cfg);
          if (cfg.runtime_mode === "production") {
            setIsProductionMode(true);
          }
          setDomains(cfg.domains.map(mapDomainDetailToInfo));
          setDomainsError(null);
          if (!initialScopesHydratedRef.current) {
            initialScopesHydratedRef.current = true;
            const matchedPreset = cfg.role_presets.find(
              (r) => r.role === activePrincipal.role
            );
            if (matchedPreset && matchedPreset.default_scopes.length > 0) {
              const sameLength =
                matchedPreset.default_scopes.length ===
                activePrincipal.allowed_scopes.length;
              const sameSet =
                sameLength &&
                matchedPreset.default_scopes.every((s) =>
                  activePrincipal.allowed_scopes.includes(s)
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
        if (err instanceof ApiClientError && err.status === 403) {
          void hydrateProductionIdentity();
          return;
        }
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
  }, [
    principal,
    authAttempt,
    configAttempt,
    isProductionMode,
    health?.hardware_profile?.runtime_mode,
  ]);

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
      setTokenSavings(null);
      setAbstainedDiagnostic(null);
      setScopeResetVersion((v) => v + 1);
    }
    setPrincipal(next);
  };

  const handleContextPackReady = (
    pack: ContextPack | null,
    savings: ScoutTokenSavingsBreakdown | null,
    diagnostic?: InsufficientContextDiagnostic | null
  ): void => {
    setContextPack(pack);
    setTokenSavings(savings);
    setAbstainedDiagnostic(diagnostic ?? null);

    if (pack) {
      const targetSessionId = pack.session_id ?? activeSessionId;
      const driftWarnings = pack.commit_drift_warnings ?? [];
      setSessions((prev) => {
        const idx = prev.findIndex((s) => s.session_id === targetSessionId);
        const nextTitle =
          pack.user_task.length > 48
            ? `${pack.user_task.slice(0, 45)}...`
            : pack.user_task;
        if (idx === -1) {
          return [
            {
              session_id: targetSessionId,
              title: nextTitle || "Scout Session",
              domain_id: pack.domain_id,
              turn_count: 1,
              compacted_tokens: pack.pack_token_count,
              updated_at: new Date().toISOString(),
              has_commit_drift: driftWarnings.length > 0,
              commit_drift_details: driftWarnings,
            },
            ...prev,
          ];
        }
        return prev.map((s) =>
          s.session_id === targetSessionId
            ? {
                ...s,
                title: s.turn_count === 0 && nextTitle ? nextTitle : s.title,
                domain_id: pack.domain_id,
                turn_count: s.turn_count + 1,
                compacted_tokens: s.compacted_tokens + pack.pack_token_count,
                updated_at: new Date().toISOString(),
                has_commit_drift:
                  s.has_commit_drift || driftWarnings.length > 0,
                commit_drift_details:
                  driftWarnings.length > 0
                    ? driftWarnings
                    : s.commit_drift_details,
              }
            : s
        );
      });
      if (pack.session_id) {
        setActiveSessionId(pack.session_id);
      }
    }
  };

  const handleClearContextPack = useCallback((): void => {
    setContextPack(null);
    setTokenSavings(null);
    setAbstainedDiagnostic(null);
  }, []);

  const handleQueryComplete = (): void => {
    void loadRoi(isProductionMode ? undefined : principal);
  };

  const handleNewSession = (): void => {
    const newId = createSessionId();
    setSessions((prev) => [
      {
        session_id: newId,
        title: `Session ${prev.length + 1}`,
        domain_id: selectedDomain,
        turn_count: 0,
        compacted_tokens: 0,
        updated_at: new Date().toISOString(),
        has_commit_drift: false,
      },
      ...prev,
    ]);
    setActiveSessionId(newId);
    setContextPack(null);
    setTokenSavings(null);
    setAbstainedDiagnostic(null);
  };

  const handleSelectDomain = useCallback(
    (nextDomain: string): void => {
      setSelectedDomain(nextDomain);
      if (nextDomain !== selectedDomain) {
        const freshId = createSessionId();
        setSessions((prev) => [
          {
            session_id: freshId,
            title:
              nextDomain === FEDERATED_ALL_DOMAIN_ID
                ? "Federated Session"
                : `${nextDomain} Session`,
            domain_id: nextDomain,
            turn_count: 0,
            compacted_tokens: 0,
            updated_at: new Date().toISOString(),
            has_commit_drift: false,
          },
          ...prev,
        ]);
        setActiveSessionId(freshId);
      }
    },
    [selectedDomain]
  );

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
            <Shield size={18} aria-hidden="true" />
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
              <Cpu size={14} className="icon-accent" aria-hidden="true" />
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
              <Activity size={14} aria-hidden="true" />
              <span>{healthError ?? "Connecting to runtime..."}</span>
            </div>
          )}

          <div
            className={`telemetry-chip ${
              dbConnected ? "telemetry-chip-ok" : "telemetry-chip-danger"
            }`}
            data-testid="health-db-badge"
          >
            <Database size={14} aria-hidden="true" />
            <span>
              {health || workspaceConfig
                ? dbConnected
                  ? pgVersionLabel
                  : `DB Disconnected (${dbStatus})`
                : "PG17 Status Pending"}
            </span>
          </div>

          {isProductionMode ? (
            <div
              className={`telemetry-chip ${
                authStatus === "ready"
                  ? "telemetry-chip-ok"
                  : authStatus === "error"
                    ? "telemetry-chip-danger"
                    : "telemetry-chip-info"
              }`}
              data-testid="production-oidc-badge"
            >
              <ShieldCheck size={14} aria-hidden="true" />
              <span>
                {authStatus === "ready"
                  ? `OIDC: ${principal.user_id} (${principal.role})`
                  : authStatus === "authenticating"
                    ? "Verifying OIDC..."
                    : "OIDC Auth Error"}
              </span>
            </div>
          ) : (
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
              <KeyRound size={14} aria-hidden="true" />
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
                  aria-label="Retry authentication"
                  onClick={() => setAuthAttempt((n) => n + 1)}
                >
                  <RefreshCw size={11} aria-hidden="true" />
                </button>
              )}
            </div>
          )}
        </div>
      </header>

      <main className="workspace-stack">
        <section className="top-scope-bar" data-testid="top-scope-bar">
          <div className="scope-summary-bar">
            <div className="scope-summary-text">
              <SlidersHorizontal size={14} className="icon-accent" aria-hidden="true" />
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
                  <ChevronDown size={13} aria-hidden="true" /> Hide Scope
                </>
              ) : (
                <>
                  <ChevronRight size={13} aria-hidden="true" /> Configure Scope
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
                        {f.id === "all" && <Layers size={13} aria-hidden="true" />}
                        {f.id === "code" && <Code size={13} aria-hidden="true" />}
                        {f.id === "wiki" && <BookOpen size={13} aria-hidden="true" />}
                        {f.id === "jira" && <Ticket size={13} aria-hidden="true" />}
                        <span>{f.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="scope-controls-grid">
                {!isProductionMode && (
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
                )}

                <DomainSelector
                  domains={domains}
                  selectedDomain={selectedDomain}
                  onSelectDomain={handleSelectDomain}
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

        <SessionBranchBar
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelectSession={setActiveSessionId}
          onNewSession={handleNewSession}
          branches={DEFAULT_BRANCHES}
          selectedBranchId={selectedBranchId}
          onSelectBranch={setSelectedBranchId}
          commitDriftWarnings={contextPack?.commit_drift_warnings}
        />

        <nav
          className="workbench-view-switcher"
          role="tablist"
          aria-label="Workbench views"
          data-testid="workbench-view-switcher"
        >
          {WORKBENCH_TABS.map((tab) => {
            const active = activeView === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                role="tab"
                id={`tab-${tab.id}`}
                aria-selected={active}
                aria-controls={`tabpanel-${tab.id}`}
                className={`workbench-view-tab ${
                  active ? "workbench-view-tab-active" : ""
                }`}
                onClick={() => setActiveView(tab.id)}
                data-testid={tab.testId}
              >
                {tab.id === "workbench" && <Code size={14} aria-hidden="true" />}
                {tab.id === "product_truth" && <ShieldCheck size={14} aria-hidden="true" />}
                {tab.id === "architecture_map" && <MapIcon size={14} aria-hidden="true" />}
                {tab.id === "specialization_admin" && <Wrench size={14} aria-hidden="true" />}
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>

        {activeView === "workbench" && (
          <section
            id="tabpanel-workbench"
            role="tabpanel"
            aria-labelledby="tab-workbench"
            className="workspace-main"
          >
            <ScoutConsole
              selectedDomain={selectedDomain}
              principal={principal}
              domains={domains}
              pipelineStages={workspaceConfig?.pipeline_stages}
              sourceFilter={sourceFilter}
              scopeResetVersion={scopeResetVersion}
              activeSessionId={activeSessionId}
              selectedBranchId={selectedBranchId}
              onSelectDomain={setSelectedDomain}
              onContextPackReady={handleContextPackReady}
              onClearContextPack={handleClearContextPack}
              onQueryStart={handleClearContextPack}
              onQueryComplete={handleQueryComplete}
            />

            <CitationInspector
              contextPack={contextPack}
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
        )}

        {activeView === "product_truth" && (
          <div
            id="tabpanel-product_truth"
            role="tabpanel"
            aria-labelledby="tab-product_truth"
          >
            <ProductTruthView
              contextPack={contextPack}
              principalRole={principal.role}
              selectedDomain={selectedDomain}
            />
          </div>
        )}

        {activeView === "architecture_map" && (
          <div
            id="tabpanel-architecture_map"
            role="tabpanel"
            aria-labelledby="tab-architecture_map"
          >
            <ArchitectureMapViewer
              contextPack={contextPack}
              selectedDomain={selectedDomain}
            />
          </div>
        )}

        {activeView === "specialization_admin" && (
          <div
            id="tabpanel-specialization_admin"
            role="tabpanel"
            aria-labelledby="tab-specialization_admin"
          >
            <SpecializationAdminView
              domains={domains}
              principal={principal}
              hardwareProfile={hw}
              adapterPoolStatus={health?.adapter_pool_status}
            />
          </div>
        )}
      </main>

      {contextPack && (
        <FrontierHandoffModal
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
