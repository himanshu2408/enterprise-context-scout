import React, { useState, useEffect } from "react";
import {
  Terminal,
  Cpu,
  Layers,
  Sparkles,
} from "lucide-react";
import {
  ContextPack,
  DomainInfo,
  PrincipalContext,
  ROIMetricsResponse,
  SSEContextPackReadyData,
} from "./types/contracts";
import {
  fetchDomains,
  fetchROIMetrics,
  MOCK_CONTEXT_PACK,
  MOCK_DOMAINS,
  MOCK_ROI_METRICS,
} from "./api/client";
import { RoleSwitcher, ROLE_DEFAULT_SCOPES } from "./components/RoleSwitcher";
import { DomainSelector } from "./components/DomainSelector";
import { ScoutConsole } from "./components/ScoutConsole";
import { CitationInspector } from "./components/CitationInspector";
import { TokenSavingsMeter } from "./components/TokenSavingsMeter";
import { GeminiDiffModal } from "./components/GeminiDiffModal";

export const App: React.FC = () => {
  const [principal, setPrincipal] = useState<PrincipalContext>({
    user_id: "demo_engineer",
    role: "engineer",
    allowed_scopes: ROLE_DEFAULT_SCOPES.engineer,
    max_session_paid_tokens: 16000,
  });

  const [selectedDomainId, setSelectedDomainId] = useState<string>(
    "trading_code_scout"
  );
  const [contextPack, setContextPack] = useState<ContextPack | null>(
    MOCK_CONTEXT_PACK
  );
  const [latestTelemetry, setLatestTelemetry] = useState<
    SSEContextPackReadyData["telemetry"] | null
  >(null);
  const [domains, setDomains] = useState<DomainInfo[]>(MOCK_DOMAINS);
  const [roiMetrics, setRoiMetrics] = useState<ROIMetricsResponse>(
    MOCK_ROI_METRICS
  );
  const [isHandoffModalOpen, setIsHandoffModalOpen] = useState(false);

  useEffect(() => {
    fetchDomains().then(setDomains);
    fetchROIMetrics().then(setRoiMetrics);
  }, []);

  const handleContextPackReady = (
    pack: ContextPack,
    telemetry?: SSEContextPackReadyData["telemetry"]
  ): void => {
    setContextPack(pack);
    if (telemetry) {
      setLatestTelemetry(telemetry);
    }
  };

  return (
    <div className="app-root">
      {/* Top Navigation Bar */}
      <header className="app-header">
        <div className="header-left">
          <div className="brand-logo">
            <Terminal className="icon-emerald" size={22} />
            <span className="brand-title">Enterprise Context Scout</span>
          </div>
          <span className="badge badge-zinc text-xs">v1.0.0 M8</span>
          <span className="badge badge-sky text-xs">
            <Cpu size={12} className="mr-1" />
            Apple M5 Pro &middot; 48GB Unified Memory
          </span>
        </div>

        <div className="header-right">
          <span className="badge badge-emerald text-xs">
            <Sparkles size={12} className="mr-1" />
            ShieldGemma 2 Guardrails Active
          </span>
          <span className="badge badge-purple text-xs">
            <Layers size={12} className="mr-1" />
            PostgreSQL 17 RLS Enforced
          </span>
        </div>
      </header>

      {/* Main Container */}
      <main className="app-main">
        {/* Row 1: Identity & Domain Selection */}
        <section className="grid-two-col mb-4">
          <RoleSwitcher
            principal={principal}
            onPrincipalChange={setPrincipal}
          />
          <DomainSelector
            selectedDomainId={selectedDomainId}
            onDomainChange={setSelectedDomainId}
            domains={domains}
          />
        </section>

        {/* Row 2: Scout Query Console */}
        <section className="mb-4">
          <ScoutConsole
            selectedDomainId={selectedDomainId}
            principal={principal}
            onContextPackReady={handleContextPackReady}
            onOpenHandoffModal={() => setIsHandoffModalOpen(true)}
          />
        </section>

        {/* Row 3: Token Savings ROI Meter */}
        <section className="mb-4">
          <TokenSavingsMeter
            telemetry={latestTelemetry}
            aggregatedMetrics={roiMetrics}
          />
        </section>

        {/* Row 4: Deterministic Citation & AST Closure Inspector */}
        <section className="mb-4">
          <CitationInspector contextPack={contextPack} />
        </section>
      </main>

      {/* Gemini Diff Modal */}
      <GeminiDiffModal
        isOpen={isHandoffModalOpen}
        onClose={() => setIsHandoffModalOpen(false)}
        contextPack={contextPack}
        principal={principal}
      />
    </div>
  );
};

export default App;
