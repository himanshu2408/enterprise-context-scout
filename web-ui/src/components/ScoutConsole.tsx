import React, { useState, useRef, useEffect } from "react";
import {
  Terminal,
  Play,
  RotateCw,
  ShieldCheck,
  ShieldAlert,
  Sparkles,
  ArrowRight,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Radio,
} from "lucide-react";
import {
  ContextPack,
  ExecutionMode,
  PrincipalContext,
  SSEBounceBackData,
  SSEContextPackReadyData,
  SSEErrorData,
  SSEStageEvent,
  SSEStageUpdateData,
} from "../types/contracts";
import { executeScoutQuery } from "../api/client";
import { streamScoutWorkflow } from "../api/sse";

export interface ScoutConsoleProps {
  selectedDomainId: string;
  principal: PrincipalContext;
  onContextPackReady: (
    pack: ContextPack,
    telemetry?: SSEContextPackReadyData["telemetry"]
  ) => void;
  onOpenHandoffModal: () => void;
}

interface StageState {
  stage: number;
  name: string;
  status: "idle" | "pending" | "running" | "completed" | "failed";
  message: string;
}

const DEFAULT_STAGES: StageState[] = [
  {
    stage: 1,
    name: "Hybrid RRF Retrieval",
    status: "idle",
    message: "Pending execution across BM25, Trigram & EmbeddingGemma-2",
  },
  {
    stage: 2,
    name: "Gemma 4 26B-A4B Synthesis",
    status: "idle",
    message: "Pending local 4-bit Apple Silicon GPU inference",
  },
  {
    stage: 3,
    name: "AST Closure & Friction Gate",
    status: "idle",
    message: "Pending 1-hop caller closure and cost floor audit",
  },
  {
    stage: 4,
    name: "Pre-Flight Deterministic Validation",
    status: "idle",
    message: "Pending AST compiler check before paid frontier handoff",
  },
];

const SAMPLE_QUERIES: Record<string, string> = {
  trading_code_scout:
    "Implement slippage model with volume-weighted square-root impact and regulatory minimum cost floor in Rust/Python bindings.",
  quant_research_scout:
    "Extract intraday momentum volatility breakout alpha signal equations and verify market transaction cost floor for US Equities and NSE.",
  federated_all:
    "Cross-domain federation: extract quant paper volatility signal and locate matching execution order routing methods in trading engine.",
};

export const ScoutConsole: React.FC<ScoutConsoleProps> = ({
  selectedDomainId,
  principal,
  onContextPackReady,
  onOpenHandoffModal,
}) => {
  const [query, setQuery] = useState(SAMPLE_QUERIES[selectedDomainId] ?? "");
  const [executionMode, setExecutionMode] = useState<ExecutionMode>("chat_qa");
  const [isStreaming, setIsStreaming] = useState(false);
  const [stages, setStages] = useState<StageState[]>(DEFAULT_STAGES);
  const [bounceBack, setBounceBack] = useState<SSEBounceBackData | null>(null);
  const [guardrailAlert, setGuardrailAlert] = useState<string | null>(null);
  const [hasContextPack, setHasContextPack] = useState(false);

  const abortStreamRef = useRef<(() => void) | null>(null);

  // Synchronize sample query when domain changes if user hasn't heavily modified it
  useEffect(() => {
    if (SAMPLE_QUERIES[selectedDomainId]) {
      setQuery(SAMPLE_QUERIES[selectedDomainId]);
    }
  }, [selectedDomainId]);

  // Clean up stream on unmount
  useEffect(() => {
    return () => {
      if (abortStreamRef.current) {
        abortStreamRef.current();
      }
    };
  }, []);

  const resetState = (): void => {
    setStages(
      DEFAULT_STAGES.map((s) => ({
        ...s,
        status: "idle",
      }))
    );
    setBounceBack(null);
    setGuardrailAlert(null);
    setHasContextPack(false);
  };

  const handleStartWorkflow = async (): Promise<void> => {
    if (!query.trim() || isStreaming) return;
    resetState();
    setIsStreaming(true);

    if (executionMode === "gemini_handoff") {
      // Synchronous REST mode
      try {
        setStages((prev) =>
          prev.map((s) => ({
            ...s,
            status: "running",
            message: "Executing synchronous ScoutWorkflow...",
          }))
        );

        const res = await executeScoutQuery(
          {
            domain_id: selectedDomainId,
            query: query.trim(),
            execution_mode: "gemini_handoff",
          },
          principal
        );

        setStages((prev) =>
          prev.map((s) => ({
            ...s,
            status: "completed",
            message: "Completed synchronously.",
          }))
        );

        setHasContextPack(true);
        onContextPackReady(res.context_pack, res.telemetry);
      } catch (err: unknown) {
        setGuardrailAlert(err instanceof Error ? err.message : String(err));
      } finally {
        setIsStreaming(false);
      }
      return;
    }

    // SSE Streaming mode
    const abortFn = streamScoutWorkflow({
      request: {
        domain_id: selectedDomainId,
        query: query.trim(),
        execution_mode: "chat_qa",
      },
      principal,
      onEvent: (event: SSEStageEvent) => {
        if (event.event === "stage_update") {
          const update = event.data as SSEStageUpdateData;
          setStages((prev) =>
            prev.map((s) =>
              s.stage === update.stage
                ? {
                    ...s,
                    status: update.status,
                    message: update.message,
                  }
                : s
            )
          );
        } else if (event.event === "bounce_back") {
          const bb = event.data as SSEBounceBackData;
          setBounceBack(bb);
        } else if (event.event === "context_pack_ready") {
          const ready = event.data as SSEContextPackReadyData;
          const pack = ready.pack ?? ready.context_pack;
          const rawTelemetry = ready.token_savings ?? ready.telemetry;
          const telemetry = rawTelemetry
            ? {
                local_tokens_consumed:
                  "local_tokens_consumed" in rawTelemetry
                    ? (rawTelemetry.local_tokens_consumed as number)
                    : rawTelemetry.local_tokens_explored,
                frontier_tokens_sent:
                  "frontier_tokens_sent" in rawTelemetry
                    ? (rawTelemetry.frontier_tokens_sent as number)
                    : rawTelemetry.pack_tokens_sent,
                frontier_tokens_saved:
                  "frontier_tokens_saved" in rawTelemetry
                    ? (rawTelemetry.frontier_tokens_saved as number)
                    : rawTelemetry.tokens_saved,
                estimated_usd_saved: rawTelemetry.estimated_usd_saved,
                savings_ratio_pct:
                  "savings_ratio_pct" in rawTelemetry
                    ? (rawTelemetry.savings_ratio_pct as number)
                    : rawTelemetry.frontier_savings_ratio_pct,
              }
            : undefined;

          if (pack) {
            setHasContextPack(true);
            onContextPackReady(pack, telemetry);
          }
        } else if (event.event === "error") {
          const err = event.data as SSEErrorData;
          setGuardrailAlert(err.message);
          setStages((prev) =>
            prev.map((s) =>
              s.status === "running" ? { ...s, status: "failed" } : s
            )
          );
        }
      },
      onError: (err: Error) => {
        setGuardrailAlert(err.message);
        setIsStreaming(false);
      },
      onComplete: () => {
        setIsStreaming(false);
      },
    });

    abortStreamRef.current = abortFn;
  };

  const handleCancelStream = (): void => {
    if (abortStreamRef.current) {
      abortStreamRef.current();
      abortStreamRef.current = null;
    }
    setIsStreaming(false);
  };

  return (
    <div className="card scout-console-card" data-testid="scout-console">
      <div className="card-header">
        <div className="card-title">
          <Terminal className="icon icon-sky" size={18} />
          <span>Local Gemma 4 Scout Console (ARC-01)</span>
        </div>

        <div className="flex-row-gap">
          {guardrailAlert ? (
            <span
              className="badge badge-rose"
              data-testid="shield-gemma-alert-badge"
            >
              <ShieldAlert size={14} className="mr-1" /> ShieldGemma 2: Blocked
            </span>
          ) : (
            <span
              className="badge badge-emerald"
              data-testid="shield-gemma-safe-badge"
            >
              <ShieldCheck size={14} className="mr-1" /> ShieldGemma 2: Active & Safe
            </span>
          )}

          <div className="mode-toggle" role="group" aria-label="Execution mode">
            <button
              type="button"
              data-testid="mode-stream-btn"
              className={`mode-btn ${executionMode === "chat_qa" ? "mode-btn-active" : ""}`}
              onClick={() => setExecutionMode("chat_qa")}
            >
              <Radio size={12} className="mr-1" /> Stream SSE
            </button>
            <button
              type="button"
              data-testid="mode-rest-btn"
              className={`mode-btn ${executionMode === "gemini_handoff" ? "mode-btn-active" : ""}`}
              onClick={() => setExecutionMode("gemini_handoff")}
            >
              REST Sync
            </button>
          </div>
        </div>
      </div>

      {guardrailAlert && (
        <div
          className="alert-banner alert-banner-rose"
          data-testid="guardrail-alert-banner"
          role="alert"
        >
          <AlertTriangle size={18} className="mr-2 flex-shrink-0" />
          <div className="alert-content">
            <div className="font-semibold">Guardrail Intervention</div>
            <div className="text-sm">{guardrailAlert}</div>
          </div>
        </div>
      )}

      {bounceBack && (
        <div
          className="alert-banner alert-banner-amber"
          data-testid="bounce-back-badge"
          role="status"
        >
          <Sparkles size={18} className="mr-2 flex-shrink-0 text-amber-400" />
          <div className="alert-content">
            <div className="font-semibold flex-row-gap">
              <span>$0 Pre-Flight Bounce-Back Remediation (Attempt #{bounceBack.bounce_count})</span>
              <span className="badge badge-emerald text-xs">Cost: $0.00 USD</span>
            </div>
            <div className="text-sm mt-1">{bounceBack.feedback}</div>
            <div className="text-xs text-muted mt-1">
              Action: {bounceBack.remediation_action}
            </div>
          </div>
        </div>
      )}

      <div className="console-input-area">
        <textarea
          data-testid="scout-query-input"
          className="textarea-console"
          rows={3}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter natural language implementation goal or quant hypothesis..."
          disabled={isStreaming}
        />

        <div className="console-action-bar">
          <div className="sample-quick-picks">
            <span className="text-xs text-muted mr-1">Quick Sample:</span>
            <button
              type="button"
              className="btn-link text-xs"
              onClick={() =>
                setQuery(
                  "Implement slippage model with volume-weighted square-root impact and regulatory minimum cost floor in Rust/Python bindings."
                )
              }
            >
              Code Scout
            </button>
            <span className="text-muted">|</span>
            <button
              type="button"
              className="btn-link text-xs"
              onClick={() =>
                setQuery(
                  "Extract intraday momentum volatility breakout alpha signal equations and verify market transaction cost floor."
                )
              }
            >
              Quant Paper
            </button>
            <span className="text-muted">|</span>
            <button
              type="button"
              className="btn-link text-xs"
              onClick={() =>
                setQuery(
                  "Explain why OrderRouter references calculate_slippage_bps without bounce-back errors."
                )
              }
            >
              Bounce Test
            </button>
          </div>

          <div className="action-buttons-group">
            {isStreaming ? (
              <button
                type="button"
                data-testid="cancel-stream-btn"
                className="btn btn-secondary"
                onClick={handleCancelStream}
              >
                Cancel Stream
              </button>
            ) : (
              <button
                type="button"
                data-testid="run-scout-btn"
                className="btn btn-primary"
                onClick={handleStartWorkflow}
                disabled={!query.trim()}
              >
                <Play size={14} className="mr-1" />
                Run Scout Workflow
              </button>
            )}

            {hasContextPack && (
              <button
                type="button"
                data-testid="open-diff-modal-btn"
                className="btn btn-emerald"
                onClick={onOpenHandoffModal}
              >
                <Sparkles size={14} className="mr-1" />
                1-Click Gemini Handoff
                <ArrowRight size={14} className="ml-1" />
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="stepper-container" data-testid="scout-stepper">
        <div className="stepper-title">Deterministic 4-Stage Workflow Progression:</div>
        <div className="stepper-grid">
          {stages.map((stage) => {
            const isRunning = stage.status === "running";
            const isCompleted = stage.status === "completed";
            const isFailed = stage.status === "failed";

            return (
              <div
                key={stage.stage}
                data-testid={`stage-card-${stage.stage}`}
                className={`stage-card ${
                  isRunning
                    ? "stage-card-running"
                    : isCompleted
                      ? "stage-card-completed"
                      : isFailed
                        ? "stage-card-failed"
                        : ""
                }`}
              >
                <div className="stage-card-top">
                  <div className="stage-num-badge">Stage {stage.stage}</div>
                  <div className="stage-status-icon">
                    {isRunning && (
                      <RotateCw
                        size={14}
                        className="animate-spin icon-sky"
                        data-testid={`spinner-stage-${stage.stage}`}
                      />
                    )}
                    {isCompleted && (
                      <CheckCircle2
                        size={14}
                        className="icon-emerald"
                        data-testid={`check-stage-${stage.stage}`}
                      />
                    )}
                    {isFailed && (
                      <AlertTriangle
                        size={14}
                        className="icon-rose"
                        data-testid={`fail-stage-${stage.stage}`}
                      />
                    )}
                    {(stage.status === "idle" || stage.status === "pending") && (
                      <Clock size={14} className="icon-muted" />
                    )}
                  </div>
                </div>

                <div className="stage-name">{stage.name}</div>
                <div className="stage-msg">{stage.message}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
