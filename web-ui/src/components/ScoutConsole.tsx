import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Cpu,
  Database,
  Loader2,
  Play,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import {
  ApiClientError,
  executeScoutQuery,
  FEDERATED_ALL_DOMAIN_ID,
} from "../api/client";
import { streamScoutWorkflow } from "../api/sse";
import {
  ConfidenceBand,
  ContextPack,
  DomainInfo,
  InsufficientContextDiagnostic,
  PipelineStageSpec,
  PrincipalContext,
  QuantStrategyBrief,
  ScoutTokenSavingsBreakdown,
  SourceFilterType,
  SSEBounceBackData,
  SSEStageEvent,
  StageConfidence,
  StageExecutionTrace,
} from "../types/contracts";

export interface ScoutConsoleProps {
  selectedDomain: string;
  principal: PrincipalContext;
  domains?: DomainInfo[];
  pipelineStages?: PipelineStageSpec[];
  sourceFilter?: SourceFilterType;
  scopeResetVersion?: number;
  onSelectDomain?: (domainId: string) => void;
  onContextPackReady: (
    pack: ContextPack | null,
    tokenSavings: ScoutTokenSavingsBreakdown | null,
    quantSpec: QuantStrategyBrief | null,
    diagnostic?: InsufficientContextDiagnostic | null,
    stageTraces?: StageExecutionTrace[]
  ) => void;
  onClearContextPack?: () => void;
  onQueryComplete?: () => void;
}

type StageStatus = "pending" | "running" | "completed" | "failed";

interface ExecutionStage {
  id: number;
  number: string;
  title: string;
  summary: string;
  whatHappens: string;
  gateCondition: string;
  status: StageStatus;
  detail: string;
  elapsedMs?: number | null;
  confidence?: StageConfidence | null;
  trace?: StageExecutionTrace | null;
}

interface DynamicSamplePill {
  id: string;
  domainId: string;
  domainTag: string;
  query: string;
}

const SOURCE_FILTER_LABELS: Record<SourceFilterType, string> = {
  all: "All Federated Sources",
  code: "Code Repos",
  wiki: "Confluence & Docs",
  jira: "Jira & Issues",
};

function buildInitialStages(
  specs?: PipelineStageSpec[]
): ExecutionStage[] {
  if (specs && specs.length > 0) {
    return specs.map((spec) => ({
      id: spec.stage_id,
      number: spec.number,
      title: spec.title,
      summary: spec.summary,
      whatHappens: spec.what_happens,
      gateCondition: spec.gate_condition,
      status: "pending",
      detail: spec.summary,
      elapsedMs: null,
      confidence: null,
      trace: null,
    }));
  }
  return [1, 2, 3, 4].map((num) => ({
    id: num,
    number: `0${num}`,
    title: `Stage ${num}`,
    summary: "Awaiting workspace pipeline specification...",
    whatHappens: "",
    gateCondition: "",
    status: "pending",
    detail: "Waiting for query execution...",
    elapsedMs: null,
    confidence: null,
    trace: null,
  }));
}

function formatConfidenceBand(band: ConfidenceBand): string {
  if (band === "high") return "High";
  if (band === "medium") return "Medium";
  return "Low";
}

function formatSignalKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function formatSignalValue(val: number | string | boolean): string {
  if (typeof val === "boolean") return val ? "Yes" : "No";
  if (typeof val === "number") {
    return Number.isInteger(val) ? String(val) : val.toFixed(3);
  }
  return val;
}

function truncatePillText(text: string, maxLen = 58): string {
  const clean = text.trim();
  if (clean.length <= maxLen) return clean;
  return `${clean.slice(0, maxLen - 1)}…`;
}

export const ScoutConsole: React.FC<ScoutConsoleProps> = ({
  selectedDomain,
  principal,
  domains = [],
  pipelineStages,
  sourceFilter = "all",
  scopeResetVersion = 0,
  onSelectDomain,
  onContextPackReady,
  onClearContextPack,
  onQueryComplete,
}) => {
  const samplePills = useMemo<DynamicSamplePill[]>(() => {
    if (domains.length === 0) return [];
    if (selectedDomain === FEDERATED_ALL_DOMAIN_ID) {
      const interleaved: DynamicSamplePill[] = [];
      const maxPerDomain = 2;
      for (let qIdx = 0; qIdx < maxPerDomain; qIdx += 1) {
        for (const d of domains) {
          const q = d.sample_queries[qIdx];
          if (q && q.trim()) {
            interleaved.push({
              id: `${d.domain_id}-${qIdx}`,
              domainId: d.domain_id,
              domainTag: d.display_name || d.domain_id,
              query: q,
            });
          }
        }
      }
      return interleaved.slice(0, 6);
    }

    const activeDomain = domains.find((d) => d.domain_id === selectedDomain);
    if (!activeDomain) return [];
    return activeDomain.sample_queries
      .filter((q) => q.trim().length > 0)
      .map((q, idx) => ({
        id: `${activeDomain.domain_id}-${idx}`,
        domainId: activeDomain.domain_id,
        domainTag: activeDomain.display_name || activeDomain.domain_id,
        query: q,
      }));
  }, [domains, selectedDomain]);

  const allBackendSampleQueries = useMemo(() => {
    const set = new Set<string>();
    for (const d of domains) {
      for (const q of d.sample_queries) {
        if (q.trim()) set.add(q.trim());
      }
    }
    return set;
  }, [domains]);

  const [query, setQuery] = useState<string>("");
  const [mode, setMode] = useState<"stream" | "rest">("stream");
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [timelineExpanded, setTimelineExpanded] = useState<boolean>(true);
  const [expandedStageId, setExpandedStageId] = useState<number | null>(null);
  const [stages, setStages] = useState<ExecutionStage[]>(() =>
    buildInitialStages(pipelineStages)
  );
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [bounceBack, setBounceBack] = useState<SSEBounceBackData | null>(null);
  const [guardrailAlert, setGuardrailAlert] = useState<{
    category: string;
    rationale: string;
  } | null>(null);
  const [shieldSafe, setShieldSafe] = useState<boolean>(false);
  const [abstainedDiagnostic, setAbstainedDiagnostic] =
    useState<InsufficientContextDiagnostic | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const cancelStreamRef = useRef<(() => void) | null>(null);
  const runStartMsRef = useRef<number | null>(null);

  // Sync stage definitions from backend workspaceConfig when idle
  useEffect(() => {
    if (!pipelineStages || pipelineStages.length === 0) return;
    setStages((prev) =>
      pipelineStages.map((spec) => {
        const existing = prev.find((s) => s.id === spec.stage_id);
        return {
          id: spec.stage_id,
          number: spec.number,
          title: spec.title,
          summary: spec.summary,
          whatHappens: spec.what_happens,
          gateCondition: spec.gate_condition,
          status: existing?.status ?? "pending",
          detail:
            existing && existing.status !== "pending"
              ? existing.detail
              : spec.summary,
          elapsedMs: existing?.elapsedMs ?? null,
          confidence: existing?.confidence ?? null,
          trace: existing?.trace ?? null,
        };
      })
    );
  }, [pipelineStages]);

  // Populate default sample query from backend domain config if input is empty or unmodified sample
  useEffect(() => {
    if (samplePills.length === 0) return;
    const firstQuery = samplePills[0].query;
    setQuery((prevQuery) => {
      const trimmed = prevQuery.trim();
      if (!trimmed || allBackendSampleQueries.has(trimmed)) {
        return firstQuery;
      }
      return prevQuery;
    });
  }, [selectedDomain, samplePills, allBackendSampleQueries]);

  useEffect(() => {
    if (scopeResetVersion > 0) {
      if (cancelStreamRef.current) {
        cancelStreamRef.current();
        cancelStreamRef.current = null;
      }
      setIsRunning(false);
      setStages(buildInitialStages(pipelineStages));
      setElapsedMs(null);
      setBounceBack(null);
      setGuardrailAlert(null);
      setShieldSafe(false);
      setAbstainedDiagnostic(null);
      setApiError(null);
      setExpandedStageId(null);
    }
  }, [scopeResetVersion, pipelineStages]);

  useEffect(() => {
    return () => {
      if (cancelStreamRef.current) {
        cancelStreamRef.current();
        cancelStreamRef.current = null;
      }
    };
  }, []);

  const getElapsedMs = (): number | null => {
    if (runStartMsRef.current === null) return null;
    return Math.max(0, Math.round(performance.now() - runStartMsRef.current));
  };

  const resetRunState = (activeStages: ExecutionStage[]): void => {
    setStages(activeStages);
    setElapsedMs(null);
    setBounceBack(null);
    setGuardrailAlert(null);
    setShieldSafe(false);
    setAbstainedDiagnostic(null);
    setApiError(null);
    setTimelineExpanded(true);
    setExpandedStageId(null);
  };

  const applyTracesAndConfidences = (
    traces: StageExecutionTrace[] | undefined,
    confidences: StageConfidence[] | undefined
  ): void => {
    const traceMap = new Map<number, StageExecutionTrace>();
    for (const t of traces ?? []) {
      traceMap.set(t.stage, t);
    }
    const confMap = new Map<number, StageConfidence>();
    for (const c of confidences ?? []) {
      confMap.set(c.stage, c);
    }
    if (traceMap.size === 0 && confMap.size === 0) return;
    setStages((prev) =>
      prev.map((st) => {
        const t = traceMap.get(st.id);
        const c = confMap.get(st.id);
        return {
          ...st,
          trace: t ?? st.trace ?? null,
          confidence: c ?? st.confidence ?? null,
          detail: t?.summary || st.detail,
          elapsedMs: t?.elapsed_ms ?? st.elapsedMs ?? null,
        };
      })
    );
  };

  const handleCancel = (): void => {
    if (cancelStreamRef.current) {
      cancelStreamRef.current();
      cancelStreamRef.current = null;
    }
    setIsRunning(false);
    const totalMs = getElapsedMs();
    if (totalMs !== null) {
      setElapsedMs(totalMs);
    }
    setStages((prev) =>
      prev.map((st) =>
        st.status === "running"
          ? {
              ...st,
              status: "failed",
              detail: "Cancelled by user.",
              elapsedMs: totalMs,
            }
          : st
      )
    );
  };

  const handleSSEEvent = (evt: SSEStageEvent): void => {
    const nowElapsed = getElapsedMs();
    if (nowElapsed !== null) {
      setElapsedMs(nowElapsed);
    }

    if (evt.event === "stage_update") {
      const {
        stage,
        stage_name,
        status,
        details,
        message,
        confidence,
        stage_trace,
        elapsed_ms,
      } = evt.data;
      const mappedStatus: StageStatus =
        status === "completed"
          ? "completed"
          : status === "failed"
            ? "failed"
            : "running";
      const detailText =
        stage_trace?.summary || details || message || "Processing...";
      const stageElapsed =
        typeof elapsed_ms === "number"
          ? elapsed_ms
          : stage_trace?.elapsed_ms ?? nowElapsed;

      setStages((prev) =>
        prev.map((st) => {
          if (st.id === stage) {
            return {
              ...st,
              title:
                st.title.startsWith("Stage ") && stage_name
                  ? stage_name
                  : st.title,
              status: mappedStatus,
              detail: detailText,
              elapsedMs: stageElapsed,
              confidence: confidence ?? st.confidence ?? null,
              trace: stage_trace ?? st.trace ?? null,
            };
          }
          if (
            st.id < stage &&
            (st.status === "pending" || st.status === "running")
          ) {
            return {
              ...st,
              status: "completed",
              detail: st.status === "pending" ? "Completed." : st.detail,
            };
          }
          if (
            mappedStatus === "completed" &&
            st.id === stage + 1 &&
            st.status === "pending"
          ) {
            return { ...st, status: "running", detail: "In progress..." };
          }
          return st;
        })
      );
      return;
    }

    if (evt.event === "bounce_back") {
      setBounceBack(evt.data);
      setStages((prev) =>
        prev.map((st) =>
          st.id === 4
            ? {
                ...st,
                status: "running",
                detail: `Self-correcting ($0): ${evt.data.feedback}`,
                elapsedMs: nowElapsed,
              }
            : st
        )
      );
      return;
    }

    if (evt.event === "context_pack_ready") {
      const pack = evt.data.pack;
      const finalTraces = evt.data.stage_traces ?? pack?.stage_traces ?? [];
      const traceMap = new Map<number, StageExecutionTrace>();
      for (const t of finalTraces) {
        traceMap.set(t.stage, t);
      }

      const confMap = new Map<number, StageConfidence>();
      for (const c of pack?.stage_confidences ?? []) {
        confMap.set(c.stage, c);
      }

      const finalDetail = pack
        ? `Verified ${pack.code_slices.length} code slice(s) & ${pack.supporting_doc_citations.length} doc citation(s) (${pack.pack_token_count} pack tokens).`
        : "Completed in grounded run.";

      setStages((prev) =>
        prev.map((st) => {
          const t = traceMap.get(st.id);
          return {
            ...st,
            status:
              t?.status === "failed" || st.status === "failed"
                ? "failed"
                : "completed",
            detail:
              t?.summary ||
              (st.id === 4 && st.status === "pending"
                ? finalDetail
                : st.status === "pending"
                  ? "Completed in grounded run."
                  : st.detail),
            elapsedMs: t?.elapsed_ms ?? st.elapsedMs ?? nowElapsed,
            confidence: confMap.get(st.id) ?? st.confidence ?? null,
            trace: t ?? st.trace ?? null,
          };
        })
      );
      setShieldSafe(true);
      setIsRunning(false);
      onContextPackReady(
        evt.data.pack,
        evt.data.token_savings,
        evt.data.quant_strategy_spec ??
          evt.data.pack?.quant_strategy_spec ??
          null,
        null,
        finalTraces
      );
      onQueryComplete?.();
      return;
    }

    if (evt.event === "error") {
      const { category, rationale, stage, status, diagnostic } = evt.data;
      setIsRunning(false);

      if (
        status === "blocked" ||
        stage === "guardrails" ||
        category === "prompt_injection" ||
        category === "security_policy" ||
        category === "SECURITY_GUARDRAIL_TRIGGERED"
      ) {
        setGuardrailAlert({
          category: category || "guardrail_block",
          rationale:
            rationale || "Blocked by local ShieldGemma / security gate.",
        });
        setStages((prev) =>
          prev.map((st) =>
            st.id === 1
              ? {
                  ...st,
                  status: "failed",
                  detail: rationale,
                  elapsedMs: nowElapsed,
                  confidence: {
                    stage: 1,
                    stage_name: st.title,
                    confidence_score: 0,
                    confidence_band: "low",
                    signals: { security_gate: "blocked", category },
                    summary: "Blocked at Stage 1 security policy gate.",
                  },
                }
              : st
          )
        );
        onClearContextPack?.();
        onContextPackReady(null, null, null, null);
        return;
      }

      if (diagnostic || category === "insufficient_context") {
        const diag: InsufficientContextDiagnostic = diagnostic ?? {
          trace_id: "abstain_sse",
          domain_id: selectedDomain,
          reason: rationale,
          is_abstention: true,
          missing_symbols: [],
          partial_candidate_files: [],
          missing_items: [],
          failure_check: category,
          paid_tokens_spent: 0,
        };
        setAbstainedDiagnostic(diag);

        const traceMap = new Map<number, StageExecutionTrace>();
        for (const t of diag.stage_traces ?? []) {
          traceMap.set(t.stage, t);
        }
        const confMap = new Map<number, StageConfidence>();
        for (const c of diag.stage_confidences ?? []) {
          confMap.set(c.stage, c);
        }

        setStages((prev) =>
          prev.map((st) => {
            const t = traceMap.get(st.id);
            if (st.id < 4) {
              return {
                ...st,
                status: t?.status === "failed" ? "failed" : "completed",
                detail: t?.summary || st.detail,
                elapsedMs: t?.elapsed_ms ?? st.elapsedMs,
                confidence: confMap.get(st.id) ?? st.confidence ?? null,
                trace: t ?? st.trace ?? null,
              };
            }
            return {
              ...st,
              status: "failed",
              detail:
                t?.summary ||
                `Abstained ($0 frontier spend): ${diag.reason}`,
              elapsedMs: t?.elapsed_ms ?? nowElapsed,
              trace: t ?? st.trace ?? null,
              confidence: confMap.get(st.id) ??
                st.confidence ?? {
                  stage: 4,
                  stage_name: st.title,
                  confidence_score: 0,
                  confidence_band: "low",
                  signals: { paid_tokens_spent: diag.paid_tokens_spent },
                  summary: "Abstained at Pre-Flight Gate ($0 frontier spend).",
                },
            };
          })
        );
        onClearContextPack?.();
        onContextPackReady(null, null, null, diag, diag.stage_traces);
        onQueryComplete?.();
        return;
      }

      setApiError(rationale || "Execution failed.");
      setStages((prev) =>
        prev.map((st) =>
          st.status === "running" || st.id === 4
            ? {
                ...st,
                status: "failed",
                detail: rationale,
                elapsedMs: nowElapsed,
              }
            : st
        )
      );
    }
  };

  const handleRun = async (): Promise<void> => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery || isRunning) return;

    if (cancelStreamRef.current) {
      cancelStreamRef.current();
      cancelStreamRef.current = null;
    }

    runStartMsRef.current = performance.now();
    setIsRunning(true);
    const freshStages = buildInitialStages(pipelineStages);
    resetRunState(
      freshStages.map((st, idx) =>
        idx === 0
          ? {
              ...st,
              status: "running",
              detail:
                "Minting signed Bearer session and running ShieldGemma check...",
            }
          : st
      )
    );

    const requestPayload = {
      domain_id: selectedDomain,
      query: trimmedQuery,
      source_filter: sourceFilter,
    };

    if (mode === "stream") {
      cancelStreamRef.current = streamScoutWorkflow({
        request: requestPayload,
        principal,
        onEvent: handleSSEEvent,
        onError: (err) => {
          cancelStreamRef.current = null;
          setIsRunning(false);
          const nowElapsed = getElapsedMs();
          if (nowElapsed !== null) setElapsedMs(nowElapsed);
          setApiError(err.message);
          setStages((prev) =>
            prev.map((st) =>
              st.status === "running" || st.id === 1
                ? {
                    ...st,
                    status: "failed",
                    detail: err.message,
                    elapsedMs: nowElapsed,
                  }
                : st
            )
          );
        },
        onComplete: () => {
          cancelStreamRef.current = null;
          setIsRunning(false);
          const nowElapsed = getElapsedMs();
          if (nowElapsed !== null) setElapsedMs(nowElapsed);
        },
      });
      return;
    }

    try {
      setStages((prev) =>
        prev.map((st) =>
          st.id <= 2
            ? {
                ...st,
                status: st.id === 2 ? "running" : "completed",
                detail:
                  st.id === 2
                    ? "Querying PostgreSQL 17 RLS + local Gemma 4 synchronously..."
                    : "Signed Bearer & ShieldGemma 2 passed.",
              }
            : st
        )
      );

      const res = await executeScoutQuery(requestPayload, principal);
      setIsRunning(false);
      const nowElapsed = getElapsedMs();
      if (nowElapsed !== null) setElapsedMs(nowElapsed);

      const finalTraces =
        res.stage_traces ??
        res.context_pack?.stage_traces ??
        res.diagnostic?.stage_traces ??
        [];

      if (res.passed && res.context_pack) {
        setStages((prev) =>
          prev.map((st) => ({
            ...st,
            status: "completed",
            detail:
              st.id === 4
                ? `Verified ${res.context_pack?.code_slices.length ?? 0} slice(s) & ${res.citations.length} citation(s) (${res.context_pack?.pack_token_count ?? 0} tokens).`
                : st.id === 3
                  ? `Synthesized locally (${res.token_savings.local_tokens_explored.toLocaleString()} tokens filtered).`
                  : "Completed.",
            elapsedMs: nowElapsed,
          }))
        );
        applyTracesAndConfidences(
          finalTraces,
          res.context_pack.stage_confidences
        );
        setShieldSafe(true);
        onContextPackReady(
          res.context_pack,
          res.token_savings,
          res.quant_strategy_spec,
          null,
          finalTraces
        );
        onQueryComplete?.();
        return;
      }

      if (res.diagnostic) {
        setAbstainedDiagnostic(res.diagnostic);
        setStages((prev) =>
          prev.map((st) =>
            st.id < 4
              ? { ...st, status: "completed", detail: "Completed." }
              : {
                  ...st,
                  status: "failed",
                  detail: `Abstained ($0 frontier spend): ${res.diagnostic?.reason ?? "Insufficient grounded evidence."}`,
                  elapsedMs: nowElapsed,
                }
          )
        );
        applyTracesAndConfidences(
          finalTraces,
          res.diagnostic.stage_confidences
        );
        onClearContextPack?.();
        onContextPackReady(
          null,
          res.token_savings,
          null,
          res.diagnostic,
          finalTraces
        );
        onQueryComplete?.();
        return;
      }

      setApiError("Query completed without context pack or diagnostic.");
    } catch (err: unknown) {
      setIsRunning(false);
      const nowElapsed = getElapsedMs();
      if (nowElapsed !== null) setElapsedMs(nowElapsed);
      if (
        err instanceof ApiClientError &&
        err.status === 403 &&
        /inject|guardrail|shield|block|policy/i.test(err.detail)
      ) {
        setGuardrailAlert({
          category: "prompt_injection",
          rationale: err.detail,
        });
        setStages((prev) =>
          prev.map((st) =>
            st.id === 1
              ? {
                  ...st,
                  status: "failed",
                  detail: err.detail,
                  elapsedMs: nowElapsed,
                }
              : st
          )
        );
        onClearContextPack?.();
        onContextPackReady(null, null, null, null);
        return;
      }

      const message =
        err instanceof Error ? err.message : "REST query request failed.";
      setApiError(message);
      setStages((prev) =>
        prev.map((st) =>
          st.status === "running" || st.id === 1
            ? {
                ...st,
                status: "failed",
                detail: message,
                elapsedMs: nowElapsed,
              }
            : st
        )
      );
    }
  };

  const handleSamplePillClick = (pill: DynamicSamplePill): void => {
    setQuery(pill.query);
    if (
      selectedDomain !== FEDERATED_ALL_DOMAIN_ID &&
      onSelectDomain &&
      pill.domainId !== selectedDomain
    ) {
      onSelectDomain(pill.domainId);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      void handleRun();
    }
  };

  const missingItemLabels = abstainedDiagnostic
    ? abstainedDiagnostic.missing_items.map((item) =>
        typeof item === "string"
          ? item
          : `${item.identifier}${item.reason ? ` (${item.reason})` : ""}`
      )
    : [];

  return (
    <section className="console-section" data-testid="scout-console">
      <div className="console-top-row">
        <div className="console-title-group">
          <h2 className="section-heading">Ask Enterprise Context Scout</h2>
          <span className="domain-context-pill">
            Target: <code>{selectedDomain}</code>
          </span>
          {sourceFilter !== "all" && (
            <span
              className="status-tag status-tag-info"
              data-testid="console-source-filter-pill"
            >
              {SOURCE_FILTER_LABELS[sourceFilter]}
            </span>
          )}
        </div>

        <div
          className="mode-toggle-group"
          role="group"
          aria-label="Execution mode"
        >
          <button
            type="button"
            className={`mode-btn ${mode === "stream" ? "mode-btn-active" : ""}`}
            onClick={() => setMode("stream")}
            disabled={isRunning}
            data-testid="mode-stream-btn"
          >
            Live Stream (SSE)
          </button>
          <button
            type="button"
            className={`mode-btn ${mode === "rest" ? "mode-btn-active" : ""}`}
            onClick={() => setMode("rest")}
            disabled={isRunning}
            data-testid="mode-rest-btn"
          >
            Fast REST Sync
          </button>
        </div>
      </div>

      {samplePills.length > 0 && (
        <div
          className="quick-picks-row"
          aria-label="Domain sample queries"
          data-testid="dynamic-sample-queries"
        >
          {samplePills.map((pill) => (
            <button
              key={pill.id}
              type="button"
              className="quick-pick-btn"
              onClick={() => handleSamplePillClick(pill)}
              disabled={isRunning}
              title={pill.query}
              data-testid={`quick-pick-${pill.id}`}
            >
              <span className="quick-pick-tag">{pill.domainTag}</span>
              <span>{truncatePillText(pill.query)}</span>
            </button>
          ))}
        </div>
      )}

      <div className="composer-box">
        <textarea
          id="scout-query-input"
          name="scout_query"
          className="query-textarea"
          rows={3}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about codebase architecture, AST callers, data pipelines, or research papers... (⌘+Enter to run)"
          data-testid="scout-query-input"
        />

        <div className="composer-actions">
          <span className="composer-hint">
            Grounded locally in PG17 + Gemma 4 before any frontier token spend (⌘+Enter)
          </span>
          <div className="action-buttons">
            {isRunning && (
              <button
                type="button"
                className="btn-secondary"
                onClick={handleCancel}
                data-testid="cancel-stream-btn"
              >
                <XCircle size={15} /> Stop
              </button>
            )}
            <button
              type="button"
              className="btn-primary"
              onClick={() => void handleRun()}
              disabled={isRunning || !query.trim()}
              data-testid="run-scout-btn"
            >
              {isRunning ? (
                <>
                  <Loader2 size={15} className="spin" /> Running Local Scout...
                </>
              ) : (
                <>
                  <Play size={15} /> Run Local Scout ($0 Frontier)
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {shieldSafe && !guardrailAlert && (
        <div
          className="status-banner status-banner-safe"
          data-testid="shield-gemma-safe-badge"
        >
          <ShieldCheck size={16} />
          <span>
            ShieldGemma 2 &amp; RBAC Security Gate Passed — Zero Frontier Leak
          </span>
        </div>
      )}

      {guardrailAlert && (
        <div
          className="alert-banner alert-banner-danger"
          data-testid="guardrail-alert-banner"
          role="alert"
        >
          <div className="alert-banner-header">
            <ShieldAlert size={18} />
            <strong data-testid="shield-gemma-alert-badge">
              Blocked by ShieldGemma 2 Security Gate ({guardrailAlert.category})
            </strong>
          </div>
          <p className="alert-banner-body">{guardrailAlert.rationale}</p>
        </div>
      )}

      {apiError && (
        <div
          className="alert-banner alert-banner-danger"
          data-testid="api-error-banner"
          role="alert"
        >
          <div className="alert-banner-header">
            <AlertTriangle size={18} />
            <strong>Request Failed (Fail-Closed)</strong>
          </div>
          <p className="alert-banner-body">{apiError}</p>
        </div>
      )}

      {bounceBack && (
        <div
          className="alert-banner alert-banner-warning"
          data-testid="bounce-back-badge"
        >
          <div className="alert-banner-header">
            <RefreshCw size={16} />
            <strong>
              $0 Self-Correction Retry
              {typeof bounceBack.bounce_count === "number"
                ? ` #${bounceBack.bounce_count}`
                : ""}
            </strong>
            <span className="status-tag status-tag-warn tabular-num">
              ${bounceBack.cost_usd.toFixed(2)} Frontier Cost
            </span>
          </div>
          <p className="alert-banner-body">{bounceBack.feedback}</p>
          {bounceBack.remediation_action && (
            <p className="alert-banner-meta">
              Action: <code>{bounceBack.remediation_action}</code>
            </p>
          )}
        </div>
      )}

      {abstainedDiagnostic && (
        <div
          className="alert-banner alert-banner-info"
          data-testid="insufficient-context-banner"
        >
          <div className="alert-banner-header">
            <ShieldCheck size={17} className="icon-emerald" />
            <strong>
              $0.00 Paid-Model Protection — Honest Abstention (
              {abstainedDiagnostic.paid_tokens_spent} Paid Tokens Spent)
            </strong>
          </div>
          <p className="alert-banner-body">{abstainedDiagnostic.reason}</p>
          <div className="diagnostic-details">
            {abstainedDiagnostic.failure_check && (
              <span>
                Gate check: <code>{abstainedDiagnostic.failure_check}</code>
              </span>
            )}
            {abstainedDiagnostic.missing_symbols.length > 0 && (
              <div>
                Missing symbols:{" "}
                <code>{abstainedDiagnostic.missing_symbols.join(", ")}</code>
              </div>
            )}
            {missingItemLabels.length > 0 && (
              <div>
                Gaps: <code>{missingItemLabels.join("; ")}</code>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="pipeline-section">
        <button
          type="button"
          className="timeline-header-btn"
          onClick={() => setTimelineExpanded((v) => !v)}
          data-testid="toggle-timeline-btn"
          aria-expanded={timelineExpanded}
        >
          <span className="timeline-title-wrap">
            {timelineExpanded ? (
              <ChevronDown size={15} />
            ) : (
              <ChevronRight size={15} />
            )}
            <span>4-Stage Pipeline X-Ray &amp; Execution Telemetry</span>
          </span>
          <span className="timeline-status-summary tabular-num">
            {stages.filter((s) => s.status === "completed").length}/4 completed
            {elapsedMs !== null && (
              <span data-testid="timeline-elapsed-ms"> ({elapsedMs}ms)</span>
            )}
          </span>
        </button>

        {timelineExpanded && (
          <div className="stages-track">
            {stages.map((st) => {
              const conf = st.confidence;
              const pct = conf
                ? Math.round(
                    Math.min(1, Math.max(0, conf.confidence_score)) * 100
                  )
                : null;
              const isSignalsOpen = expandedStageId === st.id;
              const trace = st.trace;

              return (
                <div
                  key={st.id}
                  className={`stage-step stage-step-${st.status}`}
                  data-testid={`stage-card-${st.id}`}
                >
                  <button
                    type="button"
                    className="stage-step-trigger"
                    onClick={() =>
                      setExpandedStageId(isSignalsOpen ? null : st.id)
                    }
                    aria-expanded={isSignalsOpen}
                    data-testid={`stage-toggle-${st.id}`}
                  >
                    <div className="stage-step-head">
                      <div className="stage-title-row">
                        <span className="stage-num tabular-num">
                          {st.number}
                        </span>
                        <span className="stage-name">{st.title}</span>
                      </div>
                      <div className="stage-indicator">
                        {typeof st.elapsedMs === "number" && (
                          <span className="stage-elapsed tabular-num">
                            {st.elapsedMs}ms
                          </span>
                        )}
                        {st.status === "running" && (
                          <Loader2
                            size={15}
                            className="spin icon-accent"
                            data-testid={`spinner-stage-${st.id}`}
                          />
                        )}
                        {st.status === "completed" && (
                          <CheckCircle2
                            size={15}
                            className="icon-emerald"
                            data-testid={`check-stage-${st.id}`}
                          />
                        )}
                        {st.status === "failed" && (
                          <XCircle
                            size={15}
                            className="icon-danger"
                            data-testid={`fail-stage-${st.id}`}
                          />
                        )}
                        {isSignalsOpen ? (
                          <ChevronDown size={13} className="icon-muted" />
                        ) : (
                          <ChevronRight size={13} className="icon-muted" />
                        )}
                      </div>
                    </div>

                    <p className="stage-detail">{st.detail}</p>

                    {conf && pct !== null && (
                      <div className="stage-conf-row">
                        <div className="stage-conf-track">
                          <div
                            className={`stage-conf-fill stage-conf-fill-${conf.confidence_band}`}
                            style={{ width: `${pct}%` }}
                            data-testid={`stage-confidence-bar-${st.id}`}
                          />
                        </div>
                        <span
                          className={`stage-conf-badge stage-conf-badge-${conf.confidence_band} tabular-num`}
                          data-testid={`stage-confidence-badge-${st.id}`}
                        >
                          {pct}% {formatConfidenceBand(conf.confidence_band)}
                        </span>
                      </div>
                    )}
                  </button>

                  {isSignalsOpen && (
                    <div
                      className="stage-signals-drawer"
                      data-testid={`stage-signals-${st.id}`}
                    >
                      {st.whatHappens && (
                        <div className="xray-spec-block">
                          <p className="xray-spec-text">
                            <strong>What happens:</strong> {st.whatHappens}
                          </p>
                          {st.gateCondition && (
                            <p className="xray-spec-gate">
                              <strong>Gate:</strong> {st.gateCondition}
                            </p>
                          )}
                        </div>
                      )}

                      {trace && (
                        <div
                          className="xray-trace-body"
                          data-testid={`stage-xray-trace-${st.id}`}
                        >
                          {/* Stage 1 X-Ray */}
                          {st.id === 1 && (
                            <div className="xray-kv-section">
                              <div className="xray-kv-grid">
                                {trace.principal_id && (
                                  <div className="signal-kv">
                                    <dt>Principal</dt>
                                    <dd className="tabular-num">
                                      {trace.principal_id}
                                    </dd>
                                  </div>
                                )}
                                {trace.guardrail_verdict && (
                                  <div className="signal-kv">
                                    <dt>Guardrail</dt>
                                    <dd>{trace.guardrail_verdict}</dd>
                                  </div>
                                )}
                                {trace.query_type && (
                                  <div className="signal-kv">
                                    <dt>Query Type</dt>
                                    <dd>{trace.query_type}</dd>
                                  </div>
                                )}
                                {trace.core_subject && (
                                  <div className="signal-kv">
                                    <dt>Core Subject</dt>
                                    <dd>{trace.core_subject}</dd>
                                  </div>
                                )}
                              </div>
                              {trace.principal_scopes &&
                                trace.principal_scopes.length > 0 && (
                                  <div className="xray-chips-line">
                                    <span className="field-label">
                                      RLS Scopes:
                                    </span>
                                    {trace.principal_scopes.map((sc) => (
                                      <code key={sc} className="xray-chip">
                                        {sc}
                                      </code>
                                    ))}
                                  </div>
                                )}
                              {trace.required_facets &&
                                trace.required_facets.length > 0 && (
                                  <div className="xray-chips-line">
                                    <span className="field-label">
                                      QueryPlan Facets:
                                    </span>
                                    {trace.required_facets.map((f) => (
                                      <span key={f} className="xray-chip">
                                        {f}
                                      </span>
                                    ))}
                                  </div>
                                )}
                              {trace.target_domains &&
                                trace.target_domains.length > 0 && (
                                  <div className="xray-chips-line">
                                    <span className="field-label">
                                      Candidate Domains:
                                    </span>
                                    {trace.target_domains.map((td) => (
                                      <code key={td} className="xray-chip">
                                        {td}
                                      </code>
                                    ))}
                                  </div>
                                )}
                            </div>
                          )}

                          {/* Stage 2 X-Ray */}
                          {st.id === 2 && (
                            <div className="xray-kv-section">
                              {(typeof trace.ast_one_hop_count === "number" ||
                                typeof trace.ast_two_hop_count ===
                                  "number") && (
                                <div className="xray-chips-line">
                                  <span className="field-label">
                                    AST Closure:
                                  </span>
                                  <span className="xray-chip tabular-num">
                                    1-hop: {trace.ast_one_hop_count ?? 0}
                                  </span>
                                  <span className="xray-chip tabular-num">
                                    2-hop: {trace.ast_two_hop_count ?? 0}
                                  </span>
                                </div>
                              )}
                              {trace.domain_probes &&
                                trace.domain_probes.length > 0 && (
                                  <div className="probe-matrix">
                                    {trace.domain_probes.map((probe) => (
                                      <div
                                        key={probe.domain_id}
                                        className={`probe-row ${
                                          probe.selected
                                            ? "probe-row-selected"
                                            : ""
                                        }`}
                                        data-testid={`domain-probe-${probe.domain_id}`}
                                      >
                                        <div className="probe-row-head">
                                          <span className="probe-domain-name">
                                            <Database size={11} />
                                            <strong>
                                              {probe.display_name ||
                                                probe.domain_id}
                                            </strong>
                                          </span>
                                          <span
                                            className={`status-tag ${
                                              probe.selected
                                                ? "status-tag-ok"
                                                : "status-tag-neutral"
                                            }`}
                                          >
                                            {probe.selected
                                              ? "Selected"
                                              : "Pruned"}
                                          </span>
                                        </div>
                                        <div className="probe-metrics-line tabular-num">
                                          <span>
                                            Probed: {probe.probed_hits}
                                          </span>
                                          <span>
                                            Accepted: {probe.accepted_hits}
                                          </span>
                                          <span>
                                            Rejected: {probe.rejected_hits}
                                          </span>
                                          <span>
                                            Calibrated:{" "}
                                            {probe.max_calibrated_score.toFixed(
                                              3
                                            )}
                                          </span>
                                        </div>
                                        {probe.top_hits.length > 0 && (
                                          <div className="probe-hits-list">
                                            {probe.top_hits
                                              .slice(0, 3)
                                              .map((hit) => (
                                                <div
                                                  key={hit.item_id}
                                                  className="probe-hit-item tabular-num"
                                                >
                                                  <span className="probe-hit-title">
                                                    {hit.title}
                                                  </span>
                                                  <span className="probe-hit-scores">
                                                    cos=
                                                    {hit.cosine_similarity.toFixed(
                                                      2
                                                    )}{" "}
                                                    bm25=
                                                    {hit.bm25_score.toFixed(2)}{" "}
                                                    trgm=
                                                    {hit.trigram_score.toFixed(
                                                      2
                                                    )}{" "}
                                                    cal=
                                                    {hit.calibrated_score.toFixed(
                                                      2
                                                    )}
                                                  </span>
                                                </div>
                                              ))}
                                          </div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                )}
                            </div>
                          )}

                          {/* Stage 3 X-Ray */}
                          {st.id === 3 && (
                            <div className="xray-kv-grid">
                              {trace.model_id && (
                                <div className="signal-kv">
                                  <dt>Local Model</dt>
                                  <dd>
                                    <Cpu size={11} /> {trace.model_id}
                                  </dd>
                                </div>
                              )}
                              {trace.adapter_mode && (
                                <div className="signal-kv">
                                  <dt>Adapter Mode</dt>
                                  <dd>{trace.adapter_mode}</dd>
                                </div>
                              )}
                              {typeof trace.chunks_fed === "number" && (
                                <div className="signal-kv">
                                  <dt>Full Chunks Fed</dt>
                                  <dd className="tabular-num">
                                    {trace.chunks_fed}
                                  </dd>
                                </div>
                              )}
                              {typeof trace.local_tokens_read === "number" && (
                                <div className="signal-kv">
                                  <dt>Local $0 Tokens</dt>
                                  <dd className="tabular-num">
                                    {trace.local_tokens_read.toLocaleString()}
                                  </dd>
                                </div>
                              )}
                              {trace.sufficiency_verdict && (
                                <div className="signal-kv">
                                  <dt>Sufficiency</dt>
                                  <dd>{trace.sufficiency_verdict}</dd>
                                </div>
                              )}
                              {typeof trace.grounded_facets === "number" &&
                                typeof trace.total_facets === "number" && (
                                  <div className="signal-kv">
                                    <dt>Facet Grounding</dt>
                                    <dd className="tabular-num">
                                      {trace.grounded_facets}/
                                      {trace.total_facets}
                                    </dd>
                                  </div>
                                )}
                            </div>
                          )}

                          {/* Stage 4 X-Ray */}
                          {st.id === 4 && (
                            <div className="xray-kv-section">
                              {trace.final_decision && (
                                <div className="xray-chips-line">
                                  <span className="field-label">
                                    Final Decision:
                                  </span>
                                  <span
                                    className={`status-tag ${
                                      trace.final_decision === "handoff_ready"
                                        ? "status-tag-ok"
                                        : "status-tag-warn"
                                    }`}
                                    data-testid="stage4-final-decision"
                                  >
                                    {trace.final_decision}
                                  </span>
                                  {typeof trace.paid_tokens_spent ===
                                    "number" && (
                                    <span className="xray-chip tabular-num">
                                      Paid tokens: {trace.paid_tokens_spent}
                                    </span>
                                  )}
                                </div>
                              )}
                              {trace.checks && trace.checks.length > 0 && (
                                <div
                                  className="checks-ledger-table"
                                  data-testid="stage4-checks-ledger"
                                >
                                  {trace.checks.map((chk) => (
                                    <div
                                      key={chk.check_name}
                                      className="check-ledger-row"
                                    >
                                      <span
                                        className={`status-tag ${
                                          chk.passed
                                            ? "status-tag-ok"
                                            : "status-tag-danger"
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
                              )}
                            </div>
                          )}
                        </div>
                      )}

                      {conf && Object.keys(conf.signals).length > 0 && (
                        <dl className="stage-signals-grid">
                          {Object.entries(conf.signals).map(([k, v]) => (
                            <div key={k} className="signal-kv">
                              <dt>{formatSignalKey(k)}</dt>
                              <dd className="tabular-num">
                                {formatSignalValue(v)}
                              </dd>
                            </div>
                          ))}
                        </dl>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
};
