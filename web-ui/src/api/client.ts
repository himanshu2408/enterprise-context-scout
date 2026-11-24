import {
  AdminPurgeRequest,
  AdminPurgeResponse,
  AuthSessionRequest,
  AuthSessionResponse,
  BackendDomainDetail,
  ContextPack,
  DomainInfo,
  DomainsResponse,
  HandoffApplyRequest,
  HandoffApplyResponse,
  HandoffPreviewResponse,
  HealthResponse,
  PrincipalContext,
  ROIMetricsResponse,
  ScoutQueryRequest,
  ScoutQueryResponse,
  SourceFilterType,
  WorkspaceConfigResponse,
} from "../types/contracts";

export const FEDERATED_ALL_DOMAIN_ID = "federated_all";

export class ApiClientError extends Error {
  public readonly status: number;
  public readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiClientError";
    this.status = status;
    this.detail = detail;
  }
}

interface CachedSession {
  key: string;
  token: string;
  expiresAtMs: number;
  principal: PrincipalContext;
}

let cachedSession: CachedSession | null = null;

function buildPrincipalKey(principal: PrincipalContext): string {
  const scopes = [...principal.allowed_scopes].map((s) => s.trim()).join(",");
  return `${principal.user_id.trim()}|${principal.role}|${scopes}|${principal.max_session_paid_tokens}`;
}

export function clearCachedAuthSession(): void {
  cachedSession = null;
}

async function parseErrorDetail(res: Response): Promise<string> {
  const rawText = await res.text();
  if (!rawText) {
    return `HTTP ${res.status} ${res.statusText || "Request failed"}`;
  }
  try {
    const parsed = JSON.parse(rawText) as { detail?: unknown; message?: unknown };
    if (typeof parsed.detail === "string" && parsed.detail.trim()) {
      return parsed.detail;
    }
    if (typeof parsed.message === "string" && parsed.message.trim()) {
      return parsed.message;
    }
  } catch {
    // Plain text response body
  }
  return rawText;
}

export async function exchangeAuthSession(
  principal: PrincipalContext
): Promise<AuthSessionResponse> {
  const payload: AuthSessionRequest = {
    user_id: principal.user_id.trim() || "demo_engineer",
    role: principal.role,
    allowed_scopes: principal.allowed_scopes,
    max_session_paid_tokens: principal.max_session_paid_tokens,
  };

  let res: Response;
  try {
    res = await fetch("/api/v1/auth/session", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
  } catch (err: unknown) {
    cachedSession = null;
    throw new ApiClientError(
      0,
      `Authentication service unreachable: ${err instanceof Error ? err.message : String(err)}`
    );
  }

  if (!res.ok) {
    cachedSession = null;
    const detail = await parseErrorDetail(res);
    throw new ApiClientError(res.status, detail);
  }

  const session = (await res.json()) as AuthSessionResponse;
  const ttlMs = Math.max(60, (session.expires_in_seconds ?? 3600) - 30) * 1000;
  cachedSession = {
    key: buildPrincipalKey(principal),
    token: session.access_token,
    expiresAtMs: Date.now() + ttlMs,
    principal: session.principal,
  };
  return session;
}

export async function ensureBearerToken(principal: PrincipalContext): Promise<string> {
  const key = buildPrincipalKey(principal);
  if (
    cachedSession !== null &&
    cachedSession.key === key &&
    Date.now() < cachedSession.expiresAtMs
  ) {
    return cachedSession.token;
  }
  const session = await exchangeAuthSession(principal);
  return session.access_token;
}

export async function fetchCurrentPrincipal(
  principal?: PrincipalContext
): Promise<PrincipalContext> {
  const headers: Record<string, string> = {};
  if (principal) {
    const token = await ensureBearerToken(principal);
    headers.Authorization = `Bearer ${token}`;
  } else if (cachedSession && Date.now() < cachedSession.expiresAtMs) {
    headers.Authorization = `Bearer ${cachedSession.token}`;
  }

  let res: Response;
  try {
    res = await fetch("/api/v1/auth/me", { headers });
  } catch (err: unknown) {
    throw new ApiClientError(
      0,
      `Current principal endpoint unreachable: ${err instanceof Error ? err.message : String(err)}`
    );
  }
  if (!res.ok) {
    if (res.status === 401) {
      clearCachedAuthSession();
    }
    throw new ApiClientError(res.status, await parseErrorDetail(res));
  }
  return (await res.json()) as PrincipalContext;
}

export async function fetchHealth(): Promise<HealthResponse> {
  let res: Response;
  try {
    res = await fetch("/api/v1/health");
  } catch (err: unknown) {
    throw new ApiClientError(
      0,
      `Health endpoint unreachable: ${err instanceof Error ? err.message : String(err)}`
    );
  }
  if (!res.ok) {
    throw new ApiClientError(res.status, await parseErrorDetail(res));
  }
  return (await res.json()) as HealthResponse;
}

export function mapDomainDetailToInfo(detail: BackendDomainDetail): DomainInfo {
  const manifest = detail.manifest;
  const verification = detail.adapter_verification;
  const metrics = detail.training_metrics ?? {};
  const rawLoss =
    typeof verification.final_loss === "number"
      ? verification.final_loss
      : typeof metrics.final_loss === "number"
        ? metrics.final_loss
        : typeof metrics.training_loss === "number"
          ? metrics.training_loss
          : null;

  const modalities =
    detail.modalities && detail.modalities.length > 0
      ? detail.modalities
      : manifest.modalities && manifest.modalities.length > 0
        ? manifest.modalities
        : manifest.authority_rules?.primary_truth_source === "document"
          ? (["document"] as DomainInfo["modalities"])
          : (["code"] as DomainInfo["modalities"]);

  const sampleQueries =
    detail.sample_queries && detail.sample_queries.length > 0
      ? detail.sample_queries
      : manifest.sample_queries ?? [];

  const defaultScopes =
    detail.default_scopes && detail.default_scopes.length > 0
      ? detail.default_scopes
      : manifest.default_scopes ?? [];

  return {
    domain_id: detail.domain_id,
    display_name: detail.display_name || manifest.display_name || detail.domain_id,
    description: detail.description ?? manifest.description ?? "",
    modalities,
    sample_queries: sampleQueries,
    lora_enabled:
      typeof detail.lora_enabled === "boolean"
        ? detail.lora_enabled
        : typeof manifest.lora_enabled === "boolean"
          ? manifest.lora_enabled
          : true,
    default_scopes: defaultScopes,
    doc_chunks_count: detail.doc_chunks_count ?? 0,
    documents_count: detail.documents_count ?? 0,
    code_files_count: detail.code_files_count ?? 0,
    ast_symbols_count: detail.ast_symbols_count ?? 0,
    base_model_id: manifest.base_model_id,
    lora_adapter_id: manifest.lora_adapter_id,
    lora_adapter_path: manifest.lora_adapter_path,
    primary_truth_source: manifest.authority_rules?.primary_truth_source ?? "code",
    output_contract: manifest.output_contract ?? "ImplementationContextPack",
    qlora_rank: typeof verification.r === "number" ? verification.r : null,
    qlora_alpha:
      typeof verification.lora_alpha === "number" ? verification.lora_alpha : null,
    hot_swap_latency_ms:
      typeof verification.hot_swap_latency_ms === "number"
        ? verification.hot_swap_latency_ms
        : null,
    training_loss: rawLoss,
    safetensors_bytes:
      typeof verification.safetensors_bytes === "number"
        ? verification.safetensors_bytes
        : null,
    verified: Boolean(verification.is_valid),
  };
}

export async function fetchWorkspaceConfig(
  principal: PrincipalContext
): Promise<WorkspaceConfigResponse> {
  const token = await ensureBearerToken(principal);
  let res: Response;
  try {
    res = await fetch("/api/v1/workspace-config", {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
  } catch (err: unknown) {
    throw new ApiClientError(
      0,
      `Workspace config endpoint unreachable: ${err instanceof Error ? err.message : String(err)}`
    );
  }
  if (!res.ok) {
    if (res.status === 401) {
      clearCachedAuthSession();
    }
    throw new ApiClientError(res.status, await parseErrorDetail(res));
  }
  return (await res.json()) as WorkspaceConfigResponse;
}

export async function fetchDomains(): Promise<DomainInfo[]> {
  let res: Response;
  try {
    res = await fetch("/api/v1/domains");
  } catch (err: unknown) {
    throw new ApiClientError(
      0,
      `Domains endpoint unreachable: ${err instanceof Error ? err.message : String(err)}`
    );
  }
  if (!res.ok) {
    throw new ApiClientError(res.status, await parseErrorDetail(res));
  }

  const data = (await res.json()) as DomainsResponse;
  const list = Array.isArray(data.domains) ? data.domains : [];
  return list.map(mapDomainDetailToInfo);
}

export async function fetchROIMetrics(
  principal?: PrincipalContext
): Promise<ROIMetricsResponse> {
  const headers: Record<string, string> = {};
  if (principal) {
    try {
      const token = await ensureBearerToken(principal);
      headers.Authorization = `Bearer ${token}`;
    } catch {
      // Fall through so /api/v1/telemetry/roi still executes and reports its own status/error
    }
  } else if (cachedSession && Date.now() < cachedSession.expiresAtMs) {
    headers.Authorization = `Bearer ${cachedSession.token}`;
  }

  let res: Response;
  try {
    res = await fetch("/api/v1/telemetry/roi", { headers });
  } catch (err: unknown) {
    throw new ApiClientError(
      0,
      `Telemetry endpoint unreachable: ${err instanceof Error ? err.message : String(err)}`
    );
  }
  if (!res.ok) {
    throw new ApiClientError(res.status, await parseErrorDetail(res));
  }
  return (await res.json()) as ROIMetricsResponse;
}

export async function executeScoutQuery(
  request: ScoutQueryRequest & { source_filter?: SourceFilterType },
  principal: PrincipalContext
): Promise<ScoutQueryResponse> {
  const token = await ensureBearerToken(principal);
  const bodyPayload: Record<string, unknown> = {
    domain_id: request.domain_id,
    query: request.query,
    source_filter: request.source_filter ?? "all",
  };
  if (request.trace_id !== undefined) {
    bodyPayload.trace_id = request.trace_id;
  }
  if (request.max_bounce_backs !== undefined) {
    bodyPayload.max_bounce_backs = request.max_bounce_backs;
  }
  if (request.session_id !== undefined) {
    bodyPayload.session_id = request.session_id;
  }
  if (request.branch_id !== undefined) {
    bodyPayload.branch_id = request.branch_id;
  }

  let res: Response;
  try {
    res = await fetch("/api/v1/scout/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(bodyPayload),
    });
  } catch (err: unknown) {
    throw new ApiClientError(
      0,
      `Scout query request failed: ${err instanceof Error ? err.message : String(err)}`
    );
  }

  if (!res.ok) {
    if (res.status === 401) {
      clearCachedAuthSession();
    }
    throw new ApiClientError(res.status, await parseErrorDetail(res));
  }

  return (await res.json()) as ScoutQueryResponse;
}

export async function previewHandoff(
  contextPack: ContextPack,
  modelId: "gemini-3.8-flash" | "gemini-4-argon",
  principal: PrincipalContext
): Promise<HandoffPreviewResponse> {
  const token = await ensureBearerToken(principal);
  let res: Response;
  try {
    res = await fetch("/api/v1/handoff/preview", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        pack: contextPack,
        model_id: modelId,
      }),
    });
  } catch (err: unknown) {
    throw new ApiClientError(
      0,
      `Preview handoff unreachable: ${err instanceof Error ? err.message : String(err)}`
    );
  }

  if (!res.ok) {
    const detail = await parseErrorDetail(res);
    throw new ApiClientError(res.status, `Preview failed (${res.status}): ${detail}`);
  }

  return (await res.json()) as HandoffPreviewResponse;
}

export async function applyHandoff(
  request: HandoffApplyRequest,
  principal: PrincipalContext
): Promise<HandoffApplyResponse> {
  if (!request.confirmed) {
    throw new ApiClientError(
      400,
      "Explicit human confirmation is required to apply patch (SEC-04)"
    );
  }

  const token = await ensureBearerToken(principal);
  const safePayload: HandoffApplyRequest = {
    confirmation_token: request.confirmation_token,
    confirmed: request.confirmed,
  };
  let res: Response;
  try {
    res = await fetch("/api/v1/handoff/apply", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(safePayload),
    });
  } catch (err: unknown) {
    throw new ApiClientError(
      0,
      `Apply handoff unreachable: ${err instanceof Error ? err.message : String(err)}`
    );
  }

  if (!res.ok) {
    const detail = await parseErrorDetail(res);
    throw new ApiClientError(res.status, `Apply patch failed (${res.status}): ${detail}`);
  }

  return (await res.json()) as HandoffApplyResponse;
}

export async function executeAdminPurge(
  request: AdminPurgeRequest,
  principal?: PrincipalContext
): Promise<AdminPurgeResponse> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (principal) {
    try {
      const token = await ensureBearerToken(principal);
      headers.Authorization = `Bearer ${token}`;
    } catch {
      // Production OIDC/IAP mode relies on ambient credentials if dev token exchange is disabled
    }
  } else if (cachedSession && Date.now() < cachedSession.expiresAtMs) {
    headers.Authorization = `Bearer ${cachedSession.token}`;
  }

  let res: Response;
  try {
    res = await fetch("/api/v1/admin/purge", {
      method: "POST",
      headers,
      body: JSON.stringify(request),
    });
  } catch (err: unknown) {
    throw new ApiClientError(
      0,
      `Admin purge endpoint unreachable: ${err instanceof Error ? err.message : String(err)}`
    );
  }

  if (!res.ok) {
    if (res.status === 401) {
      clearCachedAuthSession();
    }
    throw new ApiClientError(res.status, await parseErrorDetail(res));
  }

  return (await res.json()) as AdminPurgeResponse;
}
