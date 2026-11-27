import {
  PrincipalContext,
  ScoutQueryRequest,
  SourceFilterType,
  SSEBounceBackData,
  SSEContextPackReadyData,
  SSEErrorData,
  SSEStageEvent,
  SSEStageUpdateData,
} from "../types/contracts";
import { clearCachedAuthSession, resolveAuthHeaders } from "./client";

export interface SSEStreamCallbacks {
  onEvent: (event: SSEStageEvent) => void;
  onError: (err: Error) => void;
  onComplete: () => void;
}

export interface SSEStreamOptions {
  request: ScoutQueryRequest & { source_filter?: SourceFilterType };
  principal: PrincipalContext;
  onEvent: (event: SSEStageEvent) => void;
  onError: (err: Error) => void;
  onComplete: () => void;
}

function parseSSEFrame(
  rawFrame: string
): { eventName: string; dataStr: string } | null {
  const lines = rawFrame.split(/\r?\n/);
  let eventName = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (dataLines.length === 0) {
    return null;
  }
  return { eventName, dataStr: dataLines.join("\n") };
}

function dispatchSSEFrame(
  rawFrame: string,
  callbacks: SSEStreamCallbacks
): void {
  const parsed = parseSSEFrame(rawFrame);
  if (!parsed) return;

  try {
    const dataObj = JSON.parse(parsed.dataStr) as Record<string, unknown>;
    const resolvedType = parsed.eventName !== "message"
      ? parsed.eventName
      : typeof dataObj.event_type === "string"
        ? dataObj.event_type
        : "";

    if (resolvedType === "stage_update") {
      callbacks.onEvent({
        event: "stage_update",
        data: dataObj as unknown as SSEStageUpdateData,
      });
    } else if (resolvedType === "bounce_back") {
      callbacks.onEvent({
        event: "bounce_back",
        data: dataObj as unknown as SSEBounceBackData,
      });
    } else if (resolvedType === "context_pack_ready") {
      callbacks.onEvent({
        event: "context_pack_ready",
        data: dataObj as unknown as SSEContextPackReadyData,
      });
    } else if (resolvedType === "error") {
      callbacks.onEvent({
        event: "error",
        data: dataObj as unknown as SSEErrorData,
      });
    }
  } catch (parseErr: unknown) {
    callbacks.onError(
      new Error(
        `Malformed SSE frame received from server: ${
          parseErr instanceof Error ? parseErr.message : String(parseErr)
        }`
      )
    );
  }
}

async function extractHttpErrorDetail(res: Response): Promise<string> {
  const rawText = await res.text();
  if (!rawText) {
    return `Stream request rejected with HTTP ${res.status}`;
  }
  try {
    const parsed = JSON.parse(rawText) as { detail?: unknown; rationale?: unknown };
    if (typeof parsed.detail === "string" && parsed.detail.trim()) {
      return parsed.detail;
    }
    if (typeof parsed.rationale === "string" && parsed.rationale.trim()) {
      return parsed.rationale;
    }
  } catch {
    // Use plain text fallback
  }
  return rawText;
}

/**
 * Initiates a live authenticated Server-Sent Events stream against /api/v1/scout/stream.
 * Fails closed on authentication, HTTP, or transport errors without any mock fallback.
 */
export function streamScoutWorkflow(options: SSEStreamOptions): () => void {
  const { request, principal, onEvent, onError, onComplete } = options;
  const controller = new AbortController();
  let aborted = false;

  void (async () => {
    let authHeaders: Record<string, string>;
    try {
      authHeaders = await resolveAuthHeaders(principal);
    } catch (authErr: unknown) {
      if (aborted) return;
      onError(
        authErr instanceof Error
          ? authErr
          : new Error(`Failed to mint signed Bearer session: ${String(authErr)}`)
      );
      return;
    }

    if (aborted) return;

    const SECURITY_GUARDRAIL_TRIGGERED = "SECURITY_GUARDRAIL_TRIGGERED";

    const requestBody: Record<string, unknown> = {
      query: request.query,
      domain_id: request.domain_id,
      source_filter: request.source_filter ?? "all",
    };
    if (request.trace_id !== undefined) {
      requestBody.trace_id = request.trace_id;
    }
    if (request.max_bounce_backs !== undefined) {
      requestBody.max_bounce_backs = request.max_bounce_backs;
    }
    if (request.session_id !== undefined) {
      requestBody.session_id = request.session_id;
    }
    if (request.branch_id !== undefined) {
      requestBody.branch_id = request.branch_id;
    }

    let response: Response;
    try {
      response = await fetch("/api/v1/scout/stream", {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          ...authHeaders,
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });
    } catch (fetchErr: unknown) {
      if (aborted || (fetchErr instanceof Error && fetchErr.name === "AbortError")) {
        return;
      }
      onError(
        new Error(
          `SSE connection failed: ${
            fetchErr instanceof Error ? fetchErr.message : String(fetchErr)
          }`
        )
      );
      return;
    }

    if (!response.ok) {
      if (response.status === 401) {
        clearCachedAuthSession();
      }
      const detail = await extractHttpErrorDetail(response);
      if (aborted) return;
      if (response.status === 403) {
        onEvent({
          event: "error",
          data: {
            stage: "guardrails",
            status: "blocked",
            category: SECURITY_GUARDRAIL_TRIGGERED,
            rationale: detail,
            timestamp: new Date().toISOString(),
          },
        });
        onComplete();
        return;
      }
      onError(new Error(`HTTP ${response.status}: ${detail}`));
      return;
    }

    if (!response.body) {
      if (!aborted) {
        onError(new Error("SSE response body was empty."));
      }
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    try {
      while (!aborted) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        // Hold trailing '\r' so a split CRLF '\r\n\r\n' boundary across chunks is preserved.
        const trailingCr = buffer.endsWith("\r");
        const scanSlice = trailingCr ? buffer.slice(0, -1) : buffer;
        const frames = scanSlice.split(/\r?\n\r?\n/);
        buffer = (frames.pop() ?? "") + (trailingCr ? "\r" : "");

        for (const frame of frames) {
          if (frame.trim()) {
            dispatchSSEFrame(frame, { onEvent, onError, onComplete });
          }
        }
      }

      buffer += decoder.decode();
      if (!aborted && buffer.trim()) {
        const tailFrames = buffer.split(/\r?\n\r?\n/);
        for (const frame of tailFrames) {
          if (frame.trim()) {
            dispatchSSEFrame(frame, { onEvent, onError, onComplete });
          }
        }
      }
      if (!aborted) {
        onComplete();
      }
    } catch (readErr: unknown) {
      if (aborted || (readErr instanceof Error && readErr.name === "AbortError")) {
        return;
      }
      onError(
        readErr instanceof Error ? readErr : new Error(String(readErr))
      );
    } finally {
      if (aborted) {
        try {
          await reader.cancel();
        } catch {
          // Ignore stream cancel errors during abort cleanup
        }
      }
      try {
        reader.releaseLock();
      } catch {
        // Ignore if lock already released
      }
    }
  })();

  return () => {
    aborted = true;
    controller.abort();
  };
}
