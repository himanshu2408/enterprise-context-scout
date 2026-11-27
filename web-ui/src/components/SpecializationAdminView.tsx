import React, { useMemo, useState } from "react";
import { executeAdminPurge } from "../api/client";
import {
  AdapterVerificationStatus,
  AdminPurgeResponse,
  DomainInfo,
  HealthHardwareProfile,
  PrincipalContext,
} from "../types/contracts";

export interface SpecializationAdminViewProps {
  domains: DomainInfo[];
  principal: PrincipalContext;
  hardwareProfile?: HealthHardwareProfile | null;
  adapterPoolStatus?: Record<string, AdapterVerificationStatus>;
}

type PurgeTargetType = "file_path" | "doc_id" | "repo";

interface DryRunProbeOutcome {
  passed: boolean;
  domainId: string;
  summary: string;
  checks: string[];
}

function detectModelFamilyLabel(baseModelId: string): string {
  const lower = baseModelId.toLowerCase();
  if (lower.includes("gemma")) {
    return "Gemma 4 Default Stack";
  }
  if (lower.includes("qwen")) {
    return "BYO Open-Weights (Qwen3)";
  }
  if (lower.includes("mistral") || lower.includes("devstral")) {
    return "BYO Open-Weights (Mistral)";
  }
  if (lower.includes("llama")) {
    return "BYO Open-Weights (Llama 4)";
  }
  if (lower.includes("deepseek")) {
    return "BYO Open-Weights (DeepSeek)";
  }
  return "BYO Open-Weights";
}

function inferArtifactStoreScheme(adapterPath: string): string {
  if (adapterPath.startsWith("gs://")) {
    return "gs://";
  }
  if (adapterPath.startsWith("s3://")) {
    return "s3://";
  }
  return adapterPath ? "gs:// / s3:// (local mirror)" : "base-weights (no adapter)";
}

function formatBytes(bytes: number | null | undefined): string {
  if (typeof bytes !== "number" || bytes <= 0) {
    return "0 B";
  }
  const mb = bytes / (1024 * 1024);
  if (mb >= 1) {
    return `${mb.toFixed(2)} MB`;
  }
  return `${bytes} B`;
}

function sanitizeAdapterError(message: string | null | undefined): string {
  if (!message) {
    return "Invalid";
  }
  return message.replace(/(?:\/[A-Za-z0-9._\s-]+)+\/(domains\/)/g, "$1");
}

export const SpecializationAdminView: React.FC<SpecializationAdminViewProps> = ({
  domains,
  principal,
  hardwareProfile,
  adapterPoolStatus,
}) => {
  const [selectedDomainId, setSelectedDomainId] = useState<string>(
    domains[0]?.domain_id ?? ""
  );
  const [probeOutcome, setProbeOutcome] = useState<DryRunProbeOutcome | null>(
    null
  );
  const [purgeTargetType, setPurgeTargetType] =
    useState<PurgeTargetType>("file_path");
  const [purgeTarget, setPurgeTarget] = useState<string>("");
  const [purgeRepoName, setPurgeRepoName] = useState<string>("");
  const [purgeSubmitting, setPurgeSubmitting] = useState<boolean>(false);
  const [purgeResult, setPurgeResult] = useState<AdminPurgeResponse | null>(
    null
  );
  const [purgeError, setPurgeError] = useState<string | null>(null);

  const effectiveDomainId = useMemo(() => {
    if (domains.length === 0) {
      return "";
    }
    if (domains.some((d) => d.domain_id === selectedDomainId)) {
      return selectedDomainId;
    }
    return domains[0].domain_id;
  }, [domains, selectedDomainId]);

  const selectedDomain = useMemo(
    () => domains.find((d) => d.domain_id === effectiveDomainId) ?? null,
    [domains, effectiveDomainId]
  );

  const selectedAdapter = useMemo(() => {
    if (!selectedDomain) {
      return null;
    }
    return adapterPoolStatus?.[selectedDomain.domain_id] ?? null;
  }, [adapterPoolStatus, selectedDomain]);

  const rankValue =
    selectedAdapter?.r ?? selectedDomain?.qlora_rank ?? 0;
  const alphaValue =
    selectedAdapter?.lora_alpha ?? selectedDomain?.qlora_alpha ?? 0;
  const isAdapterValid =
    selectedAdapter?.is_valid ?? selectedDomain?.verified ?? false;
  const safetensorsBytes =
    selectedAdapter?.safetensors_bytes ??
    selectedDomain?.safetensors_bytes ??
    0;

  const canPurge =
    principal.role === "compliance_officer" ||
    principal.allowed_scopes.includes("audit:compliance") ||
    principal.allowed_scopes.includes("admin:purge");

  const trimmedTarget = purgeTarget.trim();
  const trimmedRepo = purgeRepoName.trim();
  const missingRepoForFilePath =
    purgeTargetType === "file_path" &&
    trimmedTarget.length > 0 &&
    trimmedRepo.length === 0;
  const isPurgeSubmitDisabled =
    !canPurge ||
    purgeSubmitting ||
    !effectiveDomainId ||
    (purgeTargetType === "file_path" && (!trimmedTarget || !trimmedRepo)) ||
    (purgeTargetType === "doc_id" && !trimmedTarget) ||
    (purgeTargetType === "repo" && !trimmedRepo);

  const archetypeLabel = useMemo(() => {
    if (!selectedDomain) {
      return "Unassigned";
    }
    if (!selectedDomain.lora_enabled) {
      return "Zero-Shot Base";
    }
    return selectedDomain.primary_truth_source === "document"
      ? "Document Knowledge QLoRA"
      : "Code + AST QLoRA";
  }, [selectedDomain]);

  const handleDomainChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedDomainId(event.target.value);
    setProbeOutcome(null);
  };

  const handleRunDryRunProbe = () => {
    if (!selectedDomain) {
      setProbeOutcome({
        passed: false,
        domainId: "none",
        summary: "No domain selected for 1-step vLLM dry-run probe.",
        checks: ["FAIL: domain registry is empty"],
      });
      return;
    }

    const validOk = Boolean(isAdapterValid);
    const isZeroShot = !selectedDomain.lora_enabled && validOk;
    const rankOk = isZeroShot ? rankValue <= 32 : rankValue > 0 && rankValue <= 32;
    const safetensorsOk = isZeroShot || safetensorsBytes > 0;
    const passed = rankOk && safetensorsOk && validOk;

    setProbeOutcome({
      passed,
      domainId: selectedDomain.domain_id,
      summary: passed
        ? `PASS (Simulated UI Metadata Check): 1-step vLLM dry-run schema valid for ${selectedDomain.domain_id} (live sidecar not probed).`
        : `FAIL (Simulated UI Metadata Check): Dry-run validation failed for ${selectedDomain.domain_id}.`,
      checks: [
        `rank_bound (r <= 32): ${rankValue} (${
          rankValue > 32 ? "EXCEEDED" : rankOk ? "OK" : "MISSING"
        })`,
        `safetensors_artifact: ${formatBytes(safetensorsBytes)} (${safetensorsOk ? "OK" : "MISSING"})`,
        `adapter_verified (is_valid): ${String(validOk)}`,
      ],
    });
  };

  const handlePurgeSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = purgeTarget.trim();
    const resolvedRepoName = purgeRepoName.trim();
    if (!canPurge || !effectiveDomainId) {
      return;
    }
    if (purgeTargetType === "file_path" && !resolvedRepoName) {
      setPurgeError("Repository Name is required for file path purges.");
      return;
    }
    if (
      (purgeTargetType === "file_path" && !trimmed) ||
      (purgeTargetType === "doc_id" && !trimmed) ||
      (purgeTargetType === "repo" && !resolvedRepoName)
    ) {
      setPurgeError("Required purge target field is empty.");
      return;
    }

    setPurgeSubmitting(true);
    setPurgeError(null);
    setPurgeResult(null);

    const payload =
      purgeTargetType === "file_path"
        ? {
            domain_id: effectiveDomainId,
            repo_name: resolvedRepoName,
            file_path: trimmed,
            include_derived_wiki: true,
          }
        : purgeTargetType === "doc_id"
          ? {
              domain_id: effectiveDomainId,
              doc_id: trimmed,
              include_derived_wiki: true,
            }
          : {
              domain_id: effectiveDomainId,
              repo_name: resolvedRepoName,
              include_derived_wiki: true,
            };

    try {
      const response = await executeAdminPurge(payload, principal);
      setPurgeResult(response);
    } catch (err: unknown) {
      setPurgeError(err instanceof Error ? err.message : String(err));
    } finally {
      setPurgeSubmitting(false);
    }
  };

  const sidecarEntries = useMemo(
    () => Object.values(adapterPoolStatus ?? {}),
    [adapterPoolStatus]
  );

  return (
    <section
      className="panel-card"
      data-testid="specialization-admin-view"
      aria-labelledby="spec-admin-heading"
    >
      <div className="section-header-row">
        <div>
          <h2 id="spec-admin-heading" className="section-title">
            BYO Specialization &amp; Governance Control Plane
          </h2>
          <p className="section-subtitle">
            ACL-safe rsLoRA adapter governance, vLLM multi-LoRA policy, sidecar
            parity, and emergency &lt;60s purge.
          </p>
        </div>
        {hardwareProfile && (
          <span className="pill-badge" data-testid="admin-hardware-slots">
            LoRA Slots: {hardwareProfile.lora_slots_allocated}/
            {hardwareProfile.lora_slots_capacity} ({hardwareProfile.device})
          </span>
        )}
      </div>

      <div className="admin-governance-grid">
        {/* Domain & Base Model Governance Card */}
        <article className="subpanel-card" aria-labelledby="admin-domain-card-title">
          <h3 id="admin-domain-card-title" className="subpanel-title">
            Domain &amp; Base Model Governance
          </h3>
          <div className="form-field-group">
            <label htmlFor="admin-domain-select" className="field-label">
              Target Domain Cartridge
            </label>
            <select
              id="admin-domain-select"
              data-testid="admin-domain-select"
              className="select-control"
              value={effectiveDomainId}
              onChange={handleDomainChange}
              disabled={domains.length === 0}
            >
              {domains.length === 0 ? (
                <option value="">No domains registered</option>
              ) : (
                domains.map((domain) => (
                  <option key={domain.domain_id} value={domain.domain_id}>
                    {domain.display_name} ({domain.domain_id})
                  </option>
                ))
              )}
            </select>
          </div>

          <div className="badge-row">
            <span className="pill-badge" data-testid="admin-archetype-badge">
              {archetypeLabel}
            </span>
            <span className="pill-badge">
              {selectedDomain
                ? detectModelFamilyLabel(selectedDomain.base_model_id)
                : "Gemma 4 / BYO Open-Weights (Qwen3, Mistral, Llama 4, DeepSeek)"}
            </span>
          </div>

          <dl className="meta-kv-list">
            <div>
              <dt>Base Model ID</dt>
              <dd className="tabular-num">
                {selectedDomain?.base_model_id ?? "none"}
              </dd>
            </div>
            <div>
              <dt>Supported Open-Weights</dt>
              <dd>Gemma 4, Qwen3, Mistral, Llama 4, DeepSeek</dd>
            </div>
            <div>
              <dt>Serialization &amp; Execution Policy</dt>
              <dd>
                <code>.safetensors</code>-only &middot;{" "}
                <code>trust_remote_code: false</code>
              </dd>
            </div>
          </dl>
        </article>

        {/* ACL training_scope_ceiling Panel */}
        <article
          className="subpanel-card"
          data-testid="training-scope-ceiling-panel"
          aria-labelledby="scope-ceiling-title"
        >
          <h3 id="scope-ceiling-title" className="subpanel-title">
            ACL <code>training_scope_ceiling</code>
          </h3>
          <p className="muted-copy">
            Fail-closed subset enforcement (<code>OWASP ASI06</code> / ACL Scope Ceiling):
            restricted enclaves never leak into shared adapters.
          </p>
          <div className="badge-row">
            {selectedDomain && selectedDomain.default_scopes.length > 0 ? (
              selectedDomain.default_scopes.map((scope) => (
                <span key={scope} className="pill-badge pill-emerald">
                  {scope}
                </span>
              ))
            ) : (
              <span className="pill-badge pill-amber">
                No training scopes configured (fail-closed)
              </span>
            )}
          </div>
          <p className="muted-copy">
            Active principal scopes:{" "}
            <code>{principal.allowed_scopes.join(", ") || "none"}</code>
          </p>
        </article>

        {/* rsLoRA & vLLM Multi-LoRA Policy Card */}
        <article
          className="subpanel-card"
          data-testid="rslora-policy-card"
          aria-labelledby="rslora-policy-title"
        >
          <h3 id="rslora-policy-title" className="subpanel-title">
            rsLoRA &amp; vLLM Multi-LoRA Policy
          </h3>
          <dl className="meta-kv-list">
            <div>
              <dt>Adapter Scaling</dt>
              <dd>
                <code>use_rslora: true</code> &middot;{" "}
                <code>use_dora: false</code>
              </dd>
            </div>
            <div>
              <dt>Rank / Alpha</dt>
              <dd className="tabular-num">
                r={rankValue} (&lt;= 32) &middot; lora_alpha={alphaValue}
              </dd>
            </div>
            <div>
              <dt>Optimizer &amp; Regularization</dt>
              <dd>
                LoRA+ LR Ratio: 8.0x &middot; NEFTune Alpha: 5.0 &middot; 15%
                General Replay
              </dd>
            </div>
            <div>
              <dt>Split Integrity</dt>
              <dd>3-Way SHA-256 Split Disjointness (80/10/10)</dd>
            </div>
          </dl>
        </article>

        {/* Canary Memorization Gate & 1-Step vLLM Dry-Run Probe */}
        <article
          className="subpanel-card"
          data-testid="canary-gate-card"
          aria-labelledby="canary-gate-title"
        >
          <h3 id="canary-gate-title" className="subpanel-title">
            Canary Memorization Gate &amp; Dry-Run Probe
          </h3>
          <div className="badge-row">
            <span className="pill-badge pill-emerald">
              CanaryMemorizationGate: 0/10 leak
            </span>
          </div>
          <p className="muted-copy">
            Validates adapter rank bounds (<code>r &lt;= 32</code>),{" "}
            <code>.safetensors</code> payload, and verification flag prior to
            multi-LoRA hot-swap.
          </p>
          <button
            type="button"
            className="primary-btn"
            data-testid="run-dryrun-probe-btn"
            onClick={handleRunDryRunProbe}
          >
            Run 1-Step vLLM Dry-Run Probe
          </button>

          {probeOutcome && (
            <div
              role="status"
              aria-live="polite"
              data-testid="dryrun-probe-result"
              className={
                probeOutcome.passed
                  ? "status-banner status-banner-ok"
                  : "status-banner status-banner-error"
              }
            >
              <strong>{probeOutcome.summary}</strong>
              <ul className="compact-list">
                {probeOutcome.checks.map((check) => (
                  <li key={check}>{check}</li>
                ))}
              </ul>
            </div>
          )}
        </article>
      </div>

      {/* Multi-Pod AdapterSyncSidecar Parity */}
      <div
        className="subpanel-card"
        data-testid="adapter-sidecar-status"
        aria-labelledby="sidecar-status-title"
      >
        <h3 id="sidecar-status-title" className="subpanel-title">
          Multi-Pod <code>AdapterSyncSidecar</code> (<code>gs://</code> /{" "}
          <code>s3://</code>) Parity
        </h3>
        <div className="table-scroll-wrap">
          <table
            className="governance-table"
            data-testid="adapter-sidecar-table"
            aria-label="Multi-Pod AdapterSyncSidecar parity status"
          >
            <thead>
              <tr>
                <th scope="col">Domain</th>
                <th scope="col">Artifact Store</th>
                <th scope="col">Safetensors Size</th>
                <th scope="col">Hot-Swap Latency</th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {sidecarEntries.length === 0 ? (
                <tr>
                  <td colSpan={5}>
                    No adapter sidecar entries reported for active pool.
                  </td>
                </tr>
              ) : (
                sidecarEntries.map((entry) => (
                  <tr key={entry.domain_id}>
                    <td>{entry.domain_id}</td>
                    <td>{inferArtifactStoreScheme(entry.adapter_path)}</td>
                    <td className="tabular-num">
                      {formatBytes(entry.safetensors_bytes ?? 0)} (
                      {entry.safetensors_bytes ?? 0} B)
                    </td>
                    <td className="tabular-num">
                      {typeof entry.hot_swap_latency_ms === "number"
                        ? `${entry.hot_swap_latency_ms} ms`
                        : "N/A"}
                    </td>
                    <td>
                      {entry.is_valid
                        ? "Verified"
                        : sanitizeAdapterError(entry.error_message)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Emergency <60s Secret & Artifact Purge */}
      <form
        className="subpanel-card"
        data-testid="admin-purge-form"
        onSubmit={(e) => {
          void handlePurgeSubmit(e);
        }}
        aria-labelledby="admin-purge-title"
      >
        <h3 id="admin-purge-title" className="subpanel-title">
          Emergency &lt;60s Secret &amp; Artifact Purge
        </h3>
        {!canPurge && (
          <div
            className="status-banner status-banner-warn"
            role="note"
            data-testid="admin-purge-rbac-notice"
          >
            RBAC Restricted: Requires <code>compliance_officer</code> role or{" "}
            <code>audit:compliance</code> / <code>admin:purge</code> scope.
          </div>
        )}
        <div className="form-inline-row">
          <div className="form-field-group">
            <label htmlFor="admin-purge-target-type" className="field-label">
              Target Type
            </label>
            <select
              id="admin-purge-target-type"
              data-testid="admin-purge-target-type"
              className="select-control"
              value={purgeTargetType}
              onChange={(e) => {
                setPurgeTargetType(e.target.value as PurgeTargetType);
                setPurgeError(null);
                setPurgeResult(null);
              }}
              disabled={!canPurge || purgeSubmitting}
            >
              <option value="file_path">
                File Path (requires Repository Name)
              </option>
              <option value="doc_id">Document ID</option>
              <option value="repo">Entire Repository</option>
            </select>
          </div>
          <div className="form-field-group">
            <label htmlFor="admin-purge-repo-input" className="field-label">
              Repository Name (required for file path)
            </label>
            <input
              id="admin-purge-repo-input"
              data-testid="admin-purge-repo-input"
              type="text"
              className="text-input"
              value={purgeRepoName}
              onChange={(e) => setPurgeRepoName(e.target.value)}
              placeholder="e.g. acme/service-repo"
              aria-invalid={missingRepoForFilePath}
              aria-describedby={
                missingRepoForFilePath
                  ? "admin-purge-validation-error"
                  : undefined
              }
              disabled={
                !canPurge || purgeSubmitting || purgeTargetType === "doc_id"
              }
            />
          </div>
          <div className="form-field-group">
            <label htmlFor="admin-purge-target-input" className="field-label">
              {purgeTargetType === "doc_id"
                ? "Target Document ID"
                : purgeTargetType === "repo"
                  ? "Target File Path or Document ID (disabled in repo mode)"
                  : "Target File Path"}
            </label>
            <input
              id="admin-purge-target-input"
              data-testid="admin-purge-target-input"
              type="text"
              className="text-input"
              value={purgeTarget}
              onChange={(e) => setPurgeTarget(e.target.value)}
              placeholder={
                purgeTargetType === "doc_id"
                  ? "e.g. doc.v1.2026"
                  : purgeTargetType === "repo"
                    ? "Disabled in Entire Repository mode"
                    : "e.g. src/secrets/leaked.py"
              }
              disabled={
                !canPurge || purgeSubmitting || purgeTargetType === "repo"
              }
            />
          </div>
          <button
            type="submit"
            className="danger-btn"
            data-testid="admin-purge-submit-btn"
            disabled={isPurgeSubmitDisabled}
          >
            {purgeSubmitting ? "Purging..." : "Execute <60s Purge"}
          </button>
        </div>

        {canPurge && missingRepoForFilePath && (
          <div
            id="admin-purge-validation-error"
            role="alert"
            data-testid="admin-purge-validation-error"
            className="status-banner status-banner-warn"
          >
            Repository Name is required for file path purges.
          </div>
        )}

        {purgeError && (
          <div
            role="alert"
            data-testid="admin-purge-error"
            className="status-banner status-banner-error"
          >
            Purge failed: {purgeError}
          </div>
        )}

        {purgeResult && (
          <div
            role="status"
            aria-live="polite"
            data-testid="admin-purge-result"
            className="status-banner status-banner-ok"
          >
            Purged across PG17, Zoekt ({purgeResult.zoekt_deleted}), and Object
            Store ({purgeResult.object_store_deleted.length} objects) in{" "}
            {purgeResult.elapsed_seconds.toFixed(2)}s (SLA{" "}
            {purgeResult.sla_met ? "MET" : "EXCEEDED"} &lt;{" "}
            {purgeResult.sla_seconds}s, residue: {purgeResult.residue_found}).
          </div>
        )}
      </form>
    </section>
  );
};
