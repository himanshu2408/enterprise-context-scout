import React, { useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  FileText,
  Filter,
  ShieldAlert,
} from "lucide-react";
import {
  ContextPack,
  RoleType,
  SideEffectCategory,
  SideEffectMatrixItem,
} from "../types/contracts";

export interface ProductTruthViewProps {
  contextPack: ContextPack | null;
  principalRole: RoleType;
  selectedDomain: string;
}

const SIGNAL_PATTERNS: Array<{
  category: SideEffectCategory;
  regex: RegExp;
  resourceLabel: string;
}> = [
  {
    category: "DB_WRITE",
    regex: /\b(INSERT|UPDATE|DELETE|ALTER|CREATE\s+TABLE|COMMIT|execute\(|session\.add|pg_)/i,
    resourceLabel: "PostgreSQL / Persistence",
  },
  {
    category: "HTTP_EGRESS",
    regex: /\b(fetch\(|axios\.|httpx\.|requests\.|urllib|http:\/\/|https:\/\/)/i,
    resourceLabel: "Outbound HTTP / Webhook",
  },
  {
    category: "AUTH_GUARD",
    regex: /\b(require_scope|PrincipalContext|JWT|Bearer|RLS|ROW\s+LEVEL\s+SECURITY|verify_token|acl)/i,
    resourceLabel: "Auth / RLS Policy Gate",
  },
  {
    category: "QUEUE_PUBLISH",
    regex: /\b(publish\(|enqueue|kafka|pubsub|celery|redis\.xadd|emit_event)/i,
    resourceLabel: "Event Bus / Job Queue",
  },
  {
    category: "ENV_CONTRACT",
    regex: /\b(os\.environ|getenv|process\.env|VITE_|SECRET_|API_KEY)/i,
    resourceLabel: "Environment & Secret Contract",
  },
];

function deriveSideEffectsFromPack(pack: ContextPack): SideEffectMatrixItem[] {
  if (pack.side_effects && pack.side_effects.length > 0) {
    return pack.side_effects;
  }

  const conflicts = pack.doc_vs_code_conflicts_detected ?? [];
  const specWarnings = pack.spec_drift_warnings ?? [];
  const hasAnyConflict = conflicts.length > 0 || specWarnings.length > 0;
  const conflictSummary = [...conflicts, ...specWarnings].join("; ");

  const derived: SideEffectMatrixItem[] = [];
  const seenKeys = new Set<string>();

  for (const slice of pack.code_slices) {
    const code = slice.verbatim_code_with_lines || "";
    for (const pattern of SIGNAL_PATTERNS) {
      if (!pattern.regex.test(code)) continue;
      const key = `${pattern.category}:${slice.file_path}:${slice.qualified_symbol}`;
      if (seenKeys.has(key)) continue;
      seenKeys.add(key);

      const matchesConflict =
        hasAnyConflict &&
        (conflictSummary.toLowerCase().includes(slice.qualified_symbol.toLowerCase()) ||
          conflictSummary.toLowerCase().includes(slice.file_path.toLowerCase()));

      derived.push({
        side_effect_id: `derived-${derived.length + 1}`,
        effect_type: pattern.category,
        qualified_symbol: slice.qualified_symbol,
        file_path: slice.file_path,
        line_range: `L${slice.start_line}-L${slice.end_line}`,
        target_resource: pattern.resourceLabel,
        spec_status: matchesConflict ? "SPEC_DRIFT_WARNING" : "VERIFIED",
        verification_note: matchesConflict
          ? `Spec drift detected against ${slice.qualified_symbol}`
          : `Grounded in ${slice.file_path} (${slice.modification_intent})`,
      });
    }
  }

  for (const conflict of conflicts) {
    derived.push({
      side_effect_id: `conflict-${derived.length + 1}`,
      effect_type: "ENV_CONTRACT",
      qualified_symbol: pack.domain_knowledge_brief?.brief_title ?? pack.domain_id,
      file_path: "spec-vs-code",
      line_range: "N/A",
      target_resource: "Documentation vs Implementation",
      spec_status: "SPEC_DRIFT_WARNING",
      verification_note: conflict,
    });
  }

  if (derived.length === 0 && pack.code_slices.length > 0) {
    const first = pack.code_slices[0];
    derived.push({
      side_effect_id: "grounded-slice-1",
      effect_type: "AUTH_GUARD",
      qualified_symbol: first.qualified_symbol,
      file_path: first.file_path,
      line_range: `L${first.start_line}-L${first.end_line}`,
      target_resource: "Pure / Read-Only Grounded Slice",
      spec_status: hasAnyConflict ? "SPEC_DRIFT_WARNING" : "VERIFIED",
      verification_note: hasAnyConflict
        ? conflictSummary
        : `No mutating side effects detected in ${first.file_path}.`,
    });
  }

  return derived;
}

function escapeCsvCell(value: string): string {
  const escaped = value.replace(/"/g, '""');
  return `"${escaped}"`;
}

function buildCsvExport(rows: SideEffectMatrixItem[]): string {
  const header = [
    "Category",
    "Symbol",
    "File",
    "Line Range",
    "Target Resource",
    "Spec Status",
    "Verification Note",
  ].join(",");

  const body = rows.map((row) =>
    [
      escapeCsvCell(row.effect_type),
      escapeCsvCell(row.qualified_symbol),
      escapeCsvCell(row.file_path),
      escapeCsvCell(row.line_range),
      escapeCsvCell(row.target_resource),
      escapeCsvCell(row.spec_status),
      escapeCsvCell(row.verification_note),
    ].join(",")
  );

  return [header, ...body].join("\n");
}

function buildMarkdownExport(
  pack: ContextPack,
  rows: SideEffectMatrixItem[],
  driftWarnings: string[]
): string {
  const lines: string[] = [
    `# Product Truth & Side-Effect Matrix — ${pack.domain_id}`,
    "",
    `- **Trace ID:** \`${pack.trace_id}\``,
    `- **Task:** ${pack.user_task}`,
    `- **Spec Drift Warnings:** ${driftWarnings.length}`,
    "",
  ];

  if (driftWarnings.length > 0) {
    lines.push("## SPEC_DRIFT_WARNING", "");
    for (const w of driftWarnings) {
      lines.push(`- ${w}`);
    }
    lines.push("");
  }

  lines.push(
    "## Side-Effect Matrix",
    "",
    "| Category | Symbol / File | Target Resource | Spec Status | Verification Note |",
    "| --- | --- | --- | --- | --- |"
  );

  for (const row of rows) {
    lines.push(
      `| ${row.effect_type} | \`${row.qualified_symbol}\` (${row.file_path}:${row.line_range}) | ${row.target_resource} | ${row.spec_status} | ${row.verification_note.replace(/\|/g, "\\|")} |`
    );
  }

  return lines.join("\n");
}

function triggerBlobDownload(filename: string, content: string, mimeType: string): void {
  if (typeof URL === "undefined" || typeof URL.createObjectURL !== "function") {
    return;
  }
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  if (typeof URL.revokeObjectURL === "function") {
    URL.revokeObjectURL(url);
  }
}

export const ProductTruthView: React.FC<ProductTruthViewProps> = ({
  contextPack,
  principalRole,
  selectedDomain,
}) => {
  const [filterMode, setFilterMode] = useState<"all" | "drift">("all");
  const [exportPreview, setExportPreview] = useState<{
    format: "csv" | "markdown";
    content: string;
  } | null>(null);

  const sideEffects = useMemo<SideEffectMatrixItem[]>(() => {
    if (!contextPack) return [];
    return deriveSideEffectsFromPack(contextPack);
  }, [contextPack]);

  const driftWarnings = useMemo<string[]>(() => {
    if (!contextPack) return [];
    return [
      ...(contextPack.spec_drift_warnings ?? []),
      ...(contextPack.doc_vs_code_conflicts_detected ?? []),
    ];
  }, [contextPack]);

  const hasDriftWarning = useMemo<boolean>(() => {
    return (
      driftWarnings.length > 0 ||
      sideEffects.some((row) => row.spec_status === "SPEC_DRIFT_WARNING")
    );
  }, [driftWarnings, sideEffects]);

  const filteredRows = useMemo<SideEffectMatrixItem[]>(() => {
    if (filterMode === "drift") {
      return sideEffects.filter((row) => row.spec_status === "SPEC_DRIFT_WARNING");
    }
    return sideEffects;
  }, [sideEffects, filterMode]);

  if (!contextPack) {
    return (
      <section
        className="product-truth-view product-truth-empty"
        data-testid="product-truth-view"
        aria-label="Product truth and side-effect matrix"
      >
        <div className="empty-state-card" data-testid="product-truth-empty">
          <FileSpreadsheet size={24} aria-hidden="true" />
          <h2 className="empty-state-title">Product Truth &amp; Side-Effect Matrix</h2>
          <p className="empty-state-body">
            Run a Scout query in <strong>{selectedDomain}</strong> to inspect
            grounded runtime side effects, doc-vs-code spec drift warnings, and
            exportable audit matrices (role: {principalRole}).
          </p>
        </div>
      </section>
    );
  }

  const handleExportCsv = (): void => {
    const csv = buildCsvExport(filteredRows);
    setExportPreview({ format: "csv", content: csv });
    triggerBlobDownload(
      `product-truth-${contextPack.domain_id}.csv`,
      csv,
      "text/csv;charset=utf-8"
    );
  };

  const handleExportMarkdown = (): void => {
    const md = buildMarkdownExport(contextPack, filteredRows, driftWarnings);
    setExportPreview({ format: "markdown", content: md });
    triggerBlobDownload(
      `product-truth-${contextPack.domain_id}.md`,
      md,
      "text/markdown;charset=utf-8"
    );
  };

  return (
    <section
      className="product-truth-view"
      data-testid="product-truth-view"
      aria-label="Product truth and side-effect matrix"
    >
      <div className="truth-header-row">
        <div>
          <h2 className="truth-title">Product Truth &amp; Side-Effect Matrix</h2>
          <p className="truth-subtitle">
            Domain: <code>{contextPack.domain_id}</code> · Trace:{" "}
            <code>{contextPack.trace_id}</code> · Role: <code>{principalRole}</code>
          </p>
        </div>

        <div className="truth-actions-bar">
          <div className="truth-filter-group" role="group" aria-label="Filter side effects">
            <Filter size={14} aria-hidden="true" />
            <button
              type="button"
              className={`btn-secondary btn-compact ${
                filterMode === "all" ? "btn-filter-active" : ""
              }`}
              aria-pressed={filterMode === "all"}
              onClick={() => setFilterMode("all")}
              data-testid="side-effect-filter-all"
            >
              All Effects ({sideEffects.length})
            </button>
            <button
              type="button"
              className={`btn-secondary btn-compact ${
                filterMode === "drift" ? "btn-filter-active" : ""
              }`}
              aria-pressed={filterMode === "drift"}
              onClick={() => setFilterMode("drift")}
              data-testid="side-effect-filter-drift"
            >
              Spec Drift Only (
              {sideEffects.filter((r) => r.spec_status === "SPEC_DRIFT_WARNING").length}
              )
            </button>
          </div>

          <div className="truth-export-group">
            <button
              type="button"
              className="btn-secondary btn-compact"
              onClick={handleExportCsv}
              data-testid="export-truth-csv-btn"
            >
              <Download size={14} aria-hidden="true" />
              <span>Export CSV</span>
            </button>
            <button
              type="button"
              className="btn-secondary btn-compact"
              onClick={handleExportMarkdown}
              data-testid="export-truth-md-btn"
            >
              <FileText size={14} aria-hidden="true" />
              <span>Export Markdown</span>
            </button>
          </div>
        </div>
      </div>

      {hasDriftWarning && (
        <div
          className="spec-drift-warning-banner"
          role="alert"
          data-testid="spec-drift-warning-banner"
        >
          <ShieldAlert size={18} aria-hidden="true" />
          <div>
            <strong>SPEC_DRIFT_WARNING:</strong> Documentation and live
            implementation diverge.
            {driftWarnings.length > 0 && (
              <ul className="spec-drift-list">
                {driftWarnings.map((warning, i) => (
                  <li key={`${i}-${warning.slice(0, 24)}`}>{warning}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}

      <div className="truth-table-wrap">
        <table
          className="truth-matrix-table"
          data-testid="side-effect-matrix-table"
        >
          <thead>
            <tr>
              <th scope="col">Category</th>
              <th scope="col">Symbol / File</th>
              <th scope="col">Target Resource</th>
              <th scope="col">Spec Status</th>
              <th scope="col">Verification Note</th>
            </tr>
          </thead>
          <tbody>
            {filteredRows.length === 0 ? (
              <tr>
                <td colSpan={5} className="truth-empty-row">
                  No side effects match the active filter.
                </td>
              </tr>
            ) : (
              filteredRows.map((row, idx) => (
                <tr
                  key={row.side_effect_id}
                  data-testid={`side-effect-row-${idx}`}
                >
                  <td>
                    <span className="effect-category-badge">{row.effect_type}</span>
                  </td>
                  <td>
                    <div className="truth-symbol-cell">
                      <code className="truth-symbol-name">
                        {row.qualified_symbol}
                      </code>
                      <span className="truth-file-loc">
                        {row.file_path} ({row.line_range})
                      </span>
                    </div>
                  </td>
                  <td>{row.target_resource}</td>
                  <td>
                    <span
                      className={`spec-status-badge ${
                        row.spec_status === "SPEC_DRIFT_WARNING"
                          ? "spec-status-drift"
                          : "spec-status-verified"
                      }`}
                    >
                      {row.spec_status === "SPEC_DRIFT_WARNING" ? (
                        <AlertTriangle size={12} aria-hidden="true" />
                      ) : (
                        <CheckCircle2 size={12} aria-hidden="true" />
                      )}
                      <span>{row.spec_status}</span>
                    </span>
                  </td>
                  <td>{row.verification_note}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {exportPreview && (
        <div className="truth-export-preview-wrap">
          <h3 className="truth-export-heading">
            Export Preview ({exportPreview.format.toUpperCase()})
          </h3>
          <pre
            className="truth-export-preview"
            data-testid="truth-export-preview"
          >
            {exportPreview.content}
          </pre>
        </div>
      )}
    </section>
  );
};
