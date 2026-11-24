import React, { useMemo, useState } from "react";
import {
  Check,
  Code,
  Copy,
  GitFork,
  Layers,
  Network,
} from "lucide-react";
import { ContextPack } from "../types/contracts";

export interface ArchitectureMapViewerProps {
  contextPack: ContextPack | null;
  selectedDomain: string;
}

type FlowchartDirection = "LR" | "TD";

interface ArchNode {
  id: string;
  label: string;
  kind: "target" | "dependency" | "caller" | "test" | "doc" | "effect";
  filePath: string;
  symbol: string;
  commitSha: string;
}

interface ArchEdge {
  from: string;
  to: string;
  label: string;
}

function sanitizeMermaidId(raw: string, fallbackIdx: number): string {
  const cleaned = raw.replace(/[^a-zA-Z0-9_]/g, "_");
  if (!cleaned) return `node_${fallbackIdx}`;
  return /^[a-zA-Z]/.test(cleaned) ? cleaned : `n_${cleaned}`;
}

function sanitizeMermaidLabel(raw: string): string {
  return raw.replace(/["`<>]/g, "'").slice(0, 48);
}

function buildGraphFromPack(pack: ContextPack): {
  nodes: ArchNode[];
  edges: ArchEdge[];
} {
  const nodes: ArchNode[] = [];
  const edges: ArchEdge[] = [];
  const seenIds = new Set<string>();

  const targetIds: string[] = [];

  pack.code_slices.forEach((slice, idx) => {
    const id = sanitizeMermaidId(`${slice.qualified_symbol}_${idx}`, idx);
    if (seenIds.has(id)) return;
    seenIds.add(id);

    let kind: ArchNode["kind"] = "dependency";
    if (slice.modification_intent === "modify_target") {
      kind = "target";
      targetIds.push(id);
    } else if (slice.modification_intent === "direct_caller") {
      kind = "caller";
    } else if (slice.modification_intent === "test_fixture") {
      kind = "test";
    }

    nodes.push({
      id,
      label: slice.qualified_symbol || slice.file_path,
      kind,
      filePath: slice.file_path,
      symbol: slice.qualified_symbol,
      commitSha: slice.git_commit_sha || "HEAD",
    });
  });

  const primaryTargetId = targetIds[0] ?? nodes[0]?.id;

  pack.code_slices.forEach((slice, idx) => {
    const id = sanitizeMermaidId(`${slice.qualified_symbol}_${idx}`, idx);
    if (!primaryTargetId || id === primaryTargetId) return;

    if (slice.modification_intent === "imported_type" || slice.modification_intent === "reference_only") {
      edges.push({ from: id, to: primaryTargetId, label: slice.modification_intent });
    } else if (slice.modification_intent === "direct_caller") {
      edges.push({ from: id, to: primaryTargetId, label: "calls" });
    } else if (slice.modification_intent === "test_fixture") {
      edges.push({ from: id, to: primaryTargetId, label: "verifies" });
    } else if (slice.modification_intent === "modify_target" && id !== primaryTargetId) {
      edges.push({ from: primaryTargetId, to: id, label: "co-modify" });
    }
  });

  pack.supporting_doc_citations.slice(0, 4).forEach((cite, idx) => {
    const id = sanitizeMermaidId(`doc_${cite.chunk_id}_${idx}`, nodes.length + idx);
    if (seenIds.has(id)) return;
    seenIds.add(id);
    nodes.push({
      id,
      label: cite.doc_title || cite.chunk_id,
      kind: "doc",
      filePath: cite.source_uri,
      symbol: cite.section_heading || cite.doc_id,
      commitSha: cite.version ?? "spec",
    });
    if (primaryTargetId) {
      edges.push({ from: id, to: primaryTargetId, label: "spec" });
    }
  });

  (pack.side_effects ?? []).slice(0, 4).forEach((effect, idx) => {
    const id = sanitizeMermaidId(`fx_${effect.effect_type}_${idx}`, nodes.length + idx);
    if (seenIds.has(id)) return;
    seenIds.add(id);
    nodes.push({
      id,
      label: `${effect.effect_type}: ${effect.target_resource}`,
      kind: "effect",
      filePath: effect.file_path,
      symbol: effect.qualified_symbol,
      commitSha: effect.spec_status,
    });
    if (primaryTargetId) {
      edges.push({ from: primaryTargetId, to: id, label: effect.effect_type });
    }
  });

  return { nodes, edges };
}

function compileMermaidSource(
  pack: ContextPack,
  direction: FlowchartDirection,
  nodes: ArchNode[],
  edges: ArchEdge[]
): string {
  if (pack.architecture_mermaid && pack.architecture_mermaid.trim().length > 0) {
    const raw = pack.architecture_mermaid.trim();
    if (/^flowchart\s+(LR|TD|TB|RL)/m.test(raw)) {
      return raw.replace(/^flowchart\s+(LR|TD|TB|RL)/m, `flowchart ${direction}`);
    }
    return `flowchart ${direction}\n${raw}`;
  }

  const lines: string[] = [`flowchart ${direction}`];
  for (const node of nodes) {
    const label = sanitizeMermaidLabel(node.label);
    lines.push(`  ${node.id}["${label}"]`);
  }
  for (const edge of edges) {
    const edgeLabel = sanitizeMermaidLabel(edge.label);
    lines.push(`  ${edge.from} -->|${edgeLabel}| ${edge.to}`);
  }
  return lines.join("\n");
}

export const ArchitectureMapViewer: React.FC<ArchitectureMapViewerProps> = ({
  contextPack,
  selectedDomain,
}) => {
  const [direction, setDirection] = useState<FlowchartDirection>("LR");
  const [showMermaidSource, setShowMermaidSource] = useState<boolean>(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  const graph = useMemo(() => {
    if (!contextPack) {
      return { nodes: [] as ArchNode[], edges: [] as ArchEdge[] };
    }
    return buildGraphFromPack(contextPack);
  }, [contextPack]);

  const mermaidSource = useMemo(() => {
    if (!contextPack) return "";
    return compileMermaidSource(contextPack, direction, graph.nodes, graph.edges);
  }, [contextPack, direction, graph.nodes, graph.edges]);

  const activeNode = useMemo(() => {
    if (graph.nodes.length === 0) return null;
    if (selectedNodeId) {
      const found = graph.nodes.find((n) => n.id === selectedNodeId);
      if (found) return found;
    }
    return graph.nodes[0];
  }, [graph.nodes, selectedNodeId]);

  const activeNodeEdges = useMemo(() => {
    if (!activeNode) return [];
    return graph.edges.filter(
      (e) => e.from === activeNode.id || e.to === activeNode.id
    );
  }, [activeNode, graph.edges]);

  if (!contextPack) {
    return (
      <section
        className="architecture-map-viewer architecture-map-empty"
        data-testid="architecture-map-viewer"
        aria-label="Architecture dependency map"
      >
        <div className="empty-state-card" data-testid="architecture-map-empty">
          <Network size={24} aria-hidden="true" />
          <h2 className="empty-state-title">Architecture Dependency Map</h2>
          <p className="empty-state-body">
            Run a Scout query in <strong>{selectedDomain}</strong> to render an
            air-gapped interactive SVG dependency graph and Mermaid flowchart.
          </p>
        </div>
      </section>
    );
  }

  const nodePositions = (() => {
    const map = new Map<string, { x: number; y: number }>();
    const cols = direction === "LR" ? 3 : 2;
    const xStep = direction === "LR" ? 230 : 280;
    const yStep = direction === "LR" ? 92 : 96;

    graph.nodes.forEach((node, idx) => {
      const col = idx % cols;
      const row = Math.floor(idx / cols);
      map.set(node.id, {
        x: 24 + col * xStep,
        y: 24 + row * yStep,
      });
    });
    return map;
  })();

  const cols = direction === "LR" ? 3 : 2;
  const rows = Math.max(1, Math.ceil(graph.nodes.length / cols));
  const svgWidth = cols * (direction === "LR" ? 230 : 280) + 40;
  const svgHeight = rows * (direction === "LR" ? 92 : 96) + 40;

  const handleCopyMermaid = async (): Promise<void> => {
    if (
      typeof navigator !== "undefined" &&
      navigator.clipboard &&
      typeof navigator.clipboard.writeText === "function"
    ) {
      await navigator.clipboard.writeText(mermaidSource);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <section
      className="architecture-map-viewer"
      data-testid="architecture-map-viewer"
      aria-label="Architecture dependency map"
    >
      <div className="arch-header-row">
        <div>
          <h2 className="arch-title">Architecture Dependency Map</h2>
          <p className="arch-subtitle">
            {graph.nodes.length} nodes · {graph.edges.length} edges · Air-gapped
            inline SVG ({direction})
          </p>
        </div>

        <div className="arch-controls-bar">
          <button
            type="button"
            className="btn-secondary btn-compact"
            onClick={() => setDirection((prev) => (prev === "LR" ? "TD" : "LR"))}
            aria-pressed={direction === "TD"}
            data-testid="arch-layout-toggle-btn"
          >
            <Layers size={14} aria-hidden="true" />
            <span>Layout: {direction === "LR" ? "Left-to-Right" : "Top-Down"}</span>
          </button>

          <button
            type="button"
            className="btn-secondary btn-compact"
            onClick={() => setShowMermaidSource((prev) => !prev)}
            aria-expanded={showMermaidSource}
            aria-controls="mermaid-source-panel"
            data-testid="toggle-mermaid-source-btn"
          >
            <Code size={14} aria-hidden="true" />
            <span>{showMermaidSource ? "Hide Mermaid" : "Mermaid Source"}</span>
          </button>
        </div>
      </div>

      <div className="arch-canvas-grid">
        <div className="arch-svg-container">
          <svg
            className="arch-svg"
            viewBox={`0 0 ${svgWidth} ${svgHeight}`}
            role="img"
            aria-label="Context pack dependency architecture diagram"
            data-testid="architecture-map-svg"
          >
            <defs>
              <marker
                id="arch-arrowhead"
                markerWidth="8"
                markerHeight="6"
                refX="7"
                refY="3"
                orient="auto"
              >
                <polygon points="0 0, 8 3, 0 6" fill="#38bdf8" />
              </marker>
            </defs>

            {graph.edges.map((edge, idx) => {
              const fromPos = nodePositions.get(edge.from);
              const toPos = nodePositions.get(edge.to);
              if (!fromPos || !toPos) return null;
              const x1 = fromPos.x + 95;
              const y1 = fromPos.y + 28;
              const x2 = toPos.x + 95;
              const y2 = toPos.y + 28;
              return (
                <g key={`${edge.from}-${edge.to}-${idx}`}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="#38bdf8"
                    strokeOpacity={0.55}
                    strokeWidth={1.5}
                    markerEnd="url(#arch-arrowhead)"
                  />
                </g>
              );
            })}

            {graph.nodes.map((node, idx) => {
              const pos = nodePositions.get(node.id) ?? { x: 20, y: 20 };
              const isSelected = activeNode?.id === node.id;
              return (
                <g
                  key={node.id}
                  transform={`translate(${pos.x}, ${pos.y})`}
                  className={`arch-svg-node arch-svg-node-${node.kind} ${
                    isSelected ? "arch-svg-node-selected" : ""
                  }`}
                  role="button"
                  tabIndex={0}
                  aria-label={`Inspect ${node.label}`}
                  onClick={() => setSelectedNodeId(node.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setSelectedNodeId(node.id);
                    }
                  }}
                  data-testid={`arch-node-${idx}`}
                >
                  <rect
                    width={190}
                    height={56}
                    rx={6}
                    fill={isSelected ? "#1e293b" : "#111620"}
                    stroke={isSelected ? "#38bdf8" : "#475569"}
                    strokeWidth={isSelected ? 2 : 1.2}
                  />
                  <text
                    x={10}
                    y={21}
                    fill="#eef2f7"
                    fontSize={11.5}
                    fontWeight={600}
                  >
                    {node.label.length > 24
                      ? `${node.label.slice(0, 23)}…`
                      : node.label}
                  </text>
                  <text x={10} y={40} fill="#94a3b8" fontSize={10}>
                    {node.kind.toUpperCase()} · {node.commitSha.slice(0, 7)}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        <aside
          className="arch-detail-panel"
          aria-label="Selected node details"
          data-testid="arch-selected-node-detail"
        >
          <h3 className="arch-detail-heading">
            <GitFork size={14} aria-hidden="true" />
            <span>Selected Node Detail</span>
          </h3>

          {activeNode ? (
            <dl className="arch-detail-list">
              <div className="arch-detail-row">
                <dt>Symbol</dt>
                <dd>
                  <code>{activeNode.symbol}</code>
                </dd>
              </div>
              <div className="arch-detail-row">
                <dt>File</dt>
                <dd>
                  <code>{activeNode.filePath}</code>
                </dd>
              </div>
              <div className="arch-detail-row">
                <dt>Commit / Version</dt>
                <dd>
                  <code>{activeNode.commitSha}</code>
                </dd>
              </div>
              <div className="arch-detail-row">
                <dt>Edges ({activeNodeEdges.length})</dt>
                <dd>
                  {activeNodeEdges.length === 0 ? (
                    <span>No incident edges</span>
                  ) : (
                    <ul className="arch-edge-list">
                      {activeNodeEdges.map((edge, i) => (
                        <li key={`${edge.from}-${edge.to}-${i}`}>
                          <code>{edge.from}</code> → <code>{edge.to}</code> (
                          {edge.label})
                        </li>
                      ))}
                    </ul>
                  )}
                </dd>
              </div>
            </dl>
          ) : (
            <p className="arch-detail-empty">Select a node to inspect.</p>
          )}
        </aside>
      </div>

      {showMermaidSource && (
        <div id="mermaid-source-panel" className="mermaid-source-panel">
          <div className="mermaid-source-header">
            <h3 className="mermaid-source-title">Mermaid Flowchart Source</h3>
            <button
              type="button"
              className="btn-secondary btn-compact"
              onClick={() => {
                void handleCopyMermaid();
              }}
              data-testid="copy-mermaid-source-btn"
            >
              {copied ? (
                <Check size={14} aria-hidden="true" />
              ) : (
                <Copy size={14} aria-hidden="true" />
              )}
              <span>{copied ? "Copied" : "Copy Source"}</span>
            </button>
          </div>
          <pre
            className="mermaid-source-code"
            data-testid="mermaid-source-code"
          >
            {mermaidSource}
          </pre>
        </div>
      )}
    </section>
  );
};
