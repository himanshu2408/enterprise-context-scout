import React, { useMemo, useState } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  GitBranch,
  History,
  Plus,
} from "lucide-react";
import {
  BranchOverlayOption,
  ScoutSessionSummary,
} from "../types/contracts";

export interface SessionBranchBarProps {
  sessions?: ScoutSessionSummary[];
  activeSessionId?: string;
  onSelectSession?: (sessionId: string) => void;
  onNewSession?: () => void;
  branches?: BranchOverlayOption[];
  selectedBranchId?: string;
  onSelectBranch?: (branchId: string) => void;
  commitDriftWarnings?: string[];
}

const DEFAULT_BRANCH_OPTIONS: BranchOverlayOption[] = [
  {
    branch_id: "head:main",
    label: "HEAD (main)",
    kind: "head",
    commit_sha: "HEAD",
    mr_iid: null,
    overlay_files_count: 0,
  },
  {
    branch_id: "mr:overlay-draft",
    label: "MR Overlay (Copy-on-Write)",
    kind: "mr_overlay",
    commit_sha: "overlay",
    mr_iid: "MR-Overlay",
    overlay_files_count: 0,
  },
];

export const SessionBranchBar: React.FC<SessionBranchBarProps> = ({
  sessions = [],
  activeSessionId,
  onSelectSession,
  onNewSession,
  branches,
  selectedBranchId,
  onSelectBranch,
  commitDriftWarnings = [],
}) => {
  const [drawerOpen, setDrawerOpen] = useState<boolean>(false);

  const branchOptions = useMemo<BranchOverlayOption[]>(() => {
    if (branches && branches.length > 0) {
      return branches;
    }
    return DEFAULT_BRANCH_OPTIONS;
  }, [branches]);

  const activeBranch = useMemo<BranchOverlayOption>(() => {
    if (selectedBranchId) {
      const found = branchOptions.find((b) => b.branch_id === selectedBranchId);
      if (found) return found;
    }
    return branchOptions[0];
  }, [branchOptions, selectedBranchId]);

  const activeSession = useMemo<ScoutSessionSummary | undefined>(() => {
    if (!activeSessionId) return sessions[0];
    return sessions.find((s) => s.session_id === activeSessionId) ?? sessions[0];
  }, [sessions, activeSessionId]);

  const hasCommitDrift =
    commitDriftWarnings.length > 0 || Boolean(activeSession?.has_commit_drift);

  const driftTooltip = useMemo(() => {
    const details = [
      ...commitDriftWarnings,
      ...(activeSession?.commit_drift_details ?? []),
    ];
    return details.length > 0
      ? details.join(" | ")
      : "HEAD commit changed since session started; re-verify grounded slices.";
  }, [commitDriftWarnings, activeSession]);

  return (
    <section
      className="session-branch-bar"
      aria-label="Session history and branch overlay controls"
      data-testid="session-branch-bar"
    >
      <div className="session-branch-bar-row">
        <div className="session-controls-group">
          <button
            type="button"
            className="btn-secondary btn-compact"
            onClick={() => setDrawerOpen((prev) => !prev)}
            aria-expanded={drawerOpen}
            aria-controls="session-history-drawer"
            data-testid="toggle-session-drawer-btn"
          >
            <History size={14} aria-hidden="true" />
            <span>
              Sessions ({sessions.length})
              {activeSession ? `: ${activeSession.title}` : ""}
            </span>
            {drawerOpen ? (
              <ChevronDown size={14} aria-hidden="true" />
            ) : (
              <ChevronRight size={14} aria-hidden="true" />
            )}
          </button>

          <button
            type="button"
            className="btn-secondary btn-compact"
            onClick={() => onNewSession?.()}
            data-testid="new-session-btn"
          >
            <Plus size={14} aria-hidden="true" />
            <span>New Session</span>
          </button>

          {hasCommitDrift && (
            <div
              className="commit-drift-badge"
              role="alert"
              title={driftTooltip}
              data-testid="commit-drift-badge"
            >
              <AlertTriangle size={14} aria-hidden="true" />
              <span>
                Commit Drift Detected
                {commitDriftWarnings.length > 0
                  ? ` (${commitDriftWarnings.length})`
                  : ""}
              </span>
            </div>
          )}
        </div>

        <div className="branch-overlay-group">
          <label htmlFor="branch-mr-picker" className="branch-picker-label">
            <GitBranch size={14} aria-hidden="true" />
            <span>Branch / MR Overlay</span>
          </label>
          <select
            id="branch-mr-picker"
            className="branch-picker-select"
            value={activeBranch.branch_id}
            onChange={(e) => onSelectBranch?.(e.target.value)}
            data-testid="branch-mr-picker"
          >
            {branchOptions.map((opt) => (
              <option key={opt.branch_id} value={opt.branch_id}>
                {opt.label}
                {opt.kind === "mr_overlay" &&
                typeof opt.overlay_files_count === "number" &&
                opt.overlay_files_count > 0
                  ? ` (${opt.overlay_files_count} files)`
                  : ""}
              </option>
            ))}
          </select>

          <span
            className={`active-branch-badge ${
              activeBranch.kind === "mr_overlay"
                ? "active-branch-badge-overlay"
                : "active-branch-badge-head"
            }`}
            data-testid="active-branch-badge"
          >
            {activeBranch.kind === "mr_overlay"
              ? `CoW Overlay${activeBranch.mr_iid ? ` · ${activeBranch.mr_iid}` : ""}`
              : `HEAD · ${activeBranch.commit_sha.slice(0, 8)}`}
          </span>
        </div>
      </div>

      {drawerOpen && (
        <div
          id="session-history-drawer"
          className="session-history-drawer"
          data-testid="session-history-drawer"
        >
          <div className="session-drawer-header">
            <h2 className="session-drawer-title">Multi-Turn Scout Sessions</h2>
            <span className="session-drawer-meta">
              {sessions.length} {sessions.length === 1 ? "session" : "sessions"}
            </span>
          </div>

          {sessions.length === 0 ? (
            <p className="session-drawer-empty">
              No prior multi-turn sessions recorded yet. Run a query or start a
              new session.
            </p>
          ) : (
            <ul className="session-history-list">
              {sessions.map((session) => {
                const isActive = session.session_id === activeSession?.session_id;
                return (
                  <li key={session.session_id} className="session-history-item">
                    <button
                      type="button"
                      className={`session-item-btn ${
                        isActive ? "session-item-btn-active" : ""
                      }`}
                      onClick={() => onSelectSession?.(session.session_id)}
                      aria-pressed={isActive}
                      data-testid={`select-session-${session.session_id}`}
                    >
                      <div className="session-item-main">
                        <span className="session-item-title">
                          {session.title}
                        </span>
                        <span className="session-item-stats tabular-num">
                          {session.turn_count}{" "}
                          {session.turn_count === 1 ? "turn" : "turns"} ·{" "}
                          {session.compacted_tokens.toLocaleString()} compacted
                          tokens · {session.domain_id}
                        </span>
                      </div>
                      <div className="session-item-badges">
                        {session.has_commit_drift && (
                          <span
                            className="session-drift-pill"
                            data-testid={`session-drift-pill-${session.session_id}`}
                          >
                            <AlertTriangle size={12} aria-hidden="true" />
                            Commit Drift
                          </span>
                        )}
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </section>
  );
};
