import React, { useMemo } from "react";
import { KeyRound, Lock, RotateCcw, ShieldCheck, User } from "lucide-react";
import {
  PrincipalContext,
  RoleType,
  WorkspaceRolePreset,
} from "../types/contracts";

export interface RoleSwitcherProps {
  principal: PrincipalContext;
  onChangePrincipal: (next: PrincipalContext) => void;
  rolePresets?: WorkspaceRolePreset[];
  availableScopes?: string[];
  authStatus?: "authenticating" | "ready" | "error";
  authError?: string | null;
  onRetryAuth?: () => void;
}

export const ROLE_DEFAULT_SCOPES: Record<RoleType, string[]> = {
  engineer: ["public_eng", "public_quant"],
  quant_researcher: ["public_quant"],
  product_manager: ["public_eng", "public_quant"],
  compliance_officer: ["public_eng", "public_quant"],
};

const ROLE_SUBTITLES: Record<RoleType, string> = {
  engineer: "Code, AST & Sandbox Patch",
  quant_researcher: "Papers, Formulas & Specs",
  product_manager: "Read-Only Architecture",
  compliance_officer: "Audit & Grounding Checks",
};

const FALLBACK_ROLE_PRESETS: WorkspaceRolePreset[] = [
  {
    role: "engineer",
    label: "Engineer",
    default_scopes: ROLE_DEFAULT_SCOPES.engineer,
  },
  {
    role: "quant_researcher",
    label: "Quant Researcher",
    default_scopes: ROLE_DEFAULT_SCOPES.quant_researcher,
  },
  {
    role: "product_manager",
    label: "Product Manager",
    default_scopes: ROLE_DEFAULT_SCOPES.product_manager,
  },
  {
    role: "compliance_officer",
    label: "Compliance Officer",
    default_scopes: ROLE_DEFAULT_SCOPES.compliance_officer,
  },
];

export const RoleSwitcher: React.FC<RoleSwitcherProps> = ({
  principal,
  onChangePrincipal,
  rolePresets,
  availableScopes,
  authStatus = "ready",
  authError = null,
  onRetryAuth,
}) => {
  const presets = useMemo(
    () =>
      rolePresets && rolePresets.length > 0
        ? rolePresets
        : FALLBACK_ROLE_PRESETS,
    [rolePresets]
  );

  const resolvedScopes = useMemo(() => {
    const set = new Set<string>(["public_eng", "public_quant", "restricted_risk"]);
    for (const preset of presets) {
      for (const sc of preset.default_scopes) {
        if (sc) set.add(sc);
      }
    }
    for (const sc of availableScopes ?? []) {
      if (sc) set.add(sc);
    }
    return Array.from(set);
  }, [presets, availableScopes]);

  const getDefaultScopesForRole = (role: RoleType): string[] => {
    const matched = presets.find((p) => p.role === role);
    if (matched && matched.default_scopes.length > 0) {
      return [...matched.default_scopes];
    }
    return [...(ROLE_DEFAULT_SCOPES[role] ?? ["public_eng"])];
  };

  const handleRoleSelect = (nextRole: RoleType): void => {
    onChangePrincipal({
      ...principal,
      role: nextRole,
      allowed_scopes: getDefaultScopesForRole(nextRole),
    });
  };

  const handleToggleScope = (scope: string): void => {
    const cleaned = principal.allowed_scopes.filter(
      (s) => s !== "unscoped_probe"
    );
    const exists = cleaned.includes(scope);
    const next = exists
      ? cleaned.filter((s) => s !== scope)
      : [...cleaned, scope];

    onChangePrincipal({
      ...principal,
      allowed_scopes: next.length > 0 ? next : ["unscoped_probe"],
    });
  };

  const handleUnscopedProbe = (): void => {
    onChangePrincipal({
      ...principal,
      allowed_scopes: ["unscoped_probe"],
    });
  };

  const handleResetDefaultScopes = (): void => {
    onChangePrincipal({
      ...principal,
      allowed_scopes: getDefaultScopesForRole(principal.role),
    });
  };

  const handleUserIdChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    onChangePrincipal({
      ...principal,
      user_id: e.target.value,
    });
  };

  const isUnscopedProbe =
    principal.allowed_scopes.length === 1 &&
    principal.allowed_scopes[0] === "unscoped_probe";

  return (
    <section
      className="scope-section role-scope-section"
      data-testid="role-switcher"
    >
      <div className="scope-section-head">
        <div className="scope-section-title">
          <User size={15} className="icon-accent" />
          <h2 className="section-heading">Persona &amp; RBAC Scope</h2>
        </div>
        <span
          className={`status-tag ${
            authStatus === "ready"
              ? "status-tag-ok"
              : authStatus === "authenticating"
                ? "status-tag-info"
                : "status-tag-danger"
          }`}
          data-testid="auth-status-badge"
        >
          <KeyRound size={12} />
          {authStatus === "ready"
            ? "Bearer Signed"
            : authStatus === "authenticating"
              ? "Minting..."
              : "Auth Failed"}
        </span>
      </div>

      {authStatus === "error" && authError && (
        <div
          className="inline-alert inline-alert-danger"
          data-testid="auth-error-banner"
        >
          <span>{authError}</span>
          {onRetryAuth && (
            <button
              type="button"
              className="btn-inline-retry"
              onClick={onRetryAuth}
              data-testid="retry-auth-btn"
            >
              Retry
            </button>
          )}
        </div>
      )}

      <div className="scope-role-top">
        <div className="user-id-inline">
          <label htmlFor="principal-user-id" className="field-label">
            User ID
          </label>
          <input
            id="principal-user-id"
            name="principal_user_id"
            type="text"
            className="text-input"
            value={principal.user_id}
            onChange={handleUserIdChange}
            placeholder="demo_engineer"
            data-testid="principal-user-id-input"
          />
        </div>

        <div className="role-pills-block">
          <span className="field-label">Role Persona</span>
          <div className="role-grid">
            {presets.map((opt) => {
              const active = principal.role === opt.role;
              const subtitle =
                ROLE_SUBTITLES[opt.role] ?? opt.default_scopes.join(", ");
              return (
                <button
                  key={opt.role}
                  type="button"
                  className={`role-btn ${active ? "role-btn-active" : ""}`}
                  onClick={() => handleRoleSelect(opt.role)}
                  data-testid={`role-btn-${opt.role}`}
                  aria-pressed={active}
                  title={subtitle}
                >
                  <span className="role-btn-title">{opt.label}</span>
                  <span className="role-btn-sub">{subtitle}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <div className="scope-rls-row">
        <div className="scope-rls-controls">
          <span className="field-label">RLS Scopes</span>
          <div className="scope-chips-row">
            {resolvedScopes.map((scope) => {
              const selected = principal.allowed_scopes.includes(scope);
              return (
                <button
                  key={scope}
                  type="button"
                  className={`scope-chip ${selected ? "scope-chip-active" : ""}`}
                  onClick={() => handleToggleScope(scope)}
                  data-testid={`scope-chip-${scope}`}
                  aria-pressed={selected}
                >
                  {scope}
                </button>
              );
            })}
          </div>
          <div className="scope-actions">
            <button
              type="button"
              className={`btn-quiet-toggle ${isUnscopedProbe ? "btn-quiet-toggle-active" : ""}`}
              onClick={handleUnscopedProbe}
              data-testid="unscoped-probe-btn"
              title="Simulate caller with zero matching corpus scopes"
            >
              Unscoped Probe
            </button>
            <button
              type="button"
              className="btn-quiet-toggle"
              onClick={handleResetDefaultScopes}
              data-testid="reset-scopes-btn"
            >
              <RotateCcw size={11} /> Reset
            </button>
          </div>
        </div>

        <div className="scope-rls-meta">
          <div className="active-scopes-bar" data-testid="active-scopes-list">
            <Lock size={12} className="icon-muted" />
            <span>
              Scopes: <code>{principal.allowed_scopes.join(", ")}</code>
            </span>
          </div>
          <div className="rbac-footnote">
            <ShieldCheck size={12} className="icon-emerald" />
            <span>
              {principal.role === "engineer"
                ? "Sandbox Patch Handoff enabled"
                : "Read-only Q&A mode"}
            </span>
          </div>
        </div>
      </div>
    </section>
  );
};
