import React from "react";
import { Shield, User, KeyRound, Check } from "lucide-react";
import { PrincipalContext, RoleType } from "../types/contracts";

export interface RoleSwitcherProps {
  principal: PrincipalContext;
  onPrincipalChange: (principal: PrincipalContext) => void;
}

export const ROLE_DEFAULT_SCOPES: Record<RoleType, string[]> = {
  engineer: ["public_eng", "repo:trading_strategies", "public_quant"],
  quant_researcher: ["public_quant", "quant_restricted", "corpus:quant_papers"],
  product_manager: ["public_eng", "public_quant"],
  compliance_officer: [
    "public_eng",
    "public_quant",
    "quant_restricted",
    "repo:trading_strategies",
  ],
};

const ROLE_DISPLAY_NAMES: Record<RoleType, string> = {
  engineer: "Lead Quantitative Engineer",
  quant_researcher: "Senior Quant Researcher",
  product_manager: "Trading Product Manager",
  compliance_officer: "Enterprise Compliance Officer",
};

export const RoleSwitcher: React.FC<RoleSwitcherProps> = ({
  principal,
  onPrincipalChange,
}) => {
  const handleRoleSelect = (role: RoleType): void => {
    const scopes = ROLE_DEFAULT_SCOPES[role];
    onPrincipalChange({
      ...principal,
      role,
      allowed_scopes: scopes,
    });
  };

  const handleUserIdChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    onPrincipalChange({
      ...principal,
      user_id: e.target.value,
    });
  };

  return (
    <div className="card role-switcher-card" data-testid="role-switcher">
      <div className="card-header">
        <div className="card-title">
          <Shield className="icon icon-sky" size={18} />
          <span>Out-of-Band Principal Identity (SEC-04)</span>
        </div>
        <div className="badge badge-purple">
          Max Session: {principal.max_session_paid_tokens.toLocaleString()} tokens
        </div>
      </div>

      <div className="role-switcher-grid">
        <div className="input-group">
          <label htmlFor="principal-user-id" className="label">
            <User size={14} className="icon-muted" /> Principal ID (X-Principal-User-Id)
          </label>
          <input
            id="principal-user-id"
            data-testid="principal-user-id-input"
            type="text"
            className="input-text"
            value={principal.user_id}
            onChange={handleUserIdChange}
            placeholder="demo_engineer"
          />
        </div>

        <div className="input-group">
          <label className="label">
            <KeyRound size={14} className="icon-muted" /> Role Profile
          </label>
          <div className="role-buttons" role="radiogroup" aria-label="Role selector">
            {(Object.keys(ROLE_DEFAULT_SCOPES) as RoleType[]).map((role) => {
              const isSelected = principal.role === role;
              return (
                <button
                  key={role}
                  type="button"
                  data-testid={`role-btn-${role}`}
                  className={`btn-pill ${isSelected ? "btn-pill-active" : ""}`}
                  onClick={() => handleRoleSelect(role)}
                  aria-checked={isSelected}
                  role="radio"
                >
                  {isSelected && <Check size={14} className="mr-1" />}
                  <span>{ROLE_DISPLAY_NAMES[role]}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <div className="scopes-container">
        <div className="scopes-label">Active RBAC / PostgreSQL 17 RLS Scopes:</div>
        <div className="scopes-list" data-testid="active-scopes-list">
          {principal.allowed_scopes.map((scope) => {
            const isRestricted =
              scope.includes("restricted") || scope.includes("strategies");
            const isQuant = scope.includes("quant");
            const badgeClass = isRestricted
              ? "badge-amber"
              : isQuant
                ? "badge-purple"
                : "badge-sky";

            return (
              <span
                key={scope}
                data-testid={`scope-chip-${scope}`}
                className={`badge ${badgeClass}`}
              >
                {scope}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
};
