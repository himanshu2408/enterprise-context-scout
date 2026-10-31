import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import {
  RoleSwitcher,
  ROLE_DEFAULT_SCOPES,
} from "../components/RoleSwitcher";
import { PrincipalContext } from "../types/contracts";

describe("RoleSwitcher Component", () => {
  const initialPrincipal: PrincipalContext = {
    user_id: "demo_engineer",
    role: "engineer",
    allowed_scopes: ROLE_DEFAULT_SCOPES.engineer,
    max_session_paid_tokens: 16000,
  };

  it("renders principal user ID and role buttons", () => {
    const onPrincipalChange = vi.fn();
    render(
      <RoleSwitcher
        principal={initialPrincipal}
        onPrincipalChange={onPrincipalChange}
      />
    );

    const input = screen.getByTestId("principal-user-id-input") as HTMLInputElement;
    expect(input.value).toBe("demo_engineer");

    expect(screen.getByTestId("role-btn-engineer")).toBeInTheDocument();
    expect(screen.getByTestId("role-btn-quant_researcher")).toBeInTheDocument();
    expect(screen.getByTestId("role-btn-product_manager")).toBeInTheDocument();
    expect(screen.getByTestId("role-btn-compliance_officer")).toBeInTheDocument();
  });

  it("renders all default scopes for engineer role", () => {
    const onPrincipalChange = vi.fn();
    render(
      <RoleSwitcher
        principal={initialPrincipal}
        onPrincipalChange={onPrincipalChange}
      />
    );

    expect(screen.getByTestId("scope-chip-public_eng")).toBeInTheDocument();
    expect(
      screen.getByTestId("scope-chip-repo:trading_strategies")
    ).toBeInTheDocument();
    expect(screen.getByTestId("scope-chip-public_quant")).toBeInTheDocument();
  });

  it("triggers onPrincipalChange when switching to quant_researcher", () => {
    const onPrincipalChange = vi.fn();
    render(
      <RoleSwitcher
        principal={initialPrincipal}
        onPrincipalChange={onPrincipalChange}
      />
    );

    const quantBtn = screen.getByTestId("role-btn-quant_researcher");
    fireEvent.click(quantBtn);

    expect(onPrincipalChange).toHaveBeenCalledTimes(1);
    expect(onPrincipalChange).toHaveBeenCalledWith(
      expect.objectContaining({
        role: "quant_researcher",
        allowed_scopes: expect.arrayContaining([
          "public_quant",
          "quant_restricted",
          "corpus:quant_papers",
        ]),
      })
    );
  });

  it("triggers onPrincipalChange when switching to compliance_officer", () => {
    const onPrincipalChange = vi.fn();
    render(
      <RoleSwitcher
        principal={initialPrincipal}
        onPrincipalChange={onPrincipalChange}
      />
    );

    const complianceBtn = screen.getByTestId("role-btn-compliance_officer");
    fireEvent.click(complianceBtn);

    expect(onPrincipalChange).toHaveBeenCalledTimes(1);
    expect(onPrincipalChange).toHaveBeenCalledWith(
      expect.objectContaining({
        role: "compliance_officer",
        allowed_scopes: expect.arrayContaining([
          "public_eng",
          "public_quant",
          "quant_restricted",
          "repo:trading_strategies",
        ]),
      })
    );
  });

  it("triggers onPrincipalChange when editing the user ID input", () => {
    const onPrincipalChange = vi.fn();
    render(
      <RoleSwitcher
        principal={initialPrincipal}
        onPrincipalChange={onPrincipalChange}
      />
    );

    const input = screen.getByTestId("principal-user-id-input");
    fireEvent.change(input, { target: { value: "lead_quant_42" } });

    expect(onPrincipalChange).toHaveBeenCalledTimes(1);
    expect(onPrincipalChange).toHaveBeenCalledWith(
      expect.objectContaining({
        user_id: "lead_quant_42",
      })
    );
  });
});
