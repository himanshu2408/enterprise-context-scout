import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ROLE_DEFAULT_SCOPES, RoleSwitcher } from "../components/RoleSwitcher";
import { TEST_ENGINEER_PRINCIPAL } from "./fixtures";

afterEach(() => {
  cleanup();
});

describe("RoleSwitcher", () => {
  it("renders personas, default scopes, and Bearer status pill", () => {
    const { rerender } = render(
      <RoleSwitcher
        principal={TEST_ENGINEER_PRINCIPAL}
        onChangePrincipal={vi.fn()}
        authStatus="ready"
      />
    );

    expect(screen.getByTestId("role-switcher")).toBeInTheDocument();
    expect(screen.getByTestId("auth-status-badge")).toHaveTextContent(
      "Bearer Signed"
    );
    expect(screen.getByTestId("active-scopes-list")).toHaveTextContent(
      "public_eng, public_docs"
    );

    rerender(
      <RoleSwitcher
        principal={TEST_ENGINEER_PRINCIPAL}
        onChangePrincipal={vi.fn()}
        authStatus="authenticating"
      />
    );
    expect(screen.getByTestId("auth-status-badge")).toHaveTextContent(
      "Minting..."
    );
  });

  it("switches persona to product_manager with its default scopes and updates user ID", () => {
    const onChange = vi.fn();
    render(
      <RoleSwitcher
        principal={TEST_ENGINEER_PRINCIPAL}
        onChangePrincipal={onChange}
      />
    );

    fireEvent.click(screen.getByTestId("role-btn-product_manager"));
    expect(onChange).toHaveBeenCalledWith({
      ...TEST_ENGINEER_PRINCIPAL,
      role: "product_manager",
      allowed_scopes: ROLE_DEFAULT_SCOPES.product_manager,
    });

    fireEvent.change(screen.getByTestId("principal-user-id-input"), {
      target: { value: "pm_alice" },
    });
    expect(onChange).toHaveBeenCalledWith({
      ...TEST_ENGINEER_PRINCIPAL,
      user_id: "pm_alice",
    });
  });

  it("switches to domain_researcher and compliance_officer with expected RBAC default scopes", () => {
    const onChange = vi.fn();
    render(
      <RoleSwitcher
        principal={TEST_ENGINEER_PRINCIPAL}
        onChangePrincipal={onChange}
      />
    );

    fireEvent.click(screen.getByTestId("role-btn-domain_researcher"));
    expect(onChange).toHaveBeenLastCalledWith({
      ...TEST_ENGINEER_PRINCIPAL,
      role: "domain_researcher",
      allowed_scopes: ["public_docs"],
    });

    fireEvent.click(screen.getByTestId("role-btn-compliance_officer"));
    expect(onChange).toHaveBeenLastCalledWith({
      ...TEST_ENGINEER_PRINCIPAL,
      role: "compliance_officer",
      allowed_scopes: ["public_eng", "public_docs"],
    });
  });

  it("supports toggling scopes, resetting defaults, and switching to valid unscoped_probe", () => {
    const onChange = vi.fn();
    render(
      <RoleSwitcher
        principal={TEST_ENGINEER_PRINCIPAL}
        onChangePrincipal={onChange}
      />
    );

    fireEvent.click(screen.getByTestId("scope-chip-public_eng"));
    expect(onChange).toHaveBeenLastCalledWith({
      ...TEST_ENGINEER_PRINCIPAL,
      allowed_scopes: ["public_docs"],
    });

    fireEvent.click(screen.getByTestId("scope-chip-docs_restricted"));
    expect(onChange).toHaveBeenLastCalledWith({
      ...TEST_ENGINEER_PRINCIPAL,
      allowed_scopes: ["public_eng", "public_docs", "docs_restricted"],
    });

    fireEvent.click(screen.getByTestId("unscoped-probe-btn"));
    expect(onChange).toHaveBeenLastCalledWith({
      ...TEST_ENGINEER_PRINCIPAL,
      allowed_scopes: ["unscoped_probe"],
    });

    fireEvent.click(screen.getByTestId("reset-scopes-btn"));
    expect(onChange).toHaveBeenLastCalledWith({
      ...TEST_ENGINEER_PRINCIPAL,
      allowed_scopes: ROLE_DEFAULT_SCOPES.engineer,
    });
  });

  it("never emits an empty allowed_scopes array when toggling off the last active scope", () => {
    const onChange = vi.fn();
    render(
      <RoleSwitcher
        principal={{
          ...TEST_ENGINEER_PRINCIPAL,
          allowed_scopes: ["public_docs"],
        }}
        onChangePrincipal={onChange}
      />
    );

    fireEvent.click(screen.getByTestId("scope-chip-public_docs"));
    expect(onChange).toHaveBeenCalledWith({
      ...TEST_ENGINEER_PRINCIPAL,
      allowed_scopes: ["unscoped_probe"],
    });
  });

  it("displays explicit auth error banner with retry button when signing fails", () => {
    const onRetry = vi.fn();
    render(
      <RoleSwitcher
        principal={TEST_ENGINEER_PRINCIPAL}
        onChangePrincipal={vi.fn()}
        authStatus="error"
        authError="Session endpoint rejected in production"
        onRetryAuth={onRetry}
      />
    );

    expect(screen.getByTestId("auth-status-badge")).toHaveTextContent(
      "Auth Failed"
    );
    expect(screen.getByTestId("auth-error-banner")).toHaveTextContent(
      "Session endpoint rejected in production"
    );
    fireEvent.click(screen.getByTestId("retry-auth-btn"));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
