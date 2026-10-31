import { describe, it, expect } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import App from "../App";

describe("App Root Component", () => {
  it("renders header with enterprise branding and hardware specifications", async () => {
    render(<App />);

    expect(screen.getByText("Enterprise Context Scout")).toBeInTheDocument();
    expect(
      screen.getByText(/Apple M5 Pro · 48GB Unified Memory/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/ShieldGemma 2 Guardrails Active/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/PostgreSQL 17 RLS Enforced/i)
    ).toBeInTheDocument();

    // Verify main components are mounted
    expect(screen.getByTestId("role-switcher")).toBeInTheDocument();
    expect(screen.getByTestId("domain-selector")).toBeInTheDocument();
    expect(screen.getByTestId("scout-console")).toBeInTheDocument();
    expect(screen.getByTestId("token-savings-meter")).toBeInTheDocument();
    expect(screen.getByTestId("citation-inspector")).toBeInTheDocument();

    // Domains loaded
    await waitFor(() => {
      expect(
        screen.getByTestId("domain-card-trading_code_scout")
      ).toBeInTheDocument();
    });
  });
});
