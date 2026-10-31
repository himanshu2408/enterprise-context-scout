import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DomainSelector } from "../components/DomainSelector";
import { MOCK_DOMAINS } from "../api/client";

describe("DomainSelector Component", () => {
  it("renders all 3 domain cards with metadata and specs", () => {
    const onDomainChange = vi.fn();
    render(
      <DomainSelector
        selectedDomainId="trading_code_scout"
        onDomainChange={onDomainChange}
        domains={MOCK_DOMAINS}
      />
    );

    expect(screen.getByTestId("domain-card-trading_code_scout")).toBeInTheDocument();
    expect(screen.getByTestId("domain-card-quant_research_scout")).toBeInTheDocument();
    expect(screen.getByTestId("domain-card-federated_all")).toBeInTheDocument();

    expect(screen.getByText("Trading Code Scout")).toBeInTheDocument();
    expect(screen.getByText("Quant Research Scout")).toBeInTheDocument();
    expect(screen.getByText("Federated Cross-Domain Scout")).toBeInTheDocument();

    // Verify QLoRA specs
    expect(screen.getAllByText("r=16, α=32").length).toBe(3);
    expect(screen.getByText("mlx:gpu Loss: 1.2578")).toBeInTheDocument();
    expect(screen.getByText("mlx:gpu Loss: 1.7812")).toBeInTheDocument();
    expect(screen.getAllByText("0.018 ms swap").length).toBe(3);
  });

  it("marks the selected domain card as active", () => {
    const onDomainChange = vi.fn();
    render(
      <DomainSelector
        selectedDomainId="quant_research_scout"
        onDomainChange={onDomainChange}
        domains={MOCK_DOMAINS}
      />
    );

    const quantCard = screen.getByTestId("domain-card-quant_research_scout");
    expect(quantCard).toHaveClass("domain-card-active");
    expect(screen.getByText("ACTIVE ADAPTER")).toBeInTheDocument();
  });

  it("triggers onDomainChange when clicking another domain card", () => {
    const onDomainChange = vi.fn();
    render(
      <DomainSelector
        selectedDomainId="trading_code_scout"
        onDomainChange={onDomainChange}
        domains={MOCK_DOMAINS}
      />
    );

    const quantCard = screen.getByTestId("domain-card-quant_research_scout");
    fireEvent.click(quantCard);

    expect(onDomainChange).toHaveBeenCalledTimes(1);
    expect(onDomainChange).toHaveBeenCalledWith("quant_research_scout");
  });

  it("supports keyboard Enter key to select a domain", () => {
    const onDomainChange = vi.fn();
    render(
      <DomainSelector
        selectedDomainId="trading_code_scout"
        onDomainChange={onDomainChange}
        domains={MOCK_DOMAINS}
      />
    );

    const fedCard = screen.getByTestId("domain-card-federated_all");
    fireEvent.keyDown(fedCard, { key: "Enter", code: "Enter" });

    expect(onDomainChange).toHaveBeenCalledTimes(1);
    expect(onDomainChange).toHaveBeenCalledWith("federated_all");
  });
});
