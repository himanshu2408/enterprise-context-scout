import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DomainSelector } from "../components/DomainSelector";
import { DomainInfo } from "../types/contracts";
import { TEST_DOMAINS } from "./fixtures";

afterEach(() => {
  cleanup();
});

describe("DomainSelector", () => {
  it("renders federated route and live registered domains without fabricated null metrics", () => {
    const unseenWithNulls: DomainInfo = {
      domain_id: "polyglot_unseen_v2",
      display_name: "Zero-Shot Verification",
      base_model_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
      lora_adapter_id: "polyglot_base",
      lora_adapter_path: "",
      primary_truth_source: "code",
      output_contract: "ImplementationContextPack",
      qlora_rank: null,
      qlora_alpha: null,
      hot_swap_latency_ms: null,
      training_loss: null,
      safetensors_bytes: null,
      verified: true,
    };

    const onSelect = vi.fn();
    render(
      <DomainSelector
        domains={[...TEST_DOMAINS, unseenWithNulls]}
        selectedDomain="trading_code_scout"
        onSelectDomain={onSelect}
      />
    );

    expect(screen.getByTestId("domain-card-federated_all")).toBeInTheDocument();
    expect(
      screen.getByTestId("domain-card-trading_code_scout")
    ).toHaveClass("domain-card-active");

    fireEvent.click(screen.getByTestId("domain-card-quant_research_scout"));
    expect(onSelect).toHaveBeenCalledWith("quant_research_scout");

    fireEvent.click(
      screen.getByTestId("toggle-adapter-metrics-polyglot_unseen_v2")
    );
    const metricsBox = screen.getByTestId("adapter-metrics-polyglot_unseen_v2");
    expect(metricsBox).toHaveTextContent("N/A");
    expect(metricsBox).not.toHaveTextContent("0.018");
  });

  it("marks selected domain card as active and displays verified adapter metrics when expanded", () => {
    render(
      <DomainSelector
        domains={TEST_DOMAINS}
        selectedDomain="quant_research_scout"
        onSelectDomain={vi.fn()}
      />
    );

    expect(
      screen.getByTestId("domain-card-quant_research_scout")
    ).toHaveClass("domain-card-active");
    expect(
      screen.getByTestId("domain-card-trading_code_scout")
    ).not.toHaveClass("domain-card-active");

    fireEvent.click(
      screen.getByTestId("toggle-adapter-metrics-trading_code_scout")
    );
    const metricsBox = screen.getByTestId("adapter-metrics-trading_code_scout");
    expect(metricsBox).toHaveTextContent("r=8, α=16");
    expect(metricsBox).toHaveTextContent("0.02 ms");
    expect(metricsBox).toHaveTextContent("0.7200");
  });

  it("supports keyboard Enter and Space keys to select a domain card", () => {
    const onSelect = vi.fn();
    render(
      <DomainSelector
        domains={TEST_DOMAINS}
        selectedDomain="trading_code_scout"
        onSelectDomain={onSelect}
      />
    );

    const quantCard = screen.getByTestId("domain-card-quant_research_scout");
    fireEvent.keyDown(quantCard, { key: "Enter" });
    expect(onSelect).toHaveBeenCalledWith("quant_research_scout");

    const codeCard = screen.getByTestId("domain-card-trading_code_scout");
    fireEvent.keyDown(codeCard, { key: " " });
    expect(onSelect).toHaveBeenCalledWith("trading_code_scout");
  });

  it("filters standalone domains by sourceFilter and shows empty hint when none match", () => {
    const { rerender } = render(
      <DomainSelector
        domains={TEST_DOMAINS}
        selectedDomain="federated_all"
        onSelectDomain={vi.fn()}
        sourceFilter="code"
      />
    );

    expect(
      screen.getByTestId("domain-card-trading_code_scout")
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("domain-card-quant_research_scout")
    ).not.toBeInTheDocument();

    rerender(
      <DomainSelector
        domains={TEST_DOMAINS}
        selectedDomain="federated_all"
        onSelectDomain={vi.fn()}
        sourceFilter="jira"
      />
    );

    expect(screen.getByTestId("domain-card-federated_all")).toBeInTheDocument();
    expect(screen.getByTestId("domains-filter-empty")).toHaveTextContent(
      "No standalone domain registered for filter jira"
    );
  });

  it("renders loading skeleton while domains are syncing", () => {
    render(
      <DomainSelector
        domains={[]}
        selectedDomain="federated_all"
        onSelectDomain={vi.fn()}
        loading={true}
      />
    );

    expect(screen.getByTestId("domains-loading")).toHaveTextContent(
      "Querying live domain adapter registry"
    );
  });

  it("renders fail-closed error banner and triggers retry when /api/v1/domains fails", () => {
    const onRetry = vi.fn();
    render(
      <DomainSelector
        domains={[]}
        selectedDomain="trading_code_scout"
        onSelectDomain={vi.fn()}
        error="HTTP 503 Service Unavailable"
        onRetry={onRetry}
      />
    );

    expect(screen.getByTestId("domains-error-banner")).toHaveTextContent(
      "HTTP 503 Service Unavailable"
    );
    expect(
      screen.queryByTestId("domain-card-trading_code_scout")
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("domains-retry-btn"));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
