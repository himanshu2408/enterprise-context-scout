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
      description: "Polyglot zero-shot codebase",
      modalities: ["code"],
      sample_queries: [],
      lora_enabled: false,
      default_scopes: ["public_eng"],
      doc_chunks_count: 0,
      documents_count: 0,
      code_files_count: 10,
      ast_symbols_count: 120,
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
        selectedDomain="code_architecture_scout"
        onSelectDomain={onSelect}
      />
    );

    expect(screen.getByTestId("domain-card-federated_all")).toBeInTheDocument();
    expect(
      screen.getByTestId("domain-card-code_architecture_scout")
    ).toHaveClass("domain-row-active");

    fireEvent.click(screen.getByTestId("domain-card-enterprise_docs_scout"));
    expect(onSelect).toHaveBeenCalledWith("enterprise_docs_scout");

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
        selectedDomain="enterprise_docs_scout"
        onSelectDomain={vi.fn()}
      />
    );

    expect(
      screen.getByTestId("domain-card-enterprise_docs_scout")
    ).toHaveClass("domain-row-active");
    expect(
      screen.getByTestId("domain-card-code_architecture_scout")
    ).not.toHaveClass("domain-row-active");

    fireEvent.click(
      screen.getByTestId("toggle-adapter-metrics-code_architecture_scout")
    );
    const metricsBox = screen.getByTestId("adapter-metrics-code_architecture_scout");
    expect(metricsBox).toHaveTextContent("r=8, α=16");
    expect(metricsBox).toHaveTextContent("2 docs · 14 files");
    expect(metricsBox).toHaveTextContent("0.7200");
  });

  it("supports keyboard Enter and Space keys to select a domain card", () => {
    const onSelect = vi.fn();
    render(
      <DomainSelector
        domains={TEST_DOMAINS}
        selectedDomain="code_architecture_scout"
        onSelectDomain={onSelect}
      />
    );

    const docsCard = screen.getByTestId("domain-card-enterprise_docs_scout");
    fireEvent.keyDown(docsCard, { key: "Enter" });
    expect(onSelect).toHaveBeenCalledWith("enterprise_docs_scout");

    const codeCard = screen.getByTestId("domain-card-code_architecture_scout");
    fireEvent.keyDown(codeCard, { key: " " });
    expect(onSelect).toHaveBeenCalledWith("code_architecture_scout");
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
      screen.getByTestId("domain-card-code_architecture_scout")
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("domain-card-enterprise_docs_scout")
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
        selectedDomain="code_architecture_scout"
        onSelectDomain={vi.fn()}
        error="HTTP 503 Service Unavailable"
        onRetry={onRetry}
      />
    );

    expect(screen.getByTestId("domains-error-banner")).toHaveTextContent(
      "HTTP 503 Service Unavailable"
    );
    expect(
      screen.queryByTestId("domain-card-code_architecture_scout")
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("domains-retry-btn"));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
