import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { TokenSavingsMeter } from "../components/TokenSavingsMeter";
import { MOCK_ROI_METRICS } from "../api/client";

describe("TokenSavingsMeter Component", () => {
  it("renders default ROI metrics and token-savings breakdown", () => {
    render(<TokenSavingsMeter aggregatedMetrics={MOCK_ROI_METRICS} />);

    // Savings ratio
    expect(screen.getByTestId("frontier-savings-ratio")).toHaveTextContent(
      "97.65%"
    );

    // Local tokens explored
    expect(screen.getByTestId("local-tokens-explored")).toHaveTextContent(
      "14,850"
    );

    // Frontier tokens sent
    expect(screen.getByTestId("frontier-tokens-sent")).toHaveTextContent("348");

    // QLoRA overhead eliminated
    expect(screen.getByTestId("qlora-overhead-eliminated")).toHaveTextContent(
      "-1634.0"
    );

    // Cost avoided
    expect(screen.getByTestId("usd-saved")).toHaveTextContent("$0.29");

    // Adapter latency
    expect(screen.getByTestId("adapter-swap-latency")).toHaveTextContent(
      "0.018 ms"
    );
  });

  it("updates live stats when custom telemetry prop is provided", () => {
    const customTelemetry = {
      local_tokens_consumed: 28400,
      frontier_tokens_sent: 512,
      frontier_tokens_saved: 27888,
      estimated_usd_saved: 0.56,
      savings_ratio_pct: 98.2,
    };

    render(
      <TokenSavingsMeter
        telemetry={customTelemetry}
        aggregatedMetrics={MOCK_ROI_METRICS}
      />
    );

    expect(screen.getByTestId("frontier-savings-ratio")).toHaveTextContent(
      "98.20%"
    );
    expect(screen.getByTestId("local-tokens-explored")).toHaveTextContent(
      "28,400"
    );
    expect(screen.getByTestId("frontier-tokens-sent")).toHaveTextContent("512");
    expect(screen.getByTestId("usd-saved")).toHaveTextContent("$0.56");
  });
});
