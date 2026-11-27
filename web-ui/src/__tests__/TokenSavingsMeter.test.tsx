import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { TokenSavingsMeter } from "../components/TokenSavingsMeter";
import { TEST_ROI_METRICS, TEST_TOKEN_SAVINGS } from "./fixtures";

afterEach(() => {
  cleanup();
});

describe("TokenSavingsMeter", () => {
  it("renders empty query state rather than fake baseline savings before any query runs", () => {
    render(
      <TokenSavingsMeter
        currentQuerySavings={null}
        roiMetrics={TEST_ROI_METRICS}
      />
    );

    expect(screen.getByTestId("empty-query-savings")).toBeInTheDocument();
    expect(
      screen.queryByTestId("frontier-savings-ratio")
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("telemetry-source-badge")).toHaveTextContent(
      "PG17 Live Ledger"
    );
    expect(screen.getByTestId("cumulative-roi-card")).toHaveTextContent(
      "249 queries"
    );
    expect(screen.getByTestId("cumulative-roi-card")).toHaveTextContent(
      "$40.11 saved"
    );
  });

  it("renders exact live query token savings breakdown and Oct 2026 rates", () => {
    const { container } = render(
      <TokenSavingsMeter
        currentQuerySavings={TEST_TOKEN_SAVINGS}
        roiMetrics={TEST_ROI_METRICS}
      />
    );

    expect(screen.getByTestId("frontier-savings-ratio")).toHaveTextContent(
      "91.0%"
    );
    expect(screen.getByTestId("local-tokens-explored")).toHaveTextContent(
      "42,000"
    );
    expect(screen.getByTestId("frontier-tokens-sent")).toHaveTextContent(
      "3,800"
    );
    expect(
      screen.getByTestId("qlora-overhead-eliminated")
    ).toHaveTextContent("420 tok");
    expect(screen.getByTestId("usd-saved")).toHaveTextContent("$0.1528");
    expect(container.querySelector(".progress-bar-fill")).toHaveStyle({
      width: "90.95%",
    });
    expect(screen.getByTestId("token-savings-meter")).toHaveTextContent(
      "Gemini 4 Carbon"
    );
    expect(screen.getByTestId("token-savings-meter")).toHaveTextContent(
      "Claude Opus 5.5"
    );
  });

  it("supports collapsed strip mode and toggles Frontier Rates drawer cleanly", () => {
    render(
      <TokenSavingsMeter
        currentQuerySavings={TEST_TOKEN_SAVINGS}
        roiMetrics={TEST_ROI_METRICS}
        defaultExpanded={false}
      />
    );

    expect(
      screen.queryByTestId("frontier-rates-drawer")
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("toggle-telemetry-strip-btn"));
    const drawer = screen.getByTestId("frontier-rates-drawer");
    expect(drawer).toBeInTheDocument();
    expect(drawer).toHaveTextContent("Gemini 4 Hydrogen");
    expect(drawer).toHaveTextContent("GPT-5.6 Sol");

    fireEvent.click(screen.getByTestId("toggle-telemetry-strip-btn"));
    expect(
      screen.queryByTestId("frontier-rates-drawer")
    ).not.toBeInTheDocument();
  });

  it("clamps progress bar width safely for zero and >100% boundary ratios", () => {
    const { container, rerender } = render(
      <TokenSavingsMeter
        currentQuerySavings={{
          local_tokens_explored: 0,
          pack_tokens_sent: 0,
          tokens_saved: 0,
          estimated_usd_saved: 0,
          frontier_savings_ratio_pct: -12,
          qlora_overhead_eliminated_tokens: 0,
        }}
        roiMetrics={null}
      />
    );

    expect(container.querySelector(".progress-bar-fill")).toHaveStyle({
      width: "0%",
    });

    rerender(
      <TokenSavingsMeter
        currentQuerySavings={{
          local_tokens_explored: 10000,
          pack_tokens_sent: 100,
          tokens_saved: 9900,
          estimated_usd_saved: 0.04,
          frontier_savings_ratio_pct: 145.2,
          qlora_overhead_eliminated_tokens: 0,
        }}
        roiMetrics={null}
      />
    );

    expect(container.querySelector(".progress-bar-fill")).toHaveStyle({
      width: "100%",
    });
  });

  it("renders loading state while ROI telemetry is pending", () => {
    render(
      <TokenSavingsMeter
        currentQuerySavings={null}
        roiMetrics={null}
        roiLoading={true}
      />
    );

    expect(
      screen.getByText("Loading cumulative PostgreSQL ROI...")
    ).toBeInTheDocument();
  });

  it("renders explicit error banner when cumulative ROI fetch fails", () => {
    render(
      <TokenSavingsMeter
        currentQuerySavings={null}
        roiMetrics={null}
        roiError="Telemetry DB unreachable"
      />
    );

    expect(screen.getByTestId("roi-error-banner")).toHaveTextContent(
      "Telemetry DB unreachable"
    );
    expect(
      screen.queryByTestId("cumulative-roi-card")
    ).not.toBeInTheDocument();
  });
});
