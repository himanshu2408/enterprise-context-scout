import { cleanup, fireEvent, render, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { clearCachedAuthSession } from "../api/client";
import { ArchitectureMapViewer } from "../components/ArchitectureMapViewer";
import { CitationInspector } from "../components/CitationInspector";
import { DomainSelector } from "../components/DomainSelector";
import { FrontierHandoffModal } from "../components/FrontierHandoffModal";
import { ProductTruthView } from "../components/ProductTruthView";
import { RoleSwitcher } from "../components/RoleSwitcher";
import { ScoutConsole } from "../components/ScoutConsole";
import { SessionBranchBar } from "../components/SessionBranchBar";
import { SpecializationAdminView } from "../components/SpecializationAdminView";
import { TokenSavingsMeter } from "../components/TokenSavingsMeter";
import {
  installMockApiFetch,
  TEST_BRANCHES,
  TEST_CODE_PACK,
  TEST_DOMAINS,
  TEST_ENGINEER_PRINCIPAL,
  TEST_HEALTH_RESPONSE,
  TEST_PIPELINE_STAGES,
  TEST_ROI_METRICS,
  TEST_SESSIONS,
  TEST_SIDE_EFFECTS,
  TEST_TOKEN_SAVINGS,
} from "./fixtures";

function resolveAriaLabelledBy(
  el: Element,
  root: HTMLElement
): string {
  const labelledBy = el.getAttribute("aria-labelledby");
  if (!labelledBy) {
    return "";
  }
  return labelledBy
    .split(/\s+/)
    .map((id) => {
      const target =
        root.querySelector(`[id="${CSS.escape(id)}"]`) ??
        document.getElementById(id);
      return target?.textContent?.trim() ?? "";
    })
    .filter(Boolean)
    .join(" ")
    .trim();
}

function getAccessibleName(el: Element, root: HTMLElement): string {
  const ariaLabel = el.getAttribute("aria-label")?.trim();
  if (ariaLabel) {
    return ariaLabel;
  }
  const labelledByText = resolveAriaLabelledBy(el, root);
  if (labelledByText) {
    return labelledByText;
  }
  const titleAttr = el.getAttribute("title")?.trim();
  if (titleAttr) {
    return titleAttr;
  }
  const text = el.textContent?.trim();
  if (text) {
    return text;
  }
  const imgAlt = el.querySelector("img[alt]")?.getAttribute("alt")?.trim();
  if (imgAlt) {
    return imgAlt;
  }
  const svgTitle = el.querySelector("svg title")?.textContent?.trim();
  if (svgTitle) {
    return svgTitle;
  }
  return "";
}

function getFormControlLabel(el: Element, root: HTMLElement): string {
  const ariaLabel = el.getAttribute("aria-label")?.trim();
  if (ariaLabel) {
    return ariaLabel;
  }
  const labelledByText = resolveAriaLabelledBy(el, root);
  if (labelledByText) {
    return labelledByText;
  }
  const id = el.getAttribute("id");
  if (id) {
    const matchingLabel = root.querySelector(`label[for="${CSS.escape(id)}"]`);
    const labelText = matchingLabel?.textContent?.trim();
    if (labelText) {
      return labelText;
    }
  }
  const parentLabel = el.closest("label");
  if (parentLabel?.textContent?.trim()) {
    return parentLabel.textContent.trim();
  }
  return "";
}

/**
 * Deterministic DOM-level WCAG 2.2 AA audit covering:
 * - Duplicate IDs (4.1.1)
 * - Interactive control accessible names (4.1.2)
 * - Form field labeling (1.3.1 / 4.1.2)
 * - Table <th scope> semantics (1.3.1)
 * - Heading hierarchy without skipped levels (1.3.1 / 2.4.6)
 * - Dialog role/modal/label semantics (4.1.2)
 * - Tablist/tab/tabpanel ARIA relationships (4.1.2)
 * - Non-decorative SVG/image alt labels (1.1.1)
 * - Positive tabIndex anti-pattern (2.4.3)
 */
export function auditWcag22AA(root: HTMLElement): string[] {
  const violations: string[] = [];

  // 1. Duplicate IDs
  const seenIds = new Map<string, number>();
  for (const el of Array.from(root.querySelectorAll("[id]"))) {
    const id = el.getAttribute("id") ?? "";
    if (!id) continue;
    const count = (seenIds.get(id) ?? 0) + 1;
    seenIds.set(id, count);
    if (count === 2) {
      violations.push(`Duplicate id="${id}"`);
    }
  }

  // 2. Interactive control accessible names
  const interactiveSelector =
    'button, a[href], [role="button"], [role="tab"], [role="menuitem"], [role="switch"]';
  for (const el of Array.from(root.querySelectorAll(interactiveSelector))) {
    if (el.getAttribute("aria-hidden") === "true") continue;
    const name = getAccessibleName(el, root);
    if (!name) {
      violations.push(
        `Interactive control <${el.tagName.toLowerCase()}> missing accessible name (testId=${el.getAttribute("data-testid") ?? "none"})`
      );
    }
  }

  // 3. Form controls labeling
  const formControls = root.querySelectorAll(
    'input:not([type="hidden"]):not([type="button"]):not([type="submit"]):not([type="reset"]), select, textarea'
  );
  for (const el of Array.from(formControls)) {
    const label = getFormControlLabel(el, root);
    if (!label) {
      violations.push(
        `Form control <${el.tagName.toLowerCase()}> missing label (id=${el.getAttribute("id") ?? "none"}, testId=${el.getAttribute("data-testid") ?? "none"})`
      );
    }
  }

  // 4. Table <th scope>
  for (const table of Array.from(root.querySelectorAll("table"))) {
    const headers = Array.from(table.querySelectorAll("th"));
    if (headers.length === 0) {
      violations.push(
        `<table> missing <th> header cells (testId=${table.getAttribute("data-testid") ?? "none"})`
      );
    }
    for (const th of headers) {
      const scope = th.getAttribute("scope");
      if (
        scope !== "col" &&
        scope !== "row" &&
        scope !== "colgroup" &&
        scope !== "rowgroup"
      ) {
        violations.push(
          `<th> "${th.textContent?.trim() ?? ""}" missing valid scope attribute`
        );
      }
    }
  }

  // 5. Heading hierarchy
  const headings = Array.from(
    root.querySelectorAll('h1, h2, h3, h4, h5, h6, [role="heading"]')
  );
  let prevLevel = 0;
  for (const h of headings) {
    const tag = h.tagName.toLowerCase();
    const level = tag.startsWith("h")
      ? Number(tag.slice(1))
      : Number(h.getAttribute("aria-level") ?? "2");
    if (!(h.textContent ?? "").trim()) {
      violations.push(`Empty heading <${tag}>`);
    }
    if (prevLevel > 0 && level > prevLevel + 1) {
      violations.push(
        `Skipped heading level: h${prevLevel} -> h${level} ("${h.textContent?.trim() ?? ""}")`
      );
    }
    prevLevel = level;
  }

  // 6. Dialog semantics
  for (const dlg of Array.from(root.querySelectorAll('[role="dialog"], dialog'))) {
    if (dlg.getAttribute("aria-modal") !== "true") {
      violations.push(
        `Dialog missing aria-modal="true" (testId=${dlg.getAttribute("data-testid") ?? "none"})`
      );
    }
    const name = getAccessibleName(dlg, root);
    if (!name) {
      violations.push(
        `Dialog missing aria-label or aria-labelledby (testId=${dlg.getAttribute("data-testid") ?? "none"})`
      );
    }
  }

  // 7. Tablist / tab ARIA
  for (const tab of Array.from(root.querySelectorAll('[role="tab"]'))) {
    if (!tab.closest('[role="tablist"]')) {
      violations.push(
        `[role="tab"] "${tab.textContent?.trim() ?? ""}" not inside [role="tablist"]`
      );
    }
    const selected = tab.getAttribute("aria-selected");
    if (selected !== "true" && selected !== "false") {
      violations.push(
        `[role="tab"] "${tab.textContent?.trim() ?? ""}" missing aria-selected="true"|"false"`
      );
    }
  }
  for (const panel of Array.from(root.querySelectorAll('[role="tabpanel"]'))) {
    if (
      !panel.getAttribute("aria-labelledby") &&
      !panel.getAttribute("aria-label")
    ) {
      violations.push(`[role="tabpanel"] missing aria-labelledby or aria-label`);
    }
  }

  // 8. Images & non-decorative SVGs
  for (const img of Array.from(root.querySelectorAll("img"))) {
    if (!img.hasAttribute("alt")) {
      violations.push(`<img> missing alt attribute`);
    }
  }
  for (const svg of Array.from(root.querySelectorAll("svg"))) {
    if (svg.getAttribute("aria-hidden") === "true") continue;
    if (svg.closest(interactiveSelector)) continue;
    const isDecorativeLucide =
      svg.classList.contains("lucide") &&
      Boolean(svg.parentElement?.textContent?.trim());
    if (isDecorativeLucide) continue;
    const role = svg.getAttribute("role");
    const hasLabel =
      Boolean(svg.getAttribute("aria-label")?.trim()) ||
      Boolean(resolveAriaLabelledBy(svg, root)) ||
      Boolean(svg.querySelector("title")?.textContent?.trim());
    if (role !== "img" || !hasLabel) {
      violations.push(
        `Non-decorative <svg> outside interactive control must have role="img" and accessible label (testId=${svg.getAttribute("data-testid") ?? "none"})`
      );
    }
  }

  // 9. Positive tabIndex
  for (const el of Array.from(root.querySelectorAll("[tabindex]"))) {
    const idx = Number(el.getAttribute("tabindex"));
    if (Number.isFinite(idx) && idx > 0) {
      violations.push(
        `Positive tabIndex=${idx} on <${el.tagName.toLowerCase()}> breaks natural focus order`
      );
    }
  }

  return violations;
}

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace("#", "").trim();
  const num = parseInt(clean, 16);
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

function relativeLuminance(hex: string): number {
  const [r, g, b] = hexToRgb(hex).map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrastRatio(fgHex: string, bgHex: string): number {
  const l1 = relativeLuminance(fgHex);
  const l2 = relativeLuminance(bgHex);
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (lighter + 0.05) / (darker + 0.05);
}

beforeEach(() => {
  clearCachedAuthSession();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("WCAG 2.2 AA Accessibility Audit Suite", () => {
  it("audits App shell with zero WCAG 2.2 AA violations", async () => {
    installMockApiFetch();
    const { container } = render(<App />);

    await waitFor(() => {
      expect(
        container.querySelector('[data-testid="hardware-profile-chip"]')
      ).not.toBeNull();
    });

    expect(auditWcag22AA(container)).toEqual([]);
  });

  it("audits FrontierHandoffModal with zero WCAG 2.2 AA violations", async () => {
    installMockApiFetch();
    const { container } = render(
      <FrontierHandoffModal
        isOpen={true}
        onClose={vi.fn()}
        contextPack={TEST_CODE_PACK}
        principal={TEST_ENGINEER_PRINCIPAL}
      />
    );

    await waitFor(() => {
      expect(
        container.querySelector('[data-testid="unified-diff-view"]')
      ).not.toBeNull();
    });

    expect(auditWcag22AA(container)).toEqual([]);
  });

  it("audits ProductTruthView, ArchitectureMapViewer, SessionBranchBar, and SpecializationAdminView with zero WCAG 2.2 AA violations", () => {
    const views = [
      <ProductTruthView
        key="truth"
        contextPack={{ ...TEST_CODE_PACK, side_effects: TEST_SIDE_EFFECTS }}
        principalRole="product_manager"
        selectedDomain="code_architecture_scout"
      />,
      <ArchitectureMapViewer
        key="arch"
        contextPack={TEST_CODE_PACK}
        selectedDomain="code_architecture_scout"
      />,
      <SessionBranchBar
        key="session"
        sessions={TEST_SESSIONS}
        activeSessionId="sess_01"
        onSelectSession={vi.fn()}
        onNewSession={vi.fn()}
        branches={TEST_BRANCHES}
        selectedBranchId="main"
        onSelectBranch={vi.fn()}
        commitDriftWarnings={["slicer/bounds.py modified"]}
      />,
      <SpecializationAdminView
        key="admin"
        domains={TEST_DOMAINS}
        principal={TEST_ENGINEER_PRINCIPAL}
        hardwareProfile={TEST_HEALTH_RESPONSE.hardware_profile}
        adapterPoolStatus={TEST_HEALTH_RESPONSE.adapter_pool_status}
      />,
    ];

    for (const view of views) {
      const { container, unmount } = render(view);
      expect(auditWcag22AA(container)).toEqual([]);
      unmount();
    }

    // Also audit SpecializationAdminView as compliance_officer with validation banner active
    const { container: complianceContainer, unmount: unmountCompliance } =
      render(
        <SpecializationAdminView
          domains={TEST_DOMAINS}
          principal={{
            ...TEST_ENGINEER_PRINCIPAL,
            role: "compliance_officer",
            allowed_scopes: ["public_eng", "audit:compliance"],
          }}
          hardwareProfile={TEST_HEALTH_RESPONSE.hardware_profile}
          adapterPoolStatus={TEST_HEALTH_RESPONSE.adapter_pool_status}
        />
      );
    expect(auditWcag22AA(complianceContainer)).toEqual([]);
    const targetInput = complianceContainer.querySelector(
      "#admin-purge-target-input"
    );
    expect(targetInput).not.toBeNull();
    if (targetInput) {
      fireEvent.change(targetInput, {
        target: { value: "src/secrets/leaked.py" },
      });
    }
    expect(
      complianceContainer.querySelector('[data-testid="admin-purge-validation-error"]')
    ).not.toBeNull();
    expect(auditWcag22AA(complianceContainer)).toEqual([]);
    unmountCompliance();
  });

  it("audits ScoutConsole, CitationInspector, RoleSwitcher, DomainSelector, and TokenSavingsMeter with zero WCAG 2.2 AA violations", () => {
    const components = [
      <ScoutConsole
        key="scout"
        selectedDomain="code_architecture_scout"
        principal={TEST_ENGINEER_PRINCIPAL}
        domains={TEST_DOMAINS}
        pipelineStages={TEST_PIPELINE_STAGES}
        onContextPackReady={vi.fn()}
      />,
      <CitationInspector
        key="cite"
        contextPack={TEST_CODE_PACK}
        tokenSavings={TEST_TOKEN_SAVINGS}
        principalRole="engineer"
        onOpenDiffModal={vi.fn()}
      />,
      <RoleSwitcher
        key="role"
        principal={TEST_ENGINEER_PRINCIPAL}
        onChangePrincipal={vi.fn()}
        authStatus="ready"
      />,
      <DomainSelector
        key="domain"
        domains={TEST_DOMAINS}
        selectedDomain="code_architecture_scout"
        onSelectDomain={vi.fn()}
      />,
      <TokenSavingsMeter
        key="roi"
        currentQuerySavings={TEST_TOKEN_SAVINGS}
        roiMetrics={TEST_ROI_METRICS}
      />,
    ];

    for (const component of components) {
      const { container, unmount } = render(component);
      expect(auditWcag22AA(container)).toEqual([]);
      unmount();
    }
  });

  it("verifies CSS contrast tokens in index.css meet WCAG 2.2 AA >= 4.5:1 ratio", async () => {
    const fsModuleName = "node:" + "fs";
    const pathModuleName = "node:" + "path";
    const procModuleName = "node:" + "process";
    const nodeFs = (await import(/* @vite-ignore */ fsModuleName)) as {
      readFileSync: (p: string, enc: string) => string;
    };
    const nodePath = (await import(/* @vite-ignore */ pathModuleName)) as {
      join: (...parts: string[]) => string;
    };
    const nodeProc = (await import(/* @vite-ignore */ procModuleName)) as {
      cwd: () => string;
    };
    const cssText = nodeFs.readFileSync(
      nodePath.join(nodeProc.cwd(), "src", "index.css"),
      "utf-8"
    );

    const tokenRegex = /--([a-z-]+):\s*(#[0-9a-fA-F]{6})\s*;/g;
    const tokens: Record<string, string> = {};
    let match: RegExpExecArray | null;
    while ((match = tokenRegex.exec(cssText)) !== null) {
      tokens[match[1]] = match[2];
    }

    const backgrounds = ["bg-canvas", "bg-surface", "bg-elevated"];
    const foregrounds = ["ink-primary", "ink-secondary", "ink-muted"];

    for (const bg of backgrounds) {
      expect(tokens[bg]).toBeDefined();
      for (const fg of foregrounds) {
        expect(tokens[fg]).toBeDefined();
        expect(contrastRatio(tokens[fg], tokens[bg])).toBeGreaterThanOrEqual(4.5);
      }
    }

    const signals = [
      "signal-sky",
      "signal-emerald",
      "signal-amber",
      "signal-rose",
    ];
    for (const sig of signals) {
      expect(tokens[sig]).toBeDefined();
      expect(contrastRatio(tokens[sig], tokens["bg-surface"])).toBeGreaterThanOrEqual(
        4.5
      );
    }
  });

  it("PR-4 (ECS-079/ECS-080): Tabpanels match aria-controls, DomainSelector avoids keyboard hijack, and CSV neutralizes formulas", () => {
    const { container } = render(
      <CitationInspector
        contextPack={TEST_CODE_PACK}
        tokenSavings={TEST_TOKEN_SAVINGS}
        principalRole="engineer"
        onOpenDiffModal={vi.fn()}
      />
    );
    const tabs = Array.from(container.querySelectorAll('[role="tab"]'));
    expect(tabs.length).toBe(4);
    for (const tab of tabs) {
      const ctrl = tab.getAttribute("aria-controls");
      expect(ctrl).toBeTruthy();
      if (tab.getAttribute("aria-selected") === "true" && ctrl) {
        expect(container.querySelector(`#${ctrl}[role="tabpanel"]`)).not.toBeNull();
      }
    }

    // CWE-1236 formula injection neutralization in ProductTruthView CSV export
    const formulaPack = {
      ...TEST_CODE_PACK,
      code_slices: [
        {
          ...TEST_CODE_PACK.code_slices[0],
          qualified_symbol: "=CMD|'/C calc'!A0",
          selection_reason: "+1+1@SUM(A1)",
        },
      ],
    };
    const truthRender = render(
      <ProductTruthView
        contextPack={formulaPack}
        principalRole="product_manager"
        selectedDomain={TEST_CODE_PACK.domain_id}
      />
    );
    fireEvent.click(truthRender.getByTestId("export-truth-csv-btn"));
    const preview = truthRender.getByTestId("truth-export-preview").textContent ?? "";
    expect(preview).toContain('"\'=CMD|\'/C calc\'!A0"');
    expect(preview).not.toMatch(/,"[=+\-@]/);

    // DomainSelector card wrapper must not hijack Enter inside nested buttons
    const onSelect = vi.fn();
    const selRender = render(
      <DomainSelector
        domains={TEST_DOMAINS}
        selectedDomain={TEST_DOMAINS[0].domain_id}
        onSelectDomain={onSelect}
      />
    );
    const card = selRender.getByTestId(`domain-card-${TEST_DOMAINS[1].domain_id}`);
    const selectBtn = card.querySelector("button");
    expect(selectBtn).not.toBeNull();
    if (selectBtn) {
      fireEvent.keyDown(selectBtn, { key: "Enter", code: "Enter" });
    }
    expect(onSelect).not.toHaveBeenCalled();
  });
});
