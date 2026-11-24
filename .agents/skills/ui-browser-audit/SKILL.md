---
name: ui-browser-audit
description: Audits the React Workbench (web-ui) via Antigravity Browser Subagent for zero external network egress, CSP compliance, WCAG 2.2 AA accessibility, and dual-persona flows. Use after UI changes or during M5/M6 verification.
---
# Air-Gapped React Workbench Browser Audit (`/ui-browser-audit`)

1. Open `http://127.0.0.1:5173` in `/browser` and check DevTools Network & Console tabs:
   assert `0` external requests (`fonts.googleapis.com` == 0) and `0` CSP/Console errors.
2. Exercise Engineer (`FrontierHandoffModal`), PM/Auditor (`ProductTruthView` +
   `ArchitectureMapViewer`), and Admin (`SpecializationAdminView`) workflows.
