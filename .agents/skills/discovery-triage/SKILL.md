---
name: discovery-triage
description: Triages newly uncovered bugs, security flaws, race conditions, or architectural mismatches without scope creep. Use whenever an unexpected defect is spotted during implementation or review.
---
# Open Issue Discovery & Self-Healing Triage (`/discovery-triage`)

1. **Architecture & Security Baseline:** Cross-check findings against
   `ARCHITECTURE_BLUEPRINT.md`, `docs/SECURITY_AND_GOVERNANCE.md`, and
   `reports/v1_ga_release_scorecard.md`.
2. **SEV-0 (In-scope blocker/bug in files you are touching):** Write a failing regression
   test first, fix the root cause cleanly, verify via `/gate-verify`, and summarize the
   fix and test evidence in the PR description.
3. **SEV-1 (Out-of-scope file or separate subsystem):** Document the finding with
   `filepath:line` and reproduction steps in the issue tracker or handoff summary; do NOT
   edit out-of-scope files during parallel runs.
4. **SEV-2 (Empirical contradiction):** Pause and present a 6-line A/B trade-off summary
   with measured evidence before modifying pinned contracts or base images.
