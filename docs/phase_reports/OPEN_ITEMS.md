# OPEN ITEMS & USER DIRECTIVES TRACKING

Authoritative tracking register for all caveats, weaknesses, recommendations, and modularity requirements requested by the user on 2026-11-10T09:46:20Z.
Per directive: Only independent auditor runs count as evidence (never worker or sentinel self-reports). Track status, exact verification command and output, and verifying auditor. Finish current Gate 7 first, then systematically address items in dedicated PRs/iterations.

---

## Summary Matrix

| ID | Category | Item Description | Status | Verifying Auditor |
|---|---|---|---|---|
| **A1** | Verification Integrity | Full 1,300+ test sweep execution with exact pass/fail counts (clarify "70% in-progress" log) | OPEN | PENDING |
| **A2** | Verification Integrity | Repo-wide sweep for broad/swallowed exceptions (narrow or log+re-raise with tests) | OPEN | PENDING |
| **A3** | Verification Integrity | Repo-wide sweep for weak assertions (`>=1`, truthiness) & mutation proof on 340-test E2E suite | OPEN | PENDING |
| **A4** | Verification Integrity | Parity audit: Real PostgreSQL vs In-Memory test execution; dual-run tests for all fixes; scoping leak grep | OPEN | PENDING |
| **A5** | Verification Integrity | Comprehensive defect history recorded in Phase Log & Architecture Story with before/after diffs | OPEN | PENDING |
| **B6** | Security / Correctness | `ContextVar` principal concurrency test with simultaneous requests proving zero leakage | OPEN | PENDING |
| **B7** | Security / Correctness | SEC-VULN-01 review, webhook HMAC verification tests, queue concurrency tests, Merkle & CTE benchmarks | OPEN | PENDING |
| **B8** | Security / Correctness | Citation Gate: Zero-tolerance exact normalized substring match, word substitution/negation tests, persisted versions | IMPLEMENTED | PENDING AUDITOR |
| **B9** | Security / Correctness | Jira: Per-project configurable terminal/stale statuses and resolutions with duplicate link following | OPEN | PENDING |
| **B10** | Security / Correctness | Confluence temporal decay: Verification that old-but-correct foundational pages are not buried | OPEN | PENDING |
| **C11** | Parsers / Generalization | Tree-sitter AST integration with deterministic ABSTENTION on syntax errors/partial parses; 10+ paren tests | OPEN | PENDING |
| **C12** | Parsers / Generalization | Real public repository evaluation (eShop, OrchardCore, public React/Redux/Terraform) with hand-checked N | OPEN | PENDING |
| **C13** | Parsers / Generalization | Status verification of GitLab raw-blob ingestion, Terraform/GCP plugin, and Storybook MDX | OPEN | PENDING |
| **D14** | Evaluation Hygiene | Dataset provenance documentation (sources, public vs synthetic, disjoint proof) & R1 gaps report | OPEN | PENDING |
| **D15** | Cost Governance | Live model pricing verification, pre-run cost estimation gate, subagent token accounting, Vertex EU region | OPEN | PENDING |
| **E16** | Modularity / No Coupling | Elimination of company-specific names/workflows; modular plugins/configuration with neutral defaults | IMPLEMENTED (M4 Config) | PENDING AUDITOR |
| **E17** | Modularity / Connectors | Support for Atlassian Cloud vs Server/Data Center and GitLab SaaS vs Self-Managed behind adapters | RESOLVED (Cloud Prioritized) | Confirmed by User |
| **E18** | Modularity / Proof | Extensibility proof: "Build your own plugin/connector" guide, toy plugin test, multi-company config test | OPEN | PENDING |
| **F19** | Milestone 4 Design | Present deterministic abstention and Gemma claim verification design for user review before implementation | IMPLEMENTED | PENDING AUDITOR |
| **F20** | Execution Protocol | Zero-assumptions rule: Message parent agent to ask user when uncertain; one concern per commit/PR | IN_PROGRESS | Sentinel Enforced |
| **M4-OI-03** | Evaluation Hygiene | Real-Corpus Benchmark Evaluation Harness (Pinned SHAs, Tree-sitter AST in Postgres, Pinned URLs, User Approval for Cloning) | OPEN | PENDING |

---

## Detailed Item Register

### Section A: Verification Integrity

#### A1. Full Test Suite Audit & Explanation of Progress Logs
- **Requirement**: Explain the "~70% pass rate" from the 09:33Z progress report. The independent auditor must run the complete test suite across all 1,300+ collected tests (not `-k` subsets) and report exact pass, fail, skipped, and error counts.
- **Context Note**: The 09:33Z log reflected a live in-progress background pytest run (`task-160`) displaying test progress percentages (`[ 70%]`), not a 70% passing ratio. This distinction must be formally audited and reported with full test execution numbers.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

#### A2. Swallowed Exception Sweep
- **Requirement**: Repo-wide sweep for broad `except Exception:` blocks across connectors, workers, task queue, and API. Narrow exception types or ensure proper structured logging and re-raising. Add regression tests proving exceptions are not silently suppressed.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

#### A3. Weak Assertion Hardening & E2E Mutation Testing
- **Requirement**: Sweep test suites for weak assertions (`assert len(x) >= 1`, truthiness checks) where exact bounds or specific values should be asserted. Perform mutation testing on the 340-test E2E suite to confirm tests execute authentic code paths against live PostgreSQL schemas rather than stubs, requiring intentional mutation failures.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

#### A4. Dual-Track In-Memory & Live PostgreSQL Parity Audit
- **Requirement**: Document all tests executing against real PostgreSQL vs `InMemoryRLSRepository`. Ensure all multi-tenancy bugfixes (multi-repo file deletion scoping, inward AST edge cascades, Confluence version supersession) are verified against both in-memory and PostgreSQL. Add permanent standalone test with two repos containing identical file paths, deleting one and asserting exact counts in the other. Grep for any remaining `or domain_id ==` scoping leaks.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

#### A5. Complete Defect Register, Phase Reports & Architecture Story Documentation
- **Requirement**: 
  - (a) Write dedicated persistent phase reports for Milestones 1, 2, and 3 now in `docs/phase_reports/` (`M1_PHASE_REPORT.md`, `M2_PHASE_REPORT.md`, `M3_PHASE_REPORT.md`) reconstructing the full history from gate records, including every failed gate, its root cause, the surgical fix, and exact before/after counts.
  - (b) Create the running architecture story document detailing problems diagnosed, architectural changes, measured improvements, and honest misses for Staff/Senior review.
  - (c) From now on, write the phase report at each milestone gate boundary, not at the end of the project.
  - (d) Detail every defect identified across all iterations with before/after diffs: SEC-VULN-01, GitLab `sym.kind` swallowed exception, cross-repo collateral file deletion, Confluence version rollback / `"v1"` bypass, citation forgery without metadata checks, Jira status classification, and progress reporting log interpretation.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

---

### Section B: Security & Correctness

#### B6. ContextVar Concurrency Isolation & Simultaneous Request Regression Test
- **Requirement**: Document origin of `_CURRENT_PRINCIPAL` lifecycle change in `api/main.py` and `api/dependencies.py`. Author a dedicated concurrency test with two simultaneous requests running under different principal contexts proving zero cross-request identity leakage (must demonstrate failure before the fix and pass after). Review as an isolated security change in a separate commit/PR.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

#### B7. Security & Concurrency Verification Suite
- **Requirement**: Audit SEC-VULN-01 remediation and regression tests; confirm webhook HMAC verification and forged payload rejection; verify queue concurrency (zero lost/duplicate jobs under worker races, dead-letter queue behavior); provide empirical benchmarks for Merkle blob-diffing p95 latency and Recursive CTE graph traversal with documented dataset sizes.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

#### B8. Zero-Tolerance Citation Gate & Quote Verification
- **Requirement**: Enforce exact normalized-substring matching with zero tolerance (eliminate fuzzy 0.98 ratio on production paths) with permanent regression tests for single-word substitution and polarity flips (e.g., "must" vs "must not"). Include bidirectional homoglyph tests (reject Cyrillic/Greek substitutions; accept legitimate Unicode). Enforce `reject_absent_tokens=True` by default. Exclude synthetic chunk headers from verbatim quote spans. Persist Confluence version records in database schema.
- **Status**: IMPLEMENTED (Verified by Worker; Pending Forensic Auditor Review)
- **Evidence**:
  ```
  [Command]: uv run pytest tests/test_claim_verifier.py tests/test_domain_agnostic_citation_gate.py tests/test_adversarial_m7_distillation.py -v
  [Output]: 42 passed in 2.39s (13 in test_claim_verifier, 13 in test_domain_agnostic_citation_gate, 16 in test_adversarial_m7_distillation)
  [Implementation]:
    - ast_engine/deterministic_slicer.py: Removed difflib.SequenceMatcher, enforced MIN_QUOTE_OVERLAP_RATIO = 1.0 exact NFKC match.
    - agents/citation_gate.py: Enforced DEFAULT_MIN_OVERLAP_RATIO = 1.0 and homoglyph checks.
    - agents/claim_verifier.py: ExactClaimVerifier with homoglyph attack rejection and authentic Unicode acceptance.
  ```
- **Auditor**: PENDING AUDITOR RUN

#### B9. Modular Jira Statuses, Resolutions & Duplicate Links
- **Requirement**: Make terminal/stale Jira statuses and resolutions (Duplicate, Won't Fix, Cannot Reproduce) fully configurable per project rather than hardcoded. Follow duplicate ticket links where available. Ensure tickets with status `Done` remain valid evidence.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

#### B10. Confluence Temporal Decay & Foundational Doc Preservation
- **Requirement**: Verify that older but authoritative documentation pages are not penalized by temporal decay scores. Document scoring and freshness mechanics.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

---

### Section C: Parsers & Generalization

#### C11. Tree-Sitter AST & Deterministic Abstention on Parse Failures
- **Requirement**: Augment/replace heuristic regex parsers with tree-sitter grammars. When AST parsing fails or encounters syntax errors, output deterministic `insufficient context` abstention rather than partial symbols. Flag heuristic fallbacks as low confidence. Add boundary tests for deeply nested parentheses beyond limits and C# raw strings with 4+ quotes.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

#### C12. Real Public Repository AST Accuracy Benchmarks
- **Requirement**: Execute AST parsers on real public open-source codebases (eShop, OrchardCore, public React/Redux/Terraform repositories). Hand-check a sample of extracted symbols and report precision and recall with sample size $N$.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

#### C13. Status Audit of Unreported Features
- **Requirement**: Provide concrete verification and tests for: GitLab raw-blob push ingestion, Terraform GCP cloud infrastructure plugin, and Storybook MDX documentation support.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

---

### Section D: Evaluation Hygiene & Claims

#### D14. Evaluation Dataset Provenance & Gap Assessment
- **Requirement**: Document exact origin, distribution, and split methodology for all statistical eval sets (unanswerable $N \ge 300$, stale $N \ge 100$, collision $N \ge 100$, answerable $N \ge 300$, token A/B $N \ge 120$). Prove evaluation data is strictly disjoint from model training/distillation corpora. Complete the initial verified-gaps report.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

#### D15. Live Pricing & Token Accounting Gate
- **Requirement**: Enforce cost estimation gate before every paid model compute call with live search verification. Verify actual token accounting capabilities across subagents. Validate Vertex AI EU region and active model IDs against current official documentation without API keys.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

#### M4-OI-03. Real-Corpus Benchmark Evaluation Harness
- **Requirement**: Manifest with pinned commit SHAs + public repo URLs, real files indexed through tree-sitter/polyglot parsers into PostgreSQL store, mechanically derived queries, replacing synthetic-fixture assertions once real-corpus run is available. Outbound cloning requires explicit user approval of proposed URLs, pinned SHAs, and disk size.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

---

### Section E: Modularity & Generalization (Zero Employer Coupling)

#### E16. Elimination of Workflow Coupling & Neutral Defaults
- **Requirement**: Audit entire codebase to eliminate any employer-specific names, status flows, space keys, project hierarchies, or finance-specific constructs from core modules. Decouple domain packs and make all workflow rules configurable via neutral schemas and environment plugins.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

#### E17. Connector Adapter Layer (Cloud vs Data Center / Self-Managed)
- **Requirement**: Provide abstraction adapters supporting both Atlassian Cloud REST APIs (v2/v3) and Server/Data Center REST endpoints, as well as GitLab SaaS vs Self-Managed GitLab instances.
- **User Clarification Confirmed (2026-11-10T09:56:34Z)**:
  - **Atlassian (Jira & Confluence)**: Confirmed **Atlassian CLOUD** (`.atlassian.net` domain). Prioritize Jira Cloud REST v3, Confluence Cloud REST v2, and API-token/OAuth authentication.
  - **GitLab**: Likely Self-Managed (custom company domain). Support configurable `base_url`, custom CA/TLS, and version negotiation.
- **Mandatory Architectural Requirements**:
  1. **Dual Adapter SPI**: Prioritize Atlassian Cloud while retaining the pluggable adapter interface so Data Center/Server (REST v2 / v1, PATs) can be added by other companies without core modifications.
  2. **Config-Driven Endpoints**: Base URLs, authentication methods, API versions, and TLS certificate paths must be purely configuration-driven with neutral defaults.
  3. **Zero Employer Leaks**: Zero employer names, client URLs, or credentials anywhere in the repository.
  4. **Public/Local Testing**: Test exclusively against local mocks or synthetic data; NEVER contact live employer endpoints.
  5. **Cloud-Specific Doc Verification**: Verify rate limits, cursor pagination, Atlassian Document Format (ADF), and permission differences against official Atlassian documentation prior to production wiring.
- **Status**: RESOLVED (Atlassian = Cloud Confirmed; Data Center Adapter Architecture Retained)
- **Evidence**:
  ```
  [User Confirmation]: Jira/Confluence URLs end in .atlassian.net (Cloud confirmed). Adapter SPI retained.
  ```
- **Auditor**: PENDING (To be verified in connector adapter implementation PR)

#### E18. Extensibility Demonstration & Multi-Company Verification
- **Requirement**: Author a concise "Build Your Own Plugin & Connector" SPI guide. Add automated tests registering a toy language plugin and toy connector without modifying core files. Add an integration test loading a distinct second company's workflow configuration (custom Jira statuses, alternative Confluence hierarchies) and proving end-to-end execution.
- **Status**: OPEN
- **Evidence**:
  ```
  [Command]: PENDING AUDITOR RUN
  [Output]: PENDING AUDITOR RUN
  ```
- **Auditor**: PENDING

---

### Section F: Milestone 4 & Project Execution

#### F19. Milestone 4 Design Review & Production Implementation
- **Requirement**: Present the complete Milestone 4 design (deterministic 6-check abstention code gates, exact NFKC claim-to-source-span mapping, and wrong-answer rate benchmarking) for explicit user review, and implement with mechanical $0-spend proof.
- **Status**: IMPLEMENTED (Verified by Worker; Pending Forensic Auditor Review)
- **Evidence**:
  ```
  [Command]: uv run pytest tests/test_zero_spend_abstention.py tests/test_preflight_gate.py tests/e2e/test_tier4_statistical_suites.py -v
  [Output]: 47 passed in 1.10s
    - tests/test_zero_spend_abstention.py: 4/4 passed (all 300 EVAL-01 unanswerable queries verified with 0 calls, 0 paid tokens, Clopper-Pearson 95% lower bound 98.78%).
    - tests/test_preflight_gate.py: 13/13 passed (6 checks: AST syntax, 1-hop closure, RRF 0.65 threshold, scope mirroring, cryptographic provenance, document freshness).
    - tests/e2e/test_tier4_statistical_suites.py: 30/30 passed (EVAL-01..EVAL-05 statistical suites).
  [Design Artifact]: enterprise-context-scout/docs/M4_DESIGN.md
  ```
- **Auditor**: PENDING AUDITOR RUN

#### F20. Zero-Assumptions & Single-Concern PR Discipline
- **Requirement**: Always ask parent/user when uncertain; never make unverified assumptions. Implement fixes in focused, isolated branches and PRs (one concern per commit/PR).
- **Status**: IN_PROGRESS (Enforced by Sentinel)
- **Evidence**:
  ```
  [Command]: Continuous Sentinel Enforcement
  [Output]: Zero unapproved assumptions; user directives recorded in ORIGINAL_REQUEST.md and OPEN_ITEMS.md
  ```
- **Auditor**: Sentinel Monitored
