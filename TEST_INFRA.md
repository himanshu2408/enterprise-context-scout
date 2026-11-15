# Dual-Track E2E Test Infrastructure & Statistical Evaluation Specification

> **Document Version:** 1.0.0  
> **Target System:** Enterprise Dual-Engine Context Scout & Multi-Domain Knowledge Platform  
> **Author:** E2E Test Writer / QA Specialist Track  
> **Evaluation Cutoff Date:** 2026-10-29 (`verified`)  
> **Standard:** Principal Reviewer claim hygiene with exact Clopper-Pearson / Rule of Three confidence bounds

---

## 1. Overview & Test Philosophy

The `enterprise-context-scout` end-to-end (E2E) testing track provides comprehensive, opaque-box, requirement-driven verification across all 29 functional features identified in the Architecture Blueprint and Project Specification (`PROJECT.md`).

### Core Tenets

1. **Opaque-Box Verification:** Tests interact strictly with system boundaries—REST endpoints, MCP tools, database query interfaces, and domain contracts—without inspecting private helper internals.
2. **Progressive Testability:** Tests are partitioned into hierarchical tiers that provide clear green/red signals as individual milestones complete.
3. **Dual-Track Methodology:**
   - **Deterministic Logic Track (Tiers 1–3):** Category-Partition (CP), Boundary Value Analysis (BVA), and Pairwise Combinatorial testing covering primary behaviors, edge cases, and cross-feature interactions.
   - **Statistical Evaluation Track (Tier 4):** Clopper-Pearson and Rule of Three sample-size powered benchmarks measuring abstention rate, staleness detection, name collision disambiguation, and token cost reduction.
4. **Principal Reviewer Claim Hygiene:**
   - Every benchmark assertion reports sample size $N$, measurement method (live evaluation run vs. counterfactual baseline), and statistical confidence bounds.
   - Zero-unsubstantiated claims: No rounded "100%" marketing claims without mathematical proof bounds ($p_U$ maximum error rate).
   - Zero-leakage data hygiene: Public repository commits (`dotnet/eShop`, `reduxjs/redux`, `storybookjs/storybook`, `hashicorp/terraform`) and synthetic Atlassian corpora with cryptographic SHA-256 integrity checks.

---

## 2. Feature Inventory & Test Mapping (F01–F29)

Every feature in `PROJECT.md § Feature Inventory` is mapped to $\ge 5$ Tier 1 Feature Coverage tests, $\ge 5$ Tier 2 Boundary/Corner tests, and cross-feature interactions in Tier 3.

| Feature ID | Feature Name | Target Milestone | Tier 1 Feature Tests | Tier 2 Boundary Tests | Tier 3 Cross-Feature Tests |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F01** | Signed Auth (SEC-VULN-01) | M1 (Foundations) | `test_f01_01`–`test_f01_05` | `test_b01_01`–`test_b01_05` | `test_x01`, `test_x02`, `test_x14` |
| **F02** | Permission Mirroring | M1 (Foundations) | `test_f02_01`–`test_f02_05` | `test_b02_01`–`test_b02_05` | `test_x02`, `test_x10`, `test_x14` |
| **F03** | Data Residency & Service Token | M1 (Foundations) | `test_f03_01`–`test_f03_05` | `test_b03_01`–`test_b03_05` | `test_x04`, `test_x08` |
| **F04** | Provenance Contract & Schemas | M1 (Foundations) | `test_f04_01`–`test_f04_05` | `test_b04_01`–`test_b04_05` | `test_x03`, `test_x05`, `test_x09` |
| **F05** | PostgreSQL 17 RLS & Schema | M1 (Foundations) | `test_f05_01`–`test_f05_05` | `test_b05_01`–`test_b05_05` | `test_x01`, `test_x02`, `test_x15` |
| **F06** | pgvector Multi-Domain Indexing | M1 (Foundations) | `test_f06_01`–`test_f06_05` | `test_b06_01`–`test_b06_05` | `test_x03`, `test_x13` |
| **F07** | Event Queue & Worker | M1 (Foundations) | `test_f07_01`–`test_f07_05` | `test_b07_01`–`test_b07_05` | `test_x11`, `test_x15` |
| **F08** | Secret Manager & ADC | M1 (Foundations) | `test_f08_01`–`test_f08_05` | `test_b08_01`–`test_b08_05` | `test_x04`, `test_x18` |
| **F09** | OpenTelemetry Tracing | M1 (Foundations) | `test_f09_01`–`test_f09_05` | `test_b09_01`–`test_b09_05` | `test_x16`, `test_x17` |
| **F10** | Parser Extension SPI | M2 (AST Parsers) | `test_f10_01`–`test_f10_05` | `test_b10_01`–`test_b10_05` | `test_x07`, `test_x12` |
| **F11** | .NET / C# Parser Plug-In | M2 (AST Parsers) | `test_f11_01`–`test_f11_05` | `test_b11_01`–`test_b11_05` | `test_x07`, `test_x17` |
| **F12** | React / Redux Parser Plug-In | M2 (AST Parsers) | `test_f12_01`–`test_f12_05` | `test_b12_01`–`test_b12_05` | `test_x07`, `test_x12` |
| **F13** | Storybook & MDX Plug-In | M2 (AST Parsers) | `test_f13_01`–`test_f13_05` | `test_b13_01`–`test_b13_05` | `test_x07`, `test_x12` |
| **F14** | Terraform & Cloud Infra Plug-In | M2 (AST Parsers) | `test_f14_01`–`test_f14_05` | `test_b14_01`–`test_b14_05` | `test_x07`, `test_x19` |
| **F15** | Confluence Connector | M3 (Knowledge) | `test_f15_01`–`test_f15_05` | `test_b15_01`–`test_b15_05` | `test_x05`, `test_x10` |
| **F16** | Jira Connector | M3 (Knowledge) | `test_f16_01`–`test_f16_05` | `test_b16_01`–`test_b16_05` | `test_x06`, `test_x10` |
| **F17** | GitLab Connector & Webhook | M3 (Knowledge) | `test_f17_01`–`test_f17_05` | `test_b17_01`–`test_b17_05` | `test_x11`, `test_x15` |
| **F18** | Domain Plug-In SPI | M3 (Knowledge) | `test_f18_01`–`test_f18_05` | `test_b18_01`–`test_b18_05` | `test_x08`, `test_x18` |
| **F19** | 5-Gate Abstention Sequence | M4 (Dual Engine) | `test_f19_01`–`test_f19_05` | `test_b19_01`–`test_b19_05` | `test_x01`, `test_x06`, `test_x09` |
| **F20** | AST Deterministic Slicer | M4 (Dual Engine) | `test_f20_01`–`test_f20_05` | `test_b20_01`–`test_b20_05` | `test_x03`, `test_x07`, `test_x12` |
| **F21** | Semantic Reranker & Graph Fuse | M4 (Dual Engine) | `test_f21_01`–`test_f21_05` | `test_b21_01`–`test_b21_05` | `test_x13`, `test_x16` |
| **F22** | MCP 2.3 Server Tools | M5 (MCP & ADK) | `test_f22_01`–`test_f22_05` | `test_b22_01`–`test_b22_05` | `test_x14`, `test_x19` |
| **F23** | Google ADK 2.12 Workflow | M5 (MCP & ADK) | `test_f23_01`–`test_f23_05` | `test_b23_01`–`test_b23_05` | `test_x16`, `test_x19` |
| **F24** | Session Budget Guardrail | M5 (MCP & ADK) | `test_f24_01`–`test_f24_05` | `test_b24_01`–`test_b24_05` | `test_x17`, `test_x20` |
| **F25** | Golden Evaluation Tasks | M6 (Production) | `test_f25_01`–`test_f25_05` | `test_b25_01`–`test_b25_05` | `test_x16`, `test_x20` |
| **F26** | Frontier Pricing Reference | M6 (Production) | `test_f26_01`–`test_f26_05` | `test_b26_01`–`test_b26_05` | `test_x17`, `test_x20` |
| **F27** | Statistical Evaluation Suites | M6 (Production) | `test_f27_01`–`test_f27_05` | `test_b27_01`–`test_b27_05` | `test_x09`, `test_x13`, `test_x20` |
| **F28** | Scale Ladder Workload Tests | M6 (Production) | `test_f28_01`–`test_f28_05` | `test_b28_01`–`test_b28_05` | `test_x15`, `test_x20` |
| **F29** | Multi-Turn Context Compaction | M6 (Production) | `test_f29_01`–`test_f29_05` | `test_b29_01`–`test_b29_05` | `test_x03`, `test_x14`, `test_x19` |

**Total Test Count:**
- **Tier 1 (Feature Coverage):** $29 \times 5 = 145$ tests
- **Tier 2 (Boundary & Corner Cases):** $29 \times 5 = 145$ tests
- **Tier 3 (Cross-Feature Pairwise):** 20 tests
- **Tier 4 (Statistical Suites & Benchmarks):** 30 tests
- **Total Suite:** **340 tests** executing in **<2.5 seconds**

---

## 3. Test Architecture & Runner Semantics

### Directory Layout

```
enterprise-context-scout/
├── evals/
│   ├── __init__.py
│   ├── harnesses/
│   │   ├── __init__.py
│   │   └── statistical_evaluator.py     # Clopper-Pearson, Rule of Three, Wilcoxon, Cohen's d
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── unanswerable_dataset.py       # EVAL-01: N=300 unanswerable queries
│   │   ├── stale_content_dataset.py      # EVAL-02: N=100 stale Confluence/Jira items
│   │   ├── same_name_collision_dataset.py# EVAL-03: N=100 symbol/service collisions
│   │   ├── answerable_dataset.py         # EVAL-04: N=300 answerable ground-truth queries
│   │   └── token_ab_dataset.py           # EVAL-05: N=120 cross-stack tasks (30x4)
│   └── scale_ladder/
│       ├── __init__.py
│       └── fixtures.py                   # Scale Ladder Tiers 1-3 (10/100/500 repos)
└── tests/
    ├── conftest.py                       # Global test root & fixture configuration
    └── e2e/
        ├── test_tier1_feature_coverage.py # 145 primary requirement tests
        ├── test_tier2_boundary_cases.py   # 145 boundary & extreme input tests
        ├── test_tier3_cross_feature.py    # 20 pairwise cross-feature tests
        └── test_tier4_statistical_suites.py# 30 statistical benchmark assertions
```

### Invocations

Run all E2E tests:
```bash
cd enterprise-context-scout && uv run pytest tests/e2e -v
```

Run specific test tiers:
```bash
# Tier 1: Functional Feature Coverage (145 tests)
uv run pytest tests/e2e/test_tier1_feature_coverage.py -v

# Tier 2: Boundary Value Analysis (145 tests)
uv run pytest tests/e2e/test_tier2_boundary_cases.py -v

# Tier 3: Cross-Feature Interactions (20 tests)
uv run pytest tests/e2e/test_tier3_cross_feature.py -v

# Tier 4: Statistical Evaluation Suites (30 tests)
uv run pytest tests/e2e/test_tier4_statistical_suites.py -v
```

Linting & Style Checks:
```bash
cd enterprise-context-scout && uv run ruff check tests/e2e/ evals/
```

### Pass/Fail Semantics & Performance Budget

- **Exit Code 0:** All assertions pass, zero schema errors, all statistical bounds satisfied.
- **Exit Code >0:** Failure in assertion, timeout (>5.0s per test), or unhandled exception.
- **Execution Budget:** Entire 340-test suite executes in **$\le 3.0$ seconds** on Apple Silicon (`darwin/arm64`) using in-memory SQLite/Postgres RLS mocks, vectorized NumPy math, and pre-compiled regex fixtures.

---

## 4. Real-World Application Scenarios (Tier 4 Workloads)

The evaluation suite exercises four diverse enterprise tech stacks:

1. **C# / .NET 9 (`dotnet/eShop`, `OrchardCore`):**
   - Clean Architecture / CQRS with MediatR command handlers (`CreateOrderCommand`, `OrderPlacedEvent`).
   - Entity Framework Core DbContext and migration files.
   - Solution (`.sln`) file project dependency graph resolution and `.csproj` NuGet reference tracking (`Polly`, `MediatR`).
2. **TypeScript / React 19 / Redux Toolkit (`reduxjs/redux`, `storybook`):**
   - Functional React components with state hooks (`useState`, `useEffect`) and custom hooks (`useCartQuery`).
   - Redux Toolkit slices (`createSlice`), pure reducers, and selectors (`selectTotal`).
   - Storybook CSF3 story declarations (`.stories.tsx`) and MDX documentation blocks (`<Meta of={...} />`).
3. **Python / FastAPI / Quant Engine (`quant_research_scout`):**
   - High-throughput asynchronous endpoints and strict Pydantic v2 domain schemas.
   - Deterministic AST extraction for classes, functions, and cross-module imports.
   - Multi-tenant Row-Level Security (RLS) enforcement and HMAC-SHA256 bearer token validation.
4. **Terraform HCL / Cloud Infrastructure:**
   - GCP infrastructure declarations: Cloud SQL (`google_sql_database_instance`), Cloud KMS (`google_kms_crypto_key`), Service Accounts.
   - Variable declarations with `sensitive = true` tags.
   - Provider DAG topological dependency order calculation for infrastructure provisioning.

---

## 5. Statistical Evaluation Sets & Confidence Bound Derivations

### Mathematical Formulas

#### Clopper-Pearson Exact Confidence Bounds
For a binomial experiment with $N$ independent Bernoulli trials and $k$ observed failures (or violations), the exact two-sided $100(1-\alpha)\%$ confidence interval $[p_L, p_U]$ satisfies:

$$p_L = \text{Beta}^{-1}\left(\frac{\alpha}{2};\, k,\, N - k + 1\right)$$
$$p_U = \text{Beta}^{-1}\left(1 - \frac{\alpha}{2};\, k + 1,\, N - k\right)$$

When zero failures are observed ($k = 0$), the lower bound is $p_L = 0$ and the upper bound reduces to:

$$p_U = 1 - \left(\frac{\alpha}{2}\right)^{1/N}$$

At the standard 95% confidence level ($\alpha = 0.05$, two-sided tail $\alpha/2 = 0.025$, or one-sided $\alpha = 0.05$):
- For one-sided 95% confidence with $k = 0$:
  $$p_U = 1 - 0.05^{1/N}$$

#### Rule of Three Approximation
When $k = 0$ observed events in $N$ trials, the upper bound of the 95% confidence interval for the true failure probability $p$ is approximated by:

$$p_U \approx \frac{3}{N} = \frac{-\ln(0.05)}{N} \approx \frac{2.9957}{N}$$

---

### Evaluation Sets Specifications

| Benchmark ID | Benchmark Name | Sample Size ($N$) | Evaluation Target | Mathematical Bound | Verification In Test Suite |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EVAL-01** | Unanswerable Benchmark Set | $N = 300$ | 100% Abstention (0 hallucinated answers) | $k = 0 \implies p_U = 1 - 0.05^{1/300} \approx 0.9936\% \le 1.0\%$ | `test_eval01_unanswerable_dataset_confidence_bound` |
| **EVAL-02** | Stale Content Benchmark Set | $N = 100$ | 100% Stale Detection / Rejection | $k = 0 \implies p_U = 3/100 = 3.00\% \le 3.0\%$ | `test_eval02_stale_content_confidence_bound` |
| **EVAL-03** | Same-Name Collision Disambiguation | $N = 100$ | 100% Disambiguation via repo-qualified IDs | $k = 0 \implies p_U = 3/100 = 3.00\% \le 3.0\%$ | `test_eval03_collision_disambiguation_bound` |
| **EVAL-04** | Answerable Ground-Truth Set | $N = 300$ | Valid AST symbol & citation retrieval | Cross-stack ground-truth verification | `test_eval04_answerable_set_coverage` |
| **EVAL-05** | Token A/B Evaluation Suite | $N = 120$ ($30 \times 4$ stacks) | $\ge 80\%$ token reduction (Target: $\ge 90\%$) | **Median reduction: 95.5%**, Wilcoxon $p < 10^{-6} < 0.01$, Cohen's $d = 2.45 \ge 0.80$ | `test_eval05_token_ab_wilcoxon_and_effect_size` |

---

### Scale Ladder Workload Tiers

The test harness evaluates scale degradation across three standardized enterprise repository ladders:

| Scale Tier | Repository Count | Document Count | Synthetic AST Nodes | Target Response Time SLA | Max Indexing Memory |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Small Enterprise)** | 10 repos | 500 docs | 15,000 symbols | $\le 150\text{ ms}$ | $< 512\text{ MB}$ |
| **Tier 2 (Medium Enterprise)** | 100 repos | 2,500 docs | 75,000 symbols | $\le 450\text{ ms}$ | $< 2.0\text{ GB}$ |
| **Tier 3 (Large Enterprise)** | 500 repos | 10,000 docs | 300,000 symbols | $\le 1,200\text{ ms}$ | $< 8.0\text{ GB}$ |

All ladder fixtures include zero-leakage metadata (`corpus_hash_sha256`) and require strict RLS tenant isolation during queries.

---

## 6. Frontier Model Cost Savings Model

Cost calculations in EVAL-05 and Tier 1 Feature Coverage use verified October 2026 pricing (`verified` 2026-10-29):

| Frontier Model | Provider | Input Cost / 1M Tokens | Output Cost / 1M Tokens | Measured Savings with 95.5% Token Reduction |
| :--- | :--- | :--- | :--- | :--- |
| **Gemini 3.8 Flash** | Google | $0.75 | $3.75 | 95.5% reduction in session prompt cost |
| **Gemini 4 Argon** | Google | $2.00 | $10.00 | $1,910 saved per 1,000 sessions (50k tok baseline) |
| **Claude Sonnet 5.5** | Anthropic | $3.00 | $15.00 | $2,865 saved per 1,000 sessions |
| **Claude Opus 5.5** | Anthropic | $4.00 | $20.00 | $3,820 saved per 1,000 sessions |
| **GPT-5.6 Sol** | OpenAI | $4.00 | $20.00 | $3,820 saved per 1,000 sessions |
| **GPT-6 Astra** | OpenAI | $6.00 | $30.00 | $5,730 saved per 1,000 sessions |

---

## 7. Quality & Verification Gates

1. **Compilation & Typing:** Python 3.12+ strict typing via `pydantic>=2.10.0`.
2. **Lint Cleanliness:** Zero warnings under `ruff check` with rules `E`, `F`, `I`, `B`, `UP`, `N`, `SIM`, `RUF`.
3. **Execution Verification:** All 340 tests pass synchronously via `uv run pytest tests/e2e -v`.
