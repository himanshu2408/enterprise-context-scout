# Dual-Track E2E Test Infrastructure & Statistical Evaluation Specification (`v1.0.0`)

> **Document Version:** `1.0.0`  
> **Target System:** Enterprise Context Scout (`v1.0.0`)  
> **Last Updated:** 2026-11-24  
> **Standard:** Zero-skip verification (`0 failed, 0 skipped`) with exact Wilson / Clopper-Pearson confidence bounds and zero synthetic-as-production misrepresentation

---

## 1. Overview & Verification Philosophy

Enterprise Context Scout enforces a zero-skip, multi-layer verification pyramid spanning static analysis, live PostgreSQL 17 `FORCE ROW LEVEL SECURITY` integration tests, `OWASP ASI01`–`ASI10` adversarial suites, Schemathesis OpenAPI fuzzing, AST mutation testing, air-gapped UI WCAG 2.2 AA audits, and statistical retrieval benchmarks.

### Core Tenets

1. **Zero-Skip & Zero-Fake-Fallback Integrity:** Tests never bypass PostgreSQL RLS via mocks in live database gates, never skip suites via `-k "not ..."`, and never weaken assertions to turn CI green.
2. **Strict Tenant Isolation in Shared-State Tests:** Every database integration test provisions unique per-test `tenant_id` UUIDs (never mutating `DEFAULT_TENANT_ID`) and runs canary checks confirming cleanup leaves unrelated rows untouched.
3. **Dual-Track Methodology:**
   - **Deterministic Engineering Track:** `84` Python `pytest` modules (`1045` tests) + `10` TypeScript `vitest` suites (`75` tests) covering `23/23` PG17 `FORCE RLS` tables (`0001`–`0023`), 12-language AST/SCIP parsers, transitive connector ACLs, `<60s` artifact purge, 3-graph ADK orchestration, BYO-LLM/multi-LoRA safety, 60-case headless backend & Stateless HTTP MCP middleware authenticity, `OWASP ASI01`–`ASI10`, Schemathesis OpenAPI fuzzing, and PITR DR.
   - **Statistical Evaluation Track:** Exact 95% Wilson and Clopper-Pearson confidence bounds across retrieval recall, citation faithfulness, 10K-page wiki freshness/conflict resolution, token economics, and mutation kill rates.
4. **Claim Hygiene:** Every reported rate includes raw counts ($k / N$), confidence interval, execution environment, and explicit disclosure of synthetic vs. real public corpora.

---

## 2. Verification Pyramid & Suite Layout (`v1.0.0`)

```text
enterprise-context-scout/
├── .agents/skills/
│   ├── gate-verify/scripts/run_compact_gates.sh   # Full zero-skip gate runner (Python + Squawk + web-ui)
│   ├── rls-pgtap-audit/scripts/check_rls_catalog.py # Live PG17 23/23 FORCE RLS & NOBYPASSRLS catalog audit
│   └── byo-lora-dryrun/scripts/check_splits_and_ceiling.py # 3-way SHA-256 split & rsLoRA policy audit
├── evals/
│   ├── harnesses/statistical_evaluator.py         # Clopper-Pearson, Rule of Three, Wilcoxon, Cohen's d
│   ├── datasets/                                  # EVAL-01..EVAL-05 (unanswerable, stale, collision, answerable, BPE token A/B)
│   └── scale_ladder/fixtures.py                   # Scale Ladder Tiers 1–3 (10 / 100 / 500 repos)
├── tests/                                         # 84 Python test modules (1045 tests)
│   ├── test_backend_middleware_authenticity.py    # 60-case 5-category backend & Stateless HTTP MCP middleware suite
│   ├── test_m1_alembic_rls_pools.py               # Alembic 0001–0023, Squawk PG17, 23/23 FORCE RLS
│   ├── test_postgres_rls_live.py                  # Live PG17 cross-tenant isolation & HNSW iterative scan
│   ├── test_m2_01..04_*.py                        # Zoekt CoW overlays, 12-lang AST/SCIP, connectors, <60s purge
│   ├── test_m3_01_02_reranker_and_three_graph_adk.py # Cross-encoder reranker & 3-graph ADK orchestrator
│   ├── test_m4_01_vllm_sidecar_and_asi05.py       # BYO vLLM, AdapterSyncSidecar, OWASP ASI05 zero host RCE
│   ├── test_m4_02_specialization_distiller_peft.py # 5-archetype specialization, rsLoRA, 0/10 CanaryGate
│   ├── test_m5_01_02_zero_trust_iap_oidc_scim_audit_mcp.py # Cloud IAP/OIDC/SCIM 2.0, WORM audit, stateless MCP
│   ├── test_owasp_asi01_to_asi10_matrix.py        # Unified OWASP ASI01–ASI10 adversarial security matrix
│   ├── test_m2_schemathesis_openapi_fuzz.py       # Schemathesis OpenAPI fuzzing (31 ops / 62 subtests)
│   ├── test_l9_mixed_tenant_50_sse_mcp_concurrency.py # 50 concurrent mixed-tenant SSE + MCP sessions
│   ├── test_m6_01_containers_helm_tf_aibom.py     # Containers, Helm, Terraform multi-cloud parity, AI-BOM
│   ├── test_m6_02_pitr_reconciler_and_chaos_dr.py # Redis quota failover, vLLM outage fallback, PITR DR
│   └── test_release_readiness_and_seed_demo.py    # 1.0.0 version alignment, seed-demo, tracked-file hygiene
└── web-ui/src/__tests__/                          # 10 Vitest suites (75 tests, incl. WCAG 2.2 AA axe audit)
```

---

## 3. Primary Verification Commands

```bash
# 1. Full zero-skip verification gate (ruff, mypy --strict, Squawk PG17, all pytest, web-ui lint/tsc/vitest/build)
bash .agents/skills/gate-verify/scripts/run_compact_gates.sh --scope all

# 2. Headless backend & Stateless HTTP MCP middleware authenticity suite (60 cases, 5 categories)
make verify-backend
uv run context-scout verify-backend --postgres --live-mlx
uv run pytest tests/test_backend_middleware_authenticity.py -q

# 3. Kernel PG17 23/23 FORCE RLS catalog & Alembic Squawk checks
uv run python .agents/skills/rls-pgtap-audit/scripts/check_rls_catalog.py
uv run python -m context_scout.db.migrations_runner --lint-squawk

# 4. BYO-LoRA 3-way split disjointness & OWASP ASI06 scope ceiling audit
uv run python .agents/skills/byo-lora-dryrun/scripts/check_splits_and_ceiling.py

# 5. Live PostgreSQL 17 evaluation harness & AST mutation gate (>= 85% kill rate)
uv run python -m context_scout.evals.runner --postgres --fail-under-threshold
uv run python -c "from context_scout.security.mutation_gate import ASTMutationGateRunner; s = ASTMutationGateRunner(min_kill_rate=0.85).run_all(); assert s.passed; print(s.overall_kill_rate)"

# 6. Air-gapped React Workbench quality & WCAG 2.2 AA suite
npm --prefix web-ui run lint && npm --prefix web-ui run typecheck && npm --prefix web-ui run test && npm --prefix web-ui run build
```

---

## 4. Polyglot Workload Stacks Covered in Tier 4 & Integration Suites

1. **C# / .NET (`eShop`, `OrchardCore` patterns):** MediatR CQRS handlers, EF Core `DbContext` migrations, `.sln`/`.csproj` project graphs.
2. **TypeScript / React 19 / Redux Toolkit / Storybook:** Hooks, `createSlice` reducers/selectors, CSF3 `.stories.tsx`, MDX documentation blocks.
3. **Python / FastAPI / Multi-Tenant Platform Services (`code_architecture_scout`):** Async FastAPI routes, Pydantic v2 contracts, PostgreSQL 17 `FORCE RLS`, cross-service API/SQL contract edges.
4. **Terraform HCL & Cloud Infrastructure:** GKE Workload Identity, Cloud SQL PG17 PSC, KMS CMEK, AWS EKS / Azure AKS / On-Prem K8s modules, `sensitive = true` handling, provider DAG ordering.

*(Finance/quant dogfooding workloads live strictly under [`examples/quant_dogfood/`](examples/quant_dogfood/) and are excluded from core contracts and domain-neutral gates.)*

---

## 5. Statistical Evaluation Bounds

### Clopper-Pearson & Wilson 95% Confidence Intervals

- **One-sided 95% Clopper-Pearson upper failure bound ($k = 0$ failures in $N$ trials):**
  $$p_U = 1 - 0.05^{1/N} \approx \frac{3}{N}$$
- **Two-sided 95% Wilson Score interval** (implemented in `src/context_scout/evals/enterprise_wiki_scale_benchmark.py` with $z = 1.959963984540054$) is reported across all proportional release gates in [`reports/v1_ga_release_scorecard.md`](reports/v1_ga_release_scorecard.md).

### Statistical Benchmark Suites (`evals/datasets/`)

| Benchmark ID | Dataset | Sample Size ($N$) | Target | Verification Test |
|---|---|---|---|---|
| **EVAL-01** | Unanswerable Queries | $N = 300$ | 100% `$0` abstention ($p_U \le 1.0\%$) | `tests/test_zero_spend_abstention.py` |
| **EVAL-02** | Stale Content | $N = 100$ | 100% stale detection ($p_U \le 3.0\%$) | `evals/datasets/stale_content_dataset.py`, `tests/test_step2_2_confluence_lifecycle.py` |
| **EVAL-03** | Same-Name Collisions | $N = 100$ | 100% repo-qualified disambiguation | `evals/datasets/same_name_collision_dataset.py`, `tests/test_m2_f08_f09_merkle_push_disambiguation.py` |
| **EVAL-04** | Answerable Ground Truth | $N = 300$ | Cross-stack symbol & citation verification | `tests/test_zero_spend_abstention.py` |
| **EVAL-05** | BPE Token A/B Suite | $N = 120$ ($30 \times 4$ stacks) | $\ge 80\%$ token reduction (Wilcoxon $p < 0.01$, Cohen's $d \ge 0.80$) | `tests/test_m0_02_governance_and_eval_fixes.py`, `evals/datasets/token_ab_dataset.py` |

---

## 6. Quality & Release Gates

All release candidates must pass [`reports/v1_ga_release_scorecard.md`](reports/v1_ga_release_scorecard.md) gates `L1`–`L12`:
1. `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src tests` (`0` errors).
2. `squawk --pg-version=17.0` (`0` DDL lock violations across Alembic `0002`–`0023`) and `23/23` PG17 `FORCE RLS` catalog verification.
3. Full zero-skip `pytest` + `vitest` via `bash .agents/skills/gate-verify/scripts/run_compact_gates.sh --scope all`.
4. Zero internal build-scaffolding identifiers across all tracked files (`tests/test_release_readiness_and_seed_demo.py::test_tracked_files_contain_zero_internal_build_scaffolding_identifiers`).
