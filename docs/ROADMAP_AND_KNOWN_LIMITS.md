# Enterprise Context Scout — Roadmap & Known Limits (`v1.0.1`)

> Concise engineering roadmap and known system boundaries for **Enterprise Context Scout (`v1.0.1`)**, replacing archived milestone logs previously under `docs/phase_reports/`.

## 1. Product Scope & Supported Sources
Enterprise Context Scout focuses strictly on enterprise engineering stack sources:
- **Source Code & VCS:** GitHub / GitHub Enterprise, GitLab / Self-Managed GitLab, Local Git, and offline Air-Gap Bundles (`deploy/airgap/create_airgap_bundle.sh`).
- **Engineering Knowledge & Tracking:** Atlassian Confluence and Jira (with hierarchical epic/parent links and MR/PR cross-links).
- **Polyglot AST & Contracts (`15` built-in plugins):** Python, TypeScript/JavaScript, React/Redux, Storybook, C#, Java/Spring, Kotlin, Go, Rust, SQL/dbt, Terraform/HCL, OpenAPI/AsyncAPI/GraphQL, Protobuf/gRPC, Templates/i18n, and SCIP indexes.

Non-engineering suites (Slack, Google Drive, SharePoint) are intentionally out of scope.

## 2. Operational Boundaries & Known Limits
1. **Air-Gapped Serving:** Production deployments pre-stage `google/embeddinggemma-2` (or approved local embedders) and base LLM weights offline; `SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS` fails closed in `production`, `prod`, `staging`, and `qa`.
2. **Container Footprint vs. GPU Specialization:** Default `api` and `worker` runtime images target `< 2.0 GiB` using the CPU PyTorch backend (`ARG TORCH_BACKEND=cpu`). GPU QLoRA distillation and `vLLM` multi-LoRA serving require installing the `gpu-train` / `cu128` extras on CUDA 12.8 nodes.
3. **Database Schema & Migrations:** `sql/001_init_schema.sql`, `sql/002_production_enterprise_schema.sql`, `src/context_scout/db/schema.py`, and Alembic `0001..0024` maintain 23-table PostgreSQL 17 `FORCE ROW LEVEL SECURITY` parity under `NOBYPASSRLS`.
4. **Remote Zoekt Multi-Tenancy:** Shared remote Zoekt shards are namespaced strictly as `{tenant_id}::{repo_name}` across search queries, reconciliation, and artifact purge operations.

## 3. Post-GA Engineering Roadmap
- **External Open-Source Benchmark Expansion:** Expand `src/context_scout/evals/unseen_ab_benchmark.py` with additional pinned public-repo evaluations across Rust, Go, and multi-service monorepos.
- **Live PostgreSQL CI Coverage Extension:** Expand container-backed `S3` + `Zoekt` integration tests using ephemeral Testcontainers/Docker Compose services.
- **Supply-Chain & Release Hardening:** Maintain SBOM (`scripts/generate_aibom.py`), Trivy CRITICAL library/IaC gates, kubeconform strict manifest checks, and SLSA Level 3 build provenance attestations (`release-supply-chain.yml`).
