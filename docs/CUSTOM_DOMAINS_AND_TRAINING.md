# Custom Domains, Ingestion & QLoRA Cartridge Training Guide

**Enterprise Context Scout** is built around declarative `DomainPack` manifests. Engineers can onboard a new codebase (e.g., platform infrastructure, microservices) or knowledge corpus (e.g., legal/compliance policies, biotech protocols, internal runbooks) **with zero edits to core Python (`src/`) or React (`web-ui/`) code**.

For custom language parsers (`LanguagePlugin`) or upstream SaaS connectors (`SourceConnectorPlugin`), see [PLUGIN_CONNECTOR_AUTHORING_GUIDE.md](./PLUGIN_CONNECTOR_AUTHORING_GUIDE.md).

---

## 1. Declarative `domains/<domain_id>/domain.yaml` Schema

Create a directory `domains/<your_domain>/` containing `domain.yaml`. On startup or request, `DomainRegistry` (`src/context_scout/agents/domain_registry.py`) validates it against `DomainPackManifest` (`src/context_scout/schemas/contracts.py`), and `GET /api/v1/workspace-config` + `federated_all` automatically expose it to the API and React workbench.

### Annotated Example (`domains/platform_infra_scout/domain.yaml`)

```yaml
domain_id: platform_infra_scout
display_name: "Platform Infrastructure & Terraform Scout"
description: "Kubernetes manifests, Terraform modules, Helm charts, and SRE runbooks."

# Google-only model policy enforced by validate_google_model_id()
base_model_id: "google/gemma-4-26B-A4B-it"
contextualizer_model_id: "google/gemma-4-E4B-it"
embedding_model_id: "google/embeddinggemma-2"
embedding_dimensions: 768

# Set lora_enabled: true for structured CodeSlicePointer JSON domains;
# set lora_enabled: false for prose/document synthesis domains (uses base Gemma-4).
lora_enabled: true
lora_adapter_id: "gemma4_26b_platform_infra"
lora_adapter_path: "domains/platform_infra_scout/adapters/gemma4_26b_platform_infra"

# Modalities indexed in this domain: any subset of ["code", "document", "issues"]
modalities:
  - "code"
  - "document"

# PostgreSQL 17 FORCE RLS scopes granted to this domain by default
default_scopes:
  - "public_eng"
  - "domain:platform_infra_scout"

# Explicit read-only MCP tool allow-list (enforced by DomainAuthorizationGate)
mcp_read_tools_allowed:
  - "code_repo_mcp:assemble_verified_context_pack"
  - "code_repo_mcp:search_code_trigram"
  - "code_repo_mcp:lookup_ast_symbol"
  - "code_repo_mcp:expand_ast_1hop_closure"
  - "code_repo_mcp:slice_file_lines"
  - "doc_corpus_mcp:search_hybrid_rrf"
  - "doc_corpus_mcp:verify_citation_quote"
  - "doc_corpus_mcp:get_document_section"

# Truth hierarchy & freshness rules
authority_rules:
  primary_truth_source: "code"          # "code" | "document"
  require_out_of_sample_check: false    # true only for quant backtest domains
  require_transaction_cost_check: false # true only for quant backtest domains
  freshness_half_life_days: 180

# Output schema contract: "ImplementationContextPack" | "DomainKnowledgeBrief" | "QuantStrategyBrief"
output_contract: "ImplementationContextPack"

# Optional source directory for CLI sync / benchmark-domain
source_dir: "data/platform_infra"

# Surfaced dynamically in GET /api/v1/workspace-config as quick-pick prompts
sample_queries:
  - "How are EKS node group IAM roles and VPC subnets wired in Terraform?"
  - "Which runbook steps govern PostgreSQL failover recovery?"
```

### How Auto-Discovery Works
1. **`GET /api/v1/workspace-config`** loads all valid `domains/*/domain.yaml` manifests via `DomainRegistry.load_all()`, queries live PostgreSQL 17 `FORCE RLS` counts for the caller's `PrincipalContext`, and populates the UI domain selector and sample queries dynamically.
2. **`federated_all` Routing:** When a query targets `federated_all`, `ScoutWorkflow` probes every domain in `DomainRegistry` that the caller's `allowed_scopes` authorize, ranks domains by calibrated relevance score (`>= CALIBRATED_RELEVANCE_FLOOR`), and routes to the best matching domain(s)—never hardcoding domain IDs.

---

## 2. CLI & Ingestion Workflows (`uv run context-scout`)

The CLI (`src/context_scout/cli.py`) manages hardware inspection, domain scaffolding, Git Merkle / multi-format ingestion, distillation, QLoRA training, querying, and benchmarking.

```bash
# 0. Inspect host RAM/VRAM, quantization backend, and safe KV-cache budget
uv run context-scout hardware

# 1. Scaffold a new DomainPack (--domain-type code | knowledge | document | research)
uv run context-scout init-domain \
  --domain-id platform_infra_scout \
  --domain-type code \
  --display-name "Platform Infrastructure Scout" \
  --source-dir /path/to/your/repos-or-docs
```

### 2.1 Codebase Ingestion (Git Merkle + 11-Language AST)
`uv run context-scout sync --domain-id <domain_id>` invokes `GitMerkleIndexer` (`src/context_scout/ingestion/git_merkle_indexer.py`):
- Diffs `git ls-tree -r HEAD` blob SHAs against `code_files.git_blob_sha` and skips unchanged files.
- Overlays uncommitted edits (`M`), untracked files (`??`), and deletions (`D`) via `git --no-optional-locks status --porcelain -unormal` in `<50 ms`.
- Parses changed files with `ASTExtractor` (Python, TypeScript/TSX, JS/JSX, Go, Rust, Java, C#, SQL, Terraform, YAML, Markdown) and links cross-service contracts via `CrossBoundaryContractLinker`.

### 2.2 Multi-Format Document Ingestion & Corpus Hygiene
`context-scout sync` and `src/context_scout/ingestion/doc_parser.py` ingest `.pdf`, `.md`, `.mdx`, `.adoc`, `.sgml`, `.tex`, `.rst`, `.json`, `.csv`, and `.html` files into PostgreSQL 17 `doc_chunks`:
- Preserves LaTeX math environments and Markdown/HTML tables intact (`LatexDocParser`, `MarkdownTableChunker`).
- Prepends 50-token contextual headers (`google/gemma-4-E4B-it`) and embeds with 768-d `google/embeddinggemma-2` (`SCOUT_EMBEDDING_DEVICE=cpu` when running alongside MLX).
- **Corpus Hygiene & Enrichment (`scripts/clean_and_enrich_corpus.py`):** Scrubs raw `.tex` proof noise, SGML/DocBook markup, and license/stub boilerplate from `doc_chunks`, and can ingest clean open-source foundational reference pages before re-indexing:
  ```bash
  .venv/bin/python scripts/clean_and_enrich_corpus.py --clean --enrich --verify
  ```

---

## 3. Teacher-Student Distillation & QLoRA Cartridge Training

### 3.1 Core Invariants (No Memorization, No Regex Self-Distillation)
1. **Never Train Weights on Dynamic Facts:** QLoRA adapters (`r=16, alpha=32`) learn **task behavior**—compact `CodeSlicePointer` JSON formatting, AST skeleton localization, and bounce-back recovery—never raw repository contents or static document facts (which live in PostgreSQL 17 under `FORCE RLS`).
2. **Never Distill from Regex/Template Scrapers:** `TeacherDistiller` (`src/context_scout/tuning/teacher_distiller.py`) generates candidate trajectories using a real reasoning teacher (`gemini-3.8-flash` or `google/gemma-4-31B-it`) grounded in **full `verbatim_content` chunks** across `sufficient`, `partial`, and `insufficient` regimes.
3. **Deterministic Rejection Sampling:** Every candidate trajectory must pass `PreFlightValidator` before entering `train.jsonl` / `valid.jsonl`, with 15–20% multi-turn bounce-back recovery examples included.
4. **Leakage-Free 85/15 Split:** Stratified by module prefix (code) or `doc_id` (documents) so evaluation repositories/documents never appear in training rows.
5. **When to Enable QLoRA (`lora_enabled`):** Enable QLoRA (`lora_enabled: true`) on **code pointer domains** where strict `CodeSlicePointer` JSON localization cuts prompt overhead by `96.65%`. Keep `lora_enabled: false` on **prose/research synthesis domains** so base `Gemma-4-26B-A4B-it` synthesizes directly from full retrieved chunks without 4-bit adapter JSON looping.

### 3.2 Distill, Train & Evaluate

```bash
# 2. Distill leakage-free JSONL datasets with PreFlightValidator rejection sampling
uv run context-scout distill \
  --domain-id platform_infra_scout \
  --source-dir /path/to/your/repos \
  --num-samples 200

# 3a. Train locally on Apple Silicon (M5 Pro 48GB) via mlx-lm (freezes all 128 MoE router.proj gates)
uv run context-scout train \
  --domain-id platform_infra_scout \
  --iters 400 \
  --mlx-native

# 3b. Or build / dispatch a GCP Vertex AI Hugging Face PEFT job (NVIDIA L4 / A100)
uv run python -m context_scout.tuning.train_vertex_peft --dry-run

# 4. Query locally ($0) or hand off to Gemini 3.8 Flash in an isolated GitWorktreeSandbox
uv run context-scout query \
  --domain-id platform_infra_scout \
  --prompt "Localize the Terraform VPC peering module and check CIDR validation" \
  --handoff-apply

# 5. Run the multi-tier domain readiness benchmark
uv run context-scout benchmark-domain \
  --domain-id platform_infra_scout \
  --max-queries 60 \
  --output-json reports/platform_infra_benchmark.json \
  --output-md reports/platform_infra_benchmark.md
```

- **Local MLX Training (`src/context_scout/tuning/train_mlx_lora.py`):** Applies rank-16 LoRA exclusively to shared attention (`q/k/v/o_proj`) and shared MLP (`gate/up/down_proj`) layers while freezing all 128 MoE `router.proj` gates, acquiring `/tmp/context_scout_gpu.lock` to avoid Metal collisions, and saving `adapters.safetensors` (~9.97 MB).
- **Cloud Vertex AI PEFT (`src/context_scout/tuning/train_vertex_peft.py`):** Generates `CustomJob` specs (`LoraConfig(r=16, lora_alpha=32)`, `g2-standard-12` + `NVIDIA_L4` in `europe-west3` by default) with GCS adapter export.
