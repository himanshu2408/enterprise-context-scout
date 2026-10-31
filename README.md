# Enterprise Context Scout & Multi-Domain Knowledge Platform

An enterprise-grade, dual-engine context discovery and verification platform built on the **Google AI & Open-Weights Stack** (`Gemma 4 26B-A4B-it` MoE, `Gemma 4 E4B-it`, `EmbeddingGemma 2`, `ShieldGemma 2` / Google Cloud Model Armor, `Google ADK 2.12`, `MCP 2.3`, and `PostgreSQL 17` with `pgvector`, `pg_trgm`, and `FORCE ROW LEVEL SECURITY`).

---

## 1. Why This Platform Exists: Frontier Economics vs. Zero-Cost Local Discovery

When engineering and quantitative research teams deploy commercial frontier models directly against entire codebases and document corpora, **80–90% of paid input tokens are wasted on exploratory search**—reading irrelevant directories, tracing dead-end call graphs, and inspecting stale documentation. Concurrently, naive vector chunking creates "half-information" hallucinations by severing functions from their imported types, callers, and test fixtures.

### The Economic Reality (October 2026 Frontier Pricing vs. Zero-Cost Local Discovery)

| Model Category | Model ID | Input / 1M Tokens | Output / 1M Tokens | Architectural Role & Economic Profile |
|:---|:---|:---:|:---:|:---|
| **Frontier Comparison** | Anthropic Claude Opus 5.5 | $4.00 | $20.00 | Premium frontier model; prohibitive for brute-force multi-turn codebase traversal |
| **Frontier Comparison** | OpenAI GPT-5.6 Sol | $4.00 | $20.00 | Premium frontier model; prohibitive for brute-force multi-turn codebase traversal |
| **Google Frontier** | Google Gemini 4 Argon (`gemini-4-argon`) | $2.00–$4.00 | $10.00–$20.00 | Google frontier flagship ($2.00/$10.00 intro to Nov 30, 2026; $4.00/$20.00 std); 1-shot complex reasoning |
| **Local Discovery Engine** | **Google Gemma 4 26B-A4B-it** | **$0.00** | **$0.00** | **100% Local Discovery & Verification**: MoE (3.8B active), runs on local hardware at zero token cost |
| **Fast Execution Target** | **Google Gemini 3.8 Flash (`gemini-3.8-flash`)** | **$0.75** | **$3.75** | **1-Shot Implementation Target**: Promo through Dec 31, 2026 ($1.50/$7.50 std); fast patch synthesis |

*(Note: Legacy 2025 models such as `gemini-2.5-pro` and `gemini-2.5-flash` are retired from frontier designation.)*

**Enterprise Context Scout** decouples **Zero-Cost Local Discovery & Verification** from **Targeted Google Model Implementation**:

1. **Zero-Cost Local Discovery & Verification Engine (100% Local, $0 Token Cost):**
   - 100% of codebase exploration and research verification runs locally with open-weights `Gemma-4-26B-A4B-it` (MoE: 25.2B total / 3.8B active parameters, 256K context) and domain-specific QLoRA adapters, absorbing 30,000–60,000 tokens of noisy multi-turn exploration at $0 API cost.
   - **Engine 1 — Codebase Intelligence (`trading_code_scout`):** Git Blob-SHA Merkle incremental indexing (`<50ms` sync), 3-way Normalized Reciprocal Rank Fusion (`pg_trgm` + `tsvector` BM25 + `EmbeddingGemma 2` 768-d), and `tree-sitter` AST symbol/call graph with bounded 1-hop AST auto-closure. Emits structured `CodeSlicePointer` objects sliced byte-for-byte with `git_commit_sha` provenance.
   - **Engine 2 — Multi-Market Quantitative Research (`quant_research_scout`):** Automated open-access harvester (arXiv + OpenAlex covering Indian `IN_NSE`, German `DE_XETRA`, and US `US_EQUITY` markets), lossless LaTeX/Docling extraction, Anthropic Contextual Retrieval (`Gemma-4-E4B-it`), and Market-Regime Quant Scepticism Gate ($\ge 15\text{ bps}$ NSE delivery, $\ge 10\text{ bps}$ Xetra, $\ge 5\text{ bps}$ US).
2. **Paid Fast Execution Engine (100% Google Model Runtime):**
   - Implementation executes in a single turn using **Google `gemini-3.8-flash`** ($0.75/1M input, $3.75/1M output promo through Dec 31, 2026), or **`gemini-4-argon`** when complex frontier architectural reasoning is required.
   - The execution model receives a compact, verified **`ImplementationContextPack`** (typically 2,000–4,000 tokens instead of 45,000+ raw repository tokens), achieving 80–90% paid token savings over ungrounded frontier queries.
3. **4-Layer Pre-Flight Quality Gate:**
   - Validates 100% internal symbol resolution, Normalized RRF floor ($\ge 0.55$), and NFKC-normalized verbatim citation matching ($\ge 0.98$) before spending a single paid Gemini token, with a local $0 bounce-back self-correction loop and a `ContextMissEvent` QLoRA retraining flywheel.

---

## 2. Monorepo Directory Layout

```text
enterprise-context-scout/
├── pyproject.toml              # Modern Python 3.12+ project config (ruff, mypy, pytest)
├── uv.lock                     # Deterministic pinned dependency lockfile
├── .github/workflows/ci.yml    # Production 6-stage GitHub Actions CI/CD pipeline with pgvector & WIF
├── docker-compose.yml          # PostgreSQL 17 + pgvector container definition
├── sql/
│   └── 001_init_schema.sql     # PostgreSQL 17 DDL: tables, indexes, and FORCE RLS policies
├── src/
│   └── context_scout/
│       ├── __init__.py         # Package root
│       ├── config.py           # Environment config & runtime hardware auto-detection
│       ├── cli.py              # CLI (`uv run context-scout`) to scaffold, distill & train custom domains
│       ├── api/                # FastAPI SSE streaming backend (M8)
│       │   ├── __init__.py
│       │   ├── main.py         # App factory, CORS middleware, OTel request tracing
│       │   ├── schemas.py      # Pydantic v2 API request/response and SSE schemas
│       │   ├── dependencies.py # Out-of-band principal extraction (SEC-04) & token store
│       │   └── routes/         # Endpoints: health, domains, scout, handoff, telemetry
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── contracts.py    # Pydantic v2 core contracts (CodeSlicePointer, ASTSymbol, etc.)
│       ├── db/
│       │   ├── __init__.py
│       │   ├── bm25.py         # Pure-Python Lucene Okapi BM25 with smoothed non-negative IDF
│       │   ├── connection.py   # Psycopg 3 connection pool with RLS context manager
│       │   ├── repositories.py # PostgresRLSRepository & InMemoryRLSRepository (3-way RRF)
│       │   └── schema.py       # Table names and database constants
│       ├── ast_engine/
│       │   ├── __init__.py
│       │   ├── parser.py       # Multi-language Tree-sitter parser (Python & TypeScript)
│       │   ├── expander.py     # ASTDependencyExpander: bounded 1-hop caller closure (SEC-03)
│       │   └── slicer.py       # DeterministicSlicer: byte-exact slicing & NFKC quote verification
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── merkle.py       # GitMerkleIndexer: Git Blob-SHA diffing (<50ms incremental sync)
│       │   ├── paper_harvester.py # Multi-market arXiv Atom & OpenAlex harvester with jittered backoff
│       │   ├── doc_parser.py   # Lossless LatexDocParser & MarkdownTableChunker
│       │   ├── contextualizer.py # Anthropic Contextual Retrieval with google/gemma-4-E4B-it
│       │   └── scepticism_gate.py # Market-Regime Scepticism Gate (statutory friction & citation check)
│       ├── trading_strategies/ # Realistic multi-file quantitative execution & alpha strategies
│       │   ├── __init__.py
│       │   ├── volatility_garch.py # GARCH(1,1) conditional volatility with quasi-MLE
│       │   ├── kelly_risk_manager.py # Multi-asset Kelly criterion with Ledoit-Wolf shrinkage
│       │   ├── order_flow_imbalance.py # Multi-level OFI with depth decay (Cont et al. 2014)
│       │   ├── pairs_trading_kalman.py # Kalman hedge ratio tracking with Joseph-form stability
│       │   └── almgren_chriss_execution.py # Optimal execution trajectory with impact parameters
│       ├── security/           # Enterprise Out-of-Band RBAC and Guardrails (SEC-04)
│       │   ├── __init__.py
│       │   ├── rbac.py         # Out-of-band PrincipalContext binding & DomainAuthorizationGate
│       │   └── guardrails.py   # PromptInjectionSanitizer, SecretRedactor, ShieldGemmaGuardrail
│       ├── mcp_servers/        # Enterprise Model Context Protocol servers (mcp v2.3.0)
│       │   ├── __init__.py
│       │   ├── common.py       # Hard payload caps (<=10 items, <=32 KB) & out-of-band context
│       │   ├── code_repo_mcp.py # Code repository MCP server (5 canonical tools + 5 aliases)
│       │   ├── quant_corpus_mcp.py # Quant corpus MCP server (4 canonical tools + 3 aliases)
│       │   └── seed_code_corpus.py # Database seeding engine for strategies and GitHub issues
│       ├── agents/
│       │   ├── __init__.py     # Agent interfaces and adapter management
│       │   ├── preflight_gate.py # 4-Check PreFlightValidator & $0 bounce-back loop
│       │   ├── scout_workflow.py # Google ADK 2.12 ScoutWorkflow & FederatedScoutCoordinator
│       │   └── gemini_executor.py # Paid Gemini executor, budget guardrails & GitWorktreeSandbox
│       ├── telemetry.py        # OpenTelemetry GenAI semantic conventions & tracer
│       ├── evals/              # CI/CD evaluation harness & golden benchmark suite
│       │   ├── __init__.py     # Package exports
│       │   ├── golden_dataset.py # Typed Pydantic benchmark suite (10 retrieval, 22 RBAC, 22 injection)
│       │   ├── harness.py      # 6 CI/CD gate metrics computation & threshold validation
│       │   └── runner.py       # CLI runner generating JSON & Markdown scorecards
│       └── tuning/             # Teacher-Student Distillation & QLoRA offline tuning engine
│           ├── __init__.py     # Package exports
│           ├── teacher_distiller.py # Distillation pipeline with PreFlightValidator rejection sampling
│           ├── train_mlx_lora.py # Local 4-bit QLoRA trainer on Apple M5 Pro with HW-01 lock
│           ├── train_vertex_peft.py # Enterprise GCP Vertex AI PEFT job spec generator
│           └── benchmark.py    # Empirical Before-vs-After benchmark runner
├── web-ui/                     # Strict TypeScript + React Enterprise UI (M8)
│   ├── package.json
│   ├── tsconfig.json           # strict: true, noImplicitAny: true, zero any
│   ├── vite.config.ts
│   ├── src/
│   │   ├── types/contracts.ts  # Mirroring Pydantic v2 schemas
│   │   ├── components/         # RoleSwitcher, DomainSelector, ScoutConsole, etc.
│   │   └── __tests__/          # 26 Vitest component tests (100% pass)
├── domains/                    # Declarative DomainPack YAML configs & trained QLoRA adapters
│   ├── trading_code_scout/
│   │   ├── domain.yaml
│   │   └── adapters/gemma4_26b_code_scout/ # Genuine trained adapters.safetensors, config, metrics
│   └── quant_research_scout/
│       ├── domain.yaml
│       └── adapters/gemma4_26b_quant_research/ # Genuine trained adapters.safetensors, config, metrics
├── reports/                    # CI/CD evaluation reports (JSON & Markdown)
│   ├── m6_eval_report.json
│   ├── m6_eval_report.md
│   ├── m7_qlora_benchmark_report.json
│   └── m7_qlora_benchmark_report.md
└── tests/
    ├── conftest.py             # Shared fixtures (temp repos, live postgres connections)
    ├── test_m1_foundation.py   # M1 contracts, schemas, and adapter validation
    ├── test_postgres_rls_live.py # M1 live PostgreSQL 17 RLS security tests
    ├── test_m2_code_engine.py  # M2 parser, expander, and slicer unit tests
    ├── test_dependency_expander.py # Bounded 1-hop expansion and budget compliance tests
    ├── test_deterministic_slicer.py # Exact slicing and NFKC verification tests
    ├── test_postgres_m2_live.py # Live Merkle indexer & DB sync tests
    ├── test_adversarial_ast_rls.py # Adversarial injection and security boundary tests
    ├── test_empirical_merkle_slicer_challenge.py # Benchmark tests (<50ms Merkle sync, >=0.98 NFKC)
    ├── test_m3_quant_engine.py # M3 paper harvester, parser, contextualizer & RRF tests
    ├── test_postgres_m3_live.py # M3 PostgreSQL 17 live RLS & 3-way RRF integration tests
    ├── test_adversarial_m3_chunker_harvester.py # Adversarial stress tests for chunkers & harvesters (17 tests)
    ├── test_adversarial_m3_rrf_scepticism.py # Adversarial stress tests for 3-way RRF & scepticism gate (24 tests)
    ├── test_real_paper_harvest_live.py # Live harvest, PDF fallback, DB persistence & gate integration (6 tests)
    ├── test_m7_teacher_distiller.py # M7 distillation, rejection sampling, 15-20% bounce-back tests
    ├── test_m7_qlora_tuning.py # M7 QLoRA memory math, HW-01 lock, router freezing & benchmark tests
    ├── test_m4_mcp_servers.py  # M4 Enterprise MCP servers, live Postgres RLS & bounds tests (18 tests)
    ├── test_m4_security.py     # M4 Out-of-band RBAC, domain gate, and guardrail tests (37 tests)
    ├── test_adversarial_security.py # M4 Adversarial security fuzzing and pool isolation tests (47 tests)
    ├── test_preflight_gate.py  # M5 4-Check pre-flight quality gate & bounce-back tests (10 tests)
    ├── test_scout_workflow.py  # M5 ADK 4-stage workflow DAG & federated coordinator tests (13 tests)
    ├── test_gemini_executor.py # M5 Gemini model policy, budget gate & worktree tests (10 tests)
    ├── test_m5_adk_workflow.py # M5 End-to-end integration & live Postgres telemetry tests (17 tests)
    ├── test_m6_eval_harness.py # M6 Golden dataset, 6 CI/CD gates, OTel GenAI traces & live PG17 (13 tests)
    └── test_m8_fastapi_sse.py  # M8 FastAPI SSE backend, RBAC, ShieldGemma, Handoff & CI/CD (25 tests)
```

---

## 3. Milestone 1 Architecture: Foundation & PostgreSQL 17 RLS Security

Milestone 1 establishes the bedrock data infrastructure, strict multi-tenant isolation, and Pydantic v2 domain models for all subsequent engines.

### 3.1 Non-Bypassable Row-Level Security (`FORCE RLS`)

In enterprise environments, data leakage between tenants or permission tiers must be prevented at the database kernel level rather than relying on application-level `WHERE` clauses.

- **Forced RLS:** Every sensitive table enforces `ALTER TABLE <table_name> FORCE ROW LEVEL SECURITY;`. This ensures that even the table owner is subject to RLS policies.
- **Role Isolation:** The application connects exclusively as the unprivileged `context_scout_app` role. Superuser accounts (`postgres`) are prohibited in runtime application connection pools to prevent accidental RLS bypass.
- **Session Scoping:** Multi-tenant boundaries are enforced per connection or transaction block using:
  ```sql
  SET LOCAL app.principal_scopes = 'tenant:acme,role:quant,repo:prod_strategies';
  ```
- **Policy Enforcement:** RLS policies evaluate `current_setting('app.principal_scopes', true)` against tenant and scope arrays using PostgreSQL string-matching and array operators:
  ```sql
  CREATE POLICY git_blobs_isolation_policy ON git_blobs
      FOR ALL
      TO context_scout_app
      USING (
          app_has_scope(tenant_id, allowed_roles)
      );
  ```

### 3.2 Relational & Vector Schema (`sql/001_init_schema.sql`)

The schema manages 6 core entities:
1. `git_blobs`: Stores file paths, Git Blob-SHAs, content, and line count for Merkle diffing.
2. `ast_symbols`: Extracted definitions (functions, classes, methods, interfaces, variables) with byte offsets and line numbers.
3. `ast_edges`: Directed caller/callee and import relationships between symbols (`CALLS`, `IMPORTS`, `EXTENDS`).
4. `document_chunks`: Text chunks indexed with both `pgvector` (768-d vector embeddings) and `tsvector` (full-text search).
5. `context_packs`: Audit log of assembled context packs with token counts and model targets.
6. `feedback_events`: Feedback telemetry tracking citation hits, misses, and model output quality.

### 3.3 Core Domain Contracts (`contracts.py`)

All cross-module payloads are strictly validated using Pydantic v2 models:
- **`PrincipalContext`:** Represents authenticated caller identity, tenant ID, and granted permission scopes.
- **`CodeSlicePointer`:** Precise pointer containing `repo_id`, `git_commit_sha`, `file_path`, `start_line`, `end_line`, `byte_offset`, and `byte_length`.
- **`ASTSymbol` & `ASTEdge`:** Structured representations of codebase symbols and directed semantic edges.
- **`QuantStrategyBrief`:** Schema for quantitative trading strategies, specifying mathematical formulas, trade parameters, and market friction rules.
- **`ContextPack`:** The final token-budgeted, security-cleared context package sent to the implementation model.

---

## 4. Milestone 2 Architecture: AST-Grounded Code Engine

Milestone 2 delivers the `trading_code_scout` engine, replacing naive vector search over code with deterministic AST analysis and Merkle-diffed indexing.

### 4.1 Git Blob-SHA Merkle Indexing (`GitMerkleIndexer`)

Traditional code search platforms re-index entire repositories on every commit, introducing multi-minute delays. The `GitMerkleIndexer` exploits Git's underlying Merkle DAG:

- **Blob-SHA Diffing:** Computes changes between commit trees by querying Git tree objects directly. If a file's Blob-SHA is identical to the indexed SHA, parsing is bypassed entirely.
- **Performance Benchmark:** Processes incremental commits and syncs database records in **$<50\text{ ms}$** (measured $\approx 33.7\text{ ms}$ in-memory, $\approx 37.4\text{ ms}$ with live PostgreSQL 17 sync).
- **Staleness Elimination:** Eliminates vector re-embedding latency for unchanged files, keeping the index synchronized in real time with active repositories.

### 4.2 Multi-Language Tree-Sitter Parser (`TreeSitterParser`)

The parser constructs concrete syntax trees (CSTs) for Python (`tree_sitter_python`) and TypeScript (`tree_sitter_typescript`), extracting:
- **Symbols:** Function definitions, classes, methods, imports, and exported variables with exact byte offsets and line ranges.
- **Directed Call Graph Edges:** Analyzes function calls, method invocations, and class inheritance, generating directed `ast_edges` between caller and callee symbols.
- **Fault-Tolerant Parsing:** Indexes syntactically malformed or in-progress code files without failing, allowing continuous indexing while developers edit.

### 4.3 Bounded 1-Hop AST Dependency Closure (`ASTDependencyExpander`, `SEC-03`)

Sending entire files or unconstrained graph closures to an LLM wastes token budget and exposes unrelated tenant code. The `ASTDependencyExpander` enforces mathematical constraints:
- **Depth Limit:** Strictly 1-hop direct callers and callees of the target symbol.
- **Caller Bound:** At most $\le 3$ direct callers per symbol.
- **Token Budget:** Total expanded closure is bounded to $\le 1,500$ tokens.
- **Diversity Selection:** When $>3$ callers exist, the expander prioritizes callers across distinct files to provide diverse usage examples rather than multiple call sites in a single file.

### 4.4 Deterministic Slicing & Quote Verification (`DeterministicSlicer`, `VAL-02`)

To eliminate hallucinated code snippets and off-by-one line errors:
- **Byte-Exact Slicing:** Extracts exact line ranges directly from source files using UTF-8 byte offsets, stamping each slice with the immutable `git_commit_sha`.
- **NFKC Quote Verification:** Before citations are passed to the context pack, citations undergo Unicode NFKC normalization and token-overlap verification:
  $$\text{Overlap Ratio} = \frac{|\text{Tokens}_{\text{citation}} \cap \text{Tokens}_{\text{source}}|}{|\text{Tokens}_{\text{citation}}|}$$
  Any citation scoring below $\ge 0.98$ is rejected by the pre-flight gate.

---

## 5. Milestone 3 Architecture: Multi-Market Quant Research Harvester & Contextual Retrieval

Milestone 3 delivers the `quant_research_scout` engine, transforming unstructured quantitative research papers into strictly partitioned, contextually situated, and empirically grounded document chunks.

### 5.1 Multi-Market Paper Harvester (`paper_harvester.py`)

Academic quantitative literature is distributed across preprint servers and open-access repositories with disparate schemas and rate constraints. The harvester automates multi-market ingestion:

- **arXiv Ingestion:** Standard-library Atom XML parser (`parse_arxiv_atom_feed`) resolving `http://www.w3.org/2005/Atom` namespaces. Handles versioned paper identifiers (e.g. `2403.99999v4` $\rightarrow$ canonical URI), extracts clean titles, summaries, and authors, and parses ISO-8601 dates with resilient `ValueError` fallback to baseline dates.
- **In-Memory LaTeX Archive Decompression & Include Resolution:**
  - `unpack_latex_archive` inspects `/e-print/{arxiv_num}` payload bytes directly in memory.
  - Automatically identifies and decompresses `.tar.gz` and standalone `.gz` archives without writing temporary files to disk, eliminating `UnicodeDecodeError` on binary gzip headers (`\x1f\x8b`).
  - `resolve_latex_includes` locates root TeX documents (`\documentclass` + `\begin{document}`) and recursively resolves nested `\input{...}`, `\include{...}`, and `\subfile{...}` directives, stripping subfile preambles to produce a single coherent LaTeX document.
- **OpenAlex REST API v2 Integration:**
  - Employs canonical OpenAlex concept IDs (`OPENALEX_CONCEPT_MAP`: Economics `C162324750`, Financial Economics `C144133570`, Finance `C149782125`, Financial System `C106159729`, Econometrics `C149923435`, Order Book `C2779835941`) to bypass deprecated field queries.
  - Enforces `has_fulltext:true`, `type:article|preprint`, `is_retracted:false`, and `sort=cited_by_count:desc` to prioritize high-impact quantitative research.
  - Integrates OpenAlex polite pool routing (`mailto=scout@example.com`) ensuring 10 req/s throughput.
  - Reconstructs linear abstract text from inverted token-index arrays (`abstract_inverted_index`).
- **Market Scope Presets (`MarketRegime`):**
  - `IN_NSE` (`MarketRegime.INDIA_NSE_BSE`): National Stock Exchange of India & Bombay Stock Exchange quantitative research (NIFTY/BANKNIFTY liquidity, STT transaction friction, market impact, order flow imbalance).
  - `DE_XETRA` (`MarketRegime.GERMANY_XETRA_DAX`): Deutsche Börse Xetra order book microstructure, DAX intraday volatility, and MiFID II European market structure.
  - `US_EQUITY` (`MarketRegime.US_NYSE_NASDAQ`): US equity/options microstructure, Reg NMS order routing, sub-penny trading, and algorithmic execution.
- **Resilient HTTP Client & Dual Fallback:** `HttpPaperHarvesterClient` implements exponential backoff ($t_{\text{sleep}} = \text{base} \times 2^{\text{attempt}} + \text{jitter}$) where $\text{base} = 3.5\text{ s}$ and $\text{jitter} \in [0.1, 0.5]\text{ s}$ to adhere to academic API rate limits. Automatically falls back from arXiv LaTeX sources to direct arXiv PDF downloads (`https://arxiv.org/pdf/{arxiv_num}.pdf`) or OpenAlex open-access PDF URLs when TeX source is unavailable.

### 5.2 Lossless LaTeX & PDF Document Parsing (`doc_parser.py`)

Standard text chunkers break documents at fixed token windows, splitting mathematical proofs and tabular data across chunk boundaries. Milestone 3 implements lossless chunkers:

- **`LatexDocParser`:**
  - Tokenizes documents into atomic blocks where math environments (`equation`, `equation*`, `align`, `align*`, `gather`, `gather*`, `multline`, `multline*`, `flalign*`, display math `$$...$$`, and inline math `$...$`) are never split across chunk boundaries.
  - Full table float preservation: matches `\begin{table} ... \begin{tabular} ... \end{tabular} ... \end{table}` and standalone `tabular` blocks as atomic units, preventing nested tabular boundary splitting under chunk token pressure.
  - Maintains a 3-level hierarchical section breadcrumb trail (`\section`, `\subsection`, `\subsubsection`) with automatic stack rollback upon section level decrements.
- **`PdfDocParser`:**
  - High-fidelity PDF fallback powered by `pypdf`.
  - Extracts full document text with automatic NUL-byte (`\x00`) sanitization to prevent PostgreSQL storage errors.
  - Regex-driven section boundary detection matching standard academic headings (`Abstract`, `Introduction`, `Methodology`, `Results`, `Discussion`, `Conclusion`, `References`).
  - Bounded token chunking (400–800 tokens) guaranteeing structural preservation when raw LaTeX source is inaccessible.
- **`MarkdownTableChunker`:**
  - Parses GitHub Flavored Markdown (GFM) pipe tables with leading and trailing whitespace tolerance.
  - Ensures multi-row data tables (e.g. 50+ row liquidity matrices) remain intact as indivisible units in `chunk.tables_markdown`.

### 5.3 Anthropic Contextual Retrieval with Gemma 4 (`contextualizer.py`)

Isolated document chunks lose their overarching document and section context, causing retrieval failures for generic queries.

- **Situated 50-Token Context Headers:** Implements Anthropic Contextual Retrieval targeting open-weights **`google/gemma-4-E4B-it`**:
  ```text
  [Context: Document "Statistical Arbitrage in Indian Equities" (Scope: IN_NSE); Section "Market Impact Model"]
  ```
- **Turn Template Formatting:** Prompts format user turns strictly using `<start_of_turn>user\n...\n<end_of_turn>\n<start_of_turn>model\n`.
- **Offline-First CI Architecture:** Includes `DeterministicContextualizer`, a zero-dependency local metadata extractor that generates compliant 50-token situated context headers for 100% deterministic, offline CI execution.

### 5.4 3-Way Normalized Reciprocal Rank Fusion (`BLK-01`)

Milestone 3 implements multimodal ranking fusion combining lexical, vector, and authority signals:

- **Ranking Modalities:**
  1. **Lexical BM25:** Pure-Python Lucene-style Okapi BM25 (`bm25.py`) with smoothed non-negative IDF ($\ln(1 + \frac{N - \text{df} + 0.5}{\text{df} + 0.5})$) for in-memory indexing, and PostgreSQL 17 `ts_rank_cd` full-text search.
  2. **Dense Vector Embeddings:** 768-dimensional cosine distance (`embedding <=> %s::vector`) targeting `EmbeddingGemma 2`.
  3. **Trigram & Authority Weight:** `similarity(verbatim_content, %s) * authority_weight` via PostgreSQL `pg_trgm`.
- **Normalization Invariant (`compute_normalized_rrf`):**
  $$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  $$\text{Normalized Score}(d) = \frac{\text{RRF}(d)}{|M| \times \frac{1}{k + 1}} \in [0.0, 1.0]$$
  - With $k = 60$ and $|M| = 3$, an item ranking #1 across all 3 modalities yields mathematically exact $1.000000$. Single hits in 1 modality yield $\approx 0.333333$.
- **Database-Kernel Multi-Tenant Isolation:**
  - Evaluated in PostgreSQL 17 CTEs with forced RLS (`ALTER TABLE doc_chunks FORCE ROW LEVEL SECURITY;`).
  - Search queries strictly enforce `doc_chunks_tenant_isolation_policy` at the storage layer prior to ranking window CTEs, eliminating cross-tenant ranking leakage.

### 5.5 Market-Regime Quant Scepticism Gate (`scepticism_gate.py`, `FIN-01` & `VAL-02`)

Backtests frequently overfit by assuming zero or unrealistically low execution friction. The Scepticism Gate validates quantitative strategy briefs against empirical market realities:

- **Statutory Transaction Cost Floors (`FIN-01`):**
  - **`IN_NSE` Delivery:** Minimum $\ge 15.00\text{ bps}$ (Securities Transaction Tax 0.1% delivery, exchange turnover fees, SEBI charges, stamp duty, GST).
  - **`IN_NSE` Intraday Futures:** Minimum $\ge 5.00\text{ bps}$ (STT 0.02% sell-side, clearing and exchange turnover charges).
  - **`DE_XETRA`:** Minimum $\ge 10.00\text{ bps}$ (Xetra trading venue fee schedule, CCP central counterparty clearing, banking settlement).
  - **`US_EQUITY`:** Minimum $\ge 5.00\text{ bps}$ (SEC Section 31 transaction fees, FINRA TAF, exchange routing and taker fees).
  - Any strategy assuming friction below the statutory floor is flagged with `FrictionVerdict.UNVERIFIED_FRICTION_WARNING` and rejected (`passed=False`).
- **Institutional Risk Thresholds:**
  - Net Sharpe degradation penalty: if gross Sharpe minus friction drag yields net Sharpe $\le 0.50$, the strategy fails after costs (`FAILS_AFTER_COSTS`).
  - Maximum Drawdown cap: strategies with maximum historical drawdown exceeding $30.0\%$ are rejected.
- **100% Verbatim Citation Verification (`VAL-02`):**
  - Every numerical claim and quote in the strategy brief is verified against indexed chunks using `DeterministicSlicer` with Unicode NFKC normalization.
  - Requires $\ge 0.98$ token overlap. Single-character or single-digit alterations (e.g. 15.2 bps altered to 15.3 bps) trigger `HallucinatedQuoteError` and immediate brief rejection.

### 5.6 Scaled Corpus Ingestion & Live PostgreSQL 17 Persistence

To validate the retrieval engine under institutional conditions, the automated harvest pipeline ingested a multi-market corpus:

- **Harvested Corpus Scale:**
  - **300 Total Verified Full-Text Papers:** Exactly 100 verified full-text papers per market (`IN_NSE`: 100, `DE_XETRA`: 100, `US_EQUITY`: 100), with 0 `.txt` abstract stubs (259 LaTeX `.tex` sources and 44 full-text `.pdf` papers).
  - **Local Storage Cache:** Cached under `data/papers/` (git-ignored) with structured metadata in `data/papers/metadata/{market}_papers.jsonl` and full source/summary text under `data/papers/content/{market}/`.
- **Database Persistence (`doc_chunks`):**
  - **14,331 Total Genuine Chunks Persisted** in local PostgreSQL 17 under domain `quant_research_scout` and scope `public_quant` with 768-dimensional `google/embeddinggemma-2` embeddings.
  - **Market Breakdown:** `DE_XETRA`: 5,013 chunks, `US_EQUITY`: 4,982 chunks, `IN_NSE`: 4,336 chunks.
  - Each chunk contains 50-token situated headers, 768-dimensional embeddings, and precomputed `tsvector` lexemes.
- **Live 3-Way Normalized RRF Verification:**
  - Query evaluation against live database chunks demonstrates strong multi-modal convergence:
    - `IN_NSE` ("order flow imbalance NIFTY market impact"): Top RRF score **0.7097** (BM25 rank: 34, Vector rank: 67, Trigram rank: 1).
    - `DE_XETRA` ("Xetra order book liquidity DAX volatility"): Top RRF score **0.6300** (BM25 rank: 227, Vector rank: 6, Trigram rank: 21).
    - `US_EQUITY` ("high frequency trading NYSE execution friction"): Top RRF score **0.7130** (BM25 rank: 16, Vector rank: 6, Trigram rank: 88).
  - All top results comfortably exceed the institutional $\ge 0.55$ Pre-Flight quality gate threshold.

---

## 6. Milestone 4 Architecture: Enterprise MCP Servers, Security Layer & Trading Strategy Package

Milestone 4 establishes production-grade enterprise interfaces implementing the Model Context Protocol (`mcp>=2.3.0`), out-of-band role-based access control, PostgreSQL 17 Forced RLS integration, and institutional quantitative trading strategies.

### 6.1 Enterprise Security Layer & Out-of-Band RBAC (`src/context_scout/security/`, `SEC-04`)

Enterprise agent architectures must guarantee zero privilege escalation and zero parameter injection leakage:

- **Out-of-Band Session Principal Binding (`rbac.py`):**
  - Session authentication strictly decoupled from LLM prompt inputs and tool parameters. `PrincipalContext` is bound out-of-band via Python `contextvars.ContextVar` (`bind_session_principal`, `get_current_principal`).
  - Zero parameter injection leakage: `validate_tool_call_arguments` blocks all case-insensitive attempts (`k.lower()`) to pass security attributes (`principal`, `principal_context`, `allowed_scopes`, `user_id`, `role`, `scopes`, `principal_scopes`, `app.principal_scopes`), and rejects NUL bytes (`\x00` in str arguments) with `ValueError`.
  - Zero parameter exposure: MCP tool schemas expose only domain business arguments; principal identity is never accepted from model generation.
- **PostgreSQL 17 Forced RLS Integration (`scoped_db_session`):**
  - Live PostgreSQL 17 session binding via `SET LOCAL app.principal_scopes = %s;` within transaction blocks.
  - Connection pool scope safety: Verified across connection pool recycling (`test_postgres_17_connection_pool_scope_isolation`) that returning pooled connections to a shared pool never bleeds principal scopes.
- **Domain Authorization Gate (`DomainAuthorizationGate`):**
  - Enforces per-DomainPack tool allow-lists defined in `domain.yaml` (`mcp_read_tools_allowed`).
  - Canonicalizes tool names and aliases (`search_code_trigram` -> `code_repo_mcp:grep_repo`, `lookup_ast_symbol` -> `code_repo_mcp:search_ast_symbols`).
  - Unauthorized tool invocations raise `DomainToolAccessDeniedError`.
- **Content Framing & Guardrails (`guardrails.py`, `SEC-01`, `OPS-01`):**
  - `PromptInjectionSanitizer`: Detects prompt injection, delimiter breakout attacks, XML/HTML tag spoofing, and system override patterns with normalized threat scoring.
  - `SecretRedactor`: Regex-based masking for high-entropy secrets (OpenAI/Google API keys, JWT tokens, AWS credentials, RSA/EC private keys) with format-preserving redaction.
  - `frame_untrusted_content`: Wraps untrusted external data within boundary-delimited tags using cryptographically random nonces to neutralize jailbreaks.
  - `ShieldGemmaGuardrail`: Integrates with Google ShieldGemma 2 with a deterministic heuristic fallback policy ensuring resilient zero-downtime content classification.

### 6.2 Enterprise Code Repository MCP Server (`src/context_scout/mcp_servers/code_repo_mcp.py`)

Provides bounded codebase intelligence over live PostgreSQL 17 under Forced RLS:

- **5 Canonical MCP Tools:**
  1. `search_code_trigram`: `pg_trgm` similarity search over indexed repository files with Forced RLS.
  2. `lookup_ast_symbol`: Exact and prefix symbol definitions across language AST trees.
  3. `expand_ast_1hop_closure`: Bounded 1-hop caller/type dependency closure ($\le 3$ callers, $\le 1,500$ tokens).
  4. `slice_file_lines`: Byte-exact line slicing with 1-indexed validation, off-by-one detection, and git blob/commit SHA provenance.
  5. `search_github_issues`: Full-text issue search over historical bug reports and architecture proposals stored in `doc_chunks`.
- **Realistic Multi-File Trading Strategy Package (`src/context_scout/trading_strategies/`):**
  - `volatility_garch.py`: Real-time conditional volatility tracking via `GARCH11Model` with constrained quasi-MLE ($\alpha + \beta \le 0.9999$) and stationarity protection.
  - `kelly_risk_manager.py`: Continuous multi-asset position sizing via `KellyRiskManager` using Ledoit-Wolf linear covariance shrinkage with eigenvalue regularisation floors (`MIN_EIGENVALUE_FLOOR = 1e-6`).
  - `order_flow_imbalance.py`: Multi-level order book pressure calculation via `OrderFlowImbalanceSignal` (Cont-Kukanov-Stoikov 2014) with exponential depth decay and GARCH scaling.
  - `pairs_trading_kalman.py`: Cointegration hedge ratio tracking via `KalmanHedgeEstimator` implementing symmetric Joseph-form covariance updates ($(I - KH)P(I - KH)^T + KRK^T$) for numerical stability under market turbulence.
  - `almgren_chriss_execution.py`: Optimal execution trajectory optimization via `AlmgrenChrissExecutionOptimizer` modeling linear permanent and quadratic temporary market impact, with exact linear TWAP fallback ($x_j = X_0 (1 - t_j / T)$) when risk aversion $\lambda \le 0$.
- **Live Database Seeding (`seed_code_corpus.py`):**
  - 43 code files, 227 AST symbols, and 2,431 directed AST edges seeded into PostgreSQL 17 under Forced RLS.

### 6.3 Quantitative Research Corpus MCP Server (`src/context_scout/mcp_servers/quant_corpus_mcp.py`)

Provides high-precision research discovery across 14,318 genuine academic chunks:

- **4 Canonical MCP Tools:**
  1. `search_quant_papers_hybrid`: 3-way Normalized RRF search across lexical BM25, dense 768-d semantic vectors (`EmbeddingGemma 2`), and trigram matches.
  2. `extract_paper_equations_and_tables`: Lossless extraction of LaTeX mathematical equations and Markdown tables.
  3. `verify_market_friction_assumptions`: Quantitative Scepticism Gate verifying transaction cost floors ($\ge 15$ bps NSE, $\ge 10$ bps Xetra, $\ge 5$ bps US) and net Sharpe degradation hurdle ($\le 0.50$ fails after costs).
  4. `compare_cross_market_regimes`: Comparative cross-market structural and regulatory analysis across Indian, German, and US markets.
- **Scaled Corpus Discovery:**
  - Full-text access across 14,318 genuine academic chunks from 300 full-text papers (`IN_NSE`, `DE_XETRA`, `US_EQUITY`).

### 6.4 Hard Payload Bounds & Truncation (`common.py`, `SEC-03`)

- **Item Clamping:** Clamps collection return payloads to at most 10 items (`MAX_PAYLOAD_ITEMS = 10`).
- **32 KB JSON Cap:** Serialized JSON payload capped at 32,768 bytes (`MAX_PAYLOAD_BYTES = 32_768`).
- **Truncation Marker:** When payload exceeds cap, trailing items are truncated and a deterministic marker `[PAYLOAD_TRUNCATED_AT_32KB]` is attached to the output payload.

---

## 7. Milestone 5 Architecture: Google ADK 2.12 Scout, 4-Check Pre-Flight Gate & Paid Gemini Worktree Handoff

Milestone 5 unifies the local open-weights discovery stack with production-grade Google ADK (`google-adk>=2.12.0`) workflows, enforcing a strict 4-check deterministic pre-flight gate that guarantees zero paid token waste on exploratory context discovery, coupled with an isolated git worktree sandbox and human confirmation gate.

### 7.1 4-Check Pre-Flight Quality Gate (`src/context_scout/agents/preflight_gate.py`)

To eliminate the 80–90% paid token wastage observed in naive agent setups, the `PreFlightValidator` acts as a zero-cost deterministic gate before any paid Gemini API call is executed:

1. **Check 1 (100% Internal Symbol Resolution):**
   - Parses AST symbol references across all candidate `code_slices`.
   - Rejects any pack containing ungrounded internal function, method, or class references.
   - Automatically executes 1-hop AST dependency closure expansion to auto-attach missing caller and type definition slices up to `MAX_EXPANSION_TOKEN_BUDGET = 1500`.
2. **Check 2 (Zero Novel Target Identifiers, `VAL-01`):**
   - Cross-checks all target modification symbols against the indexed AST symbol graph.
   - Prevents hallucinations targeting non-existent modules or symbols before code generation commences.
3. **Check 3 (Normalized RRF Floor, `BLK-01`):**
   - Enforces a statutory retrieval quality floor: requires highest normalized 3-way RRF score $\ge 0.55$.
   - Rejects low-confidence context packs with weak lexical/semantic grounding.
4. **Check 4 (100% Verbatim Citation Verification, `VAL-02`):**
   - Audits every academic paper citation against indexed `doc_chunks` using `DeterministicSlicer` with NFKC normalization ($\ge 0.98$ overlap required).
   - Rejects hallucinated quotes, misattributed claims, or tampered equation citations.

### 7.2 $0 Local Bounce-Back Loop & Context Flywheel

When any pre-flight check fails:
- The system returns `PreFlightValidationReport(passed=False)` with structured diagnostic feedback (`bounce_back_feedback`).
- Generates `InsufficientContextDiagnostic(paid_tokens_spent=0)` locally. Zero paid tokens are spent.
- Triggers up to 2 local self-correction turns with `LocalGemmaClient` at $0 API cost, searching for missing symbols or refining retrieval queries.
- Records `MissingContextRequest` flywheel events, capturing context miss patterns to inform future QLoRA distillation datasets.

### 7.3 Local Gemma Discovery Engine & Multi-Domain ADK Workflow (`src/context_scout/agents/scout_workflow.py`)

- **Local Gemma Client (`LocalGemmaClient`, `HW-01`, `ARC-02`):**
  - Targets `google/gemma-4-26B-A4B-it` (MoE: 25.2B total / 3.8B active parameters, 256K context) at $0 token cost.
  - Enforces strict 32,768-token KV cache clamping (`HW-01`) preventing out-of-memory errors on Apple Silicon MPS and local GPUs.
  - Dynamic in-memory LoRA adapter hot-swapping (`ARC-02`) via `load_lora_adapter` / `unload_lora_adapter` without full model reload.
- **4-Stage ADK Scout Workflow (`ScoutWorkflow`):**
  - **Stage 1 (Discovery):** Local Gemma queries enterprise MCP tools (`code_repo_mcp`, `quant_corpus_mcp`) across live PostgreSQL 17 under Forced RLS.
  - **Stage 2 (Expansion):** Bounded 1-hop AST closure expands caller, type, and test dependencies.
  - **Stage 3 (Assembly):** Assembles a compact, typed `ImplementationContextPack` containing byte-exact slices, verified citations, and provenance metadata.
  - **Stage 4 (Pre-Flight Validation):** `PreFlightValidator` verifies all 4 quality gates.
- **Federated Multi-Domain DAG Coordination (`FederatedScoutCoordinator`, `ARC-01`):**
  - Orchestrates multi-domain requests spanning `trading_code_scout` and `quant_research_scout`.
  - Dispatches domain scouts concurrently or sequentially, performs score-normalized RRF fusion, and produces a unified context pack with deduplicated AST slices and academic citations.

### 7.4 Paid Gemini Implementation Executor & Git Worktree Sandbox (`src/context_scout/agents/gemini_executor.py`)

- **Google Model Policy & Hard Budget Enforcement:**
  - Allowed execution models: **Google `gemini-3.8-flash`** ($0.75/$3.75 promo) for fast execution, or **`gemini-4-argon`** ($2.00–$4.00/$10.00–$20.00 std) for complex frontier reasoning.
  - Legacy 2025 models (`gemini-2.5-pro`, `gemini-2.5-flash`) are strictly rejected.
  - Hard per-turn budget caps checked via `max_budget_usd`. If projected token cost exceeds budget, execution halts before calling the model API.
- **Isolated Ephemeral Git Worktree Sandbox (`GitWorktreeSandbox`):**
  - Creates an isolated ephemeral git worktree on an ephemeral branch (`scout-exec-<uuid>`) off `repo_root`.
  - Applies synthesized unified diff patches using `git apply --whitespace=fix`.
  - Runs automated verification commands (`ruff check`, `pytest`) within the isolated worktree directory.
  - **Human Confirmation Gate:** Requires explicit human sign-off (`confirm_and_merge(confirmed=True)`) before committing. When declined (`confirmed=False`), changes are aborted with zero residual disk pollution.
  - Automatic worktree cleanup removes worktree directories and branches upon context manager exit.

### 7.5 Institutional ROI & Execution Telemetry

- **Live PostgreSQL 17 Telemetry (`execution_telemetry` table):**
  - Records every scout execution: `trace_id`, `domain_id`, `discovery_tokens_saved`, `paid_tokens_spent`, `net_dollar_savings`, `preflight_passed`, and `human_confirmed`.
  - Empirically verifies $>80\%$ token savings over brute-force frontier queries:
    $$\text{Savings \%} = \frac{\text{discovery\_tokens\_saved}}{\text{discovery\_tokens\_saved} + \text{paid\_tokens\_spent}} \times 100 \ge 80\%$$

---

## 8. Milestone 6 Architecture: Golden Dataset, CI/CD Evaluation Harness & OpenTelemetry Observability

Milestone 6 establishes automated, repeatable evaluation gates that enforce production readiness and zero-regression deployment standards across retrieval accuracy, verbatim faithfulness, token economics, security isolation, and adversarial guardrails.

### 8.1 Typed Golden Benchmark Dataset (`src/context_scout/evals/golden_dataset.py`)

Evaluation is driven by a typed golden benchmark suite built using Pydantic v2 models:
- **`GoldenCodeTask` (5 Tasks):** Real-world code localization tasks across the quantitative trading strategies (`volatility_garch`, `kelly_risk_manager`, `order_flow_imbalance`, `pairs_trading_kalman`, `almgren_chriss_execution`). Statically validates target files, target AST classes/functions, required context terms, and negative distraction keywords.
- **`GoldenQuantTask` (3 Tasks):** Multi-market literature retrieval tasks covering `IN_NSE` liquidity/impact, `DE_XETRA` volatility/microstructure, and `US_EQUITY` friction/regulations. Statically validates expected paper IDs, section titles, and verbatim equation/table quotes.
- **`GoldenFederatedTask` (2 Tasks):** End-to-end multi-domain tasks requiring simultaneous AST code slicing and academic literature grounding.
- **`RedTeamRBACQuery` (22 Queries):** Cross-role adversarial permission probes (e.g., `junior_dev` querying production risk parameters, `restricted_quant` probing non-granted market scopes, parameter injection bypass attempts).
- **`PoisonedInjectionPayload` (22 Payloads):** Adversarial prompt-injection attack vectors spanning delimiter escapes, system prompt overrides, LaTeX steganography, Unicode/bidi directionality overrides, and secret extraction probes.

### 8.2 6 CI/CD Evaluation Gates & Measured Results (`src/context_scout/evals/harness.py`)

The `EvaluationHarness` executes all benchmark tasks against live PostgreSQL 17 (under Forced RLS) and in-memory repositories, evaluating the 6 production gates:

| CI/CD Gate Metric | Description & Method | Threshold | Measured Result | Status |
|:---|:---|:---:|:---:|:---:|
| **Zero-Touch Sufficiency Rate** | Fraction of tasks where Local Gemma + Pre-Flight Gate resolves full context without requiring missing-context fallback turns | $\ge 85.0\%$ | **100.0%** | ✅ **PASS** |
| **File & Symbol Recall@5** | Fraction of golden target files and AST symbols present in the top-5 retrieved `CodeSlicePointer` items | $\ge 90.0\%$ | **100.0%** | ✅ **PASS** |
| **Verbatim Citation Faithfulness** | NFKC-normalized character/token overlap ($\ge 0.98$) of cited slices against immutable Git blobs | $== 100.0\%$ | **100.0%** | ✅ **PASS** |
| **Frontier Token Savings Ratio** | Reduction in paid frontier model input tokens relative to ungrounded raw codebase dumps ($\frac{\text{saved}}{\text{saved} + \text{paid}}$) | $\ge 80.0\%$ | **80.0%** | ✅ **PASS** |
| **RBAC Zero-Leakage Rate** | Percentage of unauthorized cross-tenant/cross-role queries yielding exactly 0 records under PostgreSQL 17 `FORCE RLS` | $== 100.0\%$ | **100.0%** | ✅ **PASS** |
| **Prompt-Injection Block Rate** | Percentage of adversarial prompt-injection payloads detected and blocked by `PromptInjectionSanitizer` | $\ge 95.0\%$ | **95.45%** | ✅ **PASS** |

### 8.3 OpenTelemetry GenAI Semantic Instrumentation (`src/context_scout/telemetry.py`)

The evaluation harness integrates natively with OpenTelemetry (`opentelemetry-api>=1.30.0` and `opentelemetry-sdk>=1.30.0`), emitting standardized traces conforming to the **OpenTelemetry GenAI Semantic Conventions**:
- **Conventions Enforced:** `gen_ai.system` (`gemma`, `gemini`), `gen_ai.request.model` (`gemma-4-26B-A4B-it`, `gemini-3.8-flash`, `gemini-4-argon`), `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, and `gen_ai.operation.name` (`workflow`, `retrieval`, `preflight_gate`).
- **Span Capture:** Instruments all scout workflow stages, tool queries, and validation passes. Operates with an `InMemorySpanExporter` in CI/CD pipelines to assert telemetry presence without external collectors.

### 8.4 Automated CLI Evaluation Runner (`src/context_scout/evals/runner.py`)

The evaluation suite is executable via CLI for automated CI/CD gating:
```bash
# Run full evaluation suite against live PostgreSQL 17 and enforce quality gates
uv run python -m context_scout.evals.runner \
  --postgres \
  --output-json reports/m6_eval_report.json \
  --output-md reports/m6_eval_report.md \
  --fail-under-threshold
```
- **Exit Code 0:** All 6 gates meet or exceed thresholds; artifact scorecards written to `reports/`.
- **Exit Code 1:** One or more gates fail threshold; blocks CI/CD merge request.

---

## 9. Milestone 7 Architecture: Teacher-Student Distillation, 1-Base-Model + Multi-LoRA Hot-Swapping & Custom Domain CLI

Milestone 7 delivers the offline Teacher-Student distillation pipeline, genuine 4-bit RAG-conditioned QLoRA fine-tuning for **`trading_code_scout`** and **`quant_research_scout`**, an in-memory multi-adapter pool over a single frozen base model, and a 3-command CLI so any developer can train a specialized agent on their own codebase or document corpus.

### 9.1 How the 1-Base-Model + Multi-Adapter Architecture Runs Underneath (`AdapterPool` & `LocalGemmaClient`)

Instead of running two separate 26B models (which would consume $>31\text{ GB}$ of memory and force slow disk reloads), the platform loads **only 1 frozen base model** (`google/gemma-4-26B-A4B-it`, `15.6 GB` in 4-bit quantization) and hot-swaps lightweight domain QLoRA adapters in memory:

- **Single Shared Base Model (`15.6 GB`):** `google/gemma-4-26B-A4B-it` (MoE: 25.2B total / 3.8B active parameters per token) stays resident once in unified/GPU memory.
- **2 Lightweight Domain QLoRA Adapters (`9.97 MB` each, `54` tensors):**
  1. **Adapter 1 — `gemma4_26b_code_scout` (`domains/trading_code_scout`):** Trained on AST symbol graphs, 1-hop dependency closures, and repository file slices to emit exact `CodeSlicePointer` JSON objects.
  2. **Adapter 2 — `gemma4_26b_quant_research` (`domains/quant_research_scout`):** Trained on retrieved academic paper chunks and market friction rules to emit verified `QuantStrategyBrief` and `DocCitationSpan` JSON objects.
- **Sub-Millisecond Hot-Swapping (`0.018 ms`):** `AdapterPool` pins all registered domain adapters (`9.97 MB` each) in memory. When a request targets `trading_code_scout`, `quant_research_scout`, or `federated_all` (which runs both sequentially), `LocalGemmaClient.load_adapter()` switches the active low-rank pointer ($h = W_0 x + \frac{\alpha}{r} B A x$) in **0.018 ms** without reloading `W_0`.

### 9.2 Distillation Pipeline with Rejection Sampling & Bounce-Back Recovery (`src/context_scout/tuning/teacher_distiller.py`)

- **Introspection over Live PostgreSQL 17 Corpus:**
  - Connects to live tables: 91 `code_files`, 841 `ast_symbols`, 11,022 `ast_edges`, and 14,323 `doc_chunks`.
  - Supports `gemini-3.8-flash` / `google/gemma-4-31B-it` Teacher execution with deterministic AST/quote synthesis fallback for reproducible offline execution without API dependencies.
- **RAG-Conditioned Prompts (Never Memorizing Dynamic Facts):**
  - Every training prompt embeds the retrieved candidate context (`[RETRIEVED_AST_CANDIDATES]` or `[RETRIEVED_DOC_CHUNKS]`), training the adapter to extract grounded symbols, line spans, and verbatim quotes from retrieved context rather than memorizing static repository text.
- **Deterministic Rejection Sampling Gate (`PreFlightValidator`):**
  - Every Teacher-generated trajectory must pass the 4-check pre-flight gate before admission to the dataset; rejects any hallucinated symbol, out-of-bounds line range, or unverified citation.
- **ML-01 15–20% Bounce-Back Recovery Trajectories:**
  - Multi-turn trajectories where Turn 1 emits an incomplete draft, Turn 2 delivers `PreFlightValidator` bounce-back feedback detailing missing internal dependencies or unverified quotes, and Turn 3 performs targeted remediation.
- **Zero-Leakage Stratified 85/15 Splits:**
  - Stratified by repository module prefix for `trading_code_scout` and by paper `doc_id` for `quant_research_scout`, with 0% overlap between `train.jsonl` and `valid.jsonl`.

### 9.3 Runtime Hardware Auto-Detection & Memory Math (`src/context_scout/config.py` & `train_mlx_lora.py`)

- **Zero Hardware Hardcoding (`detect_hardware_profile()`):**
  - At startup, `detect_hardware_profile()` automatically queries the host OS (`sysctl` on macOS, `torch.cuda` / `/proc/meminfo` on Linux) to detect CPU/GPU model, physical RAM/VRAM GB, quantization backend (`4-bit (MLX)`, `4-bit (CUDA)`, or `4-bit (PyTorch)`), and dynamically sets a safe `max_kv_tokens` ceiling (`8,192` for $<24\text{ GB}$, `16,384` for $24\text{–}35\text{ GB}$, `32,768` for $\ge 36\text{ GB}$).
  - Supports explicit `.env` overrides (`SCOUT_HARDWARE_DEVICE`, `SCOUT_HARDWARE_MEMORY_GB`, `SCOUT_LOCAL_MAX_KV_TOKENS`) for containerized or custom deployments.
- **4-bit QLoRA Memory Budget (`gemma-4-26B-A4B-it`):**
  - 4-bit base weights: **13.9–15.6 GB**
  - LoRA adapter parameters ($r=16, \alpha=32$): **9.97 MB** (`0.24 GB` with buffers)
  - AdamW 16-bit optimizer states for adapter weights: **0.96 GB**
  - Dynamic activation budget (batch size 1, sequence length 2,048, `--grad-checkpoint`): **4.5 GB**
  - **Active Peak Training Memory:** **~18.4–19.6 GB**.
- **HW-01 Process Mutual Exclusion Lock (`GPUMutualExclusionLock`):**
  - Enforces a POSIX non-blocking file lock (`fcntl.flock` on `/tmp/context_scout_gpu.lock`) preventing simultaneous GPU execution between offline QLoRA training and inference serving.

### 9.4 MoE LoRA Targeting, Strict Router Gate Freezing & 2-Epoch Training Metrics

- Gemma 4 26B-A4B features 128 Mixture-of-Experts (MoE) routing layers. Fine-tuning router gate projections on domain datasets causes expert routing collapse.
- `LocalQLoRATrainer` targets shared attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and shared MLP projections (`gate_proj`, `up_proj`, `down_proj`) with scaling $\alpha / r = 32 / 16 = 2.0$, while **strictly freezing all 128 MoE router gates** (`router.proj`).
- **2-Epoch RAG-Conditioned QLoRA Training Results (`mlx-lm` on `mlx-community/gemma-4-26b-a4b-it-4bit`):**
  - **`trading_code_scout` (`gemma4_26b_code_scout`):** `340` RAG-conditioned train samples (`2` epochs = `680` steps), `train_loss: 1.3588 -> 0.0008`, `val_loss: 0.9424 -> 0.0027`, **100% (3/3)** live `mlx_lm.generate()` JSON schema and exact AST symbol/line-span grounding.
  - **`quant_research_scout` (`gemma4_26b_quant_research`):** `334` RAG-conditioned train samples (`2` epochs = `668` steps), `train_loss: 1.4734 -> 0.0001`, `val_loss: 1.2806 -> 0.0212`, **100% (3/3)** live `mlx_lm.generate()` JSON schema and exact `chunk_id` / `verbatim_quote` grounding across single-turn and 4-turn bounce-back validation samples.

### 9.5 Enterprise GCP Vertex AI PEFT Trainer (`src/context_scout/tuning/train_vertex_peft.py`)

- Generates Hugging Face `peft.LoraConfig(r=16, lora_alpha=32)` and GCP Vertex AI `CustomJob` resource specifications (`machineSpec: g2-standard-12`, `acceleratorType: NVIDIA_L4`).
- Supports dry-run validation verifying VRAM headroom ($>2.0\text{ GB}$) and job execution without cloud billing risk.

### 9.6 Empirical Before-vs-After Benchmark Results (`reports/m7_qlora_benchmark_report.md`)

Measured across 36 held-out test cases comparing base Gemma 4-26B vs. base Gemma 4-26B + trained QLoRA adapters:

| Benchmark Metric | Base Model (Before) | Tuned QLoRA (After) | Gain / Reduction | Gate Threshold | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pydantic Schema Adherence (Zero-Shot)** | `19.44%` | **`100.0%`** | `+80.56%` | $\ge 95.0\%$ | ✅ PASS |
| **Pydantic Schema Adherence (Few-Shot)** | `69.44%` | **`100.0%`** (0-shot) | `+30.56%` | $\ge 95.0\%$ | ✅ PASS |
| **AST Skeleton Localization Recall@5** | `8.33%` | **`100.0%`** | `+91.67%` | $\ge 90.0\%$ | ✅ PASS |
| **Prompt Token Overhead per Call** | `1690.6 tokens` | `56.6 tokens` | **`-1634.0 tokens`** (`-96.65%`) | `1500–2000 tokens` | ✅ PASS |
| **Prompt Prefill Latency (TTFT)** | `120.76 ms` | `4.04 ms` | `29.89x speedup` | $< 50\text{ ms}$ | ✅ PASS |
| **In-Memory Adapter Switch Latency** | `N/A (Disk I/O)` | `0.018 ms` | `< 2 ms (ARC-02)` | $< 5.0\text{ ms}$ | ✅ PASS |

### 9.7 Training Your Own Custom Domain Agent in 3 Commands (`uv run context-scout`)

Anyone cloning this repository can create, distill, and fine-tune a brand-new specialized agent on their own codebase or document folder using the built-in `context-scout` CLI (`src/context_scout/cli.py`):

```bash
# 0. Inspect auto-detected host hardware, quantization backend, and safe KV-cache budget
uv run context-scout hardware

# 1. Scaffold a new DomainPack (choose --domain-type code or research)
uv run context-scout init-domain \
  --domain-id payments_ledger_scout \
  --domain-type code \
  --display-name "Payments Ledger Code Scout" \
  --source-dir /path/to/your/codebase

# 2. Distill RAG-conditioned JSONL training & validation sets from your folder
uv run context-scout distill \
  --domain-id payments_ledger_scout \
  --source-dir /path/to/your/codebase \
  --num-samples 200

# 3. Train the rank-16 QLoRA adapter (auto-registers in domains/<domain_id>/domain.yaml)
uv run context-scout train \
  --domain-id payments_ledger_scout \
  --iters 400
```

Once trained, restart the FastAPI server and your new domain automatically appears in `/api/v1/domains` and the React UI `DomainSelector` alongside existing domains.

---

## 10. Milestone 8 Architecture: FastAPI SSE Backend, Strict TypeScript/React UI & GitHub Actions CI/CD

Milestone 8 completes the production enterprise operational layer, delivering a low-latency FastAPI SSE streaming backend, a strict TypeScript/React web interface, and an automated 6-stage GitHub Actions CI/CD pipeline with keyless Workload Identity Federation (WIF).

### 10.1 FastAPI SSE Streaming Backend (`src/context_scout/api/`)

The backend is structured around an ASGI application factory (`create_app() -> FastAPI` in `src/context_scout/api/main.py`), CORS middleware, OpenTelemetry HTTP request tracing, and out-of-band principal extraction:

- **Out-of-Band Principal Extraction (`dependencies.py`, `SEC-04`):**
  - Session authentication strictly decouples user identity from LLM prompt inputs. `X-Principal-User-Id`, `X-Principal-Role` (`engineer`, `quant_researcher`, `product_manager`, `compliance_officer`), and `X-Principal-Scopes` are validated and bound to `contextvars.ContextVar` out-of-band.
  - Role-spoofing text in prompt inputs cannot alter the server-side `PrincipalContext`. Invalid roles return `HTTP 400`.
- **7 Production API Endpoints:**
  1. `GET /api/v1/health`: Returns service health, database connectivity check, registered domain IDs, active `AdapterPool` verification status, and auto-detected host hardware memory profile (`detect_hardware_profile()`).
  2. `GET /api/v1/domains`: Returns registered `DomainPackManifest` configs (`trading_code_scout`, `quant_research_scout`, plus any custom domains), live adapter verification (`r=16, alpha=32`, `is_moe=True`, `safetensors_bytes > 0`), and training metrics.
  3. `POST /api/v1/scout/stream`: Server-Sent Events (`text/event-stream`) streaming 4 sequential stage updates (`Stage 1: Hybrid RRF`, `Stage 2: Gemma 4 Synthesis`, `Stage 3: AST Closure / Friction Gate`, `Stage 4: Pre-Flight Validation`), `$0` bounce-back events, and verified `context_pack_ready` event.
  4. `POST /api/v1/scout/query`: Synchronous JSON execution through `ScoutWorkflow` or `FederatedScoutCoordinator`, returning full verified `ImplementationContextPack`, `PreFlightValidationReport`, OpenTelemetry trace attributes, and token savings breakdown.
  5. `POST /api/v1/handoff/preview`: Invokes `GeminiImplementationExecutor` in preview mode (`require_confirmation=True`), generating proposed unified diff, cost telemetry, and confirmation token without mutating repository files.
  6. `POST /api/v1/handoff/apply`: Enforces explicit human confirmation (`confirmed: True` + valid token) before applying patch in `GitWorktreeSandbox`, invalidating token upon consumption.
  7. `GET /api/v1/telemetry/roi`: Aggregates ROI token savings and loads Milestone 6 evaluation report (`reports/m6_eval_report.json`) and Milestone 7 QLoRA benchmark report (`reports/m7_qlora_benchmark_report.json`).

### 10.2 Strict TypeScript + React Enterprise UI (`web-ui/`)

Located in `web-ui/`, built on Vite + React 18 + strict TypeScript (`"strict": true`, `"noImplicitAny": true`, `"strictNullChecks": true`, zero `any` across codebase):

- **Contracts (`src/types/contracts.ts`):** Strictly typed interfaces mirroring Python Pydantic v2 schemas (`RoleType`, `PrincipalContext`, `CodeSlicePointer`, `DocCitationSpan`, `QuantStrategyBrief`, `ContextPack`, `SSEStageEvent`, `HandoffPreviewResponse`, `ROIMetricsResponse`).
- **Enterprise Components:**
  - `RoleSwitcher.tsx`: Interactive role switcher displaying active RBAC/RLS scopes (`public_eng`, `repo:trading_strategies`, `public_quant`, `quant_restricted`, `audit:compliance`).
  - `DomainSelector.tsx`: DomainPack and QLoRA adapter selector showing adapter status, hot-swap latency (`0.018 ms`), and training loss metrics.
  - `ScoutConsole.tsx`: Query console with 4-stage stepper, execution mode toggle (SSE Stream vs. REST Sync), and ShieldGemma 2 guardrail badge.
  - `CitationInspector.tsx`: Tabbed inspector for byte-exact `CodeSlicePointer` slices and `QuantStrategyBrief` academic citations with friction floor badges (`FIN-01`).
  - `TokenSavingsMeter.tsx`: Real-time token-savings meter showing local Gemma 4 tokens explored (`$0.00`), frontier tokens sent, and frontier token savings ratio.
  - `GeminiDiffModal.tsx`: 1-Click implementation handoff modal displaying proposed unified git diff and enforcing human approval before applying to sandbox.
- **Frontend Verification:** 41/41 component and integration tests passing in Vitest (`npm run test`); production build verified (`npm run build`).

### 10.3 Enterprise GitHub Actions CI/CD Pipeline (`.github/workflows/ci.yml`)

The production `.github/workflows/ci.yml` pipeline defines 6 ordered jobs on GitHub Actions:
1. `lint`: Ruff syntax/style check (`uv run ruff check src tests`) and prohibited terms repository hygiene scanner (`! git grep -i -E ...`).
2. `typecheck`: Strict static typechecking (`uv run mypy --strict src tests`).
3. `test`: Live PostgreSQL 17 + `pgvector` container service integration tests with automated schema bootstrap (`sql/001_init_schema.sql`) and Forced RLS policy initialization.
4. `eval-gate`: Automated 6-gate architectural evaluation runner enforcing Zero-Touch Sufficiency $\ge 85\%$, Recall@5 $\ge 90\%$, Citation Faithfulness $100\%$, Token Savings $\ge 80\%$, RBAC Zero-Leakage $100\%$, and Prompt Injection Block $\ge 95\%$.
5. `ui-verify`: Node.js 22 runner executing `npm ci`, `npm run lint`, `npm run typecheck`, `npm run test`, and `npm run build`.
6. `deploy`: Keyless Google Cloud Workload Identity Federation (WIF) OIDC deployment to Google Cloud Run (no static service account JSON keys, gated on `main` branch with manual dispatch).

---

## 11. Local Setup & Installation

### 11.1 Prerequisites

- **macOS** (Apple Silicon M-series recommended for MLX 4-bit inference/training) or **Linux** (CUDA / CPU) workstation.
- **Python 3.12+** with `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **Node.js 20+** and `npm` for web UI.
- **PostgreSQL 17** with `pgvector` and `pg_trgm` extensions (via **Docker Desktop** or **Homebrew**):
  ```bash
  # Option A: Docker Compose (runs pgvector/pgvector:pg17 on host port 54329)
  docker compose up -d postgres

  # Option B: macOS Homebrew (runs on port 5432)
  brew install postgresql@17 pgvector
  brew services start postgresql@17
  ```

### 11.2 Environment & Database Setup

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/himanshu2408/enterprise-context-scout.git
   cd enterprise-context-scout
   uv sync --extra dev
   cd web-ui && npm install && cd ..
   ```

2. **Initialize PostgreSQL database (if using local Homebrew PostgreSQL on `:5432`; Docker auto-initializes `sql/001_init_schema.sql`):**
   ```bash
   createdb context_scout
   psql context_scout -f sql/001_init_schema.sql
   ```

3. **Configure environment variables (`.env`):**
   ```bash
   cp .env.example .env
   ```
   - Set `SCOUT_DB_ADMIN_URL` and `SCOUT_DB_APP_URL` (use port `5432` for local Homebrew Postgres or port `54329` for Docker Compose).
   - Hardware profile (`device`, `memory_gb`, `max_kv_tokens`) is auto-detected at runtime (`uv run context-scout hardware`), and can be optionally overridden via `SCOUT_HARDWARE_DEVICE`, `SCOUT_HARDWARE_MEMORY_GB`, and `SCOUT_LOCAL_MAX_KV_TOKENS`.

4. **Seed Corpus & Launch Application Servers:**
   ```bash
   # Seed code AST graph and quant paper chunks into PostgreSQL 17
   uv run python -m context_scout.mcp_servers.seed_code_corpus

   # Start FastAPI SSE Backend (port 8080)
   uv run uvicorn context_scout.api.main:app --host 127.0.0.1 --port 8080 --reload

   # Start Vite React Web UI (port 5173, proxies /api -> http://127.0.0.1:8080)
   cd web-ui && npm run dev
   ```

---

## 12. Verification & Test Suite

The platform enforces zero-trust automated testing across all backend, CLI, worktree feature-build, and frontend components:

```bash
# 1. Format & Linting Check (Ruff)
uv run ruff check src tests

# 2. Strict Static Type Checking (Mypy)
uv run mypy --strict src tests

# 3. Comprehensive Unit, Live Postgres, Regression & E2E Feature-Build Suite (Pytest - 536 tests)
uv run pytest -v

# 4. End-to-End Worktree Feature Build & Custom Domain CLI Training Verification
uv run pytest -v tests/test_end_to_end_feature_build.py

# 5. CI/CD Evaluation Harness & Gate Scorecard (6 architectural gates)
uv run python -m context_scout.evals.runner --in-memory

# 6. Milestone 7 Empirical QLoRA Benchmark Suite
uv run python -m context_scout.tuning.benchmark

# 7. Web UI Lint, Typecheck, Test (41 Vitest tests), and Build
cd web-ui
npm run lint
npm run typecheck
npm run test
npm run build
cd ..
```

### Test Coverage Highlights (536 Passed Python Tests + 41 Passed Web UI Tests, 0 Failures, 0 Skips)

| Subsystem | Test Module | Tests | Scope & Invariants Tested |
|:---|:---|:---:|:---|
| **M1 Foundation** | `tests/test_m1_foundation.py` | 7 | Pydantic v2 contracts, serialization, and adapter interfaces |
| **M1 Postgres RLS** | `tests/test_postgres_rls_live.py` | 4 | Forced Row-Level Security, scope isolation, and tenant cross-access denial |
| **M2 AST Engine** | `tests/test_m2_code_engine.py` | 5 | Tree-sitter symbol extraction, call graph resolution, and error tolerance |
| **M2 Expander** | `tests/test_dependency_expander.py` | 7 | Bounded 1-hop closure, token budgeting ($\le 1,500$), and diversity selection |
| **M2 Slicer** | `tests/test_deterministic_slicer.py` | 8 | Byte-exact slicing, line range verification, and NFKC normalization |
| **M2 Postgres Live** | `tests/test_postgres_m2_live.py` | 2 | Live PostgreSQL Merkle tree diffing and database persistence |
| **M2 Security** | `tests/test_adversarial_ast_rls.py` | 7 | Security boundaries, malicious scope injection, and cycle handling |
| **M2 Performance** | `tests/test_empirical_merkle_slicer_challenge.py` | 14 | Performance assertions ($<50\text{ ms}$ Merkle sync, $\ge 0.98$ NFKC quote overlap) |
| **M3 Quant Engine** | `tests/test_m3_quant_engine.py` | 13 | Paper harvester, parser, contextualizer, and in-memory 3-way RRF tests |
| **M3 Postgres Live** | `tests/test_postgres_m3_live.py` | 2 | Live PostgreSQL 17 RLS document chunk storage and 3-way hybrid RRF |
| **M3 Adversarial Harvester** | `tests/test_adversarial_m3_chunker_harvester.py` | 17 | Nested LaTeX table float integrity, Markdown whitespace, non-ISO dates |
| **M3 Adversarial RRF & Gate** | `tests/test_adversarial_m3_rrf_scepticism.py` | 24 | RRF mathematical peak invariant, multi-market friction floors, citation tampering |
| **M3 Challenger Harvester** | `tests/test_empirical_m3_harvester_parser_challenge.py` | 19 | Empirical full-text LaTeX/PDF validation, 5 arXiv categories, 0 .txt stubs |
| **M3 Challenger Verification** | `tests/test_empirical_m3_challenger_verification.py` | 23 | Live RLS isolation on $\ge 1,000$ chunks, 3-way RRF fusion, negative topic rejection |
| **M3 Live Harvest & Ingestion** | `tests/test_real_paper_harvest_live.py` | 6 | Live arXiv tar decompression, PDF fallback, OpenAlex v2, DB persistence & Scepticism Gate |
| **M4 Enterprise MCP Servers** | `tests/test_m4_mcp_servers.py` | 18 | Live PostgreSQL 17 Forced RLS, canonical tools, payload bounds ($\le 32$ KB), strategy queries |
| **M4 Security Layer** | `tests/test_m4_security.py` | 37 | Out-of-band principal binding, domain authorization gate, prompt injection & secret guardrails |
| **M4 Adversarial Security** | `tests/test_adversarial_security.py` | 47 | Adversarial parameter injection fuzzing, connection pool isolation, NUL byte detection, TWAP fallback |
| **M5 Pre-Flight Gate** | `tests/test_preflight_gate.py` | 10 | 4-Check pre-flight quality gate, symbol auto-attachment, and $0 bounce-back tests |
| **M5 ADK Scout Workflow** | `tests/test_scout_workflow.py` | 13 | ADK 4-stage workflow DAG, self-correction, and federated coordinator tests |
| **M5 Gemini Executor** | `tests/test_gemini_executor.py` | 10 | Gemini model policy, budget gate, and git worktree sandbox tests |
| **M5 End-to-End Integration** | `tests/test_m5_adk_workflow.py` | 17 | End-to-end integration, $0 bounce-back, and live PostgreSQL 17 telemetry tests |
| **M6 Eval Harness & OTel** | `tests/test_m6_eval_harness.py` | 13 | Golden dataset, 6 CI/CD gates, OTel GenAI traces & live PG17 (100% pass) |
| **M7 Teacher Distillation** | `tests/test_m7_teacher_distiller.py` | 6 | PostgreSQL 17 introspection, rejection sampling, 15-20% bounce-back, 0% leakage |
| **M7 QLoRA & Benchmarks** | `tests/test_m7_qlora_tuning.py` | 17 | Model acquisition, memory math, HW-01 lock, router gate freezing, Vertex PEFT, benchmark |
| **M8 FastAPI SSE & CI/CD** | `tests/test_m8_fastapi_sse.py` | 25 | Health, domains, RBAC roles, ShieldGemma, SSE stream, handoff preview/apply, ROI telemetry, repo hygiene |
| **Enterprise Regression Guards** | `tests/test_enterprise_regression_guards.py` | 14 | Permanent regression tests for RAG-conditioned QLoRA, Vite proxy, CI markers, hardware auto-detection |
| **End-to-End Feature Build & CLI** | `tests/test_end_to_end_feature_build.py` | 2 | Live worktree feature synthesis + behavioral test & 3-command custom-domain CLI training |
| **Additional Challenging Suites** | `tests/test_adversarial_*.py` / `test_empirical_*.py` | 139 | Deep adversarial & empirical stress suites across M1–M8 |
| **Python Backend Total** | **31 Test Suites** | **536** | **100% Pass Rate (0 Failures, 0 Skips, 0 Errors)** |
| **Web UI Total (Vitest)** | `web-ui/src/__tests__/*.test.tsx` | **41** | **100% Pass Rate (9 Test Suites, 0 Failures)** |

---

## 13. Security Invariants & Repository Protection

- **Confidentiality:** No proprietary client names, credentials, or internal metrics are permitted in code, comments, or commit history.
- **Git Hook Protection:** Local git hooks (`.git/hooks/pre-commit` and `.git/hooks/commit-msg`) enforce regex-based secret scanning, path checks, and word-bounded term blocking on every commit.
- **Model Truthfulness:** All model specifications, pricing, and benchmark claims are verified against official documentation and live search gates.

