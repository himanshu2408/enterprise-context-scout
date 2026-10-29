# STATUS: Enterprise Context Scout & Multi-Domain Platform

**Last Updated:** 2026-10-29  
**Active Branch:** `feat/m4-mcp-servers`  
**Repository:** `https://github.com/himanshu2408/enterprise-context-scout`

## 1. Milestone Progress Dashboard

| Milestone | Scope | Status | Verification Gate |
| :--- | :--- | :---: | :--- |
| **M0: Architecture & Audit** | `ARCHITECTURE_BLUEPRINT.md`, `ARCHITECTURE_AUDIT_REPORT.md`, `AGENTS.md`, Pre-commit Security Hook | **COMPLETE** | 12/12 Audit Remediations Applied (`BLK-01`, `SEC-01..04`, `PERF-01..02`, `ARC-01..02`, `HW-01`, `FIN-01`) |
| **M1: Foundation, Postgres 17 RLS & Contracts** | `pyproject.toml`, `docker-compose.yml`, `sql/001_init_schema.sql`, Pydantic v2 Contracts, `DomainRegistry`, `AdapterPool`, RLS Repositories | **COMPLETE** | `uv run ruff check` + `uv run mypy --strict src` + `uv run pytest tests/test_m1_foundation.py` |
| **M2: Code Engine (AST + Merkle Sync + Slicer)** | Git Blob-SHA Merkle indexer, `tree-sitter` AST symbol/edge extractor, bounded 1-hop `ASTDependencyExpander` ($\le 3$ callers, $\le 1,500$ tokens), `DeterministicSlicer` | **COMPLETE** | `uv run ruff check` + `uv run mypy --strict src tests` + `uv run pytest -v` (54/54 passing) |
| **M3: Quant Paper Harvester & Contextual Retrieval** | arXiv (`.tex`/PDF) & OpenAlex harvester (`IN_NSE`, `DE_XETRA`, `US_EQUITY`), 26-concept relevance classifier, 12 negative blocking topics, Local `EmbeddingGemma 2` (768-d MPS/CPU), Lossless LaTeX/MD Chunkers, Contextualizer, 3-way Normalized RRF, Market Scepticism Gate | **COMPLETE** | `uv run ruff check` + `uv run mypy --strict src` + `uv run pytest -v` (158/158 passing, PR #3) |
| **M4: Enterprise MCP Servers (`mcp` v2.3.0)** | `code_repo_mcp.py` & `quant_corpus_mcp.py` with out-of-band `PrincipalContext` & hard payload caps | **COMPLETE** | `uv run ruff check src tests` + `uv run mypy --strict src tests` + `uv run pytest` (213/213 passing) |
| **M5: Google ADK 2.12 Scout, Pre-Flight Gate & Handoff** | ADK `Workflow` DAG, `LocalGemmaClient`, 4-Check `PreFlightValidator`, `FederatedScoutCoordinator`, `ShieldGemma 2`, Git worktree sandbox | **NEXT** | `tests/test_m5_adk_workflow.py` |
| **M6: Evaluation Harness & OpenTelemetry** | Recall@5 ($\ge 90\%$), Zero-Touch Sufficiency ($\ge 85\%$), Citation Faithfulness ($100\%$), RBAC Red-Team ($100\%$), OTel spans | **PENDING** | `tests/test_m6_eval_harness.py` |
| **M7: Teacher Distillation & QLoRA Training** | Rejection-sampled + 20% recovery trajectories (`gemini-3.8-flash` / `gemma-4-31B-it`), `mlx-lm` local M5 Pro & Vertex AI `PEFT` trainers | **PENDING** | Before-vs-After QLoRA Benchmark Report |
| **M8: FastAPI SSE, TypeScript/React UI & CI/CD** | FastAPI streaming backend, strict TS/React enterprise UI, `.gitlab-ci.yml` / GitHub Actions | **PENDING** | End-to-end API + UI test suite |

## 2. Milestone 2 Completion Details (AST-Grounded Code Engine)

Milestone 2 is verified complete with 54/54 tests passing across unit, adversarial, and live PostgreSQL 17 integration suites:
- **Git Merkle Incremental Indexing (`GitMerkleIndexer`):** Git Blob-SHA Merkle diffing verifies unchanged tree subtrees and blobs. Cache hits consistently execute under the required `<50ms` latency threshold (measured empirical averages: **~33.7ms** in-memory, **~37.4ms** live PostgreSQL 17 repository sync).
- **Tree-sitter Multi-Language AST Extraction:** Deterministic parser extracts structured symbols (`function`, `class`, `method`, `interface`, `type_alias`) and typed directed relations (`CALLS`, `IMPORTS`, `USES_TYPE`, `INHERITS`, `TESTED_BY`) across both Python and TypeScript codebases with 1-indexed byte and line spans.
- **Bounded 1-Hop AST Auto-Closure (`ASTDependencyExpander`):** Enforces strict `SEC-03` boundaries with `MAX_DIRECT_CALLERS_PER_SYMBOL = 3`, `MAX_EXPANSION_TOKEN_BUDGET = 1500`, and call-site diversity selection (different calling files/functions prioritized over duplicate call-sites), alongside matching `TESTED_BY` test fixtures and `USES_TYPE` imports.
- **Byte-Exact Slicing & Quote Verification (`DeterministicSlicer`):** Guarantees zero LLM re-typing hallucinations via byte-exact 1-indexed slicing attached to `git_commit_sha`/`git_blob_sha` provenance. Implements `VAL-02` NFKC and whitespace-normalized quote verification requiring $\ge 0.98$ token overlap.
- **Live PostgreSQL 17 RLS Integration:** Live integration with PostgreSQL 17 under `scout_app_user` with `FORCE ROW LEVEL SECURITY`, tenant scope isolation via `app.principal_scopes`, `pgvector` index queries, and `pg_trgm` similarity search.

## 3. Milestone 3 Completion Details (Quant Research Harvester, Contextual Retrieval & Scepticism Gate)

Milestone 3 is verified complete with 158/158 tests passing across all 15 test suites, including unit, live database integration, adversarial suites, challenger suites, and live arXiv/OpenAlex paper harvesting:
- **Multi-Market Quant Research Harvester (`paper_harvester.py` & `harvest_pipeline.py`):**
  - Streaming live ingestion covering arXiv Atom XML preprints across 5 quant categories (`q-fin.TR`, `q-fin.PM`, `q-fin.ST`, `q-fin.CP`, `q-fin.RM`) and OpenAlex REST API v2 using canonical concept mapping (`OPENALEX_CONCEPT_MAP`).
  - Strict `QuantRelevanceClassifier` targeting 26 quant trading concepts with immediate rejection for 12 negative blocking noise topics (`covid`, `coronavirus`, `pandemic`, `mandatory csr`, `corporate social responsibility`, `climate change`, `natural disaster`, `green bond`, `circular economy`, `brexit`, `industrial symbiotic`, `board gender`).
  - In-memory `.tar.gz` and `.gz` LaTeX archive extraction (`unpack_latex_archive`) with recursive multi-file include inlining (`resolve_latex_includes` for `\input`, `\include`, `\subfile`).
  - High-fidelity PDF fallback parser (`PdfDocParser`) via `pypdf` with academic section heading extraction, NUL-byte sanitization, and full-text validation (%PDF- header verification, $\ge 3,000$ extracted characters, $\ge 3$ chunks). Absolute ban on synthetic `.txt` abstract stubs (0 `.txt` files in corpus).
  - Resilient `HttpPaperHarvesterClient` with exponential backoff ($3.5s \times 2^n + [0.1, 0.5]s$ jitter) and OpenAlex polite pool headers (`mailto=scout@example.com`).
- **Scaled Quantitative Research Corpus Harvested & Cached:**
  - **300 Genuine Full-Text Papers Harvested:** Exactly 100 verified full-text papers per market (`IN_NSE`: 100, `DE_XETRA`: 100, `US_EQUITY`: 100).
  - 100% full-text coverage with 0 `.txt` abstract stubs: 259 LaTeX `.tex` source papers and 44 full-text `.pdf` papers.
  - Cached locally under `data/papers/` (`metadata/{market}_papers.jsonl` and `content/{market}/`).
- **Local Dense Semantic Embeddings (`LocalSemanticEmbedder`):**
  - Powered by **`google/embeddinggemma-2`** (768-dimensional Matryoshka dense vectors) running natively on Apple Silicon MPS with automated seamless CPU fallback.
  - Enforces `max_seq_length = 2048`, raw string pre-truncation to 6,000 characters, and `torch.mps.empty_cache()` lifecycle, eliminating memory pressure.
  - Cosine similarity $> 0.65$ verified on institutional quant finance concepts.
- **Live PostgreSQL 17 Persistence & 3-Way Normalized RRF Verification:**
  - **14,331 Total Genuine Chunks Persisted** in PostgreSQL 17 `doc_chunks` under domain `quant_research_scout` and scope `public_quant`.
  - Market breakdown:
    * `DE_XETRA`: **5,013 chunks** (100 verified papers)
    * `US_EQUITY`: **4,982 chunks** (100 verified papers)
    * `IN_NSE`: **4,336 chunks** (100 verified papers)
  - Live 3-way RRF retrieval verified: `IN_NSE` top score **0.7097**, `DE_XETRA` top score **0.6300**, `US_EQUITY` top score **0.7130** (all exceeding the institutional $\ge 0.55$ Pre-Flight quality gate threshold).
- **Lossless LaTeX Math & Table Chunking (`doc_parser.py`):**
  - `LatexDocParser` preserves math environments (`equation`, `align`, `gather`, `multline`, `$$...$$`, `$...$`) and full table float environments (`table` + `tabular`) as indivisible atomic units.
  - Hierarchical section breadcrumb tracking (`\section`, `\subsection`, `\subsubsection`) with stack rollback.
  - `MarkdownTableChunker` keeps GitHub-flavored markdown pipe tables intact with leading/trailing whitespace tolerance.
- **Anthropic Contextual Retrieval (`contextualizer.py`):**
  - 50-token situated context headers generated with `google/gemma-4-E4B-it` `<start_of_turn>user` turn formatting.
  - Deterministic offline fallback (`DeterministicContextualizer`) for 100% reproducible CI execution.
- **3-Way Normalized Reciprocal Rank Fusion (`BLK-01`, `bm25.py` & `repositories.py`):**
  - Fuses Lexical Okapi BM25 (Lucene smoothed non-negative IDF), Dense Vector (`embedding <=> %s::vector` targeting `EmbeddingGemma 2`), and Trigram string similarity.
  - Strictly normalized to $[0.0, 1.0]$ via $\frac{\text{raw\_score}}{\text{max\_possible}}$.
  - Multi-tenant isolation enforced at the PostgreSQL 17 kernel level via `FORCE ROW LEVEL SECURITY`.
- **Market-Regime Quant Scepticism Gate (`scepticism_gate.py`, `FIN-01` & `VAL-02`):**
  - Enforces statutory friction floors ($\ge 15.0$ bps delivery / $\ge 5.0$ bps futures for `IN_NSE`, $\ge 10.0$ bps for `DE_XETRA`, $\ge 5.0$ bps for `US_EQUITY`).
  - Penalizes net Sharpe degradation ($\le 0.50$ fails after costs) and caps maximum drawdown ($\le 30.0\%$).
  - 100% verbatim citation verification (`VAL-02`) via `DeterministicSlicer` with NFKC normalization ($\ge 0.98$ overlap), rejecting single-digit mutations with `HallucinatedQuoteError`.
- **Verification Metrics:**
  - `uv run ruff check src tests`: 0 errors.
  - `uv run mypy --strict src`: 0 type errors across 24 source files.
  - `uv run mypy tests`: 0 type errors across 15 test files.
  - `uv run pytest -v`: 158 passed in 33.91s across 15 test suites (100% pass rate, 0 failures, 0 skips).
  - Pull Request: [PR #3](https://github.com/himanshu2408/enterprise-context-scout/pull/3) (`feat/m3-quant-research-harvester` -> `main`).

## 4. Milestone 4 Completion Details (Enterprise MCP Servers & Realistic Trading Code Integration)

Milestone 4 is verified complete with 213/213 tests passing across all test suites, including unit, live PostgreSQL 17 Forced RLS integration, adversarial parameter injection, and payload bounds tests:
- **Realistic Multi-File Trading Strategy Package (`src/context_scout/trading_strategies/`):**
  - `volatility_garch.py`: Real-time conditional volatility tracking via `GARCH11Model` with constrained quasi-MLE ($\alpha + \beta \le 0.9999$) and stationarity protection.
  - `kelly_risk_manager.py`: Continuous multi-asset position sizing via `KellyRiskManager` using Ledoit-Wolf linear covariance shrinkage with eigenvalue regularisation floors (`MIN_EIGENVALUE_FLOOR = 1e-6`).
  - `order_flow_imbalance.py`: Multi-level order book pressure calculation via `OrderFlowImbalanceSignal` (Cont-Kukanov-Stoikov 2014) with exponential depth decay and GARCH scaling.
  - `pairs_trading_kalman.py`: Cointegration hedge ratio tracking via `KalmanHedgeEstimator` implementing the symmetric Joseph-form covariance update ($(I - KH)P(I - KH)^T + KRK^T$) for numerical stability under market turbulence.
  - `almgren_chriss_execution.py`: Optimal execution trajectory optimization via `AlmgrenChrissExecutionOptimizer` modeling linear permanent and quadratic temporary market impact under risk penalties.
  - All 6 files seeded into live PostgreSQL 17: 43 total code files, 227 AST symbols, and 2,431 directed AST edges.
- **Enterprise Code Repository MCP Server (`code_repo_mcp.py`):**
  - Fast, bounded codebase intelligence under Forced RLS using canonical tools:
    * `search_code_trigram`: `pg_trgm` similarity search over indexed repository files.
    * `lookup_ast_symbol`: Exact and prefix symbol definitions across language AST trees.
    * `expand_ast_1hop_closure`: Bounded 1-hop caller/type dependency closure ($\le 3$ callers, $\le 1,500$ tokens).
    * `slice_file_lines`: Byte-exact line slicing with 1-indexed validation, off-by-one detection, and git blob/commit SHA provenance.
    * `search_github_issues`: Full-text issue search over historical bug reports and architecture proposals stored in `doc_chunks`.
  - Legacy backward-compatible aliases: `search_ast_symbols`, `get_symbol_definition`, `get_symbol_callers`, `grep_repo`, `list_github_issues`.
- **Enterprise Quantitative Corpus MCP Server (`quant_corpus_mcp.py`):**
  - High-precision research corpus discovery across 14,318 genuine academic chunks:
    * `search_quant_papers_hybrid`: 3-way Normalized RRF search across lexical BM25, dense 768-d semantic vectors (`EmbeddingGemma 2`), and trigram matches.
    * `extract_paper_equations_and_tables`: Lossless extraction of LaTeX mathematical equations and Markdown tables.
    * `verify_market_friction_assumptions`: Quantitative Scepticism Gate verifying transaction cost floors ($\ge 15$ bps NSE, $\ge 10$ bps Xetra, $\ge 5$ bps US) and net Sharpe degradation hurdle ($\le 0.50$ fails after costs).
    * `compare_cross_market_regimes`: Comparative cross-market structural and regulatory analysis across Indian, German, and US markets.
  - Legacy backward-compatible aliases: `search_papers_hybrid`, `get_paper_section`, `compare_market_frictions`.
- **Out-of-Band Security & Payload Guardrails (`common.py`):**
  - `SEC-04` Out-of-Band Context: Zero schema leakage—`PrincipalContext` and `scopes` are strictly injected via Python `ContextVar` session context and stripped from MCP tool schemas. Parameter injection attempts raise `SecurityError`.
  - `SEC-03` Hard Payload Caps: Clamps collection outputs to $\le 10$ items and payload size to $\le 32$ KB (32,768 bytes) serialized JSON with `[PAYLOAD_TRUNCATED_AT_32KB]` notification.
  - `DomainAuthorizationGate`: Verifies domain-level tool execution permissions against domain YAML manifests.
- **Verification Metrics:**
  - `uv run ruff check src tests`: 0 errors.
  - `uv run mypy --strict src tests`: 0 type errors across 55 source files.
  - `uv run pytest -v tests/test_m4_mcp_servers.py`: 18/18 passed in 8.43s.
  - `uv run pytest`: 213 passed in 40.29s across 17 test suites (100% pass rate).


