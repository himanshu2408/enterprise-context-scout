# STATUS: Enterprise Context Scout & Multi-Domain Platform

**Last Updated:** 2026-10-29  
**Active Branch:** `feat/m3-quant-research-harvester`  
**Repository:** `https://github.com/himanshu2408/enterprise-context-scout`

## 1. Milestone Progress Dashboard

| Milestone | Scope | Status | Verification Gate |
| :--- | :--- | :---: | :--- |
| **M0: Architecture & Audit** | `ARCHITECTURE_BLUEPRINT.md`, `ARCHITECTURE_AUDIT_REPORT.md`, `AGENTS.md`, Pre-commit Security Hook | **COMPLETE** | 12/12 Audit Remediations Applied (`BLK-01`, `SEC-01..04`, `PERF-01..02`, `ARC-01..02`, `HW-01`, `FIN-01`) |
| **M1: Foundation, Postgres 17 RLS & Contracts** | `pyproject.toml`, `docker-compose.yml`, `sql/001_init_schema.sql`, Pydantic v2 Contracts, `DomainRegistry`, `AdapterPool`, RLS Repositories | **COMPLETE** | `uv run ruff check` + `uv run mypy --strict src` + `uv run pytest tests/test_m1_foundation.py` |
| **M2: Code Engine (AST + Merkle Sync + Slicer)** | Git Blob-SHA Merkle indexer, `tree-sitter` AST symbol/edge extractor, bounded 1-hop `ASTDependencyExpander` ($\le 3$ callers, $\le 1,500$ tokens), `DeterministicSlicer` | **COMPLETE** | `uv run ruff check` + `uv run mypy --strict src tests` + `uv run pytest -v` (54/54 passing) |
| **M3: Quant Paper Harvester & Contextual Retrieval** | arXiv (`.tex`/PDF) & OpenAlex harvester (`IN_NSE`, `DE_XETRA`, `US_EQUITY`), Lossless LaTeX/MD Chunkers, `Gemma-4-E4B-it` Contextualizer, 3-way Normalized RRF, Market Scepticism Gate | **COMPLETE** | `uv run ruff check` + `uv run mypy --strict src tests` + `uv run pytest -v` (116/116 passing, PR #3) |
| **M4: Enterprise MCP Servers (`mcp` v2.3.0)** | `code_repo_mcp.py` & `quant_corpus_mcp.py` with out-of-band `PrincipalContext` & hard payload caps | **NEXT** | `tests/test_m4_mcp_servers.py` |
| **M5: Google ADK 2.12 Scout, Pre-Flight Gate & Handoff** | ADK `Workflow` DAG, `LocalGemmaClient`, 4-Check `PreFlightValidator`, `FederatedScoutCoordinator`, `ShieldGemma 2`, Git worktree sandbox | **PENDING** | `tests/test_m5_adk_workflow.py` |
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

Milestone 3 is verified complete with 116/116 tests passing across all test suites, including unit, live database integration, two rigorous adversarial suites, and live arXiv/OpenAlex paper harvesting:
- **Multi-Market Quant Research Harvester (`paper_harvester.py`):**
  - Live ingestion covering arXiv Atom XML feeds and OpenAlex REST API v2 using canonical concept mapping (`OPENALEX_CONCEPT_MAP`).
  - Multi-market presets for `INDIA_NSE_BSE`, `GERMANY_XETRA_DAX`, and `US_NYSE_NASDAQ`.
  - In-memory `.tar.gz` and `.gz` LaTeX archive extraction (`unpack_latex_archive`) with recursive multi-file include inlining (`resolve_latex_includes` for `\input`, `\include`, `\subfile`).
  - High-fidelity PDF fallback parser (`PdfDocParser`) via `pypdf` with academic section heading extraction and NUL-byte sanitization.
  - Resilient `HttpPaperHarvesterClient` with exponential backoff ($3.5s \times 2^n + [0.1, 0.5]s$ jitter) and OpenAlex polite pool headers (`mailto=scout@example.com`).
- **Scaled Quantitative Research Corpus Harvested & Cached:**
  - **330 Total Papers Harvested:** Top 110 citation-ranked papers per market (`IN_NSE`: 110, `DE_XETRA`: 110, `US_EQUITY`: 110).
  - Cached locally under `data/papers/` (`metadata/{market}_papers.jsonl` and `content/{market}/`).
- **Live PostgreSQL 17 Persistence & 3-Way Normalized RRF Verification:**
  - **1,473 Total Chunks Persisted** in PostgreSQL 17 `doc_chunks` under domain `quant_research_scout` and scope `public_quant`.
  - Market breakdown: `DE_XETRA`: 639 chunks, `US_EQUITY`: 449 chunks, `IN_NSE`: 385 chunks.
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
  - `uv run ruff check .`: 0 errors.
  - `uv run mypy --strict src tests`: 0 type errors across 35 source files.
  - `uv run pytest -v`: 116/116 tests passing (100% pass rate in 14.9s).
  - Pull Request: [PR #3](https://github.com/himanshu2408/enterprise-context-scout/pull/3) (`feat/m3-quant-research-harvester` -> `main`).

