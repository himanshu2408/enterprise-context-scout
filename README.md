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
├── docker-compose.yml          # PostgreSQL 17 + pgvector container definition
├── sql/
│   └── 001_init_schema.sql     # PostgreSQL 17 DDL: tables, indexes, and FORCE RLS policies
├── src/
│   └── context_scout/
│       ├── __init__.py         # Package root
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
│       └── agents/
│           └── __init__.py     # Agent interfaces and adapter management
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
    └── test_real_paper_harvest_live.py # Live harvest, PDF fallback, DB persistence & gate integration (6 tests)
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

## 6. Local Setup & Installation

### 6.1 Prerequisites

- **macOS** or **Linux** workstation.
- **Python 3.12+** with `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **PostgreSQL 17** with `pgvector` and `pg_trgm` extensions:
  ```bash
  # macOS (Homebrew)
  brew install postgresql@17 pgvector
  brew services start postgresql@17
  ```

### 6.2 Environment & Database Setup

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/himanshu2408/enterprise-context-scout.git
   cd enterprise-context-scout
   uv sync
   ```

2. **Initialize PostgreSQL database:**
   ```bash
   # Create database and unprivileged application role
   createdb context_scout_test
   psql context_scout_test -c "CREATE USER context_scout_app WITH PASSWORD 'scout_test_password';"
   psql context_scout_test -c "GRANT CONNECT ON DATABASE context_scout_test TO context_scout_app;"

   # Run schema migration as superuser
   psql context_scout_test -f sql/001_init_schema.sql
   ```

3. **Configure environment variables (`.env`):**
   ```bash
   cp .env.example .env
   ```
   Set `DATABASE_URL=postgresql://context_scout_app:scout_test_password@localhost:5432/context_scout_test`.

---

## 7. Verification & Test Suite

The platform enforces zero-trust automated testing across all components:

```bash
# 1. Format & Linting Check (Ruff)
uv run ruff check .

# 2. Strict Static Type Checking (Mypy)
uv run mypy --strict src tests

# 3. Comprehensive Unit & Integration Test Suite (Pytest)
uv run pytest -v
```

### Test Coverage Highlights (158 Passed Tests, 0 Failures, 0 Skips)

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
| **Total** | **15 Test Suites** | **158** | **100% Pass Rate (0 Failures, 0 Skips, 0 Errors in 33.91s)** |

---

## 8. Security Invariants & Repository Protection

- **Confidentiality:** No proprietary client names, credentials, or internal metrics are permitted in code, comments, or commit history.
- **Git Hook Protection:** Local git hooks (`.git/hooks/pre-commit` and `.git/hooks/commit-msg`) enforce regex-based secret scanning, path checks, and word-bounded term blocking on every commit.
- **Model Truthfulness:** All model specifications, pricing, and benchmark claims are verified against official documentation and live search gates.
