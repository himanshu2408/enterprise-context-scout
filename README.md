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
│       │   ├── connection.py   # Psycopg 3 connection pool with RLS context manager
│       │   └── schema.py       # Table names and database constants
│       ├── ast_engine/
│       │   ├── __init__.py
│       │   ├── parser.py       # Multi-language Tree-sitter parser (Python & TypeScript)
│       │   ├── expander.py     # ASTDependencyExpander: bounded 1-hop caller closure (SEC-03)
│       │   └── slicer.py       # DeterministicSlicer: byte-exact slicing & NFKC quote verification
│       ├── ingestion/
│       │   ├── __init__.py
│       │   └── merkle.py       # GitMerkleIndexer: Git Blob-SHA diffing (<50ms incremental sync)
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
    └── test_empirical_merkle_slicer_challenge.py # Benchmark tests (<50ms Merkle sync, >=0.98 NFKC)
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

## 5. Local Setup & Installation

### 5.1 Prerequisites

- **macOS** or **Linux** workstation.
- **Python 3.12+** with `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **PostgreSQL 17** with `pgvector` and `pg_trgm` extensions:
  ```bash
  # macOS (Homebrew)
  brew install postgresql@17 pgvector
  brew services start postgresql@17
  ```

### 5.2 Environment & Database Setup

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

## 6. Verification & Test Suite

The platform enforces zero-trust automated testing across all components:

```bash
# 1. Format & Linting Check (Ruff)
uv run ruff check .

# 2. Strict Static Type Checking (Mypy)
uv run mypy --strict src tests

# 3. Comprehensive Unit & Integration Test Suite (Pytest)
uv run pytest -v
```

### Test Coverage Highlights (54 Passed Tests)
- `tests/test_m1_foundation.py`: Pydantic v2 contracts, serialization, and adapter interfaces.
- `tests/test_postgres_rls_live.py`: Forced Row-Level Security, scope isolation, and tenant cross-access denial.
- `tests/test_m2_code_engine.py`: Tree-sitter symbol extraction, call graph resolution, and error tolerance.
- `tests/test_dependency_expander.py`: Bounded 1-hop closure, token budgeting, and diversity selection.
- `tests/test_deterministic_slicer.py`: Byte-exact slicing, line range verification, and NFKC normalization.
- `tests/test_postgres_m2_live.py`: Live PostgreSQL Merkle tree diffing and database persistence.
- `tests/test_adversarial_ast_rls.py`: Security boundaries, malicious scope injection, and cycle handling.
- `tests/test_empirical_merkle_slicer_challenge.py`: Performance assertions ($<50\text{ ms}$ Merkle sync, $\ge 0.98$ NFKC quote overlap).

---

## 7. Security Invariants & Repository Protection

- **Confidentiality:** No proprietary client names, credentials, or internal metrics are permitted in code, comments, or commit history.
- **Git Hook Protection:** Local git hooks (`.git/hooks/pre-commit` and `.git/hooks/commit-msg`) enforce regex-based secret scanning, path checks, and word-bounded term blocking on every commit.
- **Model Truthfulness:** All model specifications, pricing, and benchmark claims are verified against official documentation and live search gates.
