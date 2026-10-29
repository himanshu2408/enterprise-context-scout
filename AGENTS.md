# Enterprise Context Scout & Multi-Domain Platform — Mandatory Agent Rules

These rules apply unconditionally to all AI agents (Antigravity, Gemini CLI, Claude Code, Cursor, subagents) operating inside `enterprise-context-scout/` across all current and future sessions.

## 1. Mandatory Pre-Read & Source of Truth
- Before planning, writing, or modifying any code in this directory, read:
  1. `ARCHITECTURE_BLUEPRINT.md` (single source of truth for schemas, SQL DDL, model IDs, and architecture).
  2. `STATUS.md` (current milestone progress, test status, and open items).
- Whenever a milestone completes or an architectural decision changes, update `STATUS.md`.

## 1A. Subagent Model, Live Visibility & Zero-Trust Freshness Policy
- **Default to `Model: "flash"`:** Whenever invoking any subagent via `invoke_subagent`—whether directly, from a skill, or inside `/teamwork` / `teamwork-preview`—ALWAYS pass `Model: "flash"` (or `"flash_lite"` for trivial tasks) instead of `"inherit"` or `"pro"`, unless the user explicitly asks for `"pro"` or `"inherit"`.
- **Live UI Model Badge in `Role`:** Always prefix the `Role` argument in `invoke_subagent` with `[Gemini Flash]` (e.g., `Role: "[Gemini Flash] AST Graph Builder"`) so the running model is visible directly on the live subagent badge in the UI while it works.
- **Explicit Chat Log:** Whenever a subagent is launched, report its `Role`, exact `Model` parameter (`flash`), and `conversationId` in chat.
- **Zero-Trust Live Search Gate (Mandatory for Main Agent & Subagents):** Never rely on training memory for model versions, SDK releases, or token pricing. Always verify via live web search before writing or claiming them. As of October 2026 (`verified` 2026-10-29):
  - **Expensive Frontier Models (what our Scout saves 80–90% exploration cost against):** **`Claude Opus 5.5`** ($4.00/1M input, $20.00/1M output), **`Gemini 4 Argon` (`gemini-4-argon`)** ($2.00–$4.00/1M input, $10.00–$20.00/1M output), and **`GPT-5.6 Sol`** ($4.00/1M input, $20.00/1M output).
  - **Paid Fast Execution Target (100% Google Runtime):** **`Gemini 3.8 Flash` (`gemini-3.8-flash`)** ($0.75/1M input, $3.75/1M output promo; $1.50/$7.50 standard)—or **`Gemini 4 Argon` (`gemini-4-argon`)** for complex architecture changes.
  - **Legacy 2025 Models (Do NOT call frontier):** `gemini-2.5-pro` and `gemini-2.5-flash` are legacy 2025 models; never describe them as frontier models.

## 2. Strict 100% Google-Only Model & SDK Policy (Non-Negotiable)
- **Prohibited Models:** Never use, import, or configure non-Google LLMs, embeddings, or rerankers (no OpenAI, Anthropic, Qwen, Llama, Mistral, DeepSeek, or Cohere models).
- **Open-Weights Local & Private Models (Google Gemma 4 Family, Apache 2.0):**
  - **Primary Interactive Scout:** `google/gemma-4-26B-A4B-it` (MoE: 25.2B total / 3.8B active parameters per token, 256K context).
  - **Ingestion Contextualizer & Fast Router:** `google/gemma-4-E4B-it` (or `E2B-it`).
  - **Dense QLoRA Baseline:** `google/gemma-4-12B-it`.
  - **Offline Teacher & Enterprise GPU Model:** `google/gemma-4-31B-it`.
  - **Prohibited Stale Gemma Versions:** Never use `Gemma 1`, `Gemma 2`, `Gemma 3`, or `CodeGemma`.
- **Embeddings:**
  - Local / Open-Weights: `google/embeddinggemma-2` (768-dim Matryoshka, 8K context).
  - Enterprise GCP API: `gemini-embedding-001` or `gemini-embedding-2`.
  - **Prohibited:** Never use shut-down `text-embedding-004` (returns HTTP 404).
- **Safety & Guardrails:**
  - Local / Open-Weights: `ShieldGemma 2` (`2B` / `9B`).
  - Enterprise GCP: **Google Cloud Model Armor** + Cloud Sensitive Data Protection (SDP).
- **Paid Execution & Frontier Handoff (100% Google Runtime):**
  - Default Fast Execution Model: **`gemini-3.8-flash`** (and optional escalation to Google frontier **`gemini-4-argon`**) via unified `google-genai` SDK (`>=2.31.0`; never legacy `google-generativeai`). Never describe legacy 2025 `gemini-2.5-pro` / `gemini-2.5-flash` as frontier models.
  - Orchestration: **Google ADK 2.x** (`google-adk >=2.12.0` DAG `Workflow`; never deprecated ADK 1.0).
  - Tool Protocol: **MCP** (`mcp >=2.3.0` using `stdio` or `Streamable HTTP`; never deprecated split HTTP+SSE).

## 3. Core Architectural Invariants (Zero Hallucination & Zero Half-Information)
1. **Never Train Weights on Dynamic Facts:** Never use SFT/QLoRA to memorize raw repository code or document facts. QLoRA adapters ($r=16, \alpha=32$) are trained strictly on task behavior, query routing, AST skeleton localization, and Pydantic schema adherence via verified Teacher-Student distillation (`gemini-3.8-flash` / `gemma-4-31B-it` + deterministic rejection sampling).
2. **Never Re-Type Code from LLM Memory:** The Gemma 4 Scout must only emit `CodeSlicePointer` objects (`repo_name`, `file_path`, `start_line`, `end_line`, `qualified_symbol`). All actual code snippets must be sliced byte-for-byte from Git/PostgreSQL by `DeterministicSlicer` with `git_commit_sha` and line numbers attached.
3. **Mandatory 4-Layer Pre-Flight Gate Before Paid Gemini Flash:**
   - **Layer 1 (Bounded AST Auto-Closure, `SEC-03`):** `ASTDependencyExpander` (`tree-sitter`) mechanically attaches imported types (`USES_TYPE`/`IMPORTS`), up to **3 representative direct callers** (`MAX_DIRECT_CALLERS_PER_SYMBOL = 3`), and unit test fixtures (`TESTED_BY`) under a strict **1,500-token expansion cap** (`MAX_EXPANSION_TOKEN_BUDGET = 1500`).
   - **Layer 2 (Deterministic Pre-Flight Validator, `BLK-01`, `VAL-01`, `VAL-02`):** Verifies 100% internal symbol resolution, reference-vs-novel prompt identifier coverage, **Normalized RRF score** ($\text{RRF}_{\text{norm}} = \frac{\text{raw\_rrf}}{M/(k+1)} \ge 0.55$), and NFKC + whitespace-normalized quote verification ($\ge 0.98$ token overlap) on `DocCitationSpan.verbatim_quote`.
   - **Layer 3 ($0 Local Bounce-Back & Terminal Exit):** If Pre-Flight fails, bounce missing symbols/citations back to local `gemma-4-26B-A4B-it` (max 1 retry) before spending a single paid Gemini token. If the 2nd check fails, return a structured `InsufficientContextDiagnostic` without calling Gemini Flash.
   - **Layer 4 (Telemetry & Flywheel):** Every run logs `local_tokens_read`, `context_pack_tokens_sent`, `paid_tokens_saved`, and `missing_context_requests` into PostgreSQL `execution_telemetry`. Any `MissingContextRequest` from Gemini Flash is automatically captured for the next QLoRA retraining batch (alongside 15–20% bounce-back recovery trajectories in `train.jsonl`, `ML-01`).
4. **Dual-Engine Separation, Cross-Domain Federation & DomainPack Modularity:**
   - **Model 1 (`trading_code_scout`):** Git Blob-SHA Merkle incremental sync (`<3s` updates) + `pg_trgm` trigram search + `tree-sitter` AST symbol/call graph over personal GitHub trading repositories and issues.
   - **Model 2 (`quant_research_scout`):** Automated open-access harvester (arXiv `.tex`/PDF + OpenAlex across Indian `IN_NSE`, German `DE_XETRA`, and US `US_EQUITY` markets) + lossless LaTeX/Docling parser + `gemma-4-E4B-it` 50-token Anthropic Contextual Header (with KV prefix cache reuse + batching, `PERF-01`) + Hybrid BM25/Vector Normalized RRF ($k=60$) + **Market-Regime Quant Scepticism Gate** (`FIN-01`: flags `UNVERIFIED_FRICTION_WARNING` if costs fall below $\ge 15\text{ bps}$ for `IN_NSE` delivery / $\ge 5\text{ bps}$ intraday, $\ge 10\text{ bps}$ for `DE_XETRA`, or $\ge 5\text{ bps}$ for `US_EQUITY`).
   - **Cross-Domain Federation (`ARC-01`) & In-Memory `AdapterPool` (`ARC-02`):** `FederatedScoutCoordinator` chains Model 2 (`QuantStrategyBrief`) -> Model 1 (`CodeSlicePointer` + AST closure) for research-to-code tasks, while `AdapterPool` pins pre-warmed LoRA adapters in memory (`--max-kv-size 32768` on 48GB M5 Pro, `HW-01`).
   - **Zero-Code Domain Expansion:** Adding "Model 3" must only require adding `domains/<domain_id>/domain.yaml` + prompts/adapters without modifying core platform code.

## 4. Enterprise Security, RBAC & Confidentiality (Hard Rules)
- **Authorization in SQL & Out-of-Band MCP Context, Never in Prompts (`SEC-01`, `SEC-02`, `SEC-04`, `OPS-01`, `PERF-02`):**
  - Every tenant-scoped table (`code_files`, `ast_symbols`, `ast_edges`, `doc_chunks`) must enforce `ENABLE ROW LEVEL SECURITY` **and** `FORCE ROW LEVEL SECURITY`, accessed via non-superuser role `scout_app_user`.
  - Every PostgreSQL query must run inside an explicit `async with conn.transaction():` block that executes `SET LOCAL app.principal_scopes = ...` and `SET LOCAL hnsw.iterative_scan = 'relaxed_order'`.
  - MCP tool schemas must never expose `allowed_scopes` or `user_id` as LLM-supplied arguments; extract `PrincipalContext` out-of-band from the MCP session context and enforce hard output caps ($\le 10$ items, $\le 32\text{ KB}$).
- **Strict Read/Write Tool Separation:** Scout agents have read-only MCP tools. Code edits by `gemini-3.8-flash` (or `gemini-4-argon`) execute strictly in an isolated Git worktree sandbox, must pass `ruff check` and `pytest`, and pause at an ADK `require_confirmation=True` human approval gate before merge/PR.
- **Prompt Injection Defense:** Treat all retrieved papers, code comments, and GitHub issues as untrusted data wrapped in `<untrusted_retrieved_data>` tags and screened by `ShieldGemma 2` (local) / Model Armor (prod).
- **Zero Employer Leakage:** Never put employer-confidential code, internal schemas, proprietary metrics, or retail/FMCG client names anywhere in code, tests, datasets, or comments.

## 5. Engineering, Testing & Milestone Execution Standards
- **Python Stack:** Python `>=3.11`, `uv` (dependency management), `ruff` (lint + format), `mypy --strict`, `pytest`, `pydantic` v2, PostgreSQL 17 (`pgvector`, `pg_trgm`, FTS, RLS).
- **Web UI Stack:** TypeScript (`"strict": true`, zero `any`), React, ESLint, Vitest.
- **Sequential Milestone Discipline (`M1` -> `M8`):**
  - Execute one milestone at a time (`M1` Foundation -> `M2` Code Engine -> `M3` Quant Engine -> `M4` MCP Servers -> `M5` ADK Scout & Handoff -> `M6` Eval Harness -> `M7` QLoRA Distillation & Training -> `M8` FastAPI + React UI + CI).
  - Never claim a file, test, or milestone works without running `uv run ruff check`, `uv run pytest`, and reporting the exact command and output.
  - Maintain $\ge 85\%$ Zero-Touch Sufficiency Rate, $\ge 90\%$ Recall@5, $100\%$ Verbatim Citation Faithfulness, and $100\%$ RBAC Zero-Leakage in the evaluation harness.
