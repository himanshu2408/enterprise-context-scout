# Enterprise Context Scout & Multi-Domain Knowledge Platform

An enterprise-grade, dual-engine context discovery and verification platform built on the **Google AI & Open-Weights Stack** (`Gemma 4 26B-A4B-it` MoE, `Gemma 4 E4B-it`, `EmbeddingGemma 2`, `ShieldGemma 2` / Google Cloud Model Armor, `Google ADK 2.12`, `MCP 2.3`, and `PostgreSQL 17` with `pgvector`, `pg_trgm`, and `FORCE ROW LEVEL SECURITY`).

## Why This Platform Exists

When frontier models (`gemini-2.5-flash` / `gemini-3.8-flash`) are pointed directly at large repositories and research corpora, **80–90% of paid input tokens are wasted on exploratory search**, and arbitrary vector chunking causes "half-information" hallucinations (missing imported types, broken callers, or shifted line numbers).

**Enterprise Context Scout** decouples **Zero-Cost Local Discovery & Verification** from **Paid Frontier Implementation**:

1. **Engine 1 — Codebase Intelligence (`trading_code_scout`):**
   - Git Blob-SHA Merkle incremental indexing (`<3s` updates on large repos).
   - 3-way Normalized Reciprocal Rank Fusion (`pg_trgm` trigram search + `tsvector` BM25 + `EmbeddingGemma 2` 768-d vectors).
   - `tree-sitter` AST symbol and call-graph index with **bounded 1-hop AST auto-closure** (automatically attaching imported types, up to 3 representative direct callers, and unit test fixtures).
   - **Deterministic Byte-for-Byte Slicer:** The open-weights Scout emits structured `CodeSlicePointer` objects (`repo`, `file_path`, `start_line`, `end_line`), and Python slices verbatim lines with `git_commit_sha`—guaranteeing zero hallucinated code.
2. **Engine 2 — Multi-Market Quantitative Research (`quant_research_scout`):**
   - Automated open-access harvester (arXiv `.tex`/PDF + OpenAlex) covering **Indian (`IN_NSE`), German (`DE_XETRA`), and US (`US_EQUITY`)** markets.
   - Lossless LaTeX & `Docling` equation/table extraction + Anthropic-style **Contextual Retrieval** (`Gemma-4-E4B-it` with KV prefix cache reuse).
   - **Market-Regime Quant Scepticism Gate:** Enforces realistic transaction cost and tax floors (`IN_NSE` $\ge 15\text{ bps}$ delivery / $\ge 5\text{ bps}$ intraday + STT check; `DE_XETRA` $\ge 10\text{ bps}$; `US_EQUITY` $\ge 5\text{ bps}$).
3. **4-Layer Pre-Flight Quality Gate & Paid Model Handoff:**
   - Validates 100% internal symbol resolution, Normalized RRF floor ($\ge 0.55$), and NFKC-normalized verbatim citation matching ($\ge 0.98$) before spending a single paid Gemini token, with a local $0 bounce-back loop and a `ContextMissEvent` QLoRA retraining flywheel.

## Documentation

- **[Architecture Blueprint (`ARCHITECTURE_BLUEPRINT.md`)](./ARCHITECTURE_BLUEPRINT.md):** Full system topology, SQL DDL, Pydantic v2 contracts, QLoRA memory math, and trade-off matrices.
- **[Enterprise Architecture Audit Report (`ARCHITECTURE_AUDIT_REPORT.md`)](./ARCHITECTURE_AUDIT_REPORT.md):** Independent 8-milestone security, math, and performance audit.
- **[Agent & Engineering Rules (`AGENTS.md`)](./AGENTS.md):** Mandatory architectural, security, and testing invariants.
