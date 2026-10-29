# Enterprise Context Scout & Multi-Domain Knowledge Platform

An enterprise-grade, dual-engine context discovery and verification platform built on the **Google AI & Open-Weights Stack** (`Gemma 4 26B-A4B-it` MoE, `Gemma 4 E4B-it`, `EmbeddingGemma 2`, `ShieldGemma 2` / Google Cloud Model Armor, `Google ADK 2.12`, `MCP 2.3`, and `PostgreSQL 17` with `pgvector`, `pg_trgm`, and `FORCE ROW LEVEL SECURITY`).

## Why This Platform Exists: Frontier Economics vs. Zero-Cost Local Discovery

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
   - **Engine 1 — Codebase Intelligence (`trading_code_scout`):** Git Blob-SHA Merkle incremental indexing (`<3s` sync), 3-way Normalized Reciprocal Rank Fusion (`pg_trgm` + `tsvector` BM25 + `EmbeddingGemma 2` 768-d), and `tree-sitter` AST symbol/call graph with bounded 1-hop AST auto-closure. Emits structured `CodeSlicePointer` objects sliced byte-for-byte with `git_commit_sha` provenance.
   - **Engine 2 — Multi-Market Quantitative Research (`quant_research_scout`):** Automated open-access harvester (arXiv + OpenAlex covering Indian `IN_NSE`, German `DE_XETRA`, and US `US_EQUITY` markets), lossless LaTeX/Docling extraction, Anthropic Contextual Retrieval (`Gemma-4-E4B-it`), and Market-Regime Quant Scepticism Gate ($\ge 15\text{ bps}$ NSE delivery, $\ge 10\text{ bps}$ Xetra, $\ge 5\text{ bps}$ US).
2. **Paid Fast Execution Engine (100% Google Model Runtime):**
   - Implementation executes in a single turn using **Google `gemini-3.8-flash`** ($0.75/1M input, $3.75/1M output promo through Dec 31, 2026), or **`gemini-4-argon`** when complex frontier architectural reasoning is required.
   - The execution model receives a compact, verified **`ImplementationContextPack`** (typically 2,000–4,000 tokens instead of 45,000+ raw repository tokens), achieving 80–90% paid token savings over ungrounded frontier queries.
3. **4-Layer Pre-Flight Quality Gate:**
   - Validates 100% internal symbol resolution, Normalized RRF floor ($\ge 0.55$), and NFKC-normalized verbatim citation matching ($\ge 0.98$) before spending a single paid Gemini token, with a local $0 bounce-back self-correction loop and a `ContextMissEvent` QLoRA retraining flywheel.

## Documentation

- **[Architecture Blueprint (`ARCHITECTURE_BLUEPRINT.md`)](./ARCHITECTURE_BLUEPRINT.md):** Full system topology, SQL DDL, Pydantic v2 contracts, QLoRA memory math, and trade-off matrices.
- **[Enterprise Architecture Audit Report (`ARCHITECTURE_AUDIT_REPORT.md`)](./ARCHITECTURE_AUDIT_REPORT.md):** Independent 8-milestone security, math, and performance audit.
- **[Agent & Engineering Rules (`AGENTS.md`)](./AGENTS.md):** Mandatory architectural, security, and testing invariants.
