# Security, Governance & Enterprise Deployment Topology

**Enterprise Context Scout** enforces access control at the database storage kernel and transport boundary—never in LLM prompts—and isolates untrusted retrieved content before any local or paid frontier model processes it.

---

## 1. Out-of-Band Authentication & Principal Binding (`src/context_scout/security/rbac.py`, `api/routes/auth.py`)

- **Out-of-Band `PrincipalContext` Extraction:**
  - `POST /api/v1/auth/session` mints short-lived (`3600s`) HMAC-SHA256 Bearer tokens for interactive local evaluation (disabled with `HTTP 403` when `is_production_environment()` is true, requiring an external IdP JWT in production).
  - `GET /api/v1/auth/me` returns the verified `PrincipalContext` (`user_id`, `role`, `allowed_scopes`, `max_session_paid_tokens`) bound to a Python `contextvars.ContextVar` at the transport boundary.
- **Fail-Closed Startup Secret Validation:** An unset signing secret is a startup error when signed authentication is enabled (`SCOUT_REQUIRE_SIGNED_AUTH=true`); production never falls back to hardcoded default secrets.
- **Zero Identity Parameters in MCP Tool Schemas:**
  - MCP tools (`code_repo_mcp`, `quant_corpus_mcp`, `doc_corpus_mcp`) accept only domain search parameters.
  - `validate_tool_call_arguments` blocks case-insensitive attempts to inject `principal`, `principal_context`, `allowed_scopes`, `user_id`, `role`, `scopes`, `principal_scopes`, or `app.principal_scopes`, and rejects NUL bytes (`\x00`).
  - `/api/v1/handoff/preview` confirmation tokens are cryptographically bound to the requesting `principal.user_id`; cross-principal `/api/v1/handoff/apply` calls return `HTTP 403`.

---

## 2. PostgreSQL 17 `FORCE ROW LEVEL SECURITY` (`sql/001_init_schema.sql`)

Tenant and repository isolation is enforced inside PostgreSQL 17 so that even a compromised application query cannot read unauthorized rows:

```sql
-- Forced RLS on all tenant-scoped tables (even table owners cannot bypass)
ALTER TABLE code_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE code_files FORCE ROW LEVEL SECURITY;
ALTER TABLE ast_symbols ENABLE ROW LEVEL SECURITY;
ALTER TABLE ast_symbols FORCE ROW LEVEL SECURITY;
ALTER TABLE ast_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE ast_edges FORCE ROW LEVEL SECURITY;
ALTER TABLE doc_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE doc_chunks FORCE ROW LEVEL SECURITY;

-- Fail-closed array overlap policy: empty/unset setting evaluates to NULL -> 0 rows
CREATE POLICY rls_code_files_scope ON code_files
    USING (allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ','));
```

- **Least-Privilege Runtime Role (`scout_app_user`):** Application connection pools (`RLSConnectionManager` via `psycopg_pool.ConnectionPool`) connect exclusively as `scout_app_user` (`NOSUPERUSER NOCREATEDB NOCREATEROLE`); superuser connections are rejected at runtime.
- **Per-Transaction Scope & HNSW Recall Binding:** Every query executes inside an explicit `async with conn.transaction():` / `with conn_mgr.scoped_transaction(principal):` block that runs:
  ```sql
  SET LOCAL app.principal_scopes = 'public_eng,repo:trading_strategies';
  SET LOCAL hnsw.iterative_scan = 'relaxed_order';
  ```
  `SET LOCAL` resets automatically at transaction end, preventing scope leakage across pooled connections, while `hnsw.iterative_scan = 'relaxed_order'` (pgvector `>=0.8.0`) prevents HNSW top-K recall collapse under selective RLS filters.

---

## 3. Prompt-Injection Defense, Secret Redaction & `$0.00` Paid-Model Protection

- **Input & Retrieved-Content Guardrails (`src/context_scout/security/guardrails.py`):**
  - `PromptInjectionSanitizer` + `ShieldGemmaGuardrail`: Blocks system-prompt overrides, delimiter breakouts, HTML-comment split injections (pre- and post-comment stripping), Unicode bidi directionality spoofing, and LaTeX comment steganography (`18/18` active attack payloads blocked).
  - `SecretRedactor`: Redacts high-entropy API keys, JWTs, AWS credentials, and PEM private keys before context leaves the retrieval boundary (`4/4` secret-extraction payloads neutralized; `22/22` combined security gate).
  - `frame_untrusted_content`: Wraps all retrieved code slices, issues, and document chunks in `<untrusted_retrieved_data>` delimiters with random nonces so retrieved text is never treated as instructions.
- **`$0.00` Paid-Model Protection (Abstention Gate):**
  - If retrieval falls below `CALIBRATED_RELEVANCE_FLOOR` (`0.34`), `PreFlightValidator` fails after bounce-back retries, or `grounded_facets == 0`, the pipeline returns `InsufficientContextDiagnostic(paid_tokens_spent=0)` without calling `gemini-3.8-flash` or `gemini-4-argon`.
- **Isolated `GitWorktreeSandbox` (`src/context_scout/agents/gemini_executor.py`):**
  - Paid code generation executes inside an ephemeral `git worktree` (`scout-exec-<uuid>`), strips cloud/API secret environment variables before running `ruff check` and `pytest`, and pauses at `require_confirmation=True` before merging.

---

## 4. Local Workstation vs. Enterprise GCP Deployment Mapping

| Layer | Local Development Stack (This Repo) | Enterprise GCP Production Equivalent |
|---|---|---|
| **Scout Inference & Multi-LoRA** | `mlx-lm` (`gemma-4-26B-A4B-it` 4-bit) + in-memory `AdapterPool` (`MAX_LOADED_MLX_MODELS = 1`) | **GKE / Vertex AI Custom Prediction** running `vLLM` with `--enable-lora` on NVIDIA L4 / A100 GPUs |
| **Embeddings (`768-dim`)** | Local `google/embeddinggemma-2` (`SCOUT_EMBEDDING_DEVICE=cpu` alongside MLX) | **Vertex AI `gemini-embedding-001`** or self-hosted `embeddinggemma-2` on GKE |
| **Multi-Tenant Storage & RLS** | Docker / Homebrew PostgreSQL 17 (`pgvector`, `pg_trgm`, `FORCE RLS`, `scout_app_user`) | **Cloud SQL for PostgreSQL 17** or **AlloyDB** with `FORCE RLS` + IAM database authentication |
| **Identity & Principal Binding** | `POST /api/v1/auth/session` + HMAC-SHA256 Bearer / `X-Principal-Signature` | **Google Cloud IAP / Okta OIDC JWT** validated at API Gateway $\rightarrow$ signed `PrincipalContext` |
| **Input/Output Guardrails** | Local `ShieldGemma 2` + `PromptInjectionSanitizer` + `SecretRedactor` | **Google Cloud Model Armor** + Cloud Sensitive Data Protection (SDP) |
| **Paid Implementation Model** | `gemini-3.8-flash` / `gemini-4-argon` via `google-genai` SDK (`>=2.31.0`) | **Vertex AI Gemini API** (`gemini-3.8-flash` / `gemini-4-argon`) with VPC Service Controls |
| **CI/CD & Cloud Auth** | GitHub Actions [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | Keyless **Workload Identity Federation (WIF)** OIDC $\rightarrow$ Artifact Registry $\rightarrow$ Cloud Run |

---

## 5. When NOT to Use This Architecture

1. **Small Repositories (`<50K` LOC) with Low Query Volume:** If the entire active codebase fits inside a cached frontier context window and daily query volume is low, native Gemini Context Caching is simpler and cheaper than operating PostgreSQL 17 + `pgvector` and a local scout tier.
2. **Sub-200ms Inline Autocomplete (FIM):** The 4-stage discovery, AST closure, and Pre-Flight validation pipeline targets asynchronous multi-file bug fixes, refactors, and research synthesis (`1–4 s` discovery latency), not keystroke-level IDE tab completion.
3. **Greenfield Scaffolding with Zero Existing Context:** When generating standalone scripts or greenfield prototypes with no dependency on existing repository symbols or proprietary literature, routing directly to `gemini-3.8-flash` avoids unnecessary retrieval overhead.
