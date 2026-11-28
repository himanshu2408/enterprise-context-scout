# Security, Governance & Enterprise Deployment Topology

**Enterprise Context Scout** enforces access control at the database storage kernel and transport boundary—never in LLM prompts—and isolates untrusted retrieved content before any local or paid frontier model processes it.

---

## 1. Out-of-Band Authentication & Principal Binding (`src/context_scout/security/rbac.py`, `api/routes/auth.py`)

- **Out-of-Band `PrincipalContext` Extraction:**
  - `POST /api/v1/auth/session` mints short-lived (`3600s`) HMAC-SHA256 Bearer tokens for interactive local evaluation (disabled with `HTTP 403` when `is_production_environment()` is true, requiring an external IdP JWT in production).
  - `GET /api/v1/auth/me` returns the verified `PrincipalContext` (`user_id`, `role`, `allowed_scopes`, `max_session_paid_tokens`) bound to a Python `contextvars.ContextVar` at the transport boundary.
- **Fail-Closed Startup Secret Validation:** An unset signing secret is a startup error when signed authentication is enabled (`SCOUT_REQUIRE_SIGNED_AUTH=true`); production never falls back to hardcoded default secrets.
- **Zero Identity Parameters in MCP Tool Schemas:**
  - MCP tools (`code_repo_mcp`, `doc_corpus_mcp`) accept only domain search parameters.
  - `validate_tool_call_arguments` blocks case-insensitive attempts to inject `principal`, `principal_context`, `allowed_scopes`, `user_id`, `role`, `scopes`, `principal_scopes`, or `app.principal_scopes`, and rejects NUL bytes (`\x00`).
  - `/api/v1/handoff/preview` confirmation tokens are cryptographically bound to the requesting `principal.user_id` and `principal.tenant_id`; cross-principal `/api/v1/handoff/apply` calls return `HTTP 403`.

---

## 2. PostgreSQL 17 `FORCE ROW LEVEL SECURITY` (`src/context_scout/db/migrations/versions/0001..0024`)

Tenant and repository isolation is enforced inside PostgreSQL 17 across all **23 public tables** (Alembic migrations `0001` through `0024`) so that even a compromised application query cannot read unauthorized rows:

```sql
-- Forced RLS on all 23 tenant-scoped tables (even table owners cannot bypass)
ALTER TABLE code_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE code_files FORCE ROW LEVEL SECURITY;
ALTER TABLE ast_symbols ENABLE ROW LEVEL SECURITY;
ALTER TABLE ast_symbols FORCE ROW LEVEL SECURITY;
ALTER TABLE ast_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE ast_edges FORCE ROW LEVEL SECURITY;
ALTER TABLE doc_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE doc_chunks FORCE ROW LEVEL SECURITY;

-- Fail-closed tenant & scope policy: empty/unset setting evaluates to NULL -> 0 rows
CREATE POLICY rls_code_files_scope ON code_files
    USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        AND allowed_scopes && string_to_array(NULLIF(current_setting('app.principal_scopes', true), ''), ',')
    );
```

- **Least-Privilege Runtime Role (`scout_app_user`):** Application connection pools (`RLSConnectionManager` via `psycopg_pool.ConnectionPool`) connect exclusively as `scout_app_user` (`NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOINHERIT`); superuser connections are rejected at runtime.
- **Per-Transaction Scope & HNSW Recall Binding:** Every query executes inside an explicit `async with conn.transaction():` / `with conn_mgr.scoped_transaction(principal):` block that runs:
  ```sql
  SELECT set_config('app.tenant_id', '<tenant-uuid>', true);
  SELECT set_config('app.principal_scopes', 'public_eng,repo:core_services', true);
  SET LOCAL hnsw.iterative_scan = 'relaxed_order';
  ```
  `SET LOCAL` / `set_config(..., true)` resets automatically at transaction end, preventing tenant or scope leakage across pooled connections, while `hnsw.iterative_scan = 'relaxed_order'` (pgvector `>=0.8.0`) prevents HNSW top-K recall collapse under selective RLS filters.

---

## 3. Prompt-Injection Defense, Secret Redaction & `$0.00` Paid-Model Protection

- **Input & Retrieved-Content Guardrails (`src/context_scout/security/guardrails.py`):**
  - `PromptInjectionSanitizer` + `ShieldGemmaGuardrail`: Local runtime executes deterministic regex, Unicode-bidi, and comment-stripping rules (`evaluate_deterministic_fallback()`, `used_fallback=True`), blocking system-prompt overrides, delimiter breakouts, HTML-comment split injections, Unicode bidi directionality spoofing, and LaTeX comment steganography (`18/18` known signature payloads blocked). **Important limitation:** Because local Stage 1 does not run live neural `ShieldGemma 2` weights, novel paraphrased prompt injections without known trigger signatures are not caught by Stage 1 regexes; defense-in-depth relies on SQL `FORCE RLS`, identity-free read-only MCP tool schemas, and `<untrusted_data_{nonce}>` delimiter framing (with Google Cloud Model Armor / neural `ShieldGemma 2` as the production target).
  - `SecretRedactor`: Redacts high-entropy API keys, JWTs, AWS credentials, and PEM private keys before context leaves the retrieval boundary (`4/4` secret-extraction payloads neutralized; `22/22` combined security gate).
  - `frame_untrusted_content`: Wraps all retrieved code slices, issues, and document chunks in per-call 128-bit random nonce delimiters (`<untrusted_data_{nonce}>...</untrusted_data_{nonce}>`) so retrieved text is never treated as instructions.
- **`$0.00` Paid-Model Protection (Abstention Gate & Air-Gap Egress Default):**
  - `allow_external_cloud_egress` defaults to `false` (configured via `config/model_registry.yaml` and `SCOUT_ALLOW_EXTERNAL_CLOUD_EGRESS=false`), ensuring zero outbound cloud calls unless explicitly opted in.
  - If retrieval falls below `CALIBRATED_RELEVANCE_FLOOR` (`0.34`), `PreFlightValidator` fails after bounce-back retries, or `grounded_facets == 0`, the pipeline returns `InsufficientContextDiagnostic(paid_tokens_spent=0)` without calling any cloud handoff model (`gemini-3.8-flash`, `gemini-4-argon`, `claude-sonnet-5.5`, `claude-opus-5.5`, `gpt-5.6-sol`, `gpt-6-astra`).
- **Zero Host Code Execution (`OWASP ASI05`, `src/context_scout/agents/gemini_executor.py`, `api/routes/handoff.py`):**
  - `0` `shell=True` calls exist in `src/context_scout/`; `POST /api/v1/handoff/apply` rejects `custom_patch` (`HTTP 400 ASI05`) and requires an atomic single-consume tenant+user confirmation token (`confirmed=True`). Host-side `pytest` invocation and arbitrary host repository patching are strictly prohibited.

---

## 4. Local Workstation vs. Enterprise GCP Deployment Mapping

| Layer | Local Development Stack (This Repo) | Enterprise GCP / VPC Production Equivalent |
|---|---|---|
| **Scout Inference & Multi-LoRA** | `mlx-lm` (`gemma-4-26B-A4B-it` 4-bit or BYO open-weights) + `AdapterPool` (`MAX_LOADED_MLX_MODELS = 1`) | **GKE / Vertex AI Custom Prediction** running `vLLM`/`SGLang` with `--enable-lora` on NVIDIA L4 / A100 / H100 GPUs (Gemma 4, `Qwen3-Coder`, `Devstral`, `DeepSeek-V3/R1`, `Llama-4`, `Phi-4`, `IBM Granite`) |
| **Embeddings (`768-dim` / `1024-dim`)** | Local `google/embeddinggemma-2` or BYO (`Qwen3-Embedding`, `BAAI/bge-m3`) | **Vertex AI `gemini-embedding-001`** or self-hosted `embeddinggemma-2` / `BAAI/bge-m3` on GKE |
| **Multi-Tenant Storage & RLS** | Docker / Homebrew PostgreSQL 17 (`pgvector`, `pg_trgm`, `FORCE RLS`, `scout_app_user`) | **Cloud SQL for PostgreSQL 17** or **AlloyDB** with `FORCE RLS` + keyless IAM database authentication |
| **Identity & Principal Binding** | `POST /api/v1/auth/session` + HMAC-SHA256 Bearer / `X-Principal-Signature` | **Google Cloud IAP / Okta / Entra ID / Keycloak OIDC JWT** validated at API Gateway $\rightarrow$ signed `PrincipalContext` |
| **Input/Output Guardrails** | Local deterministic `PromptInjectionSanitizer` + `SecretRedactor` (`ShieldGemmaGuardrail` fallback) | **Google Cloud Model Armor** + neural `ShieldGemma 2` / BYO (`Prompt-Guard-2`, `Qwen3Guard`, `Granite-Guardian`) |
| **Optional Cloud Handoff (`allow_external_cloud_egress: true`)** | Disabled by default (`false`); optional BYO cloud (`gemini-3.8-flash`, `gemini-4-argon`, `claude-*`, `gpt-*`) | **Vertex AI Gemini API** (EU-pinned `europe-west3` via keyless ADC) with VPC Service Controls or approved BYO cloud provider |
| **CI/CD & Cloud Auth** | GitHub Actions [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) (40-char SHA-pinned) + Sigstore-signed CycloneDX v1.6 AI-BOM (`aibom.cdx.json.sigstore.json` + `.pub`) | Keyless **Workload Identity Federation (WIF)** OIDC $\rightarrow$ Artifact Registry $\rightarrow$ GKE |
| **Health Probes & OTel GenAI `v1.42+`** | Split `/api/v1/health/live` & `/api/v1/health/ready` (2s TTL pooled check); `gen_ai.provider.name` + `scout.pack.tokens` + `max_tracked_tenants` cardinality cap | Kubernetes `livenessProbe` / `readinessProbe` + Cloud Trace / Managed Service for Prometheus |

---

## 5. When NOT to Use This Architecture

1. **Small Repositories (`<50K` LOC) with Low Query Volume:** If the entire active codebase fits inside a cached frontier context window and daily query volume is low, native context caching is simpler and cheaper than operating PostgreSQL 17 + `pgvector` and a local scout tier.
2. **Sub-200ms Inline Autocomplete (FIM):** The 4-stage discovery, AST closure, and Pre-Flight validation pipeline targets asynchronous multi-file bug fixes, refactors, and research synthesis (`1–4 s` discovery latency), not keystroke-level IDE tab completion.
3. **Greenfield Scaffolding with Zero Existing Context:** When generating standalone scripts or greenfield prototypes with no dependency on existing repository symbols or proprietary documentation, routing directly to a general-purpose coding model (when `allow_external_cloud_egress=true` or local BYO open-weights) avoids unnecessary retrieval overhead.
