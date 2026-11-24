# Enterprise Context Scout — Workspace Agent Instructions

## 1. Mission & Target Operating Model
Enterprise Context Scout is a 100% self-hosted, cloud- and model-agnostic Enterprise
Context Control Plane deployed inside private VPCs or air-gapped Kubernetes clusters.
- **Default Gemma Stack + BYO Open-Weights LLMs:** Preserves `google/gemma-4-*`,
  `mlx-community/gemma-4-*`, `google/embeddinggemma-2`, and `google/shieldgemma-2-2b-it`
  as seamless `$0` defaults while unlocking any enterprise's local open-source LLMs,
  embedders, rerankers, and guardrails (`Qwen3`/`Qwen3-Coder`, `Mistral`/`Devstral`,
  `DeepSeek-V3`/`R1`, `Llama 4`, `Phi-4`, `IBM Granite`, `BAAI/bge-m3`) via
  `vLLM`/`SGLang` or `mlx-lm`, plus ACL-safe `rsLoRA` specialization cartridges.
- **Authoritative Docs:** Align all changes with `ARCHITECTURE_BLUEPRINT.md`,
  `docs/SECURITY_AND_GOVERNANCE.md`, and `reports/v1_ga_release_scorecard.md`.

## 2. Non-Negotiable Engineering & Local Git Invariants
1. **Zero Vendor or Domain Lock-In:** Never block approved non-Google open-weights
   models. Never hardcode finance/quant domain concepts in `src/`, `sql/`, `evals/`, or
   `web-ui/src/` (keep finance dogfooding isolated under `examples/quant_dogfood/`).
2. **Database Kernel Security (`FORCE RLS`):** Every table in PostgreSQL 17 MUST enforce
   `tenant_id UUID NOT NULL`, `ENABLE ROW LEVEL SECURITY`, and `FORCE ROW LEVEL SECURITY`
   under `NOBYPASSRLS`. Runtime routes and distillation jobs MUST NEVER use `db_admin_url`.
3. **Zero Host Code Execution (`ASI05`):** Never apply diffs onto `get_repo_root()` or
   invoke `pytest` on the API host.
4. **Zero Fake Fallbacks or Skipped Tests:** Never fabricate heuristic answers when an
   LLM is offline, never insert `[0.0]*D` or PRNG hash embeddings, and never skip
   integration tests via `-k "not ..."`.
5. **Local Auto-Commit Hygiene:** When a step asks you to commit, verify gates are 100%
   green first, then run `git status --short && git add -A && git commit -m "..."`.
   **NEVER run `git push` and NEVER run `git add -f`** (so `.git/info/exclude` private
   docs remain untracked). In `/swarm` runs, **ONLY `@verifier`** executes `git commit`.

## 3. Eyes-Open Issue Discovery & Self-Healing Protocol
Whenever you inspect or edit code, scan module boundaries (callers, locks, SQL pools,
error handling). If you uncover an unlisted bug or vulnerability, apply `/discovery-triage`:
- **SEV-0 (In-scope blocker, gate blocker, security flaw, RLS leak, or broken contract):**
  Write a failing regression test first, fix the root cause cleanly, verify via `/gate-verify`,
  and document the fix in the PR description.
- **SEV-1 (Non-blocking bug in an untouched subsystem outside current task scope):**
  Record the exact `filepath:line` and reproduction notes in your task report or issue tracker.
  Do NOT edit out-of-scope files during parallel runs.
- **SEV-2 (Empirical measurement or library behavior contradicts documented assumptions):**
  Stop and present a 6-line trade-off summary with measured numbers and two concrete options.
