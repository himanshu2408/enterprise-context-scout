# Milestone 4 Design Specification: Deterministic Abstention & Gemma Claim Verification

**Target System:** `enterprise-context-scout`  
**Milestone:** M4 (Features F19–F22)  
**Author:** Architecture Review & Project Orchestrator  
**Status:** APPROVED CONDITIONAL (All 9 User Directives Fully Incorporated)  
**Document Path:** `enterprise-context-scout/docs/M4_DESIGN.md`  
**Authoritative Reference:** `OPEN_ITEMS.md §F19` & User Directives (2026-11-10T10:25:02Z)  

---

## 1. Executive Summary & Core Objective

In enterprise production environments, an AI context scout cannot be permitted to guess when data is missing, ambiguous, or superseded. When a frontier implementation model (e.g., `gemini-4-argon`, `claude-opus-5.5`, or `gpt-5.6-sol` costing $2.00–$4.00 / 1M input tokens) receives hallucinated or speculative context, it implements invalid code, generates phantom method calls, or writes breaking migrations.

**The Golden Principle of Milestone 4: ABSTENTION IS THE CORE REQUIREMENT.**
- If the best evidence is missing, contradictory, or weak, the system must deterministically output an explicit `InsufficientContextDiagnostic(paid_tokens_spent=0)` naming exactly what was not found.
- Abstention MUST be enforced deterministically in compiled/executable Python code gates, **never via LLM prompt instructions** ("please say I don't know").
- Paid exploration token burn on unanswerable or ambiguous tasks must be strictly **0 tokens**, verified mechanically by call-counting test harnesses.
- Over-abstention (gameable by refusing everything) is strictly controlled and measured via a dedicated Answerable Query Evaluation ($N \ge 300$).

---

## 2. October 2026 Live Model Verification & Pricing Standards

All model references and telemetry estimators in this design are verified against current October 2026 official vendor rates:
- **Local Discovery Engine:** Google `gemma-4-26B-A4B-it` (3.8B active MoE parameters) — **$0.00 API Spend / 0 Cloud Egress**.
- **Fast Cloud Implementation Sandbox:** Google `gemini-3.8-flash` (`gemini-3.8-flash`) — **$0.75–$1.50 / 1M input, $3.75–$7.50 / 1M output**.
- **Frontier Cloud Reasoning Models:**
  - Google `gemini-4-argon` (`gemini-4-argon`): **$2.00–$4.00 / 1M input, $10.00–$20.00 / 1M output**.
  - Anthropic `claude-opus-5.5`: **$4.00 / 1M input, $20.00 / 1M output**.
  - OpenAI `gpt-5.6-sol`: **$4.00 / 1M input, $20.00 / 1M output**.
*(Legacy 2025 models such as `gemini-2.5-pro` and `gemini-2.5-flash` are retired and strictly prohibited.)*

---

## 3. The 9 Structural Architecture Requirements (User Directives)

### Requirement 1: Dual Evaluation — Abstention & Over-Abstention with Token A/B
Abstention accuracy alone is trivial to game by refusing all queries. Evaluation requires balanced statistical suites:
- **EVAL-01 (Unanswerable / Insufficient Context, $N \ge 300$):** Target 100% abstention on unanswerable tasks ($300/300$), establishing a Clopper-Pearson 95% confidence lower bound $\ge 98.78\%$.
- **EVAL-04 (Answerable Query Evaluation, $N \ge 300$):** Measures wrongful refusals (false abstention rate). The scout must successfully assemble verified context packs for genuine answerable tasks with wrongful refusal rate $\le 1.0\%$ (proposed default; needs user confirmation).
- **EVAL-05 (Token Savings A/B Benchmark, $N \ge 120$):** Compares total task token spend between (a) frontier model doing direct tool-driven grep/read exploration vs. (b) 1-shot implementation using scout context packs across 4 polyglot stacks (.NET/C#, React/TS, Python, Terraform). Report-only metric: measures and reports empirical token cost reduction against the baseline with 95% confidence intervals (no fixed percentage cutoff).

### Requirement 2: Deterministic 6-Check Preflight Gate with Retrieval Relevance & RRF Thresholding
The Preflight Gate (`src/context_scout/agents/preflight_gate.py`) evaluates context through an explicit 6-check sequence:
1. **Check 1 — AST Syntax & Well-Formedness:** Verifies all code slices parse cleanly under the respective language parser (`tree-sitter`). Partial parses or syntax errors reject the item as `MALFORMED_SYNTAX`.
2. **Check 2 — 1-Hop AST Dependency Closure:** Ensures every referenced symbol resolves to an imported type definition, an external manifest dependency (`csproj`, `package.json`, `pyproject.toml`), or standard library. Unresolved internal symbols emit `MissingItemRecord(kind="code_symbol", identifier=missing_sym)`.
3. **Check 3 — Retrieval Confidence & Relevance Thresholding:** Evaluates Reciprocal Rank Fusion (RRF) and vector/trigram relevance scores against configurable threshold `min_retrieval_confidence` (default 0.65). If top retrieved items score below threshold or evidence is weak/sparse, the gate deterministically abstains with `WEAK_RETRIEVAL_CONFIDENCE`.
4. **Check 4 — Principal Scope & Tenant Permission Mirroring:** Confirms all retrieved artifacts match `PrincipalContext.allowed_scopes`. Any unauthorized item is excised. If no permitted context remains, emits `PERMISSION_DENIED_OR_OUT_OF_SCOPE`.
5. **Check 5 — Cryptographic Provenance Anchoring:** Confirms all code items contain valid `git_commit_sha`, `git_blob_sha`, and exact 1-indexed line spans matching physical storage byte-for-byte. Rejects unanchored or synthesized snippets.
6. **Check 6 — Document Freshness & Staleness Invalidation:** Invalidation of superseded Confluence page versions and Jira tickets matching configured terminal/stale statuses.

### Requirement 3: Exact Normalized-Substring Matching Only (Zero Tolerance)
- The legacy `SequenceMatcher >= 0.98` secondary match is **completely removed**.
- **Allowed Matching:** Exact Unicode NFKC-normalized substring matching only. Whitespace sequences (spaces, tabs, newlines) are collapsed to single spaces before comparison.
- **Permanent Regression Tests:** Dedicated test suite verifying:
  - Exact one-word substitutions and negations ("must" vs "must not", "allow" vs "deny") are rejected 100%.
  - Bidirectional homoglyph tests: Cyrillic/Greek homoglyphs forged into Latin quotes are strictly rejected; authentic internationalized Unicode source code is preserved and accepted.

### Requirement 4: Extractive Gemma Discovery Schema
Local Gemma output must be strictly extractive — emitting only pointers and verbatim quotes. Any free-text generated by Gemma is flagged as non-evidence and filtered out from the prompt delivered to the paid model:

```python
class CodeSpanPointer(BaseModel):
    repo_name: str
    file_path: str
    commit_sha: str
    blob_sha: str
    line_start: int
    line_end: int
    symbol_id: str | None = None


class DocSpanPointer(BaseModel):
    source_type: Literal["confluence", "jira"]
    item_id: str
    version_or_status: str
    verbatim_quote: str


class ExtractiveDiscoveryPack(BaseModel):
    query_id: str
    code_pointers: list[CodeSpanPointer]
    doc_pointers: list[DocSpanPointer]
    raw_quotes: list[str]
    non_evidence_scratchpad: str = Field(
        default="", description="Local reasoning scratchpad. NEVER passed to paid frontier models."
    )
```

### Requirement 5: Stale-Content Benchmark Target — Zero Wrong Answers ($N \ge 100$)
- In EVAL-02 ($N \ge 100$), the target is **0 wrong answers** ($100\%$ detection of superseded/stale material), not $\ge 99\%$.
- With $N = 100$ and $k = 100$ successes, the Clopper-Pearson 95% confidence interval lower bound is **$96.38\%$**.
- With $N = 300$ and $k = 300$, the lower bound reaches **$98.78\%$**.

### Requirement 6: Configurable Claim Verification Policy
The fixed 50% claim ratio is replaced by a configurable policy engine:

```python
class ClaimVerificationPolicy(str, Enum):
    STRICT_ABSTAIN_ON_ANY_UNVERIFIED = "strict_abstain"  # Any unverified claim triggers abstention
    PRUNE_AND_MARK_LOW_TRUST = (
        "prune_and_warn"  # Prune unverifiable claims, continue if verified >= min_count
    )
    MIN_VERIFIED_COUNT = "min_verified_count"  # Require at least N verified claims


class ClaimVerifierConfig(BaseModel):
    policy: ClaimVerificationPolicy = ClaimVerificationPolicy.STRICT_ABSTAIN_ON_ANY_UNVERIFIED
    min_verified_claims: int = 1
    record_pruned_audit_trail: bool = True
```
Default production policy: `STRICT_ABSTAIN_ON_ANY_UNVERIFIED`. Every unverified claim is excised and recorded in the audit log.

### Requirement 7: Independent Ground-Truth Labelling & Disjointness
- Ground-truth evaluation datasets (EVAL-01 through EVAL-05) must be mechanically generated from public open-source repositories (`dotnet/eShop`, `reduxjs/redux`, `hashicorp/terraform`, `pallets/flask`) and synthetic Atlassian fixtures.
- Test labels are generated by deterministic AST analysis scripts completely separate from the agents developing the gates.
- Evaluation data is mathematically disjoint from model training data: adapters are evaluated on held-out unseen repositories.

### Requirement 8: Decoupled Multi-Tenant Configuration (Zero Hardcoded Stacks)
All Jira terminal statuses, Confluence version rules, retrieval confidence thresholds, and claim policies are defined via configuration or plugin registries:
- `ScoutConfig` / `DomainConfig`:
  - `jira_terminal_statuses: list[str] = ["Closed", "Resolved", "Done"]`
  - `jira_negative_resolutions: list[str] = ["Won't Fix", "Duplicate", "Cannot Reproduce", "Abandoned", "Won't Do"]`
  - `min_retrieval_confidence: float = 0.65`
  - `claim_policy: ClaimVerificationPolicy = ClaimVerificationPolicy.STRICT_ABSTAIN_ON_ANY_UNVERIFIED`
- Zero hardcoded company workflows; neutral enterprise defaults across all modules.

### Requirement 9: Mechanical Zero-Spend Proof
Zero spend on abstention cannot rely on a declaration field (`paid_tokens_spent: 0`). It requires mechanical proof:
- A dedicated test fixture intercepts the Vertex AI / paid model client (`GeminiExecutor`).
- An invocation spy tracks exact API call count.
- In all 300 unanswerable test queries (EVAL-01), the test asserts:
  ```python
  assert paid_client_spy.call_count == 0
  assert paid_client_spy.tokens_billed == 0
  ```
- Any abstention path that attempts network I/O or token consumption raises a test failure.

---

## 4. Implementation Plan & Gate Qualification

1. **Step 1: Configuration & Contracts (`src/context_scout/schemas/contracts.py`, `config.py`)**
   - Implement `ExtractiveDiscoveryPack`, `ClaimVerifierConfig`, `InsufficientContextDiagnostic`.
2. **Step 2: Relevance & 6-Check Preflight Gate (`src/context_scout/agents/preflight_gate.py`)**
   - Integrate RRF thresholding, AST closure verification, scope checking, and freshness rules.
3. **Step 3: Exact-Substring Claim Verifier (`src/context_scout/agents/claim_verifier.py`)**
   - Exact Unicode NFKC substring matching; negation/homoglyph rejection; configurable policy execution.
4. **Step 4: Mechanical Zero-Spend Test Suite (`tests/test_zero_spend_abstention.py`)**
   - Spy/mock verifying 0 client invocations on abstention.
5. **Step 5: Statistical Benchmark Suites (EVAL-01 – EVAL-05)**
   - Unanswerable ($N=300$), Answerable ($N=300$), Stale ($N=100$), Collision ($N=100$), Token A/B ($N=120$).
6. **Step 6: Gate 4 Adversarial Qualification**
   - Forensic Auditor (CLEAN) + 2 Reviewers (APPROVE) + 2 Challengers (APPROVE).
