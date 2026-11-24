---
trigger: glob
description: "Self-contained Testcontainers fixtures, zero laptop DB count assumptions, real BPE tokenizers, and negative-control mutation proofs."
globs: "**/tests/**, **/evals/**, **/src/context_scout/evals/**"
---
# Tests, Benchmarks, & Evaluation Harness Rules

1. **Self-Contained Fixtures:** Seed isolated test data via `Testcontainers` or
   `ephemeral_pg_domain`. Never assert hardcoded laptop row counts (`>= 14318`,
   `>= 30000`) or `~/.cache/huggingface` snapshots.
2. **Non-Tautological Gate Verification:** Missing `chunk_id`s in `EvaluationHarness`
   MUST score `0%` (never verify `quote` against `quote`). Include negative-control
   mutation proofs on security and abstention gates.
3. **Honest Token & Recall Accounting:** Measure token savings with real BPE tokenizers
   (`AutoTokenizer`) and never label mocked generator tests as empirical benchmarks.
