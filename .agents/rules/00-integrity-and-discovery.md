---
trigger: always_on
description: "Enforces benchmark integrity, zero silent fallbacks, zero skipped tests, and the SEV-0/1/2 Open Discovery Protocol."
---
# Integrity & Open Discovery Rules

1. **No Test Cheating:** Never weaken an assertion, add an allowlist loophole, mock
   PostgreSQL in an RLS/SQL gate, or add `pytest.skip` / `-k "not ..."` to make CI green.
2. **Fail-Closed Defaults:** Missing secrets, missing `chunk_id`s in citation checks,
   offline models on synthesis paths, and unset webhook tokens MUST fail closed with
   explicit typed diagnostics (`paid_tokens_spent=0`).
3. **Issue Triage Discipline:** When uncovering an in-scope `SEV-0` bug or security flaw,
   add a permanent regression test and fix the root cause before closing; report out-of-scope
   `SEV-1` findings with `filepath:line` against `docs/SECURITY_AND_GOVERNANCE.md` and
   `reports/v1_ga_release_scorecard.md`.
4. **Compact Verification:** Prefer `bash .agents/skills/gate-verify/scripts/run_compact_gates.sh`
   so test output stays under 80 lines.
