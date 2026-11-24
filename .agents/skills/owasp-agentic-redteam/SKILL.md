---
name: owasp-agentic-redteam
description: Runs OWASP Agentic Top 10 (ASI01-ASI10) security checks, mutation testing (mutmut), and OpenAPI fuzzing (Schemathesis). Use when auditing security guardrails, authentication, MCP tools, or release gates.
---
# OWASP Agentic Red-Team & Mutation Audit (`/owasp-agentic-redteam`)

1. **ASI01:** Per-request random nonce `<untrusted_data_{nonce}>` + `<30ms` guardrail.
2. **ASI02/03:** Reject identity keys in MCP tool args; strip `normalized_rrf_score`.
3. **ASI05:** Confirm host repo patching and `pytest` execution are completely absent.
4. **ASI06:** Confirm `training_scope_ceiling` subset check + `0/10` canary extraction.
