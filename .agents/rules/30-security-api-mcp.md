---
trigger: glob
description: "Zero-trust OIDC/HMAC auth, per-request XML nonces, host-RCE ban, stateless MCP 2026-07-28, and fail-closed webhooks."
globs: "**/src/context_scout/security/**, **/src/context_scout/api/**, **/src/context_scout/mcp_servers/**"
---
# API, Security, & MCP Boundary Rules

1. **Out-of-Band Identity Only:** Extract `PrincipalContext` strictly from verified OIDC
   JWKS or HMAC tokens into a `ContextVar`. Never expose identity keys in MCP tool
   schemas. Strip client-supplied `normalized_rrf_score` overrides.
2. **No Admin DB or Fake Metrics on API Routes:** `/api/v1/telemetry/*`,
   `/api/v1/webhooks/*`, and `/api/v1/admin/*` MUST enforce RBAC and use the RLS-scoped
   pool—never `db_admin_url` or fabricated baseline numbers.
3. **Fail-Closed Webhooks:** Reject webhook requests with HTTP 401/503 when
   `SCOUT_WEBHOOK_*_SECRET` is unset unless `SCOUT_ALLOW_UNSIGNED_DEV_WEBHOOKS=true` in dev.
4. **Prompt-Injection Defense:** Wrap untrusted retrieved chunks in per-request random
   nonce tags (`<untrusted_data_{nonce}>`) and strip matching closing tags inside content.
5. **Stateless MCP `2026-07-28`:** Support horizontal scaling without sticky session
   state (`SEP-2575`), capability discovery (`SEP-2243`), and W3C `traceparent` (`SEP-2663`).
