---
name: connector-acl-sync
description: Audits SaaS and on-prem source connectors (GitLab, GitHub Enterprise, Confluence, Jira, Air-Gapped Bundle) for transitive ACL mirroring, 3-tier sync, 429 rate-limit backoff, real embeddings, and <60s secret purge. Use when editing connectors or webhooks.
---
# Enterprise Connector & ACL Sync Verification (`/connector-acl-sync`)

1. **Transitive ACLs:** GitLab 18.7+ `:read_code` (Guest = `0` code scopes); GitHub App
   repo/team ACLs; Confluence V2 `/restriction-state` + recursive ancestor
   `VIEW_RESTRICTED`; Jira `BROWSE` + Issue Security Level + Comment visibility.
2. **Zero Fake Embeddings:** Never emit `[0.0]*D` or `generate_deterministic_embedding`.
3. **3-Tier Sync & Purge:** Verify webhook + watermark + reconciliation + `429` backoff +
   `< 60 s` `purge-artifact` across all stores.
