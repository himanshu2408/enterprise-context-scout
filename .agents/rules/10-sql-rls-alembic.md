---
trigger: glob
description: "PostgreSQL 17 FORCE RLS, NOBYPASSRLS, squawk-safe Alembic migrations, halfvec(D) spaces, and pool separation."
globs: "**/sql/**, **/alembic/**, **/src/context_scout/db/**"
---
# PostgreSQL 17, RLS, & Alembic Rules

1. **100% Table RLS Coverage:** Every table MUST include `tenant_id UUID NOT NULL`,
   `ENABLE ROW LEVEL SECURITY`, and `FORCE ROW LEVEL SECURITY` with fail-closed
   `NULLIF(current_setting('app.principal_scopes', true), '')` and `app.tenant_id` policies.
2. **Least-Privilege Runtime Role:** Application pools connect exclusively as
   `NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOINHERIT` (no hardcoded passwords).
3. **Transaction-Scoped Settings:** Always bind `SET LOCAL app.tenant_id`,
   `SET LOCAL app.principal_scopes`, and `SET LOCAL hnsw.iterative_scan = 'relaxed_order'`
   inside an explicit `conn.transaction()` block.
4. **Zero-Downtime Migrations & Pools:** Lint Alembic migrations with `squawk` and
   separate interactive `query_pool` from background `ingestion_pool`.
5. **Vector Integrity:** Use `embedding_spaces` with `halfvec(D)` FP16 HNSW indexes and
   reject zero-norm `[0.0]*D` vectors at insertion time.
