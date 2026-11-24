---
name: rls-pgtap-audit
description: Audits PostgreSQL 17 pg_class and pg_roles catalogs to prove ENABLE and FORCE ROW LEVEL SECURITY on 100% of tables and zero cross-tenant leakage. Use when modifying SQL schemas, Alembic migrations, or database repositories.
---
# PostgreSQL 17 Forced RLS & Role Audit (`/rls-pgtap-audit`)

```bash
uv run python .agents/skills/rls-pgtap-audit/scripts/check_rls_catalog.py
bash .agents/skills/gate-verify/scripts/run_compact_gates.sh --scope py --pytest-target tests/test_postgres_rls_live.py
```
