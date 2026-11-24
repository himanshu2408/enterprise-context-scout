#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import psycopg


def main() -> int:
    db_url = os.environ.get(
        "SCOUT_DB_ADMIN_URL",
        "postgresql://postgres@127.0.0.1:5432/context_scout",
    )
    app_role = os.environ.get("SCOUT_DB_APP_ROLE", "scout_app_user")

    with psycopg.connect(db_url, connect_timeout=5) as conn:
        tables = conn.execute(
            "SELECT c.oid, c.relname, c.relrowsecurity, c.relforcerowsecurity "
            "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relkind = 'r' "
            "AND c.relname NOT IN ('alembic_version') ORDER BY c.relname;"
        ).fetchall()

        if not tables:
            print("[FAIL] No application tables found in schema 'public'.")
            return 1

        missing_rls = [(name, en, frc) for _oid, name, en, frc in tables if not (en and frc)]
        if missing_rls:
            print("[FAIL] Tables missing ENABLE + FORCE ROW LEVEL SECURITY:")
            for name, en, frc in missing_rls:
                print(f"  - {name}: relrowsecurity={en}, relforcerowsecurity={frc}")
            return 1

        missing_tenant_col: list[str] = []
        missing_policies: list[str] = []
        for oid, name, _en, _frc in tables:
            col_row = conn.execute(
                "SELECT a.attnotnull, format_type(a.atttypid, a.atttypmod) "
                "FROM pg_attribute a "
                "WHERE a.attrelid = %s AND a.attname = 'tenant_id' AND NOT a.attisdropped;",
                (oid,),
            ).fetchone()
            if col_row is None or not col_row[0] or col_row[1] != "uuid":
                missing_tenant_col.append(f"{name} ({col_row})")

            pol_row = conn.execute(
                "SELECT COUNT(*) FROM pg_policy WHERE polrelid = %s;",
                (oid,),
            ).fetchone()
            if pol_row is None or int(pol_row[0]) < 1:
                missing_policies.append(str(name))

        if missing_tenant_col:
            print("[FAIL] Tables missing non-null UUID tenant_id column:")
            for item in missing_tenant_col:
                print(f"  - {item}")
            return 1

        if missing_policies:
            print("[FAIL] Tables missing RLS policies in pg_policy:")
            for item in missing_policies:
                print(f"  - {item}")
            return 1

        role_row = conn.execute(
            "SELECT rolsuper, rolbypassrls, rolcreatedb, rolcreaterole, rolinherit "
            "FROM pg_roles WHERE rolname = %s;",
            (app_role,),
        ).fetchone()
        if role_row is None:
            print(f"[FAIL] Runtime role '{app_role}' does not exist.")
            return 1
        if any(role_row):
            print(
                f"[FAIL] Runtime role '{app_role}' violates least privilege "
                f"(rolsuper, rolbypassrls, rolcreatedb, rolcreaterole, rolinherit): {role_row}"
            )
            return 1

    print(
        f"[PASS] All {len(tables)} public tables enforce tenant_id UUID NOT NULL + "
        f"ENABLE/FORCE RLS + pg_policy, and '{app_role}' is strictly unprivileged "
        "(NOBYPASSRLS NOINHERIT)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
